/*
 * Copyright (C) 2026 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

// VK_LAYER_GZ_swapchain_dump
//
// Minimal Vulkan layer that intercepts vkQueuePresentKHR and writes the
// presented swapchain image (i.e. the pixels the WSI is about to display) to a
// PPM file on disk, BEFORE the present operation proceeds. Used by the
// gz-gui QSGSimpleTextureNode + QSGVulkanTexture::fromNative regression test
// to assert against the actual presented image rather than against
// QQuickWindow::grabWindow() (which diverges from the swapchain present path).
//
// Activation (per-process):
//   VK_LAYER_PATH=<dir-containing-the-json-manifest>
//   VK_INSTANCE_LAYERS=VK_LAYER_GZ_swapchain_dump
//   GZ_SWAPCHAIN_DUMP_PATH=/tmp/foo            -- output prefix (required)
//   GZ_SWAPCHAIN_DUMP_MAX_FRAMES=1             -- dump first N frames (default 1)
//
// Output:
//   <prefix>.frame_0000.ppm                    -- PPM P6 8-bit RGB
//   <prefix>.frame_0000.txt                    -- one-line metadata
//
// Hook strategy:
//   * vkCreateInstance/vkCreateDevice: standard layer chain init.
//   * vkCreateSwapchainKHR: record per-swapchain VkFormat + VkExtent2D.
//   * vkGetSwapchainImagesKHR: record the per-swapchain VkImage list.
//   * vkQueuePresentKHR: for each presented swapchain, record an extra
//     command buffer that barriers the swapchain image PRESENT_SRC_KHR ->
//     TRANSFER_SRC_OPTIMAL, vkCmdCopyImageToBuffer into a host-visible
//     buffer, barriers back to PRESENT_SRC_KHR; submit with the original
//     present-wait semaphores as wait, signal a fence; CPU-wait the fence;
//     map the buffer; write PPM. Then forward to next layer's
//     vkQueuePresentKHR with waitSemaphoreCount=0 (the original semaphores
//     were consumed by our copy submit).
//
// Supported swapchain formats:
//   * VK_FORMAT_B8G8R8A8_UNORM / _SRGB
//   * VK_FORMAT_R8G8B8A8_UNORM / _SRGB
// Other formats are written as RAW BYTES with a warning in the .txt sidecar;
// the test must verify the format matches its expectations.

// NOLINTBEGIN  -- Vulkan layer boilerplate uses C-style casts, sentinel macros,
//                  and global state by design.

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
// Re-define VK_LAYER_EXPORT here so the shared library exports the right
// symbols whether or not vk_layer.h pulled it in.
#if defined(__linux__) || defined(__APPLE__)
#  define LAYER_EXPORT extern "C" __attribute__((visibility("default")))
#else
#  define LAYER_EXPORT extern "C"
#endif

constexpr const char *kLayerName = "VK_LAYER_GZ_swapchain_dump";

std::mutex gMutex;

// Per-instance dispatch (the loader passes us a key inside each handle's chain;
// we extract it and use it as our own lookup key).
struct InstanceData
{
  VkInstance instance{VK_NULL_HANDLE};
  PFN_vkGetInstanceProcAddr nextGipa{nullptr};
  PFN_vkDestroyInstance DestroyInstance{nullptr};
  PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties{
      nullptr};
  PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties{
      nullptr};
};

struct DeviceData
{
  VkDevice device{VK_NULL_HANDLE};
  VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
  PFN_vkGetDeviceProcAddr nextGdpa{nullptr};
  PFN_vkDestroyDevice DestroyDevice{nullptr};

  // Forwarded device entry points we need.
  PFN_vkCreateSwapchainKHR CreateSwapchainKHR{nullptr};
  PFN_vkDestroySwapchainKHR DestroySwapchainKHR{nullptr};
  PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR{nullptr};
  PFN_vkQueuePresentKHR QueuePresentKHR{nullptr};
  PFN_vkCreateCommandPool CreateCommandPool{nullptr};
  PFN_vkDestroyCommandPool DestroyCommandPool{nullptr};
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers{nullptr};
  PFN_vkFreeCommandBuffers FreeCommandBuffers{nullptr};
  PFN_vkBeginCommandBuffer BeginCommandBuffer{nullptr};
  PFN_vkEndCommandBuffer EndCommandBuffer{nullptr};
  PFN_vkResetCommandBuffer ResetCommandBuffer{nullptr};
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier{nullptr};
  PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer{nullptr};
  PFN_vkCreateBuffer CreateBuffer{nullptr};
  PFN_vkDestroyBuffer DestroyBuffer{nullptr};
  PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{nullptr};
  PFN_vkAllocateMemory AllocateMemory{nullptr};
  PFN_vkFreeMemory FreeMemory{nullptr};
  PFN_vkBindBufferMemory BindBufferMemory{nullptr};
  PFN_vkMapMemory MapMemory{nullptr};
  PFN_vkUnmapMemory UnmapMemory{nullptr};
  PFN_vkCreateFence CreateFence{nullptr};
  PFN_vkDestroyFence DestroyFence{nullptr};
  PFN_vkWaitForFences WaitForFences{nullptr};
  PFN_vkResetFences ResetFences{nullptr};
  PFN_vkQueueSubmit QueueSubmit{nullptr};

  VkPhysicalDeviceMemoryProperties memProps{};

  // Per-device command pool created lazily on first present. Indexed by queue
  // family; we use the queue family the present was submitted on.
  std::unordered_map<uint32_t, VkCommandPool> cmdPools;
};

struct SwapchainData
{
  VkDevice device{VK_NULL_HANDLE};
  VkFormat format{VK_FORMAT_UNDEFINED};
  VkExtent2D extent{0u, 0u};
  std::vector<VkImage> images;
};

std::unordered_map<void *, InstanceData *> gInstanceMap;
std::unordered_map<void *, DeviceData *> gDeviceMap;
std::unordered_map<VkSwapchainKHR, SwapchainData *> gSwapchainMap;

// Vulkan layer dispatch keys. The loader stores a pointer to the dispatch
// table as the first member of every dispatchable handle (VkInstance,
// VkDevice, VkQueue, VkCommandBuffer). We use that pointer as our own key.
inline void *DispatchKey(const void *_handle)
{
  return *reinterpret_cast<void *const *>(_handle);
}

InstanceData *GetInstance(VkInstance _instance)
{
  std::lock_guard<std::mutex> lk(gMutex);
  auto it = gInstanceMap.find(DispatchKey(_instance));
  return it == gInstanceMap.end() ? nullptr : it->second;
}

InstanceData *GetInstanceByPhysical(VkPhysicalDevice _phys)
{
  std::lock_guard<std::mutex> lk(gMutex);
  auto it = gInstanceMap.find(DispatchKey(_phys));
  return it == gInstanceMap.end() ? nullptr : it->second;
}

DeviceData *GetDevice(VkDevice _device)
{
  std::lock_guard<std::mutex> lk(gMutex);
  auto it = gDeviceMap.find(DispatchKey(_device));
  return it == gDeviceMap.end() ? nullptr : it->second;
}

DeviceData *GetDeviceByQueue(VkQueue _queue)
{
  std::lock_guard<std::mutex> lk(gMutex);
  auto it = gDeviceMap.find(DispatchKey(_queue));
  return it == gDeviceMap.end() ? nullptr : it->second;
}

uint32_t SafeMaxFrames()
{
  const char *env = std::getenv("GZ_SWAPCHAIN_DUMP_MAX_FRAMES");
  if (env == nullptr || env[0] == '\0')
    return 1u;
  long v = std::strtol(env, nullptr, 10);
  if (v <= 0)
    return 1u;
  if (v > 1024)
    return 1024u;
  return static_cast<uint32_t>(v);
}

std::atomic<uint32_t> gDumpedFrames{0u};

void LayerLog(const char *_fmt, ...)
{
  static const bool verbose = (std::getenv("GZ_SWAPCHAIN_DUMP_VERBOSE")
      != nullptr);
  if (!verbose)
    return;
  std::fprintf(stderr, "[VK_LAYER_GZ_swapchain_dump] ");
  std::va_list ap;
  va_start(ap, _fmt);
  std::vfprintf(stderr, _fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
}

// Pulls VkLayerInstanceCreateInfo/VkLayerDeviceCreateInfo out of the chain
// the loader prepended. Required for layer chain walking.
VkLayerInstanceCreateInfo *FindLayerInstanceLink(const VkInstanceCreateInfo *_ci)
{
  auto *p = static_cast<const VkBaseInStructure *>(_ci->pNext);
  while (p != nullptr)
  {
    if (p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO)
    {
      auto *li = reinterpret_cast<VkLayerInstanceCreateInfo *>(
          const_cast<VkBaseInStructure *>(p));
      if (li->function == VK_LAYER_LINK_INFO)
        return li;
    }
    p = p->pNext;
  }
  return nullptr;
}

VkLayerDeviceCreateInfo *FindLayerDeviceLink(const VkDeviceCreateInfo *_ci)
{
  auto *p = static_cast<const VkBaseInStructure *>(_ci->pNext);
  while (p != nullptr)
  {
    if (p->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO)
    {
      auto *li = reinterpret_cast<VkLayerDeviceCreateInfo *>(
          const_cast<VkBaseInStructure *>(p));
      if (li->function == VK_LAYER_LINK_INFO)
        return li;
    }
    p = p->pNext;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

// Returns (bytesPerPixel, swizzleBGR).  swizzleBGR == true means the on-disk
// pixel ordering is BGRA and we must swap R<->B when writing PPM.
bool FormatInfo(VkFormat _f, uint32_t *_bpp, bool *_swizzleBgr,
    bool *_srgb)
{
  *_swizzleBgr = false;
  *_srgb = false;
  switch (_f)
  {
    case VK_FORMAT_B8G8R8A8_UNORM:
      *_bpp = 4u; *_swizzleBgr = true; return true;
    case VK_FORMAT_B8G8R8A8_SRGB:
      *_bpp = 4u; *_swizzleBgr = true; *_srgb = true; return true;
    case VK_FORMAT_R8G8B8A8_UNORM:
      *_bpp = 4u; return true;
    case VK_FORMAT_R8G8B8A8_SRGB:
      *_bpp = 4u; *_srgb = true; return true;
    default:
      *_bpp = 0u; return false;
  }
}

uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties &_props,
    uint32_t _typeFilter, VkMemoryPropertyFlags _required)
{
  for (uint32_t i = 0u; i < _props.memoryTypeCount; ++i)
  {
    if ((_typeFilter & (1u << i)) == 0u)
      continue;
    if ((_props.memoryTypes[i].propertyFlags & _required) == _required)
      return i;
  }
  return UINT32_MAX;
}

// Write a PPM P6 file from a tightly-packed RGB byte buffer.
bool WritePpm(const std::string &_path, uint32_t _w, uint32_t _h,
    const uint8_t *_rgb)
{
  FILE *f = std::fopen(_path.c_str(), "wb");
  if (f == nullptr)
    return false;
  std::fprintf(f, "P6\n%u %u\n255\n", _w, _h);
  std::fwrite(_rgb, 1u, static_cast<std::size_t>(_w) * _h * 3u, f);
  std::fclose(f);
  return true;
}

void WriteSidecar(const std::string &_path, const SwapchainData &_sc,
    uint32_t _frame, uint32_t _imageIndex, const char *_note)
{
  FILE *f = std::fopen(_path.c_str(), "w");
  if (f == nullptr)
    return;
  std::fprintf(f, "frame=%u\n", _frame);
  std::fprintf(f, "swapchain_image_index=%u\n", _imageIndex);
  std::fprintf(f, "extent=%ux%u\n", _sc.extent.width, _sc.extent.height);
  std::fprintf(f, "format=%d\n", static_cast<int>(_sc.format));
  std::fprintf(f, "note=%s\n", _note != nullptr ? _note : "");
  std::fclose(f);
}

// ---------------------------------------------------------------------------
// vkQueuePresentKHR: the heart of the layer.
// ---------------------------------------------------------------------------

VkResult DumpOneSwapchainImage(DeviceData *_dd, VkQueue _queue,
    uint32_t _queueFamily, const SwapchainData &_sc, uint32_t _imageIndex,
    uint32_t _waitSemaphoreCount, const VkSemaphore *_pWaitSemaphores,
    uint32_t _frame)
{
  // Pixel format check.
  uint32_t bpp = 0u;
  bool swizzleBgr = false;
  bool srgb = false;
  const bool known = FormatInfo(_sc.format, &bpp, &swizzleBgr, &srgb);
  if (!known)
  {
    LayerLog("present: skipping dump, format %d not supported (must be "
             "B8G8R8A8/R8G8R8A8 UNORM or SRGB)",
        static_cast<int>(_sc.format));
    return VK_SUCCESS;
  }

  const uint32_t w = _sc.extent.width;
  const uint32_t h = _sc.extent.height;
  if (w == 0u || h == 0u)
  {
    LayerLog("present: skipping dump, zero extent");
    return VK_SUCCESS;
  }

  // Lazy cmd pool for this queue family.
  VkCommandPool pool = VK_NULL_HANDLE;
  {
    std::lock_guard<std::mutex> lk(gMutex);
    auto it = _dd->cmdPools.find(_queueFamily);
    if (it == _dd->cmdPools.end())
    {
      VkCommandPoolCreateInfo pci{};
      pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pci.queueFamilyIndex = _queueFamily;
      if (_dd->CreateCommandPool(_dd->device, &pci, nullptr, &pool)
          != VK_SUCCESS)
      {
        LayerLog("present: CreateCommandPool failed");
        return VK_SUCCESS;
      }
      _dd->cmdPools[_queueFamily] = pool;
    }
    else
    {
      pool = it->second;
    }
  }

  VkCommandBuffer cb = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1u;
  if (_dd->AllocateCommandBuffers(_dd->device, &cbai, &cb) != VK_SUCCESS)
  {
    LayerLog("present: AllocateCommandBuffers failed");
    return VK_SUCCESS;
  }

  // Host-visible buffer big enough for the full image.
  const VkDeviceSize bufSize = static_cast<VkDeviceSize>(w) * h * bpp;
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bufSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (_dd->CreateBuffer(_dd->device, &bci, nullptr, &buf) != VK_SUCCESS)
    {
      LayerLog("present: CreateBuffer failed");
      _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);
      return VK_SUCCESS;
    }
    VkMemoryRequirements mr{};
    _dd->GetBufferMemoryRequirements(_dd->device, buf, &mr);
    const uint32_t typeIndex = FindMemoryType(_dd->memProps, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (typeIndex == UINT32_MAX)
    {
      LayerLog("present: no host-visible memory type");
      _dd->DestroyBuffer(_dd->device, buf, nullptr);
      _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);
      return VK_SUCCESS;
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = typeIndex;
    if (_dd->AllocateMemory(_dd->device, &mai, nullptr, &mem) != VK_SUCCESS)
    {
      LayerLog("present: AllocateMemory failed");
      _dd->DestroyBuffer(_dd->device, buf, nullptr);
      _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);
      return VK_SUCCESS;
    }
    _dd->BindBufferMemory(_dd->device, buf, mem, 0u);
  }

  // Record: PRESENT_SRC -> TRANSFER_SRC, copy, TRANSFER_SRC -> PRESENT_SRC.
  VkCommandBufferBeginInfo cbbi{};
  cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (_dd->BeginCommandBuffer(cb, &cbbi) != VK_SUCCESS)
  {
    LayerLog("present: BeginCommandBuffer failed");
    _dd->FreeMemory(_dd->device, mem, nullptr);
    _dd->DestroyBuffer(_dd->device, buf, nullptr);
    _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);
    return VK_SUCCESS;
  }

  VkImage image = _sc.images[_imageIndex];

  VkImageMemoryBarrier toSrc{};
  toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toSrc.srcAccessMask = 0;
  toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  toSrc.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toSrc.image = image;
  toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  toSrc.subresourceRange.baseMipLevel = 0u;
  toSrc.subresourceRange.levelCount = 1u;
  toSrc.subresourceRange.baseArrayLayer = 0u;
  toSrc.subresourceRange.layerCount = 1u;
  _dd->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0u, nullptr, 0u, nullptr, 1u, &toSrc);

  VkBufferImageCopy region{};
  region.bufferOffset = 0u;
  region.bufferRowLength = 0u;  // tightly packed
  region.bufferImageHeight = 0u;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0u;
  region.imageSubresource.baseArrayLayer = 0u;
  region.imageSubresource.layerCount = 1u;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {w, h, 1u};
  _dd->CmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      buf, 1u, &region);

  VkImageMemoryBarrier toPresent{};
  toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  toPresent.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  toPresent.dstAccessMask = 0;
  toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toPresent.image = image;
  toPresent.subresourceRange = toSrc.subresourceRange;
  _dd->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0u, nullptr, 0u, nullptr, 1u,
      &toPresent);

  _dd->EndCommandBuffer(cb);

  // Submit, waiting on the present's wait semaphores (so we run after the
  // app's render finished). Signal a fence; CPU-wait the fence.
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  _dd->CreateFence(_dd->device, &fci, nullptr, &fence);

  std::vector<VkPipelineStageFlags> waitStages(_waitSemaphoreCount,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.waitSemaphoreCount = _waitSemaphoreCount;
  si.pWaitSemaphores = _pWaitSemaphores;
  si.pWaitDstStageMask = waitStages.empty() ? nullptr : waitStages.data();
  si.commandBufferCount = 1u;
  si.pCommandBuffers = &cb;

  VkResult sr = _dd->QueueSubmit(_queue, 1u, &si, fence);
  if (sr != VK_SUCCESS)
  {
    LayerLog("present: QueueSubmit failed (%d)", static_cast<int>(sr));
    _dd->DestroyFence(_dd->device, fence, nullptr);
    _dd->FreeMemory(_dd->device, mem, nullptr);
    _dd->DestroyBuffer(_dd->device, buf, nullptr);
    _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);
    return sr;
  }

  // Wait up to 2s for the copy.
  _dd->WaitForFences(_dd->device, 1u, &fence, VK_TRUE, 2000000000ull);
  _dd->DestroyFence(_dd->device, fence, nullptr);

  // Map and write PPM.
  void *mapped = nullptr;
  _dd->MapMemory(_dd->device, mem, 0u, bufSize, 0, &mapped);
  if (mapped == nullptr)
  {
    LayerLog("present: MapMemory failed");
    _dd->FreeMemory(_dd->device, mem, nullptr);
    _dd->DestroyBuffer(_dd->device, buf, nullptr);
    _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);
    return VK_SUCCESS;
  }

  std::vector<uint8_t> rgb(static_cast<std::size_t>(w) * h * 3u);
  const auto *src = static_cast<const uint8_t *>(mapped);
  for (uint32_t y = 0u; y < h; ++y)
  {
    for (uint32_t x = 0u; x < w; ++x)
    {
      const std::size_t s = (static_cast<std::size_t>(y) * w + x) * bpp;
      const std::size_t d = (static_cast<std::size_t>(y) * w + x) * 3u;
      if (swizzleBgr)
      {
        rgb[d + 0u] = src[s + 2u];
        rgb[d + 1u] = src[s + 1u];
        rgb[d + 2u] = src[s + 0u];
      }
      else
      {
        rgb[d + 0u] = src[s + 0u];
        rgb[d + 1u] = src[s + 1u];
        rgb[d + 2u] = src[s + 2u];
      }
    }
  }
  _dd->UnmapMemory(_dd->device, mem);

  const char *pathPrefix = std::getenv("GZ_SWAPCHAIN_DUMP_PATH");
  if (pathPrefix == nullptr || pathPrefix[0] == '\0')
    pathPrefix = "/tmp/gz_swapchain_dump";
  char buf2[512];
  std::snprintf(buf2, sizeof(buf2), "%s.frame_%04u.ppm", pathPrefix, _frame);
  WritePpm(buf2, w, h, rgb.data());
  std::snprintf(buf2, sizeof(buf2), "%s.frame_%04u.txt", pathPrefix, _frame);
  WriteSidecar(buf2, _sc, _frame, _imageIndex,
      srgb ? "format is sRGB; values are in sRGB-encoded space" : nullptr);

  LayerLog("present: dumped frame %u image_index=%u %ux%u format=%d",
      _frame, _imageIndex, w, h, static_cast<int>(_sc.format));

  // Clean up per-frame resources.
  _dd->FreeMemory(_dd->device, mem, nullptr);
  _dd->DestroyBuffer(_dd->device, buf, nullptr);
  _dd->FreeCommandBuffers(_dd->device, pool, 1u, &cb);

  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_QueuePresentKHR(VkQueue _queue,
    const VkPresentInfoKHR *_pPresent)
{
  DeviceData *dd = GetDeviceByQueue(_queue);
  if (dd == nullptr || dd->QueuePresentKHR == nullptr)
  {
    // Shouldn't happen, but fall through.
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  const uint32_t maxFrames = SafeMaxFrames();
  const uint32_t myFrame = gDumpedFrames.fetch_add(1u);
  bool consumedSemaphores = false;
  if (myFrame < maxFrames)
  {
    // The queue family of `_queue` isn't trivially known here; the VkQueue
    // handle doesn't expose it. Best practical approach: try queue family 0
    // for the cmd pool (the test only uses queue family 0). A robust layer
    // would track queue families at vkGetDeviceQueue() time -- left as a
    // TODO since the only consumer for now is the gz-gui test.
    const uint32_t qFamilyGuess = 0u;
    for (uint32_t i = 0u; i < _pPresent->swapchainCount; ++i)
    {
      SwapchainData *sc = nullptr;
      {
        std::lock_guard<std::mutex> lk(gMutex);
        auto it = gSwapchainMap.find(_pPresent->pSwapchains[i]);
        if (it != gSwapchainMap.end())
          sc = it->second;
      }
      if (sc == nullptr)
        continue;
      const uint32_t imageIndex = _pPresent->pImageIndices[i];
      if (imageIndex >= sc->images.size())
        continue;
      // We consume the wait semaphores on the FIRST swapchain only; subsequent
      // swapchains see them as already consumed. In practice _pPresent has 1.
      const uint32_t waits = consumedSemaphores ? 0u
          : _pPresent->waitSemaphoreCount;
      const VkSemaphore *pWaits = consumedSemaphores ? nullptr
          : _pPresent->pWaitSemaphores;
      DumpOneSwapchainImage(dd, _queue, qFamilyGuess, *sc, imageIndex,
          waits, pWaits, myFrame);
      if (waits > 0u)
        consumedSemaphores = true;
    }
  }

  // Forward to the next layer / driver, with the wait semaphores cleared if
  // we already consumed them above.
  if (consumedSemaphores)
  {
    VkPresentInfoKHR pi = *_pPresent;
    pi.waitSemaphoreCount = 0u;
    pi.pWaitSemaphores = nullptr;
    return dd->QueuePresentKHR(_queue, &pi);
  }
  return dd->QueuePresentKHR(_queue, _pPresent);
}

// ---------------------------------------------------------------------------
// Swapchain tracking
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateSwapchainKHR(VkDevice _device,
    const VkSwapchainCreateInfoKHR *_pCreate,
    const VkAllocationCallbacks *_pAlloc, VkSwapchainKHR *_pSwapchain)
{
  DeviceData *dd = GetDevice(_device);
  if (dd == nullptr || dd->CreateSwapchainKHR == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  VkResult r = dd->CreateSwapchainKHR(_device, _pCreate, _pAlloc, _pSwapchain);
  if (r == VK_SUCCESS)
  {
    auto *sd = new SwapchainData();
    sd->device = _device;
    sd->format = _pCreate->imageFormat;
    sd->extent = _pCreate->imageExtent;
    std::lock_guard<std::mutex> lk(gMutex);
    gSwapchainMap[*_pSwapchain] = sd;
    LayerLog("CreateSwapchainKHR: %p %ux%u format=%d",
        static_cast<void *>(*_pSwapchain), sd->extent.width, sd->extent.height,
        static_cast<int>(sd->format));
  }
  return r;
}

VKAPI_ATTR void VKAPI_CALL Layer_DestroySwapchainKHR(VkDevice _device,
    VkSwapchainKHR _swapchain, const VkAllocationCallbacks *_pAlloc)
{
  DeviceData *dd = GetDevice(_device);
  if (dd == nullptr)
    return;
  {
    std::lock_guard<std::mutex> lk(gMutex);
    auto it = gSwapchainMap.find(_swapchain);
    if (it != gSwapchainMap.end())
    {
      delete it->second;
      gSwapchainMap.erase(it);
    }
  }
  if (dd->DestroySwapchainKHR != nullptr)
    dd->DestroySwapchainKHR(_device, _swapchain, _pAlloc);
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_GetSwapchainImagesKHR(VkDevice _device,
    VkSwapchainKHR _swapchain, uint32_t *_pCount, VkImage *_pImages)
{
  DeviceData *dd = GetDevice(_device);
  if (dd == nullptr || dd->GetSwapchainImagesKHR == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  VkResult r = dd->GetSwapchainImagesKHR(_device, _swapchain, _pCount,
      _pImages);
  if (r == VK_SUCCESS && _pImages != nullptr && _pCount != nullptr)
  {
    std::lock_guard<std::mutex> lk(gMutex);
    auto it = gSwapchainMap.find(_swapchain);
    if (it != gSwapchainMap.end())
    {
      it->second->images.assign(_pImages, _pImages + *_pCount);
      LayerLog("GetSwapchainImagesKHR: tracked %u images for swapchain %p",
          *_pCount, static_cast<void *>(_swapchain));
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// Instance / Device chain init
// ---------------------------------------------------------------------------

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Layer_GetDeviceProcAddr(
    VkDevice _device, const char *_pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Layer_GetInstanceProcAddr(
    VkInstance _instance, const char *_pName);

VKAPI_ATTR void VKAPI_CALL Layer_DestroyDevice(VkDevice _device,
    const VkAllocationCallbacks *_pAlloc)
{
  DeviceData *dd = GetDevice(_device);
  if (dd == nullptr)
    return;
  // Tear down cmd pools.
  for (auto &kv : dd->cmdPools)
  {
    if (kv.second != VK_NULL_HANDLE && dd->DestroyCommandPool != nullptr)
      dd->DestroyCommandPool(_device, kv.second, nullptr);
  }
  PFN_vkDestroyDevice next = dd->DestroyDevice;
  {
    std::lock_guard<std::mutex> lk(gMutex);
    gDeviceMap.erase(DispatchKey(_device));
  }
  delete dd;
  if (next != nullptr)
    next(_device, _pAlloc);
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateDevice(VkPhysicalDevice _phys,
    const VkDeviceCreateInfo *_pCreate, const VkAllocationCallbacks *_pAlloc,
    VkDevice *_pDevice)
{
  VkLayerDeviceCreateInfo *link = FindLayerDeviceLink(_pCreate);
  if (link == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr nextGipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr nextGdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  link->u.pLayerInfo = link->u.pLayerInfo->pNext;

  InstanceData *id = GetInstanceByPhysical(_phys);
  if (id == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  auto fpCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      nextGipa(id->instance, "vkCreateDevice"));
  if (fpCreateDevice == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;

  VkResult r = fpCreateDevice(_phys, _pCreate, _pAlloc, _pDevice);
  if (r != VK_SUCCESS)
    return r;

  auto *dd = new DeviceData();
  dd->device = *_pDevice;
  dd->physicalDevice = _phys;
  dd->nextGdpa = nextGdpa;

#define LOAD_DEV(name)                                                        \
  dd->name = reinterpret_cast<PFN_vk##name>(nextGdpa(*_pDevice, "vk" #name))
  LOAD_DEV(DestroyDevice);
  LOAD_DEV(CreateSwapchainKHR);
  LOAD_DEV(DestroySwapchainKHR);
  LOAD_DEV(GetSwapchainImagesKHR);
  LOAD_DEV(QueuePresentKHR);
  LOAD_DEV(CreateCommandPool);
  LOAD_DEV(DestroyCommandPool);
  LOAD_DEV(AllocateCommandBuffers);
  LOAD_DEV(FreeCommandBuffers);
  LOAD_DEV(BeginCommandBuffer);
  LOAD_DEV(EndCommandBuffer);
  LOAD_DEV(ResetCommandBuffer);
  LOAD_DEV(CmdPipelineBarrier);
  LOAD_DEV(CmdCopyImageToBuffer);
  LOAD_DEV(CreateBuffer);
  LOAD_DEV(DestroyBuffer);
  LOAD_DEV(GetBufferMemoryRequirements);
  LOAD_DEV(AllocateMemory);
  LOAD_DEV(FreeMemory);
  LOAD_DEV(BindBufferMemory);
  LOAD_DEV(MapMemory);
  LOAD_DEV(UnmapMemory);
  LOAD_DEV(CreateFence);
  LOAD_DEV(DestroyFence);
  LOAD_DEV(WaitForFences);
  LOAD_DEV(ResetFences);
  LOAD_DEV(QueueSubmit);
#undef LOAD_DEV

  if (id->GetPhysicalDeviceMemoryProperties != nullptr)
    id->GetPhysicalDeviceMemoryProperties(_phys, &dd->memProps);

  {
    std::lock_guard<std::mutex> lk(gMutex);
    gDeviceMap[DispatchKey(*_pDevice)] = dd;
  }
  LayerLog("CreateDevice: %p (physical=%p)", static_cast<void *>(*_pDevice),
      static_cast<void *>(_phys));
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL Layer_DestroyInstance(VkInstance _instance,
    const VkAllocationCallbacks *_pAlloc)
{
  InstanceData *id = GetInstance(_instance);
  if (id == nullptr)
    return;
  PFN_vkDestroyInstance next = id->DestroyInstance;
  {
    std::lock_guard<std::mutex> lk(gMutex);
    gInstanceMap.erase(DispatchKey(_instance));
  }
  delete id;
  if (next != nullptr)
    next(_instance, _pAlloc);
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateInstance(
    const VkInstanceCreateInfo *_pCreate, const VkAllocationCallbacks *_pAlloc,
    VkInstance *_pInstance)
{
  VkLayerInstanceCreateInfo *link = FindLayerInstanceLink(_pCreate);
  if (link == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr nextGipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  link->u.pLayerInfo = link->u.pLayerInfo->pNext;

  auto fpCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      nextGipa(VK_NULL_HANDLE, "vkCreateInstance"));
  if (fpCreateInstance == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  VkResult r = fpCreateInstance(_pCreate, _pAlloc, _pInstance);
  if (r != VK_SUCCESS)
    return r;

  auto *id = new InstanceData();
  id->instance = *_pInstance;
  id->nextGipa = nextGipa;
  id->DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
      nextGipa(*_pInstance, "vkDestroyInstance"));
  id->EnumerateDeviceExtensionProperties =
      reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
          nextGipa(*_pInstance, "vkEnumerateDeviceExtensionProperties"));
  id->GetPhysicalDeviceMemoryProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
          nextGipa(*_pInstance, "vkGetPhysicalDeviceMemoryProperties"));

  {
    std::lock_guard<std::mutex> lk(gMutex);
    gInstanceMap[DispatchKey(*_pInstance)] = id;
  }
  LayerLog("CreateInstance: %p", static_cast<void *>(*_pInstance));
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// Layer enumeration
// ---------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL Layer_EnumerateInstanceLayerProperties(
    uint32_t *_pCount, VkLayerProperties *_pProps)
{
  if (_pProps == nullptr)
  {
    *_pCount = 1u;
    return VK_SUCCESS;
  }
  if (*_pCount < 1u)
    return VK_INCOMPLETE;
  VkLayerProperties p{};
  std::snprintf(p.layerName, VK_MAX_EXTENSION_NAME_SIZE, "%s", kLayerName);
  p.specVersion = VK_API_VERSION_1_2;
  p.implementationVersion = 1u;
  std::snprintf(p.description, VK_MAX_DESCRIPTION_SIZE,
      "gz-gui regression-test swapchain dump");
  _pProps[0] = p;
  *_pCount = 1u;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_EnumerateInstanceExtensionProperties(
    const char *_pLayerName, uint32_t *_pCount,
    VkExtensionProperties *_pProps)
{
  (void)_pProps;
  if (_pLayerName != nullptr && std::strcmp(_pLayerName, kLayerName) == 0)
  {
    *_pCount = 0u;
    return VK_SUCCESS;
  }
  // Pass through.
  return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_EnumerateDeviceLayerProperties(
    VkPhysicalDevice, uint32_t *_pCount, VkLayerProperties *_pProps)
{
  return Layer_EnumerateInstanceLayerProperties(_pCount, _pProps);
}

VKAPI_ATTR VkResult VKAPI_CALL Layer_EnumerateDeviceExtensionProperties(
    VkPhysicalDevice _phys, const char *_pLayerName, uint32_t *_pCount,
    VkExtensionProperties *_pProps)
{
  if (_pLayerName != nullptr && std::strcmp(_pLayerName, kLayerName) == 0)
  {
    *_pCount = 0u;
    return VK_SUCCESS;
  }
  InstanceData *id = GetInstanceByPhysical(_phys);
  if (id == nullptr || id->EnumerateDeviceExtensionProperties == nullptr)
    return VK_ERROR_INITIALIZATION_FAILED;
  return id->EnumerateDeviceExtensionProperties(_phys, _pLayerName, _pCount,
      _pProps);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#define INTERCEPT(name)                                                       \
  if (std::strcmp(_pName, "vk" #name) == 0)                                   \
    return reinterpret_cast<PFN_vkVoidFunction>(Layer_##name)

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Layer_GetDeviceProcAddr(
    VkDevice _device, const char *_pName)
{
  INTERCEPT(GetDeviceProcAddr);
  INTERCEPT(DestroyDevice);
  INTERCEPT(CreateSwapchainKHR);
  INTERCEPT(DestroySwapchainKHR);
  INTERCEPT(GetSwapchainImagesKHR);
  INTERCEPT(QueuePresentKHR);
  if (_device == VK_NULL_HANDLE)
    return nullptr;
  DeviceData *dd = GetDevice(_device);
  if (dd == nullptr || dd->nextGdpa == nullptr)
    return nullptr;
  return dd->nextGdpa(_device, _pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Layer_GetInstanceProcAddr(
    VkInstance _instance, const char *_pName)
{
  INTERCEPT(GetInstanceProcAddr);
  INTERCEPT(CreateInstance);
  INTERCEPT(DestroyInstance);
  INTERCEPT(EnumerateInstanceLayerProperties);
  INTERCEPT(EnumerateInstanceExtensionProperties);
  INTERCEPT(EnumerateDeviceLayerProperties);
  INTERCEPT(EnumerateDeviceExtensionProperties);
  INTERCEPT(CreateDevice);
  INTERCEPT(GetDeviceProcAddr);
  // Device entry points also need to be returnable via Layer_GIPA so the
  // loader can build the device dispatch table.
  INTERCEPT(DestroyDevice);
  INTERCEPT(CreateSwapchainKHR);
  INTERCEPT(DestroySwapchainKHR);
  INTERCEPT(GetSwapchainImagesKHR);
  INTERCEPT(QueuePresentKHR);
  if (_instance == VK_NULL_HANDLE)
    return nullptr;
  InstanceData *id = GetInstance(_instance);
  if (id == nullptr || id->nextGipa == nullptr)
    return nullptr;
  return id->nextGipa(_instance, _pName);
}

#undef INTERCEPT
}  // namespace

// ---------------------------------------------------------------------------
// Exported entry points (per Vulkan layer interface 2).
// ---------------------------------------------------------------------------

LAYER_EXPORT VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *_pIface)
{
  if (_pIface->loaderLayerInterfaceVersion > 2u)
    _pIface->loaderLayerInterfaceVersion = 2u;
  _pIface->pfnGetInstanceProcAddr = Layer_GetInstanceProcAddr;
  _pIface->pfnGetDeviceProcAddr = Layer_GetDeviceProcAddr;
  _pIface->pfnGetPhysicalDeviceProcAddr = nullptr;
  return VK_SUCCESS;
}

LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance _instance, const char *_pName)
{
  return Layer_GetInstanceProcAddr(_instance, _pName);
}

LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice _device, const char *_pName)
{
  return Layer_GetDeviceProcAddr(_device, _pName);
}

// NOLINTEND
