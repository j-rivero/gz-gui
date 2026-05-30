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

// Regression test pinning down what works (and what doesn't) in
// QSGSimpleTextureNode + QSGVulkanTexture::fromNative -- the path used by
// MinimalScene's Vulkan RHI bridge (MinimalSceneRhiVulkan).
//
// Context: the O3DE/Atom zero-copy bring-up (May 2026) traced a grey-viewport
// failure to this rendering path. Cross-device sampling of the imported image
// is correct (a compute-shader texelFetch on Qt's VkDevice reads the
// producer's pixels byte-identically -- see gz-rendering's
// o3de/docs/zero-copy-interop-findings.md), yet QSGSimpleTextureNode's draw
// of the QSGVulkanTexture::fromNative-wrapped image renders uniform on the
// window. This file isolates the variables.
//
// Test 1 (FromNativeRendersLinearPattern) -- PASSES today:
//   The simplest possible fromNative case. A LINEAR-tiling, host-visible
//   VkImage on Qt's OWN VkDevice, host-written with a four-quadrant pattern.
//   Strips away every "is the producer broken?" / cross-device / OPTIMAL
//   tiling variable. PASS locks in the empirical finding that the *simple*
//   fromNative + QSGSimpleTextureNode path is correct.
//
// Test 2 (FromNativeRendersOptimalPattern) -- PASSES today:
//   OPTIMAL tiling on Qt's own VkDevice, COLOR_ATTACHMENT | SAMPLED |
//   TRANSFER_DST usage, populated via a staging buffer +
//   vkCmdCopyBufferToImage on Qt's own queue, then transitioned to
//   SHADER_READ_ONLY_OPTIMAL. Matches the O3DE-exported image's usage flags
//   but on a single device. The PASS isolates the production bug to the
//   *cross-device* dimension: OPTIMAL+COLOR_ATTACHMENT+fromNative is fine
//   on its own; only when the VkImage was created on (and exported by) a
//   SECOND VkDevice does the QSGSimpleTextureNode draw render uniform.
//
// IMPORTANT CAVEAT (recorded 2026-05-30): Tests 1-4 all PASS but their
// assertion is on QQuickWindow::grabWindow() output, not on the actual
// swapchain presentation. The two diverge: the production O3DE demo shows
// uniform 148 (Atom's background-clear colour) on screen while the same
// demo's consumer-side dumps (transfer-copy + sampler probe) show the
// producer's correct shapes, and grabWindow() of the test's view reads the
// pattern correctly even when the swapchain present is uniform white.
//
// Test 5 (FromNativeSwapchainPresentsPattern) closes this gap: it loads an
// in-tree Vulkan layer (test/regression/swapchain_dump_layer/) that hooks
// vkQueuePresentKHR, dumps the about-to-be-presented swapchain image to a
// PPM, and asserts on the PPM. Test 5 currently FAILS on NVIDIA proprietary
// driver 580 + Qt 6 with a uniform-white swapchain -- the clean isolated
// reproduction of the production bug. The Atom/o3de/cross-device path was
// a red herring: the bug exists with pure Qt, single device, single frame.
//
// Test 3 (FromNativeRendersImportedFdPattern) -- PASSES today:
//   The closest possible test-level analog to the production O3DE setup.
//   A second, private VkInstance/VkDevice stands in for Atom: it creates
//   an OPTIMAL R8G8B8A8_UNORM VkImage with VK_EXTERNAL_MEMORY_HANDLE_TYPE_
//   OPAQUE_FD_BIT + dedicated allocation (see gz-rendering/o3de/docs/
//   zero-copy-interop-findings.md hypotheses 12-15 for why dedicated is
//   required on NVIDIA), staging-uploads the four-quadrant pattern,
//   transitions to SHADER_READ_ONLY_OPTIMAL with a QFOT release to
//   VK_QUEUE_FAMILY_EXTERNAL, and exports the memory FD via
//   vkGetMemoryFdKHR. Qt's VkDevice imports the FD with VkMemory
//   DedicatedAllocateInfo (mirroring the producer), QFOT-acquires onto
//   Qt's queue, and the test wraps the resulting VkImage via fromNative
//   + QSGSimpleTextureNode. Skips cleanly if VK_KHR_external_memory_fd
//   isn't available end-to-end.
//
//   This test was EXPECTED to fail and reproduce the production bug. It
//   passes. The pattern shapes display correctly on screen even after a
//   full cross-device, OPAQUE_FD, OPTIMAL, dedicated-allocation, QFOT
//   round-trip.
//
//   So the bug is NOT in the QSGSimpleTextureNode + fromNative + cross-
//   device-FD-import path itself. It must be in something the production
//   O3DE setup adds that this test does not:
//     * Atom's pipeline leaving the image in some compressed/transient
//       state different from a clean vkCmdCopyBufferToImage.
//     * MinimalScene's threading model (render thread + Qt scene-graph
//       thread + Atom thread) -- this test uses a single QQuickItem with
//       direct updatePaintNode and no thread handoff.
//     * Re-import-on-resize churn: production creates and destroys
//       imported VkImages each window resize (the "retire old imports"
//       fix kept them alive past Qt's in-flight frame, but the wrong
//       handle could still be referenced by the QSGSimpleTextureNode).
//     * Camera->RenderTextureMetalId() pointer stability across frames.
//   The next focused diagnostic should be on MinimalScene's
//   TextureNodeRhiVulkan handoff with the GZ_GUI_VULKAN_DIAG prints
//   enabled, comparing what handle gets passed to fromNative across
//   frames against what the producer reports.

#include <gtest/gtest.h>

#include <QtGlobal>
// qtgui-config.h defines QT_FEATURE_vulkan, which QT_CONFIG(vulkan) consults.
// Without it the QT_CONFIG macro divides by an undefined token below.
#include <QtGui/qtguiglobal.h>

#include <gz/utils/ExtraTestMacros.hh>

// Compiles on every platform. The test body is guarded by QT_CONFIG(vulkan) so
// builds where Qt was configured without Vulkan support (or with a Qt < 6.0
// that lacks the QSGVulkanTexture native interface) emit a GTEST_SKIP() rather
// than a build failure. Mirrors MinimalScene's GZ_GUI_HAVE_VULKAN gating
// (src/plugins/minimal_scene/MinimalSceneRhi.hh).
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0) && QT_CONFIG(vulkan)
#define GZ_GUI_TEST_HAVE_VULKAN 1
#else
#define GZ_GUI_TEST_HAVE_VULKAN 0
#endif

#if GZ_GUI_TEST_HAVE_VULKAN
// Vulkan MUST be included before any Qt header. Qt's Vulkan integration
// defines VK_NO_PROTOTYPES (it dispatches through QVulkanFunctions); if Qt
// headers come first, every later <vulkan/vulkan.h> sees prototypes off and
// our direct vkCreateImage / vkAllocateMemory / vkMapMemory calls fail to
// resolve. Including Vulkan first means we get real prototypes, and Qt's
// subsequent VK_NO_PROTOTYPES only affects its own internal dispatch. Linking
// against the system Vulkan loader (Vulkan::Vulkan in test/regression/
// CMakeLists.txt) resolves the prototype symbols at link time.
#include <vulkan/vulkan.h>

#include <QGuiApplication>
#include <QImage>
#include <QQuickItem>
#include <QQuickView>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QSGSimpleTextureNode>
#include <QSGTexture>
#include <QSize>
#include <QtTest/QtTest>

#include <chrono>   // steady_clock + wall-clock deadline for Test 5
#include <cstdio>   // PPM reader for Test 5: std::FILE / std::fopen / std::fread
#include <cstdlib>  // std::abs for the pattern-colour tolerance check; ::setenv
#include <cstring>  // std::strcmp for Test 5 platform-detect
#include <set>
#include <string>   // PPM path strings for Test 5
#include <utility>
#include <vector>

#include <unistd.h>  // ::close for the FD cleanup path on import failure
#endif  // GZ_GUI_TEST_HAVE_VULKAN

#if GZ_GUI_TEST_HAVE_VULKAN
namespace
{
// Pattern dimensions. Small to keep the test fast; the test checks unique-colour
// count, not exact pixel positions, so resolution-independence is preserved.
constexpr int kPatternW = 64;
constexpr int kPatternH = 64;

// Four-quadrant solid pattern: red / green / blue / yellow. Choosing four
// highly distinct primaries (no two within a small RGB distance of each other)
// so the "did the texture content arrive on screen?" assertion below is
// unambiguous.
//
// IMPORTANT: these are stored as little-endian uint32_t, so the byte layout
// in memory (for a R8G8B8A8_UNORM image) is reversed from the literal: the
// constant 0xAABBGGRR ends up as bytes [RR, GG, BB, AA] = [R, G, B, A]. So
// the literal for "R=0xFF, G=0x00, B=0x00, A=0xFF" (red, opaque) is
// 0xFF0000FF. A first version of this test mistakenly used the literal in
// big-endian RGBA order, which on little-endian x86 wrote magenta/yellow
// where it wanted green/blue -- and the resulting test failure (faithfully
// showing the broken pattern!) was the experiment that *eliminated* the
// hypothesis "fromNative + QSGSimpleTextureNode is broken at the simplest
// level". Lesson recorded so the byte-order trap is visible to future readers.
constexpr uint32_t kPatternRGBA[4] = {
    0xFF0000FFu,  // red:    bytes [FF, 00, 00, FF]
    0xFF00FF00u,  // green:  bytes [00, FF, 00, FF]
    0xFFFF0000u,  // blue:   bytes [00, 00, FF, FF]
    0xFF00FFFFu,  // yellow: bytes [FF, FF, 00, FF]
};

uint32_t QuadrantColour(int _x, int _y)
{
  const int qx = (_x < kPatternW / 2) ? 0 : 1;
  const int qy = (_y < kPatternH / 2) ? 0 : 1;
  return kPatternRGBA[qy * 2 + qx];
}

// RAII holder for the Vulkan objects the test creates on Qt's QRhi device.
struct PatternImage
{
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;

  ~PatternImage()
  {
    if (this->device != VK_NULL_HANDLE)
    {
      if (this->image != VK_NULL_HANDLE)
        vkDestroyImage(this->device, this->image, nullptr);
      if (this->memory != VK_NULL_HANDLE)
        vkFreeMemory(this->device, this->memory, nullptr);
    }
  }
};

// Obtain Qt's QRhi Vulkan handles from a shown QQuickWindow. Mirrors what
// MinimalSceneRhiVulkan does to inject Qt's VkDevice into the engine -- if this
// returns false the harness, not the bug, is at fault.
bool GetQtVulkanHandles(QQuickWindow *_window, PatternImage *_out)
{
  QSGRendererInterface *rif = _window->rendererInterface();
  if (rif == nullptr ||
      rif->graphicsApi() != QSGRendererInterface::Vulkan)
  {
    return false;
  }
  _out->instance = *static_cast<VkInstance *>(rif->getResource(
      _window, QSGRendererInterface::VulkanInstanceResource));
  _out->physicalDevice = *static_cast<VkPhysicalDevice *>(rif->getResource(
      _window, QSGRendererInterface::PhysicalDeviceResource));
  _out->device = *static_cast<VkDevice *>(rif->getResource(
      _window, QSGRendererInterface::DeviceResource));
  return _out->instance != VK_NULL_HANDLE &&
      _out->physicalDevice != VK_NULL_HANDLE &&
      _out->device != VK_NULL_HANDLE;
}

// Create a LINEAR, host-visible, R8G8B8A8_UNORM VkImage and host-fill it with
// the four-quadrant pattern. Transitions to SHADER_READ_ONLY_OPTIMAL so
// QSGVulkanTexture::fromNative can wrap it directly. The simplest possible
// "the texture has content" setup: no cross-device, no tiling-swizzle, no FD
// import -- isolates the bug to Qt's draw of fromNative.
bool CreatePatternImage(PatternImage *_io)
{
  VkImageCreateInfo imgInfo{};
  imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgInfo.imageType = VK_IMAGE_TYPE_2D;
  imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imgInfo.extent = {kPatternW, kPatternH, 1u};
  imgInfo.mipLevels = 1u;
  imgInfo.arrayLayers = 1u;
  imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imgInfo.tiling = VK_IMAGE_TILING_LINEAR;
  imgInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  imgInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
  imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateImage(_io->device, &imgInfo, nullptr, &_io->image) != VK_SUCCESS)
    return false;

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(_io->device, _io->image, &req);
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(_io->physicalDevice, &mp);
  uint32_t typeIdx = UINT32_MAX;
  const VkMemoryPropertyFlags want =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (uint32_t i = 0u; i < mp.memoryTypeCount; ++i)
  {
    if ((req.memoryTypeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & want) == want)
    {
      typeIdx = i;
      break;
    }
  }
  if (typeIdx == UINT32_MAX)
    return false;

  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = typeIdx;
  if (vkAllocateMemory(_io->device, &alloc, nullptr, &_io->memory) != VK_SUCCESS)
    return false;
  if (vkBindImageMemory(_io->device, _io->image, _io->memory, 0u) != VK_SUCCESS)
    return false;

  // Fill via the LINEAR subresource layout (rowPitch may exceed width*4).
  VkImageSubresource sub{};
  sub.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  VkSubresourceLayout layout{};
  vkGetImageSubresourceLayout(_io->device, _io->image, &sub, &layout);
  void *mapped = nullptr;
  if (vkMapMemory(_io->device, _io->memory, 0u, req.size, 0u, &mapped)
      != VK_SUCCESS)
  {
    return false;
  }
  auto *base = static_cast<uint8_t *>(mapped) + layout.offset;
  for (int y = 0; y < kPatternH; ++y)
  {
    auto *row = reinterpret_cast<uint32_t *>(base + y * layout.rowPitch);
    for (int x = 0; x < kPatternW; ++x)
      row[x] = QuadrantColour(x, y);
  }
  vkUnmapMemory(_io->device, _io->memory);
  // PREINITIALIZED + HOST_COHERENT is observable by the sampler without a
  // pipeline barrier on LINEAR tiling per the Vulkan spec, so no transition
  // command-buffer is needed. fromNative is told the layout below.
  return true;
}

// Find a memory type that satisfies _typeBits + every flag in _wantProps.
int FindMemoryType(VkPhysicalDevice _phys, uint32_t _typeBits,
    VkMemoryPropertyFlags _wantProps)
{
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(_phys, &mp);
  for (uint32_t i = 0u; i < mp.memoryTypeCount; ++i)
  {
    if ((_typeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & _wantProps) == _wantProps)
      return static_cast<int>(i);
  }
  return -1;
}

// Create an OPTIMAL-tiling VkImage on Qt's device with the same usage flags
// MinimalScene's Atom-exported image has (COLOR_ATTACHMENT | SAMPLED |
// TRANSFER_DST), populate it via a staging buffer + vkCmdCopyBufferToImage,
// then transition to SHADER_READ_ONLY_OPTIMAL. This matches the production
// O3DE-exported image's image/layout lifecycle (modulo cross-device sharing),
// so a failure here vs Test 1's LINEAR PASS isolates the bug to OPTIMAL
// tiling on its own, even without a second VkDevice in the picture.
//
// Uses Qt's own VkQueue (acquired via the QSGRendererInterface
// CommandQueueResource) for the staging upload submit; vkQueueWaitIdle
// before returning so Qt's render thread can resume safely.
bool CreateOptimalPatternImage(VkQueue _queue, uint32_t _queueFamily,
    PatternImage *_io)
{
  // 1) OPTIMAL image, COLOR_ATTACHMENT | SAMPLED | TRANSFER_DST usage.
  VkImageCreateInfo imgInfo{};
  imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgInfo.imageType = VK_IMAGE_TYPE_2D;
  imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imgInfo.extent = {kPatternW, kPatternH, 1u};
  imgInfo.mipLevels = 1u;
  imgInfo.arrayLayers = 1u;
  imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateImage(_io->device, &imgInfo, nullptr, &_io->image) != VK_SUCCESS)
    return false;
  VkMemoryRequirements imgReq{};
  vkGetImageMemoryRequirements(_io->device, _io->image, &imgReq);
  const int imgType = FindMemoryType(_io->physicalDevice, imgReq.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (imgType < 0)
    return false;
  VkMemoryAllocateInfo imgAlloc{};
  imgAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  imgAlloc.allocationSize = imgReq.size;
  imgAlloc.memoryTypeIndex = static_cast<uint32_t>(imgType);
  if (vkAllocateMemory(_io->device, &imgAlloc, nullptr, &_io->memory)
      != VK_SUCCESS)
    return false;
  vkBindImageMemory(_io->device, _io->image, _io->memory, 0u);

  // 2) Staging buffer (HOST_VISIBLE) carrying the pattern.
  const VkDeviceSize stagingSize =
      static_cast<VkDeviceSize>(kPatternW) * kPatternH * 4u;
  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = stagingSize;
  bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer staging = VK_NULL_HANDLE;
  if (vkCreateBuffer(_io->device, &bufInfo, nullptr, &staging) != VK_SUCCESS)
    return false;
  VkMemoryRequirements bufReq{};
  vkGetBufferMemoryRequirements(_io->device, staging, &bufReq);
  const int bufType = FindMemoryType(_io->physicalDevice, bufReq.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (bufType < 0)
  {
    vkDestroyBuffer(_io->device, staging, nullptr);
    return false;
  }
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  VkMemoryAllocateInfo bufAlloc{};
  bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  bufAlloc.allocationSize = bufReq.size;
  bufAlloc.memoryTypeIndex = static_cast<uint32_t>(bufType);
  if (vkAllocateMemory(_io->device, &bufAlloc, nullptr, &stagingMem)
      != VK_SUCCESS)
  {
    vkDestroyBuffer(_io->device, staging, nullptr);
    return false;
  }
  vkBindBufferMemory(_io->device, staging, stagingMem, 0u);
  void *mapped = nullptr;
  vkMapMemory(_io->device, stagingMem, 0u, stagingSize, 0u, &mapped);
  auto *base = static_cast<uint32_t *>(mapped);
  for (int y = 0; y < kPatternH; ++y)
    for (int x = 0; x < kPatternW; ++x)
      base[y * kPatternW + x] = QuadrantColour(x, y);
  vkUnmapMemory(_io->device, stagingMem);

  // 3) One-shot command buffer: UNDEFINED -> TRANSFER_DST, copy, TRANSFER_DST
  //    -> SHADER_READ_ONLY_OPTIMAL. Submit + vkQueueWaitIdle.
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = _queueFamily;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  vkCreateCommandPool(_io->device, &poolInfo, nullptr, &pool);
  VkCommandBufferAllocateInfo cbAlloc{};
  cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAlloc.commandPool = pool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1u;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(_io->device, &cbAlloc, &cmd);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);

  auto Barrier = [&](VkImageLayout _old, VkImageLayout _new,
      VkAccessFlags _srcA, VkAccessFlags _dstA,
      VkPipelineStageFlags _srcS, VkPipelineStageFlags _dstS)
  {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = _srcA;
    b.dstAccessMask = _dstA;
    b.oldLayout = _old;
    b.newLayout = _new;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = _io->image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vkCmdPipelineBarrier(cmd, _srcS, _dstS, 0u, 0u, nullptr, 0u, nullptr,
        1u, &b);
  };
  Barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      0u, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
  region.imageExtent = {kPatternW, kPatternH, 1u};
  vkCmdCopyBufferToImage(cmd, staging, _io->image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &region);
  Barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1u;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(_queue, 1u, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(_queue);

  vkDestroyCommandPool(_io->device, pool, nullptr);
  vkDestroyBuffer(_io->device, staging, nullptr);
  vkFreeMemory(_io->device, stagingMem, nullptr);
  return true;
}

// ---------------------------------------------------------------------------
// Cross-device FD-import harness (FromNativeRendersImportedFdPattern, Test 3)
// ---------------------------------------------------------------------------
// The third test stands up a SECOND, private VkInstance + VkDevice on the same
// physical GPU as Qt, populates an exportable OPTIMAL VkImage there, exports
// its memory FD via vkGetMemoryFdKHR, then imports the FD onto Qt's VkDevice
// and wraps with QSGVulkanTexture::fromNative. This reproduces the cross-
// device case the production O3DE/Atom zero-copy path uses, isolated from
// Atom -- the test's "producer" stands in for Atom.
//
// All four helpers below mirror what the gz-rendering O3deVkImport.cc /
// O3deBackend.cc do in production but in test-friendly form:
//   * SetupProducerDevice    -- private VkInstance + VkDevice (matched to
//                               Qt's physical device), external_memory_fd
//                               extension enabled, function pointer cached.
//   * CreateAndExportImage   -- OPTIMAL VkImage with VkExternalMemoryImage
//                               CreateInfo + VkMemoryDedicatedAllocateInfo
//                               + VkExportMemoryAllocateInfo, staging-loaded
//                               with the pattern, transitioned to SHADER_READ
//                               with a QFOT release to VK_QUEUE_FAMILY_EXTERNAL,
//                               FD exported.
//   * ImportFdOntoQtDevice   -- creates a matching VkImage on Qt's device
//                               with VkExternalMemoryImageCreateInfo, allocates
//                               memory with VkImportMemoryFdInfoKHR +
//                               VkMemoryDedicatedAllocateInfo, binds.
//   * AcquireOntoQtQueue     -- QFOT acquire from EXTERNAL onto Qt's queue
//                               family, layout transition to SHADER_READ.
//
// Why dedicated allocation: on NVIDIA proprietary, a sub-allocated import of
// an OPAQUE_FD OPTIMAL image returns the wrong tile-swizzle to the sampler.
// gz-rendering/o3de/docs/zero-copy-interop-findings.md hypotheses 12-15 cover
// the bring-up that proved this; the helper attaches the dedicated info on
// both sides to mirror that fix.

struct ProducerDevice
{
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamily = UINT32_MAX;
  PFN_vkGetMemoryFdKHR getMemoryFdKHR = nullptr;

  ~ProducerDevice()
  {
    if (this->device != VK_NULL_HANDLE)
    {
      vkDeviceWaitIdle(this->device);
      vkDestroyDevice(this->device, nullptr);
    }
    if (this->instance != VK_NULL_HANDLE)
      vkDestroyInstance(this->instance, nullptr);
  }
};

// Tracks the producer-side VkImage / VkDeviceMemory until the test ends, so we
// can destroy them on the producer device after the consumer has finished
// sampling. The FD is consumed by Qt's vkAllocateMemory on import.
struct ProducerImage
{
  VkDevice device = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize allocationSize = 0u;

  ~ProducerImage()
  {
    if (this->device != VK_NULL_HANDLE)
    {
      if (this->image != VK_NULL_HANDLE)
        vkDestroyImage(this->device, this->image, nullptr);
      if (this->memory != VK_NULL_HANDLE)
        vkFreeMemory(this->device, this->memory, nullptr);
    }
  }
};

// Stand up a private VkInstance + VkDevice on the same physical device as Qt.
// Picks the first DISCRETE_GPU it finds (mirroring O3deVkInterop.cc); on the
// single-GPU systems this test was developed on, both VkInstances enumerate
// the same hardware as the first DISCRETE_GPU, so this matches by construction.
bool SetupProducerDevice(ProducerDevice *_out)
{
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "qsg_simple_texture_node_vulkan_producer";
  appInfo.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instInfo{};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;
  if (vkCreateInstance(&instInfo, nullptr, &_out->instance) != VK_SUCCESS)
    return false;

  uint32_t physCount = 0u;
  vkEnumeratePhysicalDevices(_out->instance, &physCount, nullptr);
  if (physCount == 0u)
    return false;
  std::vector<VkPhysicalDevice> physs(physCount);
  vkEnumeratePhysicalDevices(_out->instance, &physCount, physs.data());
  _out->physicalDevice = physs[0];
  for (VkPhysicalDevice p : physs)
  {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(p, &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
      _out->physicalDevice = p;
      break;
    }
  }

  uint32_t qfCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(_out->physicalDevice, &qfCount,
      nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(_out->physicalDevice, &qfCount,
      qfs.data());
  for (uint32_t i = 0u; i < qfCount; ++i)
  {
    if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      _out->queueFamily = i;
      break;
    }
  }
  if (_out->queueFamily == UINT32_MAX)
    return false;

  const char *exts[] = {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
  const float prio = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex = _out->queueFamily;
  queueInfo.queueCount = 1u;
  queueInfo.pQueuePriorities = &prio;
  VkDeviceCreateInfo devInfo{};
  devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devInfo.queueCreateInfoCount = 1u;
  devInfo.pQueueCreateInfos = &queueInfo;
  devInfo.enabledExtensionCount = 2u;
  devInfo.ppEnabledExtensionNames = exts;
  if (vkCreateDevice(_out->physicalDevice, &devInfo, nullptr, &_out->device)
      != VK_SUCCESS)
    return false;
  vkGetDeviceQueue(_out->device, _out->queueFamily, 0u, &_out->queue);

  _out->getMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      vkGetDeviceProcAddr(_out->device, "vkGetMemoryFdKHR"));
  return _out->getMemoryFdKHR != nullptr;
}

// Create an exportable OPTIMAL VkImage on the producer device, dedicated
// allocation with OPAQUE_FD export-memory, staging-upload the pattern, then
// transition the image to SHADER_READ_ONLY_OPTIMAL with a QFOT *release* to
// VK_QUEUE_FAMILY_EXTERNAL (so the consumer's acquire-from-EXTERNAL matches).
// Exports the memory FD via vkGetMemoryFdKHR. Returns the dedicated allocation
// size in *_outAllocSize -- the consumer needs the same size on import.
bool CreateAndExportImage(const ProducerDevice &_prod,
    ProducerImage *_outImg, int *_outFd, VkDeviceSize *_outAllocSize)
{
  _outImg->device = _prod.device;
  VkExternalMemoryImageCreateInfo extImg{};
  extImg.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  extImg.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  VkImageCreateInfo imgInfo{};
  imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgInfo.pNext = &extImg;
  imgInfo.imageType = VK_IMAGE_TYPE_2D;
  imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imgInfo.extent = {kPatternW, kPatternH, 1u};
  imgInfo.mipLevels = 1u;
  imgInfo.arrayLayers = 1u;
  imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateImage(_prod.device, &imgInfo, nullptr, &_outImg->image)
      != VK_SUCCESS)
    return false;

  VkMemoryRequirements imgReq{};
  vkGetImageMemoryRequirements(_prod.device, _outImg->image, &imgReq);
  const int imgType = FindMemoryType(_prod.physicalDevice,
      imgReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (imgType < 0)
    return false;
  VkExportMemoryAllocateInfo exportInfo{};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  VkMemoryDedicatedAllocateInfo dedicated{};
  dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated.image = _outImg->image;
  dedicated.pNext = &exportInfo;
  VkMemoryAllocateInfo imgAlloc{};
  imgAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  imgAlloc.pNext = &dedicated;
  imgAlloc.allocationSize = imgReq.size;
  imgAlloc.memoryTypeIndex = static_cast<uint32_t>(imgType);
  if (vkAllocateMemory(_prod.device, &imgAlloc, nullptr, &_outImg->memory)
      != VK_SUCCESS)
    return false;
  vkBindImageMemory(_prod.device, _outImg->image, _outImg->memory, 0u);
  _outImg->allocationSize = imgReq.size;

  // Staging upload + layout transitions (UNDEFINED -> TRANSFER_DST -> copy ->
  // SHADER_READ + QFOT release to EXTERNAL).
  const VkDeviceSize stagingSize =
      static_cast<VkDeviceSize>(kPatternW) * kPatternH * 4u;
  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = stagingSize;
  bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer staging = VK_NULL_HANDLE;
  vkCreateBuffer(_prod.device, &bufInfo, nullptr, &staging);
  VkMemoryRequirements bufReq{};
  vkGetBufferMemoryRequirements(_prod.device, staging, &bufReq);
  const int bufType = FindMemoryType(_prod.physicalDevice, bufReq.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  VkMemoryAllocateInfo bufAlloc{};
  bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  bufAlloc.allocationSize = bufReq.size;
  bufAlloc.memoryTypeIndex = static_cast<uint32_t>(bufType);
  vkAllocateMemory(_prod.device, &bufAlloc, nullptr, &stagingMem);
  vkBindBufferMemory(_prod.device, staging, stagingMem, 0u);
  void *mapped = nullptr;
  vkMapMemory(_prod.device, stagingMem, 0u, stagingSize, 0u, &mapped);
  auto *base = static_cast<uint32_t *>(mapped);
  for (int y = 0; y < kPatternH; ++y)
    for (int x = 0; x < kPatternW; ++x)
      base[y * kPatternW + x] = QuadrantColour(x, y);
  vkUnmapMemory(_prod.device, stagingMem);

  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = _prod.queueFamily;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  vkCreateCommandPool(_prod.device, &poolInfo, nullptr, &pool);
  VkCommandBufferAllocateInfo cbAlloc{};
  cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAlloc.commandPool = pool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1u;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(_prod.device, &cbAlloc, &cmd);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);

  auto Barrier = [&](VkImageLayout _old, VkImageLayout _new,
      VkAccessFlags _srcA, VkAccessFlags _dstA,
      VkPipelineStageFlags _srcS, VkPipelineStageFlags _dstS,
      uint32_t _srcQF, uint32_t _dstQF)
  {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = _srcA;
    b.dstAccessMask = _dstA;
    b.oldLayout = _old;
    b.newLayout = _new;
    b.srcQueueFamilyIndex = _srcQF;
    b.dstQueueFamilyIndex = _dstQF;
    b.image = _outImg->image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    vkCmdPipelineBarrier(cmd, _srcS, _dstS, 0u, 0u, nullptr, 0u, nullptr,
        1u, &b);
  };
  Barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      0u, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
  region.imageExtent = {kPatternW, kPatternH, 1u};
  vkCmdCopyBufferToImage(cmd, staging, _outImg->image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &region);
  // Combined TRANSFER_DST -> SHADER_READ + QFOT release to EXTERNAL. The
  // consumer (Qt's queue) issues the matching acquire from EXTERNAL.
  Barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_ACCESS_TRANSFER_WRITE_BIT, 0u,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      _prod.queueFamily, VK_QUEUE_FAMILY_EXTERNAL);
  vkEndCommandBuffer(cmd);
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1u;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(_prod.queue, 1u, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(_prod.queue);
  vkDestroyCommandPool(_prod.device, pool, nullptr);
  vkDestroyBuffer(_prod.device, staging, nullptr);
  vkFreeMemory(_prod.device, stagingMem, nullptr);

  // Export the memory FD. The consumer is then responsible for closing it via
  // vkAllocateMemory consumption on import.
  VkMemoryGetFdInfoKHR getInfo{};
  getInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  getInfo.memory = _outImg->memory;
  getInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  if (_prod.getMemoryFdKHR(_prod.device, &getInfo, _outFd) != VK_SUCCESS)
    return false;
  *_outAllocSize = imgReq.size;
  return true;
}

// Import the producer's exported FD onto Qt's VkDevice as an OPTIMAL VkImage
// with matching VkImageCreateInfo. The import uses VkMemoryDedicatedAllocate
// Info on the consumer side to mirror the producer's dedicated allocation
// (see hypothesis 12-15 in zero-copy-interop-findings.md). On success, the
// producer's FD is consumed by vkAllocateMemory.
bool ImportFdOntoQtDevice(const PatternImage &_qt, int _fd,
    VkDeviceSize _allocSize, VkImage *_outImage, VkDeviceMemory *_outMem)
{
  VkExternalMemoryImageCreateInfo extImg{};
  extImg.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  extImg.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  VkImageCreateInfo imgInfo{};
  imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgInfo.pNext = &extImg;
  imgInfo.imageType = VK_IMAGE_TYPE_2D;
  imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imgInfo.extent = {kPatternW, kPatternH, 1u};
  imgInfo.mipLevels = 1u;
  imgInfo.arrayLayers = 1u;
  imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateImage(_qt.device, &imgInfo, nullptr, _outImage) != VK_SUCCESS)
    return false;

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(_qt.device, *_outImage, &req);
  const int memType = FindMemoryType(_qt.physicalDevice, req.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (memType < 0)
    return false;
  VkImportMemoryFdInfoKHR importFd{};
  importFd.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
  importFd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  importFd.fd = _fd;
  VkMemoryDedicatedAllocateInfo dedicated{};
  dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated.image = *_outImage;
  dedicated.pNext = &importFd;
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.pNext = &dedicated;
  // Use the producer's reported dedicated allocation size, not Qt's reported
  // imgReq.size (which may differ slightly).
  alloc.allocationSize = _allocSize;
  alloc.memoryTypeIndex = static_cast<uint32_t>(memType);
  if (vkAllocateMemory(_qt.device, &alloc, nullptr, _outMem) != VK_SUCCESS)
    return false;
  // Dedicated allocation always binds at offset 0.
  vkBindImageMemory(_qt.device, *_outImage, *_outMem, 0u);
  return true;
}

// QFOT acquire from EXTERNAL onto Qt's queue family + layout transition to
// SHADER_READ_ONLY_OPTIMAL (matching the producer's release layout). Submits
// to Qt's own queue and vkQueueWaitIdles. The image is now ready for
// QSGVulkanTexture::fromNative.
bool AcquireOntoQtQueue(VkQueue _qtQueue, uint32_t _qtQF,
    const PatternImage &_qt, VkImage _image)
{
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = _qtQF;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  vkCreateCommandPool(_qt.device, &poolInfo, nullptr, &pool);
  VkCommandBufferAllocateInfo cbAlloc{};
  cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAlloc.commandPool = pool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1u;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(_qt.device, &cbAlloc, &cmd);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  VkImageMemoryBarrier b{};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.srcAccessMask = 0u;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
  b.dstQueueFamilyIndex = _qtQF;
  b.image = _image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0u, 0u, nullptr, 0u, nullptr,
      1u, &b);
  vkEndCommandBuffer(cmd);
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1u;
  submit.pCommandBuffers = &cmd;
  const VkResult res = vkQueueSubmit(_qtQueue, 1u, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(_qtQueue);
  vkDestroyCommandPool(_qt.device, pool, nullptr);
  return res == VK_SUCCESS;
}

// Decisive pattern check used by both LINEAR and OPTIMAL tests: each of the
// four quadrants of the rendered framebuffer must carry its expected pattern
// colour. A weaker "any non-uniform output" assertion would false-PASS on
// Qt's window chrome.
void AssertRenderedPattern(const QImage &_rendered, const char *_caseLabel)
{
  ASSERT_FALSE(_rendered.isNull()) << "grabWindow returned a null QImage";
  ASSERT_GE(_rendered.width(), kPatternW);
  ASSERT_GE(_rendered.height(), kPatternH);
  auto NearMatch = [](QRgb _got, uint32_t _wantRGBA, int _tol) -> bool {
    const int r = qRed(_got), g = qGreen(_got), b = qBlue(_got);
    const int wr = static_cast<int>(_wantRGBA & 0xFFu);
    const int wg = static_cast<int>((_wantRGBA >> 8) & 0xFFu);
    const int wb = static_cast<int>((_wantRGBA >> 16) & 0xFFu);
    return std::abs(r - wr) <= _tol && std::abs(g - wg) <= _tol &&
        std::abs(b - wb) <= _tol;
  };
  const int w = _rendered.width();
  const int h = _rendered.height();
  struct Probe { int x; int y; uint32_t want; const char *name; };
  const Probe probes[4] = {
      {w / 4,     h / 4,     kPatternRGBA[0], "red (top-left)"},
      {3 * w / 4, h / 4,     kPatternRGBA[1], "green (top-right)"},
      {w / 4,     3 * h / 4, kPatternRGBA[2], "blue (bottom-left)"},
      {3 * w / 4, 3 * h / 4, kPatternRGBA[3], "yellow (bottom-right)"},
  };
  for (const Probe &p : probes)
  {
    const QRgb got = _rendered.pixel(p.x, p.y);
    const uint32_t wantR = p.want & 0xFFu;
    const uint32_t wantG = (p.want >> 8) & 0xFFu;
    const uint32_t wantB = (p.want >> 16) & 0xFFu;
    const uint32_t wantRgb = (wantR << 16) | (wantG << 8) | wantB;
    EXPECT_TRUE(NearMatch(got, p.want, 16))
        << _caseLabel << ": QSGSimpleTextureNode + fromNative did not render "
           "the expected pattern colour at " << p.name
        << " (" << p.x << "," << p.y << "): got 0x"
        << QString::number(got, 16).toStdString() << " want 0x"
        << QString::number(wantRgb, 16).toStdString()
        << " (ARGB). See gz-rendering/o3de/docs/zero-copy-interop-findings.md.";
  }
}

// QQuickItem mirroring MinimalSceneRhiVulkan's role: on updatePaintNode it
// creates a QSGSimpleTextureNode wrapping the externally-owned VkImage via
// QSGVulkanTexture::fromNative. This is the EXACT path under test.
//
// Two size knobs:
//   * vkImageSize     -- the actual VkImage::extent (passed to vkCreateImage).
//   * fromNativeSize  -- the size we tell QSGVulkanTexture::fromNative.
// In Tests 1-3 the two match (the test creates the image at kPatternW x
// kPatternH and passes the same to fromNative). Test 4 deliberately
// mismatches them, mirroring the production O3DE setup where the consumer
// reports the camera's window size (e.g. 1024x670) to fromNative while the
// producer's exported VkImage is still at its initial size (e.g. 512x512).
//
// Texture-lifecycle knob (production-flavour):
//   * recreateEveryFrame -- when true, replicates MinimalSceneRhi
//                           Vulkan's pattern of `delete this->texture;
//                           this->texture = fromNative(...);` on every
//                           PrepareNode -- i.e. a NEW QSGVulkanTexture each
//                           frame, wrapping the SAME VkImage handle. Tests
//                           the hypothesis that rapidly-replaced fromNative
//                           wrappers (aliasing one VkImage) confuse Qt's
//                           QSGSimpleTextureNode draw.
class PatternItem : public QQuickItem
{
 public:
  PatternItem(QQuickItem *_parent, VkImage _image, QSize _fromNativeSize,
      bool _recreateEveryFrame = false)
      : QQuickItem(_parent), image(_image), fromNativeSize(_fromNativeSize),
        recreateEveryFrame(_recreateEveryFrame)
  {
    this->setFlag(ItemHasContents);
  }

  QSGNode *updatePaintNode(QSGNode *_old,
      QQuickItem::UpdatePaintNodeData *) override
  {
    auto *node = static_cast<QSGSimpleTextureNode *>(_old);
    if (node == nullptr)
    {
      node = new QSGSimpleTextureNode();
      this->WrapAndSet(node);
    }
    else if (this->recreateEveryFrame)
    {
      // Mirror what MinimalSceneRhiVulkan::CreateTexture does every frame:
      // delete the previous QSGVulkanTexture wrapper, allocate a new one
      // around the same VkImage handle, install on the node, mark dirty.
      this->WrapAndSet(node);
      node->markDirty(QSGNode::DirtyMaterial);
    }
    node->setRect(this->boundingRect());
    return node;
  }

 private:
  void WrapAndSet(QSGSimpleTextureNode *_node)
  {
    // fromNative is documented to wrap an externally-created VkImage in its
    // current layout; the consumer does not transition it. The size param
    // is documented as the texture's pixel dimensions.
    // https://doc.qt.io/qt-6/qsgvulkantexture.html
    QSGTexture *tex =
        QNativeInterface::QSGVulkanTexture::fromNative(this->image,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            this->window(), this->fromNativeSize);
    _node->setTexture(tex);
    _node->setOwnsTexture(true);
  }

  VkImage image{VK_NULL_HANDLE};
  QSize fromNativeSize;
  bool recreateEveryFrame{false};
};
}  // namespace

/////////////////////////////////////////////////
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRendersLinearPattern))
{
  // Force Qt to use the Vulkan RHI BEFORE the QGuiApplication is constructed
  // (the same setup MinimalScene does when GZ_GUI_RENDER_ENGINE_GUI_API_BACKEND
  // == "vulkan"). Without this Qt picks the platform default (GL on Linux).
  QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);

  int argc = 1;
  static char argv0[] = "qsg_simple_texture_node_vulkan";
  static char *argv[] = {argv0, nullptr};
  QGuiApplication app(argc, argv);

  QQuickView view;
  view.setResizeMode(QQuickView::SizeRootObjectToView);
  view.resize(kPatternW, kPatternH);

  // Show + spin the event loop so QRhi initialises its VkDevice. Without this,
  // rendererInterface()->getResource(...) returns nulls.
  view.show();
  QTest::qWaitForWindowExposed(&view);
  QCoreApplication::processEvents();

  PatternImage pi;
  ASSERT_TRUE(GetQtVulkanHandles(&view, &pi))
      << "could not get Qt's QRhi VkDevice -- did Qt actually pick the Vulkan "
         "RHI? Set QSG_INFO=1 to see Qt's chosen backend.";
  ASSERT_TRUE(CreatePatternImage(&pi))
      << "could not create the LINEAR host-visible pattern VkImage";

  // Parent a PatternItem under the view's content item. We don't load any QML
  // source; the test's QQuickItem is the only content.
  auto *parent = view.contentItem();
  ASSERT_NE(parent, nullptr);
  auto *item = new PatternItem(parent, pi.image, QSize(kPatternW, kPatternH));
  item->setSize(QSizeF(kPatternW, kPatternH));
  item->setPosition(QPointF(0.0, 0.0));

  // Trigger a redraw, then grab. grabWindow() blocks the GUI thread until the
  // window is rendered; returns the QImage of the rendered framebuffer.
  view.update();
  QImage rendered = view.grabWindow();
  AssertRenderedPattern(rendered, "LINEAR/host-visible");
  view.close();
}

/////////////////////////////////////////////////
// Closer-to-production reproducer: same pattern, but the VkImage is OPTIMAL
// tiling with COLOR_ATTACHMENT | SAMPLED | TRANSFER_DST usage (matching the
// O3DE-exported image's flags) and is populated via a staging buffer +
// vkCmdCopyBufferToImage + layout transition to SHADER_READ_ONLY_OPTIMAL on
// Qt's own VkDevice/VkQueue. The only variable removed vs production is the
// cross-device FD import. If this test passes while the production O3DE path
// still renders uniform, the bug requires the cross-device dimension; if it
// fails, we have a single-device reproducer of the QSGSimpleTextureNode bug.
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRendersOptimalPattern))
{
  QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);
  int argc = 1;
  static char argv0[] = "qsg_simple_texture_node_vulkan_optimal";
  static char *argv[] = {argv0, nullptr};
  QGuiApplication app(argc, argv);

  QQuickView view;
  view.setResizeMode(QQuickView::SizeRootObjectToView);
  view.resize(kPatternW, kPatternH);
  view.show();
  QTest::qWaitForWindowExposed(&view);
  QCoreApplication::processEvents();

  PatternImage pi;
  ASSERT_TRUE(GetQtVulkanHandles(&view, &pi));

  // Acquire Qt's main queue. CommandQueueResource is a VkQueue* in Qt 6's
  // QSGRendererInterface. The queue family index isn't exposed as a resource
  // in the Qt version targeted here, so discover it: the first queue family
  // with VK_QUEUE_GRAPHICS_BIT is the one Qt asks QRhi for, and is the one
  // CommandQueueResource was vkGetDeviceQueue()'d from. (Verified in QSG_INFO
  // logs of this RHI: queue family 0 flags=0xf -- graphics+compute+transfer.)
  QSGRendererInterface *rif = view.rendererInterface();
  VkQueue queue = *static_cast<VkQueue *>(rif->getResource(&view,
      QSGRendererInterface::CommandQueueResource));
  ASSERT_NE(queue, VK_NULL_HANDLE)
      << "could not get Qt's CommandQueueResource";
  uint32_t qfCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(pi.physicalDevice, &qfCount,
      nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(pi.physicalDevice, &qfCount,
      qfs.data());
  uint32_t queueFamily = UINT32_MAX;
  for (uint32_t i = 0u; i < qfCount; ++i)
  {
    if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      queueFamily = i;
      break;
    }
  }
  ASSERT_NE(queueFamily, UINT32_MAX)
      << "no graphics queue family on Qt's physical device";

  ASSERT_TRUE(CreateOptimalPatternImage(queue, queueFamily, &pi))
      << "could not create + populate the OPTIMAL VkImage";

  auto *parent = view.contentItem();
  ASSERT_NE(parent, nullptr);
  auto *item = new PatternItem(parent, pi.image, QSize(kPatternW, kPatternH));
  item->setSize(QSizeF(kPatternW, kPatternH));
  item->setPosition(QPointF(0.0, 0.0));

  view.update();
  QImage rendered = view.grabWindow();
  AssertRenderedPattern(rendered, "OPTIMAL/staging-uploaded");
  view.close();
}

/////////////////////////////////////////////////
// Cross-device FD-imported VkImage. A second, private VkInstance/VkDevice
// stands in for Atom, creates an exportable OPTIMAL VkImage there,
// staging-uploads the pattern, exports the memory FD; Qt's VkDevice
// imports the FD with VkMemoryDedicatedAllocateInfo and the test wraps it
// via fromNative + QSGSimpleTextureNode. This is the configuration the
// production O3DE/Atom path uses; the only variable removed vs production
// is "Atom" (replaced by a staging upload) and MinimalScene's threading
// model (replaced by a direct updatePaintNode in a single QQuickItem).
//
// Result: PASSES today. The cross-device round-trip on its own is fine.
// Combined with Tests 1 and 2's PASSes, this isolates the production bug
// to either something Atom does differently from a clean staging upload,
// or MinimalScene's threading/lifetime model -- not the Qt+Vulkan
// primitives. See the file's header comment for the next investigation
// targets.
//
// Skips cleanly if Qt's device was not created with the
// VK_KHR_external_memory_fd extension (the patched gz-gui's Application.cc
// enables it via QT_VULKAN_DEVICE_EXTENSIONS), or if the producer device
// cannot be brought up.
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRendersImportedFdPattern))
{
  QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);
  int argc = 1;
  static char argv0[] = "qsg_simple_texture_node_vulkan_imported_fd";
  static char *argv[] = {argv0, nullptr};
  QGuiApplication app(argc, argv);

  QQuickView view;
  view.setResizeMode(QQuickView::SizeRootObjectToView);
  view.resize(kPatternW, kPatternH);
  view.show();
  QTest::qWaitForWindowExposed(&view);
  QCoreApplication::processEvents();

  PatternImage qt;
  ASSERT_TRUE(GetQtVulkanHandles(&view, &qt));
  QSGRendererInterface *rif = view.rendererInterface();
  VkQueue qtQueue = *static_cast<VkQueue *>(rif->getResource(&view,
      QSGRendererInterface::CommandQueueResource));
  ASSERT_NE(qtQueue, VK_NULL_HANDLE);
  uint32_t qfCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(qt.physicalDevice, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(qt.physicalDevice, &qfCount,
      qfs.data());
  uint32_t qtQF = UINT32_MAX;
  for (uint32_t i = 0u; i < qfCount; ++i)
  {
    if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      qtQF = i;
      break;
    }
  }
  ASSERT_NE(qtQF, UINT32_MAX);

  ProducerDevice prod;
  if (!SetupProducerDevice(&prod))
  {
    GTEST_SKIP() << "could not bring up the producer VkDevice "
                    "(VK_KHR_external_memory_fd unavailable on this GPU?)";
  }

  // Producer side: create exportable image, staging-upload pattern, export FD.
  ProducerImage prodImg;
  int fd = -1;
  VkDeviceSize allocSize = 0u;
  ASSERT_TRUE(CreateAndExportImage(prod, &prodImg, &fd, &allocSize))
      << "could not create + export the producer's VkImage / FD";
  ASSERT_GE(fd, 0);

  // Consumer side: import the FD onto Qt's device, QFOT-acquire onto Qt's
  // queue. We do NOT plumb the imported VkImage through the PatternImage
  // RAII holder because Qt's queue / scene graph may still be sampling it
  // when the test exits; track it separately and free after vkDeviceWaitIdle.
  VkImage importedImage = VK_NULL_HANDLE;
  VkDeviceMemory importedMem = VK_NULL_HANDLE;
  if (!ImportFdOntoQtDevice(qt, fd, allocSize, &importedImage, &importedMem))
  {
    ::close(fd);
    GTEST_SKIP() << "vkAllocateMemory(VkImportMemoryFdInfoKHR) failed -- did "
                    "Qt enable VK_KHR_external_memory_fd on its VkDevice? "
                    "(set via QT_VULKAN_DEVICE_EXTENSIONS in the gz-gui app.)";
  }
  // The FD is now owned by Qt's vkAllocateMemory; do not close it.

  ASSERT_TRUE(AcquireOntoQtQueue(qtQueue, qtQF, qt, importedImage))
      << "could not QFOT-acquire the imported image onto Qt's queue";

  // Hand the imported VkImage to the same PatternItem the other two tests
  // use; the path under test (fromNative + QSGSimpleTextureNode) is now
  // exercised on a cross-device-imported image.
  auto *parent = view.contentItem();
  ASSERT_NE(parent, nullptr);
  auto *item = new PatternItem(parent, importedImage,
      QSize(kPatternW, kPatternH));
  item->setSize(QSizeF(kPatternW, kPatternH));
  item->setPosition(QPointF(0.0, 0.0));

  view.update();
  QImage rendered = view.grabWindow();
  AssertRenderedPattern(rendered, "cross-device FD-imported");
  view.close();

  // Tear-down: PatternItem (parented to the view) is auto-destroyed; flush
  // Qt's queue so we can safely free the imported VkImage / VkDeviceMemory.
  vkDeviceWaitIdle(qt.device);
  vkDestroyImage(qt.device, importedImage, nullptr);
  vkFreeMemory(qt.device, importedMem, nullptr);
}

/////////////////////////////////////////////////
// Test 4: replicate MinimalSceneRhiVulkan's "new fromNative wrapper every
// frame" lifecycle, on the same kind of VkImage Test 2 used.
//
// Layered diagnostics in the live demo showed:
//   * handles thread correctly through the pipeline (Atom's exported VkImage
//     == Qt-side imported VkImage == handle Qt wraps via fromNative);
//   * the imported image's content is correct (the transfer-copy and the
//     compute-shader sampler probe both read the producer's shapes byte-
//     identically: 67 unique colours);
//   * yet the screen renders uniform 148 (Atom's clear colour);
//   * fromNative is called repeatedly on the same VkImage handle as the
//     window resizes (size=0x0 then 1024x1024 then 1024x670 -- one new
//     fromNative wrapper per CreateTexture invocation), and even at steady
//     state CreateTexture fires every frame for a new wrapper around the
//     same VkImage. Both Tests 1-3 use a stable wrapper (fromNative is
//     called once in updatePaintNode's first call, then the cached node is
//     returned), so they do not exercise that lifecycle.
//
// This test layers the production "new fromNative wrapper every frame"
// lifecycle on top of Test 2's working single-device OPTIMAL setup.
// Result: PASSES (against grabWindow()), so the rapid-wrapper-replacement
// is harmless to grabWindow's readback path -- but see the file-header
// caveat: the bug is on the swapchain-present path, which grabWindow()
// does not exercise. See Test 5 (FromNativeSwapchainPresentsPattern)
// which asserts on the actual swapchain via VK_LAYER_GZ_swapchain_dump
// and reproduces the production bug in isolation (uniform white fill).
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRecreatedEveryFrame))
{
  QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);
  int argc = 1;
  static char argv0[] = "qsg_simple_texture_node_vulkan_size_mismatch";
  static char *argv[] = {argv0, nullptr};
  QGuiApplication app(argc, argv);

  QQuickView view;
  view.setResizeMode(QQuickView::SizeRootObjectToView);
  view.resize(kPatternW, kPatternH);
  view.show();
  QTest::qWaitForWindowExposed(&view);
  QCoreApplication::processEvents();

  PatternImage pi;
  ASSERT_TRUE(GetQtVulkanHandles(&view, &pi));
  QSGRendererInterface *rif = view.rendererInterface();
  VkQueue queue = *static_cast<VkQueue *>(rif->getResource(&view,
      QSGRendererInterface::CommandQueueResource));
  ASSERT_NE(queue, VK_NULL_HANDLE);
  uint32_t qfCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(pi.physicalDevice, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(pi.physicalDevice, &qfCount,
      qfs.data());
  uint32_t queueFamily = UINT32_MAX;
  for (uint32_t i = 0u; i < qfCount; ++i)
  {
    if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      queueFamily = i;
      break;
    }
  }
  ASSERT_NE(queueFamily, UINT32_MAX);
  ASSERT_TRUE(CreateOptimalPatternImage(queue, queueFamily, &pi));

  // Same setup as Test 2 (single-device OPTIMAL + staging upload), with the
  // recreate-every-frame texture lifecycle layered on top.
  auto *parent = view.contentItem();
  ASSERT_NE(parent, nullptr);
  auto *item = new PatternItem(parent, pi.image,
      QSize(kPatternW, kPatternH), /* recreateEveryFrame */ true);
  item->setSize(QSizeF(kPatternW, kPatternH));
  item->setPosition(QPointF(0.0, 0.0));

  // Force a handful of frames so updatePaintNode runs the recreate path
  // several times (the production lifecycle fires CreateTexture every frame).
  for (int i = 0; i < 8; ++i)
  {
    view.update();
    QCoreApplication::processEvents();
  }
  QImage rendered = view.grabWindow();
  AssertRenderedPattern(rendered, "OPTIMAL/fromNative-recreated-every-frame");
  view.close();
}

/////////////////////////////////////////////////
// Test 5: assert against what the WSI swapchain actually presents.
//
// Tests 1-4 all assert against QQuickWindow::grabWindow(), which a layered
// diagnostic in the live demo proved DIVERGES from the swapchain present
// path (grabWindow's QImage shows the expected pattern while an on-screen
// capture of the swapchain shows uniform Atom-clear grey for the *same*
// VkImage). To exercise the failing path, this test loads a tiny in-tree
// Vulkan layer (VK_LAYER_GZ_swapchain_dump, built next to this test) that
// hooks vkQueuePresentKHR, copies the swapchain image to a host buffer just
// before present, and writes a PPM. The test then asserts on the PPM.
//
// **Current result on NVIDIA proprietary driver 580 + Qt 6: FAILS** with a
// uniform-fill swapchain (every pixel = 0xffffffff / white, with the test's
// default QQuickWindow clear; in production the same failure shows up as
// uniform 148 = Atom's clear colour). This is a **clean isolated repro of
// the production bug**: single-device, single-process, no Atom, no cross-
// device imports, no producer/consumer threading. The only ingredients are
// QSGVulkanTexture::fromNative + QSGSimpleTextureNode wrapping a VkImage on
// Qt's own VkDevice (the same texture data Test 2 proves Qt reads back
// correctly via grabWindow). So the bug lives in Qt's QSG/QRhi draw of the
// fromNative texture into the swapchain, NOT in any of the suspects we'd
// previously chased through Atom/interop.
//
// The test is left ENABLED so the failure is visible. When the underlying
// Qt bug is fixed, this test flips green automatically; do NOT delete or
// disable it. CI on a headless QPA skips cleanly (see skip predicate below).
//
// Skips when:
//   * GZ_SWAPCHAIN_DUMP_LAYER_DIR not defined at build time (the layer was
//     not built -- Vulkan was missing);
//   * QT_QPA_PLATFORM is "offscreen" or no $DISPLAY / $WAYLAND_DISPLAY is set
//     (WSI swapchain requires a real window-system display);
//   * the layer dump file does not appear within a generous deadline.

namespace
{
// Minimal PPM P6 reader. Returns RGB888 QImage. Used by Test 5 to slurp the
// layer's swapchain dump back in and run the same 4-quadrant assertion as
// the grabWindow tests.
QImage LoadPpmAsQImage(const std::string &_path)
{
  std::FILE *f = std::fopen(_path.c_str(), "rb");
  if (f == nullptr)
    return {};
  char magic[3] = {0, 0, 0};
  if (std::fread(magic, 1u, 2u, f) != 2u || magic[0] != 'P' || magic[1] != '6')
  {
    std::fclose(f);
    return {};
  }
  int w = 0, h = 0, maxv = 0;
  if (std::fscanf(f, " %d %d %d", &w, &h, &maxv) != 3 || maxv != 255 ||
      w <= 0 || h <= 0)
  {
    std::fclose(f);
    return {};
  }
  // One whitespace separates the header from the binary blob.
  std::fgetc(f);
  QImage img(w, h, QImage::Format_RGB888);
  for (int y = 0; y < h; ++y)
  {
    if (std::fread(img.scanLine(y), 1u, static_cast<std::size_t>(w) * 3u, f) !=
        static_cast<std::size_t>(w) * 3u)
    {
      std::fclose(f);
      return {};
    }
  }
  std::fclose(f);
  return img;
}
}  // namespace

TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeSwapchainPresentsPattern))
{
#ifndef GZ_SWAPCHAIN_DUMP_LAYER_DIR
  GTEST_SKIP() << "VK_LAYER_GZ_swapchain_dump was not built -- Vulkan was "
                  "not found at CMake configure time. The swapchain-present "
                  "assertion requires the layer.";
#else
  // WSI swapchain creation requires a real window-system display. Skip on
  // offscreen / headless configurations.
  const char *qtPlatform = std::getenv("QT_QPA_PLATFORM");
  const bool isOffscreen = qtPlatform != nullptr
      && std::strcmp(qtPlatform, "offscreen") == 0;
  const bool hasDisplay = std::getenv("DISPLAY") != nullptr ||
      std::getenv("WAYLAND_DISPLAY") != nullptr;
  if (isOffscreen || !hasDisplay)
  {
    GTEST_SKIP() << "no real display ($DISPLAY/$WAYLAND_DISPLAY unset or "
                    "QT_QPA_PLATFORM=offscreen); WSI swapchain unavailable, "
                    "so vkQueuePresentKHR will not fire and the layer "
                    "produces no dump.";
  }

  // The layer is .so + .json living together in this build dir.
  const std::string layerDir = GZ_SWAPCHAIN_DUMP_LAYER_DIR;
  // The dump path is per-test to avoid stomping on a parallel run.
  const std::string dumpPrefix =
      "/tmp/qsg_simple_texture_node_vulkan_test5_swapchain";
  // Wipe any previous-run artifacts so we can later prove this run produced
  // the file we read back.
  std::remove((dumpPrefix + ".frame_0000.ppm").c_str());
  std::remove((dumpPrefix + ".frame_0000.txt").c_str());

  // The Vulkan loader honours these env vars at vkCreateInstance time, so
  // they MUST be set before QGuiApplication's QVulkanInstance is created
  // (which happens during QQuickWindow::setGraphicsApi(Vulkan) at the latest
  // QML scene-graph init).
  ::setenv("VK_LAYER_PATH", layerDir.c_str(), 1);
  ::setenv("VK_INSTANCE_LAYERS", "VK_LAYER_GZ_swapchain_dump", 1);
  ::setenv("GZ_SWAPCHAIN_DUMP_PATH", dumpPrefix.c_str(), 1);
  ::setenv("GZ_SWAPCHAIN_DUMP_MAX_FRAMES", "1", 1);

  QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);
  int argc = 1;
  static char argv0[] = "qsg_simple_texture_node_vulkan_swapchain_present";
  static char *argv[] = {argv0, nullptr};
  QGuiApplication app(argc, argv);

  QQuickView view;
  view.setResizeMode(QQuickView::SizeRootObjectToView);
  view.resize(kPatternW, kPatternH);
  view.show();
  QTest::qWaitForWindowExposed(&view);
  QCoreApplication::processEvents();

  PatternImage pi;
  ASSERT_TRUE(GetQtVulkanHandles(&view, &pi));
  QSGRendererInterface *rif = view.rendererInterface();
  VkQueue queue = *static_cast<VkQueue *>(rif->getResource(&view,
      QSGRendererInterface::CommandQueueResource));
  ASSERT_NE(queue, VK_NULL_HANDLE);
  uint32_t qfCount = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(pi.physicalDevice, &qfCount,
      nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(pi.physicalDevice, &qfCount,
      qfs.data());
  uint32_t queueFamily = UINT32_MAX;
  for (uint32_t i = 0u; i < qfCount; ++i)
  {
    if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      queueFamily = i;
      break;
    }
  }
  ASSERT_NE(queueFamily, UINT32_MAX);
  ASSERT_TRUE(CreateOptimalPatternImage(queue, queueFamily, &pi));

  // Same setup as Test 2 (single-device OPTIMAL + staging upload).
  auto *parent = view.contentItem();
  ASSERT_NE(parent, nullptr);
  auto *item = new PatternItem(parent, pi.image,
      QSize(kPatternW, kPatternH));
  item->setSize(QSizeF(kPatternW, kPatternH));
  item->setPosition(QPointF(0.0, 0.0));

  // Pump frames until the layer has written the dump file. Bounded by a
  // generous wall-clock deadline (5s); a working layer typically writes the
  // first frame in <100ms after exposure.
  const std::string dumpPath = dumpPrefix + ".frame_0000.ppm";
  const auto deadline = std::chrono::steady_clock::now()
      + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline)
  {
    view.update();
    QCoreApplication::processEvents();
    QTest::qWait(50);  // give Qt's render thread a chance to present
    std::FILE *probe = std::fopen(dumpPath.c_str(), "rb");
    if (probe != nullptr)
    {
      std::fclose(probe);
      break;
    }
  }

  std::FILE *probe = std::fopen(dumpPath.c_str(), "rb");
  if (probe == nullptr)
  {
    view.close();
    FAIL() << "VK_LAYER_GZ_swapchain_dump did not produce a PPM at "
           << dumpPath << " within 5 s. Likely causes: the platform's Qt "
              "QPA does not route through the layer-instrumented loader; "
              "Qt's QVulkanInstance was created before VK_LAYER_PATH was "
              "honoured; or vkQueuePresentKHR is never called (no WSI). "
              "Run with GZ_SWAPCHAIN_DUMP_VERBOSE=1 for layer-side logs.";
  }
  std::fclose(probe);

  QImage presented = LoadPpmAsQImage(dumpPath);
  ASSERT_FALSE(presented.isNull())
      << "could not parse swapchain dump PPM at " << dumpPath;
  AssertRenderedPattern(presented, "swapchain-present (layer-dump)");
  view.close();
#endif  // GZ_SWAPCHAIN_DUMP_LAYER_DIR
}
#else  // GZ_GUI_TEST_HAVE_VULKAN
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRendersLinearPattern))
{
  GTEST_SKIP() << "Qt was built without Vulkan support; "
                  "QSGVulkanTexture::fromNative is not available.";
}
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRendersOptimalPattern))
{
  GTEST_SKIP() << "Qt was built without Vulkan support; "
                  "QSGVulkanTexture::fromNative is not available.";
}
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRendersImportedFdPattern))
{
  GTEST_SKIP() << "Qt was built without Vulkan support; "
                  "QSGVulkanTexture::fromNative is not available.";
}
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeRecreatedEveryFrame))
{
  GTEST_SKIP() << "Qt was built without Vulkan support; "
                  "QSGVulkanTexture::fromNative is not available.";
}
TEST(QsgSimpleTextureNodeVulkan,
    GZ_UTILS_TEST_ENABLED_ONLY_ON_LINUX(FromNativeSwapchainPresentsPattern))
{
  GTEST_SKIP() << "Qt was built without Vulkan support; "
                  "QSGVulkanTexture::fromNative is not available.";
}
#endif  // GZ_GUI_TEST_HAVE_VULKAN
