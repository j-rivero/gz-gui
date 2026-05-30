# VK_LAYER_GZ_swapchain_dump

A tiny in-tree Vulkan layer that dumps the swapchain image **just before
`vkQueuePresentKHR`** to a PPM file on disk, so callers can assert (or
visually inspect) what the WSI is actually about to present — independent of
Qt-side readback APIs like `QQuickWindow::grabWindow()` which a layered live-
demo diagnostic proved diverge from the swapchain present path on at least
NVIDIA proprietary 580 + Qt 6.

The primary consumer is the
`qsg_simple_texture_node_vulkan.cc` regression test (specifically Test 5,
`FromNativeSwapchainPresentsPattern`), which loads the layer to assert that
the swapchain present contains the same pattern the test uploaded to the
backing `VkImage`. The layer is general-purpose, though, and the same
manifest + `.so` can be loaded into any Vulkan process to capture
presentations.

## Building

The layer is built whenever Qt is found *and* `find_package(Vulkan)` succeeds
at gz-gui configure time. Build artefacts land in the build tree:

```
<build>/test/regression/swapchain_dump_layer/
├── libVkLayer_gz_swapchain_dump.so
└── VkLayer_gz_swapchain_dump.json
```

The manifest's `library_path` is set to `./libVkLayer_gz_swapchain_dump.so`,
so the loader resolves it relative to the directory it scans. No install step
is required.

## Activating in any Vulkan process

```bash
export VK_LAYER_PATH=<build>/test/regression/swapchain_dump_layer
export VK_INSTANCE_LAYERS=VK_LAYER_GZ_swapchain_dump
export GZ_SWAPCHAIN_DUMP_PATH=/tmp/my_dump          # path prefix (required)
export GZ_SWAPCHAIN_DUMP_MAX_FRAMES=1               # default 1
export GZ_SWAPCHAIN_DUMP_VERBOSE=1                  # optional layer-side logs
<your-vulkan-app>
```

Outputs (per dumped frame `N`):

- `${GZ_SWAPCHAIN_DUMP_PATH}.frame_NNNN.ppm` — PPM P6, 8-bit RGB.
- `${GZ_SWAPCHAIN_DUMP_PATH}.frame_NNNN.txt` — one-line metadata
  (`frame`, `swapchain_image_index`, `extent`, `format`, `note`).

The layer reads the env vars at module load and on each present, so changes
to `GZ_SWAPCHAIN_DUMP_MAX_FRAMES` between runs take effect immediately. The
frame counter is global across all swapchains in the process.

## Sanity check

```bash
VK_LAYER_PATH=<build>/test/regression/swapchain_dump_layer \
  vulkaninfo --summary | grep -i swapchain_dump
```

Expected output line:

```
VK_LAYER_GZ_swapchain_dump  ...  1.2.0  version 1
```

## Supported swapchain formats

- `VK_FORMAT_B8G8R8A8_UNORM` / `_SRGB` (most common on X11/xcb)
- `VK_FORMAT_R8G8B8A8_UNORM` / `_SRGB`

Other formats are skipped with a warning when `GZ_SWAPCHAIN_DUMP_VERBOSE=1`.
Add the format to `FormatInfo()` in `swapchain_dump_layer.cc` to handle more.

## How the dump is taken

On entry to `vkQueuePresentKHR`, for each presented swapchain image:

1. Lazy-create a `VkCommandPool` + per-frame `VkCommandBuffer`.
2. Allocate a host-visible `VkBuffer` sized `W * H * bpp`.
3. Record: `PRESENT_SRC_KHR → TRANSFER_SRC_OPTIMAL` barrier,
   `vkCmdCopyImageToBuffer`, `TRANSFER_SRC_OPTIMAL → PRESENT_SRC_KHR` barrier.
4. Submit with the present's wait semaphores; signal a fence; CPU-wait the
   fence so the buffer is host-coherent.
5. Map the buffer, BGRA→RGB swizzle if needed, write PPM and sidecar.
6. Forward to the next-layer `vkQueuePresentKHR` with the wait semaphores
   cleared (they were consumed by the copy submit).

## Limitations

- Synchronous copy on the present queue: one full `vkQueueWaitIdle`-style
  CPU wait per dumped frame. Not for production; intended for regression
  tests and offline diagnostics.
- Queue family is assumed to be 0 when creating the command pool. A robust
  layer would track per-queue families at `vkGetDeviceQueue` time.
- The layer does not write sRGB-to-linear conversion. If the swapchain is
  an sRGB format the dumped bytes are the sRGB-encoded values; the sidecar's
  `note=` field flags this.
- Single-instance, single-process. The layer keeps global state behind a
  mutex; it's safe for multiple devices in the same process but not designed
  for high throughput.

## See also

- `../qsg_simple_texture_node_vulkan.cc` — primary consumer (Test 5).
- `gz-rendering/o3de/docs/zero-copy-interop-findings.md` — the diagnostic
  trail that led to needing this layer.
