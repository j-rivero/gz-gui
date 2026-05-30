# VK_LAYER_GZ_swapchain_dump
#
# A tiny Vulkan layer used by the qsg_simple_texture_node_vulkan regression
# test to dump the swapchain image just before vkQueuePresentKHR, so the test
# can assert against what is actually presented (rather than against
# QQuickWindow::grabWindow(), which diverges from the present path).
#
# Built only when Qt was found with Vulkan support; OFF when Vulkan is absent
# (the test gracefully GTEST_SKIPs in that case).
#
# Output:
#   * libVkLayer_gz_swapchain_dump.so  (in the test/regression/swapchain_dump_layer/ build dir)
#   * VkLayer_gz_swapchain_dump.json  (same dir)
# The test sets VK_LAYER_PATH to that directory at runtime.

if (NOT Vulkan_FOUND)
  return()
endif()

add_library(VkLayer_gz_swapchain_dump SHARED swapchain_dump_layer.cc)
target_link_libraries(VkLayer_gz_swapchain_dump PRIVATE Vulkan::Vulkan)
target_compile_features(VkLayer_gz_swapchain_dump PRIVATE cxx_std_17)
set_target_properties(VkLayer_gz_swapchain_dump PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  # The Vulkan loader resolves the manifest's "library_path" relative to the
  # manifest directory. Force the .so into the same dir as the .json so the
  # relative library_path "./libVkLayer_gz_swapchain_dump.so" resolves.
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  # Strip any per-config subdir (Debug/Release) that multi-config generators
  # would otherwise tack on.
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_CURRENT_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_CURRENT_BINARY_DIR}")

# Generate the manifest JSON pointing at the just-built shared library.
# The Vulkan loader resolves the library_path relative to the manifest's dir.
set(LAYER_LIBRARY_PATH "./libVkLayer_gz_swapchain_dump.so")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_gz_swapchain_dump.json.in
  ${CMAKE_CURRENT_BINARY_DIR}/VkLayer_gz_swapchain_dump.json
  @ONLY)

# Expose the manifest dir to the parent scope so the test target can hand it
# to the regression test via a compile-time #define.
set(GZ_SWAPCHAIN_DUMP_LAYER_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    CACHE INTERNAL "Build dir of VK_LAYER_GZ_swapchain_dump")
