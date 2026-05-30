gz_get_sources(tests)

# qsg_simple_texture_node_vulkan reproduces the QSGSimpleTextureNode +
# QSGVulkanTexture::fromNative grey-render bug (see the test's header comment
# and gz-rendering/o3de/docs/zero-copy-interop-findings.md). It calls Vulkan
# APIs directly, so it needs the Vulkan loader library and the Qt Quick module
# for QQuickView. The test gracefully GTEST_SKIPs when Qt was built without
# Vulkan support, so the dependency is optional.
find_package(Vulkan QUIET)
set(_qsg_vk_libs "")
if (Vulkan_FOUND)
  list(APPEND _qsg_vk_libs Vulkan::Vulkan)
endif()

# Optional Vulkan-layer subdirectory used by the swapchain-present test
# variant. Builds libVkLayer_gz_swapchain_dump.so + its manifest in the
# build tree; the test loads it via VK_LAYER_PATH at runtime.
if (Vulkan_FOUND)
  add_subdirectory(swapchain_dump_layer)
endif()

gz_build_tests(TYPE REGRESSION
               SOURCES ${tests}
               LIB_DEPS
                 Qt${QT_MAJOR_VERSION}::Quick
                 Qt${QT_MAJOR_VERSION}::Test
                 ${_qsg_vk_libs}
               ENVIRONMENT GZ_GUI_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})

# Hand the test the absolute build path of the layer manifest, plus declare a
# build-order dep so the layer .so + .json exist by the time the test runs.
# Target naming convention: gz_build_tests prefixes with "${TEST_TYPE}_".
set(_qsg_vk_test_target REGRESSION_qsg_simple_texture_node_vulkan)
if (Vulkan_FOUND AND TARGET ${_qsg_vk_test_target})
  target_compile_definitions(${_qsg_vk_test_target} PRIVATE
    GZ_SWAPCHAIN_DUMP_LAYER_DIR="${GZ_SWAPCHAIN_DUMP_LAYER_DIR}")
  add_dependencies(${_qsg_vk_test_target} VkLayer_gz_swapchain_dump)
endif()
