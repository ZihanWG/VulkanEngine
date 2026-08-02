# Unit tests for GPU-independent engine logic. Catch2 v3 is fetched on demand.
# The tests run headless and never initialize Vulkan or touch a GPU. Most cases
# link only VulkanEngineCore; test_scene_builder additionally links
# VulkanEngineRuntime because SceneBuilder operates on RenderObject (whose Mesh
# member is an RAII Vulkan buffer), so the binary links -- but never calls into --
# the Vulkan loader / VMA. Linking is symbol-driven, so only the objects
# SceneBuilder actually references are pulled in.
include(FetchContent)

set(VULKAN_ENGINE_CATCH2_TAG "v3.7.1")

find_package(Catch2 3 CONFIG QUIET)
if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG ${VULKAN_ENGINE_CATCH2_TAG}
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(VulkanEngineTests
    test_runtime_settings.cpp
    test_settings_clamp.cpp
    test_bounds.cpp
    test_cascades.cpp
    test_cluster_grid.cpp
    test_material_asset.cpp
    test_material_alpha.cpp
    test_mesh_lod.cpp
    test_skeletal_animation.cpp
    test_gltf_skinned_import.cpp
    test_job_system.cpp
    test_editor_camera.cpp
    test_scene_builder.cpp
    test_pipeline_cache_header.cpp
)

target_compile_definitions(VulkanEngineTests PRIVATE
    "VULKAN_ENGINE_ASSET_DIR=\"${VULKAN_ENGINE_ASSET_DIR}\""
)

# VulkanEngineRuntime PUBLIC-links VulkanEngineCore, so listing Core here too would
# only produce a duplicate-library warning -- the runtime link brings it in.
target_link_libraries(VulkanEngineTests PRIVATE
    VulkanEngineRuntime
    Catch2::Catch2WithMain
)

if(MSVC)
    target_compile_options(VulkanEngineTests PRIVATE /W4 /permissive- /EHsc /utf-8)
else()
    target_compile_options(VulkanEngineTests PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Register every TEST_CASE with CTest so `ctest` reports them individually.
if(DEFINED catch2_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()
include(Catch)
catch_discover_tests(VulkanEngineTests)
