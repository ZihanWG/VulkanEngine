# Unit tests for GPU-independent engine logic. Catch2 v3 is fetched on demand.
# The tests run headless and never initialize Vulkan or touch a GPU. Most cases
# link only VulkanEngineCore; test_scene_builder additionally links
# VulkanEngineRuntime because SceneBuilder operates on RenderObject (whose Mesh
# member is an RAII Vulkan buffer), so the binary links -- but never calls into --
# the Vulkan loader / VMA. Linking is symbol-driven, so only the objects
# SceneBuilder actually references are pulled in.
#
# That property is load-bearing on Windows, not just tidy. Vulkan is reached
# through volk and is loaded on demand, but SDL3 is a real DLL import: the moment
# a test references anything that pulls Application.cpp or Window.cpp, this
# executable gains a startup dependency on SDL3.dll, which does not sit beside it
# in the build tree. Catch2's catch_discover_tests runs this binary at build time
# to enumerate cases, so the failure appears as a build error --
# "Result: Exit code 0xc0000135" (STATUS_DLL_NOT_FOUND) -- and never as a test
# failure. macOS does not reproduce it, because dylib rpaths resolve.
#
# If you need to cover logic that currently lives beside Vulkan or SDL code,
# lift that logic into VulkanEngineCore rather than copying DLLs next to this
# target: a copy would make this whole class of layering mistake silent again.
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
    test_punctual_shadow_atlas.cpp
    test_irradiance_probes.cpp
    test_volumetric_fog.cpp
    test_ktx2.cpp
    test_texture_cook.cpp
    test_material_asset.cpp
    test_material_alpha.cpp
    test_mesh_lod.cpp
    test_skeletal_animation.cpp
    test_gltf_skinned_import.cpp
    test_job_system.cpp
    test_editor_camera.cpp
    test_scene_builder.cpp
    test_pipeline_cache_header.cpp
    test_render_graph.cpp
    test_render_scale.cpp
    test_dynamic_resolution.cpp
    test_occlusion_yield.cpp
    test_draw_batching.cpp
    test_asset_load_stats.cpp
    test_frame_clock.cpp
    test_command_line.cpp
    test_image_compare.cpp
    test_transient_memory_plan.cpp
    test_camera_framing.cpp
)

target_compile_definitions(VulkanEngineTests PRIVATE
    "VULKAN_ENGINE_ASSET_DIR=\"${VULKAN_ENGINE_ASSET_DIR}\""
    "VULKAN_ENGINE_CONFIG_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../config\""
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
