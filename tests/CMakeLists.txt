# Unit tests for GPU-independent engine logic. Catch2 v3 is fetched on demand so
# running these tests does not require a Vulkan SDK or a GPU.
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
    test_bounds.cpp
    test_job_system.cpp
    test_editor_camera.cpp
)

target_link_libraries(VulkanEngineTests PRIVATE
    VulkanEngineCore
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
