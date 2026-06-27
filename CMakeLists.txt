cmake_minimum_required(VERSION 3.25)

project(VulkanEngine
    VERSION 0.1.0
    DESCRIPTION "Modern C++20 Vulkan 1.3 engine skeleton"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VULKAN_ENGINE_FETCH_DEPS "Fetch missing third-party dependencies with FetchContent" ON)
option(VULKAN_ENGINE_BUILD_MACOS_BUNDLE "Build VulkanEngine as a macOS .app bundle" OFF)
option(VULKAN_ENGINE_BUILD_TESTS "Build unit tests for GPU-independent engine logic" ON)
option(VULKAN_ENGINE_ENABLE_SANITIZERS "Build with AddressSanitizer + UndefinedBehaviorSanitizer (GCC/Clang)" OFF)

if(VULKAN_ENGINE_BUILD_MACOS_BUNDLE AND NOT APPLE)
    message(WARNING "VULKAN_ENGINE_BUILD_MACOS_BUNDLE is only supported on Apple platforms; building a normal executable.")
endif()

# Apply sanitizers directory-wide (before any targets are defined) so every
# translation unit -- including the fetched dependencies the tests link against
# -- is instrumented consistently. ASan/UBSan are a GCC/Clang feature; MSVC has
# its own /fsanitize=address but no UBSan, so we skip it there.
if(VULKAN_ENGINE_ENABLE_SANITIZERS)
    if(MSVC)
        message(WARNING "VULKAN_ENGINE_ENABLE_SANITIZERS supports GCC/Clang only; ignoring on MSVC.")
    else()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

include(FetchContent)

# FetchContent fallbacks are pinned to stable tags so clean checkouts and CI
# resolve reproducible dependency sources instead of moving default branches.
set(VULKAN_ENGINE_SDL3_TAG "release-3.4.8")
set(VULKAN_ENGINE_GLM_TAG "1.0.3")
set(VULKAN_ENGINE_VOLK_TAG "vulkan-sdk-1.4.328.1")
set(VULKAN_ENGINE_VMA_TAG "v3.3.0")

find_package(Vulkan REQUIRED)

find_package(SDL3 CONFIG QUIET)
if(NOT TARGET SDL3::SDL3 AND NOT TARGET SDL3::SDL3-shared AND NOT TARGET SDL3::SDL3-static)
    if(NOT VULKAN_ENGINE_FETCH_DEPS)
        message(FATAL_ERROR "SDL3 was not found. Install SDL3 or configure with VULKAN_ENGINE_FETCH_DEPS=ON.")
    endif()

    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG ${VULKAN_ENGINE_SDL3_TAG}
    )
    FetchContent_MakeAvailable(SDL3)
endif()

find_package(glm CONFIG QUIET)
if(NOT TARGET glm::glm)
    if(NOT VULKAN_ENGINE_FETCH_DEPS)
        message(FATAL_ERROR "GLM was not found. Install GLM or configure with VULKAN_ENGINE_FETCH_DEPS=ON.")
    endif()

    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG ${VULKAN_ENGINE_GLM_TAG}
    )
    FetchContent_MakeAvailable(glm)
endif()

# Resolve volk. The LunarG Vulkan SDK (used by CI) ships a header-only volk config
# that defines volk::volk_headers. That target both clashes with a FetchContent
# volk (duplicate volk::volk_headers alias) and lacks the compiled volk.c the
# engine links against. So when dependency fetching is enabled, always fetch volk
# -- which provides the compiled volk::volk target -- instead of probing for a
# system volk that may only ship the headers interface.
if(VULKAN_ENGINE_FETCH_DEPS)
    set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG ${VULKAN_ENGINE_VOLK_TAG}
    )
    FetchContent_MakeAvailable(volk)
else()
    find_package(volk CONFIG QUIET)
    if(NOT TARGET volk::volk AND NOT TARGET volk)
        message(FATAL_ERROR "Volk (compiled volk::volk target) was not found. Configure with VULKAN_ENGINE_FETCH_DEPS=ON.")
    endif()
endif()

find_package(VulkanMemoryAllocator CONFIG QUIET)
if(NOT TARGET GPUOpen::VulkanMemoryAllocator AND NOT TARGET VulkanMemoryAllocator)
    if(NOT VULKAN_ENGINE_FETCH_DEPS)
        message(FATAL_ERROR "Vulkan Memory Allocator was not found. Install VMA or configure with VULKAN_ENGINE_FETCH_DEPS=ON.")
    endif()

    FetchContent_Declare(vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG ${VULKAN_ENGINE_VMA_TAG}
    )
    FetchContent_MakeAvailable(vma)
endif()

set(SDL3_TARGET "")
if(TARGET SDL3::SDL3)
    set(SDL3_TARGET SDL3::SDL3)
elseif(TARGET SDL3::SDL3-shared)
    set(SDL3_TARGET SDL3::SDL3-shared)
elseif(TARGET SDL3::SDL3-static)
    set(SDL3_TARGET SDL3::SDL3-static)
endif()

set(VOLK_TARGET "")
if(TARGET volk::volk)
    set(VOLK_TARGET volk::volk)
elseif(TARGET volk)
    set(VOLK_TARGET volk)
endif()

set(VMA_TARGET "")
if(TARGET GPUOpen::VulkanMemoryAllocator)
    set(VMA_TARGET GPUOpen::VulkanMemoryAllocator)
elseif(TARGET VulkanMemoryAllocator)
    set(VMA_TARGET VulkanMemoryAllocator)
elseif(vma_SOURCE_DIR)
    add_library(vma_headers INTERFACE)
    target_include_directories(vma_headers INTERFACE "${vma_SOURCE_DIR}/include")
    set(VMA_TARGET vma_headers)
endif()

if(NOT SDL3_TARGET)
    message(FATAL_ERROR "SDL3 target was not found after dependency setup.")
endif()
if(NOT VOLK_TARGET)
    message(FATAL_ERROR "Volk target was not found after dependency setup.")
endif()
if(NOT VMA_TARGET)
    message(FATAL_ERROR "VMA target was not found after dependency setup.")
endif()

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_sdl3.cpp
    external/imgui/backends/imgui_impl_vulkan.cpp
    external/imguizmo/ImGuizmo.cpp
)

target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
    external/imguizmo
)

target_compile_definitions(imgui PUBLIC
    VK_NO_PROTOTYPES
    IMGUI_IMPL_VULKAN_USE_VOLK
)

target_link_libraries(imgui PUBLIC
    ${SDL3_TARGET}
    ${VOLK_TARGET}
)

# GPU-independent engine logic (serialization, logging, math helpers). Kept as a
# separate static library so the unit tests can link it without pulling in Vulkan.
find_package(Threads REQUIRED)

add_library(VulkanEngineCore STATIC
    src/core/JobSystem.cpp
    src/core/Logger.cpp
    src/renderer/CascadeMath.cpp
    src/renderer/EditorCamera.cpp
    src/renderer/RuntimeSettings.cpp
)

target_include_directories(VulkanEngineCore PUBLIC src)
target_include_directories(VulkanEngineCore SYSTEM PUBLIC external)

target_link_libraries(VulkanEngineCore PUBLIC glm::glm Threads::Threads)

target_compile_definitions(VulkanEngineCore PUBLIC
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

if(MSVC)
    target_compile_options(VulkanEngineCore PRIVATE /W4 /permissive- /EHsc /utf-8)
else()
    target_compile_options(VulkanEngineCore PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(VulkanEngine
    src/main.cpp
    src/assets/AssetManager.cpp
    src/core/Application.cpp
    src/core/PngWriter.cpp
    src/core/Window.cpp
    src/renderer/BindlessTextureHeap.cpp
    src/renderer/ClusteredLighting.cpp
    src/renderer/GpuProfiler.cpp
    src/renderer/Mesh.cpp
    src/renderer/PostProcessStack.cpp
    src/renderer/RenderGraph.cpp
    src/renderer/Renderer.cpp
    src/renderer/RendererDebugUi.cpp
    src/renderer/ScreenshotCapture.cpp
    src/ui/ImGuiLayer.cpp
    src/rhi/VulkanBuffer.cpp
    src/rhi/VulkanBrdfLut.cpp
    src/rhi/VulkanCommandContext.cpp
    src/rhi/VulkanComputePipeline.cpp
    src/rhi/VulkanContext.cpp
    src/rhi/VulkanDebugUtils.cpp
    src/rhi/VulkanDescriptor.cpp
    src/rhi/VulkanDevice.cpp
    src/rhi/VulkanEnvironmentMap.cpp
    src/rhi/VulkanImage.cpp
    src/rhi/VulkanMemory.cpp
    src/rhi/VulkanPipeline.cpp
    src/rhi/VulkanShadowMap.cpp
    src/rhi/VulkanSwapchain.cpp
    src/rhi/VulkanSync.cpp
    src/rhi/VulkanTimestampQuery.cpp
    src/rhi/VulkanTexture.cpp
)

if(APPLE AND VULKAN_ENGINE_BUILD_MACOS_BUNDLE)
    set_target_properties(VulkanEngine PROPERTIES
        MACOSX_BUNDLE TRUE
        OUTPUT_NAME VulkanEngine
        MACOSX_BUNDLE_BUNDLE_NAME VulkanEngine
        MACOSX_BUNDLE_GUI_IDENTIFIER com.zihanw.VulkanEngine
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION}
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MacOSXBundleInfo.plist.in"
    )
endif()

target_include_directories(VulkanEngine PRIVATE src)
target_include_directories(VulkanEngine SYSTEM PRIVATE external)

target_link_libraries(VulkanEngine PRIVATE
    VulkanEngineCore
    ${SDL3_TARGET}
    ${VOLK_TARGET}
    ${VMA_TARGET}
    glm::glm
    imgui
)

if(TARGET Vulkan::Headers)
    target_link_libraries(VulkanEngine PRIVATE Vulkan::Headers)
else()
    target_link_libraries(VulkanEngine PRIVATE Vulkan::Vulkan)
endif()

target_compile_definitions(VulkanEngine PRIVATE
    VK_NO_PROTOTYPES
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    $<$<CONFIG:Debug>:VULKAN_ENGINE_ENABLE_VALIDATION=1>
    $<$<NOT:$<CONFIG:Debug>>:VULKAN_ENGINE_ENABLE_VALIDATION=0>
)

if(MSVC)
    target_compile_options(VulkanEngine PRIVATE /W4 /permissive- /EHsc /utf-8)
else()
    target_compile_options(VulkanEngine PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(ASSET_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(CONFIG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/config")
cmake_path(CONVERT "${SHADER_BINARY_DIR}" TO_CMAKE_PATH_LIST VULKAN_ENGINE_SHADER_DIR NORMALIZE)
cmake_path(CONVERT "${ASSET_SOURCE_DIR}" TO_CMAKE_PATH_LIST VULKAN_ENGINE_ASSET_DIR NORMALIZE)
cmake_path(CONVERT "${CONFIG_SOURCE_DIR}" TO_CMAKE_PATH_LIST VULKAN_ENGINE_CONFIG_DIR NORMALIZE)
set(SHADER_SOURCES
    "${SHADER_SOURCE_DIR}/simple.vert"
    "${SHADER_SOURCE_DIR}/simple.frag"
    "${SHADER_SOURCE_DIR}/simple_bindless.frag"
    "${SHADER_SOURCE_DIR}/skybox.vert"
    "${SHADER_SOURCE_DIR}/skybox.frag"
    "${SHADER_SOURCE_DIR}/shadow.vert"
    "${SHADER_SOURCE_DIR}/cull.comp"
    "${SHADER_SOURCE_DIR}/cluster_build.comp"
    "${SHADER_SOURCE_DIR}/light_cull.comp"
    "${SHADER_SOURCE_DIR}/depth_pyramid.comp"
    "${SHADER_SOURCE_DIR}/luminance.comp"
    "${SHADER_SOURCE_DIR}/luminance_histogram.comp"
    "${SHADER_SOURCE_DIR}/fullscreen.vert"
    "${SHADER_SOURCE_DIR}/bloom_extract.frag"
    "${SHADER_SOURCE_DIR}/bloom_blur.frag"
    "${SHADER_SOURCE_DIR}/bloom_downsample.frag"
    "${SHADER_SOURCE_DIR}/bloom_upsample.frag"
    "${SHADER_SOURCE_DIR}/taa_resolve.frag"
    "${SHADER_SOURCE_DIR}/exposure_reduce.comp"
    "${SHADER_SOURCE_DIR}/composite.frag"
)

target_compile_definitions(VulkanEngine PRIVATE
    "VULKAN_ENGINE_SHADER_DIR=\"${VULKAN_ENGINE_SHADER_DIR}\""
    "VULKAN_ENGINE_ASSET_DIR=\"${VULKAN_ENGINE_ASSET_DIR}\""
    "VULKAN_ENGINE_CONFIG_DIR=\"${VULKAN_ENGINE_CONFIG_DIR}\""
)

set(GLSLC_EXECUTABLE "")
if(Vulkan_GLSLC_EXECUTABLE AND EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
    set(GLSLC_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}")
else()
    find_program(GLSLC_EXECUTABLE glslc)
endif()
if(GLSLC_EXECUTABLE)
    file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")
    set(SHADER_OUTPUTS "")

    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME "${SHADER}" NAME)
        set(SHADER_OUTPUT "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT "${SHADER_OUTPUT}"
            COMMAND "${GLSLC_EXECUTABLE}" "${SHADER}" -o "${SHADER_OUTPUT}" --target-env=vulkan1.3
            DEPENDS "${SHADER}"
            COMMENT "Compiling shader ${SHADER_NAME}"
            VERBATIM
        )
        list(APPEND SHADER_OUTPUTS "${SHADER_OUTPUT}")
    endforeach()

    add_custom_target(VulkanEngineShaders DEPENDS ${SHADER_OUTPUTS})
    add_dependencies(VulkanEngine VulkanEngineShaders)
else()
    message(FATAL_ERROR "glslc was not found. Milestone 2 needs compiled SPIR-V shaders; install the Vulkan SDK.")
endif()

if(APPLE AND VULKAN_ENGINE_BUILD_MACOS_BUNDLE)
    add_custom_command(TARGET VulkanEngine POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "$<TARGET_BUNDLE_DIR:VulkanEngine>/Contents/Resources"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${ASSET_SOURCE_DIR}"
            "$<TARGET_BUNDLE_DIR:VulkanEngine>/Contents/Resources/assets"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${SHADER_BINARY_DIR}"
            "$<TARGET_BUNDLE_DIR:VulkanEngine>/Contents/Resources/shaders"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CONFIG_SOURCE_DIR}"
            "$<TARGET_BUNDLE_DIR:VulkanEngine>/Contents/Resources/config"
        COMMENT "Copying VulkanEngine resources into the app bundle"
        VERBATIM
    )
endif()

if(WIN32)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT VulkanEngine)

    if(TARGET SDL3::SDL3-shared)
        add_custom_command(TARGET VulkanEngine POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:SDL3::SDL3-shared>"
                "$<TARGET_FILE_DIR:VulkanEngine>"
        )
    endif()
endif()

if(VULKAN_ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
