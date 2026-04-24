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

include(FetchContent)

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
        GIT_TAG main
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
        GIT_TAG master
    )
    FetchContent_MakeAvailable(glm)
endif()

find_package(volk CONFIG QUIET)
if(NOT TARGET volk::volk AND NOT TARGET volk)
    if(NOT VULKAN_ENGINE_FETCH_DEPS)
        message(FATAL_ERROR "Volk was not found. Install Volk or configure with VULKAN_ENGINE_FETCH_DEPS=ON.")
    endif()

    set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(volk)
endif()

find_package(VulkanMemoryAllocator CONFIG QUIET)
if(NOT TARGET GPUOpen::VulkanMemoryAllocator AND NOT TARGET VulkanMemoryAllocator)
    if(NOT VULKAN_ENGINE_FETCH_DEPS)
        message(FATAL_ERROR "Vulkan Memory Allocator was not found. Install VMA or configure with VULKAN_ENGINE_FETCH_DEPS=ON.")
    endif()

    FetchContent_Declare(vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG master
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

add_executable(VulkanEngine
    src/main.cpp
    src/core/Application.cpp
    src/core/Logger.cpp
    src/core/Window.cpp
    src/renderer/Renderer.cpp
    src/rhi/VulkanBuffer.cpp
    src/rhi/VulkanCommandContext.cpp
    src/rhi/VulkanContext.cpp
    src/rhi/VulkanDescriptor.cpp
    src/rhi/VulkanDevice.cpp
    src/rhi/VulkanImage.cpp
    src/rhi/VulkanMemory.cpp
    src/rhi/VulkanPipeline.cpp
    src/rhi/VulkanSwapchain.cpp
    src/rhi/VulkanSync.cpp
)

target_include_directories(VulkanEngine PRIVATE src)

target_link_libraries(VulkanEngine PRIVATE
    ${SDL3_TARGET}
    ${VOLK_TARGET}
    ${VMA_TARGET}
    glm::glm
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
set(SHADER_SOURCES
    "${SHADER_SOURCE_DIR}/simple.vert"
    "${SHADER_SOURCE_DIR}/simple.frag"
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
    message(WARNING "glslc was not found. Milestone 1 does not need shaders; install the Vulkan SDK before Milestone 2.")
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
