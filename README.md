# VulkanEngine

Modern C++20 Vulkan 1.3 renderer skeleton inspired by the educational flow of [Sascha Willems' HowToVulkan](https://github.com/SaschaWillems/HowToVulkan), but split into engine-style modules instead of a single tutorial file.

This first milestone opens an SDL3 window, creates a Vulkan 1.3 device through Volk, creates a swapchain, and clears the current swapchain image every frame using Dynamic Rendering and Synchronization2.

## Dependencies

Required:

- CMake 3.25+
- C++20 compiler, MSVC recommended on Windows
- Vulkan SDK with Vulkan 1.3 headers and `glslc`
- SDL3
- Volk
- Vulkan Memory Allocator
- GLM

The CMake project first looks for installed packages. If they are missing, `VULKAN_ENGINE_FETCH_DEPS=ON` downloads SDL3, Volk, VMA, and GLM with FetchContent.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
.\build\Debug\VulkanEngine.exe
```

For CLion, open this folder as a CMake project and use a Debug profile. Validation layers are enabled only in Debug builds.

## Architecture

- `Application` owns the `Window` and `Renderer`.
- `Window` owns SDL initialization, the native window, Vulkan instance extensions, and surface creation.
- `Renderer` owns the frame loop and orchestrates frame resources.
- `VulkanContext` owns the Vulkan instance, debug messenger, surface, selected device, queues, and VMA allocator.
- `VulkanDevice` selects a Vulkan 1.3 GPU, finds queue families, enables Synchronization2, Dynamic Rendering, Buffer Device Address, and descriptor indexing features when supported.
- `VulkanSwapchain` owns swapchain images, image views, image layout tracking, and a depth image for later milestones.
- `VulkanCommandContext` owns the graphics command pool and per-frame command buffers.
- `VulkanSync` owns per-frame fences and semaphores.
- `VulkanImage` and `VulkanBuffer` are RAII resource wrappers around Vulkan objects plus VMA allocations.

## Vulkan Initialization Flow

1. Initialize Volk global function loading.
2. Create `VkInstance` with SDL3-required extensions and optional debug utils.
3. Install the validation debug messenger in Debug builds.
4. Create `VkSurfaceKHR` from the SDL3 window.
5. Select a Vulkan 1.3 physical device with graphics and present support.
6. Create a logical device with Vulkan 1.3 feature chains.
7. Load device functions through Volk.
8. Create a VMA allocator with Buffer Device Address support.
9. Create swapchain images, image views, depth image, command buffers, and synchronization objects.

## One-Frame Rendering Flow

1. Wait for the current frame fence.
2. Acquire the next swapchain image with an image-available semaphore.
3. Reset the fence and command buffer.
4. Record Synchronization2 image barriers:
   - previous layout to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`
   - `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`
5. Begin Dynamic Rendering with a clear color attachment.
6. End rendering and submit with `vkQueueSubmit2`.
7. Present the image and recreate the swapchain if it is out of date.

## Next Milestones

Milestone 2 should compile `simple.vert` and `simple.frag`, create a graphics pipeline using Dynamic Rendering state, and issue `vkCmdDraw` for a triangle.

Milestone 3 should expand `VulkanBuffer` into a staging upload path and add vertex and index buffer helpers.

Milestone 4 should connect `Mesh`, `Material`, `Camera`, transforms, and MVP shader data into a simple render object list.

Milestone 5 should turn the descriptor indexing support into an optional bindless-style texture array path.
