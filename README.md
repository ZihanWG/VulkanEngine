# VulkanEngine

Modern C++20 Vulkan 1.3 renderer skeleton inspired by the educational flow of [Sascha Willems' HowToVulkan](https://github.com/SaschaWillems/HowToVulkan), but split into engine-style modules instead of a single tutorial file.

The current milestone opens an SDL3 window, creates a Vulkan 1.3 device through Volk, creates a swapchain, uploads cube geometry into GPU-local vertex and index buffers, and draws a rotating depth-tested object every frame using Dynamic Rendering and Synchronization2.

## Dependencies

Required:

- CMake 3.25+
- C++20 compiler, MSVC recommended on Windows
- Vulkan SDK with Vulkan 1.3 headers and `glslc` for shader compilation
- SDL3
- Volk
- Vulkan Memory Allocator
- GLM

The CMake project first looks for installed packages. If they are missing, `VULKAN_ENGINE_FETCH_DEPS=ON` downloads SDL3, Volk, VMA, and GLM with FetchContent.

Milestone 2 and later require `glslc`. CMake compiles shaders into the build-directory shader folder (`<build>/shaders`) and embeds that absolute shader directory in the executable, so running from Visual Studio, CLion, or PowerShell does not depend on the current working directory.

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
- `VulkanSwapchain` owns swapchain images, image views, color/depth image layout tracking, and the depth image used by Dynamic Rendering.
- `VulkanCommandContext` owns the graphics command pool and per-frame command buffers.
- `VulkanSync` owns per-frame fences and semaphores.
- `VulkanPipeline` loads compiled SPIR-V shader modules and creates a Dynamic Rendering graphics pipeline.
- `VulkanBuffer` owns `VkBuffer` plus VMA allocation, supports CPU-visible uploads, staging copies, and optional Buffer Device Address lookup.
- `VulkanImage` owns `VkImage` plus VMA allocation and image view lifetime.

## Vulkan Initialization Flow

1. Initialize Volk global function loading.
2. Create `VkInstance` with SDL3-required extensions and optional debug utils.
3. Install the validation debug messenger in Debug builds.
4. Create `VkSurfaceKHR` from the SDL3 window.
5. Select a Vulkan 1.3 physical device with graphics and present support.
6. Create a logical device with Vulkan 1.3 feature chains.
7. Load device functions through Volk.
8. Create a VMA allocator with Buffer Device Address support.
9. Create swapchain images, image views, depth image, graphics pipeline, command buffers, and synchronization objects.

## One-Frame Rendering Flow

1. Wait for the current frame fence.
2. Acquire the next swapchain image with an image-available semaphore.
3. Reset the fence and command buffer.
4. Record Synchronization2 image barriers:
   - previous layout to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`
   - `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`
5. Transition the depth image to `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL`.
6. Begin Dynamic Rendering with clear color and depth attachments.
7. Bind the graphics pipeline.
8. Upload per-frame MVP data and push that buffer's device address.
9. Set dynamic viewport and scissor from the current swapchain extent.
10. Bind the device-local vertex and index buffers.
11. Draw the rotating cube with `vkCmdDrawIndexed`.
12. End rendering and submit with `vkQueueSubmit2`.
13. Present the image and recreate the swapchain if it is out of date.

## Milestone 2: Triangle Rendering

`src/shaders/simple.vert` and `src/shaders/simple.frag` are compiled by CMake into SPIR-V files under the build directory. `VulkanPipeline` loads those `.spv` files, creates shader modules, creates an empty pipeline layout, and builds a graphics pipeline with `VkPipelineRenderingCreateInfo`.

The empty pipeline layout still matters because Vulkan pipelines always need a layout describing descriptor sets and push constants. It is empty for the triangle milestone, but future MVP data, materials, and bindless descriptors will extend it.

Dynamic Rendering does not use a legacy `VkRenderPass`, so the pipeline declares compatible color and optional depth formats through `VkPipelineRenderingCreateInfo`. Viewport and scissor are dynamic states so resizing the window does not require rebuilding the pipeline when only the extent changes.

## Milestone 3: Vertex/Index Buffer Rendering

`VulkanBuffer` is now the RAII owner for buffer handles and VMA allocations. CPU-visible buffers can be filled through `upload`, while GPU-local buffers use a temporary staging buffer and a one-time `vkCmdCopyBuffer` submission. The copy command records a Synchronization2 buffer barrier so transfer writes are visible to vertex and index fetch.

The renderer uses an explicit `Vertex` layout with position and color, device-local vertex and index buffers, and `vkCmdDrawIndexed`. The pipeline receives explicit vertex binding and attribute descriptions, and `simple.vert` reads locations 0 and 1 instead of generating positions from `gl_VertexIndex`.

## Milestone 4: Depth And MVP

Dynamic Rendering now binds both color and depth attachments. The swapchain depth image is transitioned with Synchronization2 into `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL` before rendering, and the graphics pipeline enables depth testing with the swapchain depth format.

The renderer owns one hard-coded colored cube. Each frame updates an MVP matrix with GLM (`GLM_FORCE_DEPTH_ZERO_TO_ONE` is enabled by CMake) and writes it to that frame's CPU-visible storage buffer. Those frame-data buffers are created with Buffer Device Address support, so the renderer can query each `VkDeviceAddress`.

The vertex shader reads the MVP through `GL_EXT_buffer_reference`. A small vertex-stage push constant carries only the `VkDeviceAddress` of the current frame-data buffer, so no descriptor set is used for MVP data in this milestone.

## Next Milestones

Milestone 5 can extract the hard-coded geometry and temporary MVP calculation into simple `Mesh`, `Camera`, transform, and render-object structures.

Later milestones can turn descriptor indexing support into an optional bindless-style texture array path.
