# VulkanEngine

Modern C++20 Vulkan 1.3 renderer skeleton inspired by the educational flow of [Sascha Willems' HowToVulkan](https://github.com/SaschaWillems/HowToVulkan), but split into engine-style modules instead of a single tutorial file.

The current milestone opens an SDL3 window, creates a Vulkan 1.3 device through Volk, creates a swapchain, uploads cube geometry into GPU-local vertex and index buffers, and draws a rotating textured cube every frame using Dynamic Rendering and Synchronization2.

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

Milestone 2 and later require `glslc`. CMake compiles shaders into the build-directory shader folder, for example `build/shaders`, and embeds that absolute shader directory into the executable, so running from Visual Studio, CLion, or PowerShell does not depend on the current working directory.

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
- `VulkanTexture` owns a sampled 2D image, image view, sampler, and staging upload path.
- `Mesh`, `RenderObject`, `Transform`, and `Camera` provide the first renderer-side scene abstractions without introducing ECS or a render graph.

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
8. Bind the texture descriptor set.
9. Update the render object's transform and upload camera-derived MVP data.
10. Push the MVP buffer device address.
11. Set dynamic viewport and scissor from the current swapchain extent.
12. Bind the device-local vertex and index buffers.
13. Draw render objects with `vkCmdDrawIndexed`.
14. End rendering and submit with `vkQueueSubmit2`.
15. Present the image and recreate the swapchain if it is out of date.

## Milestone 2: Triangle Rendering

`src/shaders/simple.vert` and `src/shaders/simple.frag` are compiled by CMake into SPIR-V files under the build directory. `VulkanPipeline` loads those `.spv` files, creates shader modules, creates a pipeline layout, and builds a graphics pipeline with `VkPipelineRenderingCreateInfo`.

The pipeline layout still matters because Vulkan pipelines always need a layout describing descriptor sets and push constants. The current renderer uses a descriptor set for texture sampling and a vertex-stage push constant for frame data.

Dynamic Rendering does not use a legacy `VkRenderPass`, so the pipeline declares compatible color and optional depth formats through `VkPipelineRenderingCreateInfo`. Viewport and scissor are dynamic states so resizing the window does not require rebuilding the pipeline when only the extent changes.

## Milestone 3: Vertex/Index Buffer Rendering

`VulkanBuffer` is now the RAII owner for buffer handles and VMA allocations. CPU-visible buffers can be filled through `upload`, while GPU-local buffers use a temporary staging buffer and a one-time `vkCmdCopyBuffer` submission. The copy command records a Synchronization2 buffer barrier so transfer writes are visible to vertex and index fetch.

The renderer uses an explicit `Vertex` layout with position and color, device-local vertex and index buffers, and `vkCmdDrawIndexed`. The pipeline receives explicit vertex binding and attribute descriptions, and `simple.vert` reads locations 0 and 1 instead of generating positions from `gl_VertexIndex`.

## Milestone 4: Depth And MVP

Dynamic Rendering now binds both color and depth attachments. The swapchain depth image is transitioned with Synchronization2 into `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL` before rendering, and the graphics pipeline enables depth testing with the swapchain depth format.

Milestone 4 introduced a colored cube, depth testing, and per-frame MVP data. Each frame writes an MVP matrix to that frame's CPU-visible storage buffer. Those frame-data buffers are created with Buffer Device Address support, so the renderer can query each `VkDeviceAddress`.

The vertex shader reads the MVP through `GL_EXT_buffer_reference`. A small vertex-stage push constant carries only the `VkDeviceAddress` of the current frame-data buffer, so no descriptor set is used for MVP data.

## Milestone 5: Scene Abstractions

The hard-coded cube vertex and index data has moved out of `Renderer` and into `Mesh::createCube()`. `Mesh` owns the GPU-local vertex and index buffers for that built-in cube.

`Renderer` now owns a `Camera`, one cube `Mesh`, and a list of `RenderObject` entries. Each `RenderObject` references a `Mesh` and owns its own `Transform`, giving the renderer a simple draw list instead of direct single-cube draw state.

The per-frame MVP is now generated from `Camera + Transform`, then uploaded through the existing Buffer Device Address storage-buffer path. A vertex-stage push constant passes the frame-data buffer address to the shader, which reads the MVP through `GL_EXT_buffer_reference`.

## Milestone 6: Basic Texture Descriptor

Texture sampling now uses a conventional Vulkan descriptor set while the MVP path remains unchanged. `VulkanTexture` creates a GPU-local RGBA8 image, uploads CPU-generated pixels through a staging buffer, transitions the image with Synchronization2 barriers, owns the image view, and creates the sampler used by the fragment shader.

The renderer creates one procedural checkerboard texture in code, allocates one descriptor set at set 0/binding 0, and updates it as a `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`. The pipeline layout now contains both that descriptor set layout and the existing vertex-stage push constant range for the Buffer Device Address frame data.

The cube vertex layout now includes UV coordinates at location 2. The vertex shader passes UVs to the fragment shader, and the fragment shader samples the checkerboard texture. File texture loading, material abstractions, and bindless descriptor arrays are later milestones.

## Next Milestones

Later milestones can introduce texture file loading, material objects, lighting, and optional descriptor indexing for larger texture arrays.
