# VulkanEngine

Modern C++20 Vulkan 1.3 renderer skeleton inspired by the educational flow of [Sascha Willems' HowToVulkan](https://github.com/SaschaWillems/HowToVulkan), but split into engine-style modules instead of a single tutorial file.

The current milestone opens an SDL3 window, creates a Vulkan 1.3 device through Volk, creates a swapchain, uploads cube geometry with normals into GPU-local vertex and index buffers, loads a small RGBA texture from disk with a procedural fallback, and draws multiple independently rotating textured cubes with minimal directional lighting and a PCF-filtered directional shadow map every frame using Dynamic Rendering and Synchronization2.

## Dependencies

Required:

- CMake 3.25+
- C++20 compiler, MSVC recommended on Windows
- Vulkan SDK with Vulkan 1.3 headers and `glslc` for shader compilation
- SDL3
- Volk
- Vulkan Memory Allocator
- GLM
- stb_image, vendored as `external/stb_image.h`

The CMake project first looks for installed packages. If they are missing, `VULKAN_ENGINE_FETCH_DEPS=ON` downloads SDL3, Volk, VMA, and GLM with FetchContent.

Milestone 2 and later require `glslc`. CMake compiles shaders into the build-directory shader folder, for example `build/shaders`, and embeds that absolute shader directory into the executable, so running from Visual Studio, CLion, or PowerShell does not depend on the current working directory.

Milestone 9 embeds the source `assets` directory path into the executable. The demo tries to load `assets/textures/checker.png`; if that file is missing or cannot be decoded, the renderer falls back to its procedural checkerboard texture.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
.\build\Debug\VulkanEngine.exe
```

For CLion, open this folder as a CMake project and use a Debug profile. Validation layers are enabled only in Debug builds.

The run path for the demo texture is `assets/textures/checker.png`. CMake embeds the source asset directory, and the renderer uses the procedural checkerboard fallback if that PNG is missing.

## Validated Environment

Validated locally on:

- Windows
- Visual Studio 2022 MSVC x64
- Vulkan SDK 1.4.328.1
- NVIDIA GeForce RTX 3080 Ti Laptop GPU

Galaxy overlay layer naming warnings may appear in Debug runs. They come from an external Vulkan layer and are unrelated to renderer validation.

## Architecture

- `Application` owns the `Window` and `Renderer`.
- `Window` owns SDL initialization, the native window, Vulkan instance extensions, and surface creation.
- `Renderer` owns the frame loop and orchestrates frame resources.
- `VulkanContext` owns the Vulkan instance, debug messenger, surface, selected device, queues, and VMA allocator.
- `VulkanDevice` selects a Vulkan 1.3 GPU, finds queue families, enables Synchronization2, Dynamic Rendering, Buffer Device Address, and descriptor indexing features when supported.
- `VulkanSwapchain` owns swapchain images, image views, color/depth image layout tracking, and the depth image used by Dynamic Rendering.
- `VulkanCommandContext` owns the graphics command pool and per-frame command buffers.
- `VulkanSync` owns per-frame image-available semaphores and fences, plus render-finished semaphores scoped per swapchain image.
- `VulkanPipeline` loads compiled SPIR-V shader modules and creates a Dynamic Rendering graphics pipeline.
- `VulkanBuffer` owns `VkBuffer` plus VMA allocation, supports CPU-visible uploads, staging copies, and optional Buffer Device Address lookup.
- `VulkanImage` owns `VkImage` plus VMA allocation and image view lifetime.
- `VulkanTexture` owns a sampled image, VMA allocation, image view, and sampler, and uploads RGBA8 texture data through a staging buffer. It can load image files through stb_image, generate mipmaps on the GPU when supported, or use a procedural checkerboard fallback.
- `VulkanShadowMap` owns the fixed-size sampled depth image, image view, sampler, and current layout used by the directional shadow pass.
- `Mesh`, `Material`, `RenderObject`, `Transform`, and `Camera` provide the first renderer-side scene abstractions without introducing ECS or a render graph.

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
4. Update all object transforms.
5. Upload per-object MVP/model/light/light-MVP data into the current frame's object-data buffer.
6. Record the command buffer.
7. Transition the shadow map to `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL`.
8. Begin depth-only Dynamic Rendering for the shadow pass.
9. Bind the shadow pipeline and draw each `RenderObject` with the BDA object-data push constant.
10. End the shadow pass and transition the shadow map to `VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL`.
11. Transition the swapchain image to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`.
12. Transition the main depth image to `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL`.
13. Begin main Dynamic Rendering with clear color and depth attachments.
14. Bind the main graphics pipeline.
15. Set dynamic viewport and scissor from the current swapchain extent.
16. For each `RenderObject`, bind its material descriptor set 0.
17. Push that object's object-data buffer device address.
18. Bind the object's device-local vertex and index buffers.
19. Draw the object with `vkCmdDrawIndexed`.
20. End Dynamic Rendering.
21. Transition the swapchain image to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
22. Submit with `vkQueueSubmit2`.
23. Present the image and recreate the swapchain if it is out of date.

## Milestone 2: Triangle Rendering

`src/shaders/simple.vert` and `src/shaders/simple.frag` are compiled by CMake into SPIR-V files under the build directory. `VulkanPipeline` loads those `.spv` files, creates shader modules, creates a pipeline layout, and builds a graphics pipeline with `VkPipelineRenderingCreateInfo`.

The pipeline layout still matters because Vulkan pipelines always need a layout describing descriptor sets and push constants. At this stage, the renderer used a vertex-stage push constant for MVP data, while descriptor sets were intentionally left for later texture and sampler work.

Dynamic Rendering does not use a legacy `VkRenderPass`, so the pipeline declares compatible color and optional depth formats through `VkPipelineRenderingCreateInfo`. Viewport and scissor are dynamic states so resizing the window does not require rebuilding the pipeline when only the extent changes.

## Milestone 3: Vertex/Index Buffer Rendering

`VulkanBuffer` is now the RAII owner for buffer handles and VMA allocations. CPU-visible buffers can be filled through `upload`, while GPU-local buffers use a temporary staging buffer and a one-time `vkCmdCopyBuffer` submission. The copy command records a Synchronization2 buffer barrier so transfer writes are visible to vertex and index fetch.

The renderer uses an explicit `Vertex` layout with position and color, device-local vertex and index buffers, and `vkCmdDrawIndexed`. The pipeline receives explicit vertex binding and attribute descriptions, and `simple.vert` reads locations 0 and 1 instead of generating positions from `gl_VertexIndex`.

## Milestone 4: Depth And MVP

Dynamic Rendering now binds both color and depth attachments. The swapchain depth image is transitioned with Synchronization2 into `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL` before rendering, and the graphics pipeline enables depth testing with the swapchain depth format.

Milestone 4 introduced a colored cube, depth testing, and per-frame MVP data. Each frame wrote an MVP matrix to that frame's CPU-visible storage buffer. Those buffers were created with Buffer Device Address support, so the renderer could query each `VkDeviceAddress`.

The vertex shader reads the MVP through `GL_EXT_buffer_reference`. A small vertex-stage push constant carries only the `VkDeviceAddress` of the MVP data, so no descriptor set is used for MVP data in this milestone.

## Milestone 5: Scene Abstractions

The hard-coded cube vertex and index data has moved out of `Renderer` and into `Mesh::createCube()`. `Mesh` owns the GPU-local vertex and index buffers for that built-in cube.

`Renderer` now owns a `Camera`, one cube `Mesh`, and a list of `RenderObject` entries. Each `RenderObject` references a `Mesh` and owns its own `Transform`, giving the renderer a simple draw list instead of direct single-cube draw state.

The MVP is generated from `Camera + Transform`, then uploaded through the existing Buffer Device Address storage-buffer path. A vertex-stage push constant passes the MVP data address to the shader, which reads the MVP through `GL_EXT_buffer_reference`. Descriptor sets are not used for MVP data.

## Milestone 6: Basic Texture Descriptor

Milestone 6 is implemented and introduces descriptor sets only for texture sampling. MVP still uses the existing Buffer Device Address storage-buffer path, with a vertex-stage push constant carrying the current MVP data address. The vertex shader still reads MVP through `GL_EXT_buffer_reference`; it has not moved to a uniform buffer descriptor.

The texture binding contract is:

- set 0, binding 0
- `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`
- fragment shader visibility

At Milestone 6, the texture was still a CPU-generated RGBA8 checkerboard. No image files were loaded at that stage, and no `stb_image` dependency was used yet.

Texture upload uses a CPU-visible staging buffer, a GPU-local `VkImage`, `vkCmdCopyBufferToImage`, and Synchronization2 image barriers:

- `VK_IMAGE_LAYOUT_UNDEFINED` to `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`
- `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`

The frame binding flow is now:

1. Bind pipeline.
2. Bind texture descriptor set 0.
3. Push object MVP buffer device address.
4. Bind vertex and index buffers.
5. Draw indexed.

Milestone 9 later adds stb_image-based file texture loading and GPU mipmap generation. At Milestone 6, bindless descriptors, lighting, model loading, and render graph work were still future milestones.

## Milestone 7: Basic Material Abstraction

`Material` is now the minimal link between a render object and texture sampling state. It stores a debug name, references a base color `VulkanTexture`, and stores the descriptor set used by the fragment shader's texture binding.

`Renderer` still owns the actual checkerboard `VulkanTexture`, the checkerboard `Material`, the cube `Mesh`, the `Camera`, and the `RenderObject` list. `RenderObject` now references both `Mesh` and `Material`, while continuing to own its `Transform` and debug name.

The descriptor contract is unchanged from Milestone 6:

- set 0, binding 0
- `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`
- fragment shader visibility

MVP data still uses Buffer Device Address plus a vertex-stage push constant. The vertex shader still reads the MVP through `GL_EXT_buffer_reference`, and MVP data has not moved into descriptor uniform buffers.

The per-object draw flow is now:

1. Bind pipeline.
2. For each `RenderObject`, bind its material descriptor set.
3. Push the object MVP buffer device address.
4. Bind the object's mesh vertex and index buffers.
5. Draw indexed.

This milestone does not add PBR, lighting, bindless descriptors, descriptor indexing texture arrays, file texture loading, or model loading.

## Milestone 8: Multi-Object Scene

Milestone 8 is implemented. `Renderer` now draws a small scene with several cube `RenderObject` entries: center, left, right, and elevated cubes. Each render object references the shared cube `Mesh`, references the shared checkerboard `Material`, owns its own `Transform`, and carries a debug name.

MVP data is now per object. Each frame owns one CPU-visible storage buffer large enough for multiple `ObjectFrameData` entries. `updateFrameData()` animates object transforms independently, computes `projection * view * model` for each object, and uploads the resulting MVP matrices into that frame's object-data buffer.

The shader still uses `GL_EXT_buffer_reference`. For each draw, the renderer pushes the Buffer Device Address of the current object's `ObjectFrameData` entry to the vertex stage. MVP data still does not use uniform buffer descriptors.

The texture path is unchanged in Milestone 8: texture/sampler data still uses descriptor set 0 binding 0 as a combined image sampler visible to the fragment shader.

This milestone does not add lighting, PBR, bindless descriptors, file texture loading, model loading, ECS, ImGui, or a render graph.

## Milestone 9: File Texture Loading and Mipmaps

Milestone 9 adds stb_image-based file texture loading and GPU mipmap generation while keeping the existing renderer contracts intact. `VulkanTexture::createFromFile()` loads image data from disk with stb_image, forces RGBA8 pixels, uploads through a CPU-visible staging buffer, and stores the result in a GPU-local `VkImage` allocated with VMA.

When mipmap generation is requested, the texture computes `floor(log2(max(width, height))) + 1` mip levels, creates the image with transfer source, transfer destination, and sampled usage, then generates the mip chain on the GPU with `vkCmdBlitImage`. Synchronization2 image barriers transition all levels from `UNDEFINED` to `TRANSFER_DST_OPTIMAL`, copy level 0, move each previous level to `TRANSFER_SRC_OPTIMAL`, blit into the next level, and finally transition every level to `SHADER_READ_ONLY_OPTIMAL`.

If the format does not support the blit path needed for this simple GPU mip generation, the texture falls back to one mip level. The sampler uses linear min/mag filtering, linear mip filtering, repeat addressing, and a `maxLod` matching the texture mip count. Anisotropy stays disabled for now because the current device wrapper does not explicitly expose and enable `samplerAnisotropy`.

The texture/sampler descriptor contract remains set 0, binding 0 as a combined image sampler. MVP data still uses Buffer Device Address plus a vertex-stage push constant, and the vertex shader still reads per-object MVP data through `GL_EXT_buffer_reference`; MVP data has not moved into uniform buffer descriptors.

`Renderer` tries to load `assets/textures/checker.png` into the existing `checkerboardMaterial_`. If the asset is absent or stb_image fails to decode it, the procedural checkerboard path remains as the fallback. `Material` remains minimal: debug name, base color texture pointer, and descriptor set. This milestone does not add PBR parameters, normal maps, material asset files, bindless descriptors, model loading, lighting, ECS, ImGui, or a render graph.

## Milestone 10: Basic Directional Lighting

Milestone 10 is implemented and adds minimal, non-PBR directional lighting to the textured cube scene. Mesh vertices now contain position, color, UV, and normal attributes. The built-in cube still uses duplicated vertices per face so each face has clean flat normals and UVs.

The vertex shader keeps the existing `GL_EXT_buffer_reference` path. A vertex-stage push constant still carries the Buffer Device Address of the current object's `ObjectFrameData` entry. That entry now contains MVP, model, light direction, light color, and ambient color values. MVP/object data has not moved to uniform-buffer descriptors or any other descriptor set.

Normals are transformed to world space in the vertex shader with `transpose(inverse(mat3(model)))`, then passed to the fragment shader. The fragment shader keeps the texture/sampler at descriptor set 0 binding 0, samples the base color texture, and applies a simple Lambert diffuse term with a small ambient contribution:

```glsl
baseColor * vertexColor * (ambient + diffuse)
```

`Material` remains minimal: debug name, base color texture pointer, and descriptor set. PBR, specular BRDFs, normal maps, IBL/image-based lighting, material parameter buffers, bindless descriptors, model loading, ImGui, ECS, and render graph work remain future milestones.

## Milestone 11: Basic Shadow Mapping

Milestone 11 adds a minimal directional shadow map for the existing directional light. A fixed 2048x2048 sampled depth image is rendered first with a depth-only Dynamic Rendering pass from a simple orthographic light camera covering the cube scene. The shadow pass uses a vertex-only pipeline, depth writes, and static depth bias to reduce acne.

The main pass samples that depth image in the fragment shader and performs one manual depth comparison. The base color texture remains descriptor set 0 binding 0, and the shadow map is descriptor set 0 binding 1. One material descriptor set is still used for now; there are no descriptor arrays or bindless resources.

Object data still uses Buffer Device Address plus a vertex-stage push constant. `ObjectFrameData` now contains `mvp`, `model`, `lightMvp`, light direction, light color, and ambient color. MVP and lighting data have not moved to UBO descriptors.

The Milestone 11 shader/resource contract is:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- object data = Buffer Device Address plus a vertex-stage push constant
- shadow pass = depth-only Dynamic Rendering from the directional light
- main pass = fragment shader samples the shadow map and applies a single depth comparison

This is intentionally not a cascaded shadow implementation. PCF, cascaded shadow maps, better shadow filtering, stable texel snapping, broader scene fitting, PBR, normal maps, multiple lights, model loading, ECS, ImGui, and a render graph remain future work.

## Milestone 12: Shadow Quality Improvements

Milestone 12 improves the existing directional shadow map without changing the renderer structure. The shadow pass is still one depth-only Dynamic Rendering pass, and the main graphics pipeline still samples the shadow map from descriptor set 0 binding 1.

The fragment shader now uses simple manual 3x3 PCF by averaging neighboring shadow-map depth comparisons. This softens jagged shadow edges compared with the Milestone 11 single-sample comparison while keeping sampler compare mode disabled for now.

`Renderer` now owns tunable shadow settings for shadow-map resolution, small shader-side constant/slope bias values, PCF enable/radius, and static rasterizer depth-bias factors. The static rasterizer depth bias remains on the shadow pipeline to reduce acne, while the shader-side bias stays small to avoid obvious peter panning.

The directional light projection now comes from a documented fixed bounding sphere that covers the current rotating cube demo. This keeps the orthographic near/far planes stable and gives the fixed 2048 shadow map a tighter useful area. This is acceptable for the current static demo scene, but it is still not cascaded shadow mapping, camera-frustum fitting, or texel snapping.

The Milestone 12 resource contract remains:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- object data = Buffer Device Address plus a vertex-stage push constant
- no PBR, normal maps, bindless descriptors, model loading, ECS, ImGui, or render graph

Future shadow and lighting work includes cascaded shadow maps, texel snapping for stable shadows, variance or EVSM shadows, PBR, IBL, and a render graph.

## Next Milestones

Future milestones can build on this multi-object material foundation with richer material parameters, improved shadow quality, bindless descriptors, model loading, and render graph work once the minimal texture, lighting, and shadow path is stable.
