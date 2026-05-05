# VulkanEngine

Modern C++20 Vulkan 1.3 renderer skeleton inspired by the educational flow of [Sascha Willems' HowToVulkan](https://github.com/SaschaWillems/HowToVulkan), but split into engine-style modules instead of a single tutorial file.

The current milestone opens an SDL3 window, creates a Vulkan 1.3 device through Volk, creates a swapchain, uploads cube geometry with normals and tangents into GPU-local vertex and index buffers, loads small RGBA base color, normal, and metallic-roughness textures from disk with procedural fallbacks, creates and renders a simple procedural environment cubemap as a skybox background, creates low-frequency diffuse irradiance and mipmapped prefiltered specular cubemaps from the same procedural environment colors, generates a 2D split-sum BRDF LUT, and draws multiple independently rotating textured cubes with tangent-space normal mapping, direct-light Cook-Torrance GGX material response, diffuse and specular image-based lighting, directional lighting, and a PCF-filtered directional shadow map every frame using Dynamic Rendering and Synchronization2.

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

Milestone 9 embeds the source `assets` directory path into the executable. The demo tries to load `assets/textures/checker.png`, while later material milestones also load `assets/textures/checker_normal.png` and `assets/textures/checker_mr.png`; if any of those files are missing or cannot be decoded, the renderer falls back to procedural textures.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
.\build\Debug\VulkanEngine.exe
```

For CLion, open this folder as a CMake project and use a Debug profile. Validation layers are enabled only in Debug builds.

The run paths for the demo textures are `assets/textures/checker.png`, `assets/textures/checker_normal.png`, and `assets/textures/checker_mr.png`. CMake embeds the source asset directory, and the renderer uses procedural fallbacks if those PNGs are missing.

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
- `VulkanEnvironmentMap` owns a cube-compatible sampled image, cube image view, and clamp sampler. The renderer uses one generated cubemap for the visible skybox, a second low-frequency generated cubemap for diffuse irradiance, and a mipmapped generated cubemap for prefiltered specular IBL.
- `VulkanBrdfLut` owns the generated 2D `VK_FORMAT_R8G8_UNORM` split-sum BRDF lookup texture used by specular IBL.
- `VulkanShadowMap` owns the fixed-size sampled depth image, image view, sampler, and current layout used by the directional shadow pass.
- `Mesh`, `Material`, `RenderObject`, `Transform`, and `Camera` provide the first renderer-side scene abstractions without introducing ECS or a render graph.

## Current Descriptor Contract

Material descriptor set 0:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- set 0 binding 2 = normal map combined image sampler
- set 0 binding 3 = metallic-roughness combined image sampler
- set 0 binding 4 = diffuse irradiance cubemap combined image sampler
- set 0 binding 5 = prefiltered specular cubemap combined image sampler
- set 0 binding 6 = BRDF LUT combined image sampler

Skybox descriptor set:

- skybox set 0 binding 0 = visible environment cubemap combined image sampler

The material descriptor set above is still separate from the skybox descriptor set. The material shader samples the diffuse irradiance cubemap at binding 4 for environment diffuse lighting, the prefiltered specular cubemap at binding 5, and the BRDF LUT at binding 6; the skybox continues to sample the visible environment cubemap from its own set. There is still no Kulla-Conty term, bindless descriptor path, or model-loading path.

Object and material scalar data still use Buffer Device Address plus a vertex-stage push constant. MVP, model, light, camera, base-color factor, metallic factor, and roughness factor data have not moved into descriptor UBOs.

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
5. Upload per-object MVP/model/light/light-MVP/material data into the current frame's object-data buffer.
6. Record the command buffer.
7. Transition the shadow map to `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL`.
8. Begin depth-only Dynamic Rendering for the shadow pass.
9. Bind the shadow pipeline and draw each `RenderObject` with the BDA object-data push constant.
10. End the shadow pass and transition the shadow map to `VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL`.
11. Transition the swapchain image to `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`.
12. Transition the main depth image to `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL`.
13. Begin main Dynamic Rendering with clear color and depth attachments.
14. Set dynamic viewport and scissor from the current swapchain extent.
15. Bind the skybox pipeline and skybox descriptor set 0.
16. Push the skybox inverse view-projection matrix and draw a fullscreen triangle.
17. Bind the main graphics pipeline.
18. For each `RenderObject`, bind its material descriptor set 0.
19. Push that object's object-data buffer device address.
20. Bind the object's device-local vertex and index buffers.
21. Draw the object with `vkCmdDrawIndexed`.
22. End Dynamic Rendering.
23. Transition the swapchain image to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`.
24. Submit with `vkQueueSubmit2`.
25. Present the image and recreate the swapchain if it is out of date.

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
- shadow pass = depth-only Dynamic Rendering from the directional light
- main pass = fragment shader shadow-map sampling with simple 3x3 manual PCF
- no PBR, normal maps, bindless descriptors, model loading, ECS, ImGui, or render graph

Future shadow and lighting work includes cascaded shadow maps, texel snapping for stable shadows, variance or EVSM shadows, PBR, IBL, and a render graph.

## Milestone 13: Basic PBR Material Parameters

Milestone 13 adds minimal PBR-style material parameters without changing the descriptor layout. `Material` now stores `baseColorFactor`, `metallic`, and `roughness` in addition to its debug name, base color texture pointer, and descriptor set.

Material parameters are passed through the existing Buffer Device Address object-data path. Each `ObjectFrameData` entry now includes `baseColorFactor`, `materialParams`, and `cameraPosition`; `materialParams.x` is metallic, `materialParams.y` is roughness, and `materialParams.zw` are reserved.

The fragment shader still samples the base color texture from descriptor set 0 binding 0 and the shadow map from descriptor set 0 binding 1. It multiplies the texture by `baseColorFactor`, then applies a simple non-IBL diffuse plus Blinn-style specular approximation controlled by roughness and metallic.

This was not full PBR yet. At Milestone 13 there was still no BRDF LUT, IBL, Kulla-Conty multi-scattering compensation, normal maps, metallic/roughness texture maps, bindless material descriptors, model loading, ECS, ImGui, or render graph.

Cook-Torrance GGX, normal mapping, and metallic-roughness texture support are now covered by later milestones. Remaining material and lighting work is tracked in the Next Milestones section.

## Milestone 14: Cook-Torrance GGX Direct Lighting

Milestone 14 replaces the Milestone 13 Blinn-style specular approximation with a direct-light Cook-Torrance GGX BRDF in the fragment shader. The renderer still samples the base color texture from descriptor set 0 binding 0 and the shadow map from descriptor set 0 binding 1, with material values coming through the existing Buffer Device Address object-data path.

The shader computes base color from the texture multiplied by `baseColorFactor`, reads metallic from `materialParams.x`, and reads roughness from `materialParams.y`. Roughness is clamped to `[0.04, 1.0]` to avoid unstable highlights. The direct light BRDF now uses the GGX / Trowbridge-Reitz normal distribution function, Smith geometry function, and Schlick Fresnel approximation. Metallic and roughness now affect the diffuse/specular energy split, `F0`, highlight width, and specular intensity.

Lighting was still direct lighting only at Milestone 14. The PCF-filtered directional shadow factor still modulated the direct light, and ambient remained a simple unshadowed term. There was still no IBL, split-sum BRDF LUT, Kulla-Conty multi-scattering compensation, normal maps, metallic/roughness textures, bindless descriptors, model loading, ECS, ImGui, or render graph.

Normal mapping and metallic-roughness maps are now implemented in Milestones 15 and 16. Remaining material and lighting work is tracked in the Next Milestones section.

## Milestone 15: Basic Normal Mapping

Milestone 15 adds basic tangent-space normal mapping while keeping the renderer architecture simple. Mesh vertices now contain position, color, UV, normal, and tangent attributes. The built-in cube still uses duplicated vertices per face, and its tangents are hardcoded per face; general tangent generation for imported meshes is future work.

`Material` can now reference a normal map in addition to its base color texture. Descriptor set 0 keeps the existing bindings and adds one new sampler:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- set 0 binding 2 = normal map combined image sampler

The vertex shader reads the tangent at location 4, transforms the normal and tangent to world space, computes the bitangent from `cross(normal, tangent) * tangent.w`, and passes the TBN basis to the fragment shader. The fragment shader samples the normal map, decodes the tangent-space normal from `[0, 1]` to `[-1, 1]`, transforms it through TBN, and uses that world-space normal for Cook-Torrance GGX direct lighting and the shadow bias path.

`Renderer` loads `assets/textures/checker_normal.png` when present. If that file is missing or cannot be decoded, it creates a small procedural flat normal texture with RGBA `(128, 128, 255, 255)` and still binds it at descriptor set 0 binding 2. This keeps materials descriptor-complete without adding shader branching or dynamic descriptor behavior.

Object data still uses Buffer Device Address plus a vertex-stage push constant. Normal map state stays in the material descriptor set; `ObjectFrameData` is unchanged. This milestone is still not IBL, a BRDF LUT, Kulla-Conty multi-scattering compensation, bindless descriptors, model loading, glTF, ECS, ImGui, or a render graph.

## Milestone 16: Metallic-Roughness Texture Map

Milestone 16 adds a basic metallic-roughness texture map while keeping the same simple material and object-data architecture. `Material` can now reference a metallic-roughness texture in addition to its base color and normal textures. `Renderer` loads `assets/textures/checker_mr.png` when available; if the file is missing or cannot be decoded, it creates a small procedural fallback texture instead.

Object data still uses Buffer Device Address plus a vertex-stage push constant, and `materialParams.xy` remain the scalar metallic and roughness factors.

Descriptor set 0 is still the material/shadow texture set:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- set 0 binding 2 = normal map combined image sampler
- set 0 binding 3 = metallic-roughness combined image sampler

The metallic-roughness texture uses the R channel as the metallic factor and the G channel as the roughness factor. B and A are unused by the shader. The fragment shader multiplies the sampled texture values by the scalar material factors:

```glsl
metallic = clamp(materialMetallic * textureMetallic, 0.0, 1.0);
roughness = clamp(materialRoughness * textureRoughness, 0.04, 1.0);
```

The procedural fallback uses neutral R/G factors so the existing scalar `Material::metallic` and `Material::roughness` values remain the visible fallback behavior.

GGX direct lighting uses the resulting metallic and roughness values together with the existing normal map and PCF shadow paths. This is still direct lighting only: no IBL, no split-sum BRDF LUT, no Kulla-Conty multi-scattering compensation, and not full glTF material support.

## Milestone 17: IBL Preparation and Environment Texture Infrastructure

Milestone 17 prepares the renderer for image-based lighting without changing the lighting model yet. A new `VulkanEnvironmentMap` wrapper owns a cube-compatible `VkImage`, VMA allocation, `VK_IMAGE_VIEW_TYPE_CUBE` view, and clamp-to-edge sampler. The renderer creates a small generated six-face RGBA8 cubemap during scene setup so later skybox and image-based-lighting milestones have a real GPU cubemap resource to build from.

The environment map upload path uses the same explicit staging-buffer and Synchronization2 style as the existing texture code: all six cube faces are copied into array layers 0 through 5, then transitioned to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. At Milestone 17, the shader pipeline did not sample this cubemap, so descriptor set 0 remained unchanged and no environment descriptor binding was added in that milestone.

This is intentionally infrastructure only. There is still no split-sum BRDF LUT, no Kulla-Conty multi-scattering compensation, no bindless descriptors, no skybox draw, no environment prefiltering, and no model loading.

## Milestone 18: Skybox Rendering

Milestone 18 renders the procedural environment cubemap from Milestone 17 as a skybox background. The skybox uses a fullscreen triangle, a separate graphics pipeline, and a separate descriptor set layout where skybox set 0 binding 0 is the visible environment cubemap combined image sampler.

At Milestone 18, material descriptor set 0 still contained only the mesh texture and shadow bindings from 0 through 3. Object data continued to use Buffer Device Address plus the existing vertex-stage push constant. The skybox has its own descriptor set and its own vertex-stage push constant containing the inverse view-projection matrix with camera translation removed.

The main Dynamic Rendering pass clears color/depth, draws the skybox first with depth writes disabled, then draws the normal `RenderObject` meshes as before. The shadow pass is unchanged and still runs before the main pass.

Milestone 19 kept the visible skybox cubemap and diffuse irradiance cubemap as separate resources. The skybox cubemap remains the background source, while mesh materials sample the diffuse irradiance cubemap for ambient/environment diffuse lighting.

Later environment work can still add Kulla-Conty multi-scattering compensation, HDR environment loading, bindless descriptors, model loading, and a render graph.

## Milestone 19: Diffuse IBL Irradiance

Milestone 17 created the reusable environment cubemap resource, and Milestone 18 rendered that cubemap as a visible skybox. Milestone 19 adds simple diffuse image-based lighting while keeping the visible skybox path unchanged. The renderer now owns both `environmentMap_` for the skybox and `diffuseIrradianceMap_` for mesh materials.

The diffuse irradiance cubemap is generated procedurally on the CPU from the same six environment face colors, stored as a small low-frequency RGBA8 cubemap, uploaded through the existing `VulkanEnvironmentMap` staging-buffer path, and sampled as a cube image.

Mesh material descriptor set 0 now adds one fragment-stage binding:

- set 0 binding 0 = base color texture
- set 0 binding 1 = shadow map
- set 0 binding 2 = normal map
- set 0 binding 3 = metallic-roughness map
- set 0 binding 4 = diffuse irradiance cubemap
- skybox set 0 binding 0 = visible environment cubemap

The fragment shader samples `uDiffuseIrradianceMap` with the current world-space normal after tangent-space normal mapping. Diffuse IBL contributes `irradiance * baseColor * (1.0 - metallic)` as the ambient/environment diffuse term, with the old ambient color retained only as a small fallback. Direct Cook-Torrance GGX lighting, Schlick Fresnel, Smith geometry, the GGX NDF, metallic-roughness sampling, normal mapping, and PCF shadow filtering remain unchanged; the shadow factor still affects direct lighting only.

This milestone is diffuse IBL only. There is still no prefiltered specular environment map, split-sum BRDF LUT, Kulla-Conty multi-scattering compensation, HDR environment loading, bindless descriptors, model loading, ECS, ImGui, or render graph.

## Milestone 20: Specular IBL and BRDF LUT

Milestone 20 adds basic split-sum specular image-based lighting while keeping the skybox descriptor set separate and keeping object/material scalar data on the Buffer Device Address plus vertex-stage push-constant path.

The renderer now owns `environmentMap_` for the visible skybox, `diffuseIrradianceMap_` for diffuse IBL, `prefilteredEnvironmentMap_` for specular IBL, and `brdfLutTexture_` for the split-sum BRDF lookup. The prefiltered specular cubemap is generated on the CPU from the existing procedural environment colors as a mip chain: low roughness mips preserve the face gradients, and higher roughness mips blend toward low-frequency face/global colors. This is a readable approximation, not full importance-sampled environment prefiltering.

The BRDF LUT is a generated 256x256 `VK_FORMAT_R8G8_UNORM` 2D texture. It stores the split-sum scale/bias terms from a small CPU-side Hammersley/GGX integration.

Material descriptor set 0 now contains:

- set 0 binding 0 = base color texture
- set 0 binding 1 = shadow map
- set 0 binding 2 = normal map
- set 0 binding 3 = metallic-roughness map
- set 0 binding 4 = diffuse irradiance cubemap
- set 0 binding 5 = prefiltered specular cubemap
- set 0 binding 6 = BRDF LUT

The fragment shader combines direct Cook-Torrance GGX lighting, PCF shadows on direct light only, diffuse IBL from the irradiance cubemap, and specular IBL from the prefiltered environment plus BRDF LUT. This is still not Kulla-Conty, HDR environment loading, bindless rendering, descriptor indexing arrays, model loading, or a render graph.

## Next Milestones

Future milestones can build on this multi-object material foundation with:

- Kulla-Conty multi-scattering compensation
- HDR environment loading
- proper importance-sampled prefiltering
- bindless descriptors
- model loading
- render graph
