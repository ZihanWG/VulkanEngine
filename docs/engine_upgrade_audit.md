# Engine Upgrade Audit

Phase 0 records the current renderer shape before any behavioral changes. The
project is already a feature-rich Vulkan renderer demo, so the upgrade should
grow existing systems instead of replacing them.

## Current Repository Shape

- `src/core/` owns application startup, SDL3 window creation, event polling, and
  the main loop.
- `src/rhi/` owns Vulkan instance/device/swapchain/resource wrappers,
  Synchronization2 helpers, debug labels, timestamp queries, VMA-backed buffers
  and images, texture upload, shadow maps, environment maps, and BRDF LUTs.
- `src/renderer/` owns frame orchestration, scene data, draw item construction,
  material state, the bindless material texture heap, runtime settings, and the
  current manual render graph.
- `src/ui/` owns Dear ImGui integration through `ImGuiLayer`.
- `src/shaders/` contains all GLSL sources compiled by CMake with `glslc`.
- `assets/` contains the committed test glTF and fallback textures. Large HDR
  environment assets are intentionally external.
- `config/` contains the runtime settings example JSON.
- `external/` vendors Dear ImGui, `stb_image.h`, `tiny_gltf.h`, and
  `json.hpp`.
- `.github/workflows/windows-ci.yml` configures the Windows MSVC Vulkan build,
  compiles shaders, and builds the renderer.

## Build And Shader Path

The root `CMakeLists.txt` defines a C++20 executable and an `imgui` static
library. It first tries installed packages, then fetches SDL3, GLM, Volk, and
Vulkan Memory Allocator from pinned tags when `VULKAN_ENGINE_FETCH_DEPS=ON`.
The Vulkan SDK is required for headers and `glslc`.

CMake embeds these absolute paths into the executable:

- `VULKAN_ENGINE_SHADER_DIR`
- `VULKAN_ENGINE_ASSET_DIR`
- `VULKAN_ENGINE_CONFIG_DIR`

Shader sources are compiled into the build directory with:

```text
glslc <shader> -o <shader>.spv --target-env=vulkan1.3
```

The current shader set covers main PBR shading, a bindless variant, skybox,
shadow depth, compute frustum culling, log-average luminance, histogram
exposure, fullscreen triangle, bloom extract, bloom blur, and final composite.

## Runtime Entry Points

- `src/main.cpp` constructs `ve::Application` and calls `run()`.
- `Application::initialize()` creates `Window`, then `Renderer`, then forwards
  SDL events to `Renderer::handleEvent()`.
- `Application::mainLoop()` polls events and calls `Renderer::drawFrame()` until
  the window closes.
- `Renderer::drawFrame()` is the frame entry point for all rendering work.

## Vulkan Device, Swapchain, And Resource Ownership

- `VulkanContext` owns the Vulkan instance, debug messenger, SDL surface,
  selected physical/logical device, queues, and VMA allocator.
- `VulkanDevice` requires Vulkan 1.3, Dynamic Rendering, Synchronization2,
  Buffer Device Address, separate depth/stencil layouts, and the swapchain
  extension. Descriptor indexing, multi-draw indirect, first-instance indirect
  drawing, and indirect count drawing are enabled only when supported.
- `VulkanSwapchain` owns swapchain images, image views, a main depth image, and
  tracked image layouts.
- `VulkanSync` owns per-frame acquire semaphores/fences and per-swapchain-image
  render-finished semaphores.
- `VulkanCommandContext` owns one resettable graphics command pool and the
  per-frame primary command buffers.
- `VulkanBuffer`, `VulkanImage`, `VulkanTexture`, `VulkanShadowMap`,
  `VulkanEnvironmentMap`, and `VulkanBrdfLut` are RAII wrappers around Vulkan
  handles and VMA allocations.
- `VulkanDebugUtils` wraps optional `VK_EXT_debug_utils` object names and command
  labels.
- `VulkanTimestampQuery` owns a timestamp query pool shared across frame slots.

## One-Frame Flow

The current frame flow is:

1. Skip when minimized, update CPU frame timing, and recreate the swapchain if
   the window resize flag is set.
2. Wait for the current frame fence.
3. Read previous GPU readbacks for exposure, culling counters, and timestamp
   results without stalling the current frame.
4. Acquire a swapchain image and wait for any older frame still using that image.
5. Reset the fence and current frame command buffer.
6. Build ImGui draw data.
7. Update CPU-side scene/draw data, culling inputs, indirect commands, cascades,
   and per-draw `ObjectFrameData`.
8. Record render commands through the manual `RenderGraph`.
9. Submit with `vkQueueSubmit2`, signal a per-image render-finished semaphore,
   mark timestamp queries submitted, and present.
10. Recreate the swapchain on out-of-date/suboptimal presentation.

## Current Render Graph

`renderer::RenderGraph` is intentionally manual. It records pass metadata for
debug UI and centralizes image transitions and Dynamic Rendering setup for:

- `CSMShadowPass`
- `MainHDRPass`
- `BloomExtractPass`
- `BloomBlurPass`
- `LuminancePass`
- `HistogramExposurePass`
- `CompositePass`
- `ImGuiPass`

It currently handles selected image resources only: scene color, bloom extract,
bloom ping/pong, the shadow-map array, swapchain color images, and main depth.
It does not own resource creation, infer dependencies, track buffers, cull
passes, alias memory, allocate transient resources, or schedule async compute.

## Scene, Mesh, Material, And Draw Data

Current renderer-side scene data is deliberately simple:

- `Camera` stores position, target, up vector, FOV, and near/far planes.
- `Transform` stores position/rotation/scale or a matrix override.
- `RenderObject` stores a transient debug ID, mesh/material pointers, optional
  material table pointer, source type, transform, debug name, and an animation
  flag.
- `Mesh` owns GPU-local vertex/index buffers, submesh primitive ranges, local
  bounds, and a debug name.
- `Material` stores texture pointers, descriptor set, bindless indices, PBR
  scalar factors, source type, and fallback flags.
- `DrawItem` is private to `Renderer` and maps object/submesh/material data into
  direct or indirect draw commands.
- `MeshDrawBatch` groups contiguous draw items by mesh for indirect batch
  submission.

There is no ECS, persistent UUID, object visibility flag, editable transform
workflow, asset registry, or scene serialization yet. Imported glTF objects use
`Transform::fromMatrix()`, so editing them later requires either matrix
decomposition or preserving a separate local transform representation.

## glTF Loading Path

`Mesh::createFromGltf()` uses tinygltf with a custom image callback that captures
encoded image bytes instead of letting tinygltf decode images. It supports ASCII
`.gltf` and binary `.glb`, triangle primitives, positions, normals, UV0,
tangents, indices, node hierarchy traversal, and node world transforms.

`Renderer::createScene()` tries:

1. `assets/models/test_mesh.gltf`
2. `assets/models/test_mesh.glb`
3. Built-in cube fallback scene

The renderer converts loaded texture/material info into internal `VulkanTexture`
and `Material` objects. glTF base color textures are uploaded as sRGB; normal and
metallic-roughness textures are linear. Missing data falls back to procedural or
neutral textures. Unsupported glTF areas include animation, skinning, morph
targets, cameras, lights, alpha modes, occlusion textures, emissive textures,
and generated tangents.

## ImGui Debug UI

`ImGuiLayer` owns the Dear ImGui context, SDL3 backend, Vulkan backend,
descriptor pool, and cached preview descriptors. It uses the ImGui Vulkan backend
with Dynamic Rendering and recreates backend state when swapchain format or image
count changes.

`Renderer::buildDebugUi()` currently exposes:

- Runtime settings save/load/reset.
- Tone mapping and exposure controls.
- Bloom controls.
- CSM controls.
- GPU culling toggles and status.
- Environment status.
- Render graph table.
- GPU timing history table.
- Culling and exposure plots.
- Read-only scene hierarchy.
- Read-only material inspector.
- Texture debug views.
- Render target and CSM cascade debug views.

This is a debug UI, not an editor. There is no docking layout, transform editing,
scene serialization, asset browser, material editing, or ImGuizmo integration.

## GPU Culling

The renderer builds CPU `DrawItem` records every frame. The CPU path tests
object AABBs against the camera frustum and records indirect commands for visible
draw items.

When GPU culling is active, `cull.comp` reads one `GpuCullDrawItem` per draw
item, tests its world-space AABB against pushed frustum planes, writes compacted
or zeroed `VkDrawIndexedIndirectCommand` records, and increments per-batch
visible counters. The same compute shader is reused for shadow-caster culling by
pushing cascade light-frustum planes.

The current system is GPU frustum culling, not occlusion culling. Draw batches
are still CPU-built, bounds are AABB-based, and the implementation is capped by
the current fixed `kMaxDrawItems`/`kMaxFrameObjects` limits.

## Post Processing And Exposure

The main scene renders to `sceneColor_` as `VK_FORMAT_R16G16B16A16_SFLOAT`.
Current post-processing is:

- Half-resolution bloom extraction from HDR scene color.
- Separable horizontal and vertical blur through bloom ping/pong images.
- Log-average luminance compute pass with per-frame readback.
- Histogram exposure compute pass with per-frame readback.
- Composite pass that applies bloom intensity, exposure, and Reinhard or ACES
  tone mapping before swapchain output.

Exposure readback is intentionally delayed until the corresponding frame fence
has completed. There is no GPU-only exposure reduction, mip-chain bloom, TAA,
motion vector buffer, history color, or HDR swapchain output yet.

## CSM Shadows

The shadow system uses a depth 2D array owned by `VulkanShadowMap`. Each active
cascade renders one array layer with depth-only Dynamic Rendering. The renderer
computes practical split depths, fits a light-space orthographic projection to
each camera frustum slice, optionally snaps the light-view center to the shadow
texel grid, and stores cascade matrices/splits in `ObjectFrameData`.

The main shaders sample the shadow array as `sampler2DArray`, choose a cascade
from camera-view depth, and use manual depth comparisons with optional PCF. The
current implementation does not include cascade blending, stable crop matrices,
per-cascade resolution controls, alpha-tested casters, VSM/EVSM, or occlusion.

## Existing Profiling

The project already has a useful profiler foundation:

- `VulkanTimestampQuery` creates timestamp queries per frame slot.
- It reads `timestampPeriod` and queue-family `timestampValidBits`.
- It disables itself gracefully when timestamp support or
  `vkCmdWriteTimestamp2` is unavailable.
- `Renderer` records timestamps for Shadow/CSM, Main, Skybox, RenderObjects,
  Bloom, AutoExposure/Luminance, HistogramExposure, and Composite.
- Debug labels already wrap major passes and nested draw regions.
- ImGui already displays current, average, max, and history plots for timing
  ranges plus CPU frame time.

Phase 1 should therefore evolve this into a named `GpuProfiler` or similar
system and fill gaps instead of creating a parallel profiler. Current gaps are
per-pass scoped naming, separate bloom subpass timings, GPU culling timings,
ImGui pass timing, clearer total frame accounting, and a visible unavailable
state in the UI.

## Milestone History

The README contains milestone documentation from early triangle rendering
through Milestone 48. Recent history shows the renderer has progressed through
histogram exposure, ImGui debug UI, render graph/timing visualization, persistent
runtime settings, scene hierarchy inspection, material/texture debug views, and
render-target/CSM cascade debug views.

Recent git history also matches that progression:

- Render target debug views and CSM cascade visualization.
- Read-only material inspector and texture debug views.
- ImGui scene hierarchy viewer.
- Persistent runtime settings serialization.
- Render graph visualization and GPU timing graphs.
- ImGui debug UI overlay.
- Histogram exposure and post-processing milestones.

## Risk Areas For Later Phases

### Synchronization And Layout Tracking

The render graph manually tracks image layouts by mutating layout variables
owned by `Renderer`, `VulkanSwapchain`, and `VulkanShadowMap`. RenderGraph 2.0
must preserve those contracts while expanding to logical resource handles. The
riskiest transitions are:

- Shadow map depth attachment to depth read-only.
- Scene color color attachment to shader read.
- Bloom ping/pong color attachment to shader read across sequential passes.
- Swapchain undefined/color attachment/present layout transitions.
- Main depth reuse across frames after swapchain recreation.

Buffer synchronization is still outside the graph. GPU culling, shadow culling,
luminance, and histogram passes all use explicit Synchronization2 buffer
barriers before indirect draws, copies, or host reads. Moving these passes into a
new graph without buffer access modeling would be a regression.

### Descriptor Lifetime

Material descriptors point at renderer-owned textures, shadow maps, IBL maps,
and the BRDF LUT. Bindless indices are assigned by pointer identity through
`BindlessTextureHeap`. ImGui preview descriptors are cached separately and must
be invalidated when texture/render-target views are destroyed or recreated.

Future asset editing or hot reload must avoid invalidating:

- `RenderObject::mesh` and `RenderObject::material`.
- `RenderObject::materialTable`.
- Texture pointers stored in `Material`.
- Bindless texture registrations keyed by texture address.
- ImGui preview descriptor cache entries.

An asset manager should introduce stable handles before vectors of meshes,
textures, or materials are resized/replaced at runtime.

### Swapchain Recreation

`Renderer::recreateSwapchain()` currently uses `vkDeviceWaitIdle`, recreates the
swapchain, per-image render-finished semaphores, ImGui backend state, and all
post-process resources/descriptors. It also resets post-process image layouts and
conditionally recreates pipelines when formats change.

Any future transient resources, render graph resources, profiler resources tied
to image count, or ImGui preview descriptors must participate in this path. The
timestamp query pool is frame-slot based today and is not swapchain-dependent.

### Resource Destruction

Most RHI wrappers destroy Vulkan resources immediately in `reset()`/destructors.
This is safe during startup, shutdown, and current swapchain recreation because
the renderer waits idle. It is not safe for hot reload or editor-side asset
replacement unless deferred destruction or frame-fence retirement is added.

Texture, environment, BRDF LUT, and buffer uploads currently use one-time command
buffers followed by `vkQueueWaitIdle`. That is acceptable for load-time paths but
should not become a runtime editor hot-reload pattern without a staging/upload
queue lifetime plan.

### GPU Culling Counters

Main culling counters use per-frame visible-count readback buffers. Shadow
culling reuses the same shadow visible-count and indirect buffers for each
cascade as each cascade is culled and drawn. This keeps rendering coherent, but
aggregate shadow readback/debug accounting is fragile because the readback buffer
contains the last copied cascade data rather than an explicit per-cascade
history. Phase 5 should introduce explicit visibility classification buffers if
the UI needs per-object or per-cascade visibility.

### CPU/GPU Struct Contracts

`ObjectFrameData`, `PushConstants`, `GpuCullDrawItem`, and the shader-side buffer
reference/storage-buffer structs are tightly coupled by std430 layout and static
asserts. Any change to material data, culling data, TAA matrices, motion vectors,
or light buffers must update C++ and GLSL together.

### Scene Editing

Built-in objects use decomposed transforms. Imported glTF objects currently use a
matrix override, which is accurate for static rendering but awkward for editable
position/rotation/scale. Phase 2 should either decompose imported transforms at
load time or make the inspector explicitly handle matrix-backed transforms.

## Recommended Implementation Plan

1. Phase 1 should wrap or extend `VulkanTimestampQuery` into a named profiler
   API, add scoped pass records, split current coarse timers where useful, and
   keep the existing non-blocking frame-latency readback model.
2. Phase 2 should add stable runtime scene IDs, object visibility flags, and
   transform/camera/light editing without changing the draw pipeline. JSON
   serialization should initially cover what can be restored reliably.
3. Phase 3 should introduce stable asset handles before enabling material hot
   reload or editor-side material saves, so current raw pointers and bindless
   registrations do not become invalid.
4. Phase 4 should evolve the manual render graph incrementally. Start with
   logical handles and debug metadata, then move image transitions, then add
   buffer access tracking for culling and exposure before pass culling.
5. Phase 5 should add Hi-Z and occlusion as a conservative extension of the
   existing GPU culling path, with CPU fallback and explicit debug counters.
6. Phase 6 should keep the existing bloom and exposure paths as fallback while
   adding mip-chain bloom and any GPU-only exposure/TAA foundations.
7. Phase 7 should document only implemented behavior and keep planned features
   separated from current capabilities.

## Phase 0 Result

This audit is documentation-only. It intentionally makes no renderer behavior,
build-system, shader, or asset changes.
