# Render Graph 2.0

Phase 4 upgrades the earlier manual pass list into a small engine-style render graph. The goal is to centralize pass/resource declarations and conservative image transitions without changing the renderer's visible output or rewriting unrelated renderer systems.

## Current Architecture

`Renderer` still owns the frame loop and records the Vulkan commands for each pass. `RenderGraph` now owns the logical frame description:

- logical texture and buffer handles
- imported and transient resource descriptions
- per-pass read/write declarations
- conservative image layout/access transition inference
- pass execution, side-effect, and culling metadata
- ImGui debug data for pass/resource visualization

The graph keeps the existing `begin*Pass` / `end*Pass` entry points so profiler scopes, debug labels, and command recording stay at the existing call sites.

## Resource Handles

Textures and buffers are referenced by frame-local handles:

- `RGTextureHandle`
- `RGBufferHandle`

Handles are lightweight indices that are stable for one `beginFrame` / `endFrame` build. Pass declarations use handles instead of exposing `VkImage` or `VkBuffer` directly. `RenderGraphContext` can resolve handles to Vulkan handles for future graph-executed pass callbacks, but the current renderer still records most commands directly.

## Imported And Transient Resources

Imported resources are owned outside the graph:

- swapchain color image
- main depth image
- cascaded shadow map array
- depth pyramid image
- per-frame luminance/histogram readback buffers

Transient graph resources are logical frame/render-target resources:

- `SceneColor`
- `BloomExtract`
- `BloomPing`
- `BloomPong`

In Phase 4, these transient resources are described and transitioned by the graph, but their physical `VulkanImage` allocation still lives in `Renderer` and is recreated with the swapchain/post-process resize path. This avoids a resource ownership rewrite while preparing for a future graph allocator or pool.

## Pass Declarations

Pass declarations are recorded with `RenderGraphBuilder`:

- `readTexture`
- `writeTexture`
- `readWriteTexture`
- `readBuffer`
- `writeBuffer`
- `readWriteBuffer`
- `sideEffect`

The currently declared passes are:

- `CSMShadowPass`
- `MainGpuCullingPass`
- `MainHDRPass`
- `DepthPyramidPass`
- `BloomExtractPass`
- `BloomBlurHorizontal`
- `BloomBlurVertical`
- `LuminancePass`
- `HistogramExposurePass`
- `CompositePass`
- `ImGuiPass`

Shadow GPU culling buffer barriers, main GPU culling buffer barriers, luminance buffer copy/readback, histogram buffer copy/readback, and portfolio screenshot copy are still manually synchronized in `Renderer.cpp`.

## Transition Inference

Texture declarations use `RGAccess` values such as:

- `ColorAttachmentWrite`
- `DepthStencilAttachmentWrite`
- `ShaderRead`
- `StorageImageReadWrite`
- `TransferSrc`
- `TransferDst`
- `Present`

At pass begin, the graph maps the declared access to a conservative Synchronization2 image barrier. It tracks the current layout and previous access for graph-managed/imported textures and updates the owning layout state for swapchain, depth, shadow, scene color, and bloom images.

The implementation favors correctness and debug readability over barrier minimization. Same-layout write-after-write or write-after-read cases can still emit ordering barriers.

## Side Effects And Culling

Passes can be marked as side-effecting. Side-effect passes are never culled. Current side-effect passes include:

- `MainGpuCullingPass`
- `DepthPyramidPass`
- `LuminancePass`
- `HistogramExposurePass`
- `CompositePass`
- `ImGuiPass`

The graph computes basic pass liveness from declared reads/writes. A pass with no side effects and unused outputs is marked culled. Current renderer-visible passes remain live because their outputs feed later passes or external readback/present behavior.

## Debug UI

The ImGui Render Graph panel shows:

- declared pass order
- execution status
- culled status/reason
- side-effect flag
- read and write resources with declared access
- generated image barrier count/summary
- resource name, kind, lifetime, extent/size, format, mip/layer count, initial layout, and final layout

## Known Limitations

- No async compute scheduling.
- No multi-queue scheduling.
- No memory aliasing.
- No resource pooling overhaul.
- Transient scene/bloom resources are graph-described but still physically allocated by `Renderer`.
- GPU culling and readback buffer barriers remain manual.
- Portfolio screenshot copy remains manual because it temporarily transitions the swapchain between `CompositePass` and `ImGuiPass`.
- Barriers are conservative and not heavily optimized.
- Not all descriptor-driven sampled resources are graph-owned yet, including material textures, IBL cubemaps, BRDF LUT, and render-target preview descriptors.
- There is no node-editor view yet.
