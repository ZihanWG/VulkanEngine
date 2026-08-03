# Render Graph 2.0

Phase 4 upgraded the earlier manual pass list into a small engine-style render graph. The current goal is to centralize pass/resource declarations plus conservative image and selected buffer synchronization without changing the renderer's visible output or rewriting unrelated renderer systems.

## Current Architecture

`Renderer` still owns the frame loop and records the Vulkan commands for each pass. `RenderGraph` now owns the logical frame description:

- logical texture and buffer handles
- imported and transient resource descriptions
- per-pass read/write declarations
- conservative image layout/access transition inference
- conservative buffer access barrier inference for selected declared pass-to-pass dependencies
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
- punctual (spot/point) shadow atlas, when the atlas allocated successfully
- TAA history read/write images
- depth pyramid image
- per-frame luminance, histogram, exposure, and culling buffers

Transient graph resources are logical frame/render-target resources:

- `SceneColor`
- `BloomExtract`
- `BloomPing`
- `BloomPong`
- `BloomMipDownsampleN`
- `BloomMipUpsampleN`

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
- `PunctualShadowAtlasPass` when at least one punctual light was assigned an atlas tile
- `VolumetricFogPass` when fog is enabled; it declares only its shadow-map read, since its output volumes are not graph resources
- `MainGpuCullingPass`
- `MainHDRPass`
- `DepthPyramidPass`
- `TAAResolvePass` when TAA is enabled
- `BloomExtractPass`
- `BloomBlurHorizontal`
- `BloomBlurVertical`
- `BloomDownsampleMipN`
- `BloomUpsampleMipN`
- `LuminancePass`
- `HistogramExposurePass`
- `CompositePass`
- `ImGuiPass`

The graph now owns selected declared buffer pass-to-pass barriers:

- `MainGpuCullingPass` storage writes to the main indirect command buffer and visible-count buffer before `MainHDRPass` indirect reads
- `LuminancePass` storage writes to luminance partials before `HistogramExposurePass` reads them
- `HistogramExposurePass` exposure-state writes before `CompositePass` reads the GPU exposure buffer

Shadow GPU culling barriers, main GPU culling reset/copy/readback barriers, histogram reset and in-pass histogram-to-exposure-reduce barriers, exposure debug host-read visibility, and portfolio screenshot copy barriers are still manually synchronized in `RendererRecord.cpp`.

## Synchronization Inference

Texture declarations use `RGAccess` values such as:

- `ColorAttachmentWrite`
- `DepthStencilAttachmentWrite`
- `ShaderRead`
- `StorageImageReadWrite`
- `TransferSrc`
- `TransferDst`
- `Present`

At pass begin, the graph maps the declared access to a conservative Synchronization2 image barrier. It tracks the current layout and previous access for graph-managed/imported textures and updates the owning layout state for swapchain, depth, shadow, scene color, and bloom images.

Buffer declarations use existing buffer-oriented `RGAccess` values such as:

- `StorageBufferRead`
- `StorageBufferWrite`
- `StorageBufferReadWrite`
- `IndirectRead`
- `TransferSrc`
- `TransferDst`
- `HostRead`

At pass begin, the graph maps buffer accesses to conservative Synchronization2 buffer barriers. It tracks the previous stage/access for each imported buffer during the frame and emits full-buffer barriers when a declared pass-to-pass dependency includes a write-after-read, read-after-write, or write-after-write hazard. Read-after-read usage does not emit a barrier.

The implementation favors correctness and debug readability over barrier minimization. Same-layout write-after-write or write-after-read cases can still emit ordering barriers.

## Side Effects And Culling

Passes can be marked as side-effecting. Side-effect passes are never culled. Current side-effect passes include:

- `MainGpuCullingPass`
- `DepthPyramidPass`
- `LuminancePass`
- `HistogramExposurePass`
- `CompositePass`
- `ImGuiPass`

The graph computes basic pass liveness from declared reads/writes. A pass with no side effects and unused outputs is marked culled. Current renderer-visible passes remain live because their outputs feed later passes or external readback/present behavior. `TAAResolvePass` is not side-effecting, but it stays live when enabled because bloom, exposure, and composite read the resolved history image as the active HDR source.

## Debug UI

The ImGui Render Graph panel shows:

- declared pass order
- execution status
- culled status/reason
- side-effect flag
- read and write resources with declared access
- generated image and buffer barrier count/summary
- resource name, kind, lifetime, extent/size, format, mip/layer count, initial layout, and final layout

## Known Limitations

- No async compute scheduling.
- No multi-queue scheduling.
- No memory aliasing.
- No resource pooling overhaul.
- Transient scene/bloom resources and persistent TAA history resources are graph-described but still physically allocated by `Renderer`.
- Shadow GPU culling buffers are not graph-declared yet, so their reset/dispatch/draw/readback barriers remain manual.
- Intra-pass buffer sequencing remains manual when a buffer is filled, dispatched against, copied, or made host-visible inside one renderer command block.
- Portfolio screenshot copy remains manual because it temporarily transitions the swapchain between `CompositePass` and `ImGuiPass`.
- Barriers are conservative and not heavily optimized.
- Not all descriptor-driven sampled resources are graph-owned yet, including material textures, IBL cubemaps, BRDF LUT, and render-target preview descriptors.
- There is no node-editor view yet.
