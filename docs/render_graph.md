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

### Barrier batching

Every barrier a pass infers sits between the same two points in the command
stream -- after whatever the previous pass recorded, before this pass's first
command -- so they are accumulated and submitted as a single
`vkCmdPipelineBarrier2` rather than one call per resource. This changes how many
barrier commands carry the frame's synchronization, not what is synchronized:
the inferred barrier set is byte-for-byte the same.

The one case that cannot be batched is a resource transitioned twice inside a
single pass. Two barriers for the same resource in one `VkDependencyInfo` are
unordered with respect to each other, so the accumulated set is submitted before
the second transition is recorded (`barrierBatchNeedsFlush`, unit tested).
Tracked state is still updated at record time, not at flush time, so the second
transition derives its barrier from what the first one leaves behind exactly as
it did when every transition submitted on its own.

No declared pass currently declares the same resource twice, so that flush path
is dormant today. `RenderPassNode::generatedBarrierSubmitCount` above one is the
signal that a pass has started doing it -- with one standing exception:
`ImGuiPass` always reports two, because its transition of the swapchain into the
present layout is recorded after the pass body and cannot join the pass's own
set.

Measured on the default scene at 1280x720 with default settings, on lavapipe
under the headless-render job's flags:

| | Per frame |
| --- | --- |
| executed passes | 23 |
| inferred barriers | 57 |
| `vkCmdPipelineBarrier2` calls, before | 57 |
| `vkCmdPipelineBarrier2` calls, after | **22** |

The barrier count is identical on both sides, which is the property that matters:
the batching moved 57 barriers from 57 commands into 22, and added none.

This is a command-count reduction, **not a measured frame-time win** -- no
timing claim is attached to it. It is also the prerequisite for tightening the
alias-handoff source scope, since that only pays off once a pass's barriers
share one dependency.

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
- generated image and buffer barrier count/summary, and the number of
  `vkCmdPipelineBarrier2` submissions they were batched into
- resource name, kind, lifetime, extent/size, format, mip/layer count, initial layout, and final layout

## Transient Memory Aliasing

Resources whose lifetimes do not overlap can share bytes in one allocation.
`renderer::planTransientMemory` (GPU-free, unit tested) assigns offsets by greedy
interval packing, `computeTextureLifetimes` supplies the intervals from the
declared passes after culling, and `rhi::VulkanTransientMemoryPool` owns the
allocation that `VulkanImage::createAliased` binds images into.

Measured at 2560x1440: the whole transient set is 123.74 MiB and would pack into
82.62 MiB. Only the bloom chain is wired -- 41.12 MiB of images into a 23.64 MiB
pool, **17.48 MiB saved**.

### What it costs, and why it stays off

Measured A/B, interleaved with a repeated control (drift 0.48%, limit 1%):

| | Frame total |
| --- | --- |
| aliasing off | 15.236 ms |
| aliasing on | 15.412 ms |
| delta | **+0.176 ms (+1.2%)** |

**Quote the frame total and nothing else from that series.** Several per-pass
rows clear their own noise floor and are still not believable: `PunctualShadowAtlas`
and `ImGuiPass` moved, and aliasing the bloom chain cannot plausibly touch either.
`Bloom Downsample Chain` moved -0.050 ms while the frame moved +0.176 ms, and two
passes moving opposite ways on this tile-based target means work shifted across a
pass boundary rather than appearing or vanishing. The decomposition is not
trustworthy here; the frame total is.

So the trade is 17.48 MiB for 1.2% of frame time, on a machine reporting a
13 GiB device-local budget. That is a bad trade, and it is why
`enableTransientAliasing` **defaults off** and why the remaining transients were
not wired up: more aliased resources means more handoffs, so the cost would grow
while the memory it buys stays unscarce.

The conservative source scope on the handoff barrier is the prime suspect -- it
waits on `ALL_COMMANDS`/`MEMORY` rather than the union of the predecessors that
actually own the overlapping bytes. Tightening it is the only change that could
plausibly move this verdict, and it should be measured the same way before
anything else is aliased.

Scope is memory only. Every subsystem still creates, owns, views, and describes
its own images; just the backing memory moves. The pool declines rather than
fails when no memory type accepts every image, and a single failed binding demotes
the whole chain back to private allocations rather than leaving it half aliased.

### Proving the memory is actually shared

`vmaCreateAliasingImage2` returning `VK_SUCCESS` only establishes that the driver
accepted the binding. A driver that quietly gave each image private memory would
look identical, until a transient resource silently stopped being overwritten.

`--probe-aliasing` settles it: it clears one alias and reads the other back
through a second image, using the same barrier shape the allocator emits at every
handoff. On MoltenVK the write is observed and validation stays clean.

The evidence is asymmetric on purpose. A positive proves sharing; a negative does
not disprove it, because reading an alias requires transitioning it from
`VK_IMAGE_LAYOUT_UNDEFINED`, which the spec licenses to discard contents. The
probe reports that case as "not observable" rather than as a failure.

### The alias-handoff barrier

The first use of a pool-bound resource in a frame inherits bytes another resource
owned earlier, so two things hold that do not for a private image: its contents
are genuinely undefined (hence `VK_IMAGE_LAYOUT_UNDEFINED` as the old layout,
whatever layout it was left in last frame), and the barrier must wait for
whatever wrote those bytes, which its own `lastAccess` does not track.

The source scope is deliberately conservative -- `ALL_COMMANDS` /
`MEMORY_READ|WRITE` -- rather than the exact union of overlapping predecessors.
Tracking predecessors would mean threading the memory plan through the barrier
path, this graph's barriers are already documented as conservative, and the cost
is a measurement question. Tightening it is measurement-driven, not a correctness
fix.

### Why it is off by default, and a known wart

The plan needs resource lifetimes, which only exist once a frame has been
recorded, so the pool is applied *after the first frame* rather than at
resource-creation time. Applying it recreates the post-process resources, which
resets the auto-exposure accumulator.

The consequence is measurable and worth stating plainly. With aliasing on, the
rendered HDR content is provably identical -- average scene luminance and
histogram-clipped luminance match the non-aliased run to four decimal places --
but the exposure accumulator converges from a different starting point. At frame
60 that shows up as 68% of pixels differing by exactly one LSB; by frame 400 it
is 2%, still one LSB, and shrinking. It is a startup transient, not corruption:
two aliased runs are byte-identical to each other.

Removing it means applying the plan before the first frame, which needs a cached
plan from a previous run rather than a measured one. Until then the setting stays
off by default, and the golden-image job is unaffected because it renders with
the default.

## Known Limitations

- The graph does not schedule across queues. Async compute exists in the engine --
  `ClusterBuild` and `LightCull` run on a dedicated compute queue overlapping the
  shadow passes (see [async_compute.md](async_compute.md)) -- but the renderer
  owns that submission and its semaphores, not the graph. The graph records those
  passes as ordinary compute nodes.
- Memory aliasing is implemented for the bloom chain only, and is **off by
  default** (`enableTransientAliasing`). See "Transient memory aliasing" below.
- No resource pooling overhaul.
- Transient scene/bloom resources and persistent TAA history resources are graph-described but still physically allocated by `Renderer`.
- Shadow GPU culling buffers are not graph-declared yet, so their reset/dispatch/draw/readback barriers remain manual.
- Intra-pass buffer sequencing remains manual when a buffer is filled, dispatched against, copied, or made host-visible inside one renderer command block.
- Portfolio screenshot copy remains manual because it temporarily transitions the swapchain between `CompositePass` and `ImGuiPass`.
- Barriers are conservative and not heavily optimized. They are batched per
  pass (see "Barrier batching"), but their stage/access scopes are unchanged.
- Not all descriptor-driven sampled resources are graph-owned yet, including material textures, IBL cubemaps, BRDF LUT, and render-target preview descriptors.
- There is no node-editor view yet.
