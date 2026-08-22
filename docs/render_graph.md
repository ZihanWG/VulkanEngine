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

## Resource Versions

A handle names a resource **and which of its versions**. Every declared write
produces a new version and the builder hands back a handle for it, so a
declaration threads the result forward:

```cpp
frame_.sceneColor = builder.writeTexture(frame_.sceneColor, RGAccess::ColorAttachmentWrite, "...");
```

Barriers and resource lookup ignore the version -- every version of a resource
is the same `VkImage`. What it buys is that a reader names the version it
consumes, so the declared data flow no longer depends on where a pass happens to
sit in the file. That is what lets `validatePassOrder` judge an order other than
the recorded one, and it makes holding a handle from before an intervening write
a detectable mistake (`StaleVersionRead`) instead of an invisible one.

This frame's version counts, at 1280x720 with the default settings: `SceneColor`
reaches 4 (main, phase 2, SSR trace, transparent), `VelocityBuffer` and
`NormalRoughnessGBuffer` 3, `SwapchainColor`, `MainDepth` and `DepthPyramidHiZ` 2.

### What versioning found

The post-process source was snapshotted into `FrameState` *before any pass had
declared anything*, and five passes read it: bloom extract, the first mip-chain
downsample, luminance, histogram exposure, and composite. With versions turned
on, all five report `StaleVersionRead` -- they name version 0 of a target that
four writes later is at version 4.

The consequence is not cosmetic. With those stale handles the derived dependency
graph puts the entire post-process chain at a longest chain of **9** instead of
**17**, because it believes bloom, luminance and composite depend on the scene
colour *before the main pass wrote it*. A scheduler acting on that graph would
hoist the whole post-process chain above the main pass.

The fix is to resolve the source at the point of use rather than snapshot it:
`RenderGraph::postProcessSource()` returns the TAA resolve target or the scene
colour, reading whichever member currently holds the latest version. Both the
defect and the fix were checked on a real frame: restoring the snapshot behaviour
reports the five findings and drops the chain to 9; the fix reports none.

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

Everything else is still synchronized by hand, in the subsystem that owns the
resource rather than in one place. `RendererRecord.cpp` itself no longer records
a single barrier; the remaining manual sites are:

| File | What it synchronizes by hand |
| --- | --- |
| `GpuCulling.cpp` | Main cull reset/dispatch/draw, phase-2 count reset, visible-count copy and readback, and the shadow cull equivalents |
| `PunctualShadows.cpp` | Punctual cull dispatch and its indirect/count consumers |
| `VirtualShadowMapPass.cpp` | Page-marking and page-cull request buffers, which are not graph resources |
| `PostProcessStack.cpp` | Histogram reset, the in-pass histogram-to-exposure reduce, and exposure host-read visibility |
| `IrradianceProbeVolume.cpp` | Probe shading-param upload and the atlas update dispatches |
| `DepthPyramid.cpp` | The mip chain's write-to-read sequencing inside its own pass |
| `ClusteredLighting.cpp` | Cluster build to light cull, on the async compute queue |
| `VolumetricFogPass.cpp` | Injection to scattering, between its own volumes |
| `ScreenshotCapture.cpp` | Portfolio screenshot copy, which transitions the swapchain between `CompositePass` and `ImGuiPass` |

Two of those categories are deliberate rather than pending: barriers
for intra-pass sequencing (the depth pyramid's mip chain, fog's volumes, the
histogram reduce) and for readback/host visibility stay explicit even for
resources the graph does declare.

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

## Declaration Validation

The declarations describe what each pass touches. Nothing used to check that the
description was self-consistent, so a pass could read a transient nothing had
produced and the only symptom would be wrong pixels -- no validation error, no
crash, nothing in a log. `validateDeclarations` (GPU-free, unit tested) runs
every frame after culling and checks the three ways a declaration set can be
wrong without looking at a device:

| Issue | Meaning |
| --- | --- |
| `ReadsContentNoPassProduced` | A pass reads a graph-managed resource no earlier surviving pass wrote this frame, without saying the read is deliberate. Either the pass is ordered wrongly, or it means to read the previous frame and should declare that. |
| `HistoryReadOfAliasedResource` | A previous-frame read of a pool-bound resource. Aliasing hands those bytes to another resource and the handoff barrier discards the contents, so there is no previous frame left to read. |
| `ResourceDeclaredTwice` | One pass declares the same resource more than once. Usually a copy-paste slip, and it costs a barrier submission, since two barriers for one resource cannot share a dependency info. |
| `HistoryReadAfterProducer` | A previous-frame read placed after a pass that writes the resource this frame. The declaration says "last frame" and the schedule hands it this frame's data. Unlike the two rules above, this one applies to imported resources too: what is wrong is the ordering, not who owns the memory. |
| `StaleVersionRead` | A read naming an older version than the one current at that point: the handle was taken before an intervening write and never refreshed. Every version is the same image, so the pixels are right; what is wrong is that the declared data flow points at the wrong producer. |

Imported resources are exempt from the first two: their contents persist by
contract, which is what importing them means.

The check reports, it does not throw. Every rule here has a legitimate-looking
shape that only the author can adjudicate, and a graph that refuses to render is
a worse diagnostic than one that renders and says what looks wrong. Findings
appear in the ImGui Render Graph panel, on the pass that declared them, and are
queryable through `RenderGraph::declarationIssues()`.

### Declaring a previous-frame read

`RenderGraphBuilder::readHistoryTexture` declares a read of what a resource held
at the end of the previous frame. It is identical to `readTexture` in barrier
terms -- the layout and scopes do not care where the pixels came from -- and
exists so that reading before anything wrote the resource reads as intent rather
than as an ordering mistake. The frame's history reads are:

- `MainHDRPass` and `MainHDRPhase2` sampling ambient occlusion, which GTAO only
  writes later in the frame
- `VsmPageMarkPass` and `MainGpuCullingPass` sampling the Hi-Z depth pyramid
- `TAAResolvePass` sampling the previous HDR history image
- `TransparentPass` and `CompositePass` sampling ambient occlusion **when GTAO is
  off**, because then nothing produces that image at all; with GTAO on the same
  reads are ordinary same-frame dependencies on the blur, and are declared as such

That last split is the point of the whole exercise: the declaration now states
which frame the data comes from, and it changes with the setting that actually
changes the answer.

The default configuration reports zero issues. That was checked both ways --
injecting an undeclared history read and a duplicate declaration into
`MainHDRPass` makes the panel report three findings on that pass, and turning
`CompositePass`'s bloom read into a history read reports the fourth rule.
Removing the injections returns it to zero.

### What it does not check

Liveness across frames. A history read is only satisfied if *this* frame's
producer runs, and culling does not know that: a pass whose output is read only
by the next frame's history read can still be culled. Nothing in the current
frame graph is in that shape, and fixing it means teaching `cullUnusedPasses`
about cross-frame edges, which is a larger change than this check.

## The Derived Dependency Graph

`computePassSchedule` (GPU-free, unit tested) derives, from the declarations
alone, which passes each pass must follow. The edges are the three orderings a
resource forces: read-after-write, write-after-read, and write-after-write. A
history read makes no read-after-write edge -- it consumes the previous frame,
not this frame's producer -- which is the one place that distinction changes the
graph rather than only the diagnostics.

### Judging an order

Read-after-write edges resolve through the version a handle names, not through
whichever write happens to be most recent, so the edge set is a property of the
declarations rather than of where a pass sits. `validatePassOrder` takes a
proposed execution order and reports the edges it breaks -- and it does reject
orders, which is the point of the versioning work.

Checked on a real frame at 1280x720:

| Proposed order | Violations |
| --- | --- |
| the recorded order | 0 |
| exposure pair hoisted 13 slots, to just after `TransparentPass` | **0** |
| the same hoist one step further, above `TransparentPass` | **2** |

So moving `LuminancePass` and `HistogramExposurePass` out from behind the bloom
chain is provably legal, and moving them one step too far is provably not. The
recording order itself has not changed; what exists now is the means to check a
change before making it.

### Slack

How tightly the declared data flow pins the frame down. Measured on the default
scene at 1280x720:

| | |
| --- | --- |
| recorded passes | 25 |
| longest dependency chain | **17** |

So the frame is recorded 25 steps deep but the declarations only force 17. Where
that headroom sits:

| Pass | Recorded slot | Earliest legal | Slack |
| --- | --- | --- | --- |
| `LuminancePass` | 21 | 8 | **13** |
| `HistogramExposurePass` | 22 | 9 | **13** |
| `CompositePass` | 23 | 15 | 8 |
| `BloomDownsampleMip0`..`BloomUpsampleMip0` | 14-20 | 8-14 | 6 |
| `DepthPyramidPass` | 10 | 5 | 5 |
| `MainHDRPass`..`TransparentPass` | 3-9 | 1-7 | 2 |

The auto-exposure pair is the clearest: both depend only on `TransparentPass`,
and both are recorded after the entire bloom chain they have nothing to do with.
The bloom chain and the exposure pair are independent branches off the same
producer, and neither constrains the other.

`DepthPyramidPass` is the same shape at smaller scale -- it needs only the
phase-2 main pass, but is recorded after the transparent and screen-space work.

### What the analysis surfaced

`CompositePass` reads `bloomPong` *and* the mip-chain output unconditionally, so
both bloom chains -- the legacy blur and the downsample/upsample chain -- are
declared live and run every frame, ten of the twenty-five passes between them.
Culling cannot drop either while the composite declaration reads both. Whether
the shader needs both bound is a renderer question this document does not settle;
the point is that the schedule made it visible.

## The Graph Drives The Post-Process Tail

`computeExecutionOrder` turns the derived dependency graph into an order: Kahn's
algorithm over the predecessor sets, breaking ties on declaration index. The tie
break is the point. Every derived edge points backwards by construction, so the
result is exactly the order the passes were declared in — handing the order to
the graph changes nothing about what runs or when. What it changes is who
decides, and a unit test pins that the two still agree.

`Renderer::recordRenderCommands` hands the graph seven recorders through
`RenderGraph::recordScheduledUnits`:

| Unit | Anchor pass |
| --- | --- |
| `recordDepthPyramidCommands` | `DepthPyramidPass` |
| `recordTaaResolveCommands` | `TAAResolvePass` |
| `recordLegacyBloomCommands` | `BloomExtractPass` |
| `recordMipChainBloomCommands` | `BloomDownsampleMip0` |
| `recordLuminanceCommands` | `LuminancePass` |
| `recordHistogramCommands` | `HistogramExposurePass` |
| `recordCompositeCommands` | `CompositePass` |

A recorder spanning several declared passes anchors on the first of them; the
mip-chain recorder covers seven. This is the region already factored far enough
to be invoked rather than called in sequence, and it is also where all of the
frame's scheduling slack is. Everything before it — the shadow, culling, probe,
main and screen-space passes — is still recorded in place.

**No reordering happens.** The scheduled order is today's order, the golden image
is bit-identical, and the proven-legal exposure hoist stays unapplied.

### The backstop

The declarations in `buildFrameGraphDeclarations` and the recording spread across
nine translation units are two sequences kept in step by hand. Nothing compared
them. `beginDeclaredPass` now records the order passes were actually begun in,
and `endFrame` checks it:

- `validatePassOrder` — dependencies the recording order broke;
- `unrecordedPasses` — scheduled passes never recorded at all. `validatePassOrder`
  sees a missing pass only through the edges it breaks, so one with no
  predecessors would otherwise pass unnoticed.

Both are reported in the Render Graph panel, never thrown, for the same reason
`validateDeclarations` reports.

### What the backstop found on its first run

Two passes were declared and scheduled every frame but never recorded, which
means the graph's model of the frame — its culling, its barriers, its resource
lifetimes — described work that did not happen:

- **`CSMShadowPass`.** The cascade cache skips the redraw when every cascade is
  clean (`if (!cascadeNeedsRedraw(cascadeIndex)) continue;`), but the declaration
  was unconditional. On a cached frame the main pass read a shadow map the graph
  believed had just been written.
- **`LuminancePass`.** `recordLuminanceCommands` returns early in histogram mode,
  which is the default, so `HistogramExposurePass` declared a read of partials
  nothing had produced.

Both now follow the pattern the punctual atlas and the VSM page pool already
used: the pass is declared only when the renderer will record it, while the
resource stays imported and its readers stay declared, so it keeps the layout its
sampler claims. The predicates are shared rather than duplicated —
`Renderer::anyCascadeShadowRedrawRequired` and
`PostProcessStack::willRecordLuminancePass` are called both by the declaration
and by the recorder, so the two cannot drift apart again. Collapsing the
luminance recorder's three early-outs into that one predicate is what closed it;
the first attempt covered only the first of them, and the backstop said so.

The default frame now reports 23 scheduled passes, 23 recorded, no violations.
Checked the other way too: recording the tail in reverse reports three broken
dependencies (`CompositePass` before `BloomBlurVertical`, `BloomUpsampleMip0`,
and `HistogramExposurePass`).

### The precondition for ever reordering

Culling, `validateDeclarations`, and `computeTextureLifetimes` all sweep
declaration order. They are correct today **only because the scheduled order
equals it**. A policy that actually reorders has to move those analyses onto the
scheduled order first, or they will describe a frame that does not happen.
Barriers do not have this problem: they are derived from live tracked state as
each pass is recorded, so they follow whatever order the recording takes.

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
- declaration issues found for the pass, and a frame-level count above the table
- each pass's derived predecessor count, earliest legal slot, and slack, with the
  predecessor names in a tooltip, plus the frame's longest dependency chain
- resource versions are not shown; they are visible through the pass declarations
- whether the recorded order matched the schedule, and any pass that was declared
  and scheduled but never recorded
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
- Declaration validation does not model cross-frame liveness; see "What it does
  not check".
- The graph orders the post-process tail and nothing else. The rest of the frame
  is recorded in place, and making it schedulable means factoring the main HDR
  pass (~300 inline lines), the shadow, probe and culling regions into callable
  units.
- No reordering is performed. The scheduled order equals the declaration order by
  construction; see "The precondition for ever reordering".
- Versions are per frame and per resource, not per subresource, so a pass writing
  one mip of an image advances the version of the whole image.
