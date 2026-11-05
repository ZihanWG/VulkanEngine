# Render Graph

The frame is a render graph. Every pass is declared once, with the resources it
reads and writes and the callback that records its commands; the graph culls
what nothing needs, checks the declarations against each other, orders the
passes, infers the barriers between them, and records each one through its
callback.

It is deliberately small. It runs on one queue, it does not reorder passes, its
barriers are conservative and whole-resource, and transient memory aliasing
exists but is off by default. Each of those is a measured or argued decision
rather than an omission, and "Known Limitations" says which.

## Overview

A frame runs in five steps:

1. `RenderGraph::beginFrame` imports the frame's resources -- the swapchain image,
   the shadow maps, the per-frame buffers, the scene-sized targets -- and puts
   their handles on the frame's blackboard.
2. The passes are declared. The renderer's are in `Renderer::declareFrameGraph`
   (`RendererGraph.cpp`): the virtual shadow map, cascade and punctual shadow
   passes, culling, fog, probes, the geometry region, the frame captures and the
   overlay. The post-process stack's are in `PostProcessStack::declareGraphPasses`:
   TAA, both bloom chains, exposure and the composite.
3. `compile` culls dead passes, validates the declarations and derives the
   execution order.
4. `recordPasses` records every scheduled pass: for each, it emits the barriers
   its declarations imply, opens its debug label and GPU profiler scope, and
   calls its callback.
5. `endFrame` checks that what was recorded matches what was declared, and
   closes the command buffer. Submission stays with `Renderer::drawFrame`.

Declaration order is the frame's order. Every edge the graph derives points
backwards, and the schedule breaks ties on declaration index, so the order it
computes is the order the passes were declared in -- which is what lets the
graph make the order its own decision without changing the frame.

## Resources

### Handles, versions and the blackboard

Passes name textures and buffers by `RGTextureHandle` and `RGBufferHandle`,
frame-local indices that stay valid from `beginFrame` to `endFrame`. Nothing in a
declaration sees a `VkImage` or `VkBuffer`; a callback resolves handles through
its `RenderGraphPassContext`.

A handle names a resource **and which of its versions**. Every declared write
produces a new version and the builder hands back a handle for it, so a
declaration threads the result forward:

```cpp
board.sceneColor = builder.writeTexture(board.sceneColor, RGAccess::ColorAttachmentWrite, "...");
```

`board` is the frame's `RenderGraphBlackboard`, from `RenderGraph::blackboard()`:
the handles `beginFrame` imported, which declarations read and write back.
Storing the returned handle there is what makes a later reader name that
version.

Barriers and resource lookup ignore the version -- every version of a resource
is the same `VkImage`. What it buys is that a reader names the version it
consumes, so the declared data flow does not depend on where a pass sits in the
file. That is what lets `validatePassOrder` judge an order other than the
recorded one, and it makes holding a handle from before an intervening write a
detectable mistake (`StaleVersionRead`) instead of an invisible one.

#### What versioning found

The post-process source used to be snapshotted *before any pass had declared
anything*, and five passes read it: bloom extract, the first mip-chain
downsample, luminance, histogram exposure, and composite. With versions turned
on, all five reported `StaleVersionRead` -- they named version 0 of a target that
four writes later was at version 4.

The consequence was not cosmetic. With those stale handles the derived
dependency graph put the post-process chain at a longest chain of **9** instead
of **17**, because it believed bloom, luminance and composite depended on the
scene colour *before the main pass wrote it*. A scheduler acting on that graph
would have hoisted the whole post-process chain above the main pass.

The source is resolved at the point of use instead:
`RenderGraphBlackboard::postProcessSource()` returns the TAA resolve target or
the scene colour, whichever currently holds the latest version. Both the defect
and the fix were checked on a real frame: restoring the snapshot reported the
five findings and dropped the chain to 9; the fix reports none.

### Imported and transient resources

Imported resources are owned outside the graph, and their contents persist by
contract:

- the swapchain colour image and the main depth image
- the cascaded shadow map array, the punctual shadow atlas and the virtual shadow
  map page pool, each when it exists
- the TAA history pair and the depth pyramid
- the froxel fog volumes and the irradiance probe atlases, including the capture
  atlas and its depth
- the ray-traced shadow mask's images and the top-level acceleration structure's
  buffer, on frames that trace
- the per-frame culling, luminance, histogram and exposure buffers

Transient resources are the frame's scene-sized and bloom-sized targets:
`SceneColor`, the velocity and normal-roughness G-buffer, the ambient occlusion
target and its half-resolution raw input, SSR's scene-colour copy, and every
image in both bloom chains. The graph describes them and tracks their state;
their physical images are still created by the subsystems that own them and
recreated on resize. "Transient Memory Aliasing" is how the graph can let some of
them share memory.

## Declaring a pass

```cpp
passes.taaResolve = graph.addPass(
    "TAAResolvePass", RenderPassType::TaaResolve, RenderPassExecutionType::Graphics, /*sideEffect=*/false,
    [&](RenderGraphBuilder& builder) {
        builder.readTexture(board.sceneColor, RGAccess::ShaderRead, "...");
        board.taaHistoryWrite = builder.writeTexture(board.taaHistoryWrite, RGAccess::ColorAttachmentWrite, "...");
        builder.colorAttachment(board.taaHistoryWrite, VK_ATTACHMENT_LOAD_OP_CLEAR, kOpaqueBlack);
    },
    {"TAAResolvePass", "TAAResolvePass", [this](RenderGraphPassContext& context) {
         recordTaaResolveCommands(context.commandBuffer());
     }});
```

The setup callback runs inside `addPass`, so it may capture locals by reference.
The executor -- a `RenderGraphPassExecutor` -- runs when the pass is recorded and
carries three things: the debug label and GPU profiler scope the pass is reported
under, and the callback.

What the builder declares:

| Call | Declares |
| --- | --- |
| `readTexture`, `readBuffer` | a read of what this frame produced |
| `readHistoryTexture` | a read of what the resource held at the end of the previous frame |
| `readTextureForLayout` | a read whose contents do not matter -- see "Layout-only reads" |
| `writeTexture`, `writeBuffer`, `readWriteTexture`, `readWriteBuffer` | a write, producing the next version |
| `sideEffect` | that the pass must run even if nothing reads what it writes |
| `colorAttachment` | the colour target the graph opens a rendering scope over around the callback |
| `transitionAfterPass` | a transition recorded behind the pass body: the swapchain's move to `PRESENT_SRC` after the overlay |
| `orderAfter` | that the pass follows an earlier one although no declared resource says so |

### Read kinds

What a read consumes. Barriers do not distinguish them -- a layout is a layout --
but the rest of the graph does:

| Kind | Consumes | Depends on the producer | Keeps the producer alive |
| --- | --- | --- | --- |
| `Produced` | what this frame's graph produced | yes | yes |
| `History` | what the previous frame left | no | yes |
| `LayoutOnly` | nothing | no | **no** |

#### Previous-frame reads

`readHistoryTexture` is identical to `readTexture` in barrier terms. It exists so
that reading before anything wrote the resource reads as intent rather than as
an ordering mistake, and so that culling keeps the pass the next frame needs
(see "Passes the next frame needs"). The frame's history reads are:

- `MainHDRPass` and `MainHDRPhase2` sampling ambient occlusion, which GTAO only
  writes later in the frame
- `VsmPageMarkPass` and `MainGpuCullingPass` sampling the Hi-Z depth pyramid
- `TAAResolvePass` sampling the previous HDR history image
- `TransparentPass` and `CompositePass` sampling ambient occlusion **when GTAO is
  off**, because then nothing produces that image at all; with GTAO on the same
  reads are ordinary same-frame dependencies on the blur, and are declared as such

That last split is the point: the declaration states which frame the data comes
from, and it changes with the setting that actually changes the answer.

#### Layout-only reads

A pass sometimes has to sample a resource it does not want. `composite.frag`
binds both bloom chains and selects one in the shader, so the losing binding must
still point at an image in a readable layout -- sampling one in
`VK_IMAGE_LAYOUT_UNDEFINED` is not allowed -- while its contents are discarded.
`readTextureForLayout` declares exactly that. A producer whose only reader is
layout-only is dead, so culling removes it, and the content rules in
`validateDeclarations` skip layout-only reads entirely.

`CompositePass` declares the selected chain with `readTexture` and the loser with
`readTextureForLayout`, both chains' passes are declared from the same predicate,
`PostProcessStack::willRecordMipChainBloom`, and the composite's `bloomMethod`
push constant reads it too, so the shader cannot select a chain that was never
filled.

Measured on the default scene at 1280x720 when it landed, passes actually
recorded per frame:

| Configuration | Before | After | Culled |
| --- | --- | --- | --- |
| mip chain (default) | 23 | **20** | the three legacy passes |
| `useMipChain` off | 23 | **16** | the seven mip-chain passes |

Both configurations ran with no declaration issues, no order violations, no
unrecorded passes, and the validation layer clean. The captured frame was
bit-identical to the pre-change capture, which is the point: the passes that went
away were producing something nothing read.

What the removed passes cost, on lavapipe -- not a frame-time claim for real
hardware, and not quotable as one; it establishes that the removed work was not a
rounding error. Two Release builds, the change reverted and restored, four
interleaved 60-second runs:

| Removed pass | Median, before |
| --- | --- |
| `BloomBlurVertical` | 4.215 ms |
| `BloomBlurHorizontal` | 4.153 ms |
| `BloomExtractPass` | 1.620 ms |

These are top-level scopes, and after the change they do not appear in the log at
all -- 46 occurrences of `BloomExtractPass` in a before run, zero in an after
run. The mip chain was unchanged across the two builds (6.835 -> 7.215 and
5.602 -> 5.709), which is what confirms the builds differed only in the legacy
chain. The frame-total comparison from the same series is **void** and
deliberately not quoted: the repeated controls drifted 1.3% and 2.3% against a 1%
limit, the same order as the effect. A software rasterizer pays for a fullscreen
pass in CPU fill where a GPU pays in bandwidth, so neither the milliseconds nor
their share transfers; three fullscreen passes at bloom resolution, removed, is
the part that does.

### Raster targets and the transitions behind a pass

A raster pass whose rendering scope is fixed names its colour target with
`colorAttachment`, and the graph opens and closes the scope around the callback:
the TAA resolve, the legacy bloom passes, every mip-chain level, the composite and
the overlay.

The rest open their scope from the callback, because its shape is decided while
recording or its attachments are more than one colour target:

| Helper | Pass | Why the callback opens it |
| --- | --- | --- |
| `beginShadowCascadeRendering`, `beginLayeredShadowRendering` | `CSMShadowPass` | which cascades the cache says to redraw |
| `beginPunctualShadowRendering` | `PunctualShadowAtlasPass` | whether the cached atlas loads or clears |
| `beginVsmPageRendering` | `VsmPagePass` | whether the cached page pool loads or clears |
| `beginProbeCaptureRendering` | `ProbeCapture` | colour plus depth |
| `beginDepthPrepassRendering` | `DepthPrepass` | depth only |
| `beginMainHdrPassRendering`, `beginMainHdrPhase2Rendering` | `MainHDRPass`, `MainHDRPhase2` | three colour targets plus depth |
| `beginSsrTraceRendering`, `beginGtaoRendering`, `beginGtaoBlurRendering` | SSR's trace, GTAO and its blur | sub-rect render areas |
| `beginTransparentRendering` | `TransparentPass` | three colour targets plus depth, all loaded |

Each checks that it is called from a callback of the right pass type, and emits
no barriers -- the graph already has -- with one exception: each cascade after the
first re-emits `CSMShadowPass`'s barriers, as each cascade's own begin always
did, which is what keeps the recorded barrier set unchanged.

### Work passes and explicit ordering

Some work has no graph resource of its own: what it writes is a buffer the graph
does not track, or nothing on the GPU at all. `addWorkPass` declares it as a
side-effect pass with no resources, and where its order matters it says so with
`orderAfter`, an edge `computePassSchedule` treats like one a resource implies.

| Work pass | Ordered after | Ordered before |
| --- | --- | --- |
| `VsmPageCullPass` | `VsmPageMarkPass` | `VsmPagePass` |
| `VsmPagePoolIdle`, the pool's one-time layout move when marking runs without page rendering | | `MainHDRPass` |
| `CSMShadowCache`, the cascade cache bookkeeping | `CSMShadowPass` | |
| `FogVolumeInit`, the fog volume's first-use clear | | `VolumetricFogPass`, `MainHDRPass` |
| `ClusterBuild`, light assignment on the graphics queue | `MainGpuCullingPass` | `VolumetricFogPass`, `MainHDRPass` |
| `ProbeShadingParams` | | `ProbeCapture`, `ProbeTrace`, `IrradianceProbeUpdate`, `MainHDRPass` |
| `FrameCapture`, the portfolio screenshot and the capture taken before the overlay | `CompositePass` | `ImGuiPass` |
| `FrameCaptureAfterOverlay` | `ImGuiPass` | |

`orderAfter` only names earlier passes -- the builder refuses a later one, and the
schedule ignores one that points forward or at a culled pass. `CSMShadowCache` is
declared every frame inside the `CSMShadowPass` scope, so that profiler number
exists every frame, at its real cost of nothing on a fully cached frame.
`ClusterBuild` is declared only when the cluster passes record on this queue; the
async path submits them on the compute queue, which
[async_compute.md](async_compute.md) describes and the graph does not model.

CPU state the recorded frame leaves behind is not a pass: invalidating the depth
pyramid on a frame that did not build it, marking the ambient-occlusion target
written, and the VSM page and punctual atlas counters of a frame without those
passes are set after `recordPasses`, because nothing recorded reads any of them.

### Profiler scopes over several passes

Some profiler numbers cover a run of passes rather than one: the bloom mip chain
is timed as "Bloom Downsample Chain" and "Bloom Upsample Chain", the ray-traced
shadow filter's three stages as one "RayTracedShadowFilter", and the cascade cull
and cascades as one "CSMShadowPass". Those are graph scopes, declared by
bracketing `addPass` calls with `pushScope` and `popScope`. The graph opens a
scope ahead of the first of its passes that records, and closes it when the frame
moves to a pass outside it; `planScopeTransition`, a pure function, decides which
scopes close and open between two passes.

### The passes

In declaration order, which is the frame's order. A pass in parentheses is
declared only on the frames its condition names.

| Region | Passes | Declared in |
| --- | --- | --- |
| Virtual shadow maps | (`VsmPageMarkPass`), (`VsmPageCullPass`, `VsmPagePass`), (`VsmPagePoolIdle`) | `Renderer::declareFrameGraph` |
| Cascaded shadows | (`ShadowGpuCullingPass`), (`CSMShadowPass`), `CSMShadowCache` | |
| Punctual shadows | (`PunctualShadowCullPass`), (`PunctualShadowAtlasPass`) | |
| Culling and lighting | `FogVolumeInit`, (`MainGpuCullingPass`), (`ClusterBuild`), (`VolumetricFogPass`) | |
| Ray-traced scene | (`RayTracingSceneBuild`), on frames that trace sun shadows or the probe gather | |
| Probes | `ProbeShadingParams`, (`ProbeCapture` or `ProbeTrace`), (`IrradianceProbeUpdate`) | |
| Geometry | (`DepthPrepass`), (`RayTracedShadowTrace`, `RayTracedShadowTemporal`, the filter's `RayTracedShadowRadius`, `RayTracedShadowBlurH` and `RayTracedShadowBlurV`, `RayTracedShadowResolve`), `MainHDRPass`, (`DepthPyramidMidPass`, `MainGpuCullingPhase2`, `MainHDRPhase2`), (`SSRCopyPass`, `SSRTracePass`), (`GTAOPass`, `GTAOBlurPass`), (`TransparentPass`), (`DepthPyramidPass`) | |
| Post-process | (`TAAResolvePass`), `BloomExtractPass`, `BloomBlurHorizontal`, `BloomBlurVertical`, `BloomDownsampleMipN`, `BloomUpsampleMipN`, (`LuminancePass`), (`HistogramExposurePass`), `CompositePass` | `PostProcessStack::declareGraphPasses` |
| Output | `FrameCapture`, `ImGuiPass`, `FrameCaptureAfterOverlay` | `Renderer::declareFrameGraph` |

Both bloom chains are always declared; culling removes the one the composite does
not sample. A conditional pass is declared from the same predicate its recorder
uses -- `GpuCulling::willRecordMainCull`, `PunctualShadows::willRecordCull`,
`PostProcessStack::willRecordLuminancePass` and the others -- so a declaration and
its recording cannot drift apart; "The backstop" is how the ones that did were
found.

## Compile

`compile` runs three GPU-free, unit-tested analyses over the declarations, in
order: culling, validation, and the schedule.

### Culling

`cullUnusedPasses` sweeps backwards over the passes with a set of resources
something still needs. A pass with no side effect is culled when none of its
writes is needed, or when it writes nothing at all; a culled pass does not
propagate its reads, which is what
makes culling transitive back through a chain. A layout-only read does not count
as a use, which is what removes the bloom chain the composite does not sample.
Side-effect passes are never culled: the main cull writes a readback buffer the
CPU reads, the overlay writes the image that is presented, every work pass writes
something the graph does not track.

A culled pass is not recorded: `buildExecutionPlan` leaves it out.

#### Passes the next frame needs

A pass can produce something only the *next* frame reads. Nothing in this frame's
declarations consumes it, so a plain backward sweep drops it, and the frame after
samples whatever was left behind.

`DepthPyramidPass` is exactly that shape: the phase-2 re-test reads the mid-frame
build, so nothing reads the end-of-frame one, and the next frame's page marking
and main cull declare history reads on it. It used to survive on a hand-set
side-effect flag -- a fine outcome and a bad mechanism, because a pass added in
that shape without the flag was culled silently.

`cullUnusedPasses` seeds its needed set from every history read before
sweeping, which is a reader at the end of the frame: the sweep already knows how
to keep the last writer of a needed resource and drop the ones before it. A
history read does not mark the resource needed at its own position, because what
it consumes is the previous frame. `DepthPyramidPass` is declared without the
side-effect flag, which is the proof the rule is load-bearing: with the seeding
removed the pass was culled, and the recorder of the time tried to record it
anyway and threw. That was checked, not assumed. The other side-effect flags were
not audited; removing one is a per-pass argument, not a sweep.

### Declaration validation

Nothing used to check that the declarations were self-consistent, so a pass could
read a transient nothing had produced and the only symptom would be wrong pixels
-- no validation error, no crash, nothing in a log. `validateDeclarations` checks
the ways a declaration set can be wrong without looking at a device:

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

The default configuration reports zero issues. That was checked both ways when the
rules landed -- injecting an undeclared history read and a duplicate declaration
into `MainHDRPass` made the panel report three findings on that pass, and turning
`CompositePass`'s bloom read into a history read reported the fourth rule.

### The schedule

`computePassSchedule` derives, from the declarations alone, which passes each
pass must follow. The edges are the three orderings a resource forces --
read-after-write, write-after-read, write-after-write -- plus every `orderAfter`
edge. A history read makes no read-after-write edge: it consumes the previous
frame, not this frame's producer. Read-after-write edges resolve through the
version a handle names, not through whichever write happens to be most recent, so
the edge set is a property of the declarations rather than of where a pass sits.

`computeExecutionOrder` turns that into an order: Kahn's algorithm over the
predecessor sets, breaking ties on declaration index. Every derived edge points
backwards by construction, so the result is the declaration order -- a unit test
pins that the two agree -- and a cycle, which the declarations cannot produce,
would leave the passes involved out of the order rather than emit them anywhere.

#### Judging an order

`validatePassOrder` takes a proposed order and reports the edges it breaks, and it
does reject orders. Checked on a real frame at 1280x720 when it landed:

| Proposed order | Violations |
| --- | --- |
| the recorded order | 0 |
| exposure pair hoisted 13 slots, to just after `TransparentPass` | **0** |
| the same hoist one step further, above `TransparentPass` | **2** |

So moving `LuminancePass` and `HistogramExposurePass` out from behind the bloom
chain is provably legal, and moving them one step too far is provably not.

#### Slack

How tightly the declared data flow pins the frame down: each pass's earliest legal
slot, its slack against where it is recorded, and the frame's longest chain are
shown in the debug panel. Measured on the default scene at 1280x720 when the
analysis landed, the frame was recorded 25 steps deep but the declarations forced
only 17:

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
The slack sits in the tail; the head of the frame has almost none, and reordering
on a single queue would not be a performance win if it could move. Moving a pass
to another queue would be -- which is the one change that would make scheduling
matter.

The same analysis is what showed that ten of the twenty-five passes were bloom,
split across two independent chains of which the composite used one, and led to
layout-only reads.

#### Which analyses a reordering would break

Nothing reorders today, but each analysis was examined for what a reordering
would do to it:

**`computeTextureLifetimes` had to move, and has.** Its intervals decide which
resources may share bytes, so an interval describing a timeline other than the
one the passes run on lets two simultaneously live resources overlap in memory --
and nothing catches that: not the validation layer, not a barrier, not a test
that does not happen to read the clobbered pixels. It takes the execution order
and numbers its slots by position in it.

**`cullUnusedPasses` does not need to.** Its answer is invariant under any legal
reordering. The derived edges fix the relative order of every pair of passes that
touch a common resource, and two passes touching no common resource cannot affect
each other's liveness.

**`validateDeclarations` splits.** Its content rules -- `ResourceDeclaredTwice`,
`ReadsContentNoPassProduced`, `HistoryReadOfAliasedResource` -- are invariant for
the same reason. Its two ordering rules, `StaleVersionRead` and
`HistoryReadAfterProducer`, are not, and must not be: they report that the
declared order and the declared data flow disagree, which is a statement about
the declarations.

**Barriers** have none of this problem: they are derived from live tracked state
as each pass is recorded, so they follow whatever order the recording takes.

**The geometry region's callbacks** would: see "Passes share drawing state
through the command buffer".

## Recording

`recordPasses` walks `buildExecutionPlan` -- every scheduled pass with a
callback, in execution order -- and for each one:

1. moves between graph scopes, opening the ones the pass sits in;
2. opens the pass's debug label, then its GPU profiler scope;
3. emits the barriers its declarations imply, batched into one dependency;
4. opens the rendering scope over its `colorAttachment`, if it named one;
5. calls the callback with a `RenderGraphPassContext`;
6. closes the rendering scope and records any `transitionAfterPass`;
7. closes the profiler scope and the label.

Label and profiler scope sit outside the barriers because that is where every
hand-written recorder put them, so each pass's timestamp range covers its
barriers and the profiler history stays comparable across the change. The
profiler belongs to the renderer; the graph is handed two hooks for it.

While a callback runs the graph marks itself busy, and the in-callback rendering
helpers check that the running pass is of the type they open a scope for. A
callback that leaves a rendering scope of its own open throws.

### Passes share drawing state through the command buffer

The main HDR pass sets up what phase 2 and the transparent pass draw with -- the
viewport, the global descriptor set, the indirect buffers and the base push
constants -- and hands those values to them in a `MainPassFrameState` it fills
when it records. It is filled at the point the values have always been computed,
because the base push constants include whether the ambient-occlusion history is
usable, and GTAO changes that later in the frame.

Each geometry callback binds exactly what the hand-written recorder bound at that
point, which kept the command stream unchanged when the recorder was split. The
cost is that callbacks still rely on Vulkan state an earlier pass left bound:
phase 2 and the transparent pass rebind their own pipeline and sets, but a pass
that did not would draw with whatever the pass before it bound. Nothing reorders
passes, so nothing exercises that; a scheduler that moved a geometry pass would
first have to make each callback bind everything it uses.

### The frame trace

Each run logs the last frame's recorded passes, in order, with the barriers the
graph emitted for each:

```
Render graph frame trace: ... MainHDRPass[6/0/1] DepthPyramidMidPass[2/0/1] ...
```

as `Name[image/buffer/submits]`. A `--deterministic` run ends on the same frame
every time, so a change to who records a pass has to leave this line unchanged
-- a work pass that was added carries no barriers, `[0/0/0]`. That is a stronger
check than the golden image, since a missing barrier rarely shows up in pixels.
Together with the per-frame GPU scope list (`--gpu-timings-every-frame`), it is
what every step of the move to callbacks was checked against; see "How it got
here".

## Synchronization Inference

Texture declarations use `RGAccess` values such as `ColorAttachmentWrite`,
`DepthStencilAttachmentWrite`, `ShaderRead`, `StorageImageReadWrite`,
`TransferSrc`, `TransferDst` and `Present`. As each pass is recorded, the graph
maps its declared accesses to conservative Synchronization2 image barriers. It
tracks the current layout and previous access of every graph texture, and writes
the layout back to the owner of an imported one -- the swapchain, the shadow maps,
the depth pyramid -- so the owner's own view of the image stays true.

Buffer declarations use `StorageBufferRead`, `StorageBufferWrite`,
`StorageBufferReadWrite`, `StorageBufferClearAndReadWrite`, `IndirectRead`,
`TransferSrc`, `TransferDst`, `HostRead`, and three acceleration-structure
accesses:

- `AccelerationStructureBuild` and `AccelerationStructureRead`, declared on a
  top-level acceleration structure's storage buffer by the build and by the
  passes that trace against it ([ray_traced_shadows.md](ray_traced_shadows.md)).
  Adding them found that the write mask had no acceleration-structure write bit,
  which made a build followed by a trace look like two reads and emit nothing.
- `AccelerationStructureComputeRead`, the same read from the ray-traced shadow
  mask's trace pass (`RayTracedShadowTrace`), scoped to the compute stage.

Buffer accesses become conservative full-buffer barriers. What the graph tracks
per buffer, for the frame (`BufferSyncState`, applied by `advanceBufferSync`):

- the last write's stage and access;
- every reader since that write -- or since the frame began, while nothing has
  written;
- the stage/access scopes the last write has been made visible to: the second
  scope of each barrier emitted since it, kept as pairs, because a barrier makes
  its accesses visible to its own stages only.

From that:

- **A write** waits for the last write and for every reader since it. Naming
  only the most recent reader, as the graph used to, leaves the earlier ones
  racing the write: no barrier chains from them.
- **A read** gets a barrier from the last write unless one since has already
  named the read's stage together with its access. The first reader after a
  write gets one; so does a later reader in a stage or with an access no earlier
  barrier reached; a repeat read in a stage already reached gets none.
- **A read before any write** this frame, and a buffer's first use of any kind,
  gets nothing: buffer state is rebuilt every frame, so there is nothing in the
  frame to order against. Cross-frame edges stay with the subsystems (see "What
  subsystems still write by hand").

Until this tracking existed, a read after a read emitted nothing whatever its
stage, so only the first reader after a write was ordered against it. The
ray-traced shadow mask is what hit it: an `api_dump` showed its compute-stage
read of the top-level acceleration structure taking the build's only barrier,
with the main pass's fragment-stage rays unordered behind it. The mask's access
named the fragment stage too as a stopgap; it names the compute stage alone now,
and the main pass gets its own barrier
([ray_traced_shadows.md](ray_traced_shadows.md#synchronization)).

Images keep the pairwise rule -- the last access against the next one -- and do
not need more. Every read of one layout names the same stages
(`textureAccessState`), and a change of layout is itself a barrier whose first
scope is the previous access, so each one chains from the one before it.

`tests/test_render_graph.cpp` drives `advanceBufferSync` through sequences: a
second reader in another stage, repeat reads in a stage already reached, a write
after two readers, a write after another write, visibility per stage/access pair,
and what happens past the state's room for scopes. Each rule was checked by
breaking it and watching a case fail.

What it changed when it landed, counted over 60 frames of each configuration
with the old rule evaluated beside the new one (a temporary log, not committed):

| Scene | Settings | Buffer barriers per frame, before -> after | Of which new | Wider first scope |
| --- | --- | --- | --- | --- |
| default | defaults | 10 -> 10 | 0 | 2 |
| default | ray-traced shadows | 9 -> 9 | 0 | 2 |
| default | ray-traced shadows, screen-space mask | 9 -> 10 | 1 | 2 |
| default | ray-traced shadows, 2-degree sun (filtered mask) | 9 -> 10 | 1 | 2 |
| sponza | defaults | 8 -> 8 | 0 | 2 |
| sponza | ray-traced shadows | 8 -> 8 | 0 | 2 |
| sponza | ray-traced shadows, screen-space mask | 8 -> 8; 5 on the frame that builds | 1 on that frame | 2 |
| sponza | ray-traced shadows, 2-degree sun (filtered mask) | 8 -> 8; 5 on the frame that builds | 1 on that frame | 2 |

No barrier disappeared. The one new barrier is the main pass's own, from the
acceleration-structure build, on a frame that both builds and runs the mask's
trace -- every frame on the default scene, whose skinned caster moves, and only
the first on Sponza, which keeps its structure after that. It joins the pass's
existing `vkCmdPipelineBarrier2` rather than adding one. The soft mask's filter
passes read and write images only, so they add nothing here. The two wider scopes
are `MainGpuCullingPhase2`'s writes to the main cull's indirect commands and
visible counts: their first scope used to be the draw-indirect stage of the
reads before them and now also names the phase-one cull's write. That adds no
wait -- by then the depth transition after the main pass has waited for its
fragment tests and every stage before them, and the Hi-Z barrier for the
pyramid's compute -- so the frame cost was not measured.

The implementation favours correctness and debug readability over barrier
minimization. Same-layout write-after-write or write-after-read cases can still
emit ordering barriers.

### Checking the inferred barriers against something

Everything above is inference. The unit tests cover the pure derivation
functions, the golden gate compares pixels, and the validation layer's core checks
confirm each call is well formed; none of the three can see a dependency that is
missing. A barrier naming the wrong pipeline stage is a legal barrier, and the
frame it fails to order still renders correctly on hardware that happened to
serialize the work anyway.

Synchronization validation can see it, and `--sync-validation` turns it on
(`VK_EXT_layer_settings`, `validate_sync`, set at instance creation because there
is no runtime toggle). Asking for it on a build with the validation layer
compiled out, or against a layer too old to provide the extension, is a hard
failure rather than a quiet downgrade -- a run that skipped the check and
rendered a clean frame would otherwise read as evidence about synchronization it
never examined.

`--sync-validation-selftest` is the negative control. It records two overlapping
writes to one buffer with no barrier between them, reports whether the layer
caught it, and exits without rendering. Both writes are individually legal, so
core validation has nothing to say about them. Without this, "rendered a frame,
nothing reported" and "the setting was ignored" are the same observation.

Sync hazards are tallied apart from other validation errors -- on the layer's own
`SYNC-` message id, not on message text -- and the `Validation tally:` line
carries the count.

#### What the first run found

Thirty hazards on the default scene, in four classes, all of them the same
mistake: a barrier naming a narrower pipeline stage than the command it was
meant to order.

| Class | Count | Cause |
| --- | --- | --- |
| SSR half-res copy | 22 | `TransferSrc`/`TransferDst` mapped to `COPY`, and the pass records `vkCmdBlitImage`, which runs in the blit stage |
| GPU culling count buffers | 3 | a pass that clears with `vkCmdFillBuffer` and then dispatches declared only the shader scope |
| Swapchain acquire | 3 | the transition out of `PRESENT_SRC` had a source scope of `NONE`, so it was ordered against nothing and could run ahead of the semaphore wait |
| Probe shading params | 2 | one buffer served every frame in flight, and the barrier after the update said nothing about the previous frame's read |

`VK_PIPELINE_STAGE_2_COPY_BIT` covers `vkCmdCopy*` and nothing else;
`vkCmdBlitImage` is in the blit stage and `vkCmdFillBuffer`/`vkCmdUpdateBuffer`
are in the clear stage. The graph maps both transfer accesses to `ALL_TRANSFER`,
which covers copy, blit, resolve and clear alike, and a test pins the stage for
textures and buffers alike. The same narrow spelling was repeated by hand in
`VirtualShadowMapPass` and `IrradianceProbeVolume`; both were fixed.

`RGAccess::StorageBufferClearAndReadWrite` exists because a pass declares a
resource once while these passes genuinely touch it twice -- a transfer clear and
then a shader read-write. It names the union of the two scopes. Three
declarations used the narrower access wrongly (both GPU culling passes and the
histogram pass); only two produced a hazard, because the third's previous use was
separated by a frame wait.

None of the four changed a pixel: the frame 30 capture was byte-identical before
and after. That is the expected result and the reason the class survived -- this
hardware was already serializing the work these barriers failed to order. The
headless job's configuration sweep has run under `--sync-validation` ever since
(see [headless_ci.md](headless_ci.md)).

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
transition derives its barrier from what the first one leaves behind.

`RenderPassNode::generatedBarrierSubmitCount` above one is the signal that a pass
has started declaring a resource twice -- with two standing exceptions: `ImGuiPass`
reports two, because its transition of the swapchain into the present layout is
recorded after the pass body, and `CSMShadowPass` reports up to one per redrawn
cascade, because each cascade re-emits the pass's barriers.

Measured on the default scene at 1280x720 with default settings, on lavapipe under
the headless-render job's flags, when batching landed:

| | Per frame |
| --- | --- |
| executed passes | 23 |
| inferred barriers | 57 |
| `vkCmdPipelineBarrier2` calls, before | 57 |
| `vkCmdPipelineBarrier2` calls, after | **22** |

The barrier count is identical on both sides, which is the property that matters:
the batching moved 57 barriers from 57 commands into 22, and added none. This is a
command-count reduction, **not a measured frame-time win** -- no timing claim is
attached to it. It is also the prerequisite for tightening the alias-handoff
source scope, since that only pays off once a pass's barriers share one
dependency.

### What subsystems still write by hand

A barrier belongs to the graph when it sits between two passes, and to the
subsystem when it sits between two commands inside one pass, or orders something
the graph does not track. That line is why the manual count will never reach zero.

| File | What it synchronizes by hand |
| --- | --- |
| `GpuCulling.cpp` | each cull's fill-to-dispatch reset, the phase-2 count reset, and the copies to the readback buffers with their host-visibility barriers |
| `PunctualShadows.cpp` | the punctual cull's reset and dispatch |
| `VirtualShadowMapPass.cpp` | the page-marking and page-cull request buffers, which are not graph resources |
| `PostProcessStack.cpp` | the histogram reset, the in-pass histogram-to-exposure reduce, and exposure host-read visibility |
| `IrradianceProbeVolume.cpp` | the shading-parameter upload, including the previous frame's read of the one buffer every frame shares, and the atlas update's dispatches |
| `DepthPyramid.cpp` | the mip chain's write-to-read sequencing inside its own pass |
| `ClusteredLighting.cpp` | cluster build to light cull, and the lists to their readers' stages |
| `VolumetricFogPass.cpp` | injection to integration, between its own volumes |
| `RayTracingScene.cpp`, `RayTracingSkinning.cpp` | the bottom-level builds at load, and the skinned positions to the refit inside the scene build |
| `RayTracedShadowMask.cpp` | returning an image to the layout the graph tracks |
| `ScreenshotCapture.cpp` | the frame captures' copies, which move the swapchain image out of its tracked layout and back |
| `RendererRecord.cpp` | `VsmPagePoolIdle`'s one-time move of the page pool, which is not a graph resource when page rendering is off |

Now declared, so the graph emits the handoff between passes:

| Resource | Producer | Consumer |
| --- | --- | --- |
| The three froxel volumes | `VolumetricFogPass` | `MainHDRPass` samples the integrated one |
| Shadow cull indirect commands and counts | `ShadowGpuCullingPass` | `CSMShadowPass` draws them indirectly |
| Punctual cull indirect commands and counts | `PunctualShadowCullPass` | `PunctualShadowAtlasPass` draws them indirectly |
| Main cull indirect commands and counts | `MainGpuCullingPass` | `MainHDRPass` draws them indirectly |
| Luminance partials and exposure state | `LuminancePass`, `HistogramExposurePass` | `HistogramExposurePass`, `CompositePass` |

The froxel volumes also removed `VolumetricFogPass`'s `sideEffect` exemption: it
had one because nothing downstream declared a read on its outputs, and liveness
would have culled it.

Two more have the shape of a conversion and are blocked by something other than
the graph:

- **The VSM page cull's request buffers.** The cull is a pass, `VsmPageCullPass`,
  ordered between marking and the page pass, but the buffers it writes are built
  per frame inside the recorder -- the caster batches and the per-page command
  stride -- so they are not graph resources and carry their own barriers.
  Hoisting that batch build into frame prep is the prerequisite for declaring
  them, and it is a change to where CPU work happens rather than a graph change.
- **The probe shading parameter buffer.** Its upload is a pass,
  `ProbeShadingParams`, ordered before every reader. Its cross-frame hazard -- one
  buffer serving every frame in flight -- is handled by a barrier before the
  update, and would not be the graph's to emit in any case: buffer state is rebuilt
  every frame, so the graph has nothing to order a frame's first write against.

## The backstop

Declarations and recording used to be two sequences kept in step by hand. Each
callback now sits beside its declaration, but whether a declared pass records
anything, and whether its recorder agrees with its declaration about when it
runs, are still claims the graph checks. `beginDeclaredPass` records the order
passes were actually begun in, and `endFrame` checks it:

- `validatePassOrder` -- dependencies the recording order broke;
- `unrecordedPasses` -- scheduled passes never recorded at all. `validatePassOrder`
  sees a missing pass only through the edges it breaks, so one with no
  predecessors would otherwise pass unnoticed.

Both are reported in the Render Graph panel, never thrown, for the same reason
`validateDeclarations` reports. Across a run they are rolled up into one greppable
line printed at shutdown, beside the frame trace:

```
Render graph backstop: 0 order violations, 0 unrecorded passes, 0 declaration issues over 40 frames
```

Peaks across the run rather than the last frame's counts, because a violation in
frame 7 that clears again by frame 30 still has to be reported; the line carries
the frame count, because zero findings over zero frames is what a run that
rendered nothing looks like. The headless job renders
[every swept configuration](../config/ci/README.md) past it.

### What the backstop found

Every finding below was a declaration describing work that did not happen -- its
culling, its barriers, its resource lifetimes modelling a frame other than the one
submitted -- and none was visible to any other gate, because each Vulkan call was
still legal.

On its first run, on the default configuration:

- **`CSMShadowPass`.** The cascade cache skips the redraw when every cascade is
  clean, but the declaration was unconditional. On a cached frame the main pass
  read a shadow map the graph believed had just been written.
- **`LuminancePass`.** Its recorder returns early in histogram mode, which is the
  default, so `HistogramExposurePass` declared a read of partials nothing had
  produced.

Both now follow the pattern the punctual atlas and the VSM page pool already
used: the pass is declared only when the renderer will record it, while the
resource stays imported and its readers stay declared, so it keeps the layout its
sampler claims. The predicates are shared rather than duplicated --
`Renderer::anyCascadeShadowRedrawRequired` and
`PostProcessStack::willRecordLuminancePass` are asked by both the declaration and
the recorder. Collapsing the luminance recorder's three early-outs into that one
predicate is what closed it; the first attempt covered only the first of them,
and the backstop said so.

Then on the configurations that had not been looked at, once the sweep ran past
the backstop:

- **`HistogramExposurePass`** was declared unconditionally while its recorder
  returns early under manual exposure.
- **`DepthPyramidPass`** was declared every frame while the recorder skips the
  build whenever nothing consumes the pyramid, so with occlusion culling off the
  graph modelled a pass that never ran. Its declaration is gated on the same
  latched decision the recorder reads -- latched, because
  `isDepthPyramidBuildRequired()` reads state the yield controller can move
  between declaring and recording.
- **`TransparentPass`** was declared whenever anything was blended, but the
  transparent pipeline binds the bindless heap as its second descriptor set, so
  with bindless off the recorder skipped it.

And then on the configurations that still did not run. "Every swept
configuration" had been doing more work than it had earned: the leg list had
grown one entry at a time as features shipped, and nothing had counted what it
reached -- twenty-one legs moving twenty-one of 137 settings keys, with most of
the settings that select a code path rendered one way only. Counting it found:

- **`MainGpuCullingPass`** was declared unconditionally, while
  `GpuCulling::recordMainCull` returns without recording on the CPU culling
  fallback and on an empty draw list. Every frame of that configuration reported
  one unrecorded pass and one order violation, and no leg turned GPU culling off.
  It is declared from `GpuCulling::willRecordMainCull` now, the predicate the
  recorder itself returns on.

Every one of these fixes was bookkeeping: the configurations captured either side
of them were bit-identical. What changed is that the graph's model of the frame
matches the frame that was submitted. The sweep's coverage is now checked rather
than assumed, by `tools/check_ci_settings_coverage.py` in a job of its own -- see
[headless_ci.md](headless_ci.md) and [`config/ci/`](../config/ci/README.md).

## Debug UI

The ImGui Render Graph panel shows:

- declared pass order, execution status, and culled status with its reason
- the side-effect flag
- read and write resources with their declared access
- generated image and buffer barrier counts, and the number of
  `vkCmdPipelineBarrier2` submissions they were batched into
- declaration issues found for each pass, and a frame-level count above the table
- each pass's derived predecessor count, earliest legal slot and slack, with the
  predecessor names in a tooltip, plus the frame's longest dependency chain
- whether the recorded order matched the schedule, and any pass that was declared
  and scheduled but never recorded
- each resource's name, kind, lifetime, extent or size, format, mip and layer
  count, initial layout and final layout

Resource versions are not shown; they are visible through the pass declarations.

## Transient Memory Aliasing

Resources whose lifetimes do not overlap can share bytes in one allocation.
`renderer::planTransientMemory` (GPU-free, unit tested) assigns offsets by greedy
interval packing, `computeTextureLifetimes` supplies the intervals -- numbered by
position in the execution order, not by declaration index -- and
`rhi::VulkanTransientMemoryPool` owns the allocation that
`VulkanImage::createAliased` binds images into.

Measured at 2560x1440: the whole transient set is 123.74 MiB and would pack into
82.62 MiB. At the time only the bloom chain was wired -- 41.12 MiB of images into
a 23.64 MiB pool, **17.48 MiB saved**; the G-buffer pair joined it later, below.

### It once regressed to never activating at all

The measurement below was taken while the feature worked. It stopped working
afterwards, silently, and stayed broken because it is off by default.

The planner only requests transients a pass actually uses this frame -- correct,
since an unused resource has no lifetime to place. The binder then required a
placement for **every** image in the bloom chain, treating a missing one as a
failure and falling the whole chain back to private allocations. Those two rules
are incompatible the moment part of the chain is culled, and the two bloom paths
cull each other: with `useMipChain` on, `BloomExtract` and `BloomPing` have no
pass; with it off, the mip images do not. So one group was always unused, always
unplaced, and always failed the bind:

```
useMipChain on : Bloom image 'BloomExtract' could not be bound into the transient pool; ...
useMipChain off: Bloom image 'BloomMipDownsample0' could not be bound into the transient pool; ...
```

Both configurations, which is to say all of them. Turning
`renderer.enableTransientAliasing` on did nothing but log a warning.

The fix distinguishes the two reasons an image can lack a placement.
`BloomAliasPlacement::unusedThisFrame` marks the ones the planner left out on
purpose; those get a private allocation, which is safe precisely because no pass
reads or writes them. An image that *is* used and still has no placement remains a
planning bug and still tears the whole chain down, which is the case the
all-or-nothing rule exists for.

### The G-buffer joins the pool, which is where the memory actually was

Bloom alone saved 2.85 MiB. The lifetimes said where the rest was:
`NormalRoughnessGBuffer` (7.5 MiB) and `VelocityBuffer` (3.75 MiB) both stop being
read at pass 7, while the bloom chain does not start until pass 9 -- 11.25 MiB
sitting idle for two thirds of the frame with nothing reusing it.

They are now bound into the same pool. Measured on an RTX 3080 Ti Laptop,
`--scene stress` at 1280x720:

| | images | pool | saved |
| --- | --- | --- | --- |
| bloom only, `useMipChain` on | 7.23 MiB | 4.38 MiB | 2.85 MiB |
| bloom only, `useMipChain` off | 7.50 MiB | 5.62 MiB | 1.88 MiB |
| **with the G-buffer, `useMipChain` on** | **18.48 MiB** | **11.25 MiB** | **7.23 MiB** |
| **with the G-buffer, `useMipChain` off** | **18.75 MiB** | **11.25 MiB** | **7.50 MiB** |

The pool is now exactly the G-buffer pair's 11.25 MiB, which is to say the whole
bloom chain fits inside their bytes after pass 7 and costs nothing further. That
matches the ceiling `--probe-aliasing` reports for the frame as a whole (28.79 MiB
of transients into a 21.56 MiB pool), so this is the end of what these lifetimes
allow rather than a step towards it.

Zero validation errors in every configuration, and all six scene presets pass the
golden gate (`--channel-tolerance 1 --max-differing-fraction 0`) against the
private-allocation reference. `--probe-aliasing` confirms the driver really shares
the bytes rather than merely accepting the binds.

**What made it safe to extend.** Three things, and skipping any of them would have
corrupted pixels rather than failed:

- Both images are declared to the graph with `aliased = true`. Without that the
  graph emits no handoff barrier and the bloom writes land on live G-buffer bytes.
- `recreatePostProcessResources` already rebinds SSR and GTAO against a fresh
  normal-roughness view, so recreating the image pool-bound does not leave stale
  descriptors. That path existed for resize; aliasing only needed it to be true.
- Neither is history-read, which is now enforced rather than checked once; see
  below.

**The membership is a name list on purpose.** `isAliasableTransient` in
`Renderer.cpp` names them. Whether a resource can be recreated as aliased is a fact
about the code that owns it, not about how it is declared to the graph, and adding
a name without giving it a recreate-and-rebind path binds it into shared memory
while stale descriptors still point at the image it used to be. `SceneColorHDR` and
`AmbientOcclusion` are the obvious next candidates and are deliberately absent:
both live nearly the whole frame, so they would take pool space without freeing
any.

### History reads are what the lifetime rule had to grow a guard for

Aliasing was safe before this only by accident of scope: bloom is not history-read,
so the question never arose. Extending the pool made it worth answering properly.

`computeTextureLifetimes` derives an interval from one frame's usages, and for a
resource something reads with `RGReadKind::History` that interval is not the one
its bytes have to survive. Written at slot 17 and history-read at slot 4, it looks
free over [0, 3] -- and whatever the allocator packs there overwrites, at the start
of frame N+1, exactly what frame N left for it to read. The read happens after the
clobber, so nothing reports it: not the validation layer, not a barrier, only the
pixels.

Such a texture now comes back live for the whole frame, whatever its usages say,
which leaves it in the pool but stops anything reusing its bytes. Three cases in
`tests/test_render_graph.cpp` pin it, including that an ordinary `Produced` read
still gets its tight interval -- the guard has to be specific, or it would forbid
the sharing the allocator exists to do.

Today that covers `AmbientOcclusion`, `DepthPyramid` and the TAA history, none of
which are in the alias list. The TAA disocclusion test leans on it: the previous
frame's view depth travels in the history's alpha
([taa.md](taa.md#disocclusion-rejection)), which is only right if nothing reuses
the history's bytes between the frame that writes it and the frame that reads it.

### What it costs, and why it stays off

Measured A/B, interleaved with a repeated control (drift 0.48%, limit 1%). Hardware is not recorded in the commit that introduced this (`2924483`, 2026-08-16, inside the unlabelled window described in [profiling.md](profiling.md#which-machine-a-number-came-from)); a 15 ms frame on the default scene is M3-class, against 1.754 ms on the RTX 3080 Ti, so read this as the tiler. The statistic is a median, not the p10 the harness now quotes. The conclusion -- a memory saving bought with frame time -- is a ratio and does not depend on which machine it was taken on, but the absolutes do.

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
not wired up then (the G-buffer pair above joined afterwards, because that is
where the memory turned out to be): more aliased resources means more handoffs, so the cost would grow
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

#### lavapipe cannot answer this; do not re-run it there

Measured and rejected as a method. An interleaved A/B of
`renderer.enableTransientAliasing` on Mesa's lavapipe passed its own control-drift
gate (0.54%, limit 1%) and still produced nothing usable: every pass moved the
same direction, including `ClusterBuild` at -17%, `LightCull` at -17%,
`MainGpuCullingPass` at -13% and `CSMShadowPass` at -6% -- passes whose buffers
aliasing does not touch. A frame total 2.6% *faster* with aliasing on also
contradicts the +1.2% measured on real hardware.

The reason is structural rather than statistical, so more samples will not fix
it. On a software rasterizer a pipeline barrier is a scheduler sync point, not a
cache flush and pipeline drain, so the cost this section is asking about does not
exist on that driver. What the numbers most likely reflect is the ~6 MiB smaller
resident set changing CPU cache pressure for the whole frame -- an effect of the
rasterizer, not of the barrier.

Answering it needs the hardware the +1.2% was measured on.

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

- **One queue.** Async compute exists in the engine -- the cluster build and light
  cull run on a dedicated compute queue so they can overlap the shadow passes (see
  [async_compute.md](async_compute.md)) -- but the renderer owns that submission
  and its semaphores, not the graph. On the graphics-queue fallback they record in
  the `ClusterBuild` work pass and carry their own barriers to their readers; on
  the async path the graph has no pass for them at all.
- **No reordering.** The execution order equals the declaration order by
  construction. The schedule could move passes legally; what would have to change
  first is in "Which analyses a reordering would break".
- **Geometry callbacks share bound state.** See "Passes share drawing state through
  the command buffer".
- **Conservative barriers.** They are batched per pass (see "Barrier batching"),
  but their stage and access scopes are unoptimized. They are checked, however:
  see "Checking the inferred barriers against something".
- **Whole-resource tracking.** Versions, layouts and barriers are per resource,
  not per subresource. See "Per-subresource tracking has no consumer here" for the
  survey that says this costs nothing today and for what would change that.
- **Memory aliasing is off by default** (`enableTransientAliasing`), and covers the
  bloom chain and the velocity/thin-G-buffer pair. See "Transient Memory
  Aliasing".
- **Physical allocation stays with the subsystems.** Transient targets and the TAA
  history are graph-described but allocated and recreated by the code that owns
  them.
- **Intra-pass sequencing is manual, and always will be.** The graph emits at pass
  boundaries, so a buffer filled, dispatched against, copied and made host-visible
  inside one pass carries its own barriers between those steps; see "What
  subsystems still write by hand".
- **Not every sampled resource is a graph resource**, including material textures,
  IBL cubemaps, the BRDF LUT and render-target preview descriptors.
- **Declaration validation does not model cross-frame liveness** beyond history
  reads keeping their producer alive. A history read is only satisfied if *this*
  frame's producer runs; seeding culling from history reads covers that, but
  validation itself judges one frame at a time.
- **No node-editor view.**

### Per-subresource tracking has no consumer here, surveyed

Barriers cover the whole image -- `baseMipLevel = 0, levelCount = mipLevels`,
`baseArrayLayer = 0, layerCount = arrayLayers` -- and a write advances the
version of the whole resource. That reads like a gap. It was surveyed across all
27 configurations the headless job ran at the time, by logging every graph
texture with more than one subresource together with every pass that declares
it:

| Resource | Subresources | Declared by |
| --- | --- | --- |
| `DepthPyramidHiZ` | 11 mips | `DepthPyramidPass` (storage write), plus `MainGpuCullingPass`, `MainGpuCullingPhase2`, `VsmPageMarkPass` (sampled reads) |
| `CascadedShadowMapArray` | 4 layers | `CSMShadowPass` (depth write), plus `MainHDRPass`, `VolumetricFogPass`, `ProbeCapture` (sampled reads) |

Those two are the whole list, in every configuration, and **every declaration on
them covers the entire resource**. The pyramid build writes every mip and its
readers sample every mip; the cascade pass declares the whole array and its
readers sample every cascade. Nothing asks for a subset, so nothing is lost by
not offering one.

Three things make it more than a coincidence:

- The bloom mip chain is not a mip chain in the Vulkan sense. Each level is its
  own `VulkanImage` and its own graph resource, so it is already tracked
  separately.
- `DepthPyramid`'s per-mip barriers are not layout transitions. They are
  `GENERAL` to `GENERAL` write-to-read dependencies between consecutive
  dispatches, inside a single pass. Pass-boundary tracking, however fine-grained,
  would not remove one of them.
- Every owner of a multi-subresource image stores exactly one `VkImageLayout` --
  `VulkanShadowMap::layout_`, `DepthPyramid::layout_` -- and the graph points at
  that field. The engine cannot represent a divergent image at all, so
  subresource state in the graph alone would be state nothing else could honour.

What would change the answer is a pass that touches part of an image and leaves
it divergent at a pass boundary: per-cascade shadow rendering that skips cached
layers rather than declaring the array, a mip-chain effect built as mips rather
than as separate images, or per-layer VSM page rendering. Any of those needs the
owners' single layout field to go first.

Until then the assumption is not resting on this document: a wrong layout in an
inferred barrier is exactly what synchronization validation reports, and every
configuration in the sweep runs under it.

### Declaration allocates for debug text, measured and left alone

The declaration path allocates once per frame for every piece of debug text it
carries: `addPass` takes `std::string name` by value, every usage stores a
`std::string description`, and `textureResourceHandle` copies the resource name
into `RenderResourceHandle`. When it was measured, across ~23 live passes and 99
usage declarations, that was upwards of 220 heap allocations a frame, none of which the GPU ever sees.

Rewriting them as `std::string_view` onto static storage -- plus a static table
for the two mip names that are built with `std::to_string`, and resolving the
resource name by index rather than copying it -- was benchmarked in isolation
against the real string lengths and counts: **10.0-11.1 us/frame today, 4.5-4.9
us/frame after, so 5.5-6.3 us recoverable**.

That is not worth taking. On an RTX 3080 Ti Laptop, `--scene stress`, Release,
frame prep CPU ran 0.99 ms average against a 1.01 ms GPU frame when this was
measured, so the recovery is **0.6% of the CPU frame** -- below the 0.49%
control drift the A/B harness itself reports on a good run, which means it could
not be demonstrated after the fact.

The denominator has since moved: the object transform cache and the parallel
punctual cache-key build took frame prep down about 19% on the same scene, which
raises the recovery to roughly 0.75% of it. That is still under the drift gate,
so the conclusion is unchanged -- but the figure above is the one from the day
it was taken, not a current reading. Against that, the change touches `RGResourceUsage`, `RenderResourceHandle`,
the debug UI that reads them, and the tests over the analysis functions, and it introduces a lifetime rule (views into `textures_[i].desc.name`
would dangle when the vector reallocates a short, SSO-stored name) that the
current owning strings do not have.

Worth revisiting only if the declaration set grows by an order of magnitude, or
if declaration moves off the recording thread and makes the allocator contended.

## How it got here

The graph started as a description beside a hand-sequenced frame: the renderer's
recorders ran in a fixed order, each calling a `begin*Pass` / `end*Pass` pair that
emitted the pass's inferred barriers. Declarations lived in the graph, recording
in the renderer, and the backstop existed to notice the two drifting apart.

It became the thing that records the frame in steps, each of which had to leave
the recorded frame unchanged -- the frame trace identical, the per-frame GPU scope
lists identical, the captures matching -- checked on the RTX 3080 Ti across the
CI configurations and more (VSM stages, every ray-traced shadow variant, the
frame captures):

1. the graph took over the order, sequencing named recorders ("units") by the
   position of a pass each was anchored on;
2. passes moved from units to execute callbacks the graph calls, single passes
   first, then the multi-pass and multi-instance ones, then the geometry region;
3. the remaining units became work passes with explicit `orderAfter` edges;
4. the declarations moved out of the graph to sit beside their callbacks.

Two lessons from the way, both of which the structure now enforces.

### Declaration order has to match recording order

While nothing acted on the declared order, the declarations and the recording
could disagree for free, and they did: `VolumetricFogPass`, `ProbeCapture` and the
irradiance probe update were declared before `MainGpuCullingPass` and recorded
after it. Handing the order to the graph turned that latent disagreement into a
real reordering -- fog sorted ahead of the synchronous cluster build, whose light
lists fog injection walks. The graph did not model that dependency, and could
not: the cluster buffers are not graph resources and the async path submits them
on another queue. The declarations moved to where the passes actually run.

The rule this leaves: **an ordering constraint the graph does not model has to be
stated**, which is what `orderAfter` is for. Fog and the main pass name
`ClusterBuild` that way, and the probe passes name `ProbeShadingParams`.

### Why units went

Work without a pass of its own was a unit: a recorder anchored on a pass, running
right after it -- or, when its anchor was not scheduled, with the last anchored
unit registered before it. That inherited position moved whenever a unit before
it went. When fog became a callback the fog unit went, and the probe recorder,
which inherited from it on frames without a capture (the cold-start seed and the
debug-pattern path), inherited from the cluster build instead and recorded
`IrradianceProbeUpdate` ahead of `VolumetricFogPass`. Nothing depends on the order
of the two, so no barrier, backstop result or pixel changed, and the default
captures never showed it, because their last frame has a capture. The per-frame
GPU scope list did.

A pass cannot drift that way: its position is its declaration, and what it
depends on is either a resource or an `orderAfter` edge.
