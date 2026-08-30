# Design Decisions

Why the renderer is built the way it is — the trade-offs behind the major
subsystems, written so they can be defended in a technical interview. Each entry
is *decision → why → trade-offs → what I'd do with more time*.

## Clustered (Forward+) lighting

**Decision.** Assign lights to a 32×18×24 view-space froxel grid in compute, then
shade each fragment against only the lights in its cluster.

**Why.** A plain forward loop is `O(fragments × lights)` and collapses past a
handful of lights. The three classic scaling answers are deferred shading, tiled,
and clustered:

- **Deferred** decouples shading from geometry but needs a fat G-buffer, burns
  bandwidth, and makes MSAA and transparency awkward. This renderer is forward
  (HDR `R16G16B16A16`), so deferred would have been a much larger rewrite.
- **Tiled** (2D screen tiles) is simple but over-includes lights along depth
  discontinuities — a tile spanning near and far geometry gathers everything in
  between.
- **Clustered** adds depth slices (the third grid axis), so a light only affects
  the froxels it actually overlaps. It works with forward shading, MSAA, and
  transparency, and reuses the GPU-driven story already in the engine.

**Trade-offs.** A fixed `kMaxLightsPerCluster` (64) caps lights per froxel; the
index list is preallocated at full size rather than globally compacted, trading
memory for zero atomic contention. The froxel grid is rebuilt each frame even
though it only depends on the projection.

**More time.** Shadow-casting point lights (cube/atlas), a global compacted index
list, and only rebuilding froxel AABBs when the projection changes.

**Measured and rejected: culling lights at an "effective" radius.** The cluster
test uses each light's authored range, while the shader's windowed inverse-square
falloff (`rangeFade = 1 - (d/range)^4`, squared) drives the contribution toward
zero well before that. Shrinking the cull radius to where the contribution stops
mattering should therefore shorten every cluster's light list for free -- the
per-light slot load being the documented cost of the loop.

It is not free. Scaling the cull radius uniformly, default scene, control
repeated:

| Cull radius | `MainHDRPass` | Average scene luminance |
| --- | --- | --- |
| 100% | 9.85 / 9.81 ms | 0.3129 / 0.3121 |
| 85% | 9.39 ms | 0.3033 (-3.1%) |
| 70% | 9.37 ms | 0.2651 (-15%) |
| 50% | 8.55 ms | 0.2208 (-29%) |

The image darkens faster than the pass shortens, and the saving stops improving
between 85% and 70%. With two dozen overlapping lights, the aggregate of many
individually-small contributions near the range boundary is a visible fraction of
the image, so there is no threshold that is both worth having and invisible.

The sweep also bounds the whole idea: even at a 50% radius -- throwing away 87%
of the light volume -- `MainHDRPass` only falls 13%. The pass is not dominated by
walking over lights that contribute nothing; it is dominated by shading lights
that do. Fewer PCF taps and a back-face early-out were rejected earlier on the
same grounds.

## Back-face culling is off, and that is measured

**Decision.** Every graphics pipeline uses `VK_CULL_MODE_NONE`.

**Why.** Not an oversight — it was tested. The standard argument for enabling it
is that closed geometry generates twice the fragments without it, and the main
pass is the most expensive shader in the engine. That argument is about
immediate-mode GPUs. This renderer's target is an Apple M3 through MoltenVK,
where the tiler's hidden-surface removal already discards occluded fragments
before the fragment shader runs — and the back faces of closed geometry are
exactly that: occluded. There is nothing left for the cull to save.

Measured on the default scene at render scale 1.0, A/B/A/B:

| Main pipeline cull mode | Frame total | `MainHDRPass` |
| --- | --- | --- |
| `NONE` | 14.848 / 14.712 ms | 10.216 / 9.994 ms |
| `BACK_BIT` | 14.730 / 14.829 ms | 10.058 / 10.098 ms |

Both metrics land inside the control's own run-to-run spread. Scoped honestly:
this is the default scene, which is geometry-light. On a vertex- or
binning-bound scene the answer could differ, since culling removes primitives
before rasterization regardless of HSR.

**Trade-offs.** `Material::doubleSided` stays metadata-only as a consequence
(`Material.h`), and single-sided geometry viewed from behind is shaded with a
normal pointing away from the viewer. With no performance reason to enable
culling, wiring `doubleSided` becomes a pure correctness change that can only
make geometry disappear, so it is a deliberate decision rather than a cleanup.

**More time.** Wire `doubleSided` to the cull mode for correctness on
back-facing single-sided surfaces, and re-measure on a geometry-heavy scene
where the primitive-rate saving, rather than the fragment saving, is the point.

## Graphics pipelines are looked up by state, not by name

**Decision.** `rhi::VulkanPipelineStore` owns every `Renderer`-built graphics
pipeline, keyed on `rhi::PipelineKey` -- an owning, hashable value form of
`VulkanPipelineCreateInfo`. The 13 pipeline members are non-owning
`rhi::PipelineRef`s into it.

**Why.** Declaring pipelines one member at a time gives no way to ask whether the
state a caller wants has already been compiled, and two things followed from that.

Identical state was compiled more than once under different member names. The
punctual shadow atlas, the VSM page pool and the CSM cascades are all
`rhi::VulkanShadowMap`, and every `VulkanShadowMap` resolves its depth format
through the same `chooseShadowMapFormat()` on the same device -- so the formats
are always equal, and the depth-only caster pipelines built against them were
always the same pipeline. A comment in `createVsmPagePipeline` asserted the
opposite for months. Measured on this machine with `--vsm shadows`:

| Configuration | Requests | Pipelines | Shared |
| --- | --- | --- | --- |
| Default (VSM off) | 9 | 8 | `SkinnedShadowPipeline` = `SkinnedPunctualShadowPipeline` |
| `--vsm shadows` | 12 | 9 | the above + `SkinnedVsmPagePipeline`; `PunctualShadowPipeline` = `VsmPagePipeline` |

Five named pipelines resolve to two objects once every optional path is on.

Second, format-change detection was a hand-maintained condition. Each pipeline
kept a shadow copy of its formats (`pipelineColorFormat_`,
`shadowPipelineDepthFormat_`, ...) that `Renderer::pipelineNeedsRecreate` compared
against the live ones. A key subsumes that: any state difference is a miss by
construction rather than by a clause someone remembered to write. (That condition
is still in place -- retiring it needs `PostProcessStack`'s pipelines routed too,
or it would be left half automatic and half hand-written.)

**Normalization is deliberately minimal.** The two failure modes are not
symmetric: treating significant state as irrelevant hands back a pipeline built
for a different configuration, quietly, while treating irrelevant state as
significant costs one extra object. So the key normalizes only what
`VulkanPipeline::create()` proves cannot reach the driver -- the
`colorFormat`/`colorFormats` redundancy, and `depthFormat`/`depthWriteEnable` when
the depth test is off. `depthCompareOp` and the depth-bias factors are kept
verbatim even where they look inert, because `create()` writes them
unconditionally. The `VkPipelineCache` handle is excluded outright: it is a
compile hint, not state.

**Trade-offs.** Keys hold `VkDescriptorSetLayout` handles, which would dangle if a
layout were recreated under a live entry, and nothing in the key could detect it.
The store is therefore reset at the top of `Renderer::createPipeline()` -- already
the one point where every layout, format and shader is re-derived -- so an entry
never outlives a handle it was built from and no eviction policy is needed. A
shared pipeline also keeps the debug name its first requester gave it, so a
capture labels the VSM page pass with the punctual pipeline's name; every
requested name is logged at startup so the sharing is visible rather than
puzzling.

Verified pixel-identical against the pre-change build in both configurations
(0/3686400 pixels differ, max channel delta 0), with no validation errors.

`PostProcessStack` borrows the store the same way it borrows every other service.
Its six graphics pipelines are six distinct fragment shaders, so they add no
sharing -- the point there is single ownership and a single reset, not a collapse.
With them routed the store holds 14 pipelines from 15 requests by default, 15 from
18 under `--vsm shadows`.

**The reset rule needs every ref reissued, and that bit immediately.**
`createProbeCapturePipeline()` was reachable only from the constructor, never from
`createPipeline()`, so the reset destroyed its entry while `probeCapturePipeline_`
still pointed at it -- a dangling pointer introduced by the routing itself, latent
because `pipelineNeedsRecreate` only fires when a format actually changes. It is
registered from `createPipeline()` now. Because that invariant is implicit and the
next pipeline could break it the same way, `PipelineRef` also carries the store's
generation: a ref the rebuild forgot to reissue reads as `VK_NULL_HANDLE` -- the
"feature unavailable" path every call site already handles -- instead of as a
pointer into freed memory. Verified by forcing a second `createPipeline()` after
the probes exist: 19 requests with the fix, 18 without.

**More time.** Route the compute pipelines, then delete `pipelineNeedsRecreate`
entirely -- with the store, an unconditional rebuild is all hits, but only once
`reset()` stops being unconditional (a mark-and-sweep over one build generation
would drop exactly the entries nobody re-requested). Shader hot-reload becomes
tractable after that: reloading is invalidating the entries whose shader path
changed.

## Buffer-device-address for per-frame GPU data

**Decision.** Deliver per-frame buffers (object data, the light list, the cluster
grid + index list) to shaders by pushing their device address in a push constant,
not by binding descriptor sets.

**Why.** Per-frame data needs one copy per frame-in-flight to avoid a frame
writing a buffer another frame is still reading. With descriptor sets that means
either per-frame descriptor sets or `UPDATE_AFTER_BIND` plus careful
synchronization. With BDA the address is just a value set at record time, so
per-frame updates are hazard-free and there is no descriptor churn. The compute
culling passes still use descriptor sets — they fit the bound-resource model and
mirror the existing `cull.comp` setup.

**Trade-offs.** BDA needs `bufferDeviceAddress` and is less tooling-friendly than
named bindings; pointer mistakes surface as GPU faults rather than validation
errors.

## GPU-driven visibility

**Decision.** Cull on the GPU (`cull.comp`) into compacted multi-draw-indirect
commands, with bindless material textures so draws don't rebind per material.

**Why.** It keeps the CPU off the per-object critical path and lets one
`vkCmdDrawIndexedIndirectCount` issue a whole batch. Bindless removes per-draw
descriptor binding, which is what makes a single indirect batch viable.

**Trade-offs.** Indirect + bindless raise the floor of complexity and need
fallbacks (legacy descriptor sets, direct draws) for drivers without indirect
count or descriptor indexing.

## Two-phase Hi-Z occlusion culling

**Decision.** Build a max-depth pyramid and test object AABBs against the
*previous* frame's pyramid, then re-test whatever that rejected against a
pyramid rebuilt from this frame's own depth. On by default.

**Why.** Same-frame occlusion is a chicken-and-egg problem (you need depth to cull
the draws that produce depth). Reprojecting last frame's depth is the standard
conservative compromise, but on its own it drops objects that became visible
this frame -- they pop in a frame late, which is why it started out opt-in. The
second phase re-tests exactly those candidates and rescues them within the same
frame, which is what made it safe to enable by default. See
[gpu_culling.md](gpu_culling.md).

## Render graph with barrier inference

**Decision.** A renderer-owned graph with logical texture/buffer handles and
declared per-pass reads/writes that infers conservative Synchronization2 image
transitions.

**Why.** Hand-written barriers across a dozen passes are where renderers rot. Even
a compact graph that infers the common transitions removes a class of bugs and
documents pass dependencies. It is deliberately *not* an async-compute or
memory-aliasing scheduler — that complexity wasn't worth it for a single-queue
renderer.

**Trade-offs.** A few barriers (shadow culling resets, intra-pass copies, host
readbacks, the clustered build→cull→fragment chain) are still explicit where the
graph's conservative model would be coarser than hand-tuned ones.

## Synchronization2 + Dynamic Rendering

**Decision.** Use `VK_KHR_synchronization2` and dynamic rendering instead of
render passes/framebuffers.

**Why.** Sync2's `VkDependencyInfo` makes barriers far more readable (stage/access
pairs per resource), and dynamic rendering drops the render-pass/framebuffer
boilerplate that doesn't earn its keep outside tiler subpass merging — a good fit
for a desktop/MoltenVK renderer on Vulkan 1.3.

## Histogram auto-exposure on the GPU

**Decision.** Build a luminance histogram in compute, reduce to an exposure value
in a GPU buffer the composite reads directly; the CPU only reads it a frame late
for the debug UI.

**Why.** Keeping exposure on the GPU avoids a stall waiting on a readback in the
hot path; the one-frame-late CPU copy is only for display, so the latency is
invisible.

## RAII RHI + subsystem extraction

**Decision.** Wrap every Vulkan object in a move-only RAII type (`src/rhi/`) and
pull cohesive features into subsystems (`ClusteredLighting`, `PostProcessStack`,
`ScreenshotCapture`) that borrow shared services by reference.

**Why.** RAII makes resource lifetime obvious and exception-safe. Extracting
subsystems keeps `Renderer` an orchestrator instead of a monolith, and gives each
feature a testable seam — `ClusteredLighting` owns its buffers, pipelines, and
descriptors and exposes a small record/update API.

## Testing strategy without a GPU

**Decision.** Put GPU-independent math (cascade fitting, froxel cluster math,
bounds/frustum, settings clamping) in headers the shaders mirror, and unit-test
them headless with Catch2; run the suite under ASan/UBSan in CI.

**Why.** The parts most likely to be wrong — projection/depth-slice math, cluster
indexing, AABB tests — are pure functions. `ClusterGrid.h` is the source of truth
for the grid dimensions and the math; a round-trip test projects a froxel's center
and asserts it maps back to that froxel, cross-checking the build shader against
the fragment lookup. CI can't run a GPU, so this is what keeps the math honest.

**Trade-offs.** The CPU reference and the GLSL are maintained in parallel; the
shared constants and the round-trip test are the guard against drift.
