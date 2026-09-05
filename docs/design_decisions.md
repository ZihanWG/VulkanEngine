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
immediate-mode GPUs. On an Apple M3 through MoltenVK, the tiler's hidden-surface
removal already discards occluded fragments before the fragment shader runs — and
the back faces of closed geometry are exactly that: occluded. There is nothing
left for the cull to save **there**.

> **This conclusion is device-class-specific and does not survive an IMR.**
> Re-measured on an RTX 3080 Ti Laptop (Ampere, no HSR), `--scene stress`,
> A/B/A/B interleaved through `tools/dev/measure_gpu.py`:
> `MainHDRPass` 0.344 → 0.199 ms (**−42%**) against a control drift of 0.003 ms,
> and frame total 1.042 → 0.905 ms (−13%). The saving the tiler made
> unnecessary is fully present here, and it is one of the largest single wins
> available on this hardware.
>
> Enabling it also uncovered a latent bug, since fixed. With auto-exposure and
> SSR disabled to isolate rasterization, the default scene moved 59188 pixels by
> more than 8/255, and the mask landed on *every sphere* plus the alpha-cutout
> panel's holes — never on the boxes, the ground or the pillar. Culling `FRONT`
> instead removed 97.6% of the frame, which confirmed the rest of the scene was
> wound counter-clockwise correctly. Together those said `createUvSphere` emitted
> inverted winding, and `VK_CULL_MODE_NONE` had masked it for the life of the
> renderer: the spheres were rendering their own interiors. The generation moved
> to `renderer/PrimitiveGeometry.h` so it could be asserted on, and
> `tests/test_primitive_geometry.cpp` now fails on that winding. Afterwards the
> same comparison moves 3406 pixels, all of them the cutout panel — which is
> arguably a fix rather than a regression, since through a hole you should see
> the background, not the panel's own back face.
>
> `Material::doubleSided` has since been wired to the cull mode. The main pass
> holds a second pipeline built with `VK_CULL_MODE_NONE`; draw items carry the
> flag, sort on it after bucket, and `buildMeshDrawBatches` breaks a run when it
> changes, because a batch is one indirect draw with one pipeline bound. With
> culling off the two pipeline requests are byte-identical, so the store returns
> one pipeline for both refs and the recorded command stream is unchanged.
>
> It is still off by default, now for a different reason: the flip is not
> pixel-neutral, and `tests/golden/lavapipe_frame30.png` is compared with
> `--max-differing-fraction 0`. All six scene presets render with zero validation
> errors under culling and the captures are visually identical, but 1 to 64 pixels
> of 921600 move per preset (default 40, stress 39, sunlit 64, fragment-stress 0).
> The golden has to be regenerated on lavapipe in the same change, so flipping the
> default belongs in a Linux run. The runtime toggle is
> `renderer.enableBackfaceCulling`.
>
> Re-measured on an RTX 3080 Ti Laptop, `--scene stress`, A/B/A/B x3, control
> drift 0.49% against the 1% limit: `MainHDRPass` 0.321 -> 0.196 ms (-38.9%),
> frame total 1.016 -> 0.942 ms (-7.3%). Three later passes move the other way and
> the gate calls them attributable rather than noise -- `Transparent` +9.0%,
> `SSRTrace` +4.9%, `CSMShadowPass` +7.1% -- so quote the frame total, not the
> pass delta alone.

Measured on an Apple M3, default scene at render scale 1.0, A/B/A/B:

| Main pipeline cull mode | Frame total | `MainHDRPass` |
| --- | --- | --- |
| `NONE` | 14.848 / 14.712 ms | 10.216 / 9.994 ms |
| `BACK_BIT` | 14.730 / 14.829 ms | 10.058 / 10.098 ms |

Both metrics land inside the control's own run-to-run spread. Scoped honestly:
this is the default scene, which is geometry-light. On a vertex- or
binning-bound scene the answer could differ, since culling removes primitives
before rasterization regardless of HSR.

**Trade-offs.** While the toggle is off, single-sided geometry viewed from behind
is still shaded with a normal pointing away from the viewer -- honouring
`doubleSided` changes which pipeline a draw binds, not how a back face is shaded.
The wiring also costs one batch break per bucket where two-sided materials appear,
which the sort key keeps to one rather than one per alternation.

**More time.** Regenerate `tests/golden/lavapipe_frame30.png` on lavapipe and flip
the default in the same change. The geometry-heavy re-measure this line used to ask
for has been done and is quoted above, the UV sphere winding it exposed is fixed,
and `doubleSided` now reaches the cull mode.

## Graphics pipelines are looked up by state, not by name

**Decision.** `rhi::VulkanPipelineStore` owns every `Renderer`-built graphics
pipeline, keyed on `rhi::PipelineKey` -- an owning, hashable value form of
`VulkanPipelineCreateInfo`. Every pipeline member it covers is a non-owning
`rhi::PipelineRef` into it.

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
| Default (VSM off) | 22 | 21 | `SkinnedShadowPipeline` = `SkinnedPunctualShadowPipeline` |
| Bindless off | 19 | 18 | as default |
| `--vsm shadows` | 25 | 22 | the above + `SkinnedVsmPagePipeline`; `PunctualShadowPipeline` = `VsmPagePipeline` |

Five named pipelines resolve to two objects once every optional path is on. The
default configuration understates it because virtual shadow maps ship off, so
four of the caster pipelines are never requested -- `--vsm shadows` is what
exercises them.

Second, format-change detection was a hand-maintained condition. Each pipeline
kept a shadow copy of its formats (`pipelineColorFormat_`,
`shadowPipelineDepthFormat_`, ...) that `Renderer::pipelineNeedsRecreate` compared
against the live ones. A key subsumes that: any state difference is a miss by
construction rather than by a clause someone remembered to write. That condition
is still in place; why it could not simply be deleted once everything was routed
is the last section here.

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

**The store holds compute pipelines too, but the line is lifetime, not type.**
`ComputePipelineKey` is the same idea with three fields -- shader, set layouts,
push ranges -- since a compute pipeline has no raster, blend, depth or attachment
state, and nothing to normalize. What decides whether a pipeline belongs here is
the reset rule: **only a pipeline `createPipeline()` rebuilds can live in the
store**, because a ref the rebuild does not reissue points at a destroyed entry.

By that test seven more pipelines moved in -- SSR's trace, GTAO's main and blur,
the depth pyramid's, and the three exposure compute pipelines -- all of them built
only from `createPipeline()`. The eight compute pipelines owned by
`ClusteredLighting`, `GpuCulling`, `PunctualShadows`, `VolumetricFogPass`,
`VirtualShadowMapPass` and the probe volume stayed where they are: they are
created inside those subsystems' `createResources()` calls, their lifetime is
their subsystem's resources, and a reset would destroy them with nothing to
rebuild them. Routing them "for consistency" would have reproduced the probe-
capture bug eight times.

Under `--vsm shadows` the store now holds 22 pipelines from 25 requests. None of
the seven collapsed, which was the expectation -- every `.comp` and every
post-process fragment shader here is distinct. The gain is one owner, one reset
and one generation, not a saving.

Verified that the reissue rule holds by forcing a second `createPipeline()`: all
25 requests come back.

**Trade-off this introduced.** The exposure fallback paths
(`disableAutoExposureFallback` and friends) used to destroy their pipelines;
clearing a ref now only marks the feature unavailable, and the pipeline is
reclaimed at the next `createPipeline()`. That is a few unused compute pipelines
resident in a configuration where auto-exposure has already failed.

**More time.** Delete `pipelineNeedsRecreate`. With the store an unconditional
rebuild is all hits, but only once `reset()` stops being unconditional -- and the
obvious fix, a mark-and-sweep over one build generation, does not work while some
pipelines are created outside any build. Giving the subsystem-lifetime pipelines
their own scope, or making every pipeline creation funnel through one place, is
the prerequisite. Shader hot-reload becomes tractable after that: reloading is
invalidating the entries whose shader path changed.

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

## Optional device extensions are a table, not an #if

**Decision.** Optional device extensions go through a request table resolved by a
pure function, `selectOptionalExtensions` in `src/rhi/DeviceExtensionSelection.h`,
which returns one outcome per request saying whether it was taken and, if not,
which of the four reasons applied. `VulkanDevice` evaluates whatever
`VkPhysicalDevice*Features` predicate a row needs and passes the answer in as a
bool, so the policy itself names no Vulkan type and is exercised headless by
`tests/test_device_extension_selection.cpp`.

**Why.** The engine requires exactly one device extension -- swapchain -- and had
no machinery for optional ones at all. The single optional extension it did use,
`VK_KHR_portability_subset`, was an `#if defined(__APPLE__)` in the middle of the
list, and every optional *feature* fallback is still hand-written at its call
site. That shape does not extend: a ray query path for probe capture pulls in
`VK_KHR_acceleration_structure`, which pulls in `VK_KHR_deferred_host_operations`,
and enabling a dependent whose dependency was refused is a validation error rather
than a degraded path. Getting that wrong is silent on the machine that happens to
have all three.

So dependency refusal propagates, and it does so through a fixed point rather
than assuming the table is topologically sorted -- for the same reason
`TransientMemoryPlan` sorts its tie-break by name: a policy whose answer depends
on how the table happens to be written is one that changes when somebody regroups
it for readability. There is a test that runs the same table forwards and
backwards and asserts the decisions match.

Moving portability_subset onto the table also made it slightly more correct. The
spec requires it to be enabled whenever the device exposes it, which is a property
of the driver, not of the host OS; asking the device instead of the compiler
covers a layered implementation on a non-Apple host, which the `#if` did not.

The outcomes feed a startup capability report that replaced five scattered log
lines. Each row names the path that is actually live, because "descriptor indexing
is not fully supported" answers only half of what a bug report needs -- the other
half is what the renderer does instead:

```
Device capability report:
  bindless material textures : on  -- descriptor indexing, with update-after-bind
  multi-draw indirect        : on  -- object data indexed by firstInstance
  indirect draw count        : on  -- vkCmdDrawIndexedIndirectCount, max 4294967295
  layered cascade shadows    : on  -- multiview, one pass for all cascades
  independent blend          : on  -- per-attachment blend state
  async compute queue        : on  -- dedicated compute-only family 2
  transfer queue             : on  -- dedicated DMA family 1
  VK_KHR_portability_subset  : off -- not exposed by the device; required by portability-layer drivers such as MoltenVK
```

A single `Logger::warn` follows when any row is off, so a degraded run stays
greppable at one level the way the individual warn lines used to be.

**Trade-offs.** Nine capability booleans still live as separate members and
getters on `VulkanDevice`, so adding a *feature* fallback is still three edits.
Only the extension half is table-driven. The report is also the third place a
capability is spelled out, after the member and the getter, and nothing enforces
that a new capability gets a row.

**More time.** Fold the nine booleans into one `DeviceCapabilities` struct the
report iterates, so a capability that is negotiated but not reported becomes a
compile error rather than an omission. Timeline semaphores deliberately do not
belong in this table: they are a mandatory Vulkan 1.2 core feature and the device
already requires 1.3, so they need enabling, not negotiating.

## Frame pacing is a timeline semaphore, not a fence per slot

**Decision.** One device-level timeline semaphore counts submitted frames. Every
graphics submit signals the next value; each frame slot remembers the value its
own last submit will signal, and waiting for that value is what used to be
`vkWaitForFences` on that slot's fence. `imagesInFlight_` holds timeline values
instead of fence handles. There are no fences left in the frame path.

Acquire and present keep binary semaphores, because `VK_KHR_swapchain` accepts
nothing else on either side. The per-swapchain-image `renderFinished` semaphore
and the reason it is owned per image rather than per frame slot are unchanged.

`timelineSemaphore` is enabled unconditionally and is deliberately not a row in
the optional-extension table: it is a mandatory Vulkan 1.2 core feature and
`isDeviceSuitable` already refuses anything below 1.3, so no device reaching that
line could answer no. A fallback there would be unreachable code pretending to be
caution.

**Why.** The fence-per-slot model is coarse: a fence answers one question, for
one slot, to the host only. A timeline value is meaningful to anything that needs
to know how far the GPU has got -- another queue, an upload path, a frame limiter
-- which is what makes multi-queue scheduling in the render graph and any future
parallel recording expressible rather than bolted on.

It also removed an ordering hazard rather than moving it. The old code had to
reset the slot's fence *after* a successful acquire, because resetting before an
out-of-date early return would leave the fence unsignalled with no submission
that could ever signal it. The timeline version publishes the slot's value only
after the submit returns, so a frame that returns early or throws leaves the slot
on its previous, already-signalled value. Zero means "never submitted" and a wait
for zero returns immediately, which is what `VK_FENCE_CREATE_SIGNALED_BIT` used
to buy.

**Verification.** All six scene presets, 60 frames each with
`--fail-on-validation-error`: zero validation errors and captures bit-identical
to the fence version. A 900-frame soak on `--scene stress` likewise.

**Trade-offs.** The frame timeline is never reset, including across swapchain
recreation -- a timeline that went backwards would make an older recorded value
look unreachable forever. That is correct but it means the value is a
process-lifetime counter, so anything that ever wants to persist or compare it
across a device loss has to account for that.

## Frames in flight is a setting, and its ceiling is TAA

**Decision.** `renderer.framesInFlight` sizes `frames_`, clamped to
`[1, kMaxFramesInFlight]` by `clampFramesInFlight`, a pure function separate from
`clampRuntimeSettings` because it is startup-only -- it is read once, before the
renderer has frames to clamp anything against, while `clampRuntimeSettings` runs
after every live edit. Default 2, unchanged.

**Why the ceiling is 3.** Not hardware. The TAA history ping-pong has two slots
advanced by their own counter, independent of the frame slot, so beyond history
slots plus one a frame could rewrite an image an older in-flight frame still
reads. `tests/test_settings_clamp.cpp` pins the relationship so raising the
ceiling without adding history slots has to argue with a test.

**Verification.** 1, 2 and 3 all run 120 frames clean under validation. With
auto-exposure off, all three produce **bit-identical** frames -- so the frame
count changes no rendering. With auto-exposure on they differ, by a peak channel
delta of 9 at one frame and 4 at three against the two-frame result on
`--scene default` at frame 30. That is the exposure readback lagging by the slot
count and the adaptation converging along a different path, not a correctness
problem; the same reasoning applies to the GPU-culling, occlusion-yield and VSM
page-request readbacks, which all lag by this value.

**Trade-offs.** Every per-frame resource is sized off `frames_.size()`, so 3
costs a third more of every per-frame buffer, query pool and descriptor set, and
buys nothing measurable on a scene that is not CPU-starved. 1 is the useful
non-default: it serializes the frame, which is what a latency measurement or a
capture that must not overlap frames wants.

**More time.** Give TAA as many history slots as there are frames in flight, and
the ceiling stops being a TAA number. Until then, note that any A/B that changes
this value has to pin exposure first, or it will be reading adaptation, not the
thing it meant to measure.

## Parallel command recording was measured before it was built, and the measurement redirected it

**Decision.** Recording stays single-threaded. What shipped instead is the
instrumentation that answers whether it should be: `Record CPU` next to the
existing `Frame prep CPU` in the timings block, and a `Record CPU by unit`
breakdown logged separately.

**Why.** The case for threading recording rested on a number nobody had. Having
it changed the shape of the answer twice.

First, recording turned out to be worth attacking. On an RTX 3080 Ti Laptop,
Release:

| scene | GPU frame | prep CPU | record CPU | record share of CPU |
| --- | --- | --- | --- | --- |
| `stress` | 1.024 ms | 1.079 ms | 0.530 ms | 33% |
| `default` | 1.754 ms | 0.059 ms | 0.150 ms | 72% |
| `fragment-stress` | 2.149 ms | 0.089 ms | 0.182 ms | 67% |

`stress` is the interesting row: CPU work totals 1.6 ms against a 1.0 ms GPU
frame, so that scene is CPU-bound and recording is a third of it. The other two
are comfortably GPU-bound, where removing CPU recording time buys nothing.

Second, and decisively, the per-unit breakdown showed the cost is not spread
across the seventeen scheduled units. One unit held 63% of it and the rest were
under 0.015 ms each. Splitting seventeen recorders into primary/secondary halves
would have parallelised a workload that is nearly serial by construction --
Amdahl caps the win at about 1.6x on recording, for a refactor touching every
recorder in the frame.

**What the number actually pointed at.** The expensive unit is the punctual
shadow atlas, not the cascade pass -- `renderPassTypeName` returns "Shadow" for
both, which is why the breakdown now carries the unit index as well. Its cost is
the CPU caster cull, and `punctualShadows.gpuCasterCulling` already exists to
remove it: 0.33 -> 0.02 ms on that unit, 0.52 -> 0.21 ms on the frame's recording
total. One settings toggle, against a seventeen-recorder refactor, for more than
threading could have delivered.

That toggle is off for a documented reason, and re-reading it is what closed the
loop: `docs/punctual_shadows.md` says it pays only with enough draw items and a
platform that has `vkCmdDrawIndexedIndirectCount`, and names both as false where
it was measured -- the demo scene, on MoltenVK. Both are true here.

The same page records a blended-caster trap, and that page was out of date: the
mask it asks for already exists, in a side buffer rather than inside the 64-byte
cull record. What actually blocked the flip was a different bug, since found and
fixed: the punctual indirect path bound one mesh -- `allDrawItems_.front().mesh`
-- for a whole slot and drew every surviving caster in that slot against it, so
casters from any other mesh indexed into the wrong buffers. The culling agreed
exactly between the two paths; only the draw did not. `stress` had more
mesh-spanning slots than `default` and still rendered identically, which is what
made it the kind of bug that survives testing.

The fix stopped compacting survivors with an atomic and writes each one in place,
which restores the contiguous per-batch runs the replay needs to rebind per mesh.
GPU culling is now bit-identical to CPU culling on all six presets, and the CPU
recording win is unchanged by the extra per-batch draw calls.

In-place writing fixed the geometry but gave up compaction, so every slot then
submitted each batch's whole range instead of just its survivors -- about 58k
mostly-zeroed commands a frame on `stress`, and the atlas GPU pass moved 0.030 ->
0.325 ms. Compacting per (slot, mesh batch) gets both: a batch's region stays a
contiguous run of one mesh, so the replay still rebinds, and the per-batch
survivor count is what `vkCmdDrawIndexedIndirectCount` consumes. With that, the
atlas pass sits inside its own drift again and the residual GPU cost is the
~0.017 ms cull dispatch.

The default is still off. Five A/B attempts were refused by the drift gate (best
1.4%), so the frame total remains unmeasured on this laptop, and a trade that
looks like +0.017 ms GPU against -0.31 ms CPU is still not a number. Flipping it
needs a gate-passing run on a machine with a stable clock and a golden image
regenerated on lavapipe.

The GPU half of the trade is not settled either. Two A/B attempts were refused by
the drift gate at 3.9% and 7.6%, so nothing here is quotable; both pointed the
same way the original MoltenVK measurement did, which is enough to say the CPU
win cannot be banked on its own.

**Trade-offs.** Two `steady_clock` reads per unit per frame, ~34 calls, which is
noise at ~25ns each. The breakdown is logged outside the `GPU timings:` block on
purpose: `tools/dev/measure_gpu.py` treats every two-space indented line after
that header as a GPU scope, so CPU numbers left inside it get parsed and quoted
as pass timings.

**More time.** Add a per-draw-item caster mask so the indirect path can reject
blended casters, re-measure, and then revisit the `gpuCasterCulling` default.
Parallel recording only becomes the right lever after that, because only then is
the remaining recording cost spread widely enough for threading to reach it.

## Asynchronous pipeline compilation, measured and not taken

**Decision.** Pipeline creation stays synchronous and `VulkanPipelineStore` stays
single-threaded and lock-free. What shipped is the number that decides it: a
`Pipeline build: N ms` line next to the store contents.

**Why.** Async PSO compilation exists to solve two problems, and this engine has
neither at a size worth paying for.

*Startup.* On an RTX 3080 Ti Laptop, Release, timed around `Renderer::createPipeline()`:

| | Pipeline build |
| --- | --- |
| warm `VkPipelineCache` | 8.3 / 8.5 / 8.9 ms |
| cold cache (deleted before each run) | 28.1 / 28.4 / 28.6 ms |

Total process wall time for `--exit-after-frames 1` is 700-830 ms. So pipeline
building is **1.1% of startup warm and about 3.5% cold**. A perfect N-way split
would recover at most ~20 ms of a cold start, and not even that: every create
shares one `VkPipelineCache`, which the implementation is allowed to lock
internally because the cache is not created with
`VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT`, so parallel creates can
serialise inside the driver regardless.

*Runtime hitches.* This is the real reason engines compile asynchronously, and it
does not arise here. `Renderer::createPipeline()` has exactly two callers: startup,
and `recreateSwapchain` behind a `pipelineNeedsRecreate` test that only fires when
an attachment format changes or a handle is null. A plain window resize does not
rebuild anything, and the settings that would change pipeline state --
`enableBackfaceCulling`, `enableBindlessMaterialTextures`, `enableLayeredCascades`
-- are startup-only by design, precisely so they cannot. The engine effectively
never creates a pipeline mid-run.

**Specialization constants were considered with it, and rejected for a different
reason.** `simple_bindless.frag` is 1182 lines but has only eight
push-constant-driven branches, five of which are debug views that are off in a
normal frame. The three functional ones -- ambient occlusion, clustered versus
brute-force lighting, and fog -- are *uniform* branches, the cheap kind, and
turning them into specialization constants would multiply the main pipeline into
eight permutations. The plausible win is register pressure rather than branching,
which needs occupancy tooling to see, and the loss is that this repo would acquire
the shader permutation system that `docs/asset_system.md` and
`src/rhi/PipelineKey.h` both record as a deliberate absence. Eight permutations for
three uniform branches, with no way to measure the result on this machine, is not
a trade to make on reasoning alone.

**Trade-offs.** A cold start still pays ~28 ms on one thread, and a future format
change still rebuilds every pipeline synchronously. Both are accepted.

**More time.** Revisit if either premise changes: a permutation system that makes
the pipeline count grow by an order of magnitude, or a code path that creates
pipelines during play. The `Pipeline build` line is there to notice the first one
happening.

## Shader hot reload watches the build output, not the GLSL

**Decision.** `renderer.enableShaderHotReload` polls the compiled SPIR-V directory
once a second and, when its digest changes, does `waitIdle()` + `createPipeline()`.
Off by default; runtime-toggleable, since it owns no resource.

**Why the build output.** Shaders are compiled by CMake through `glslc`, so the
workflow is already edit, build the shader target, look at the window -- and the
only step missing was the last one. Watching the `.spv` files means no runtime
shader compiler, no second toolchain, and a reload that runs exactly the modules
the next launch would have loaded. The digest is `rhi::hashShaderDirectory`, the
same function the persisted `VkPipelineCache` is keyed on, so "the shaders
changed" has one definition here rather than two that can disagree.

**Two polls, not one.** `glslc` writes the modules one at a time, so a digest
taken while a build is running describes a directory that is part old and part
new. Acting on it would compile a module that is about to be replaced, or a
truncated one. Requiring the same new digest on two consecutive polls means the
directory stopped changing for at least a second first.

**Verified.** With the toggle on, adding a module to the shader directory and then
removing it produced two detections and two clean rebuilds while the engine kept
running -- "Shader directory changed" followed by "Shader hot reload complete"
each time. With the toggle off, all six scene presets are bit-identical to before
the feature existed.

**Trade-offs.** A reload is a full `waitIdle()` and a full rebuild, about 17 ms on
this machine, so it is a visible hitch rather than a seamless swap; for a
development aid that is the right trade against tracking which pipelines a given
module feeds. And `createPipeline()` resets the store before rebuilding, so a
shader that fails to load leaves the frame empty until a good build lands. The
digest is deliberately not advanced on failure so the next good build recovers,
and the error says so rather than letting a black screen speak for it.

## Bindless materials share one environment set, and the texture cap was self-imposed

**Decision.** While bindless material textures are active, every material points at
one shared descriptor set instead of allocating its own copy, and the heap reserves
4096 descriptors per texture class rather than 256.

**Why one set.** Set 0 has fourteen bindings, and only three of them --  base
colour, normal, metallic-roughness -- are the material's own. The other eleven are
global: the cascade array, the punctual atlas, the IBL cubemaps, the BRDF LUT, the
fog volume, the probe atlases and their parameter buffer, ambient occlusion, and
the VSM page pool. Under bindless the shader samples the three from the heap in
set 1, so every material's set was a copy of the same eleven descriptors -- and the
recorder only ever bound one of them anyway, `globalMaterialDescriptorSet()`
returning `materialVariants_.front().descriptorSet`.

So this changes nothing observable, which is exactly why it is safe: all six scene
presets are bit-identical afterwards. What it changes is that the sharing is now a
fact rather than an accident. Concretely, on `--scene stress` with sixteen
materials:

```
bindless on : Material descriptor sets allocated: 1 (cap 256, bindless: one shared environment set)
bindless off: Material descriptor sets allocated: 16 (cap 256, per material)
```

The count is logged after `createScene()` because that is the one point where every
startup material exists, and because "every material shares one set" should be
checkable from a log rather than inferred from the code. Without bindless nothing
changes: bindings 0/2/3 are live per material there, and sharing would hand every
material the first one's textures.

It also lifts the material count off `kMaxMaterialDescriptorSets`. A scene with more
than 256 materials used to fail to allocate; under bindless it now cannot.

**Why 4096.** The 256 was self-imposed, not a device limit. `create()` already
queries the per-class descriptor budget, clamps to it, and warns -- and on an
RTX 3080 Ti with update-after-bind that budget reads **349514**, so the default was
leaving three orders of magnitude unused while `registerTexture` throws the moment
a scene needs one more. glTF imports are user-supplied, so that cliff is reachable
by an asset rather than only by a code change. 4096 costs 3 x 4096 slots in one
pool, covers imports far larger than anything this repo ships, and devices that
cannot afford it are protected by the clamp that was already there. The final
capacity and the queried budget are now both logged, so the headroom on a given
device is visible before an import discovers it.

**What was considered and left out.** Two other gaps in the same area have no
present-tense justification, and building for them would be speculative:

- *A free list for texture indices.* Indices are handed out with `nextIndex++` and
  never returned. But nothing in the engine unregisters a texture -- there is no
  `unregisterTexture`, and no caller that would want one. A free list would be
  machinery with no user until texture streaming or scene unloading exists, and
  those are the changes that should bring it.
- *Widening the heap's stage flags.* The layout is `VK_SHADER_STAGE_FRAGMENT_BIT`,
  and all three consumers -- `simple_bindless.frag`, `shadow_masked.frag`,
  `probe_capture.frag` -- are fragment shaders. Adding vertex or compute visibility
  would be for no reader.

**Trade-offs.** The shared set is allocated by whichever material is built first
and then reused, so it carries that material's base colour, normal and
metallic-roughness in bindings 0/2/3. Nothing samples them under bindless, but they
are not meaningless-looking either, which could mislead someone reading a capture.
The resize refresh (`refreshMaterialAmbientOcclusionDescriptors`) now writes the
shared set once instead of walking every material, which is the same descriptor it
was writing N times before.

**More time.** Bindless still covers only material textures: buffers are reached
through buffer-device-address push constants rather than a descriptor heap, and the
three fixed arrays are three bindings rather than one heap indexed by a type tag.
Neither is costing anything measurable today.
