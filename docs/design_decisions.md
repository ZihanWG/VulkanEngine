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
