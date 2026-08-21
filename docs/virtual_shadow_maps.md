# Virtual Shadow Maps

Directional-light shadows as a clipmap of fixed-size pages, rendered on demand
and cached per page, instead of four camera-fitted cascades.

**Off by default, and the cascades keep rendering underneath even when it is
on**, so the fallback is always one frame away. Three nested toggles, each an
A/B point:

| stage | setting | what it does |
| --- | --- | --- |
| marking | `enableMarking` | **startup-only** — allocates the subsystem, and counts the pages the frame needs |
| rendering | `enablePageRendering` | allocates a physical page per request and draws it |
| sampling | `enableShadows` | samples the pool instead of the cascades |

Building it in that order is what let three design errors surface before anything
downstream depended on them — see
[What the measurement changed](#what-the-measurement-changed).

> **Spot-checked by eye, not gated.** A side-by-side A/B on the default scene
> showed no obvious breakage — no holes, no seams, no acne — and the residency
> counters read as designed. That is a spot check on one static camera with much
> of the shadowed ground hidden behind the debug window, not a verification: see
> [Why there is no pixel gate](#why-there-is-no-pixel-gate).

## Why, given the cascades already cache

[`docs/render_scale.md`](render_scale.md) measures `CSMShadowPass` at 0.44-0.45 ms
once the cascade culls were collapsed into one dispatch, and
[`docs/design_decisions.md`](design_decisions.md) records that `MainHDRPass` is
dominated by shading punctual lights rather than by walking over them — fewer PCF
taps having already been measured and rejected there. Shadow *rendering* is not
the problem on this renderer; shadow *sampling* and shadow *resolution* are.

So the reason to want a VSM here is resolution and stability — a clipmap puts
millimetre-scale texels near the camera and removes the cascade split entirely —
not frame time. No performance claim appears in this document, because none has
been measured.

## The model

The light's XY plane is tiled by an **absolute** grid of square pages, one grid
per clipmap level, with level *L*'s pages twice the world size of level *L-1*'s.

A page's world rect depends only on the light basis and the level — never on the
camera. That is the whole reason page caching can work. A clipmap centred on the
camera and snapped to texels would give every page a slightly different world
rect every frame, so every page's contents would change every frame and nothing
would ever be reusable.

The camera decides which finite **window** of that infinite grid is addressable:
`kVsmPagesPerLevelAxis` squared, centred on the camera's own page. A page's slot
inside the window is its absolute coordinate taken **modulo** the axis, not its
offset from the window's corner. Window-relative indexing would renumber all 256
slots the moment the window scrolled by one page; wrapping renumbers only the row
or column that actually changed identity.

| constant | value | what it fixes |
| --- | --- | --- |
| `kVsmPageSize` | 128 | texels per page edge |
| `kVsmPagesPerLevelAxis` | 16 | pages per level edge — a **coverage** constant, see below |
| `kVsmMaxClipmapLevels` | 12 | ~8 km of reach at a 4 m level-0 extent |
| `kVsmPagePoolSize` | 4096 | physical pool edge, 1024 pages |
| `kMaxVsmPagesPerFrame` | 128 | pages the render phase may draw per frame |

The virtual set (12 x 256 = 3072 pages) is deliberately **larger** than the pool
(1024), so the page table is a real indirection and the rendering phase needs a
residency policy. What makes that tractable is the measurement below: the
resident set is around 100 pages, so an allocator over 1024 slots is never under
pressure.

## The page table checks identity, not just residency

`VsmPageTableEntry` records the **absolute coordinates** a physical page
currently holds, not merely that it holds something. That field is the one doing
the work.

A slot is a toroidal wrap, so when the window scrolls the same slot comes to name
a different absolute page. An entry recording only "resident" would then hand a
reader depth that was rendered for somewhere else — a wrong shadow that looks
plausible, in a place unrelated to the scroll that caused it. Storing the
identity lets every reader check it, which makes scroll invalidation free instead
of a CPU sweep that has to be kept in step with the window.

`rendered` is separate from residency for the same class of reason. Allocation
and drawing are different steps: a page can own physical space for a frame before
its depth exists (the per-frame draw budget, or a scroll that reused the slot),
and sampling an allocated-but-undrawn page is sampling whatever the previous
occupant left there. It is set only after the draws are actually in the command
buffer.

What the identity check **cannot** catch is a change that leaves every page's
identity intact while changing what its depth should contain: the light moving,
the clipmap settings changing, a scene switch. There is no single edit site to
hook for the first of those — the debug UI drags the light direction every frame
— so `updateResidency` hashes its own inputs and drops residency when they move,
the same way the cascade cache does rather than tracking dirty flags.

## A caster that moves has to dirty its own pages

The same blind spot, one level down and worse: an object moves, and every page
its shadow is in keeps both its coordinates and its physical page. Nothing about
the page changed, so nothing invalidates it, and its depth silently goes on
describing where the object used to be.

The cascades never had this problem — `CascadeShadowCaster` hashes every caster's
model matrix into the cascade key, so any movement re-renders the cascade. Doing
the same per page would mean a per-page caster list on the CPU, which is 100
pages times the draw-item count every frame.

Instead the renderer works from the objects. Each frame it builds one
`ShadowCacheKey` per object — the model matrix, then the mesh, material, index
range and bucket of every draw item belonging to it — and compares it against
what that object contributed last time. For each object whose key moved, it drops
the depth of every addressable page overlapping **both** its previous and its
current world bounds: the old pages still hold its shadow, and the new ones do
not hold it yet. Cost is proportional to what actually changed, which on a static
scene is nothing at all.

Three details that are load-bearing:

- **The model matrix is hashed, not the world bounds.** Rotating a symmetric
  object leaves its AABB identical while changing every shadow it casts.
- **Every level is invalidated, not just the object's own.** The marking pass
  requests one level coarser as a fallback, so an object's shadow can be resident
  at more than one level at once; dropping only the finest would leave a stale
  coarse copy for the sampling walk to find.
- **Mesh and material are hashed by pointer**, which is unique only inside one
  scene. `resetSceneState` clears the per-object keys for the same reason it
  clears the cascade and atlas caches — the buffers behind those addresses are
  replaced wholesale, so a reused address would compare equal to something
  unrelated.

Blended geometry is skipped: the page cull filters it out before anything is
drawn, so a change to it cannot change any page.

## Rendering the dirty pages

`VsmPagePass` is shaped exactly like the punctual shadow atlas pass, because it
is the same problem: one dynamic-rendering scope over the whole pool, a
`vkCmdClearAttachments` rect per dirty page, then a viewport, a scissor and one
indirect draw per page. A single scope is what lets the untouched pages keep
their depth; per-page passes would be a command encoder each, which is the cost
[`docs/render_scale.md`](render_scale.md) already measured for the cascades.

It reuses `shadow_punctual.vert` and its push block outright. A VSM page and an
atlas tile are the same operation — one depth-only draw of a rect with that
rect's own projection pushed — so the only reason the pipeline is separate is the
pool's own depth format and bias.

Caster culling is one compute dispatch over every (dirty page, draw item) pair,
recorded before the scope opens: compute cannot run inside a rendering scope, and
the scope has to stay single. Each page compacts its survivors into its own
region of the indirect buffer, capped at `kMaxVsmCastersPerPage`. Going over the
cap is **counted and reported**, not silently dropped — the counter keeps
climbing past the cap so the readback shows real demand rather than a page that
merely looks full.

## Page marking reads the *previous* frame's depth

Shadows have to be rendered before the main pass, but "which pages are needed" is
a question about this frame's visible surfaces. This renderer has no depth
prepass, so `vsm_page_mark.comp` reads the depth the Hi-Z pyramid already holds,
paired with the view-projection stored when that pyramid was built — exactly the
pairing the phase-1 occlusion test in [`cull.comp`](../src/shaders/cull.comp)
already relies on.

It samples **mip 0**, which is a 1:1 copy of the main depth buffer, and takes the
minimum over each block of `markBlockStride` pixels. Mip 0 rather than a reduced
level because the deeper mips store the *maximum* depth of their footprint, and a
farther depth reconstructs to a farther surface, which selects a *coarser* level
than the nearest surface in that footprint needs. Under-requesting resolution is
the one error this pass must not make.

The cost is that the request set lags the camera by a frame. That does not
produce holes: clipmap levels nest, so a page that is not resident yet falls back
to a coarser level that is. The marking pass also marks one level coarser than
the one it selected, which is what makes that fallback a guarantee rather than a
hope.

### The pyramid now has two consumers

`isDepthPyramidBuildRequired()` used to be "is Hi-Z occlusion culling on", and
the occlusion-yield controller was free to suspend the build when occlusion was
culling nothing. Page marking is a second consumer that the yield decision knows
nothing about, and a suspended build calls `depthPyramid_.invalidate()`.

Left unhandled this is not subtle but it is quiet: marking measures one frame
correctly and then reports **zero** forever. The predicate now returns true
whenever marking is active, independently of the yield state.

## What the measurement changed

Measured on the M3 at 1080p with validation layers on, cameras left at their
preset defaults (`--scene`), reading the once-per-second report block. The counts
were identical across every block of each run rather than merely close, which is
what a static camera should produce and is itself a check that the request set is
stable frame to frame.

The first design used `kVsmPagesPerLevelAxis = 8`. It produced this:

| scene | texels/pixel | requested pages |
| --- | --- | --- |
| default | 1.0 | 42 |
| geometry stress | 1.0 | 38 |
| geometry stress | 0.25 | **0** |

Zero. Not a blurrier shadow — no shadow at all. Every page the finer request
selected fell outside its own level's window, because window size scales with the
selected level and a *finer* level has a *smaller* window.

The relation the constant has to satisfy: a point at distance *d* selects a level
whose texel is about `d / projScaleY` across, so that level's window spans about
`(axis/2) * kVsmPageSize * d / projScaleY`. For the point to fall inside the
window that selected it,

```
axis >= 2 * projScaleY / kVsmPageSize
```

which is ~12 at 1080p. Hence `kVsmPagesPerLevelAxis = 16`. And because a constant
that has to be right for a resolution is a constant that will be wrong at some
other resolution, `vsmSelectLevel` now takes the **maximum of the quality level
and `vsmMinLevelForCoverage`** — a level finer than its window can reach has no
slot for the point at all, so quality may only ever ask for something coarser.

After both changes:

| scene | texels/pixel | requested pages | levels used |
| --- | --- | --- | --- |
| default | 1.0 | **99** | L1..L5 |
| geometry stress | 0.25 | **60** | L4..L6 |
| geometry stress | 1.0 | **60** | L4..L6 |
| geometry stress | 4.0 | **25** | L5..L7 |

Two things fall out of this table.

**Page count tracks the screen, not the scene.** 2322 draw items ask for fewer
pages than 11 do — the stress camera is simply further back. This is the expected
behaviour of a screen-driven page set, and it is now measured rather than assumed.

**Level 0 is unreachable at ordinary camera distances, and that is the same
finding.** The default scene reports `L0=0`, `Levels touched: 1..5`: at a 4 m
level-0 extent, that level's window reaches only 7 x 0.25 = 1.75 m from the
camera, so the coverage bound rules it out before quality is consulted. The
`Finest texel: 0.0020 m` the panel advertises is real but unusable unless
something is within 1.75 m. Turning `level0Extent` down makes that worse, not
better.

**Requesting finer than the window can deliver changes nothing.** Stress at 0.25
and at 1.0 return identical counts because coverage binds at every pixel there.
Substituting the coverage relation, the level it forces still delivers about
`projScaleY / ((axis/2 - 1) * kVsmPageSize)` = 771 / (7 x 128) ~ 0.86 texels per
pixel at 1080p — good quality, but it means **effective resolution is bounded by
`kVsmPagesPerLevelAxis`, not by `level0Extent`**. Turning `level0Extent` down
past that point buys nothing.

`kMaxVsmPagesPerFrame` was raised from 64 to 128 on the strength of the 99. A
budget under the resident set is not wrong — caching is what keeps the per-frame
redraw far below it — but a cold start has to fill the whole set, and at 64 that
takes two frames of visibly wrong shadow.

### The cache holds

With page rendering on and a static camera, the default scene reports:

```
requested pages: 99   addressable now: 99/99
resident: 99/1024     cached this frame: 99
drawn this frame: 0/128 (99 since start)
over budget 0, refused 0, evicted 0, casters over the per-page cap: 0
```

99 pages drawn once at startup, then nothing. That is the absolute page grid
paying off: every page's world rect is unchanged frame to frame, so every entry's
identity still matches and no page needs redrawing.

The cumulative "since start" count is load-bearing in that readout. A warmed-up
clipmap draws nothing, so `drawn this frame: 0` on its own cannot distinguish a
working cache from a page pass that never ran at all — which is exactly the state
the first version of this was accidentally in, because the page pipeline was
created before the pool existed and was silently skipped.

### Invalidation, and the first real page-pass cost

No scriptable scene animates a caster — only the demo lights move, and lights are
not casters — so the path was exercised by temporarily patching one object to
drift a hundredth of a unit per frame. That object turned out to be the ground
plane, which makes it close to a worst case: its footprint covers most of the
resident set.

```
casters changed: 1     pages they invalidated: 84
resident: 100/1024     cached this frame: 14
drawn this frame: 84/128     over budget 0, refused 0, evicted 0
```

84 invalidated, 84 redrawn, the remaining 14 still cached. The numbers matching
exactly is the check: every page the object dropped is a page the next update
queued, and nothing else moved.

That run is also the first time the page pass had real work, so it is the first
cost figure for it — from a deliberately pathological load, one run, n=6 report
blocks, **no A/B and no repeated control**, so treat it as an order of magnitude
and not as a measurement:

| pass | median |
| --- | --- |
| `VsmPageMark` | 0.32 ms |
| `VsmPageCull` | 0.04 ms |
| `VsmPagePass` (84 pages) | 0.70 ms |
| `CSMShadowPass` (4 cascades, same frame) | 0.19 ms |

The designed steady state is the other end of that range: zero pages redrawn, so
only the marking pass runs at all. What the experiment bounds is the cost when
the scene fights the cache — a ground plane moving every frame invalidates its
whole footprint, and there is no cache design that avoids that.

### The marking pass was reading every pixel

`VsmPageMark` measured **0.90 ms** on a 15 ms frame, for a pass whose entire
output is a few hundred bytes. Raising `markBlockStride` from 4 to 16 only moved
it from 0.90 to 0.67 ms, which is the giveaway: the cost was proportional to the
frame's pixel count, not to the thread count, because each thread looped over
every pixel of its block.

Capping the block scan at 2x2 taps decouples the two:

| | stride 4 | stride 8 |
| --- | --- | --- |
| whole-block scan | 0.90 ms | 0.72 ms |
| 2x2 taps | 0.73 ms | **0.41 ms** |

Page counts were **identical** at every one of those points — 99 on the default
scene, 60 on geometry stress — so the default stride moved to 8. The taps make
"nearest surface in this block" into "nearest of a few samples", which can miss a
thin nearby surface and pick one level coarser; that is the same graceful
degradation an unresident page already falls back to, and it loses resolution
rather than correctness.

## Everything is allocated only if it was on at startup

The page pool is 4096x4096 D32 = **64 MiB**, plus ~2 MiB of per-frame cull
buffers and a 144 KB page table. Holding that for a subsystem that is off by
default would be a worse trade than the 17.48 MiB of bloom aliasing this project
already [measured and rejected](design_decisions.md) as a default.

So `enableMarking` is startup-only, like `CsmSettings::cascadeCount` and
`enableLayeredCascades`, and for the same kind of reason: allocating the pool on
a runtime toggle would mean recreating the page pipeline (it bakes the pool's
depth format), and there is no device-idle wait available in the steady-state
frame path. With it off, none of the pool, the cull buffers, the page table or
the marking resources exist — measured: no VSM log lines, no allocations, no
behaviour change at all. The two toggles below it stay live.

## Sampling: walk up until something is resident

`vsmShadowFactor` in
[`virtual_shadow_map.glsl`](../src/shaders/virtual_shadow_map.glsl) takes a world
position, applies the normal offset, projects into light space, and then walks
from the finest plausible level up to the coarsest, taking the first level that
is all three of: inside this frame's window, marked `rendered`, and holding the
absolute page it claims to. The walk **is** the fallback that makes the request
latency survivable — a fine page may not be resident yet, but clipmap levels
nest, so a coarser one covering the same world is. The result degrades to a
blurrier shadow, never to a hole. Nothing resident at all returns fully lit,
which is what the cascades do outside their range.

The starting level is `vsmMinLevelForCoverage`, not the full selection. It only
has to be a lower bound: the walk finds the level the marking pass actually
chose, so a start that is too fine costs a few failed table reads and never a
wrong answer — and it saves carrying `projScaleY` into the fragment stage.

Every PCF tap is clamped inside its own page before it reaches the sampler.
Neighbouring pool texels belong to a different page — a different world location,
possibly a different clipmap level — so a tap that walks out compares against
unrelated depth. The punctual atlas learned this the same way, and its
[notes](punctual_shadows.md) describe the seams it drew.

VSM **replaces** the cascades rather than blending with them. The two disagree
about texel size everywhere, so a cross-fade would show the disagreement as a
seam instead of hiding it.

The fragment shader reads `FrameConstants` directly through the push block's
`frameConstantsAddress`, at the offset the vertex stage already uses. The
alternative was seven more `flat` varyings carrying a mat4 and three vec4s that
are identical for every fragment in the frame.

## Why there is no pixel gate

The obvious check — capture a frame with cascades, capture one with VSM, compare
— does not work on this scene, and the reason is worth recording because it
invalidates the control, not just the experiment.

Two captures of the **same** configuration at the same frame number differ by
8.5% of pixels (max channel delta 85). The cause is
`updateDemoLights(elapsedSeconds)`: the animated light swarm advances on
wall-clock time, so frame 240 of two runs is at two different animation phases.
Turning auto-exposure off does not help; it makes the mismatch worse, because the
manual exposure is a different constant.

So the 53% difference measured between the cascade and VSM captures says nothing:
it is smaller than the control's own noise in max-delta terms and larger in pixel
count, and neither number is trustworthy. **Any pixel gate for this feature needs
a fixed timestep or a scene with no animation**, and neither exists yet.

What *is* checked: the pass runs with zero validation errors on MoltenVK, the
page pool fills with real depth, the residency counters behave as designed, and a
human has eyeballed the A/B on the default scene without finding anything wrong.

One trap that A/B sets, worth knowing before anyone repeats it: the two captures
differ in overall tone, and that difference is **not** the shadows. The sky
changes too, and the sky samples no shadow — it is the animated light swarm at a
different phase, the same thing that makes the pixel gate impossible. Toggling
`enableShadows` back and forth within a second isolates the shadow term from it.

## Where it sits in the frame

`VsmPageMarkPass` is declared first in `declareGeometryPasses()`, before
`CSMShadowPass`: it reads the pyramid the *previous* frame left behind, so it has
to precede anything this frame writes. It is a compute pass with no rendering
scope, declared `sideEffect` because the bitmask it writes is not a graph
resource and liveness analysis would otherwise cull it away.

`VsmPagePass` follows it, declared only on frames that actually have dirty pages.
The pool is imported and read by the main pass regardless, so a frame that
redraws nothing still reaches the layout its sampler claims — the same asymmetry
the punctual shadow atlas uses.

The graph-visible dependencies are the `ShaderRead` on the depth pyramid, the
depth-attachment write on the pool, and the main pass's read of it. The buffer
clears, both dispatches, the indirect-command handoff, and the readback copies
carry explicit barriers, the same split `GpuCulling` uses for its own counters.

**Creation order matters and is not obvious.** The page pool is fixed-size and
follows neither the swapchain nor the depth pyramid, so it is created next to the
cascaded shadow map — and it has to be before `createPipeline()`, because the
page pipeline bakes the pool's depth format. Created after, the pipeline is
skipped and page rendering silently never turns on. The marking half is the
opposite: it samples the depth pyramid, so it is created and rebound with it on
every resize.

## Ownership and layout

| resource | owner | lifetime |
| --- | --- | --- |
| mark pipeline, descriptor layout/pool | `VirtualShadowMapPass` | recreated with the depth pyramid |
| page-request buffer | `VirtualShadowMapPass` | per frame in flight, device-local |
| page-request readback buffer | `VirtualShadowMapPass` | per frame in flight, host-visible |
| mark params buffer | `VirtualShadowMapPass` | per frame in flight, host-visible |
| page pool (4096x4096 depth) | `VirtualShadowMapPass` | startup to shutdown, fixed-size |
| page table buffer | `VirtualShadowMapPass` | per frame in flight, host-visible + device address |
| page cull pipeline + buffers | `VirtualShadowMapPass` | startup to shutdown |
| page graphics pipeline | `Renderer` | recreated with the other shadow pipelines |
| depth pyramid image/sampler | `DepthPyramid` | borrowed by reference |
| shared cull input buffer | `GpuCulling` | borrowed per dispatch |

Marking descriptor set: binding 0 = depth pyramid combined image sampler,
binding 1 = page-request storage buffer, binding 2 = mark params storage buffer.

Page-cull descriptor set: binding 0 = shared cull input, binding 1 = per-page
compacted indirect commands, binding 2 = per-page visible counts plus a trailing
over-cap counter, binding 3 = six frustum planes per page, binding 4 = caster
flags.

The page table is reached two ways on purpose: the cull and page passes bind it
as a storage buffer, and the main pass fragment shader will reach it by device
address, because the main pass push-constant block is already full at its
128-byte guaranteed size.

`VsmMarkParams` (C++, `VirtualShadowMapPass.cpp`) mirrors `VsmMarkParamsBuffer`
(GLSL, `vsm_page_mark.comp`) at 176 bytes, offsets pinned with `static_assert`.

The GPU-free clipmap math in
[`VirtualShadowMap.h`](../src/renderer/VirtualShadowMap.h) is the unit-tested
reference copy; [`virtual_shadow_map.glsl`](../src/shaders/virtual_shadow_map.glsl)
is the shader-side duplicate, the same arrangement `ClusterGrid.h` /
`cluster_build.comp` and `MeshLod.h` / `cull.comp` already use.

## Settings

`VsmSettings` (`RuntimeSettings.h`), persisted under `"vsm"`, surfaced in
**Shadows -> Virtual (clipmap)**:

| field | default | effect |
| --- | --- | --- |
| `enableMarking` | `false` | **startup-only**; allocates the subsystem and runs the marking pass |
| `enableShadows` | `false` | samples the pool instead of the cascades |
| `clipmapLevels` | 8 | active levels |
| `level0Extent` | 4.0 m | world span of the finest level's grid |
| `texelsPerPixel` | 1.0 | above 1 selects coarser levels; below, finer |
| `enablePageRendering` | `false` | allocates and draws pages; still samples nothing |
| `markBlockStride` | 8 | pixels per marking thread along each axis |

The numeric fields are clamped by `renderer::clampVsmClipmapSettings`, which
`clampRuntimeSettings` delegates to rather than repeating — a second copy of the
bounds could drift and would be invisible until a page landed somewhere
impossible.

The per-frame page counts are printed in the once-per-second `GPU timings:` block
as well as shown in the panel, because `--capture-frame` excludes ImGui and a
GPU-derived number that exists only on screen cannot be checked from a headless
or scripted run.

## Limitations

- **Visual correctness is unverified**, and no pixel gate is possible on this
  scene (above).
- **Skinned geometry is not a caster at all.** The skinned demo mesh is drawn
  directly rather than as a `RenderObject`, so it is absent from the draw-item
  list every shadow path walks — VSM and cascades alike. Skinning would also
  defeat the invalidation above on its own terms: it moves vertices without
  touching the model matrix, so the key would not change.
- **Bindless path only.** `simple.frag`, the fallback for devices without
  descriptor indexing, still uses the cascades. Adding VSM there means a second
  descriptor layout for a path that only exists on hardware this feature is not
  aimed at.
- **Page casters draw at authored detail.** The page cull emits no LOD level, so
  a page draws level 0 where the cascades would have picked a simplified one.
  Matches the punctual atlas, which does the same; worth revisiting once the
  pass has a cost worth attributing.
- **Cutout casters throw a solid silhouette.** The page pipeline has no
  alpha-tested variant, so `MASK` geometry casts as if opaque — the same
  fallback the cascades use when the bindless heap is missing.
- **The request set lags by the frames in flight**, both because it is derived
  from the previous frame's depth and because the bitmask is read back when its
  frame slot comes round again.
- **Effective resolution is capped by `kVsmPagesPerLevelAxis`**, not by
  `level0Extent`; see the measurement above.
- **A marking block that straddles a page seam only marks the page its centre
  lands in.** The coarser-level mark covers it, at that level's resolution.
- **Only the default and geometry-stress scenes have been measured**, both with
  a static camera. No moving-camera or Sponza page churn number exists yet, and
  churn is what decides whether the per-frame budget is the right shape.
