# Virtual Shadow Maps

Directional-light shadows as a clipmap of fixed-size pages, rendered on demand
and cached per page, instead of four camera-fitted cascades.

**Nothing here samples the pool yet, so no pixel on screen changes.**
Directional shadows still come entirely from the cascades
([`CascadeMath.h`](../src/renderer/CascadeMath.h)), and the whole subsystem is
off by default. What exists is everything up to and including rendering:

| stage | setting | what it does |
| --- | --- | --- |
| marking | `enableMarking` | works out which pages the frame needs, and counts them |
| rendering | `enablePageRendering` | allocates a physical page per request and draws it |
| sampling | — | not implemented |

The toggles nest, each an A/B point, so the page pass can be measured on its own
without anything downstream depending on it. Building it in this order is what
let two design errors surface before the expensive parts were written — see
[What the measurement changed](#what-the-measurement-changed).

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
| `enableMarking` | `false` | runs the marking pass; draws nothing |
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

- **Nothing samples the pool.** Directional shadows are still entirely the
  cascades'. The pool fills with correct depth and is then read by nobody.
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
