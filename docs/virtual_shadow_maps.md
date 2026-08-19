# Virtual Shadow Maps

Directional-light shadows as a clipmap of fixed-size pages, rendered on demand
and cached per page, instead of four camera-fitted cascades.

**Phase 1 is what is implemented, and Phase 1 draws nothing.** It measures which
pages this frame's visible surfaces would need. Directional shadows still come
entirely from the cascades ([`CascadeMath.h`](../src/renderer/CascadeMath.h));
the whole subsystem is off by default and, when on, changes no pixel. The point
is to size the physical pool and the per-frame page budget from a measurement
rather than a guess — and it already paid for itself, see
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

## Where it sits in the frame

`VsmPageMarkPass` is declared first in `declareGeometryPasses()`, before
`CSMShadowPass`: it reads the pyramid the *previous* frame left behind, so it has
to precede anything this frame writes. It is a compute pass with no rendering
scope, declared `sideEffect` because the bitmask it writes is not a graph
resource and liveness analysis would otherwise cull it away.

The only graph-visible dependency is the `ShaderRead` on the depth pyramid. The
buffer clear, the dispatch, and the readback copy carry explicit barriers, the
same split `GpuCulling` uses for its own counter readback.

## Ownership and layout

| resource | owner | lifetime |
| --- | --- | --- |
| mark pipeline, descriptor layout/pool | `VirtualShadowMapPass` | startup to shutdown |
| page-request buffer | `VirtualShadowMapPass` | per frame in flight, device-local |
| page-request readback buffer | `VirtualShadowMapPass` | per frame in flight, host-visible |
| mark params buffer | `VirtualShadowMapPass` | per frame in flight, host-visible |
| depth pyramid image/sampler | `DepthPyramid` | borrowed by reference |

Descriptor set (compute only): binding 0 = depth pyramid combined image sampler,
binding 1 = page-request storage buffer, binding 2 = mark params storage buffer.

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
| `markBlockStride` | 4 | pixels per marking thread along each axis |

The numeric fields are clamped by `renderer::clampVsmClipmapSettings`, which
`clampRuntimeSettings` delegates to rather than repeating — a second copy of the
bounds could drift and would be invisible until a page landed somewhere
impossible.

The per-frame page counts are printed in the once-per-second `GPU timings:` block
as well as shown in the panel, because `--capture-frame` excludes ImGui and a
GPU-derived number that exists only on screen cannot be checked from a headless
or scripted run.

## Limitations

- **Nothing is rendered or sampled.** There is no physical page pool, no page
  table, no page rendering, and no sampling path. Directional shadows are still
  entirely the cascades'.
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
