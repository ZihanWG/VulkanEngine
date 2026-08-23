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

> **Gated, and the gate found a leak.** `tools/dev/vsm_ab.sh` captures a
> reproducible frame with and without VSM; both configurations repeat byte for
> byte, so the comparison between them means something. The first thing it
> measured was every umbra reading ~15% of the sun too bright, from a
> shadow-compare bias carried over from the cascades in the wrong depth
> normalization. That is fixed and measured; one lit-surface discrepancy is still
> open. See [The pixel gate, and the leak it found](#the-pixel-gate-and-the-leak-it-found).

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

### What the key deliberately leaves out

Each omission is only safe because the feature it would cover does not exist
here yet, and each becomes a stale-shadow bug on the day it lands:

| omitted | why it is safe today | add it when |
| --- | --- | --- |
| cull-selected LOD level | the page cull emits no level; pages always draw authored geometry | page LOD lands, which it should not — see below |
| raster depth bias | a compile-time constant in `ShadowSettings`, no UI, cannot move | it becomes a setting |

`alphaCutoff` **is** hashed now, indirectly and by accident of the bucket: a
material promoted to or from `MASK` changes the draw item's bucket, which the key
does hash. A cutoff edited *within* `MASK` does not change the key and will leave
a stale cutout shadow. That is a narrower hole than the one the cascades close by
hashing the value outright, and it is worth closing the same way if the material
inspector's cutoff slider ever gets used in anger.

The cascades hash all three, which is why `CascadeShadowCaster` looks richer than
the per-object key here. The light direction, the clipmap settings and a scene
switch are covered, but elsewhere — the first two by `updateResidency` hashing
its own inputs, the third by `resetSceneState`.

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
the scope has to stay single. Going over the per-page cap is **counted and
reported**, not silently dropped — the counter keeps climbing past the cap so the
readback shows real demand rather than a page that merely looks full.

### Opaque and cutout casters get separate regions

A page compacts its survivors into a region per **caster bucket**, not one region
per page, and the pass draws each page twice — opaque with the depth-only
pipeline, alpha-tested with `shadow_vsm_masked.vert` + `shadow_masked.frag`.

That split is forced by the platform. With no indirect-count, each draw submits
its region's whole stride and relies on the commands the cull zeroed being
no-ops; a shared region would therefore run the cutout commands through the
opaque pipeline as well, and a perforated panel would go back to throwing a solid
rectangle. The two sweeps are ordered by bucket rather than interleaved per page
because the pipeline switch is the expensive part and the masked one also rebinds
a descriptor set.

Which bucket a caster lands in travels beside the shared cull input, in the same
per-draw-item flag array that already carried "does this cast" — the input record
is exactly 64 bytes with no spare field. When the bindless heap is missing there
is no masked pipeline, and the flag routes cutout casters into the opaque bucket
so they still cast, solidly. That is the same fallback the cascades use.

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

## Reading what the clipmap is actually doing

The counters say how many pages are resident and how many were cached. They
cannot say **where**, and where is the whole question once the camera scrolls the
window or a caster dirties a footprint.

**Shadows → Virtual (clipmap) → Page residency** draws one grid per clipmap
level: 16x16 cells, one per slot in that level's window, green where the page
holds depth, blue where it is allocated but not drawn yet, dark where there is
nothing, and the camera's own page outlined. It reads the same page table the
GPU reads, so it cannot drift from what the sampler will really find — including
the identity check: a slot holding a page that has scrolled away shows as empty,
because that is exactly what the sampler makes of it.

Blue is not an error. Allocation and drawing are separate steps, so a page can
own space for a frame before the page pass fills it.

The **pool depth preview** underneath is the physical view: 32x32 pages of the
4096² image, in allocation order rather than world order. Unlike the punctual
atlas preview — which [warns you not to trust it](punctual_shadows.md) because
perspective depth piles up against 1.0 — this one is worth reading: pages are
rendered with an orthographic projection, so their depth is linear.

## Page LOD was considered and rejected

The page cull emits no LOD level, so a page draws authored geometry where a
cascade would have drawn a simplified chain. That looks like an obvious gap. It
is not worth closing, for two measured reasons.

**There is no time to save.** In the steady state the page pass does not run at
all — every page is cached, nothing is redrawn, and `VsmPagePass` never even
appears in the profiler. The only run where it had work was the deliberately
pathological one (a ground plane moving every frame): 0.70 ms for 84 pages.
Optimising the triangle count of a pass that costs nothing on a scene that is not
fighting its own cache is the same trade this project already rejected for
[bloom aliasing and effective-radius light culling](design_decisions.md).

**The saving would land where the pages are cheapest.** LOD selection is driven
by distance to camera, and so is the clipmap level. A page fine enough for
simplification to be visible is a page whose casters are near the camera, which
selects level 0 anyway; the pages that would actually simplify are the coarse,
far ones — of which there are few, and whose casters are small on screen.

It also costs more than it looks: the LOD table has to reach the page cull, the
level has to be packed into `firstInstance` the way `cull.comp` does, and the
per-object caster key has to start hashing the selected level or a level change
alone would leave a stale page.

If the page pass ever shows up in a real frame's profile, this is the first thing
to revisit — with a measurement, not on principle.

## The pixel gate, and the leak it found

There is one, and building it falsified the claim that stood here before.

That claim was that a pixel gate "needs a fixed timestep or a scene with no
animation, and neither exists yet". The measurement behind it was real -- two
captures of the same configuration differed by 8.5% of pixels, because
`updateDemoLights(elapsedSeconds)` advances the light swarm on wall-clock time --
but the conclusion was wrong, and had been wrong since before this feature
started: `--deterministic` landed in `92feaac`, ahead of the first VSM commit.
It pins the frame clock to a fixed 1/60 s, which makes the light swarm, the
skeletal delta, exposure adaptation and every animated transform a function of
the frame number, and `docs/headless_ci.md` had already measured three runs of
`--deterministic --capture-frame 60` producing byte-identical PNGs. The control
that failed was simply run without the flag.

Re-run with it, on MoltenVK, at frame 60 of the default scene:

| configuration | repeats | differing pixels |
| --- | --- | --- |
| cascades (`--vsm off`) | 3 | **0 / 3686400** |
| VSM (`--vsm shadows`) | 3 | **0 / 3686400** |

The second row is the one that was not obvious. Page residency is cross-frame
state -- last frame's depth pyramid marks this frame's requests, which allocate
pages that persist and are re-used -- and a fixed timestep turning all of that
into a function of the frame number was a claim, not a given. It reproduces.

`tools/dev/vsm_ab.sh` runs the whole thing: both controls first, then the
cascade-versus-VSM comparison at three tolerances. It refuses to print a
comparison whose control did not reproduce. It is deliberately **not** a CI job:
pixel determinism holds here and does not hold on lavapipe, where the same commit
rendered three times produced 0, then 433, then 0 differing pixels.

### Read the tolerance-3 diff, not the tolerance-0 one

Changing the shadow term changes scene luminance, auto-exposure follows it, and
the entire frame lands one quantization step away. So a raw comparison reports
~11% of pixels differing, and ~87% of those are a delta of 1 in a smooth
gradient: sky, sphere bodies, the lit floor. They are real and they mean nothing.
At `--channel-tolerance 3` that noise is gone and what is left is 1.7% of the
frame, which is the shadows.

### What the gate found: every umbra was leaking

The first comparison it produced was not the expected "shapes agree, edges
differ". Every cast shadow read **uniformly lighter under VSM**, by 7.6/255 in a
64x64 patch of the deepest umbra -- against a lit floor at 80/255 and a cascade
umbra at 31.8/255, that is about 15% of the sun leaking into shadow that should
have none.

The cause was a units mismatch in the shadow-compare bias. The sampler took
`CsmSettings::depthBiasConstant` -- 0.002 -- and subtracted it from the page's
normalized depth. But that constant is a fraction of a *cascade's* ortho depth
box, tens of world units, while a page's depth axis spans `2 * depthRange`, 500
world units by default. The same 0.002 therefore meant centimetres in one place
and a **whole world unit** in the other. Zeroing it dropped the umbra straight
back onto the cascades' value (31.95 against 31.84), which is what confirmed the
mechanism rather than merely fitting it.

Zero is not the fix, though: it puts the acne back. The bias is now expressed in
**texels of whichever clipmap level the lookup lands on**, converted to world
units at that level and only then into the page's depth normalization. Texels
rather than world units because level *L*'s texel is 2^L times level 0's, so the
depth error one texel can hide scales with the level -- one figure in texels is
right at every level, one in world units at exactly one of them.

The default was then swept rather than guessed, on the same reproducible frame
(all values are the red channel's mean over a fixed patch; the cascade column is
the reference each row is trying to match):

| `depthBiasTexels` | lit face | lit sphere | umbra |
| --- | --- | --- | --- |
| *cascades (reference)* | *63.57* | *138.93* | *31.84* |
| old, mis-scaled | 63.56 | 138.87 | **39.40 leaking** |
| 2 | 57.09 | 130.39 | 31.91 |
| 8 | 57.09 | 131.36 | 31.91 |
| 32 | 57.09 | **139.00** | 31.88 |
| **64 (shipped)** | 57.09 | **138.97** | **31.86** |
| 128 | 57.09 | 138.90 | 32.38 leaking |
| 256 | **63.56** | 138.86 | **39.40 leaking** |

The window is [32, 128] and 64 sits in the middle of it. Below 32 the scene
self-shadows its own lit surfaces; by 128 the umbra starts lifting back toward
the leak the setting exists to remove.

With 64 shipped, VSM differs from the cascades over **1.7% of the frame at a
maximum channel delta of 10**, and the difference image is shadow *outlines* --
the umbra interiors now agree, and what is left is the penumbra, which is exactly
where two shadow techniques at different resolutions are supposed to disagree.

### What it did not settle, and what was ruled out

One column of that table stays stubborn: a lit face reads 57.09 against the
cascades' 63.57 at every bias in the window, and only "recovers" at 256, which is
also where the umbra leak returns -- so that recovery is peter-panning erasing the
difference, not fixing it. It is ~6.5/255 on certain lit surfaces and uniform
across each face rather than shaped like a cast shadow.

Seven hypotheses have since been measured and eliminated, each on the same
reproducible frame. The value in the list is that it is *narrowing*: whoever picks
this up should not spend the day re-running these.

| hypothesis | test | result |
| --- | --- | --- |
| VSM depth bias too small | 2, 8, 32, 64, 128 texels | **57.09 at every one** |
| VSM samples too coarse a level | `texelsPerPixel` 0.25 → 4.0, levels L1–L4 | **57.09 at every one** |
| The cascade is under-biased and only *looks* right | cascade constant bias / 10 | **63.57, unchanged** |
| The cascade is resolution-limited | 2048 → 4096 → 8192 | **63.57, unchanged** |
| The cascade clips the occluder away in depth | `zPadding` x5, x25 | **63.57** (x25 lifts the *umbra* to 35.72 on precision alone) |
| PCF taps clamped at a page seam over-occlude | radius 0, 1, 2, both paths | **unchanged** (the radius does reach the GPU -- it moves 6 pixels elsewhere) |
| The sampler lacks the cascades' slope-scaled bias term | added `max(const, slope * (1 - N.L))`, slope 128 / 512 / 2048 texels | **the face and the umbra move together**: 128 changes nothing, 512 gives 60.43 with the umbra already at 39.12, 2048 gives 63.56 with the umbra fully leaking at 39.40 |

That last row is the one this document previously led with, and it is now the
most thoroughly dead: a slope term was the one shape that could plausibly have
biased a grazing face hard while leaving the umbra alone, and it does not
separate them. Which says the two surfaces have similar `N.L` -- the "cube top"
is about as horizontal as the floor -- so no function of the surface angle can
tell them apart.

One neighbouring fact fell out of the same sweep and is worth keeping: shadow LOD
*does* move the cascade umbra. `lod.shadowBias = 0` or `forcedLod = 0` takes it
from 31.84 to 35.72, because the simplified caster's silhouette is not the
authored one. Pages draw authored geometry and cascades draw the cull-selected
level, so the two paths disagree there by construction -- but it is not this
difference, which survives forcing both to level 0.

So the cascade says "lit" under every knob it has, and the page pool says
"partially occluded" under every knob it has, and no bias function of depth or
surface angle separates the affected face from the umbra. What has *not* been done is looking
at what is actually in the page that covers that surface; the residency grid says
where pages are, not what depth they hold. That is the next instrument to build,
and it is now capturable — `--capture-include-ui` puts the debug panel in a
scripted capture.

### Cutout page shadows do resolve their holes

The perforated panel is in the default startup scene -- `appendPortfolioShowcase`
builds it, and `createScene()` falls to that showcase whenever no sample scene
was fetched -- so the same deterministic capture covers the masked caster bucket
without any UI driving.

At tolerance 3 the difference under the panel traces **the bottom row of
perforations**, one arc per hole, rather than a solid band. A solid silhouette
would have produced the band. Sampling a 300x40 strip of that shadow confirms the
sign: VSM reads 60.06 against the cascades' 59.53, so more light comes through the
holes, not less -- the higher-resolution page resolves openings the cascade's
filter blurs shut.

### What the scene cannot show

The directional shadows in every demo scene here are low contrast: the umbra sits
at 31.8/255 against a lit floor at 80/255, because ambient and punctual lighting
dominate the sun. That is why an eyeball A/B found nothing wrong -- there is
little to see -- and it is why the numbers above are all patch means from a
reproducible capture rather than descriptions of an image. A scene with a
dominant sun would make this feature much easier to judge, and none exists yet.

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
| `texelsPerPixel` | 1.0 | above 1 selects coarser levels; below 1 is clamped by the coverage bound (see Limitations) |
| `enablePageRendering` | `false` | allocates and draws pages; still samples nothing |
| `markBlockStride` | 8 | pixels per marking thread along each axis |
| `depthBiasTexels` | 64 | shadow-compare bias, in texels of the sampled level |

The numeric fields are clamped by `renderer::clampVsmClipmapSettings`, which
`clampRuntimeSettings` delegates to rather than repeating — a second copy of the
bounds could drift and would be invisible until a page landed somewhere
impossible.

The per-frame page counts are printed in the once-per-second `GPU timings:` block
as well as shown in the panel, because `--capture-frame` excludes ImGui and a
GPU-derived number that exists only on screen cannot be checked from a headless
or scripted run.

### Marking without page rendering left the pool lying about its layout

`--vsm mark` allocates the pool -- that is what the startup-only marking toggle
decides -- but runs no page pass, and the material set binds the pool either way.
So every main-pass draw statically accessed a descriptor promising
`DEPTH_READ_ONLY_OPTIMAL` for an image still in `UNDEFINED`: eleven validation
errors a frame.

It had been there since the pool was introduced. What was missing was a way to
*select* the configuration: the three stage toggles lived in a git-ignored
settings file and behind a startup-only checkbox, so "marking on, rendering off"
was reachable in principle and never actually run. The flag that made the A/B
scriptable is what walked into it on its first pass through the matrix.

The fix is a one-shot explicit barrier into the layout the descriptor was written
with. Explicit rather than graph-declared on purpose: in this mode the pool is
not a graph resource at all -- it is handed to `beginFrame` only when page
rendering is active -- which is exactly the case the manual-barrier rule is for.
The contents stay undefined, which is sound rather than lucky: the only shader
that samples the pool is gated on `enableShadows`, and that cannot be on without
page rendering.

## Limitations

- **One lit-surface discrepancy is unexplained.** Certain lit faces read ~6.5/255
  darker than under the cascades at every depth bias that keeps the umbra honest.
  Seven hypotheses have been measured and eliminated, including the slope-scaled
  bias term this doc used to lead with. See
  [What it did not settle, and what was ruled out](#what-it-did-not-settle-and-what-was-ruled-out).
- **`texelsPerPixel` below 1.0 does nothing on this scene, by design.** Asking
  for finer levels is clamped by the coverage bound — `vsmSelectLevel` returns
  `max(quality, coverage)` — so 0.25 and 1.0 produce the identical 99-page set
  and byte-identical pixels. Above 1.0 it works normally, because coarser is
  always addressable: 2.0 gives 97 pages, 4.0 gives 37, 8.0 gives 18. This is the
  `kVsmPagesPerLevelAxis` cap below wearing a different hat, not a broken
  setting; an earlier note here called it unexplained because only the clamped
  half of the range had been measured.
- **The demo scenes barely show a directional shadow.** Umbra 31.8/255 against a
  lit floor at 80/255, so every judgement here is a patch mean from a
  reproducible capture rather than something visible at a glance.
- **Page casters draw authored geometry.** No LOD, deliberately — see above.
- **The skinned caster costs 12 pages a frame** on the default scene: it
  invalidates and redraws the pages its bounds cover on every frame it animates,
  leaving 87 of 99 resident pages cached. Animated geometry has no cacheable
  shadow, so that is inherent rather than a tuning failure — but it is the one
  thing here that defeats page caching by construction. Its key is a pose digest
  rather than a transform, for exactly the reason the invalidation section above
  describes; see [skeletal_animation.md](skeletal_animation.md).
- **Bindless path only.** `simple.frag`, the fallback for devices without
  descriptor indexing, still uses the cascades. Adding VSM there means a second
  descriptor layout for a path that only exists on hardware this feature is not
  aimed at.
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
