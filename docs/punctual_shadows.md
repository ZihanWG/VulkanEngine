# Punctual Shadows (Spot / Point Shadow Atlas)

Clustered lighting could put hundreds of dynamic point and spot lights on
screen, but none of them cast: the only shadow source was the directional CSM.
This subsystem gives punctual lights a shared depth atlas so spot lights shadow
the same geometry the sun does.

Both spot and point lights cast. A spot takes one atlas tile; a point light
takes six, one per cube face.

## Why an atlas rather than per-light shadow maps

A shadow map per light means a texture, a view, and a descriptor per light,
which does not survive a light count in the hundreds. One atlas keeps it to a
single image and a single descriptor binding, and turns "which lights cast this
frame" into a tile-allocation problem instead of a resource-allocation problem.

The atlas is 4096x4096, handed out in power-of-two tiles of 1024, 512, or 256px
by a quadtree allocator. A light dominating the view gets four times the
resolution of one barely visible, instead of both getting the same fixed tile.

## Tile allocation

The allocator starts each frame with the atlas cut into largest-class tiles.
Allocating a smaller class splits a free larger tile into four and banks three
for later, so neither class has a reserved region it cannot lend out — a frame
wanting a few big tiles and many small ones packs either way.

Nothing is freed mid-frame and the whole structure is rebuilt every frame, so
there is no coalescing pass and no fragmentation that outlives a frame.

Cube faces reserve **all six or none**, via a snapshot-and-roll-back. A light
that got four of six faces would sample cleared tiles for the other two, which
reads as light leaking through solid geometry — worse than not casting at all.

Because tiles now vary in size, a slot's rect no longer follows from its index.
The rect travels with the slot record and the atlas pass reads it back for the
viewport/scissor. Slot *indices* stay sequential, which is what keeps the float
encoding and the consecutive cube-face property working unchanged.

## Priority

Lights are ranked by **projected pixel radius** — `range / distance *
projScaleY`, using the same `projScaleY` the mesh-LOD selection uses, so "how
big does this look" means one thing across the engine. That single number drives
both who gets a tile and how big a tile they get.

This is what distance-only ordering could not express: a large light far away
can outrank a small one nearby, and it should.

A light whose radius encloses the camera needs separate handling, because
`range / distance` stops describing a footprint there. It ranks by how deeply it
encloses the camera instead, on a scale that starts exactly where the
non-enclosing branch ends — so a light drifting across its own radius does not
pop between size classes.

Saturating all such lights to one value instead looks harmless and is not: they
tie, the tiebreak falls back to light index, and priority silently degenerates
into "whichever lights come first". In a scene of overlapping lights that is
most of them, which makes the whole ranking inert.

Point lights are **demoted one size class**. A point light buys six tiles rather
than one, so ranking it against a spot on the same threshold lets a single point
light swallow six of the largest tiles and starve everything behind it. Each
cube face also only covers 90 degrees of the sphere, so a face genuinely
warrants less resolution than the light's whole footprint implies — the
demotion is a coarse stand-in for that, not purely a budget hack. On the demo
scene it is the difference between 2 point lights at 81% occupancy and 4 at 43%.

The sort orders *indices*, never the light array itself — the cluster
light-culling pass indexes into that array by position, so reordering it would
reassign every froxel's light list. Ties break on light index so equally-ranked
lights do not shuffle between frames and flicker their shadows.

## Point lights: six faces, one budget

A point light is shadowed by six 90-degree faces, one per signed axis, packed
into **consecutive** slots. Consecutive is load-bearing: the light stores only a
base slot, and the shader adds the face index it derives from the
light-to-fragment direction. `allocateRange` therefore fails whole rather than
partially reserving, so a light can never get a truncated cube.

Six tiles per light means the atlas oversubscribes as soon as a swarm shows up,
so how many point lights may cast is capped by a visible **Max shadowed point
lights** slider rather than an implicit limit. Which ones is decided by the
priority ranking above.

### The face convention is the thing to get right

The CPU builds one projection per face and the shader picks a face from the
light-to-fragment direction. If those two disagree, every point light samples
the wrong tile — and that fails *plausibly*, as shadows in roughly the right
place with the wrong contents, rather than as an obvious crash.

So the order (+X, -X, +Y, -Y, +Z, -Z) and the tie-breaking `>=` comparisons are
written identically in `pointShadowFaceIndex` and its GLSL mirror, and a unit
test pins the invariant directly: for directions all over the sphere, the face
selection picks the face whose projection actually contains that direction, and
all six faces are exercised.

That test has a blind spot worth naming, because a real bug lived in it. It
feeds the *same* position to both sides, so it can only catch a disagreement in
the convention itself — never a disagreement in the **input**. The shader
originally picked the face from the unbiased world position but projected the
normal-offset-biased one, and near a boundary the offset pushes a sample into
the neighbouring face's cone. It then projected outside the selected face's
frustum, the bounds test read that as "outside the light", and every face
boundary got a bright fully-lit seam.

The fix is structural rather than a correction: the bias is computed first and
one position drives both the face selection and the projection, so the two
cannot take different inputs. (Reading the normal bias before the face is known
is exact, not an approximation — all six faces of a light carry identical
params.)

## Pass structure

```
CSMShadowPass            directional cascades, unchanged
PunctualShadowAtlasPass  one rendering scope over the whole atlas
MainHDRPass              samples both
```

The atlas pass opens **one** `vkCmdBeginRendering` covering the entire atlas and
isolates each slot with a viewport and scissor. That means:

- the depth clear runs once for the whole image rather than per tile, and
- slots that got no casters stay cleared at the far plane instead of holding a
  previous frame's depth.

Per-slot the pass pushes that slot's light-space view-projection, then draws the
casters that survive a CPU frustum test against the slot's own frustum.

## Why a separate vertex shader

The CSM path precomputes a per-object light MVP for each of its four cascades
and indexes that array in `shadow.vert`. That does not scale to 64 slots — it
would mean 64 matrices per object in `ObjectFrameData`.

`shadow_punctual.vert` inverts the arrangement: the slot's view-projection
arrives as a **push constant**, one push per slot rather than per slot per
object, and is combined with the object's model matrix in the shader. The cost
is one extra matrix multiply per vertex; the saving is an `ObjectFrameData` that
does not grow with the atlas.

The push-constant block carries explicit padding, because GLSL rounds the `mat4`
up to its 16-byte alignment and it therefore lands at offset 16, not 8, after
the buffer-device-address field. `PunctualShadowPushConstants` static-asserts
that offset so the two cannot drift.

## Slot encoding

The slot index rides in `GpuLight::spotScaleOffset.z`, which was unused padding
— the same trick the mesh-LOD work used to fit LOD indices into
`GpuCullDrawItem`. The light record stays 64 bytes and no varying or descriptor
had to grow.

The value is a float, so the encoding is pinned down in one place
(`punctualShadowSlotToFloat` / `punctualShadowSlotFromFloat`):

- a valid slot round-trips exactly — slot counts are far below the 24-bit
  exact-integer range of a float;
- anything else becomes `-1.0`, which the shader tests with a `< 0.0` compare.

**That sentinel is what guards the buffer dereference in the shader.** The CPU
stamps it into every light whenever the atlas is unavailable or the toggle is
off, and the slot-buffer address is only non-zero when at least one light got a
tile. So "this light has a valid slot" always implies "the slot buffer is
valid", and the shader needs no null-address test.

The decoder is written as a negated compare (`!(encoded >= 0.0f)`) so a NaN
decodes to unshadowed rather than sliding through into an out-of-bounds fetch.

## Sampling

`punctualShadowFactor` in `simple_bindless.frag` runs *after* the cheap
rejections in `evaluatePunctualLight` (range, cone, zero attenuation), so fully
attenuated fragments never pay for the atlas fetch.

It offsets the sample position along the surface normal before projecting.
Normal-offset bias fixes acne more cheaply than a large constant depth bias and
does not detach contact shadows the way peter-panning bias does; both biases are
per-slot in `GpuShadowSlot::params` and tunable live from the debug panel.

Both biases are **slope-scaled** by the sine of the angle between the surface
and the light, matching the shape `shadowDepthBias` uses on the CSM path. That
term is not optional: one shadow texel spans the most depth at grazing
incidence, so a flat constant bias leaves acne on every surface edge-on to the
light. Under a downward spot that is the vertical side of every object, and it
reads as combed streaks along the shadow boundaries. Keeping the head-on term
small is what stops the fix from turning into peter panning.

Fragments that project outside the slot's frustum return "lit". For a spot that
is exactly the region outside the lit cone, where the falloff has already
reached zero, so it changes nothing visually.

Filtering is a 3x3 PCF in atlas UV space, with every tap **clamped into the
slot's own tile**.

That clamp is load-bearing rather than defensive. Neighbouring atlas texels
belong to a different tile — for a cube face the adjacent face, and with the
quadtree allocator possibly an unrelated light — so a tap that walks out
compares against unrelated depth. On a spot this only ever happened at the cone
edge, where the falloff is already zero, so it was invisible and the original
code skipped the clamp on exactly that reasoning. On a cube face the tile border
is the *middle* of the lit scene, and the same mismatch draws hard seams along
every face boundary.

## Render graph integration

The atlas is an imported graph texture, and the pass declares a depth-attachment
write on it; `MainHDRPass` declares the matching sampled read. The graph infers
the barriers in both directions, so there are no hand-written image barriers.

One subtlety is load-bearing: the **write** pass is only declared when a light
actually got a tile, but the **read** is declared whenever the atlas exists. A
frame that casts nothing therefore still gets the transition into the layout the
material descriptors record. Without that, a cold start with punctual shadows
disabled would leave the image in `UNDEFINED` while binding 7 claimed
`DEPTH_READ_ONLY_OPTIMAL`.

## GPU caster culling

Optional, off by default, and honestly a pessimisation at this scene's size.

The CPU path frustum-tests every draw item against every slot, which is
`O(slots x draw items)`. The GPU path replaces that with **one** dispatch over
every (slot, draw item) pair, writing each slot's survivors into its own region
of an indirect buffer so the atlas pass issues one draw call per slot.

One dispatch rather than one per slot is forced by where it has to run: compute
cannot be recorded inside a dynamic-rendering scope, and the atlas pass is a
single scope so its cached tiles survive the partial clear. Everything is
therefore culled before the scope opens — and once that is true, batching it
into one dispatch is both less code and avoids per-slot dispatch overhead.

**Measured on the demo scene**, which is why it defaults off:

| | CPU cull + record | Atlas GPU | Cull GPU |
| --- | --- | --- | --- |
| CPU culling | ~40us (release) | 0.19ms | — |
| GPU culling | lower | 0.33-0.43ms | 0.02-0.08ms |

It moves work off the CPU and makes the GPU pass *slower*. The reason is the
missing `vkCmdDrawIndexedIndirectCount` on MoltenVK: without it each slot submits
the full draw-item count as indirect commands and relies on the zeroed ones being
no-ops, so a slot that culls everything still costs a command per draw item. The
CSM path has the same limitation for the same reason.

So it pays off when two things hold, neither true here: enough draw items that
the CPU cost stops being ~40us, and a platform with indirect-count so the draw
side scales with survivors rather than candidates. It is kept as a toggle so the
comparison is something you run rather than something you argue about, and the
panel reports the CPU cost it would remove next to the GPU cost it would add.

**One trap worth recording.** The cull input buffer is shared with the main and
CSM paths and carries no bucket, and `GpuCullDrawItem` is exactly 64 bytes with
no spare field. The CSM path filters blended geometry at *replay* time, which an
indirect per-slot draw cannot do because the commands are already compacted by
then. Without a separate per-draw-item caster mask, enabling GPU culling
silently reintroduces alpha-blended geometry casting opaque shadows — a bug this
engine already had and fixed once.

## Caching

Every allocated tile used to re-render every frame, including for a static light
over static geometry. The pass now hashes everything it consumes and skips
entirely when that hash is unchanged: no clear, no draws, and no layout
transition, since the atlas keeps the sampled layout the main pass left it in
and the main pass wants that same layout next frame.

Invalidation is **per tile**, not per frame. A tile only cares about the casters
inside its own frustum, so one moving object re-renders the handful of tiles
that can see it rather than the whole atlas.

**Tile rect is the cache identity, not slot index.** Slot indices shift between
frames as lights are reordered by priority, but a tile is a fixed region of the
image. If the tile at rect R holds the render described by hash H, then any slot
this frame wanting rect R with hash H already has its contents there, whichever
light it belongs to. So the resident map is keyed by packed rect, and the
packing is exact rather than itself a hash -- two tiles colliding into one key
would let one tile's contents pass for another's.

Preserving cached tiles means the attachment loads instead of clearing, and the
dirty tiles are cleared individually with one batched `vkCmdClearAttachments`.
The first frame after the atlas image is created still clears everything, since
nothing in it can be trusted then; that also purges resident entries for tiles
the full clear wiped but did not redraw.

**A content hash rather than dirty flags.** The failure mode of caching a shadow
is a stale one, which shows up far from its cause and reads as a rendering bug.
Dirty flags spread the risk across every input that can change and every place
that changes one; hashing the inputs concentrates it into a single enumeration
that is either complete or not. What is hashed:

| Input | Why |
| --- | --- |
| Per-slot view-projection | Carries the light's position, direction, range, cone, and near plane |
| Per-slot atlas rect | Where the tile renders and at what resolution |
| Caster mesh, index range, bucket | Which geometry is drawn |
| Caster model matrix | Where it is |
| Raster depth bias settings | Pipeline state the tiles are rendered with |

Only the casters that survive *that tile's* frustum cull are hashed, and that
cull mirrors the one the recording pass performs. If the two ever diverge the
hash stops describing what actually gets drawn, which is the one way this can
silently go wrong.

The per-slot frustum cull is a pure function of the slot projections and caster
transforms, so its result needs no separate hashing. Floats are hashed by exact
bit pattern, so this is a strict "identical inputs" test and never approximate.

Two things the hash cannot express, handled separately:

- **A recreated atlas image.** Its contents are undefined even when the key
  matches, so `createShadowMap` drops the cache outright.
- **An empty scene.** That path returns before the hash is recomputed, so it
  clears the flag rather than inheriting the previous frame's answer.

**Where the hash is computed matters.** It runs after `updateAnimatedTransforms`
and `buildDrawItems`, not alongside the slot assignment. Hashing at assignment
time would read the *previous* frame's transforms and draw items, so every
change would be noticed exactly one frame late -- a single stale shadow frame on
every edit, which is both visible and hard to attribute.

The debug panel reports how many tiles were redrawn out of how many exist, plus
how many frames were skipped entirely, so the hit rate is measured rather than
assumed. Measured on the demo scene:

| Scene state | Tiles redrawn |
| --- | --- |
| Lights orbiting (the default) | 25 of 25 -- every projection changes, so nothing is reusable |
| Lights frozen, nothing moving | 0, the pass is skipped outright |
| Lights frozen, one object rotating | 9-10 of 25, varying as it moves through frustums |

The middle row is the case the whole mechanism exists for, and the last row is
the one per-tile invalidation adds over a single frame-wide hash.

## Blended geometry does not cast

Alpha-blended draw items are skipped in the atlas pass, matching the rule
[transparency.md](transparency.md) specifies for `BLEND`. They are skipped at
**replay** rather than filtered out of the caster list, the same way the main
HDR pass handles these buckets: the GPU cull still emits commands for those
slots and they are simply never drawn, which keeps every batch's compacted
command range valid. Filtering the lists instead would desync the batch ranges
from the cull input.

## Controls

Debug panel → tick **Advanced mode** → **Scene** tab → **Shadows** → *Punctual
(spot/point)*.
The controls are behind the advanced-mode gate, which is where the cascaded
shadow settings have always lived:

| Control | Effect |
| --- | --- |
| Cast punctual shadows | Master toggle; off stamps the unshadowed sentinel into every light |
| Slots used | Tiles allocated this frame, with atlas occupancy by area — with mixed tile sizes a slot count says nothing about how full the atlas is |
| Depth bias | Constant offset applied to the compared depth |
| Normal bias | World-space offset along the surface normal before projecting |
| Max shadowed point lights | Budget, in lights; each costs 6 tiles. Nearest to the camera are served first |
| Caster draws recorded | Draws the atlas pass issued last frame; zero with slots > 0 means it culled everything |
| Assignment churn | Lights that gained or lost a shadow since last frame; persistently non-zero is what popping looks like |
| Debug: shadow term only | Renders the punctual visibility term as greyscale instead of shaded colour |
| Atlas depth preview | Samples the atlas image directly (see the caveat below) |

**Read the caster-draw count, not the preview image.** The preview is a plain
`ImGui::Image` with a multiplicative tint, so it cannot expand contrast near
1.0 — and a perspective projection puts almost all of its depth range up there.
A correctly rendered tile therefore reads as solid far-plane colour, identical
to an empty one. The CSM cascade previews do not have this problem only because
an orthographic projection gives linear depth.

So the useful signal is `Caster draws recorded`: zero with slots > 0 means the
atlas pass culled or skipped everything; non-zero means the pass ran and any
remaining problem is in the lookup. Making the preview itself readable needs a
linearising remap, which the shared preview helper has no place to put.

That splits the diagnosis cleanly across the two halves:

| Symptom | Meaning |
| --- | --- |
| `Caster draws recorded` is 0 with slots > 0 | The atlas pass culled or skipped everything |
| Draws > 0 but **Debug: shadow term only** is flat white | The atlas is written but never sampled — the lookup is wrong |
| Draws > 0 and the debug view shows structure | Both halves work; the term is simply subtle in the shaded image |

The debug view exists because a single spot among dozens of clustered lights is
easy to lose in the beauty image — "I cannot see a shadow" is not evidence of a
bug on its own.

Read the debug view as a **min across the lights that actually reach each
fragment**, not as a preview of the shaded result. Several casters overlapping
darkens a great deal of the frame there while the shaded image changes far less,
because shading also weighs each light's attenuation. The gate matters:
`evaluatePunctualLight` applies the shadow only after its range and cone
early-outs, so the debug path has to reproduce them — without that it reports
occlusion from lights contributing no light at all, which min()s the frame to
near-black and reads as a catastrophic bug that is not there.

## Testing

`tests/test_punctual_shadow_atlas.cpp` covers the GPU-free core the same way
`ClusterGrid.h` and `CascadeMath.h` are covered:

- slots tile the atlas without overlapping, and every tile lands inside it;
- slot UV rects agree with their pixel rects;
- the allocator hands out each slot once and then degrades to the sentinel
  rather than failing;
- slot indices survive the float round-trip, including the NaN and
  out-of-range cases;
- a spot projection centers its cone axis in the tile, and depth increases with
  distance from the light;
- points outside the cone project off the tile, which is what makes the shader's
  bounds test equivalent to the cone cutoff;
- degenerate inputs (zero direction, a cone at/over 90 degrees, zero range)
  produce finite, non-degenerate matrices.

## Limitations

- **The skinned caster is exercised here but not visually verified.** It is
  drawn into every tile its bounds reach (6 of 25 on the default scene, printed
  beside the caster draw count) and its pose digest enters those tiles' cache
  keys, so an animating rig dirties them. But the demo scene's punctual shadows
  are too dim, and the mesh too close to the frame edge, to compose an A/B from:
  the frame changes by at most one quantization step. See
  [skeletal_animation.md](skeletal_animation.md).
- **A tile is all-or-nothing internally.** One moved caster inside a tile's
  frustum redraws that whole tile, including the casters in it that did not
  move. That is inherent to a shadow map: the depth buffer is not separable per
  caster.
- **Invalidation is conservative about frustum membership.** A caster whose
  bounds intersect a tile's frustum dirties it even if it is fully occluded by
  something nearer the light and could not change the result.
- **Tile assignment is recomputed from scratch every frame**, and priority
  depends on the camera, so moving the camera reshuffles which lights hold tiles
  and can flip a light between size classes at a threshold. Both pop visibly.
  Fixing it needs hysteresis on the size class and stable light identity across
  frames, neither of which exists yet.
- **Priority ignores occlusion and the view frustum.** A light directly behind
  the camera ranks by projected size like any other, so it can take a tile that
  a visible light then cannot have.
- **The point-light demotion is a fixed one class**, not derived from the actual
  per-face footprint.
- **GPU caster culling is off by default, and on this scene it is a net loss.**
  See [GPU caster culling](#gpu-caster-culling).
- **No filtering beyond 3x3 PCF.** No variance/moment maps, no contact-hardening.
- **No cross-face filtering.** PCF is clamped inside a face rather than
  continuing onto the neighbour, so filtering narrows slightly at face
  boundaries instead of blending across them. Correct, but not seamless; proper
  cross-face taps need neighbour-face lookups.
- **The atlas preview cannot show perspective depth.** See [Controls](#controls);
  the caster-draw counter covers the diagnostic need in the meantime.
