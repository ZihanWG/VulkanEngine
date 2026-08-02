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

A light whose radius encloses the camera clamps to maximum priority rather than
letting the projection blow up or go negative — something wrapped around the
camera is not a candidate for demotion.

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

Filtering is a 3x3 PCF in atlas UV space. Taps stay inside the tile except
within one texel of its border — and that border is the edge of the light's
cone, where the falloff is already zero.

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

## Blended geometry does not cast

Alpha-blended draw items are skipped in the atlas pass, matching the rule
[transparency.md](transparency.md) specifies for `BLEND`. They are skipped at
**replay** rather than filtered out of the caster list, the same way the main
HDR pass handles these buckets: the GPU cull still emits commands for those
slots and they are simply never drawn, which keeps every batch's compacted
command range valid. Filtering the lists instead would desync the batch ranges
from the cull input.

## Controls

Debug panel → tick **Advanced mode** → **Shadows** → *Punctual (spot/point)*.
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

- **No caching.** Every allocated face re-renders every frame, even for a light
  and geometry that have not moved. Static lights over static geometry are the
  obvious win here and it is not taken.
- **Priority ignores occlusion and the view frustum.** A light directly behind
  the camera ranks by projected size like any other, so it can take a tile that
  a visible light then cannot have.
- **The point-light demotion is a fixed one class**, not derived from the actual
  per-face footprint.
- **Uniform tile size.** A light one metre away gets the same 512px tile as one
  at the far plane.
- **CPU caster culling.** Each slot frustum-tests every draw item on the CPU
  rather than going through the existing GPU shadow-cull path.
- **No filtering beyond 3x3 PCF.** No variance/moment maps, no contact-hardening.
- **The atlas preview cannot show perspective depth.** See [Controls](#controls);
  the caster-draw counter covers the diagnostic need in the meantime.
