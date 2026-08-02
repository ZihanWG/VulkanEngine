# Punctual Shadows (Spot / Point Shadow Atlas)

Clustered lighting could put hundreds of dynamic point and spot lights on
screen, but none of them cast: the only shadow source was the directional CSM.
This subsystem gives punctual lights a shared depth atlas so spot lights shadow
the same geometry the sun does.

Status: **spot lights cast; point lights do not yet.** A point light needs six
cube faces, and the atlas pass only records one projection per slot so far.
Point lights are shaded normally, they just never occlude.

## Why an atlas rather than per-light shadow maps

A shadow map per light means a texture, a view, and a descriptor per light,
which does not survive a light count in the hundreds. One atlas keeps it to a
single image and a single descriptor binding, and turns "which lights cast this
frame" into a tile-allocation problem instead of a resource-allocation problem.

The atlas is 4096x4096 split into a uniform 8x8 grid of 512px tiles — 64 slots.
Uniform tiles make allocation a bump counter. Variable tile sizes (so a nearby
light gets a sharper map) are deliberately absent until there is a priority pass
to decide who deserves the big tiles; see [Limitations](#limitations).

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
| Slots used | Tiles allocated this frame, out of 64 |
| Depth bias | Constant offset applied to the compared depth |
| Normal bias | World-space offset along the surface normal before projecting |
| Atlas depth preview | Samples the atlas image directly, so occupied tiles and their depth content are visible |

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

- **Point lights do not cast.** Six cube faces per light are not recorded yet.
- **No priority or budget.** Slots are handed out in light order until the atlas
  runs dry; an over-budget scene silently drops the lights that come last rather
  than the least important ones. This only starts to matter once point lights
  land, since each one costs six tiles and the demo swarm goes to 512 lights.
- **Uniform tile size.** A light one metre away gets the same 512px tile as one
  at the far plane.
- **CPU caster culling.** Each slot frustum-tests every draw item on the CPU
  rather than going through the existing GPU shadow-cull path.
- **No filtering beyond 3x3 PCF.** No variance/moment maps, no contact-hardening.
- **The atlas preview cannot show perspective depth.** See [Controls](#controls);
  the caster-draw counter covers the diagnostic need in the meantime.
