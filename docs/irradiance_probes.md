# Irradiance Probes (Real-Time Global Illumination)

Every lighting term in this engine before probes was either direct or
screen-space. IBL supplied a constant environment irradiance that knew nothing
about the scene; SSR and GTAO only know what is already on screen. Nothing
carried light from one surface to another, so a wall could not tint the floor
beside it and a shadowed alcove was lit by exactly the same ambient value as an
open field.

This subsystem is a grid of probes that gather the scene and hand that light
back to shading: indirect diffuse, from geometry the camera cannot see.

## Why not DDGI

The obvious answer for probe GI in 2020s renderers is DDGI, where each probe
fires rays every frame and the update is a ray-tracing dispatch.

**This was chosen on a target that reported neither `VK_KHR_ray_query` nor
`VK_KHR_acceleration_structure`.** Of the 131 device extensions an Apple M3
through MoltenVK exposes, none are ray tracing, so DDGI as normally implemented
was not merely slow there, it was unavailable.

That constraint is device-specific and no longer holds everywhere this builds.
On an RTX 3080 Ti Laptop (NVIDIA 616.56) `rayQuery`, `accelerationStructure` and
`VK_KHR_ray_tracing_pipeline` are all present, so DDGI is available on that
machine and the rasterised capture below is a portability floor rather than the
only option. Note `rayTracingInvocationReorderReorderingHint` is `NONE` on
Ampere -- shader execution reordering is Ada hardware, so a DDGI gather here has
to tolerate divergence rather than assume it is reordered away.

Where ray query is present, the capture can now be traced instead of rasterised
(`gi.rayTracedGather`, off by default): see [The ray-traced
gather](#the-ray-traced-gather). It replaces only the capture, exactly as the
paragraph above predicted, and by default it samples the way DDGI does: a fixed
budget of rays per probe, evenly spread and randomly rotated on every update
([Rotated rays](#rotated-rays)).

What would carry over unchanged is most of this document: the octahedral tile
layout, the border texel that fixes the seam, and the Chebyshev visibility
weight that stops light leaking through walls are properties of how probe
irradiance is *stored and sampled*, not of how its radiance was gathered. Only
the capture would be replaced.

So probe radiance is gathered the way it was before ray tracing: **rasterise the
scene from the probe's own position** into a small cube, then convolve that cube
into the probe's stored irradiance. The technique is older and the update is
heavier per probe, which is what makes the amortisation below load-bearing
rather than an optimisation.

One thing that looked reusable and was not: the engine's existing IBL
convolution is entirely **CPU-side** — `createDiffuseIrradianceFromRgba32fFaces`
and friends operate on CPU pixel spans and upload the result. Feeding a GPU
capture through it would need a readback per probe per frame. The convolution
here is its own compute pass.

## Storage: two octahedral atlases

The grid is **8 x 4 x 8 = 256 probes**, deliberately coarse. Irradiance is low
frequency, and the cost of a probe here is a scene capture rather than a handful
of rays, so resolution goes into update rate instead of probe count.

Each probe owns one tile in each of two atlases.

| | format | core per probe | atlas | size |
| --- | --- | --- | --- | --- |
| Irradiance | RGBA16F | 8x8 | 320x80 | 200 KB |
| Visibility | RG16F | 16x16 | 576x144 | 324 KB |

Both are formats Vulkan **requires** storage-image support for, so there is no
format-fallback path to get wrong.

Visibility is stored at twice the resolution because it does a different job.
Irradiance is low frequency by construction — it is an integral over a
hemisphere. Visibility has to say where a wall is, and a wall is an edge.

It stores **mean distance and mean distance squared**, the two moments a
Chebyshev bound needs. Distances are clamped to `kProbeMaxDistance` (64), which
is bounded so the squared channel stays inside a half float's range — the
squared channel is what runs out first, not the linear one.

Tiles run X then Y across an atlas row and Z down the rows, so a tile coordinate
follows from the linear probe index with the same X-fastest order `probeIndex`
uses. One atlas row is one horizontal slab of the grid, which is what makes the
debug preview readable without counting texels.

### The border is not padding

A tile is its octahedral square **plus one texel on every side**, and that
border is the part most worth getting right.

The octahedral square's edges are seams: texels on opposite sides of an edge are
neighbours on the sphere but far apart in the image. Hardware bilinear filtering
knows nothing about that, so a sample near an edge blends against a texel
pointing somewhere else entirely. The border holds the wrapped copy of the
interior texels the seam actually joins to, which turns the whole tile into
something a plain linear sampler can read.

**The square folds onto its own edge**, mirrored about that edge's midpoint. So:

- a border row or column is the **adjacent** core row or column, reversed;
- a border corner is the **diagonally opposite** core corner.

Wrapping to the *opposite* edge — the way an equirectangular map wraps — is the
plausible mistake, and it is wrong by most of a hemisphere.

The UV mapping puts every lookup inside the probe's own core square, at least
half a texel from the tile's outer edge. The widest bilinear footprint therefore
reaches this tile's border and stops there; it can never reach the neighbouring
probe.

## Capture

Six cube faces per probe, rasterised into a scratch atlas that holds only the
probes being captured right now — one row per capture slot, one column per face.

Six rasterisations rather than one octahedral pass because the octahedral map is
not a projective transform: no vertex shader can produce it.

The face order, the projections and the face directions are the ones
[punctual_shadows.md](punctual_shadows.md) already defines for point-light shadow
cubes. Reused rather than re-derived — a second cube convention in the same
engine is exactly the kind of thing that produces plausible-but-rotated lighting.

The capture pass is one `vkCmdBeginRendering` over the whole capture atlas with a
viewport per (probe, face), the same arrangement the punctual shadow atlas uses
and for the same reasons: the clear runs once for the image, and 96 tiles do not
each pay for a render pass.

### Face resolution is set by the convolution, not the raster

16x16 per face. The rasterisation would happily be finer — it is a handful of
tiny viewports — but **every convolution output texel integrates over every
captured texel**, so this squared times six is the inner loop. 16 gives 1536
samples per probe, the same order as DDGI's 256 rays. Doubling it would
quadruple the convolution.

### What the capture shades

| kept | dropped |
| --- | --- |
| Albedo, lambert from the directional light | Specular of every kind |
| Lambert from every punctual light | Normal mapping |
| Cascaded shadows, punctual shadows | IBL |
| Ambient, emissive | |

The dropped terms describe view-dependent detail, and nothing view-dependent
survives an integral over 8x8 directions. Dropping them is free, not an
approximation.

Punctual lights are read as a **flat array**, not through the per-froxel cluster
lists. Those lists are built for the camera's view frustum, and a probe is not
on screen — there is no froxel to look it up in. Looping every light instead is
affordable precisely because the capture is tiny; see the measurements below,
which say the light loop is not what costs.

### The clear value is the sky, not black

The capture clears to the scene's ambient term. Black would be a correctness
error rather than a cosmetic one: outdoors most of a probe's hemisphere is sky,
and treating it as unlit makes every probe far too dark **while still looking
like plausible, merely moody, indirect light**.

Found by reading the atlas back, not by looking at it:

| capture clear | min | mean | non-zero texels |
| --- | --- | --- | --- |
| Black | 0.000 | 0.020 | 66570 / 76800 |
| Ambient | 0.047 | 0.143 | 76800 / 76800 |

## The ray-traced gather

`gi.rayTracedGather` fills the same capture atlas with rays. `ProbeTrace` is one
compute dispatch, a workgroup per (cube face, capture slot) and an invocation per
texel, and each invocation fires one ray from the probe along **exactly the
direction the convolution will read that texel back as** -- the jittered
`probeCubeTexelDirection`, now shared by both shaders through
`probe_cube.glsl`. Everything after the capture atlas is unchanged: convolution,
border, hysteresis, multi-bounce, the shading lookup. Without ray query or the
bindless path the setting stays a request and captures stay rasterised; the log
says which gather ran (`Irradiance probe gather: ...`).

Making the trace land in the raster capture's texels, rather than in a ray
layout of its own, is what made it checkable. The two gathers can be compared
image to image under one convolution, so a difference is a difference in what
was gathered and nothing else.

### What a hit shades

The surface shading is one function, `probeCaptureRadiance` in
`probe_capture_shading.glsl`, called by the raster capture per fragment and by
the trace per hit, along with the punctual-light loop, its shadow-atlas tap and
the multi-bounce lookup. Before this they lived in `probe_capture.frag` alone; a
second copy is how the two gathers would have drifted apart. What is per-gather:

| | raster capture | ray-traced gather |
| --- | --- | --- |
| Surface | rasterised, closest by depth | closest hit; cutouts confirmed where their alpha test passes, using the shadow rays' own test |
| Normal | vertex normal, flipped to face the probe | the same, interpolated from the vertex buffer by device address |
| Sun visibility | the cascades, sampled by camera view depth | **a shadow ray** from the hit |
| Texture level | derivatives at 16x16 | a ray cone (`probeRayConeTextureLod`) |
| Nothing hit | the clear: ambient, far distance | the same values, written by the miss |

The shadow ray is the one deliberate change in the result. The cascades are
fitted to the camera, so the raster capture shadows a surface only if the camera
happens to have it inside its cascade range; a probe behind the camera, or past
the last split, gathers unshadowed sunlight. A ray answers anywhere.

The texture level matters more than it looks. A ray has no derivatives, and
level 0 would alias a whole texture down to the one texel each ray happens to
land on. The cone is the angle one capture texel spans at a face's centre (2 / 16
of a radian), widened by the hit's incidence; the level is the one whose texels
are as wide as the cone where it lands, from the hit triangle's world and UV
areas. The C++ mirror is unit-tested for the doubling rule, texture size, tiling,
the grazing clamp and degenerate input, and for the spread being the angle
between adjacent capture texels.

The skinned caster is the one instance whose geometry record names bare
pre-skinned positions rather than `renderer::Vertex`, because the shadow rays
never looked at an opaque hit's attributes. Its TLAS instance now carries
`kRtSkinnedInstanceTag` in its shader-binding-table offset, a field ray queries
never consume, and a hit on it shades with the face normal and an untextured
lookup. The pre-skin's barrier now also makes its positions visible to compute
shader reads.

### Rotated rays

The first version of the trace fired one ray per capture texel, along the cube
directions the raster capture sees: 1536 rays a probe, the same 1536 every update
apart from the sub-texel jitter. `gi.raysPerProbe` (default 768) replaces that
with DDGI's sampling. Each probe fires that many rays along a spherical Fibonacci
set -- equal-area bands, golden-ratio azimuth steps, so any count covers the
sphere evenly -- and the whole set is turned by a fresh, uniformly random
rotation on every update (Shoemake's method, seeded from the update index so a
run is reproducible). Zero keeps the per-texel cube rays.

Two things make this cheap to carry:

- **The rays live in the capture atlas.** A capture slot holds 1536 texels, so
  up to 1536 rays per probe wrap through the slot's own rows. No new image, no new
  graph resource, nothing for the update pass to declare differently.
- **The convolution only changes its sample list.** Each sample is a direction, a
  share of the sphere and a recorded radiance and distance; for rotated rays the
  share is a constant the normalisation divides out, where cube texels need their
  solid angle. The cube path is the original nested loop and renders the probe
  view bit-identically to before.

The trace's push block had no room for a rotation, so the batch's probe indices
went from four `ivec4`s to one `uvec4` of 8-bit indices (the grid has 256 probes),
which the unit tests pin, along with even coverage of the sphere, rigid and
uniformly distributed rotations, and one texel per ray inside its slot.

#### Measured: fewer rays, better result

Against a converged reference -- 1536 rotated rays at hysteresis 0.97, 2000
frames -- each layout at the default hysteresis 0.7, 16 probes a frame, scene
frozen, over 32 consecutive frames of the probe-only view (debug gain 0.5).
Error is the mean absolute difference from the reference; flicker is the
per-pixel standard deviation over the 32 frames, in 8-bit steps:

| rays per probe | Cornell error | Cornell flicker, mean / p99 | default error | default flicker, mean / p99 |
| --- | --- | --- | --- | --- |
| 1536, fixed cube texels | 0.0068 | 0.19 / 1.28 | 0.0032 | 0.19 / 0.86 |
| 256, rotated | 0.0071 | 0.25 / 1.64 | 0.0048 | 0.30 / 1.65 |
| 512, rotated | 0.0066 | 0.16 / 1.61 | 0.0025 | 0.15 / 0.80 |
| **768, rotated** | **0.0062** | **0.10 / 0.71** | 0.0029 | **0.11 / 0.45** |
| 1536, rotated | 0.0061 | 0.05 / 0.50 | 0.0029 | 0.10 / 0.44 |

At the same 1536 rays, rotating them cuts the flicker two- to fourfold: the
fixed directions alias, and the jitter cycles through a short sequence of offsets
that the hysteresis then shows as a pattern. 768 rotated rays already beat 1536
fixed ones on every column, which is what sets the default. The flicker figures
are below one 8-bit step, so they rank the layouts rather than measure them
absolutely.

The convolution's inner loop is the sample count, so the default halves the
pass that dominated the probe cost:

| Sponza, 4 probes / frame, traced gather | cube texels (0) | 768 rotated | change |
| --- | --- | --- | --- |
| Frame total | 5.176 | 4.681 | **-9.6%** |
| `IrradianceProbeUpdate` | 0.692 | 0.291 | **-57.9%** |
| `ProbeTrace` | 0.047 | 0.038 | -19.1% |

Same protocol as the table above (pinned 800 / 7001 MHz, A/B/A/B/A/B, p10 in ms),
control drift 0.72%. The first two attempts at this series drifted 4.0% and
7.5% and were discarded rather than quoted.

One regression was found and undone on the way. The first version folded the cube
path into the same flat loop as the rays, indexing face and texel by division;
the cube convolution then read 0.761-0.784 ms in three series against 0.692 with
the original nested loops restored. The nested loops are what ships.

### What it changes about the frame

- **No per-probe CPU work.** The raster capture frustum-culls and records every
  draw item for six faces per probe; the trace is one push and one dispatch
  whatever the scene holds.
- **The cascades can go.** The trace reads no cascaded shadow map, so on a frame
  that also traces the camera's shadows (`rayTracedShadows.enabled`) nothing
  reads the cascades any more and no cascade is redrawn -- the raster capture was
  one of the readers that kept them alive. `CSMShadowPass` is still declared and
  records nothing, the same as a frame whose cascade cache hit.
- **The TLAS is built for either consumer.** The scene is traced when the sun's
  shadow rays or the probe gather need it; the main passes still declare their
  TLAS read only on frames that trace the sun, so a frame on which only probes
  trace leaves the main shader on its shadow-map path. `RayTracingSceneBuild` is
  declared ahead of the probe passes now, since `ProbeTrace` is its first reader.
- **The probe parameter buffer is visible to compute.** Its update barrier used
  to name the fragment stage only, which would have let the trace read last
  frame's grid position.

### Checked against the raster capture

The probe-only view (`gi.debugIrradianceOnly`, debug gain 0.5, 16 probes a frame,
`--deterministic`, frame 130, RTX 3080 Ti):

| scene | mean, raster | mean, traced | MAE | pixels off by more than 2/255 |
| --- | --- | --- | --- | --- |
| Cornell box | 0.5357, 0.5428, 0.5135 | 0.5357, 0.5427, 0.5134 | 0.0006 | 1.4% |
| Default | 0.2769, 0.2787, 0.2838 | 0.2841, 0.2784, 0.2828 | 0.0034 | 13.9% |

The Cornell box has the sun off and is untextured, so the two gathers ought to
agree there, and they do, red/green split included (1.037 left and 0.941 right in
both gathers). That is the test that directions, texel placement, normals and
shading match. The default scene has a sun and differs by more, with no pixel off
by more than 16/255. That is where the shadow ray and the cascades disagree,
though the attribution is by elimination rather than measured term by term.

The match proves nothing if the traced result is not what is displayed, so it has
a negative control. Tinting the trace's output red turned the Cornell probe view
from (0.536, 0.543, 0.513) to (0.536, 0.189, 0.223).

### Cost

Sponza, Release, RTX 3080 Ti Laptop with clocks pinned at 800 / 7001 MHz,
`--every-frame --freeze-animation`, interleaved A/B/A/B, p10 in ms. A is the
raster capture and B the traced gather, every other setting identical:

| configuration | frame, raster | frame, traced | change | ProbeCapture | ProbeTrace | control drift |
| --- | --- | --- | --- | --- | --- | --- |
| 4 probes / frame (default) | 6.102 | 5.107 | **-16.3%** | 0.510 | 0.041 | 0.95% |
| 16 probes / frame | 7.003 | 5.024 | **-28.3%** | 1.920 | 0.036 | 0.10% |
| 4, ray-traced sun shadows on | 5.045 | 4.557 | **-9.7%** | 0.509 | 0.041 | 0.44% |
| 4, ray-traced sun shadows on, cascade cache off | 6.492 | 4.613 | **-28.9%** | 0.513 | 0.041 | 0.48% |
| Cornell box, 16 probes / frame | 1.799 | 1.757 | -2.3% | 0.069 | 0.033 | 0.17% |

Four things the table says:

- **The raster capture's cost is the scene's, the trace's is not.** On Sponza the
  raster capture scales with what it draws -- 0.51 ms at 4 probes, 1.92 ms at 16
  -- because every probe rasterises six faces of 103 primitives. The trace stays
  near 0.04 ms at either count. On the Cornell box, a handful of boxes, the two
  are close, which is the same statement from the other end.
- **The frame saves more than the two passes differ by**, about twice as much at
  4 probes (0.995 ms against 0.469). The pass timestamps do not cover whatever
  the raster capture costs between passes, so the frame total is the number to
  quote; where the rest goes is not measured.
- **The cascade case is where tracing the sun pays twice.** With the cascade
  cache off -- standing in for a camera that moves, which invalidates it -- the
  raster capture keeps all four cascades redrawn (`CSMShadowPass` 0.785 ms,
  `ShadowGpuCulling` 0.027), and the traced gather removes the last reason to.
  With the cache on and the camera frozen neither side redraws, so that row shows
  the gather alone.
- **The convolution got slightly dearer**: `IrradianceProbeUpdate` +0.031 to
  +0.043 ms (+4-7%) in every series, attributable each time. It runs the same
  shader over the same atlas either way; the cause is not isolated.

Validation layers are compiled out of Release, so these runs say nothing about
validation. The traced legs are clean under validation and synchronization
validation in Debug -- with the standing caveat that synchronization validation
sees neither acceleration-structure accesses nor device-address reads, both of
which this pass makes.

## Convolution

One workgroup per probe, 16x16 threads. Every thread walks the whole capture and
accumulates the texels on its own side of the sphere. That looks quadratic and is
fine at these sizes: the inner loop is 1536 iterations and every thread in the
workgroup reads the same texel at the same time, so the loads broadcast out of
cache rather than scattering.

Threads do double duty. All 256 compute a visibility texel; the first 8x8 also
compute an irradiance texel. Two dispatches would read more clearly but would
fetch the whole capture twice, and the fetch is the entire cost.

| | filter | result |
| --- | --- | --- |
| Irradiance | cosine lobe, normalised | mean radiance arriving at a surface facing that way |
| Visibility | cos^20 lobe, normalised | mean distance and mean distance squared |

Normalising by the accumulated weight rather than multiplying by a constant is
what makes the result independent of how the capture is discretised.

### Solid angle is required, not a refinement

Cube texels are not equal in solid angle: a face's corners subtend up to three
times less than its centre. Integrating without the Jacobian over-counts the
corners of every face, and the result is a **smooth bias** — the GI comes out "a
bit wrong" everywhere rather than visibly broken somewhere, which is the hardest
kind of error to notice.

## Shading lookup

The main fragment shader blends the eight probes surrounding each shading point.
Each probe's weight is the product of three things:

1. **Trilinear** position in the grid cell.
2. **Backface rejection**, a wrapped cosine rather than a cutoff. A hard cutoff
   drops probes discontinuously as a surface turns, which draws a line across
   otherwise smooth shading where no geometry changes.
3. **Visibility**, a Chebyshev bound from the stored moments, cubed.

The eight weights are renormalised afterwards, so a point beside a wall takes its
light from the probes it can still see rather than simply going dark.

Both the visibility and backface terms carry a floor. A weight that reaches
exactly zero puts a hard edge in the shading where it crosses, and that edge
corresponds to nothing in the scene.

### Visibility is the whole reason the depth atlas exists

Trilinear blending on its own happily takes light from a probe on the far side of
a wall. That failure reads as **a room being softly lit from nowhere** rather
than as an addressing bug, and it is the characteristic artefact of probe GI.

The bound is cubed because the raw Chebyshev falloff is far too gentle to read as
a wall — light bleeds a long way past the occluder.

**`abs()` on the variance is load-bearing, not defensive.** The moments are read
through a bilinear filter, and interpolating two texels can produce a mean square
below the squared mean even though no single texel can. The variance then goes
negative and the weight comes back negative, which *subtracts* light — a black
fringe that reads as an occlusion artefact rather than as arithmetic. A unit test
pins it.

### Sampling is offset off the surface

Biased along both the normal and the view direction. A normal-only offset barely
clears the surface at grazing angles, which is exactly where a plane otherwise
samples probes whose stored visibility says the plane itself is in the way — and
shades itself black.

The bias is the one parameter that has to be set against a scene rather than
derived. Too small and flat surfaces self-occlude; too large and light leaks
through thin geometry.

### Where the grid parameters live

A uniform buffer at **set 0 binding 11**, alongside the two atlases at bindings 9
and 10. Neither of the two obvious homes worked:

- **Push constants**: the main pass's block had 24 bytes left and this needs 32.
- **`ObjectFrameData`**: it would fit, but that struct's layout is duplicated in
  six shaders and the array stride depends on it. A missed one reads every object
  past the first at the wrong offset — silently, and only for objects that are
  not the first.

One buffer rather than one per frame, refreshed by a 48-byte `vkCmdUpdateBuffer`
inside the frame that reads it, so there is no in-flight host write for a
descriptor to have to avoid.

Probe irradiance **replaces** the constant environment irradiance rather than
adding to it: both answer "what diffuse light arrives here", and summing them
would double-count. Intensity carries the off state, so nothing downstream needs
a separate flag — the same shape `fogMaxDistance` and the punctual shadow slot
sentinel use.

## Multi-bounce

The capture reads the irradiance the grid already holds and uses it as the
surface's indirect light. The next capture then sees light that has bounced one
more time, so bounces accumulate over cycles rather than being computed in one
pass. Nothing new is bound for it: the capture already binds the material
descriptor set, which is where both probe atlases and the grid parameters live.

Indirect light is **interpolated** between the constant ambient and the grid
value rather than summed. Summing would count the sky twice — the atlas already
contains it, having been captured against a sky-coloured clear — and it would
make the feedback unbounded instead of a convex combination.

It is a feedback loop, and the weight is what bounds it. Each round multiplies by
`albedo * weight`, so the series settles at `1 / (1 - albedo * weight)`;
`probeBounceAmplification` is that number, and the debug panel reports it for a
bright surface so the knob is not just a slider with a shrug attached. The
weight is capped below one because at exactly one a perfectly white surface does
not converge — physically correct and useless as a renderer.

The weight is held at zero until every probe has been captured once. Before that
most of the grid still holds its neutral seed, and blending a surface toward that
would make the scene *darker* — the opposite of what another bounce should do.

### Measured, and the scene decides the answer

The same measurement on two scenes, with the probe term rendered on its own:

| scene | bounce 0 | bounce 0.95 | change |
| --- | --- | --- | --- |
| Open platform, strong sun | 0.2771 | 0.2833 | +2.2% |
| Cornell box | 0.4517 | 0.5920 | **+31%** |

On the open scene the effect was barely above noise, and that is not a defect —
most of what a probe there sees is either directly lit or sky, so the grid's
irradiance is already close to the ambient constant it replaces. The closed room
is where light has somewhere to be trapped, and there a second bounce is a third
of the indirect term.

It also strengthens the colour bleeding, which is the part a single bounce cannot
do at all: light that has hit the red wall twice is redder than light that hit it
once. Mean irradiance ratio of the probes on each side of the room:

| bounce weight | left (red wall) R/G | right (green wall) R/G | separation |
| --- | --- | --- | --- |
| 0.0 | 1.197 | 0.965 | 1.24x |
| 0.95 | 1.370 | 0.923 | **1.48x** |

The loop is stable at every weight tested — the trajectory settles rather than
climbing.

## Amortised update

Probes are captured round robin, a few per frame. That is the whole cost control,
and it sets two things at once: what a frame costs, and how many frames the grid
takes to catch up with a change.

Plain round robin rather than importance ordering. The property that matters is
that any one probe's staleness is **bounded** — a scheme that picked probes by
importance would let a probe nothing currently looks at go stale indefinitely,
and staleness in GI reads as light lagging the scene rather than as a missing
update.

### Accumulation needs jitter, or it does nothing at all

This is the part of the design that is not obvious, and getting it wrong would
have shipped something that looked implemented and did nothing.

Unlike DDGI, which fires randomly rotated rays each frame, **this capture is
deterministic**: the same 1536 fixed directions every update. Re-capturing an
unchanged scene reproduces the previous result bit for bit, so blending
successive captures averages identical numbers.

So each update slides the whole capture by a sub-texel offset from a Halton
(2, 3) sequence — the same sequence the TAA jitter uses — and the convolution
subtracts the same offset when it reconstructs directions. Successive captures
then sample different points inside each texel, and the accumulated result
carries angular detail no single 16x16 capture holds.

The jitter is applied as a **clip-space translation after the view-projection**,
not a term poked into the combined matrix. See [Traps](#traps-worth-recording).

Hysteresis is held at **zero until every probe has been captured once**. With a
round-robin cursor, most of the grid is still holding its neutral seed through
the first cycle, and blending against that would leave the whole grid permanently
darker than the scene — an error that would look like the GI simply being weak.

It is also lower than a per-frame accumulator would use (0.7 against DDGI's
~0.97), because a probe here is re-captured once per *cycle* rather than every
frame: the same hysteresis costs sixty times the latency.

## The Cornell box scene

Under **Scene** tab → **Scene Presets → Cornell box**. A closed room with white floor, ceiling
and back wall, one red and one green side wall, two blocks, and a single light
below the ceiling. Open toward the camera, because a sealed box has nothing to
look through.

It exists because the default scene cannot show this subsystem working. An open
grey platform under strong sun has almost nothing to bleed and almost nothing to
occlude, so every measurement taken on it lands near the noise floor however
correct the implementation is. That is a property of the scene, and no amount of
work on the probes fixes it.

Loading it does four things beyond placing geometry, each of which the scene is
useless without:

- **Switches the sun off.** It would flood the room through the open side and
  swamp the one thing the scene exists to show.
- **Replaces the orbiting light swarm with one overhead light.** The swarm lights
  every surface directly; the floor between the blocks has to be lit by bounce
  alone or there is nothing to look at.
- **Fits the probe grid to the interior**, inset so no probe lands inside a wall
  — a probe in solid geometry sees only that geometry and hands it to everything
  nearby.
- **Turns probes on**, since that is the point.

Surfaces are matte and the white is deliberately not 1.0: colour bleeding scales
with saturation, specular would not survive the convolution anyway, and a
perfectly white room is exactly the case where the bounce feedback series stops
converging.

What it demonstrates, measured rather than asserted — mean probe irradiance on
each side of the room:

| probes | R | G | R/G |
| --- | --- | --- | --- |
| Left half (red wall) | 1.067 | 0.841 | 1.269 |
| Right half (green wall) | 0.963 | 1.019 | 0.946 |

A 34% swing in the red/green ratio across the room, from walls the probes never
see directly. That is indirect light carrying colour, which is the whole claim.

## Pass structure

```
CSMShadowPass            directional cascades
PunctualShadowAtlasPass  spot/point tiles
ClusterBuild / LightCull
ProbeCapture             graphics: 6 faces x N probes into the capture atlas,
                         reading last frame's probe atlas for the extra bounce
  or ProbeTrace          compute: one ray per capture texel into the same atlas
                         (gi.rayTracedGather), after RayTracingSceneBuild
IrradianceProbeUpdate    compute: convolve, then wrap the octahedral border
MainHDRPass              samples both probe atlases
```

Capture sits after both shadow passes because the radiance it records is
shadowed by them, and after the cluster passes only because it is convenient —
it does not use their output.

### Render graph integration

Both probe atlases and both capture targets are imported. The capture pass
declares its reads of the cascaded shadow map and the punctual atlas, so neither
is still a depth attachment when it samples them.

Two subtleties are load-bearing:

- **The main pass declares its read of the probe atlases whenever they exist**,
  not only when a probe updated — the same asymmetry the punctual shadow atlas
  uses. Without it a cold start would leave the images in `UNDEFINED` while a
  material descriptor claimed a sampled layout. This engine has hit that class of
  bug twice.
- **The update pass is marked a side effect** even though its outputs are graph
  resources with a declared reader. The atlases persist across frames by design,
  so the pass's real consumer is the *next* frame's update, which liveness
  analysis cannot see. Without it the pass survives only because the main pass
  declares a read it does not yet sample, and removing that read culls the pass
  while the renderer still records it — which throws rather than degrades.

## Measurements

Demo scene, **Release build**, min and median over ~25 samples.

### GPU

| probes/frame | ProbeCapture | IrradianceProbeUpdate |
| --- | --- | --- |
| 4 (default) | 0.003 / 0.006 ms | 0.202 / 0.263 ms |
| 16 (max) | 0.085 / 0.333 ms | 0.342 / 0.468 ms |

**The convolution dominates, not the capture.** Even with every punctual light
evaluated for every capture fragment, the capture is cheaper than the
convolution's 1536-sample inner loop. Per-probe light culling is therefore not
implemented — there is no measurement that justifies it. The lever that matters
is the face resolution, which the convolution's cost is quadratic in.

These numbers match the Debug build's almost exactly, and that is not an
accident worth being suspicious of: the shaders are compiled by the same `glslc`
invocation regardless of `CMAKE_BUILD_TYPE`, so the SPIR-V is **byte-identical**
between builds, and these are timestamp queries over that identical binary. The
usual caution about Debug timings applies to CPU work, not to this.

### CPU

Which is where it does apply. Culling and recording the capture pass is the one
probe cost a build type actually changes, and it changes it by an order of
magnitude:

| probes/frame | Release | Debug |
| --- | --- | --- |
| 4 (default) | 25 / 45 us | 168 / 597 us |
| 16 (max) | 50 / 158 us | — |

At 16 probes the pass records 234 draws on this scene — six frustum-culled faces
per probe, from a scene of 11 draw items. That is the number that scales with
scene complexity rather than with probe resolution, and it is the one to watch on
a heavier scene.

### Effect on the image

From the engine's own scene-luminance readout:

| | average scene luminance |
| --- | --- |
| Lookup off (constant IBL) | 0.3044 |
| Intensity 1 | 0.2673 |
| Intensity 4 | 0.3965 |

Punctual lights in the capture, measured with the probe term rendered on its own
so the whole image is the signal:

| | probe-only luminance |
| --- | --- |
| Directional only | 0.1497, essentially static at the ambient value |
| With punctual lights | 0.26-0.28, varying as the demo's lights orbit |

Accumulation, over a 26-second run with the lights orbiting:

| hysteresis | per-sample stdev | range |
| --- | --- | --- |
| 0.0 | 0.000312 | 0.001100 |
| 0.95 | 0.000189 | 0.000700 |

## Controls

The **Probes** tab of the Inspector window, not a section of the main panel,
and not behind the advanced-mode gate. The atlas previews are how this subsystem
is checked at all, and they want to stay visible *while* the probe controls are
changed, so the controls and the previews share one tab beside the main panel
rather than competing for the main panel's height. Loading the Cornell box
preset opens the Inspector on this tab.

| Control | Effect |
| --- | --- |
| Enable irradiance probes | Master toggle. Off still seeds the atlases once so nothing samples them out of an undefined layout |
| Probes per frame | Round-robin batch size; 0 pauses capture without losing the cursor |
| Intensity | How strongly probe irradiance replaces the constant environment term. Zero disables the lookup outright |
| Surface bias | How far off a surface the grid is sampled from. The one parameter that wants tuning per scene |
| Hysteresis | Fraction of a probe's previous value kept on re-capture. Zero overwrites, which is the reference to compare against |
| Debug: probe irradiance only | The indirect term alone, bypassing exposure and tone mapping |
| Debug pattern | Fills tiles with the direction each texel stands for instead of captured radiance |
| Debug gain | Display gain for the previews and the probe-only view |

**The probe-only view bypasses the display pipeline deliberately.** Auto-exposure
exists to cancel overall brightness changes, so a debug view of a term that *is*
an overall brightness change shows almost nothing however correct it is —
toggling intensity moved the measured scene luminance by 12% and the displayed
image by nearly zero. The bypass skips ambient occlusion, bloom, exposure and
tone mapping.

Two caveats on that view: it reports the gathered irradiance **before** intensity
is applied, so that slider keeps its own meaning instead of doubling as a
brightness knob; and its background is the **skybox**, not a probe value — the
skybox is drawn by its own pass, whose push constants are already at the 128-byte
guaranteed minimum with no room for a debug flag.

The **Debug pattern** is the standing check on the octahedral border: with it on,
a correct border makes neighbouring tiles meet with no seam, and a wrong one
draws a one-texel frame around every tile.

## Testing

`tests/test_irradiance_probes.cpp`, covering the GPU-free core the way
`ClusterGrid.h` and `PunctualShadowAtlas.h` are covered. The tests worth
describing are the ones aimed at failures that are *quiet*.

**The octahedral border, three ways.** A structural test pins the rule (corners
diagonal, edges reversed, core texels identity). That is not enough on its own —
a mirrored-the-wrong-way rule still lands in the core and passes every bounds
check. So a second test fills a tile with the direction each texel stands for,
filters it the way the GPU would, and measures the angle error, against both
wrong borders for contrast:

| border rule | max angle error at the seam |
| --- | --- |
| Octahedral (correct) | 0.1565 rad — equal to the tile interior's error to five significant figures |
| Clamp-to-edge | 0.2403 rad |
| Torus wrap | > 0.5 rad |

Clamp-to-edge being only 1.5x worse is the point: the mild mistake survives
casual inspection, which is why the test measures rather than eyeballs.

**Solid angle by convergence, not by tolerance.** The weights over all six faces
must sum to 4*pi. Because that sum is a midpoint rule it approaches the answer
rather than hitting it, so the test requires the error to *keep falling* as the
grid refines — a plausible-but-wrong Jacobian can land within a few percent at
one resolution, but only the right one converges.

**Constant radiance in, constant radiance out.** A uniform field must convolve
back to exactly that constant in every direction, which pins normalisation and
solid-angle weighting together.

**Capture directions against the real matrix.** Each texel's direction is
projected back through the actual face view-projection and must land on that
texel's own pixel. This is the test that catches a basis derived wrongly, and it
did — twice.

**Chebyshev visibility.** Monotonic past the occluder, in [0,1], sharper
occluders reject harder than blurry ones, filtered moments that give a negative
raw variance still produce a finite non-negative weight, and a NaN distance
decodes to visible rather than sliding into the divide.

**Round robin visits every probe once per cycle**, for several batch sizes,
because bounded staleness is the property — not "eventually".

## Traps worth recording

**Absence of validation errors proved nothing, twice.** The ImGui atlas preview
was being *clipped*, so no draw was issued — and a deliberately wrong image
layout claimed for it did not trip validation either. The cold-start layout was
eventually verified by reading back the layout pointer the render graph writes
(`UNDEFINED` on frame 1, `SHADER_READ_ONLY_OPTIMAL` from frame 2), and the
preview only by re-running the same negative control after moving it into its own
window, where it produced
`VUID-vkCmdDrawIndexed-imageLayout-00344` as it should have all along.

**The jitter went into the wrong matrix.** Offsetting column 2 of the *combined*
view-projection scales the shift by world z rather than view z, so it varied
across the face instead of being the constant sub-texel slide it had to be. The
round-trip test caught it, after first catching a plain sign error in the same
line. It is now a clip-space translation applied after the view-projection.

**A measurement that could not resolve its signal.** Scene luminance showed no
difference between hysteresis settings, because probes are a small share of it
and the two runs tracked each other almost value for value. Rendering the probe
term alone makes the whole image the measurement, and the damping is then plain.
The first measurement was not evidence of no effect; it was evidence of an
insufficient instrument.

**The debug view clipped at its own defaults.** It multiplied the gathered
irradiance by the user's intensity before display, so at intensity 1 and gain 6 a
typical value came out at 1.5 and the screen went white — the only way to see
anything was to drag a setting with a real meaning down to 0.2 and use it as a
brightness knob.

## Limitations

- **Indirect diffuse only.** No indirect specular. A mirror still reflects only
  what SSR can find on screen.
- **Multi-bounce is a fixed weight, not an energy-exact solve.** It converges to
  the right shape but the weight is a damping factor chosen for stability rather
  than derived, so absolute indirect intensity is a tuning parameter.
- **The grid is fixed and hand-placed.** Origin and spacing are settings. There
  is no fitting to scene bounds, no cascaded or nested volumes, and no probe
  relocation — a probe that lands inside geometry stays there and contributes
  whatever it sees from in there.
- **A probe outside the volume clamps to the edge probes.** Correct in the sense
  of bounded, but a large scene with a small grid is lit by its border.
- **Visibility is a statistical bound, not an occlusion test.** Thin geometry,
  and surfaces close to a probe relative to the 16x16 visibility resolution, can
  still leak.
- **The depth lobe exponent is blunted** to 20 against DDGI's 50, to stay stable
  at 1536 samples. Sub-texel jitter reduces aliasing but does not decorrelate
  captures the way a random rotation would, so raising it is not yet earned.
- **Capture shading is not the main pass's shading.** No normal maps, no
  specular, and the lambert is unnormalised, so probe irradiance is
  self-consistent rather than physically matched to the direct term.
- **Cascaded shadows in the raster capture only work inside the CSM's range.** A
  probe beyond the cascade distance gathers unshadowed sunlight. The ray-traced
  gather shadows its hits with rays and does not have this limit.
- **Rotated rays still feed the round-robin update.** A probe is updated once a
  cycle with the cycle-length hysteresis, not every frame at DDGI's 0.97, and the
  depth lobe is still the blunted 20 -- whether the rotation has earned a sharper
  one is the next measurement, not a conclusion.
- **Below about 512 rays the rotated layout loses to the fixed one** on these
  scenes (256 is worse on every column), so the setting is not a free quality
  knob in both directions.
- **The skinned caster shades flat in the ray-traced gather**: face normal, no
  texture. Its geometry record holds bare positions.
- **The update is not budgeted in time**, only in probe count. The debug panel
  reports the capture's CPU cost, so the feedback exists, but nothing adapts the
  batch size to it — a heavier scene makes each capture more expensive and the
  count stays where it was set.
- **Only this scene has been measured.** Eleven draw items and a handful of
  lights is a light load for a pass that loops every light for every capture
  fragment; the conclusion that the light loop is not what costs is specific to
  that.
- **The default scene does not show it well.** An open platform with grey
  materials has little to bleed and little to occlude, so the effect there is
  real but subtle. Use the Cornell box preset to see it, and treat any
  measurement taken on the default scene as a lower bound.
- **The Cornell box is a diagnostic, not a validation suite.** It has no
  reference image and nothing compares against one, so it shows that light is
  transported and coloured, not that the amount is right.
