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

**This target reports neither `VK_KHR_ray_query` nor
`VK_KHR_acceleration_structure`.** Of 131 device extensions, none are ray
tracing; MoltenVK does not expose them. DDGI as normally implemented is not
merely slow here, it is unavailable.

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

## Pass structure

```
CSMShadowPass            directional cascades
PunctualShadowAtlasPass  spot/point tiles
ClusterBuild / LightCull
ProbeCapture             graphics: 6 faces x N probes into the capture atlas
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

Demo scene, **Debug build** (Release timings not yet taken; Debug numbers here
have run ~10x off Release elsewhere in this project and are not a basis for a
performance claim).

| probes/frame | ProbeCapture | IrradianceProbeUpdate |
| --- | --- | --- |
| 4 (default) | 0.003 ms (at the timer floor) | 0.214 ms |
| 16 (max) | 0.327 ms | 0.471 ms |

**The convolution dominates, not the light loop.** Even with every punctual light
evaluated for every capture fragment, the capture is cheaper than the
convolution's 1536-sample inner loop. Per-probe light culling is therefore not
implemented — there is no measurement that justifies it.

Effect on the image, from the engine's own scene-luminance readout:

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

Its own window, **Irradiance Probes**, and not behind the advanced-mode gate.
The main debug panel's sections are all default-open, so it is already taller
than the display; anything appended lands below the fold where ImGui clips it
rather than drawing it. That is not merely inconvenient — the atlas previews are
how this subsystem is checked at all.

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
- **Single bounce.** The capture evaluates direct lighting; it does not sample
  the probe atlas itself, so light does not bounce twice. Feeding the previous
  frame's irradiance back into the capture is the standard next step and is not
  implemented.
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
- **Cascaded shadows in the capture only work inside the CSM's range.** A probe
  beyond the cascade distance gathers unshadowed sunlight.
- **The update is not budgeted in time**, only in probe count. A scene with far
  more draw items would make each capture more expensive with no feedback.
- **Release timings have not been taken.** Every number above is from a Debug
  build.
- **The demo scene does not show it well.** An open platform with grey materials
  has little to bleed and little to occlude; the effect is real but subtle. A
  closed, coloured room would demonstrate it far better and does not exist yet.
