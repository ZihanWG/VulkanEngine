# Volumetric Fog

Froxel volumetric fog: a 3D volume over the view frustum holding the
participating medium, lit and integrated in compute, then applied in the main
HDR pass as a single texture fetch.

Off by default — fog changes the look of every shot, so it is opt-in the way
GTAO and TAA are.

Both the directional light and the punctual (spot/point) lights contribute, so
shadowed lights cast visible shafts through the medium.

## Why a froxel volume

Ray-marching the depth buffer per pixel makes cost scale with screen resolution
and gives every pixel an independent, noisy estimate. A froxel volume decouples
fog cost from resolution entirely: the medium is evaluated once per froxel at
160x90x64, and the pixel shader does one trilinear fetch.

It also makes the fog *volumetric* rather than a screen-space effect — the
medium exists in front of, between, and behind geometry, so light shafts and
occlusion work out of the integration rather than needing a separate pass.

## Passes

```
CSMShadowPass            directional cascades
PunctualShadowAtlasPass  spot/point tiles
VolumetricFogPass        inject -> integrate          <-- here
MainHDRPass              samples the integrated volume
```

**Injection** (`fog_inject.comp`, one invocation per froxel) reconstructs the
froxel's world position, evaluates the medium's density there, lights it with
the directional light — shadowed by the same cascaded shadow map the opaque pass
uses — and writes `rgb` = light scattered toward the eye per unit length, `a` =
extinction per unit length.

**Integration** (`fog_integrate.comp`, one invocation per *column*) marches
front to back through Z, accumulating in-scattering and transmittance, and
writes the running total into every slice. A column per thread rather than a
froxel per thread is the whole point: the march has to be sequential, because
light from a far slice must travel back through everything nearer the eye.

The output is arranged so the apply step is one fetch and one lerp:
`scene * a + rgb`.

## Where the fog is applied

In the **main HDR fragment shader**, not in composite.

Composite has no depth bound, and finding a fragment's froxel needs its view
depth — applying there would have meant making the depth attachment samplable
just for fog. The main pass already carries `vViewDepth` as a varying, because
clustered lighting needs it to pick a froxel. Reusing it costs nothing.

It also puts fog into scene colour *before* bloom and TAA, which is where fog
belongs: bright fog should bloom, and fog should be temporally filtered along
with everything else.

The push constant carries the fog max distance, and zero means off — the same
"one value carries the disabled state" shape the punctual shadow slot sentinel
uses, so no separate flag is needed.

## Relationship to the clustered light grid

The volume is addressed like the clustered lighting grid — screen tiles in XY,
exponential slices in Z — but is deliberately *not* the same grid:

- the light grid is 16x9x24, far too coarse to represent fog;
- fog wants its own, much nearer far plane (64 units by default, versus the
  scene's far plane) so its 64 slices land where fog is actually visible.

What is shared is the addressing scheme, and that is what lets a fog froxel find
the light cluster covering it and walk **that cluster's light list** rather than
every light in the scene — the same amortisation the opaque pass gets.

That mapping is the one place the two systems have to agree, and getting it
wrong hands a froxel some other cluster's lights, which looks like fog lit
slightly wrong rather than anything obviously broken. So it is a GPU-free
function (`fogFroxelClusterIndex`) defined as "the cluster the fragment shader
would pick for a fragment at this froxel's screen position and view depth", and
a unit test asserts exactly that against `ClusterGrid.h`'s own `clusterIndex`
instead of re-deriving the arithmetic.

The fog volume also uses its own near plane (0.5 rather than the camera's 0.1),
because an exponential distribution anchored at 0.1 spends most of its slices in
the first metre where almost nothing is visible.

## Light shafts

Punctual lighting inside the medium reuses what the surrounding systems already
built: the per-cluster light lists from the clustered lighting passes, and the
tiles from the punctual shadow atlas. Fog builds neither of its own.

That also fixes the pass ordering. Fog injection has to run *after* the cluster
build and light cull, because it walks their output, and *before* the main HDR
pass, which samples the volume. It sits between them.

The shadow lookup is deliberately simpler than the surface one in
`simple_bindless.frag`: no normal-offset bias, no slope scaling, and a single
tap instead of 3x3 PCF. That is not a shortcut. Those biases exist to stop a
surface shadowing itself, and a participating medium has no surface to
self-shadow; and froxels are far coarser than pixels with a trilinear filter on
the way out, so extra taps would be filtered away regardless.

Attenuation, the range cutoff, and the spot cone are evaluated exactly as the
surface path does them, so a light does not appear to reach further through fog
than it does across a floor.

## Temporal filtering

One sample per froxel per frame aliases badly under a high-frequency shadow: a
froxel is either fully lit or fully shadowed, so shaft edges crawl and flicker
as anything moves.

The fix is the standard pair. The sample position is jittered inside its froxel
each frame using Halton 2/3/5 over a 16-frame cycle -- the same sequence the TAA
jitter uses, extended to the volume's third axis so slices are sampled at
varying depths too. The result is then blended against the previous frame's
volume, reprojected through the previous view-projection.

The jitter is not optional decoration: without it every frame samples the froxel
centre, the blend averages identical samples, and the filter converges to
exactly the unfiltered result. The jitter is what gives the history something
new to integrate.

Two guards matter:

- **Reprojection rejects anything off screen or outside the volume last frame.**
  There is no history for it, and clamping would smear one edge froxel across
  the whole disoccluded region.
- **The history is rejected if it is negative or NaN.** This is a feedback loop
  writing into itself, so one bad value would be blended forward forever rather
  than washing out. For the same reason both scatter volumes are cleared at
  startup rather than left `UNDEFINED`.

The volumes ping-pong, so a descriptor set bakes in which one is written and
which is sampled as history -- hence two sets per frame, one per parity.

Cost is negligible next to the light loop: the fog passes stay at roughly 1.0ms
with filtering on.

## Fog on the sky

The skybox is infinitely far, so it samples the volume's far slice -- the total
fog along the whole view ray. Without it the world fogs out while the sky behind
stays perfectly crisp, which reads as a hole in the fog rather than as depth.

It samples **unconditionally**, with no enable flag, because the skybox push
constants are already two mat4s -- exactly the 128-byte guaranteed minimum, with
nowhere to put one. Instead, switching fog off re-clears the volume to its
neutral state (zero scattering, unit transmittance), which makes the sample a
no-op. That is detected from the fog-active state each frame rather than in the
UI callback, so every path that disables fog is covered.

## Per-light importance culling

A froxel bounds what each light could still contribute before doing the work,
and skips it when even that bound is negligible. The bound is
`brightest channel * intensity / distance^2 * maxPhase(anisotropy)`.

**Conservative, not approximate.** The range fade, the spot cone, and the shadow
term are each at most 1, and the phase is at most `maxHenyeyGreensteinPhase`, so
the bound can only ever overstate the real contribution — a light that would
have been visible is never dropped. Using the brightest channel rather than
luminance keeps that true for strongly tinted lights. A unit test pins the phase
bound against the phase function over a full sweep of angles and anisotropies,
including that it is *tight* and not merely safe: a loose bound would make the
cull silently never fire.

**Where the test goes is the whole optimisation, and the first placement was
wrong.** It originally sat just before the shadow-atlas fetch and the phase
function, on the assumption that those dominate. Measuring said otherwise:
culling *every* light there saved under 10%. What the loop actually costs is
walking the list and the attenuation arithmetic — particularly the `pow()` in
the range fade. Moving the test above all of that is what made it worth having.

Measured on the demo scene, taking the minimum and median over ~19 samples,
because single averages on this machine are noise-dominated:

| Threshold | Median | Note |
| --- | --- | --- |
| 0 (off) | 1.13 ms | Reference |
| 0.05 (default) | ~1.15 ms | Within noise here; culls only the outer shell |
| 0.2 | 0.80 ms | Clear win, starts trading fidelity |
| Cull everything | 0.70 ms | The floor — what the light loop costs in total |

The default is deliberately conservative. At 0.05 with the demo's lights it only
culls beyond ~7.1 of 8 units, where `rangeFade²` has already reduced the true
contribution to under 0.007 — genuinely invisible, but also a small enough share
of the work that the win does not clear the measurement noise on this machine.
The win is real and clear at higher thresholds, and grows with light density:
the demo has only a handful of lights per cluster, which is the case this helps
least.

## The slab integral

The integration uses the analytic integral of scattering across a homogeneous
slab, `(1 - exp(-sigma * d)) / sigma`, not `scattering * thickness`.

The naive form ignores that light scattered in near the back of a slice is
partly absorbed before it leaves, and overestimates badly once a slice is
optically thick — dense fog blows out to white instead of saturating at the
medium's own colour. A unit test pins exactly that: at extreme density the
result converges to the scattering albedo rather than growing without bound.

This matters more here than in a per-pixel ray march because froxel slices are
*thick*: exponential spacing makes the last slice more than ten times the depth
of the first.

## Controls

Debug panel → **Volumetric Fog**:

| Control | Effect |
| --- | --- |
| Enable | Master toggle; off costs nothing, not even the dispatches |
| Density | Extinction per unit length |
| Max distance | Depth range of the volume; shorter concentrates the 64 slices where fog is visible |
| Anisotropy | Forward scattering; positive gives the halo when looking toward a light |
| Height falloff | Zero is uniform; larger pulls fog into a ground layer |
| Base height | World height the layer sits at |
| Ambient scale | Keeps unlit fog from going black |
| Scattering color | Medium albedo |

## Testing

`tests/test_volumetric_fog.cpp` covers the GPU-free core, the same way
`ClusterGrid.h` and `PunctualShadowAtlas.h` are covered:

- the slice/depth mapping round-trips — the injection pass turns a slice into a
  world position and the apply pass turns a depth back into a slice, and a
  mismatch would make fog slide relative to the geometry it sits in front of;
- slices grow with depth and exactly tile the volume, with no gap or overlap;
- an empty medium leaves the background untouched (transmittance 1, scatter 0);
- transmittance decreases monotonically and stays in [0, 1] — above 1 would
  brighten the background instead of fogging it;
- dense fog converges to the albedo instead of blowing out;
- a near opaque slice suppresses a bright slice behind it, which is the
  weighting that stops fog glowing through denser fog;
- the phase function integrates to 1/(4pi) when isotropic, points forward for
  positive anisotropy, and stays finite at the degenerate endpoints;
- height fog is flat at and below its base and decays above it.

## Limitations

- **Reprojection is camera-only.** A moving light drags its shaft through the
  history rather than being rejected, so a fast-moving light smears at high
  history-blend values. Proper rejection needs a per-froxel validity signal, not
  just the previous camera.
- **Uniform medium.** No noise or density texture, so the fog has no internal
  structure.
- **Punctual shafts cost.** Every froxel walks its cluster's light list, which
  takes the fog passes from roughly 0.3ms to roughly 1.0ms on the demo scene.
  Importance culling trims that but cannot remove the list walk itself.
  Temporal filtering adds nothing measurable on top.
- **The cull threshold is absolute, not relative.** It is a radiance bound, so a
  value tuned for one scene's light intensities does not transfer to another
  with very different ones. A threshold relative to the froxel's own accumulated
  scattering would, but needs two passes over the light list.
- **The volume is a fixed 160x90x64** regardless of resolution or fog distance.
