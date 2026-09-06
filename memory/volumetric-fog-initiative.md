---
name: volumetric-fog-initiative
description: Froxel volumetric fog SHIPPED — all phases merged to main and pushed (ccf0f63, 7f270e5, 572b5c1); branch feature/fog-light-shafts deleted 2026-08-05 after confirming it was fully merged
metadata: 
  node_type: memory
  type: project
  originSessionId: df3509ca-ddf6-43e5-9b25-e03a9a8bfe0d
  modified: 2026-08-02T14:45:59.584Z
---

Froxel volumetric fog, started 2026-08-02 right after
[[punctual-shadows-initiative]] shipped. Phase 1 merged to main as `ccf0f63` (`--no-ff`) and pushed to origin. Picked from the 2026-08-02 engine review as the highest
value-per-effort item, because the froxel grid already existed.

Phase 1 (`9be8b41`): 160x90x64 volume, two compute passes (inject, integrate),
applied in the main HDR pass. Directional light only.

Design decisions worth not re-deriving:
- **Fog applies in the main HDR fragment shader, not composite.** Composite has
  no depth bound and locating a froxel needs view depth; the main pass already
  carries `vViewDepth` for clustered lighting. Also lands fog before bloom/TAA.
- **The fog volume is NOT the cluster light grid.** 16x9x24 is too coarse and
  fog wants a much nearer far plane (64 vs the scene far plane). Only the
  addressing scheme is shared — that is what lets Phase 2 reuse cluster light
  lists.
- **Integration uses the analytic slab integral**, not scattering * thickness.
  Exponential slices make the last slice 10x+ the first, thick enough that the
  naive form blows dense fog out to white instead of saturating at the albedo.
- Fog is off by default (opt-in like GTAO/TAA).

Phase 2 (`35d227b`) + Phase 3 (`db8f40d`), merged as `7f270e5` (--no-ff) and
pushed; user confirmed the visuals:
- Phase 2: punctual light shafts. Froxels walk the cluster light list and sample
  the punctual shadow atlas. Required moving the fog pass *after* cluster
  build/light cull (it had been recorded with the shadow passes, reading last
  frame's lists). Cost 0.32ms -> ~1.0ms.
- Phase 3: temporal jitter (Halton 2/3/5, 16-frame) + reprojection against a
  ping-ponged history volume, plus skybox fog. The jitter is load-bearing —
  without it the blend averages identical centre samples and converges to the
  unfiltered result. Skybox samples unconditionally because its push constants
  are already at the 128-byte minimum; fog-off re-clears the volume to neutral
  instead.

**Per-light importance culling (merged `572b5c1`, pushed).** A froxel bounds each
light's possible contribution (brightest channel * intensity / d^2 * phase peak)
and skips it below a threshold. Conservative by construction — range fade, spot
cone and shadow are each <=1 and the phase <= maxHenyeyGreensteinPhase — so a
visible light can never be dropped.

**The lesson: placement was the whole optimisation, and my first guess was
wrong.** I put the test before the shadow fetch + phase assuming those dominate.
Measuring said no — culling *every* light there saved <10%. The real cost is the
list walk and the attenuation math ahead of it (notably the `pow()` in range
fade). Moving the test above all of that is what made it work.

Measured (min+median over ~19 samples; single averages here are so noisy they
reported a threshold as *slower* than no threshold): off 1.13ms, default 0.05
~1.15ms (within noise), 0.2 -> 0.80ms, cull-everything floor 0.70ms. Default is
deliberately conservative and its win does NOT clear the noise on this machine —
recorded honestly rather than dressed up. Win grows with light density; the demo
has few lights per cluster, the case this helps least.

Threshold is an *absolute radiance* bound so it does not transfer between scenes;
exposed as a slider rather than baked. Finding where it becomes visible needs
eyes on the image.

Still open: reprojection is camera-only (moving lights smear), uniform medium
(no noise), fixed 160x90x64 volume, threshold is absolute not relative.

Correction recorded: "transparent geometry is not fogged" was never true — the
transparent pass reuses simple_bindless.frag.

Verification: 124/124 tests, validation clean with fog off AND forced on.
~0.32ms without shafts, ~1.0ms with; temporal filtering adds nothing measurable.
User confirmed the visuals before merging.

Note: I hit the *same* cold-start layout hazard here that
[[punctual-shadows-initiative]] had already taught me and that I had written
into its docs — an optional subsystem's image sitting in UNDEFINED while a
material descriptor claims a sampled layout. Worth checking for by reflex
whenever a new optional resource joins the material descriptor set.
