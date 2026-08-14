# Screen-Space Reflections

Glossy screen-space reflections traced against the main depth buffer, blended
into scene color before TAA so the temporal filter integrates the march jitter.
Implementation: [`src/renderer/ScreenSpaceReflections.*`](../src/renderer/ScreenSpaceReflections.h)
and [`src/shaders/ssr_trace.frag`](../src/shaders/ssr_trace.frag). On by default.

## Thin G-buffer

The forward renderer has no G-buffer, so the main HDR pass writes a third MRT
attachment (`NormalRoughnessGBuffer`, RGBA16F): octahedral-encoded world-space
shading normal (RG, including normal mapping), roughness (B), metallic (A).
The skybox writes roughness 1 so sky pixels never trace. The attachment is
owned by PostProcessStack next to the velocity target and is also read by the
GTAO pass (see [gtao.md](gtao.md)), which reuses this normal.

## Pass structure

1. **SSRCopyPass** (transfer) — copies `SceneColorHDR` into `SsrSceneColorCopy`.
   The trace samples the copy because reading the same image it blends into
   would be a feedback loop.
2. **SSRTracePass** (graphics) — fullscreen pass over the depth buffer:
   - reconstructs the view-space position from depth (inverse of the same
     jittered projection the rasterizer used), decodes the world-space normal,
     and reflects the view ray;
   - marches up to `maxSteps` steps of equal *screen-space* stride along the
     ray's projected segment (start offset jittered by interleaved gradient
     noise, which TAA integrates), carrying `1/depth` — which is what varies
     linearly across the screen — and inverting it per step to compare against
     the depth buffer with a view-space `thickness` test;
   - refines the hit with `refinementSteps` bisection iterations on the same
     screen-space parameter;
   - samples the scene-color copy at the hit and outputs the signed correction
     `(color - prefilteredEnv) * specularWeight * replacement` (see
     [Energy conservation](#energy-conservation)), where
     `replacement = clamp(confidence * intensity, 0, 1)` and
     `confidence = screenEdgeFade * roughnessFade * towardCameraFade`.
   The pipeline uses ONE + ONE additive blending into `SceneColorHDR` (LOAD),
   so the shader pre-multiplies every weight and a zero output is a no-op.

Pass order: main HDR (± two-phase occlusion) → SSR copy → SSR trace → final
depth pyramid → TAA → bloom/exposure/composite. Running before TAA gives the
reflections neighborhood clamping and temporal accumulation for free.

## Controls

`Screen-Space Reflections` panel / `ssr` settings block: `enabled`, `maxSteps`
(8–128), `refinementSteps` (0–8), `maxDistance` (view units), `thickness`,
`intensity`, `maxRoughness` (surfaces rougher than this trace nothing — the
roughness fade approaches zero there), `screenEdgeFade`. All clamped in
`clampRuntimeSettings` (unit-tested) and persisted with the other settings.

`intensity` scales how far the reflection *replaces* the IBL, and the combined
weight saturates at 1. Its 0–4 range predates the conservation fix, when the
output was purely additive and 4.0 just meant "brighter"; against a signed
correction an unclamped 4.0 would subtract four times the specular IBL the main
pass wrote, pushing scene colour negative into bloom and the exposure histogram.
Above 1.0 it now means "reach full replacement at lower confidence".

Requires a samplable main depth image (same gate as SSAO); otherwise the panel
reports unavailable and the passes are skipped.

## Limitations

- Screen-space only: anything off-screen or occluded contributes nothing. This is
  now a graceful fallback rather than a hole: the pass emits a *correction* toward
  the traced colour, so no hit means the IBL specular simply stands.
- The thin G-buffer stores no albedo, so the trace weights the swap with a
  grayscale F0 while the main pass used an albedo-tinted one. The two weights do
  not cancel, and a residual `prefilteredEnv * (w_tinted - w_grayscale)` survives
  the correction. It is one-directional: base colour is at most 1 per channel, so
  the grayscale F0 is always the larger and SSR always *over*-subtracts the
  environment specular, never under-subtracts — the benign direction, a slight
  darkening rather than energy gain. Bounded per channel by
  `(1 - baseColour) * brdf.x`, and exactly zero on dielectrics, whose F0 is 0.04
  on both sides. In the default portfolio scene only `PolishedMetalSmall` is both
  metallic and smooth enough to trace at all; its base colour (0.82, 0.85, 0.88)
  makes the residual a near-uniform ~15% darkening of that sphere's environment
  specular. Removing it needs tinted F0 in the G-buffer, which has no spare
  channel — see [Why the exact fix is not worth it](#why-the-exact-fix-is-not-worth-it).
- The multi-scatter compensation the main pass adds on top of `specularIbl` is
  not subtracted either, so that part stays double-counted. It is structurally
  self-limiting: the term scales with `roughness^2`, while SSR's `roughnessFade`
  drives confidence to zero as roughness approaches `maxRoughness`, so the two
  barely overlap. `RoughMetal` (roughness 0.60, strength 0.70), where the term is
  large, is skipped outright by the `roughness >= maxRoughness` early-out;
  `PolishedMetalSmall` (0.23, 0.40) contributes `0.23^2 * 0.40 ~= 2%`. The peak
  across the whole band is ~5% of the environment term.
- No Hi-Z acceleration (the existing pyramid is
  max-depth, built for occlusion; a min-depth pyramid is future work), no
  roughness-cone blur, no half-res trace.

## Energy conservation

The pass used to blend `reflection * fresnel * confidence` additively onto scene
colour that **already contained the main pass's specular IBL** for that pixel, so
a mirror received its specular roughly twice. That was described here as a
standard first-pass trade-off, which understated a conservation bug.

It now emits a signed difference instead, keeping the additive blend:

```
want:  mix(iblSpec, ssr, conf)  =  iblSpec + (ssr - iblSpec) * conf
emit:  (ssrColour - prefilteredEnv) * specularWeight * conf
```

Both terms carry a split-sum weight `F * brdf.x + brdf.y`. Where the two `F`s
agree it factors out of the difference, and that is what carries the argument on
a thin G-buffer. Confidence 0 leaves the IBL untouched; confidence 1 replaces it
entirely.

Where they disagree it does not factor out — and here they do disagree, because
this pass reconstructs a grayscale F0 while the main pass used an albedo-tinted
one. What survives is `prefilteredEnv * (w_tinted - w_grayscale)`. That is a
residual, not a return of the double-count: it is bounded, always an
over-subtraction rather than an addition, and zero on dielectrics. An earlier
revision of this document (and of the shader header) claimed an imprecise F0
"cannot reintroduce the double-count" and that the weight factors out
unconditionally. That overstated the factorisation, which holds per channel only
when both sides use the same `F`. See Limitations for the measured bound.

Verified by what SSR does to average scene luminance, which is the bug's
signature. Against the SSR-off baseline it added **+0.93%** before, and **-0.2%**
after — a replacement redistributes specular rather than brightening the frame.

The trace therefore needs the prefiltered environment cube and the BRDF LUT. Both
are built *after* `ScreenSpaceReflections::createResources` runs, so they are
bound by a separate targeted write that is re-applied after every recreate —
SSR resources are created twice during startup alone, and binding them once from
the environment path silently lost them. `frameSsrActive_` requires
`isIblBound()`, so the pass cannot run without knowing what it is replacing.

### Why the exact fix is not worth it

Both residuals above exist because the trace *reconstructs* what the main pass
wrote instead of reading it. The exact fix is to stop reconstructing: have the
main pass emit its `specularIbl` — tinted F0, multi-scatter and all — to a fourth
render target, and have SSR subtract that texture verbatim. Both residuals go to
zero at once, and the duplicated split-sum math disappears rather than being
mirrored more faithfully.

It is not worth what it costs here. The thin G-buffer is `R16G16B16A16_SFLOAT`
with every channel already spoken for (octahedral normal `xy`, roughness `z`,
metallic `w`), so this cannot be packed into what exists; it needs a new
attachment. That is +4 bytes/pixel on `MainHDRPass`, which is 56–78% of the
frame and the pass every other optimisation here has been fighting, plus a
tile-memory cost on this TBDR that would have to be measured rather than assumed.
The blast radius is four pipeline declarations (main, phase-2, transparent,
skybox), `PostProcessStack`, the render graph, both `simple_bindless.frag` and
the `simple.frag` fallback, and SSR's descriptor set.

Against that: a one-directional ~15% darkening of one sphere's environment
specular in the default scene, and a multi-scatter residual that is structurally
capped near 5% because it grows with `roughness^2` exactly where SSR is fading
out. Revisit if a strongly tinted metal (gold, copper) becomes a scene the engine
is judged on — there the F0 residual reaches 30–65% per channel and the trade
inverts.

## Marching in screen space

The march steps uniformly along the ray's **screen-space** projection, not by a
fixed view-space distance. A fixed view-space step covers many pixels near the
camera and a fraction of one far away, so it skipped geometry close up while
spending most of its steps re-reading the same distant texel. Depth is carried as
its reciprocal, which is what varies linearly across the screen, and inverted per
step — exact, and independent of the projection's `w` row.

`SSRTrace` fell from 0.627 ms to 0.502 ms: fewer wasted samples, and rays leave
the screen sooner so the loop exits earlier.

The ray origin is biased along the surface normal, scaled by view depth so the
bias stays constant in pixels. The old march did not need this — its first sample
sat half a fixed world-space step clear of the surface — but a screen-space step
shrinks in world terms as the camera closes in, so without the bias the first
sample lands on the originating surface and it reflects itself. That appeared as a
brightening rather than as visible garbage, and was caught by the same luminance
check.
