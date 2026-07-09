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
   - marches up to `maxSteps` fixed view-space steps (start offset jittered by
     interleaved gradient noise, which TAA integrates), comparing ray depth
     against the depth buffer with a view-space `thickness` test;
   - refines the hit with `refinementSteps` bisection iterations;
   - samples the scene-color copy at the hit and outputs
     `color * fresnel * confidence * intensity` with
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

Requires a samplable main depth image (same gate as SSAO); otherwise the panel
reports unavailable and the passes are skipped.

## Limitations

- Screen-space only: anything off-screen or occluded contributes nothing
  (edge fade hides most of it; no fallback blend to IBL by hit confidence yet).
- Reflections add on top of the existing IBL specular rather than replacing it,
  so mirror-like surfaces can read slightly bright (standard first-pass SSR
  trade-off; documented rather than hidden).
- The thin G-buffer stores no albedo, so metal reflections use a grayscale
  F0 approximation (untinted).
- Linear march with fixed steps — no Hi-Z acceleration (the existing pyramid is
  max-depth, built for occlusion; a min-depth pyramid is future work), no
  roughness-cone blur, no half-res trace.
