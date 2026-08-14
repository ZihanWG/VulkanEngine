# Ground-Truth Ambient Occlusion (GTAO)

Horizon-based ambient occlusion (Jimenez et al. 2016, *Practical Realtime
Strategies for Accurate Indirect Occlusion*) computed from the main depth buffer
and the thin G-buffer normal, then applied to the ambient term by the main pass.
It replaces the earlier depth-only SSAO that ran inline in the composite shader.
Implementation: [`src/renderer/GroundTruthAmbientOcclusion.*`](../src/renderer/GroundTruthAmbientOcclusion.h),
[`src/shaders/gtao.frag`](../src/shaders/gtao.frag), and
[`src/shaders/gtao_blur.frag`](../src/shaders/gtao_blur.frag). Off by default.

## Why GTAO over the old SSAO

The previous SSAO reconstructed a face normal from depth derivatives and summed a
depth-only occlusion factor on a Vogel disk — cheap but crude, with no real
horizon and heavy noise. GTAO reuses the thin G-buffer normal that SSR already
writes (octahedral world normal → view space) and integrates the *actual*
cosine-weighted visibility arc, so contact shadows and creases read correctly.

## Pass structure

The subsystem mirrors SSR: it owns its pipelines, per-frame params buffer, and
descriptors, borrows the rendering services + `SsaoSettings` by reference, and is
gated on a samplable main depth image (`available()`), same as SSR.

1. **GTAOPass** (graphics, **half resolution**) — a fullscreen horizon search
   into a half-res `R8_UNORM` raw-AO target (owned by the subsystem, wrapped by
   the render graph like SSR's scene-color copy). For each pixel it:
   - reconstructs the view-space position from depth and decodes the view-space
     normal from the thin G-buffer;
   - sweeps `sliceCount` slices around the view direction (rotation jittered by
     interleaved gradient noise so the denoise integrates it out);
   - for each slice, marches `stepsPerSlice` steps to each side and tracks the
     maximum horizon cosine, attenuated past the radius by a `falloff` term;
   - clamps the two horizons into the hemisphere around the projected normal and
     evaluates the closed-form GTAO arc integral, weighted by the projected
     normal length;
   - writes `pow(1 - occlusion·intensity, power)` as the visibility term.
2. **GTAOBlurPass** (graphics, **full resolution**) — a joint-bilateral upsample
   + denoise: for each full-res pixel it gathers the half-res raw-AO
   neighborhood, weighting each tap by a spatial Gaussian and by view-space depth
   similarity measured against the full-res depth buffer. This removes the slice
   noise and upsamples in one pass without bleeding across depth edges. Output is
   the full-res `AmbientOcclusion` target (owned by PostProcessStack).
3. **MainHDRPass** (the *next* frame) — samples the AO target at set 0 binding 12
   and multiplies it into the ambient *diffuse* term only. The specular IBL is
   left unoccluded: AO answers the diffuse hemisphere-visibility question, and
   reusing it for specular was always an approximation. It also has to stay off
   the specular half for SSR to stay conservative — the trace subtracts the
   specular IBL it replaces and cannot see a factor applied after the fact, so an
   AO-attenuated `specularIbl` would be over-subtracted into negative scene
   colour in exactly the creases AO darkens. Because GTAO runs after the main
   pass, what the main pass reads is the previous frame's result, so the sample is
   reprojected along the motion vector the shader already computes for TAA. A
   reprojection landing off screen has no history and falls back to unoccluded.
   The transparent pass declares the same read, since it reuses this fragment
   shader.

   `CompositePass` still samples the term at binding 4 and multiplies it into the
   whole scene colour, but only when `ambientOnly` is off — see below.

Pass order: main HDR (± two-phase occlusion) → SSR → GTAO trace → GTAO upsample →
depth pyramid → TAA → bloom/exposure/composite.

## Half resolution

The horizon search is the expensive part, so it runs at half resolution (a
quarter of the pixels, ~4× fewer searches). The joint-bilateral upsample restores
full resolution using the full-res depth buffer to reject taps on other surfaces,
so silhouettes stay sharp. Measured on macOS/MoltenVK (Debug): the trace dropped
from ~9.7 ms full-res to ~2.7 ms, plus ~0.6 ms for the upsample.

"Full resolution" here means the internal render extent, not the window: under a
render scale below 1.0 both targets shrink with everything else, and the trace
stays half of whatever that is. See [render_scale.md](render_scale.md).

## Controls

`Ambient Occlusion (GTAO)` panel / `SsaoSettings`: `enabled`, `radius` (view
units), `intensity`, `power`, `sliceCount` (1–8), `stepsPerSlice` (2–16), and
`falloff`. The `GTAO Visibility` render-target preview shows the denoised term.
Requires a samplable main depth image; otherwise the panel reports unavailable
and the passes are skipped.

## Limitations

- **Occlusion lags one frame.** The main pass reads the previous frame's AO,
  because GTAO needs the depth and normals that pass produces. Static scenes are
  unaffected; under fast camera motion contact shadows trail slightly, and newly
  disoccluded pixels have no history and read as unoccluded until the next frame.
  A depth prepass would avoid this at the cost of submitting all geometry twice.
- `SsaoSettings::ambientOnly` (default on) selects this. Turning it off restores
  the older whole-scene multiply in the composite, which also darkens direct
  lighting -- physically wrong, but kept as an A/B reference.
- No multi-bounce: the GTAO multi-bounce fit needs per-pixel albedo, which the
  thin G-buffer does not store. Future work alongside a fuller G-buffer.
- Spatial denoise only — no temporal accumulation yet (the velocity buffer TAA
  already uses would let the slice/step counts drop further).
- Screen-space: occlusion from off-screen or depth-occluded geometry is missing,
  and the half-res trace can miss sub-pixel contact detail that the upsample
  cannot fully recover.
