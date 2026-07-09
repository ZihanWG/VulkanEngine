# Ground-Truth Ambient Occlusion (GTAO)

Horizon-based ambient occlusion (Jimenez et al. 2016, *Practical Realtime
Strategies for Accurate Indirect Occlusion*) computed from the main depth buffer
and the thin G-buffer normal, then multiplied into scene color by the composite.
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
3. **CompositePass** — samples the AO term at binding 4 and multiplies it into
   scene color when GTAO is enabled (the former depth-for-SSAO binding).

Pass order: main HDR (± two-phase occlusion) → SSR → GTAO trace → GTAO upsample →
depth pyramid → TAA → bloom/exposure/composite.

## Half resolution

The horizon search is the expensive part, so it runs at half resolution (a
quarter of the pixels, ~4× fewer searches). The joint-bilateral upsample restores
full resolution using the full-res depth buffer to reject taps on other surfaces,
so silhouettes stay sharp. Measured on macOS/MoltenVK (Debug): the trace dropped
from ~9.7 ms full-res to ~2.7 ms, plus ~0.6 ms for the upsample.

## Controls

`Ambient Occlusion (GTAO)` panel / `SsaoSettings`: `enabled`, `radius` (view
units), `intensity`, `power`, `sliceCount` (1–8), `stepsPerSlice` (2–16), and
`falloff`. The `GTAO Visibility` render-target preview shows the denoised term.
Requires a samplable main depth image; otherwise the panel reports unavailable
and the passes are skipped.

## Limitations

- Applied as a scene-color multiply in the composite, so it darkens direct
  lighting too. Physically AO should modulate only indirect/ambient (IBL); doing
  that correctly needs the main pass to sample AO, which requires either a depth
  prepass or previous-frame reprojection. Documented rather than hidden.
- No multi-bounce: the GTAO multi-bounce fit needs per-pixel albedo, which the
  thin G-buffer does not store. Future work alongside a fuller G-buffer.
- Spatial denoise only — no temporal accumulation yet (the velocity buffer TAA
  already uses would let the slice/step counts drop further).
- Screen-space: occlusion from off-screen or depth-occluded geometry is missing,
  and the half-res trace can miss sub-pixel contact detail that the upsample
  cannot fully recover.
