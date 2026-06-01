# Temporal AA Foundation

Phase 6B adds a conservative Temporal AA foundation. It is intentionally limited to jitter plus HDR history resolve, so it can coexist with the existing renderer without introducing motion vectors, temporal upscaling, FSR/DLSS, or a broader camera/object velocity pipeline.

## Runtime Controls

TAA is off by default. It can be enabled from:

- `VulkanEngine Debug` -> `Temporal AA`
- `config/runtime_settings.example.json` and saved runtime settings under the `taa` object

Runtime fields:

- `enabled`: runs the TAA resolve pass and routes post-processing through resolved history.
- `jitterEnabled`: applies subpixel projection jitter while TAA is active.
- `neighborhoodClampEnabled`: clamps previous history to the current 3x3 color neighborhood.
- `feedback`: history blend factor, clamped to `0.0..0.98`.

The ImGui panel also exposes current jitter, history validity, read/write history indices, and a manual `Reset TAA History` button.

## Frame Flow

1. `MainHDRPass` renders skybox and meshes into `SceneColorHDR`.
2. `DepthPyramidPass` still builds Hi-Z from main depth after the HDR pass.
3. If TAA is enabled, `TAAResolvePass` samples current `SceneColorHDR` plus the previous history image and writes the current history image.
4. Bloom, luminance, histogram exposure, and composite sample the active HDR source: resolved TAA history when TAA is enabled, otherwise `SceneColorHDR`.
5. After post-processing uses the current history image, the renderer flips the ping-pong history index for the next frame.

## Jitter And Matrices

The renderer uses an 8-sample Halton sequence in bases 2 and 3. Jitter is applied to the projection matrix as a subpixel NDC offset for skybox and mesh rendering.

CPU frustum culling, CSM setup, depth-pyramid validity checks, and GPU culling inputs stay on the unjittered view-projection matrix. This keeps Phase 6B scoped to image resolve behavior and avoids changing culling stability.

## History Resources

Two full-resolution `VK_FORMAT_R16G16B16A16_SFLOAT` images are recreated with the swapchain. Render Graph imports them as persistent resources:

- `TAAHistoryRead`
- `TAAHistoryWrite`

`TAAResolvePass` reads the previous history and writes the current history. Later post-process passes read the current history through descriptor variants.

History is invalidated on swapchain/post-process recreation, runtime TAA setting changes, camera resets/edits, scene load, portfolio mode application/restoration, material asset save/reload, and explicit debug reset.

## Resolve Shader

`src/shaders/taa_resolve.frag` performs:

- current HDR sample
- optional previous history sample
- current-frame 3x3 min/max neighborhood
- optional neighborhood clamp of history
- bounded feedback blend

The shader uses same-UV history sampling. This is deliberate for Phase 6B: there is no velocity buffer, reprojection matrix path, per-object motion, disocclusion mask, reactive mask, or temporal upscaler.

## Validation Surface

Expected debug signals:

- Render Graph shows `TAAResolvePass` when TAA is enabled.
- GPU Profiler shows `TAAResolvePass` when TAA is enabled.
- Render target debug views show both TAA history images once they reach shader-read layout.
- Toggling TAA or pressing reset makes history invalid for the next resolve.

## Limitations

- No motion vectors.
- No depth reprojection.
- No automatic camera-cut detection beyond explicit invalidation paths.
- No temporal upscaling.
- No FSR/DLSS/XeSS.
- No per-material reactive mask.
- Animated object history is not motion-compensated.
