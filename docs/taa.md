# Temporal AA

Phase 6B added a conservative TAA foundation (jitter + same-UV HDR history resolve).
The motion-vector upgrade extends it into a real reprojecting TAA: the main pass
writes a velocity buffer, and the resolve reprojects history along per-pixel motion
with closest-depth velocity dilation. Temporal upscaling and FSR/DLSS remain out of
scope.

## Runtime Controls

TAA is off by default. It can be enabled from:

- `VulkanEngine Debug` -> `Temporal AA`
- `config/runtime_settings.example.json` and saved runtime settings under the `taa` object

Runtime fields:

- `enabled`: runs the TAA resolve pass and routes post-processing through resolved history.
- `jitterEnabled`: applies subpixel projection jitter while TAA is active.
- `neighborhoodClampEnabled`: clamps previous history to the current 3x3 color neighborhood.
- `reprojectionEnabled`: reprojects the history sample along the velocity buffer
  (camera + rigid object motion). Off falls back to the Phase 6B same-UV sampling
  for A/B comparison of ghosting under motion.
- `feedback`: history blend factor, clamped to `0.0..0.98`.

The ImGui panel also exposes current jitter, history validity, read/write history indices, and a manual `Reset TAA History` button.

## Velocity Buffer

The main HDR pass renders into two color attachments (MRT): `SceneColorHDR` plus a
render-extent `R16G16_SFLOAT` velocity buffer holding UV-space motion vectors
(`currentUV - previousUV`).

- **Meshes** (`simple.vert` / `simple_skinned.vert`): `ObjectFrameData` carries the
  unjittered current and previous-frame MVP matrices (`currMvpNoJitter` /
  `prevMvpNoJitter`). The vertex shaders emit both clip positions, and the fragment
  shaders derive the NDC delta. Rasterization keeps using the jittered `mvp`, so
  jitter never contaminates velocity.
- **Previous-frame state**: `Renderer::capturePreviousFrameMatrices()` runs after
  frame submission and stores the unjittered view-projection plus each
  `RenderObject::previousModelMatrix`. After a history reset the first frame
  reprojects with zero motion.
- **Skinned meshes**: velocity uses the current frame's skinned position with the
  previous rigid MVP, so it captures camera + rigid object motion but not
  joint-space motion (that would need the previous frame's joint palette).
- **Skybox**: the fragment shader projects the view direction with `w = 0` through
  the previous view-projection (translation drops out), producing rotation-only sky
  motion — correct for an infinitely distant environment.

## Frame Flow

1. `MainHDRPass` renders skybox and meshes into `SceneColorHDR` + `VelocityBuffer`.
2. `DepthPyramidPass` still builds Hi-Z from main depth after the HDR pass.
3. If TAA is enabled, `TAAResolvePass` samples current `SceneColorHDR`, velocity,
   main depth, and the previous history image, then writes the current history image.
4. Bloom, luminance, histogram exposure, and composite sample the active HDR source: resolved TAA history when TAA is enabled, otherwise `SceneColorHDR`.
5. After post-processing uses the current history image, the renderer flips the ping-pong history index for the next frame.

## Jitter And Matrices

The renderer uses an 8-sample Halton sequence in bases 2 and 3. Jitter is applied to the projection matrix as a subpixel NDC offset for skybox and mesh rendering.

CPU frustum culling, CSM setup, depth-pyramid validity checks, and GPU culling inputs stay on the unjittered view-projection matrix.

## History Resources

Two `VK_FORMAT_R16G16B16A16_SFLOAT` images at the render extent are recreated with the swapchain. Render Graph imports them as persistent resources:

- `TAAHistoryRead`
- `TAAHistoryWrite`

`TAAResolvePass` reads the previous history and writes the current history. Later post-process passes read the current history through descriptor variants.

History is invalidated on swapchain/post-process recreation, runtime TAA setting changes, camera resets/edits, scene load, portfolio mode application/restoration, material asset save/reload, and explicit debug reset. With reprojection active, ordinary camera motion no longer needs a reset — the resets remain for discontinuities (cuts, scene swaps).

## Resolve Shader

`src/shaders/taa_resolve.frag` performs:

- current HDR sample
- closest-depth 3x3 velocity dilation (silhouette edges reproject with the
  foreground object's motion); skipped when the main depth image cannot be sampled
- history sample at the velocity-reprojected UV, rejected when it lands off-screen
- current-frame 3x3 min/max neighborhood
- optional neighborhood clamp of history
- bounded feedback blend

## Validation Surface

Expected debug signals:

- Render Graph shows `TAAResolvePass` when TAA is enabled, with `VelocityBuffer`
  written by `MainHDRPass` and read by the resolve.
- GPU Profiler shows `TAAResolvePass` when TAA is enabled.
- Render target debug views show both TAA history images once they reach shader-read layout.
- Toggling TAA or pressing reset makes history invalid for the next resolve.
- Toggling `Motion reprojection` off while orbiting the camera brings back the
  Phase 6B ghosting; on, moving objects and camera motion stay sharp.

## TAA makes latent depth bugs visible

Jitter moves the rasterized position a fraction of a pixel every frame, so the
interpolated depth at each pixel centre moves with it. Any depth comparison that
was *marginal* — coplanar surfaces, a bias that only just clears acne — stops
being resolved the same way every frame and starts alternating.

That turns a static, easily-missed seam into a visible flicker, and it means
"this only flickers with TAA on" is usually **not** a TAA bug. Bisect it with the
panel's three toggles, which separate the causes cleanly:

| Symptom | Reading |
| --- | --- |
| Flicker stops with **jitter** off, frozen in one of two states | The current frame genuinely alternates. Look for marginal depth comparisons, not for a history bug. |
| Flicker stops with **neighborhood clamp** off, settling on a blend | Same cause — history is now low-passing a real two-state signal instead of tracking it. |
| Flicker stops with **motion reprojection** off | A history-path bug: velocity disagreeing with what the pixel actually shows. |

The first two point at the scene, not the resolve. Found this way: the portfolio
glass panes had their bottom face exactly on the floor's top plane, so their
depth test against the floor flipped every frame and a strip along the floor line
flickered. The fix was in `SceneBuilder` (see `kPortfolioFloorSinkDepth`), not
here. Blended geometry shows it worst, since it only depth-tests — a flip decides
whether the pixel has the surface on it at all.

## Limitations

- Skinned joint-space motion is not captured (previous-palette velocity is future work).
- No disocclusion mask or confidence weighting beyond neighborhood clamping.
- No temporal upscaling.
- No FSR/DLSS/XeSS.
- No per-material reactive mask.
- Editor object teleports produce one frame of large velocity (clamped by the
  neighborhood bound) rather than a per-object history reset.
