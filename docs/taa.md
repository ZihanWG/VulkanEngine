# Temporal AA

Phase 6B added a conservative TAA foundation (jitter + same-UV HDR history resolve).
The motion-vector upgrade extended it into a real reprojecting TAA: the main pass
writes a velocity buffer, and the resolve reprojects history along per-pixel motion
with closest-depth velocity dilation.

It is now also the **upsampler**. The resolve runs at presentation resolution and
reconstructs each output pixel from the jittered low-resolution samples nearest
to it, so render scale below 1.0 no longer ends in a bilinear stretch. See
[Temporal upsampling](#temporal-upsampling) below.

## Runtime Controls

TAA is off by default. It can be enabled from:

- `VulkanEngine Debug` -> `Image` tab -> `Temporal AA`
- `config/runtime_settings.example.json` and saved runtime settings under the `taa` object

Runtime fields:

- `enabled`: runs the TAA resolve pass and routes post-processing through resolved history.
- `jitterEnabled`: applies subpixel projection jitter while TAA is active.
- `neighborhoodClampEnabled`: bounds previous history by the current nine-tap neighbourhood.
- `varianceClipping`, `varianceGamma`, `rejectionFeedback`: stricter rejection,
  both toggles off by default -- see [Ghosting rejection](#ghosting-rejection).
- `disocclusionRejection`, `disocclusionTolerance`, `debugDisocclusion`: drop the
  history where depth says it shows a surface that is no longer there. On by
  default at a relative tolerance of `0.05` (clamped to `0.005..0.5`); the debug
  toggle paints rejected pixels red -- see
  [Disocclusion rejection](#disocclusion-rejection).
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
- **Skinned meshes**: the previous clip position is the vertex skinned by the
  *previous* frame's joint palette and projected by the previous rigid MVP, so
  velocity carries joint-space motion as well as camera and rigid object motion.
  Each per-frame palette buffer holds both palettes -- this frame's, then the
  previous frame's at `kSkinPreviousPaletteOffset` -- so the pair is guarded by
  the one fence that already guards the slot. Reading the previous frame's
  *slot* instead would race the CPU rewriting it for the next frame, and with
  one frame in flight there is no other slot. A rebuilt rig, or the first frame,
  reports zero joint motion rather than a velocity from an unrelated pose.
- **Skybox**: the fragment shader projects the view direction with `w = 0` through
  the previous view-projection (translation drops out), producing rotation-only sky
  motion — correct for an infinitely distant environment.

## Frame Flow

1. `MainHDRPass` renders skybox and meshes into `SceneColorHDR` + `VelocityBuffer`.
2. `DepthPyramidPass` still builds Hi-Z from main depth after the HDR pass.
3. If TAA is enabled, `TAAResolvePass` reconstructs from `SceneColorHDR`, velocity,
   main depth and the previous history, and writes the current history **at
   presentation resolution** -- this is the upsample.
4. Bloom, luminance, histogram exposure, and composite sample the active HDR source: resolved TAA history when TAA is enabled, otherwise `SceneColorHDR`.
5. After post-processing uses the current history image, the renderer flips the ping-pong history index for the next frame.

## Jitter And Matrices

The renderer uses an 8-sample Halton sequence in bases 2 and 3. Jitter is applied to the projection matrix as a subpixel NDC offset for skybox and mesh rendering.

CPU frustum culling, CSM setup, depth-pyramid validity checks, and GPU culling inputs stay on the unjittered view-projection matrix.

## History Resources

Two `VK_FORMAT_R16G16B16A16_SFLOAT` images at the **presentation** extent, recreated with the swapchain. They are the one screen-space pair that is written in full rather than into the render sub-rect, because they hold the upsampled result. Render Graph imports them as persistent resources:

- `TAAHistoryRead`
- `TAAHistoryWrite`

`TAAResolvePass` reads the previous history and writes the current history. Later post-process passes read the current history through descriptor variants.

The history's alpha channel is not colour. With disocclusion rejection on, the
resolve writes `1 + view depth` of the surface each pixel shows into it, for the
next frame's test; otherwise it writes `1`. Every downstream consumer reads
`.rgb` only, and the offset keeps the render-target preview, which draws the
image through ImGui's alpha blend, opaque.

History is invalidated on swapchain/post-process recreation, runtime TAA setting changes, camera resets/edits, scene load, portfolio mode application/restoration, material asset save/reload, and explicit debug reset. With reprojection active, ordinary camera motion no longer needs a reset — the resets remain for discontinuities (cuts, scene swaps).

## Resolve Shader

`src/shaders/taa_resolve.frag` runs at presentation resolution and performs:

- a nine-tap weighted reconstruction of the current frame from the low-resolution
  source (see below)
- closest-depth 3x3 velocity dilation (silhouette edges reproject with the
  foreground object's motion); skipped when the main depth image cannot be sampled
- history sample at the velocity-reprojected UV, rejected when it lands off-screen
  or when depth says it is disoccluded (see below)
- a neighbourhood box from the same nine taps -- one gather, two jobs
- optional clamp of history into that box
- bounded feedback blend

## Temporal upsampling

The resolve is what turns a low-resolution frame into an output-resolution one,
so it is the only pass whose viewport is the presentation extent while its
sources are the render sub-rect.

**Why the jitter is what makes it possible.** The projection is offset by
`+jitter` each frame, so a source texel whose centre is `c` holds what an
unjittered frame would have at `c - jitter`. Undoing that gives every sample a
real sub-pixel position. An output pixel is then reconstructed from the samples
nearest to *it*, weighted by how far each one actually landed:

```
weight = exp(-2.29 * d^2)      d in source pixels
```

One texel away keeps 10% of the weight, half a texel 57%. Tighter and an output
pixel that no sample landed near goes noisy; looser and the reconstruction is a
blur, which would make undoing the jitter pointless.

**The distinction that matters.** Accumulating plain bilinear taps can only ever
converge on an upscale of itself, however many frames it runs for -- the
information about where each sample really was has already been thrown away by
the time it is accumulated. Reconstructing first is what lets the sequence
converge on the image.

**The history stops being a low-resolution source.** Scene colour, velocity and
depth share the render sub-rect and are sampled through `sourceUvScale`; the
history is written in full and is sampled directly. Neighbourhood offsets are
one *source* texel expressed in output space, which means dividing the texel size
by that scale. Measuring the box in output texels instead would clamp the history
against an upscaled blur of itself and stop it rejecting anything.

**Everything downstream becomes full-resolution.** `activePostProcessSource()` in
`PostProcessStack` reports the allocation with a uv scale of 1 when TAA is
active, and bloom, luminance, the histogram and the composite follow without
individually knowing why. The composite's sharpen filter goes with it: it exists
to undo the composite's own bilinear stretch, and with the stretch gone its gate
("is the composite actually stretching") is false.

## Ghosting rejection

Two stricter history-rejection mechanisms, **both off by default**, under
`Image -> Temporal AA -> Ghosting rejection`.

They are off because every form of stricter rejection costs accumulated history,
which is the thing temporal upsampling exists to gather. They were written for
ghosting that turned out not to be present in this scene -- eleven static objects
and orbiting lights leave camera motion as nearly the only motion -- and with
nothing wrong to reject they are cost without benefit. They are kept because
ghosting is a real failure mode and content with object motion will produce it.

- **Variance clipping** bounds the history by the neighbourhood's mean plus or
  minus `varianceGamma` standard deviations in YCoCg, rather than by its RGB
  extremes. An extremes box is set by its two most extreme samples, so a single
  bright speck widens it enough for a ghost to sit inside. YCoCg is not the point
  in itself: putting luma on one axis is, because a ghost almost always differs
  from where it drifted to in brightness, and in RGB that difference is spread
  across three correlated channels where each alone looks unremarkable. The
  history is clipped toward the box centre rather than clamped per channel --
  per-channel clamping moves each axis independently and shifts hue, so a red
  ghost over grey becomes a grey-red that was nowhere in the neighbourhood.
- **Rejection feedback** is what actually removes a ghost. A clipped history is
  only the nearest *plausible* colour and still receives most of the pixel, so it
  still trails. The distance it had to travel to become acceptable is itself the
  ghosting signal, and that distance drives its feedback down. It is equally the
  mechanism that costs most when there is no ghost.

## Disocclusion rejection

A pixel is disoccluded when the surface it shows now was hidden last frame,
behind something that has since moved off it. Its reprojected history then holds
the occluder, not the surface. The neighbourhood clamp catches that only when the
occluder's colour falls outside the current neighbourhood; one that happens to
sit inside it survives and trails. Depth does not have that blind spot, because
the occluder was at a different distance.

**Where the previous depth comes from.** Not the Hi-Z pyramid. Its mip 0 is a
copy of main depth, but by the time the resolve runs, the two-phase mid-frame
rebuild and the end-of-frame build have both refilled it from *this* frame, and
the occlusion-yield controller can suspend the build altogether. The history
carries it instead: the resolve records, in the history's alpha, the view depth
of the reconstruction tap that dominates each output pixel -- the surface that
pixel's colour is mostly made of. Sky records as `60000`, far past any surface
and still inside 16-bit float range. At the format's 11-bit mantissa a recorded
depth is good to about 0.05%, two orders of magnitude inside the default
tolerance.

**What it is compared with.** For each of the nine source taps, the depth that
surface *would* have had last frame if it had stood still: the previous frame's
clip `w` at the tap's world position. The CPU folds that into two rows once a
frame (`taaPreviousDepthRows` in `src/renderer/TaaDisocclusion.h`), so per tap it
is two dot products and a divide, with no matrix inverse in the shader. The rows
are built from the *jittered* view-projection, because that is what rasterised
the depth buffer; jitter shears only x and y, so it leaves the depths alone.

**The test.** The four history texels of the bilinear footprint at the
reprojected position are gathered, and the history is rejected when *none* of
them lies inside the range of expected depths across the 3x3, widened by the
tolerance. Four texels and a range rather than one depth against one, because
the test also runs on silhouettes, where both the footprint and the
neighbourhood straddle an edge -- demanding that every texel match would reject
history exactly where anti-aliasing needs it. Sky in the neighbourhood leaves the
range open above. A texel with no recorded depth (alpha `1`, from a history
written with the test off) cannot disagree, so turning the test on never
rejects the frame it was turned on.

A rejected pixel takes the current frame's reconstruction alone, the same as an
off-screen reprojection. The test needs a samplable main depth and reprojection;
without either, and with the setting off, the resolve reads no depth for it and
writes alpha `1`.

**What it does on the scenes here.** RTX 3080 Ti Laptop, 1280x720,
`--deterministic`, frame 30 of 40. With the camera still, rejection changes
pixels only around the animated skinned mesh: 20 on the default scene, 117 on
`--scene sunlit`. There is no headless camera-motion flag, so the camera was
orbited by a patch applied locally for the run (0.01 rad and a 0.3% dolly per
frame, not committed); rejection then changed 0.29% and 0.14% of the frame, and
the debug view shows single-pixel lines along the trailing edge of each
silhouette and nothing across flat surfaces. Its GPU cost has not been measured
-- nine depth fetches and one gather per output pixel, beside the nine depth
reads velocity dilation already makes.

**Rejection off is not bit-identical to before the test existed.** With the
setting off, a TAA frame differs from the previous resolve on 941 of 921600
pixels on the default scene, almost all by at most one level. The difference is in the
compiled shader, not the logic: the same source with the depth path compiled
out matches the old output exactly, and writing a constant alpha with the path
left in reproduces the drift, so nothing the alpha carries reaches colour. A
larger shader rounds a little differently, and TAA feedback carries it forward.

## Validation Surface

Expected debug signals:

- Render Graph shows `TAAResolvePass` when TAA is enabled, with `VelocityBuffer`
  written by `MainHDRPass` and read by the resolve.
- GPU Profiler shows `TAAResolvePass` when TAA is enabled.
- Render target debug views show both TAA history images once they reach shader-read layout.
- Toggling TAA or pressing reset makes history invalid for the next resolve.
- Toggling `Motion reprojection` off while orbiting the camera brings back the
  Phase 6B ghosting; on, moving objects and camera motion stay sharp.
- `Debug: paint disoccluded pixels` paints each rejected pixel red. Orbiting the
  camera draws thin lines along the trailing edges of silhouettes; red across a
  flat surface, or everywhere at once, means the expected depth is wrong.

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

- Skinned velocity reflects the pose last *uploaded*, which is the pose last
  rendered except across a frame that advanced the animation and then did not
  submit (a swapchain recreate). That frame's motion is folded into the next one.
- No confidence weighting from the reconstruction: an output pixel that no
  sample landed near is blended the same as one a sample landed on.
- Not FSR2 or DLSS. The upsampling here is reconstruction plus the existing
  neighbourhood rejection; the hand-tuned machinery those spend most of their
  complexity on -- locks, reactive masks, shading-change detection -- is absent.
- Disocclusion rejection reprojects the expected depth through the camera only.
  Velocity is two-dimensional and carries no change in depth, so an object that
  moves along the view direction by more than the tolerance in one frame
  rejects its own history and loses its accumulated anti-aliasing while it does.
- The tolerance is relative, so two surfaces within it of each other -- a contact
  shadow line, a thin gap -- cannot be told apart by depth; the neighbourhood
  clamp is the only guard there.
- Blended surfaces write no depth. Across glass the test judges the opaque
  surface behind it, reprojected along the glass's own velocity.
- No per-material reactive mask.
- Editor object teleports produce one frame of large velocity (clamped by the
  neighborhood bound) rather than a per-object history reset.
