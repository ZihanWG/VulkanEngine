# Render Scale

Decouples the resolution the scene is *shaded* at from the resolution it is
*presented* at. Everything up to and including TAA and the bloom chain runs at
the render extent; the composite pass upscales to the swapchain, and the ImGui
overlay stays native.

Default is 1.0 (native) — this changes nothing unless the slider is moved.

## Why this and not something else

By 2026-08 the frame was thoroughly fragment-bound and had been mapped in
detail: `MainHDRPass` is 56% of the default scene's frame and 78% of the
fragment stress scene's, and roughly two thirds of the pass is the clustered
punctual light loop. That loop is **real shading work, not waste** — the cheap
wins (a duplicated debug shadow lookup, a duplicated 96-byte slot load) were
already found and taken, and two further micro-optimisations were measured and
rejected because the cost is memory traffic per light, not tap count.

When per-pixel cost cannot go down, the remaining lever is the pixel count. That
is an architectural knob rather than a local one, and unlike the alternatives it
is cheap to build and cheap to reverse:

- **Deferred / visibility buffer.** Would not help here. The main cost is
  shading *visible* pixels, and this is a tile-based deferred GPU (Apple M3 via
  MoltenVK) that already resolves hidden surfaces in hardware, so there is
  little overdraw for a G-buffer to remove — while the extra attachment
  bandwidth is exactly what a tiler is worst at.
- **Fewer or cheaper lights.** Already pushed: the cluster grid was refined to
  32×18×24 for a 19% `MainHDRPass` win, and per-light data was packed down.
- **Temporal upscaling (FSR/DLSS-style).** The quality answer, and a natural
  follow-on — but it needs this plumbing first regardless.

## Measured

Default scene, Debug build, 2560×1440, Apple M3. Medians of the last three
GPU-timing prints per run; the 1.0 control was re-measured after the series and
held (16.7 → 17.3 ms), so the deltas are not machine drift.

| Scale | Render extent | Frame total | `MainHDRPass` |
| --- | --- | --- | --- |
| 1.00 | 2560×1440 | 16.9 ms | 9.2 ms |
| 0.75 | 1920×1080 | 11.0 ms (−35%) | 5.4 ms (−41%) |
| 0.50 | 1280×720 | 6.2 ms (−63%) | 2.4 ms (−74%) |

The saving is *more* than the pixel ratio at 0.5 (56% of the frame gone for 75%
of the pixels gone). Two reasons: `Transparent` shares the same fragment shader
and falls with it (1.5 → 0.1 ms), and LOD selection is driven by projected pixel
radius, so a smaller render target legitimately picks coarser meshes.

## What is sized by the render extent

Everything screen-space except the three things listed under "and what is not":

- main depth (`MainDepth`) and the Hi-Z depth pyramid built from it
- `SceneColorHDR`, the velocity buffer, the thin G-buffer (normal/roughness)
- GTAO (its half-res trace target is half of *this*, not half of the window)
- the SSR scene-colour copy
- both TAA history images
- the legacy bloom targets and the whole mip chain
- the luminance/histogram compute dispatch dimensions, which read scene colour

...and what is not:

- **`CompositePass`** — its viewport is the swapchain. It samples scene colour
  through a `VK_FILTER_LINEAR` sampler with normalised UVs, so the upscale costs
  nothing extra and needed no shader change.
- **`ImGuiPass`** — runs after the composite, so the debug UI and any text stay
  crisp at any scale.
- **Portfolio screenshots** — captured from the composited swapchain image, so
  they come out at window resolution.

Two frame-setup values also follow the render extent, deliberately:

- **TAA jitter** is an NDC offset of half a *render* pixel. Against the window
  extent it would be the wrong size and the sequence would not resolve.
- **GPU cull LOD selection and the occlusion pass's screen-size thresholds** are
  in rendered pixels. A half-scale frame genuinely needs less mesh detail.

## Implementation

The scale is one persisted float (`renderScale.scale` in
`config/runtime_settings.json`), clamped to `[0.25, 1.0]` by
`clampRenderScale`. Above 1.0 would be supersampling, which the composite's
bilinear stretch is the wrong filter for; below 0.25 nothing survives.

`renderer::RenderResolution` (`src/renderer/RenderResolution.h`) holds the pair
of extents. `Renderer` owns one and every resolution-dependent subsystem —
`PostProcessStack`, `DepthPyramid`, `ScreenSpaceReflections`,
`GroundTruthAmbientOcclusion` — borrows it by const reference, the same
reference-borrowing pattern they already use for the swapchain. A setter per
subsystem was the alternative and it has four places to forget on a resize; a
subsystem that missed one would allocate a target at the wrong size and fail
somewhere else entirely.

The pure math (`clampRenderScale`, `scaledRenderDimensions`) lives in
`src/renderer/RenderScale.h`, free of Vulkan so it compiles into
`VulkanEngineCore` and is unit-tested without a device. It rounds to nearest
rather than truncating, treats NaN as native, and never returns a zero
dimension — a zero-sized attachment is invalid and every caller would otherwise
need its own guard.

### The main depth image is the awkward one

`MainDepth` is owned by `VulkanSwapchain` (it has always shared the swapchain's
lifetime) but must now be sized like the *internal* targets. It cannot simply be
created at the scaled size inside `VulkanSwapchain::create`, because the render
extent can only be derived after the swapchain has picked its actual size — the
surface may not grant the requested one.

So creation stays two-step: the swapchain creates depth at its own extent, then
`Renderer::updateRenderResolution` calls `VulkanSwapchain::resizeDepthImage`
with the render extent. That call is a no-op when the sizes already match, so at
scale 1.0 nothing is allocated twice. `depthExtent()` is what `RenderGraph`
imports as the `MainDepth` resource extent, and that extent becomes the
`renderArea` of every pass writing it.

### Changing the scale at runtime

A scale change resizes exactly the set of targets a window resize does, so it
takes the same path. The UI slider writes `renderScaleSettings_`; `drawFrame`
notices `renderResolution_` disagrees with it and calls `recreateSwapchain()` at
the top of the frame, which is the only point where destroying in-flight targets
is safe. The slider itself edits a pending value and commits on release —
committing waits for the device to go idle and rebuilds everything, which is not
something to do once per dragged frame.

## Using it

Debug panel → **Render Scale** (visible in both simple and advanced mode). The
slider and the 100/75/50/33% preset buttons both commit immediately on release;
the panel reports the two extents and the resulting shaded-pixel percentage.

Turn TAA on with it. Render scale trades spatial detail for speed and TAA
recovers some of that detail across frames, which is why the two ship together
in every engine that has them.

## Limitations

- The upscale is a plain bilinear stretch in the composite. There is no sharpen
  pass and no temporal upscaling (FSR/DLSS/XeSS-style), so at 0.5 and below
  edges are visibly soft.
- The scale is static. Dynamic resolution — driving it from a frame-time target
  — is the obvious next step and needs no new plumbing, only a controller.
- Non-uniform scaling (different X and Y) is not supported; the aspect ratio is
  preserved so the projection needs no change.
