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

Requesting a change and applying one are deliberately separate. The UI slider —
and the dynamic-resolution controller below — only write
`renderScaleSettings_.scale`. `drawFrame` notices `renderResolution_` disagrees
with it and calls `applyRenderScaleChange()` at the top of the frame, which is
the only point where destroying in-flight targets is safe. One mechanism serves
both requesters, and the controller's decision stays visible on the slider and in
the saved settings rather than living somewhere private.

`applyRenderScaleChange` is cheaper than the window-resize path it replaced:
`recreateSwapchain` also rebuilds the swapchain, its semaphores and the ImGui
backend, none of which a scale change touches. What it does do is idle the
device, recreate every screen-sized target, and invalidate the TAA history and
the depth pyramid — both hold a frame of data at the previous resolution, and the
pyramid would reject visible geometry if it were trusted.

The slider itself edits a pending value and commits on release, since each commit
costs a rebuild.

## Dynamic resolution

Off by default, and that default is deliberate: a portfolio engine is usually
being *measured*, and a resolution that moves on its own invalidates every A/B
comparison in the profiler panel.

`renderer::DynamicResolutionController` (`src/renderer/DynamicResolution.h`)
picks the scale from measured frame time. Like the render-scale math it is free
of Vulkan and unit-tested without a device — it decides a number and nothing
else, so all of its behaviour is testable.

### The signal is GPU frame time, not the CPU frame delta

The CPU delta includes the present wait, so under vsync it reads as the refresh
interval no matter how much headroom the GPU has — a controller driven by it
would never raise the scale. It also absorbs the CPU cost of *applying* a scale
change, which would feed the controller its own cost and walk it downward.

GPU frame total is the quantity render scale actually moves. The consequence,
stated plainly: a CPU-bound frame is invisible to this controller. That is
correct — lowering the resolution would not fix it.

### Median, not average

Samples are reduced with a median over a 9-frame window. A single hitch — a
shader compile, a texture upload, another process taking the GPU — must not drop
the resolution, and an exponential average cannot promise that at any usable
weight: slow enough to absorb a 4× spike is too slow to respond to real load. A
median over an odd window is immune to outliers up to half the window by
construction.

This is also why `Renderer` passes only *fresh* readings. GPU timestamps arrive
a few frames late, so `gpuFrameTimeHistory_.latest()` repeats between readbacks;
feeding repeats would fill the window with duplicates of one sample and defeat
the whole point. `pushGpuTimingSample` publishes the new value and
`updateDynamicResolution` consumes it once.

### How it decides

Cost is per pixel and pixel count goes as the square of the scale, so the scale
that would land on target is `currentScale * sqrt(target / measured)`. That
over-corrects by whatever share of the frame is resolution-independent (shadows,
culling, the cluster passes), so the request is capped to a step rather than
applied outright — and the controller is asymmetric, because dropping frames is
happening now while unused headroom only costs sharpness:

| | |
| --- | --- |
| max step down | 0.15 |
| max step up | 0.05 |
| deadband | target × [0.85, 1.0] — no action on budget |
| quantisation | 0.05, snapped toward the direction of travel |
| settle window | 12 measurements ignored after a change |

The settle window must outlast the sample window or a decision gets made from
frame times measured at the previous resolution; a `static_assert` pins that.
Snapping toward the direction of travel rather than to nearest matters too:
rounding to nearest can land back on the scale it started from, and then the
controller makes no progress while remaining convinced it should.

The unit tests cover the properties that are easy to get wrong and invisible at
runtime: a converged controller stops changing the scale (closed-loop, frame
time proportional to pixel count), a single spike moves nothing, bounds hold
even when stored the wrong way round, and no measurement means no decision.

### Measured, and its honest cost

Default scene, target 8 ms, min 0.25, Debug:

```
1.00 -> 0.85 -> 0.70 -> 0.60 -> 0.55, then no further changes
holds ~7.0 ms against the 8 ms budget (deadband 6.8-8.0)
```

Four changes to converge, then it stops — which is the behaviour that matters,
since **each change costs a one-frame CPU hitch**: it idles the device and
rebuilds every screen-sized target. Measured 27.4 / 18.0 / 13.8 / 11.6 ms for
the four changes above (the first is highest — allocations and descriptor pools
are cold).

That cost is the reason for the quantisation, the deadband and the settle
window: converging in a handful of steps and then holding still is affordable,
continuous adjustment would not be. The debug panel reports the last apply cost
so the trade is visible rather than assumed.

**The alternative, not built:** allocate every target once at the maximum scale
and render into a sub-rect, so changing scale is only a viewport change and
costs nothing. That is what makes continuous adjustment viable and it is the
right end state — but it means every pass that samples a partially-filled target
needs its UVs scaled to the valid region (composite, TAA, bloom, SSR, GTAO, the
Hi-Z occlusion test in `cull.comp`), and per-mip rounding in the bloom chain
makes reading past the valid region an easy mistake with visible consequences.
Rebuild-on-change converges in four hitches and never lies about the result;
that is the better trade until the sub-rect work is done properly.

## Using it

Debug panel → **Render Scale** (visible in both simple and advanced mode). The
slider and the 100/75/50/33% preset buttons both commit immediately on release;
the panel reports the two extents and the resulting shaded-pixel percentage.

Under **Dynamic resolution** in the same section: the enable toggle, a target
expressed as FPS (the stored value is the millisecond budget), and the scale
bounds. Enabling it disables the manual slider, since the controller owns the
value from then on. The readouts are the median GPU frame time, the change
count, a "settling" line while the controller is deliberately not acting, and
the cost of the last apply.

Turn TAA on with it. Render scale trades spatial detail for speed and TAA
recovers some of that detail across frames, which is why the two ship together
in every engine that has them.

## Limitations

- The upscale is a plain bilinear stretch in the composite. There is no sharpen
  pass and no temporal upscaling (FSR/DLSS/XeSS-style), so at 0.5 and below
  edges are visibly soft.
- Applying a scale rebuilds every screen-sized target and costs a one-frame CPU
  hitch (12-27 ms here). Dynamic resolution converges in a handful of steps and
  then holds still, so this is bounded rather than continuous — but the sub-rect
  approach described above is what would remove it.
- The controller only sees GPU frame time, so it cannot respond to a CPU-bound
  frame. Correct, but worth knowing when a target is not being met.
- Non-uniform scaling (different X and Y) is not supported; the aspect ratio is
  preserved so the projection needs no change.
