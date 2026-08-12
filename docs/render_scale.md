# Render Scale

Decouples the resolution the scene is *shaded* at from the resolution it is
*presented* at. The main pass and the screen-space effects run at the render
extent; the ImGui overlay stays native.

Where the image returns to presentation resolution depends on TAA. With it off,
the composite upscales. With it on, the **TAA resolve** does — it reconstructs at
presentation resolution from the jittered low-resolution samples, and everything
after it, bloom included, is already full size. See [taa.md](taa.md).

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
- **Temporal upscaling (FSR/DLSS-style).** The quality answer, and it needed
  this plumbing first regardless. Since built: the TAA resolve now reconstructs
  at presentation resolution, so with TAA on the composite does no stretching at
  all. See [taa.md](taa.md).

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
- the legacy bloom targets and the whole mip chain, and the luminance/histogram
  dispatch dimensions — but only while TAA is off; with it on they follow the
  resolved history and are full size

Every one of those is *allocated* at the maximum render resolution and only
**written** in its top-left sub-rect -- see "Sub-rect rendering" below. The list
above is what the render extent decides gets written, not what gets created.

...and what is not:

- **`CompositePass`** — its viewport is the swapchain. It samples scene colour
  through a `VK_FILTER_LINEAR` sampler with normalised UVs, so the upscale costs
  nothing extra and needed no shader change.
- **The TAA history pair** — written in full at presentation resolution,
  because they hold the upsampled result.
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

### Changing the scale at runtime

Requesting a change and applying one are deliberately separate. The UI slider —
and the dynamic-resolution controller below — only write
`renderScaleSettings_.scale`. `drawFrame` notices `renderResolution_` disagrees
with it and calls `applyRenderScaleChange()` at the top of the frame. One
mechanism serves both requesters, and the controller's decision stays visible on
the slider and in the saved settings rather than living somewhere private.

**Applying a scale costs 0.008 ms** and reallocates nothing: it moves viewports
and uv scales. `applyRenderScaleChange` only idles the device and rebuilds when
the *allocation* extent moved, which is a window resize.

Three things are still invalidated on every change even though no storage moves
— the TAA history, the depth pyramid and the ambient-occlusion history. All
three hold a frame of data in the previous sub-rect, and staleness has nothing
to do with whether the storage moved. That stops being obvious precisely when
nothing is being reallocated any more.

### Sub-rect rendering

Every screen-space target is created at the maximum render resolution and only
its top-left sub-rect is rendered into. The cost is memory — a 0.5-scale frame
owns four times the texels it writes — and one obligation on every consumer:
scale its UVs by the target's written/allocated ratio and stay inside. Past the
written region lies whatever the last larger frame left there, which reads as a
smear along the right and bottom edges and raises no validation error.

The invariant that keeps the conversion honest, written into
`PostProcessStack.h`, is that three quantities which used to be one number are
not any more:

| | |
| --- | --- |
| texel size | `1 / ALLOCATED` — a texel is physical; the sub-rect changes how many are written, not how big one is |
| viewport, dispatch bounds | `USED` |
| uv scale | `USED / ALLOCATED`, **per target** |

Per target is load-bearing. Derived targets — half-resolution ones, mip chains —
round each side independently, so their ratio drifts from the scene's and a
single global uv scale is wrong everywhere except the top level.
`src/shaders/sub_rect.glsl` holds the scale-and-clamp helpers.

Three things in it are easy to get wrong and are commented where they live:

- **Two offset conventions coexist and both are correct.** `composite` and
  `taa_resolve` add tap offsets in *allocated* texels after scaling; the bloom
  blur, downsample and upsample tent and `gtao_blur` add them before, so they
  divide their texel size by the uv scale to express the same physical step.
- **The depth pyramid needs a ratio per mip**, recomputed in `cull.comp` from the
  written and allocated base sizes. It is exact because repeated `max(1, n/2)`
  equals `max(1, n >> k)` — and it only works because the pyramid sampler is
  `mipmapMode NEAREST`, so exactly one level is read. Linear mip filtering would
  blend two levels with two different scales and could not be expressed as a
  single uv at all.
- **The clamp must be inert at scale 1.0.** Bounding the far edge at the last
  texel's *centre* unconditionally discards the bilinear blend across the
  outermost half-texel, and in the small bloom mips that ring is a large share of
  the image: it moved average scene luminance by 2%. A fully written target gets
  no bound, since `CLAMP_TO_EDGE` already does the right thing.

Only the *fetches* are scaled. The same UVs also feed view-space reconstruction
in SSR and both GTAO passes, and those stay in written-region space — they
describe where the fragment is on screen, not where its data sits in an
allocation.

The slider commits on release rather than per dragged frame. That is now a
courtesy rather than a necessity.

## Sharpening

A bilinear stretch is a low-pass filter, and at 0.5 the result was judged too
soft to use. `CompositePass` therefore ends with a contrast-adaptive sharpen,
strength `renderScale.sharpness` (default 0.5).

**It only runs when the frame is upscaled.** `PostProcessStack` zeroes the
strength when `RenderResolution::isNative()`, so a native frame is bit-identical
to what it produced before the filter existed and the default look is unchanged
for anyone who never moves the scale. The panel says `(inactive at 1.00)` rather
than pretending the slider does something.

Three decisions in it are worth keeping:

**Taps are one source texel apart, not one output pixel.** This is the whole
trick. Between two source texels a bilinear magnification is a linear ramp, and
the second difference of a linear ramp is zero — so output-pixel-spaced taps
would find no contrast to restore anywhere except exactly at source texel
centres, and the sharpening would come out modulated by the upscale grid. One
source texel is the spacing at which real detail exists. The shader gets it from
`textureSize(uSceneColor, 0)`, so no push constant carries it and it cannot go
stale against a resize.

**It sharpens the tone-mapped image, not linear HDR.** Same local contrast in a
bright region and a dark one means very different linear magnitudes, so
sharpening before the tone curve would boost highlights far more than shadows.
`toneMapScene` was factored out of `main` so each tap can be mapped before the
filter sees it. Bloom is deliberately shared across the taps rather than
re-fetched — it is a heavily blurred mip chain, carries no detail worth
sharpening, and reusing it keeps the whole thing to four extra fetches.

**Anti-ringing clamp.** The result may not leave the range already present in
the five taps. An unsharp mask without that bound overshoots into haloes at every
high-contrast edge, which reads as worse than the softness it was fixing.

Cost: **`CompositePass` roughly triples.** Back-to-back A/B/A/B at scale 0.5 on
a 2560×1440 window gave 0.362 / 1.460 / 0.474 / 1.227 ms, so about +0.8 ms. The
absolute values in that series are inflated — the machine had thermally degraded
over a long run of GPU measurements, and the scale-0.5 control read 13-14 ms
against the 6.2 ms measured cold — but the ratio held across both pairs.

The tone-curve evaluations are free; the four extra fetches are not. The output
is 3.7 M pixels, and at scale 0.5 the taps sit two output pixels apart, so each
one is its own cache line.

Worth it in proportion: scale 0.5 saves about 10 ms and this spends under one of
them.

### The effect is content-dependent, and that is not a hedge

The correction is a rim one *source* texel wide — two output pixels at scale 0.5
— with magnitude roughly `sharpness × localContrast / 4`, where "local" means
*within one source texel*. On a bright edge of 50% display contrast at strength
0.5 that is about 16/255, plainly visible. On a dim 10% edge it is 3/255 and
stays invisible, which is the point rather than a shortfall: amplifying
low-contrast detail is amplifying noise.

**Verified by A/B on both kinds of content**, same camera, sharpness swept
0.04 → 0.94:

| Scene | Visible? | Why |
| --- | --- | --- |
| Portfolio showcase | **No** — indistinguishable at any strength | Spheres, a panel with big holes, a gradient backdrop, all procedural solid colours. Almost nothing exists at the source-texel scale, so the filter correctly finds nothing to do. |
| Geometry stress | **Yes** | 2311 small objects put silhouette density right at the render-resolution scale, which is the frequency band this operates in. |

So judging it on the portfolio scene answers a different question than it looks
like it answers. That scene is also where the *softness* the filter exists to
fix is least visible, for exactly the same reason — the two cancel out and the
whole comparison reads as "no change".

**Show sharpen delta** in the panel settles it in one frame when the argument is
not convincing: it replaces the image with an amplified `|sharpened - original|`,
so black means the filter changed nothing there.

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

Four changes to converge, then it stops. **Applying each one costs 0.008 ms** —
sub-rect rendering means nothing is reallocated, so a change moves viewports and
uv scales and that is all. The debug panel still reports the last apply cost.

That was not always true, and the history is worth keeping because it shaped the
controller. Each change used to cost a 12-27 ms CPU hitch, of which 60-75% was
`vkDeviceWaitIdle` — and replacing that wait with waiting on the frame fences
measured *slower*, because it is real in-flight GPU work rather than driver
overhead. A wait cannot be made cheap; it can only be avoided.

The quantisation, the deadband and the settle window were sized for that world,
where converging in a handful of steps and then holding still was affordable and
continuous adjustment was not. They stay: a controller that thrashes is still
wrong even when thrashing is free, because every change invalidates the TAA
history and the depth pyramid.

## Using it

Debug panel → **Performance** tab → **Render Scale** (visible in both simple and
advanced mode). The
slider and the 100/75/50/33% preset buttons both commit immediately on release;
the panel reports the two extents and the resulting shaded-pixel percentage.

Under **Dynamic resolution** in the same section: the enable toggle, a target
expressed as FPS (the stored value is the millisecond budget), and the scale
bounds. Enabling it disables the manual slider, since the controller owns the
value from then on. The readouts are the median GPU frame time, the change
count, a "settling" line while the controller is deliberately not acting, and
the cost of the last apply.

Turn TAA on with it, and note which artefact each one fixes — they are not the
same and it is easy to conflate them. Judged on screenshots at 2560×1440, 0.5
without TAA is not usable, but what makes it unusable is **aliasing**: the cutout
panel's hole grid breaks into irregular blocks and per-pixel noise magnifies into
a visible quilt on the floor. Aliasing is TAA's job, and sharpening makes it
worse, not better. **Softness** is the artefact that remains once TAA is on, and
that is the one sharpening addresses.

Reading the first as the second is a live trap — this document did it, and a
sharpen filter got built on the strength of screenshots that were actually
showing aliasing. The filter is still the right tool for softness; the ordering
is TAA first, then sharpen what is left.

Two notes on comparing screenshots at all: the demo lights orbit, so two shots
taken at different moments differ in lighting as well as in setting, and only
sharpness is comparable between them. And pick a scene with detail near the
render-resolution scale, per the table above.

## Limitations

- With TAA **off**, the upscale is bilinear plus the sharpen above, and that is
  all it can be: spatial sharpening re-emphasises what survived and cannot invent
  sub-pixel detail. With TAA **on** the resolve reconstructs at presentation
  resolution instead and the composite stops stretching, which is why the sharpen
  gate is "is the composite stretching" rather than "is the frame native".
  It is not FSR2 or DLSS -- no locks, reactive masks or disocclusion detection.
- A 0.5-scale frame owns four times the texels it writes. That is the price of
  the sub-rect design, and it is paid in memory rather than in time.
- The controller only sees GPU frame time, so it cannot respond to a CPU-bound
  frame. Correct, but worth knowing when a target is not being met.
- Non-uniform scaling (different X and Y) is not supported; the aspect ratio is
  preserved so the projection needs no change.
