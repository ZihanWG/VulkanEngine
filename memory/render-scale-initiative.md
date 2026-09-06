---
name: render-scale-initiative
description: "Render scale initiative COMPLETE and pushed: scale + dynamic resolution + sharpen + sub-rect rendering; -63% frame at 0.5, and a scale change now costs 0.008 ms"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4da1878a-f14b-40e8-b1c0-fc00e696ebfd
  modified: 2026-08-11T07:15:40.748Z
---

The first **architecture-level** answer to a frame that local optimisation had
run out of road on. `2fb1e3e` (render scale) + `babdb3c` (dynamic resolution),
**merged `c657f35` `--no-ff` and pushed** 2026-08-11; branch deleted. Merged main
re-verified: 211 tests, 0 validation errors, 16.7 ms at defaults (unchanged).

## The visual verdict (2026-08-11, from the user's screenshots)

**0.5 without TAA is not usable — but the artefact is ALIASING, not softness**
(hole grid breaking into blocks, per-pixel noise magnified into a quilt). I
misread it as softness and justified the sharpen filter on it. Aliasing is TAA's
job; sharpening makes it worse. **Softness is what remains once TAA is on**, and
that is what sharpening actually addresses. TAA is a precondition for scale < 1,
not an option.

**Judging any of this needs content with detail at the render-resolution scale.**
The portfolio scene has none — spheres, a big-holed panel, a gradient backdrop,
all procedural solid colours — so both the softness *and* the sharpening are
invisible there, and the comparison reads as "no change" from both directions.
The **geometry stress scene** (2311 small objects) is where the sharpen A/B was
finally visible. Do not re-run this comparison on the portfolio scene.

Use the **perforated cutout panel** as the A/B test pattern; it breaks long
before the metal spheres or the glass. And note the demo lights orbit, so
screenshots taken at different moments have different lighting — only sharpness
is comparable between them, not colour or brightness.

## What it is

Everything screen-space shades at `renderScale x` the window and `CompositePass`
upscales to the swapchain. Default 1.0, so it is inert until the slider moves.
Doc: `docs/render_scale.md`. Debug panel → **Render Scale** (both simple and
advanced mode).

Measured, default scene, Debug, 2560x1440, M3. **Re-measured 2026-08-12 on a
settled machine (`0e42613`, pushed) — these supersede the earlier numbers, which
carried thermal drift.** One series back to back, 1.0 control repeated at the
end and back within 0.4% (17.72 -> 17.79), which is what makes it valid. The
machine was NOT quiet (WindowServer/Safari/WeChat active) so absolutes are still
a few % high; the deltas are sound.

```
1.00   2560x1440   frame 17.75 ms   MainHDRPass 9.7
0.75   1920x1080   frame 11.52 ms   MainHDRPass 5.78  (-35% / -40%)
0.50   1280x720    frame  6.90 ms   MainHDRPass 2.66  (-61% / -73%)
0.25    640x360    frame  5.39 ms   MainHDRPass 1.37  (-70% / -86%)
1.00 TAA on        frame 19.17 ms   TAAResolve 1.292
0.50 TAA on        frame  8.90 ms   TAAResolve 1.187
```

Beats the pixel ratio at 0.5 because `Transparent` shares the fragment shader
and falls with it and LOD picks coarser meshes for a smaller target.

**TAAU costs ~2 ms of the 10.85 ms that 0.5 saves** (resolve 1.19 + post-process
chain moving to full res). Half the saving survives, with reconstructed detail.

**~4 ms of the frame did not scale with render resolution at all** — at 0.25,
MainHDRPass was only 1.37 of 5.39, and CSMShadowPass 1.31 was the largest
remaining pass (PunctualShadowAtlas 0.13 and CompositePass 0.22 are NOT
factors). That floor has since been roughly halved by the union shadow cull —
0.25 now runs ~3.3 ms — see [[shadow-cascade-cost]]. The dynamic-resolution
minimum still stays at 0.5. Scaling the shadow maps with the render scale was
measured and buys nothing; cascade count is the lever.

## Why this and not deferred — do not re-litigate

Deferred / visibility buffer is the wrong move **on this hardware**: the cost is
shading *visible* pixels, and Apple's TBDR resolves hidden surfaces in hardware,
so there is little overdraw for a G-buffer to remove — while the extra
attachment bandwidth is what a tiler is worst at. Same reason a depth prepass is
not worth adding here. See [[mainhdrpass-attribution]] for why per-pixel cost
could not go lower.

## Design points worth not re-deriving

- **`RenderResolution` is borrowed by const reference** by the four
  resolution-dependent subsystems (PostProcessStack, DepthPyramid, SSR, GTAO),
  the same Design-B pattern they use for the swapchain. A setter per subsystem
  has four places to forget on a resize.
- **`MainDepth` lives in `VulkanSwapchain`** but must be sized with the internal
  targets, and the render extent is only derivable *after* the swapchain picks
  its actual size. Hence two-step: swapchain creates depth at its own extent,
  `Renderer::updateRenderResolution` follows with `resizeDepthImage`, which
  no-ops when the sizes match. `RenderGraph` imports `swapchain.depthExtent()`,
  not `extent()`, and that becomes the renderArea of every pass writing it.
- **TAA jitter and the GPU cull's LOD/occlusion thresholds follow the render
  extent on purpose** — the jitter is half a *render* pixel, the thresholds are
  in rendered pixels.
- **Composite needed no shader change**: it already sampled scene colour through
  a `VK_FILTER_LINEAR` sampler over normalised UVs.
- The slider commits on `IsItemDeactivatedAfterEdit`, not per frame — each
  commit idles the device and rebuilds everything.

## How it was verified without eyes on the screen

0 validation errors at 1.0 / 0.75 / 0.5, and **average luminance matched to
0.2% across scales** (0.3117 vs 0.3111). That is the cheap correctness signal
for a resolution change: a shifted or partial image would move it a lot. Same
trick as the layout-change check in [[gpu-cpu-struct-layout]].

## Phase 2: dynamic resolution (`babdb3c`) — DONE

`DynamicResolutionController`, GPU-free and unit-tested. Off by default. Target
8 ms on the default scene: 1.00 → 0.85 → 0.70 → 0.60 → 0.55, then **stops**,
holding ~7.0 ms.

Things worth not re-deriving:

- **GPU frame total, never the CPU frame delta.** The CPU delta is pinned to the
  refresh interval under vsync (so the scale would never rise) and it absorbs the
  cost of *applying* a change, which would make the controller chase itself down.
- **Median over 9 samples, not an EMA.** A unit test ("one hitch must not move
  the scale") killed the EMA outright: no weight absorbs a 4x spike while still
  responding to real load. This also forced `Renderer` to publish only *fresh*
  readings — GPU timestamps land a few frames late, so
  `gpuFrameTimeHistory_.latest()` repeats, and repeats would fill the window with
  duplicates. `freshGpuFrameMs_` is consumed once.
- **`sqrt(target/measured)`** because pixels go as scale²; then capped to a step
  (0.15 down / 0.05 up) because the resolution-independent share of the frame
  makes the raw request over-correct.
- **Snap toward the direction of travel, not to nearest** — nearest can land back
  on the current scale and stall progress.
- **Settle window (12) must exceed sample window (9)**, pinned by `static_assert`.

**The honest cost: each change is a one-frame CPU hitch of 12-27 ms** (idle the
device, rebuild every screen-sized target). Bounded because the controller
converges and stops. The fix is the sub-rect design below.

## Where it ended up

- **Sub-rect rendering** -- **COMPLETE: merged `47421b2`, `e60da53`, `096a9cf`,
  all pushed.** A render-scale change now costs **0.008 ms** (was 12-27 ms).
  Batches 1 and 2 plus the payoff. Every screen-space target is allocated at the maximum render
  resolution and written only in its top-left sub-rect.

  Two measurements that shaped it:
  - **The apply hitch is 60-75% `vkDeviceWaitIdle`** (wait ~10 ms, rebuild
    ~3-6 ms). Replacing it with waiting on the frame fences measured *slower*
    (15.8 vs 11.8 ms) -- the wait is real in-flight GPU work, **not** driver
    overhead. You cannot make the wait cheap, only not wait.
  - Bloom is ~0.4 ms at scale 1.0 vs ~0.13 at 0.5, so leaving the bloom chain at
    full size to dodge per-mip uvScale would cost 0.27 ms forever.

  **The invariant** (in `PostProcessStack.h`): texel size = `1/ALLOCATED`;
  viewport and dispatch bounds = `USED`; uvScale = `USED/ALLOCATED` **per
  target**. Derived targets round each side independently, so a global uvScale is
  the classic wrong answer. `src/shaders/sub_rect.glsl` has the helpers.

  **The stopping rule** that made this safe: the allocation flip and the
  consumer-side scaling must be ONE commit. In between, consumers read stale
  image data from the last larger frame along the right/bottom edges, and no
  validation layer says a word.

  Traps hit, all now in code comments:
  - **Main depth WAS the last holdout and is now sub-rected too.** Leaving it at
    the written extent let several consumers skip their uv scaling -- a real
    saving that bought the wrong thing: once the rebuild was skipped it meant
    destroying an image in flight (25 VUIDs). Its consumers (taa_resolve,
    ssr_trace, gtao, gtao_blur) now scale their fetches. **Only the fetches** --
    the same UVs feed view-space reconstruction and must stay in written-region
    space.
  - **Two offset conventions coexist**: composite/taa_resolve add tap offsets in
    *allocated* texels **after** scaling; bloom blur/downsample/upsample-tent and
    gtao_blur add them **before**, so they divide their texel size by the uvScale.
  - **The depth pyramid needs a per-mip ratio**, computed in `cull.comp` as
    `max(base >> mip, 1)` on both sides -- exact, because repeated `max(1, n/2)`
    equals `max(1, n >> k)`. **This only works because the pyramid sampler is
    mipmapMode NEAREST**; linear mip filtering would blend two different scales.
  - **The clamp must be inert at scale 1.0.** Bounding the far edge at the last
    texel's *centre* unconditionally discards the outermost half-texel's bilinear
    blend, which in the tiny bloom mips moved scene luminance by **2%**. A fully
    written target gets no bound.

  **The payoff (`096a9cf`)**: `applyRenderScaleChange` skips the rebuild and the
  waitIdle entirely when the allocation has not moved (only a window resize moves
  it). The luminance partial buffer had to stop being sized from the written
  extent first -- it is now sized from the allocation with the group and partial
  counts derived at record time. **Storing a derived count IS the bake-in**; that
  is the shape of the last blocker, worth recognising if another one appears.

  Still invalidated on every scale change even though nothing is reallocated:
  TAA history, depth pyramid, **and the AO history** -- all three hold a frame in
  the previous sub-rect, and staleness is unrelated to whether storage moved.
  That stops being obvious exactly when reallocation stops.

  Known cosmetic gap: ImGui render-target previews show the whole allocation,
  including the unwritten region, below scale 1.0.

  **Verification without eyes**: average luminance vs the pre-work baselines
  **0.3117 at 1.0 / 0.3111 at 0.5** -- reading unwritten texels moves it.
  **Run duration must match**: the demo lights orbit, so a 12s and a 13s run
  sample different light positions and comparing across them compares lighting,
  not code. Latest: 0.3118 / 0.3119, and 1.0/0.75/0.5/0.25 agree to 0.0001.
  The luminance check is blind to a few-pixel-wide edge -- the user checks the
  right/bottom edges on the **geometry stress scene** for that.
- ~~A sharpen pass after the upscale~~ — DONE, **merged `a77ad03` and pushed**.
  Unsharp mask in the composite on the *tone-mapped* image, gated on
  `!isNative()`, default strength 0.5. Two traps, both hit:
  **(1)** taps must be one *source* texel apart — output-pixel taps find zero
  second difference along a bilinear ramp. **(2)** the anti-ringing limiter must
  bound the overshoot, **not forbid it**: clamping into the five-tap range (the
  obvious choice, shipped first) removes almost the whole effect, because at an
  edge the centre tap is already its neighbourhood's extreme. The user reported
  "感觉没什么变化" and a 1D numeric model confirmed the ramp ends came out
  bit-identical. Now bounds overshoot at 25% of local contrast.
  Cost: CompositePass ~triples, ~+0.8 ms at 0.5 on 2560x1440, paid
  unconditionally while the benefit is content-dependent — that is the whole
  argument about the default, and the stress-scene A/B is what settled it at 0.5.
  Ships with a **Show sharpen delta** debug view (amplified |after-before|)
  because a rim this fine is otherwise unarguable by eye.
- Temporal upscaling (FSR-style) — the quality answer, much larger.
