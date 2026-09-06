---
name: mainhdrpass-attribution
description: "Ablation results for MainHDRPass: the debug-only punctual shadow lookup cost 2.9ms (fixed, branch perf/skip-debug-shadow-lookup); CSM sampling is only 0.45ms; series stopped when the machine thermally degraded"
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-08T05:54:41.025Z
---

Started 2026-08-06 because `MainHDRPass` was ~14 ms of a 23 ms frame and nested
profiler scopes cannot attribute it on this TBDR (see
[[gpu-profiler-nested-scopes]]). Method: ablate one term in
`src/shaders/simple_bindless.frag`, rebuild (~10 s), run 35 s, take medians,
revert. `Transparent` reuses the same fragment shader, so it moves too and is
part of the signal, not noise.

## Found: half of all punctual shadow sampling was thrown away

**`MainHDRPass` 14.074 -> 11.197 ms (-20%), `Transparent` 2.438 -> 1.929,
frame 23.032 -> 19.410 ms (-16%).** Non-overlapping ranges on all three.

Every fragment ran the 9-tap punctual shadow atlas PCF **twice per light**:
once in `evaluatePunctualLight` for shading, once via
`punctualShadowDebugFactor` to accumulate `punctualVisibility`. That variable is
read in exactly one place — the overlay gated on `pc.debugPunctualShadows` — so
with the overlay off (the default) the second lookup was pure waste.

Fixed by gating the accumulation on the same flag that consumes it. **MERGED AND
PUSHED** as `8899b81` (`--no-ff`); main == origin/main; branch deleted. Confirmed
on merged main once the machine had settled: frame 19.425, MainHDRPass 11.113,
Transparent 2.019 — reproducing the ablation figures.

Generalisable: grep this shader (and others) for values computed unconditionally
but consumed only under a debug flag. `punctualVisibility` was the expensive one;
`clusterLightCount` and `probeIrradiance` are the same shape but cheap.

## The completed map (2026-08-07, machine quiet enough)

Against an 10.876 ms `MainHDRPass` baseline, ablating each term:

```
clustered punctual light loop (whole)   -6.92 ms   (-64%)
  of which punctual shadow PCF          -2.88 ms
  of which per-light shading math       -4.04 ms   (by subtraction)
IBL (irradiance + prefiltered + BRDF)   -0.75 ms
CSM sampleShadowFactor                  -0.45 ms   (measured 2026-08-06)
volumetric fog                           0        (off by default, block skipped)
GTAO                                     0        (off by default)
```

**The punctual light loop is roughly two thirds of the pass.** The scene is 24
orbiting point lights (`demoLightCount_ = 24`, range 8.0) plus one overhead spot,
through a 16x9x24 cluster grid — a reasonable Forward+ load, not an absurd one.

The earlier INCONCLUSIVE reading on the shadow lookup was **purely machine
degradation**; re-measured clean it is 2.88 ms, the same as the debug duplicate
cost, which makes sense since both ran the identical 9-tap PCF.

**Unlike the three wins before it, this is real shading work, not waste.**

## Two optimisations tried and REJECTED on measurement (2026-08-07)

**1. Fewer PCF taps — not worth it. The taps are not the cost.** Replacing the
3x3 loop with a single tap saved only **0.51 ms** of the 2.88 ms the whole
`punctualShadowFactor` costs. So ~2.4 ms is everything *around* the taps: the
96-byte `GpuShadowSlot` load (read twice — once for `params.y` at the base slot,
once in full after face selection), the `viewProjection` transform, and the
bounds tests. **This function is memory-bound on per-light data, not ALU-bound on
PCF.** Any future attempt should attack the slot load, not the tap count.

Also found while scoping it: **the size classes do not discriminate on this
scene.** Instrumented histogram over 25,800 assignments: class 0 = 4% (the single
overhead spot), class 1 = 96% (all 24 orbiting point lights), class 2 = never.
So "adaptive taps by size class" degenerates to all-or-nothing here — there is no
per-light importance signal to adapt to without a new one.

**2. Back-facing early-out — no win.** `normalLight` was computed *after* the
shadow lookup and the result is scaled by it, so a surface facing away pays for
the lookup and returns zero. Hoisting `dot(normal, lightDirection) <= 0` above
the lookup is mathematically equivalent and looked like free money. Measured
back to back: 11.879 -> 12.286 ms, i.e. no gain and possibly a small loss. The
hypothesis that "roughly half the lights are behind any surface" is wrong for
lights that survive the range test (range 8.0) — those are mostly on the lit
side already, so the branch adds divergence without skipping much.

Both experiments were reverted; nothing committed.

## DONE: the duplicate slot load is gone (merged `ae17c88`, pushed)

Shipped 2026-08-07 once the machine settled (load 2.36, no Safari GPU process;
two controls agreed to 0.24 ms). **MainHDRPass 11.214 -> 10.457 ms (-7%), frame
19.418 -> 18.279 ms.** Every changed run beat every control run across 5 runs,
which is the bar to use when an effect is only ~3x the control spread.

`normalBias` now rides in `GpuLight::spotScaleOffset.w`, written in the same
loop that resets the slot index (`RendererFrame.cpp` ~290) — that loop runs after
`updateDemoLights` rebuilds the lights and before any slot is assigned, so it is
always current. The shader reads an identical float, since both it and the
record's `params.y` come from `PunctualShadows::normalBias_`.

**`params.y` was deliberately NOT removed**: `probe_capture.frag` (~line 202) has
the same two-read pattern and still uses it. Deleting it would have silently
broken probe capture's shadow bias. Probe capture is off by default and costs
~0.3 ms, so it did not get the same treatment — but the pattern there is
identical if it ever becomes worth it.

The original reasoning, kept because it is what made the fix findable:

`punctualShadowFactor` in `src/shaders/simple_bindless.frag` reads the slot
buffer **twice**:

```
line 501:  float normalBias = pc.punctualShadowSlots.slots[baseSlotIndex].params.y;
line 515:  GpuShadowSlot slot = pc.punctualShadowSlots.slots[slotIndex];
```

They cannot simply be merged, and this is the trap: the face index for a point
light is selected from the *biased* position, which needs `normalBias` — so the
first read must happen before the second's index is known. (Selecting the face
from the unbiased direction is the bug Milestone 59 documents, so do not "fix"
it that way.)

**The fix is to stop putting `normalBias` in the slot at all.** Verified while
scoping:
- `GpuLight::spotScaleOffset.w` is **unused** — grep found no `.w` or `[3]`
  reader anywhere in shaders or C++. `GpuLight` is already loaded in the loop and
  passed into `punctualShadowFactor`, so carrying `normalBias` there costs
  nothing at runtime and removes read 1 entirely.
- `normalBias_`/`constantBias_` are **members of `PunctualShadows`, i.e. global**,
  not per-light (`PunctualShadows.cpp` ~125 and ~159 build every record from the
  same two values). So a push constant would work too, but push-constant space is
  tight and `spotScaleOffset.w` is free — prefer it.

Expected win is bounded by the ~2.4 ms non-tap portion, and removing one of two
reads is plausibly 0.5-1 ms. **Measure it, do not assume**, and follow the
protocol in [[exposure-reduce-serial-bottleneck]]: back-to-back control, and
re-check the control every few runs since it drifted 10.885 -> 11.879 over four
runs during this session.

`Transparent` tracks `MainHDRPass` throughout because it reuses the same
fragment shader (1.759 -> 0.380 with the light loop gone).

## The machine degrades under a long measurement series

After roughly a dozen consecutive 35 s GPU runs, the **control** (identical code,
already measured at 19.410 ms frame / 11.197 ms MainHDRPass) came back at 45.6 ms
/ 27.4 ms, then 38.5 ms with a range of [19.170, 95.393]. Load average 5.74.

So: **an ablation series has a limited budget of trustworthy runs.** Re-measure
the control every few ablations, not just at the start — if the control has
moved, discard everything since the last good one. This is the stronger version
of the back-to-back rule in [[exposure-reduce-serial-bottleneck]]: back-to-back
pairing is necessary but not sufficient, because both halves of a pair can be
degraded together.


## 2026-08-14: effective-radius light culling MEASURED AND REJECTED (`b1a0a50`)

Idea: the cluster cull uses each light's authored range, but the windowed
inverse-square falloff (`rangeFade = 1 - (d/range)^4`, squared) drives the
contribution toward zero well before it, so a tighter cull radius should shorten
every cluster's light list for free.

It is not free. Uniform radius sweep, control repeated:

```
100%   MainHDRPass 9.85 / 9.81    luminance 0.3129 / 0.3121
 85%               9.39                     0.3033  (-3.1%)
 70%               9.37                     0.2651  (-15%)
 50%               8.55                     0.2208  (-29%)
```

The image darkens faster than the pass shortens, and the saving flattens between
85% and 70%. Two dozen overlapping range-8 lights mean the aggregate of many
small near-boundary contributions is a visible fraction of the image; there is no
threshold both worth having and invisible. **Don't re-offer this.**

The useful part is the bound: at a 50% radius (87% of the light volume thrown
away) MainHDRPass falls only 13%. **The loop is not spending its time on lights
that contribute nothing** -- which closes the "find work that produces nothing"
line of attack on this pass. The cheap waste elsewhere in the frame is mined out
(see [[occlusion-yield-initiative]], [[shadow-cascade-cost]],
[[exposure-reduce-serial-bottleneck]]); what is left in MainHDRPass is real
shading, and moving it means a quality trade.
