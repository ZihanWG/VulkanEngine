---
name: exposure-reduce-serial-bottleneck
description: "Exposure path optimisation SHIPPED — shared-memory histogram + parallel reduce (6.07→0.33ms), then the redundant second full-screen scan removed (738a99b, −1.9%)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-06T13:29:59.023Z
---

**DONE AND SHIPPED 2026-08-06.** Two changes, merged `--no-ff` and pushed:
`122fe7b` (shared-memory histogram) and `42e5d59` (parallel reduction).
Together: **`Histogram Exposure` 6.07 ms -> 0.33 ms**, the frame roughly 28 -> 23 ms.

## What was wrong, and why it was the same bug twice

Both shaders in the exposure path were written as if serial work were free.

1. `luminance_histogram.comp` — one invocation per pixel doing a global
   `atomicAdd` into one of 256 bins: ~3.7M atomics contending on 256 addresses.
   Fixed by staging per-workgroup in shared memory and flushing only non-empty
   bins. 6.07 -> 1.96 ms.
2. `exposure_reduce.comp` — `local_size 1,1,1` dispatched `(1,1,1)`, i.e. **one
   GPU thread**, walking one luminance partial per 16x16 tile. At this drawable
   that is 14,400 dependent global reads with nothing to hide the latency
   behind. Fixed by a 256-thread workgroup: threads stride the partials and
   combine through a shared-memory tree, bins staged into shared memory by the
   same threads while those reads are in flight. 2.03 -> 0.33 ms.

The percentile walk stays serial on thread 0 deliberately — it is inherently
sequential (each bin's cumulative range depends on all bins before it) and is
only `binCount` iterations out of shared memory once staged.

Drawable size matters here and is easy to miss: the window is 1280x720 but
`src/core/Window.cpp` sets `SDL_WINDOW_HIGH_PIXEL_DENSITY`, so on this M3 the
actual drawable is 2560x1440.

## The methodology lesson — run the control back to back

Comparing the reduction against an **earlier session's** baseline showed the
frame saving (0.95 ms) as much smaller than the pass saving (1.63 ms), with
`MainHDRPass` apparently up 13.78 -> 14.41. That reads as "my change slowed
something else down" and would have sent the next session chasing nothing.

Re-running the *old* shader back to back (stash, rebuild, run, pop) showed
`MainHDRPass` at 14.238 on the control too — it had drifted up between sessions
on its own, probably thermal. Against the control the frame saving is 1.42 ms
and matches the pass saving.

**On this machine, cross-session baselines are not comparable for anything
under ~1 ms. Re-run the before-state back to back.** Cheap: stash the shader,
rebuild (~10 s), run 35 s.

## Verifying exposure changes

The engine logs `average luminance` and `histogram clipped luminance` once per
second. The first is exactly what `readLogAverageLuminance` computes and the
second exactly what the histogram feeds, so together they are a direct assertion
on both shaders — no readback plumbing needed. Their *ratio* is scene-independent
and so survives the scene animating between runs.

Expect the last digits to move on the reduction: a tree sums in a different order
than a serial loop, and over 14,400 terms the tree is the more accurate of the
two. Observed 0.3184 -> 0.3195 against a within-run spread of 0.30-0.37.

Both fixes are guarded by `kHistogramSharedBinCapacity` in `ExposureTypes.h`,
which static_asserts that `kHistogramBinCount` fits the shared arrays.

See [[gpu-profiler-nested-scopes]] for how the pass was measured (nested scopes
around a *compute dispatch* are valid; inside a render pass they are not).
**`MainHDRPass` is now ~14 ms of a 23 ms frame and is what is left.**


## 2026-08-14: the second full-screen scan was redundant (`738a99b`, pushed)

In Histogram mode `LuminancePass` still ran -- its guard checked "is auto
exposure on", not the mode -- scanning the whole scene image a second time for a
value nothing read. `exposure_reduce.comp` uses `averageLuminance` in Histogram
mode only as an empty-histogram fallback and for the debug readout.

Fix: derive the geometric mean from the histogram bins the reduce stage already
walks. **The bins ARE the luminance distribution.** A/B/A/B: frame 15.094/15.037
-> 14.779/14.797 ms (-1.9%), `average luminance` identical to four decimals
(0.3144), exposure unchanged.

**The wrong turn is the lesson.** I first folded the log-average reduction into
the histogram compute pass (it already fetches every texel). That worked but the
shared-memory reduction alone cost 0.17 of the 0.29 saved -- the REDUCTION is
most of that pass, not the texel fetch. Reusing an existing data structure beat
reusing an existing memory read.

Do NOT simply delete the luminance pass in Histogram mode: `average luminance`
drops to the clamp floor (0.1800), and that readout is the eyes-free
verification signal this project relies on (see [[gpu-cpu-struct-layout]]).

Still open, deliberately: in **LogAverage** mode the histogram pass (0.33 ms)
runs only to feed the "histogram clipped luminance" readout. Not symmetric --
a single average cannot reconstruct a distribution -- so removing it really would
lose the number, and LogAverage is not the default.
