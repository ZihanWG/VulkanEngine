---
name: gpu-profiler-nested-scopes
description: "GPU profiler scopes nested inside a render pass read ~0 on this TBDR and are meaningless; top-level passes are accurate. Plus: the exposure histogram is the frame's 2nd-biggest cost for a fixable reason."
metadata: 
  node_type: memory
  type: project
  originSessionId: 30ea2028-e930-4922-9e8f-d02ba071e94d
  modified: 2026-08-06T06:41:53.957Z
---

Investigated 2026-08-05 after the irradiance-probe work made the timings look
wrong. Findings are measured, and one plausible hypothesis was **disproved** —
recorded so it is not re-tried.

## Scopes nested inside a render pass are meaningless here

Measured on Release, medians over ~12 frames, demo scene:

```
MainHDRPass       13.512 ms   <- top-level render pass, ACCURATE
  Skybox           0.002 ms
  RenderObjects    0.073 ms   <- these three total 0.09 ms
  SkinnedMesh      0.017 ms
                   ^ 13.42 ms belongs to no child scope
```

Top-level passes are fine: SSRTrace 0.92, Transparent 2.45, DepthPyramid 0.68,
CompositePass 0.20, ClusterBuild 0.014.

**Cause: tile-based deferred rendering.** On Apple GPUs via MoltenVK the fragment
work for a render pass executes when the pass *resolves*, so a timestamp written
between draw calls inside the pass measures only command-recording/vertex work.
All the fragment cost lands at `vkCmdEndRendering`, outside every nested scope.

`docs/profiling.md` currently says only "Parent scopes include child scope work",
which invites reading children as a valid breakdown. They are not — they read ~0
regardless of what is inside them. **MainHDRPass's 13.5 ms is real**, not an
artefact.

## Disproved hypothesis — do not re-try

Suspected the `TOP_OF_PIPE` (begin) / `BOTTOM_OF_PIPE` (end) pairing in
`GpuProfiler::beginScope`/`endScope` made each scope absorb preceding in-flight
work. Changed begin to `BOTTOM_OF_PIPE` and measured: **medians essentially
unchanged** (MainHDRPass 13.512 -> 13.445, Transparent 2.452 -> 2.360, SSRTrace
0.917 -> 0.947). The pairing is not the problem. Experiment reverted.

Also a process note: the first single-frame block after that change looked
*twice as bad*, which was an outlier. Single blocks are useless here — take
medians, as this project's own rule says.

## Found on the way: the exposure histogram is the 2nd-biggest frame cost

`Histogram Exposure` is a stable **6.18 ms** (min 5.72, max 6.38, Release) out of
roughly 28 ms of measured passes.

`src/shaders/luminance_histogram.comp` is 34 lines with **no shared memory**: one
invocation per pixel at full resolution, each doing a global `atomicAdd` into one
of 256 bins. That is ~2M global atomics contending on 256 addresses — the slowest
possible histogram.

Standard fix: per-workgroup `shared uint localBins[256]`, then one global
`atomicAdd` per bin per group. Cuts global atomics from ~2M to
`workgroupCount * 256`, typically 10-50x. Should be worth ~15-20% of the frame.

## Both follow-up tasks DONE, MERGED AND PUSHED 2026-08-06

`a6267c6` shared-memory histogram, `4fdfece` docs, merged `--no-ff` as `122fe7b`
and pushed; main == origin/main; branch deleted. Re-verified on main after the
merge: Release + Debug build, 150/150, 0 VUIDs on a real M3 run, and the timing
reproduced (1.963 ms, frame 23.976).

**Measured, Release, demo scene, medians over 28 samples (n=28 both sides):**

```
Histogram Exposure   6.072 [5.495-6.410]  ->  1.958 [1.899-2.186]   -68%
Frame total         28.050 [27.319-28.436] -> 23.870 [23.470-24.843] -15%
```

The two deltas agree to 0.07 ms and the ranges do not overlap. Output unchanged:
histogram-derived clipped luminance median 0.3599 -> 0.3601, a gap 30x smaller
than the frame-to-frame sd of 0.021. (That log line — `histogram clipped
luminance`, printed once/sec alongside `average luminance` — is the correctness
check for any future exposure change; average luminance comes from the separate
unmodified LuminancePass, so their *ratio* is scene-independent.)

Two things that made this measurable, worth reusing:
- `Renderer::tryPrintGpuTimings` dumps every scope to the log once per second, so
  a 30s background run yields ~28 samples. There is no env var or CLI flag; just
  run the binary and parse. Median script kept at scratchpad `stats.py`.
- The shader clamps the pushed bin count to a shared-array capacity constant;
  `kHistogramSharedBinCapacity` in `ExposureTypes.h` static_asserts against it so
  raising `kHistogramBinCount` can't silently drop the top bins.

The classic bug avoided: the old shader early-`return`ed for out-of-range pixels.
With barriers that must become an `if`, or the workgroup deadlocks on non-uniform
control flow.

**`MainHDRPass` is now 13.6 ms of a 23.9 ms frame (57%)** and is the obvious next
target — but per the nested-scope finding above, no amount of extra instrumentation
inside it will attribute that cost. It has to be split into real passes first.

See [[instrument-before-guessing-runtime-bugs]] and
[[vulkanengine-cannot-run-in-sandbox]].
