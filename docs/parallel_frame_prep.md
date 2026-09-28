# Parallel Frame Preparation

CPU frame preparation — everything `Renderer::updateFrameData` does between
input handling and command recording — is task-parallel: the per-object and
per-draw-item loops chunk across a bounded set of `ve::JobSystem` workers while
the main thread claims chunks alongside them. Command recording stays single-threaded by
design: the renderer is GPU-driven (multi-draw indirect + GPU culling), so the
CPU records a handful of indirect draws per pass and recording is not the
bottleneck; the CPU cost lives in per-object data preparation.

## The primitive: `JobSystem::parallelFor`

`src/core/JobSystem.h` — splits `[0, count)` into disjoint contiguous chunks,
runs them on up to `maxHelpers` pool workers plus the calling thread, and
returns when every chunk completed. Ranges below `minChunkSize` run inline
(dispatch would cost more than the work). Every chunk runs even when one throws,
and one chunk exception is rethrown on the caller. Because chunks never overlap,
bodies may write per-index data without locking; it must not be called from a
worker thread (nested waits could starve the pool). Unit-tested in
`tests/test_job_system.cpp` (coverage, inline fallback, distribution, the helper
bound, completion while every worker is busy, empty range, exceptions).

Two properties matter, and both were measured before they were chosen:

- **Chunks are claimed, not assigned.** The caller and the woken helpers take
  chunks from a shared atomic counter, up to four chunks per participant. A
  helper that wakes late finds less (or nothing) left, so the caller never waits
  on a thread that has not started -- it only waits for chunks already running.
  The first implementation handed one fixed chunk to each worker, so every call
  paid the slowest wake-up in the pool, and on a hybrid CPU that is often an
  efficiency core.
- **Each call wakes a bounded number of helpers, once.** `framePrepParallelFor`
  passes `kFramePrepMaxHelpers = 6`; load-time callers (BRDF LUT, cubemap
  prefilter) keep the default, the whole pool. Frame prep loops take tens of
  microseconds each and several run back to back per frame, so a wake-up is not
  small next to the work.

### What the wake-up policy cost

Release, `--scene stress` (2322 draw items), i9-12900HX (24 threads, so 23
workers) with an RTX 3080 Ti Laptop, medians of ~31 one-second blocks per run.
Only CPU timings were read, and they reproduce unpinned.

Before either change, several loops were **slower in parallel than serial** --
`main culling frame data` 0.105 ms parallel against 0.059 ms with the toggle
off. Claimed chunks alone recovered about a tenth of frame prep, but command
recording, which is single-threaded and runs afterwards, got ~15% slower in every
unit at once. That is interference, not code: a 23-thread wake storm per
dispatch, several dispatches per frame. Bounding the wake-up count fixed both.
One binary, only the policy varied, interleaved over three rounds:

| policy | frame prep CPU |
| --- | --- |
| wake all 23 at once | 0.922 / 0.976 / 0.965 ms |
| wake 6 at once | **0.708 / 0.701 / 0.699 ms** |
| wake 4, each helper wakes 1 more while chunks remain | 0.807 / 0.857 / 0.792 ms |
| wake 4, each helper wakes 2 more while chunks remain | 0.980 / 0.968 / 1.358 ms |

On-demand recruitment lost to a fixed bound: every hop adds a wake-up latency,
so on a short loop the chain arrives after the work is gone or keeps waking
threads that find nothing. An earlier single-run sweep of the bound put 2-3
visibly below 5-8, which was a plateau; 6 sits inside it.

Shipped change against the previous `main` binary, A/B/A/B/A/B after a
discarded warm-up run:

| scope | before | after |
| --- | --- | --- |
| frame prep CPU | 0.902 / 0.901 / 0.920 ms | **0.616 / 0.615 / 0.616 ms** (-32%) |
| punctual shadow cache | 0.184 / 0.183 / 0.201 | 0.092 / 0.091 / 0.092 |
| world bounds | 0.136 / 0.137 / 0.138 | 0.074 / 0.074 / 0.075 |
| shadow frame data | 0.144 / 0.144 / 0.143 | 0.095 / 0.094 / 0.095 |
| object frame data upload | 0.144 / 0.145 / 0.144 | 0.107 / 0.108 / 0.108 |
| main culling frame data | 0.106 / 0.106 / 0.106 | 0.065 / 0.065 / 0.065 |
| record CPU | 0.254 / 0.266 / 0.271 | 0.267 / 0.274 / 0.277 |

`--scene default` (11 draw items, below every chunk threshold) moved from
0.057 / 0.056 / 0.061 to 0.047 / 0.047 / 0.047 ms with record CPU unchanged.
Load-time `parallelFor` work was not re-timed; it wakes the same whole pool as
before and gains only the claimed chunks. The bound is hardware-shaped: a pool
of seven workers (Apple M3) is barely affected by it, and a different CPU may
put the plateau elsewhere, so re-sweep before moving it.

## What runs in parallel

All dispatches go through `Renderer::framePrepParallelFor`, which honors the
runtime toggle (below) and falls back to the inline path when disabled:

1. **Object transform cache** (`updateFrameObjectTransforms`) — every active
   object's model matrix and world AABB are computed once per frame into
   `frameModelMatrices_` and `frameWorldBounds_`. Before the bounds half existed,
   `RenderObject::worldBounds()` (a model-matrix build + 8-corner AABB transform)
   was re-derived up to seven times per object per frame across visibility, four
   shadow cascades, and the two GPU-cull input builds.

   The matrix half came later, and for a larger reason: the punctual shadow cache
   key composed one matrix per *(atlas slot, draw item)* pair rather than per
   object, so the cost scaled with the atlas rather than the scene. The 17-scope
   CPU profile put `updatePunctualShadowCacheState` at 30% of frame prep, three
   times the next builder, which is what sent anyone looking here.

   One consumer deliberately stays off the cache: `updateVsmResidency` runs near
   the top of `drawFrame`, before `updateFrameData` rebuilds the array, so it
   composes its own matrices rather than hashing the previous frame's.
2. **Per-object frame data** (`uploadObjectFrameData`) — the heaviest loop:
   six mat4 multiplies (jittered MVP, unjittered current/previous MVP for
   motion vectors, four cascade light MVPs) plus material lookups per draw
   item, written into disjoint `ObjectFrameData` slots.
3. **CPU frustum culling** (`buildVisibleDrawItems`) — per-object AABB tests
   with per-chunk stat counters reduced into atomics. The visibility flags use
   `std::vector<uint8_t>` rather than `std::vector<bool>`: parallel chunks
   write disjoint indices, which `vector<bool>`'s packed bits would turn into
   data races.
4. **Shadow cascades** (`buildShadowFrameData`) — the per-cascade draw-item
   filter + batch build runs as one job per cascade (each cascade writes only
   its own slot); the stats reduction happens after the join. The inner
   per-object loop stays serial because `parallelFor` must not nest.
5. **GPU-cull input builds** (`updateGpuCullInputBuffer`,
   `updateGpuShadowCullInputBuffer`) — per-draw-item AABB/command fill from the
   bounds cache.

6. **Punctual shadow cache keys** (`updatePunctualShadowCacheState`) — one
   content hash per atlas slot, each walking every draw item. Each slot builds
   its key in a local `PunctualShadowCacheKey` and writes only its own entry in
   a per-slot dirty array; the dirty *list* is drained from that array in slot
   order afterwards, because it decides what the atlas pass records and a list
   ordered by whichever chunk finished first would make the recorded frame
   differ run to run.

   This is the one dispatch that does **not** take the default minimum chunk,
   and it is worth reading for why. Sizing the chunk by slot count is wrong in
   both directions, and both were measured on this loop:

   | chunking | `--scene stress` | `--scene default` |
   | --- | --- | --- |
   | default 64-slot minimum | one chunk, fully serial | correct |
   | one slot per chunk | **-22.3%** on the scope | **+23.9%** on frame prep |

   With 11 draw items a slot is nearly empty, so ~64 jobs cost more to hand out
   than to run. The chunk is therefore derived from the **total work** — the
   number of draw-item tests — rather than from the slot count, so the same
   expression reaches full parallelism on `stress` and stays serial on
   `default`. `framePrepParallelFor` has an explicit-minimum overload for this.

Kept serial: draw-item append and the mesh-batch scans (order-dependent),
`stable_sort` by mesh, buffer uploads, the skinned-mesh tail slot, and the
drain of the punctual dirty-slot list above.

## Verifying it

`GPU Profiler` panel in the debug UI:

- `Parallel frame prep (JobSystem)` checkbox — A/B toggle, applied next frame.
- `Frame prep CPU: current (avg, max)` — wall-clock of `updateFrameData`,
  measured around the whole prep block each frame.
- The worker count is shown next to the checkbox
  (`hardware_concurrency() - 1`).

Expected signal: with a few hundred+ objects (e.g., the occlusion test scene),
frame-prep time drops several-fold with the toggle on; with tiny scenes the
loops fall below `minChunkSize` and run inline, so the numbers converge — that
is the chunking working as intended, not a missing speedup.

## Threading contract

- Worker bodies only read shared frame state (camera, settings, cascade
  matrices, materials) and write disjoint per-index slots.
- All Vulkan calls stay on the main thread; `parallelFor` returns before any
  upload or command recording touches the produced data.
- `JobSystem` is also used for glTF texture decode at load time; frame prep and
  loading share the same pool.
