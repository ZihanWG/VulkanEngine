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

   Across frames, `objectTransformCache_` keeps each object's matrix and world
   AABB together with the transform and local bounds they came from
   (`renderer::refreshCachedObjectTransform`, GPU-free and unit-tested), so a
   static object copies last frame's bits instead of recomposing them. Composing
   was ~75 ns per object on one thread (49 ns for the TRS matrix, 26 ns for the
   eight-corner bounds). The key is the inputs' **values compared bit for bit**,
   not a dirty flag. Every writer of a transform is therefore covered without
   knowing the cache exists, a slot reused by a different object simply misses,
   and a hit is exactly the bits a fresh compose would give. That exactness is
   why -0.0 and 0.0 count as different keys.
2. **Per-object frame data** (`uploadObjectFrameData`) — one mat4 multiply
   (the unjittered previous-frame MVP for motion vectors; everything shared by
   the frame lives in `FrameConstants`) plus material lookups per draw item.
   Each chunk writes its 192-byte records **straight into the mapped
   per-frame buffer**, once and in full (see "Writing GPU records in place"
   below).
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
   bounds cache. Both go through one writer, `writeGpuCullInput`, which differs
   per caller only in the batch list, command slots per draw item, and debug
   caster isolation. Batch membership is resolved serially first, so the
   parallel fill writes each 64-byte record exactly once, in place.

### Writing GPU records in place

The object-data and cull-input buffers are host-visible memory the device reads
directly (`VMA_MEMORY_USAGE_AUTO` with sequential host writes), which on a
discrete GPU is write-combined memory across PCIe. They used to be built in a
zeroed `std::vector`, then copied in with `VulkanBuffer::upload`. Timed inside
the function on `--scene stress` (2322 draw items), the copy cost more than the
work:

| per frame | allocate + zero | parallel fill | single-threaded copy |
| --- | --- | --- | --- |
| object frame data (446 KB) | 0.014 ms | 0.025 ms | **0.068 ms** |
| main cull input (149 KB) | 0.004 ms | 0.017 ms (+0.006 batch pass) | **0.023 ms** |

The copy ran at ~6.5 GB/s, about what one core gets writing combined stores
over PCIe. The chunks now write through `VulkanBuffer::mapRange` and the
function ends with `flush` + `unmap`. Each record is assembled in a local
and stored whole: a partial or repeated write into write-combined memory is
the slow case. A record with nothing to describe is written zeroed, as the
vector's value-initialised slot was. Buffers, lifetimes, the frame-slot fence
that protects them, and the bytes the GPU reads are all unchanged. Captures of
`--scene default`, `--scene stress`, and the default scene with
`csm.debugOnlyShadowCasterObject` set are byte-identical before and after (the
isolation case differs from a plain capture in 11% of pixels, so it does
exercise the path).

Same hardware and protocol as above, against the bounded-wake-up build:

| scope | before | after |
| --- | --- | --- |
| frame prep CPU | 0.614 / 0.611 / 0.617 ms | **0.541 / 0.547 / 0.541 ms** (-12%) |
| object frame data upload | 0.108 / 0.108 / 0.109 | 0.075 / 0.075 / 0.076 |
| shadow frame data | 0.094 / 0.094 / 0.095 | 0.075 / 0.075 / 0.075 |
| main culling frame data | 0.064 / 0.064 / 0.065 | 0.046 / 0.046 / 0.046 |
| record CPU | 0.272 / 0.265 / 0.272 | 0.282 / 0.281 / 0.282 |

Record CPU rose by ~0.012 ms in every pair, against 0.07 ms saved in prep. The
cause is not established; one candidate is PCIe writes still draining while the
driver writes command memory.

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

Kept serial: draw-item append and the mesh-batch scans (order-dependent), the
draw-item sort, buffer uploads, the skinned-mesh tail slot, and the drain of
the punctual dirty-slot list above.

The sort is `renderer::sortDrawItemsForBatching`: a stable counting sort over
the distinct (bucket, doubleSided, mesh) keys. A frame has a handful of those
against thousands of items. It produces exactly `std::stable_sort`'s order,
which the tests check on random inputs, and it falls back to `std::stable_sort`
past 64 distinct keys. The blend bucket's back-to-front sort computes each
item's camera distance once instead of twice per comparison.

### Static objects and the sort

Release, `--scene stress`, same hardware and protocol as the tables above,
against the in-place-records build:

| scope | before | after |
| --- | --- | --- |
| frame prep CPU | 0.543 / 0.541 / 0.544 ms | **0.461 / 0.466 / 0.461 ms** (-15%) |
| world bounds | 0.075 / 0.074 / 0.075 | 0.031 / 0.032 / 0.032 |
| draw items | 0.088 / 0.087 / 0.088 | 0.049 / 0.050 / 0.049 |
| record CPU | 0.262 / 0.276 / 0.278 | 0.266 / 0.275 / 0.277 |

On one thread, the stable sort had been 0.044 ms and the blend sort 0.014 ms.
`--scene default`, whose objects animate and so miss the cache, stayed at
0.045-0.047 ms. Captures of the default, stress, orbiting-camera stress and
fragment-stress scenes are byte-identical before and after.

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
