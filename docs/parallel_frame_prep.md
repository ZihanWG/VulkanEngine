# Parallel Frame Preparation

CPU frame preparation — everything `Renderer::updateFrameData` does between
input handling and command recording — is task-parallel: the per-object and
per-draw-item loops chunk across the `ve::JobSystem` worker pool while the main
thread executes one chunk itself. Command recording stays single-threaded by
design: the renderer is GPU-driven (multi-draw indirect + GPU culling), so the
CPU records a handful of indirect draws per pass and recording is not the
bottleneck; the CPU cost lives in per-object data preparation.

## The primitive: `JobSystem::parallelFor`

`src/core/JobSystem.h` — splits `[0, count)` into disjoint chunks, runs them on
the pool workers plus the calling thread, and returns when every chunk
completed. Ranges below `minChunkSize` run inline (dispatch would cost more
than the work). The first chunk exception is rethrown on the caller. Because
chunks never overlap, bodies may write per-index data without locking; it must
not be called from a worker thread (nested waits could starve the pool).
Unit-tested in `tests/test_job_system.cpp` (coverage, inline fallback,
distribution, empty range, exception propagation).

## What runs in parallel

All dispatches go through `Renderer::framePrepParallelFor`, which honors the
runtime toggle (below) and falls back to the inline path when disabled:

1. **World-bounds cache** (`updateFrameWorldBounds`) — every active object's
   world AABB is computed once per frame into `frameWorldBounds_`. Before this
   existed, `RenderObject::worldBounds()` (a model-matrix build + 8-corner AABB
   transform) was re-derived up to seven times per object per frame across
   visibility, four shadow cascades, and the two GPU-cull input builds.
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

Kept serial: draw-item append and the mesh-batch scans (order-dependent),
`stable_sort` by mesh, buffer uploads, and the skinned-mesh tail slot.

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
