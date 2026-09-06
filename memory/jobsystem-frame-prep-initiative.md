---
name: jobsystem-frame-prep-initiative
description: "JobSystem parallelFor + task-parallel frame prep SHIPPED: GPU A/B verified by user, merged to main (28f2a7d), pushed 2026-07-02"
metadata: 
  node_type: memory
  type: project
  originSessionId: cda20ab6-69d8-493b-bb76-507a127de6d4
---

Started 2026-07-02 right after shipping [[taa-motion-vectors-initiative]] — priority #3 of the gap analysis ("JobSystem takes over the frame loop"). Implemented on branch `feature/jobsystem-frame-prep` (commit d34614c off main 7d5702c).

What shipped:
- `JobSystem::parallelFor(count, minChunkSize, body)` — disjoint chunks across pool + calling thread, inline below minChunkSize, first chunk exception rethrown, NOT re-entrant from workers (documented). 5 new Catch2 cases in tests/test_job_system.cpp (62 → 67 total tests).
- Per-frame world-bounds cache `Renderer::frameWorldBounds_` — worldBounds() was re-derived up to 7×/object/frame (visibility + 4 cascades + 2 GPU-cull input builds); now computed once, in parallel.
- Parallelized via `Renderer::framePrepParallelFor` (honors toggle, min chunk 64): ObjectFrameData fill (heaviest: 6 mat4 muls/item), CPU frustum culling (stats via per-chunk locals → atomics; vector<bool> → vector<uint8_t> to avoid packed-bit races), one job per CSM cascade (inner loop serial — no nesting), both GPU-cull input fills.
- Deliberately serial: draw-item append + mesh-batch scans (order-dependent), stable_sort, uploads, skinned tail. Command recording stays single-threaded BY DESIGN — engine is GPU-driven (MDI), recording is not the CPU cost; this is the interview talking point.
- Debug UI (GPU Profiler panel): "Parallel frame prep (JobSystem)" A/B checkbox + worker count + "Frame prep CPU: cur (avg, max)" from a new DebugHistory measured around updateFrameData in drawFrame.
- docs/parallel_frame_prep.md + docs/README.md index + root README.

Status: SHIPPED. User GPU-verified the A/B toggle 2026-07-02, merged to main (28f2a7d) and pushed. Remaining from the 2026-07-02 gap analysis: (4) two-phase Hi-Z occlusion default-on, (5) async compute in RenderGraph, (6) SSR or DDGI flagship.
