---
name: two-phase-occlusion-initiative
description: "Two-phase Hi-Z occlusion culling (default ON) SHIPPED: GPU-verified by user, merged to main (5c7d868), pushed 2026-07-05"
metadata: 
  node_type: memory
  type: project
  originSessionId: cda20ab6-69d8-493b-bb76-507a127de6d4
---

Started 2026-07-05 — priority #4 of the 2026-07-02 gap analysis. Implemented on branch `feature/two-phase-hiz-occlusion` (commit 9f930b2 off main 28f2a7d).

Architecture (GPU timeline): cull P1 (frustum + prev-frame pyramid via its stored VP — camera-still gate REMOVED; occluded-in-frustum items marked in a per-item phase-result buffer, binding 5) → main HDR P1 → DepthPyramidMidPass rebuild → cull P2 (resets only per-batch counts via vkCmdFillBuffer over kBatchVisibleCountBufferSize, stats persist; re-tests candidates with current VP; 5th stats counter = rescued) → MainHDRPhase2 (LOAD color/velocity/depth, replays per-batch DrawIndexedIndirectCount) → final pyramid rebuild.

Key implementation facts:
- Requires the bindless MDI indirect-count path (P2 re-fills the same compacted batch regions P1's draws consumed); otherwise per-frame fallback to old single-phase camera-still behavior. Predicate resolved once per frame into `Renderer::frameTwoPhaseOcclusionActive_` (in buildMainCullingFrameData) and threaded to graph declarations via `RenderGraphFrameResources::twoPhaseOcclusionEnabled`.
- `GpuCullFrameParams` grew a second mat4 (`occlusionViewProjectionPhase2` = current unjittered VP), 128 → 192 bytes.
- `DepthPyramid::recordCommands(..., bool midFrame)` selects DepthPyramidMidPass vs DepthPyramidPass graph slots; profiler scopes "DepthPyramidMid"/"DepthPyramid".
- Readback copy moved out of recordMainCull (copyReadback param) so combined counters copy after the final phase.
- Defaults flipped: `enableGpuOcclusionCulling = true`, new `enableTwoPhaseOcclusion = true` (serialized; round-trip tested). UI: "Two-phase occlusion (Hi-Z re-test)" checkbox + "Phase-2 rescued" counter in GPU Culling panel; "Two-phase occlusion: active/inactive" status line.
- Phase-2 draw replay is INLINE in recordRenderCommands (PushConstants lives in RendererInternal.h anonymous namespace — can't appear in Renderer.h signatures).

Status: SHIPPED. GPU-verified by user 2026-07-05 (Occlusion culled + Phase-2 rescued both nonzero while the camera moves), merged to main (5c7d868), pushed. IMPORTANT verify learning: the user's Mac/MoltenVK has NO vkCmdDrawIndexedIndirectCount ("Main indirect count path: fallback"), so the original count-path-only gating never activated there — fixed by supporting the non-compacted path (phase 2 rewrites fixed per-item slots, zeroing non-rescued commands; replay uses plain vkCmdDrawIndexedIndirect). Count path machines (user's Windows RTX) use the compacted re-fill. User GPU-verify: occlusion test scene → orbit camera (occlusion should stay ON while moving now), check "Phase-2 rescued" goes nonzero when objects disocclude, no popping/holes; A/B the two-phase checkbox. Remaining from gap analysis: (5) async compute in RenderGraph, (6) SSR or DDGI. [[vulkanengine-cannot-run-in-sandbox]], [[user-prefers-phased-checkpoints]], [[no-claude-coauthor-trailer]].
