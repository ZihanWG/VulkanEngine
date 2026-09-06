---
name: async-compute-initiative
description: "Async compute for clustered lighting SHIPPED: GPU-verified on M3 (dedicated compute family via MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1, zero sync validation errors), merged 49838ae + pushed 2026-07-06 alongside the background-session bindless update-after-bind fix"
metadata: 
  node_type: memory
  type: project
  originSessionId: cda20ab6-69d8-493b-bb76-507a127de6d4
---

Started 2026-07-06 — priority #5 of the 2026-07-02 gap analysis. Implemented on branch `feature/async-compute-clustered` (commit 8d0e9ce off main 5c7d868).

Design (deliberately minimal — NO graphics CB split, NO render-graph multi-queue):
- ClusterBuild + LightCull are not render-graph passes (they manage their own barriers), which made them cheap to relocate.
- Async CB recorded + submitted right after CPU frame prep, BEFORE the graphics CB is recorded (GPU overlaps with CPU recording too). Graphics submit waits the per-frame binary semaphore at FRAGMENT_SHADER — first stage reading the cluster buffers via BDA; shadow passes are depth-only (no fragment shader) so they overlap.
- Queue selection in VulkanDevice: dedicated compute-only family > second graphics-family queue (priority 0.5) > unavailable (graceful inline fallback). MoltenVK default exposes NO async queue — verify with MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1.
- Cross-family: all ClusteredLighting buffers created VK_SHARING_MODE_CONCURRENT (new VulkanBufferCreateInfo::sharedQueueFamilies span) — no QFOT. recordLightCull(asyncQueue=true) skips the trailing compute→fragment barriers (FRAGMENT invalid on compute-only queue; semaphore provides the dependency). Build→cull barrier stays (same CB).
- CB/semaphore reuse guarded transitively by the frame fence (graphics waited on async → fence implies async done).
- New rhi::VulkanAsyncCompute (pool + per-frame CBs + binary semaphores). `frameAsyncComputeActive_` resolved at end of updateFrameData (clustered active && toggle && available).
- enableAsyncCompute setting (default on, serialized + round-trip tested); Lights (Clustered) panel checkbox + active/inactive/unavailable status. Known limitation: GPU profiler timestamps live on the graphics CB, so ClusterBuild/LightCull rows vanish while async is active (documented, noted in panel). docs/async_compute.md + README.

Status: headless-verified (ci-debug + ci-asan 67/67), committed, NOT merged. User verify on Mac: (1) default run → panel shows "unavailable", rendering unchanged; (2) `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1 ./build/debug/VulkanEngine` → active, crank lights to hundreds, no validation errors, lighting correct, A/B toggle. Remaining from gap analysis: (6) SSR or DDGI flagship. [[vulkanengine-cannot-run-in-sandbox]], [[user-prefers-phased-checkpoints]], [[no-claude-coauthor-trailer]].
