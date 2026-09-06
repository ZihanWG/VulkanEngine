---
name: taa-motion-vectors-initiative
description: "Motion-vector TAA (velocity buffer MRT + reprojected history) SHIPPED: GPU-verified by user, merged to main, pushed 2026-07-02"
metadata: 
  node_type: memory
  type: project
  originSessionId: cda20ab6-69d8-493b-bb76-507a127de6d4
---

Started 2026-07-02 after a gap-analysis session ranked "real TAA with motion vectors" as priority #2 (after merging the tier-1 branch, [[tier1-engine-improvements]]). Implemented in one session on branch `feature/taa-motion-vectors` (commit a030d67, branched off main after the tier-1 merge).

What shipped:
- Main HDR pass is now MRT: SceneColorHDR + full-res `R16G16_SFLOAT` VelocityBuffer (UV-space motion). `VulkanPipelineCreateInfo` gained an optional `colorFormats` span for MRT pipelines.
- `ObjectFrameData` grew `currMvpNoJitter`/`prevMvpNoJitter` (560 → 688 bytes; std430 struct mirrored in simple.vert / simple_skinned.vert / shadow.vert). Raster still uses jittered mvp so jitter never enters velocity.
- `Renderer::capturePreviousFrameMatrices()` runs AFTER queue submit (ordering matters: skybox push constants are recorded late and must still see prev matrices). Per-object prev model lives on `RenderObject::previousModelMatrix`.
- Skybox: `SkyboxPushConstants` is now exactly 128 bytes {invVP, prevVP} — the old exposure/tonemap fields were dead (never read by the shader). Sky velocity = rotation-only reprojection via w=0 direction projection.
- TAA resolve: 4-binding descriptor set (color/history/velocity/depth), closest-depth 3x3 velocity dilation (skipped when depth can't be sampled), off-screen history rejection, existing neighborhood clamp kept. New `taa.reprojectionEnabled` setting + "Motion reprojection" ImGui toggle for A/B ghosting demo.
- Known limitation (documented in docs/taa.md): skinned meshes use rigid-only velocity; previous-palette joint motion is future work.

Status: SHIPPED. User GPU-verified 2026-07-02 (clamp-off A/B confirmed reprojection works; residual ghosting with clamp off is expected — moving lights/shading + disocclusion, which clamp handles). Merged to main (7d5702c) and pushed together with the tier-1 merge. During verify we also fixed two latent render-graph debug-table bugs: ScrollX tables collapsing to a bare scrollbar (pin explicit height + ScrollY) and stretch-proportional column sizing degenerating under ScrollX (switch to SizingFixedFit + fixed widths for wrapped columns; table IDs renamed to dodge stale imgui.ini widths).

Remaining priority list from the 2026-07-02 gap analysis: (3) JobSystem takes over the frame loop, (4) two-phase Hi-Z occlusion default-on, (5) async compute in RenderGraph, (6) SSR or DDGI flagship. Working style: [[user-prefers-phased-checkpoints]], [[no-claude-coauthor-trailer]].
