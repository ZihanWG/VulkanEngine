---
name: gtao-initiative
description: GTAO ambient occlusion initiative — Phase 1 (core pass) shipped on branch feature/gtao; Phases 2-3 pending
metadata: 
  node_type: memory
  type: project
  originSessionId: f737c2c3-11d0-49f9-94ec-8e8b6e5b38b6
---

Replacing the crude depth-only inline SSAO with ground-truth ambient occlusion (Jimenez et al. 2016). Chosen as the highest-ROI next step from the [[ssr-initiative]] gap analysis (reuses the SSR thin G-buffer normals). Follows the SSR subsystem as the architectural template.

**Phase 1 — core GTAO pass: DONE, committed on branch `feature/gtao` (commit 65ffc76), NOT yet merged to main, NOT pushed.**
- `src/shaders/gtao.frag`: multi-slice horizon-search GTAO, decodes the octahedral world normal from the thin G-buffer (same as ssr_trace.frag), transforms to view space, integrates the cosine-weighted arc.
- `GroundTruthAmbientOcclusion` subsystem mirrors `ScreenSpaceReflections`; gated on `swapchain_.depthSupportsSampling()`.
- `PostProcessStack` owns the R8_UNORM visibility target (`ambientOcclusion_`) next to velocity/normalRoughness; composite samples it at **binding 4** (replaced the former depth binding) and multiplies into scene color when enabled.
- Render graph: new `RenderPassType::Gtao` + `begin/endGtaoPass`, AO resource imported/transitioned, composite reads AO instead of main depth.
- `SsaoSettings` repurposed for GTAO knobs (radius/intensity/power/sliceCount/stepsPerSlice/falloff); removed old `bias`/`sampleCount`. Debug UI panel "Ambient Occlusion (GTAO)" updated. **Off by default** (user validates the visual look).
- Verified on macOS/MoltenVK: builds clean, 0 validation errors, GTAO pass executes (GPU profiler shows a "GTAO" row, ~9.7ms full-res debug on MoltenVK — Phase 3 half-res will cut ~4x). Visual look still needs user validation on RTX.

**Phase 2 — denoise: DONE, committed on `feature/gtao` (commit e53ba10), verified on MoltenVK (0 validation errors, GTAO ~9.7ms + GTAOBlur ~1.9ms rows).**
- `src/shaders/gtao_blur.frag`: depth-aware 5x5 bilateral (view-Z edge-stopping).
- GTAO subsystem now owns a raw-AO target (wrapped by the graph like SSR's scene-color copy); trace writes raw, blur denoises into the composite target. Both passes share one descriptor layout; blur reuses the params SSBO for depth linearization.
- Render graph: `RenderPassType::GtaoBlur` + `begin/endGtaoBlurPass`, raw-AO imported.
- Debug UI: "GTAO Visibility" render-target preview of the denoised term.
- Deferred: multi-bounce needs a G-buffer albedo we don't store (documented future work). `SsaoSettings`→`GtaoSettings` rename not done (cosmetic).

**Phase 3 — half-res + upsample + docs: DONE, committed on `feature/gtao` (commit 5e4419b), verified on MoltenVK (0 validation errors).**
- Trace runs at HALF resolution into a half-res raw-AO target; `gtao_blur.frag` rewritten as a joint-bilateral upsample (3x3 half-res gather, full-res depth edge-stopping) that denoises + upsamples in one full-res pass. GTAO ~9.7ms→~2.7ms, GTAOBlur ~0.6ms.
- Renderer imports the raw-AO resource at its actual (half) extent; trace viewport = half, upsample viewport = full.
- docs/gtao.md added; README (Highlights row, feature bullet, deep-dives table, limitations) + docs/README.md index + ssr.md normal-reuse note updated.

**SHIPPED: all three phases merged to main (merge commit f9cb5d7, --no-ff) and pushed to origin 2026-07-09.** Branch `feature/gtao` deleted after merge. GTAO off by default — visual look still needs RTX validation by user (toggle via "Ambient Occlusion (GTAO)" panel + "GTAO Visibility" preview).

**Temporal accumulation (optional Phase 4) — pending:** reuse velocity buffer to accumulate AO across frames; would let slice/step counts drop. Design choice still open: AO applied to whole scene color in composite (approximation); "indirect-only" needs main-pass integration.

See [[user-prefers-phased-checkpoints]] and [[no-claude-coauthor-trailer]]. Engine can't run in sandbox but does from session on user Mac ([[vulkanengine-cannot-run-in-sandbox]]).
