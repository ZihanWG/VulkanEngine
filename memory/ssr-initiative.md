---
name: ssr-initiative
description: "Screen-space reflections + thin G-buffer SHIPPED: GPU-verified on M3, merged to main (c6ba7a6), pushed 2026-07-07 — gap analysis COMPLETE (all 6 items shipped)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 56036396-ea5b-4b5e-914b-c44ed9915145
---

Started 2026-07-07 — the final item (#6) of the 2026-07-02 gap analysis; user chose SSR over DDGI. Implemented on branch `feature/ssr` off main 49838ae.

Architecture:
- Thin G-buffer: main pass 3rd MRT attachment (RGBA16F: octahedral world-space shading normal RG, roughness B, metallic A; skybox writes roughness 1). Image owned by PostProcessStack next to velocity; format kNormalRoughnessFormat in RendererInternal.h. Also positions SSAO→GTAO upgrade.
- New `renderer::ScreenSpaceReflections` subsystem (Design B): owns SsrSceneColorCopy image (trace source — avoids read/write feedback loop on scene color), per-frame SsrParams SSBO (view/proj/invProj + march/weight vec4s, 224B), 4-binding descriptor sets (depth/G-buffer/copy/params), additive-blend fullscreen pipeline (new VulkanPipelineCreateInfo::enableAdditiveBlend, ONE+ONE).
- ssr_trace.frag: view-space fixed-step march (IGN-jittered start, per-64-frame counter), thickness test, binary refinement, output = color × fresnel(grayscale F0 from metallic) × (screenEdgeFade × roughnessFade × towardCameraFade) × intensity, pre-multiplied.
- Graph: RenderPassType::Ssr; SSRCopyPass (Transfer) + SSRTracePass (Graphics, LOAD sceneColor) declared between the two-phase block and the final depth pyramid, gated by RenderGraphFrameResources::ssrEnabled (= Renderer::frameSsrActive_, resolved in updateFrameData; params uploaded there with the JITTERED projection — must match rasterized depth).
- Runs BEFORE TAA so the temporal filter integrates march jitter.
- SsrSettings (RuntimeSettings.h): enabled=true default, maxSteps 48, refinementSteps 5, maxDistance 30, thickness 0.35, intensity 1.0, maxRoughness 0.6, screenEdgeFade 0.1. Serialized + clampRuntimeSettings signature grew (ssr param — test_settings_clamp harness updated), round-trip tested (68 tests total). "Screen-Space Reflections" debug panel.
- Availability gate = samplable main depth (like SSAO); unavailable → passes skipped.

Documented limitations (docs/ssr.md): additive on top of IBL specular (slight double-brightening on mirrors), grayscale metal F0 (no albedo in G-buffer), linear march (existing pyramid is MAX-depth so no Hi-Z trace; min-pyramid future work), screen-space-only misses.

Status: SHIPPED. GPU-verified by user 2026-07-07 (metal sphere correctly reflecting neighbors; during verify we fixed ImGui ID conflicts — SSR sliders needed ##ssr suffixes because CollapsingHeader does not scope IDs and the Lights panel also has an 'Intensity' slider). Merged to main (c6ba7a6), pushed. This completed ALL SIX items of the 2026-07-02 gap analysis: tier-1 infra, motion-vector TAA, JobSystem frame prep, two-phase Hi-Z occlusion, async compute, SSR. Natural next candidates (not committed to): min-depth pyramid for Hi-Z SSR acceleration, SSAO->GTAO using the new thin G-buffer, DDGI, asset pipeline (KTX2/meshoptimizer). Note: user has an untracked assets/materials/portfolio_ground.material.json from SSR testing. [[vulkanengine-cannot-run-in-sandbox]], [[user-prefers-phased-checkpoints]], [[no-claude-coauthor-trailer]].
