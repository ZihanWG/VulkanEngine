---
name: clustered-lighting-initiative
description: Interview-readiness initiative adding clustered (Forward+) lighting as the flagship feature
metadata: 
  node_type: memory
  type: project
  originSessionId: bb7471bd-8ada-486d-90bb-a62898959740
---

Initiative (started 2026-06-25): make VulkanEngine interview-ready for a **game-engine programmer** role, judged mostly on demo + talking points, ~2-week budget. Flagship feature chosen by the user = **clustered (Forward+) many-light rendering** (over skeletal animation / SSR / GPU particles, which are named as future work).

Work is on branch **feat/clustered-lighting** (off main). Plan file: `~/.claude/plans/parallel-mixing-pearl.md`. Phased commits, GPU-verified locally by the user (sandbox can't run GPU — see [[vulkanengine-cannot-run-in-sandbox]]); checkpoints per [[user-prefers-phased-checkpoints]].

Status:
- **Phase 1 done** (commit): `ClusteredLighting` subsystem + per-frame BDA light buffer; point/spot lights; brute-force multi-light shading in `simple_bindless.frag`.
- **Phase 2 done** (commit): `cluster_build.comp` + `light_cull.comp`, 16×9×24 froxel grid, per-cluster light index list, fragment reads grid via push-constant addresses; ClusterBuild/LightCull GPU profiler scopes.
- **Phase 3 done** (commit): `ClusterGrid.h` (GPU-free froxel math, single source of truth for grid dims), `tests/test_cluster_grid.cpp` (5 Catch2 cases incl. froxel round-trip; 41 tests total), cluster heatmap debug view, "Lights (Clustered)" ImGui panel, animated demo light swarm (count slider 0-512).
- **Phase 4 done** (commit): README rewritten for hiring managers (CI badges, pitch, Mermaid frame-flow diagram, highlights matrix, Clustered Lighting section, updated resume copy), `docs/design_decisions.md` talking-points doc. Demo GIF still needs a local GPU capture to `screenshots/clustered_lighting_demo.gif`.
- **Emissive maps done** (commit): glTF emissiveFactor + emissiveTexture; emissive maps reuse the sRGB base-color bindless array; `ObjectFrameData` gained a trailing `emissiveFactor` vec4 (560-byte layout, both vertex shaders kept in sync); added pre-tonemap so it blooms; hero ceramic glows by default; HDR-editable in the Material Inspector.
- Also added `docs/clustered_lighting.md` (per-system deep-dive matching gpu_culling.md/render_graph.md).

**Shipped** 2026-06-27: merged to main via `--no-ff` merge commit (0683c29) grouping the 6 feature commits, **pushed to origin/main**, and the feat/clustered-lighting branch **deleted** locally + on origin. main builds clean + 41/41 tests pass. Initiative complete.

Demo GIF captured + wired into README hero (committed/pushed), portfolio still refreshed to clustered look (downscaled to 1.6 MB). User confirmed the GPU path works (they ran F12).

Emissive follow-ups DONE (pushed to main 66abdea): (1) emissive persists in material-asset JSON — MaterialAsset.emissiveFactor serialized; AssetManager moved into VulkanEngineCore (it's GPU-free) with 2 new Catch2 round-trip tests (43 tests total); (2) legacy simple.frag applies the emissive factor. Established session pattern: small polish commits go straight to main + push (feature branch is gone).

Remaining only if the user wants a new initiative: the next flagship — skeletal animation or SSR (both named as future work in the README). That's a multi-day effort needing the user to pick which.

Verification tools for the user: in-app "Lights (Clustered)" panel — toggle "Cluster heatmap" (froxel light-count viz = proof clustering works), toggle "Use clustered culling" for brute-force A/B, crank count to 512 for GPU-profiler timings.

clang-format note: local clang-format is **22.1.8** and reformats the repo's already-clean code (version mismatch). Do NOT run it on existing files — hand-format to match; the repo CI does not enforce clang-format.
