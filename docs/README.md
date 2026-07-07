# VulkanEngine Documentation

Start here for focused technical notes. The root `README.md` is the portfolio
overview; these documents describe implementation details and limitations.

- [Design Decisions](design_decisions.md): The trade-offs behind the major
  subsystems (clustered lighting, GPU-driven visibility, BDA, render graph,
  Sync2, testing) — written as interview-ready talking points.
- [Build Guide](build.md): Required tools, Vulkan SDK setup, CMake
  presets, shader compilation, CI behavior, and common build failures.
- [macOS / MoltenVK Build](build_macos.md): LunarG SDK + MoltenVK setup, the
  app-bundle target, and the double-clickable launcher.
- [Engine Upgrade Audit](engine_upgrade_audit.md): Current repository shape,
  implemented renderer scope, frame flow, and accurate Phase 7 limitations.
- [Frame Flow And Descriptor Contract](frame_flow.md): The full per-frame pass
  ordering and the descriptor-set layout contract.
- [Parallel Frame Preparation](parallel_frame_prep.md): JobSystem
  parallelFor, the per-frame world-bounds cache, which frame-prep loops run on
  workers, and the A/B toggle + CPU timing readout.
- [GPU Profiling](profiling.md): Timestamp query profiler design, frame-latency
  readback model, profiled ranges, and known timing limitations.
- [Render Graph 2.0](render_graph.md): Logical resource handles, pass
  declarations, conservative image transitions, selected buffer barriers,
  liveness metadata, and graph UI.
- [GPU Culling](gpu_culling.md): GPU frustum culling, optional conservative
  previous-frame Hi-Z occlusion, occlusion test scene, counters, and fallbacks.
- [Async Compute](async_compute.md): The async compute queue selection,
  overlapping ClusterBuild/LightCull with the shadow passes, the semaphore
  model, and concurrent-sharing buffers.
- [Clustered Lighting](clustered_lighting.md): Forward+ froxel grid, GPU light
  culling, per-froxel shading, buffer layout, barriers, heatmap, and testing.
- [Skeletal Animation](skeletal_animation.md): GPU-free animation core (pose
  sampling + hierarchy flatten), GPU linear-blend skinning, the procedural demo,
  and the unit-tested bind-pose invariant.
- [Screen-Space Reflections](ssr.md): The thin G-buffer, the copy + trace
  passes, march/refinement parameters, and current limitations.
- [Post-Processing](post_processing.md): HDR scene color, mip-chain bloom, GPU
  exposure state, tone mapping, and active HDR source routing.
- [Temporal AA Foundation](taa.md): Optional TAA pass, Halton jitter, HDR
  history resolve, invalidation, debug signals, and limitations.
- [Editable Scene Workflow](scene_editing.md): Runtime object editing, JSON
  scene save/load, missing saved-scene behavior, and serialization scope.
- [Asset Manager And Material Assets](asset_system.md): Material JSON schema,
  path-based handles, runtime mapping, scene metadata, and non-goals.
- [Portfolio Capture](portfolio_capture.md): Portfolio capture mode, the studio
  setup, and overlay-free F12 screenshot export.
- [Milestone History](milestones.md): Incremental build history and design
  decisions behind the current renderer (moved out of the root README).

## Portfolio Summary Text

C++20 Vulkan 1.3 real-time renderer featuring clustered (Forward+) lighting,
PBR/IBL, cascaded shadows, HDR post-processing, mip-chain bloom, GPU exposure,
Render Graph metadata, GPU timestamp profiling, GPU culling, optional Hi-Z
occlusion, TAA foundation, editable scene tools, JSON material assets, and
overlay-free portfolio capture.
