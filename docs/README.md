# VulkanEngine Documentation

Start here for focused technical notes. The root `README.md` is the portfolio
overview; these documents describe implementation details and limitations.

- [Build Guide](build.md): Required tools, Vulkan SDK setup, CMake
  presets, shader compilation, CI behavior, and common build failures.
- [Engine Upgrade Audit](engine_upgrade_audit.md): Current repository shape,
  implemented renderer scope, frame flow, and accurate Phase 7 limitations.
- [GPU Profiling](profiling.md): Timestamp query profiler design, frame-latency
  readback model, profiled ranges, and known timing limitations.
- [Render Graph 2.0](render_graph.md): Logical resource handles, pass
  declarations, conservative image transitions, selected buffer barriers,
  liveness metadata, and graph UI.
- [GPU Culling](gpu_culling.md): GPU frustum culling, optional conservative
  previous-frame Hi-Z occlusion, occlusion test scene, counters, and fallbacks.
- [Post-Processing](post_processing.md): HDR scene color, mip-chain bloom, GPU
  exposure state, tone mapping, and active HDR source routing.
- [Temporal AA Foundation](taa.md): Optional TAA pass, Halton jitter, HDR
  history resolve, invalidation, debug signals, and limitations.
- [Editable Scene Workflow](scene_editing.md): Runtime object editing, JSON
  scene save/load, missing saved-scene behavior, and serialization scope.
- [Asset Manager And Material Assets](asset_system.md): Material JSON schema,
  path-based handles, runtime mapping, scene metadata, and non-goals.

## Portfolio Summary Text

C++20 Vulkan 1.3 real-time renderer featuring PBR/IBL, cascaded shadows, HDR
post-processing, mip-chain bloom, GPU exposure, Render Graph metadata, GPU
timestamp profiling, GPU culling, optional Hi-Z occlusion, TAA foundation,
editable scene tools, JSON material assets, and overlay-free portfolio capture.
