---
name: tier1-engine-improvements
description: "Tier-1 engine-improvement round; phases 1-2 (PipelineCache + BuiltinTextureFactory) merged to main and PUSHED 2026-07-02 (GPU smoke-tested); GTAO phase 3 not started"
metadata: 
  node_type: memory
  type: project
  originSessionId: 49172fdc-f05d-4e46-bb65-58328dbc76e0
---

Started 2026-07-01 after an engine gap-analysis vs modern industrial engines. Agreed "Tier 1" = 3 changes, being implemented on branch `feature/tier1-pipelinecache-gtao-refactor` (branched off `main` at fdf8361). Approach: adversarially-reviewed design pass (a Workflow) first, then per-phase commits, each build + `ctest --preset ci-debug` verified headless (baseline 56 tests → 62 after phase 1).

- **Phase 1 DONE + committed (69c6513): disk-backed VkPipelineCache.** New GPU-free `src/rhi/VulkanPipelineCache.{h,cpp}` (path resolve + header validation + atomic write). Cache owned by `VulkanDevice` (created end of `createLogicalDevice`, destroyed top of `cleanup` before `vkDestroyDevice`), forwarded via `VulkanContext::pipelineCache()`, threaded into all 17 pipeline-create sites via an optional `pipelineCache` field on the two pipeline create-info structs (defaults `VK_NULL_HANDLE` = cold). Loaded blob validated against the running GPU via full `VkPipelineCacheHeaderVersionOne` (vendorID/deviceID/UUID) before use. Writable path: Windows `%LOCALAPPDATA%/VulkanEngine/`, else `$XDG_CACHE_HOME` or `~/.cache/VulkanEngine/`, else temp. 6 new headless Catch2 cases (`tests/test_pipeline_cache_header.cpp`).
- **Phase 2 DONE + committed (54d8386): extracted `renderer::BuiltinTextureFactory`.** Moved 7 procedural/asset-fallback built-in texture methods out of Renderer (Renderer.cpp 5202 → 4932 LOC). Stateless leaf, borrows context + writes into caller-owned `VulkanTexture&` refs (SceneBuilder style); the 7 texture members + 2 asset-loaded flags stay on Renderer so all downstream readers are untouched. Takes the asset dir as a param (does NOT include RendererInternal.h) to avoid dragging the god-object header back in. Bodies moved verbatim.
- **Phase 3 NOT STARTED: SSAO → standalone GTAO pass.** User chose to GPU-verify phases 1-2 FIRST (2026-07-01). Full design + adversarial review saved during the session (new `AmbientOcclusion` subsystem like DepthPyramid, `ao_generate.comp` + `ao_blur.comp`, RenderGraph closed-enum additions, CompositePushConstants reshape, composite.frag rewire). User still to choose full standalone-pass version vs a lighter in-composite upgrade when they resume.

**Update 2026-07-02: phases 1-2 merged into `main` and pushed** (user ran the merged build several times during the TAA verify session — pipeline cache smoke test implicitly passed). GTAO phase 3 remains not started.

Phases 1-2 build clean + 62/62 tests pass but are NOT GPU-verified — the engine can't run in-sandbox ([[vulkanengine-cannot-run-in-sandbox]]). Working style: [[user-prefers-phased-checkpoints]], [[no-claude-coauthor-trailer]]. Note the pre-existing `-Wparentheses-equality` warning at `PostProcessStack.cpp:1407` (`if ((frameCount_ == 0u))`) is not ours.
