---
name: vulkanengine-improvement-roadmap
description: "Status of the 4-phase improvement initiative for the Vulkan engine (refactor, threading, tests, SSAO)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 23f06752-0feb-47dc-97c2-9bdca3b36d09
---

VulkanEngine is a portfolio project. In a session on 2026-06-19 we ran a phased improvement roadmap (plan file: `~/.claude/plans/vulkan-engine-wise-twilight.md`). Completed and committed on `main`:

- Phase 0: stopped tracking `build-mac/` build artifacts + `.DS_Store`; fixed `.gitignore`.
- Phase 1: `VulkanEngineCore` static lib (GPU-free) + Catch2 unit tests (`tests/`) + ctest in CI.
- Phase 2a: split ~1900 lines of ImGui debug UI out of the 11.5k-line `Renderer.cpp` into `RendererDebugUi.cpp`; shared file-local helpers moved to `RendererInternal.h` (methods stay `Renderer::` members — a TU split, not a separate class).
- Phase 3: `ve::JobSystem` thread pool (`src/core/JobSystem.*`); glTF textures now decode on workers, upload on main thread.
- Phase 4: SSAO computed in the composite fragment shader from the main depth buffer.

**Important: Phase 4 SSAO defaults OFF and was never visually verified** — the assistant cannot render the Vulkan app in-sandbox. It compiles/links and is wired correctly, but quality/Y-depth-convention need local verification (toggle in debug UI: "Ambient Occlusion (SSAO)"). See [[vulkanengine-cannot-run-in-sandbox]].

CI follow-up (same session): the Linux GitHub Actions build had never passed (bad apt package names) — fixed to install the Vulkan SDK from the LunarG apt repo (`ubuntu-24.04`), plus a volk duplicate-target fix (always FetchContent volk when fetching deps, since the LunarG SDK ships header-only `volk::volk_headers`). Windows CI was already green.

**New direction (2026-06-20): user wants this to become an interactive editor like Unity/Unreal.** Milestone A "interactive viewport" DONE + committed (`96e6cb6`): `EditorCamera` (GPU-free, unit-tested) fly/orbit/pan/dolly; `Ray`+`Aabb::intersectRay` (unit-tested) for click-to-select picking; vendored ImGuizmo (`external/imguizmo/`) translate/rotate/scale handles. Controls: RMB+WASD fly, LMB pick, Alt+LMB orbit, MMB pan, scroll speed, W/E/R gizmo op, X world/local. 22/22 tests green. **Gizmo orientation + pick accuracy need local visual verification (Vulkan Y-flip).** Next milestones (not done): B = docking panels + scene-in-viewport-panel + object create/delete/duplicate; C = ECS/scene-graph + play/edit mode; D = asset browser + drag-drop import.

Roadmap stretch goal not done: Phase 2b (extract a PostProcessPipeline subsystem). Related working-style: [[user-prefers-phased-checkpoints]].
