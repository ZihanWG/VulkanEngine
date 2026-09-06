# VulkanEngine Agent Guide

## Project Intent

VulkanEngine is a C++20 / Vulkan 1.3 real-time renderer and graphics-programming portfolio. Optimize for explicit ownership, inspectable GPU contracts, measurable behavior, and small verifiable milestones. Do not expand it into a generic game engine unless the task explicitly asks for that.

Current code is the source of truth. Documentation explains intent but can lag behind implementation.

## Before Changing Anything

1. Run `git status --short` and preserve every pre-existing change. Never discard or rewrite unrelated work.
2. Trace the real call path and read the relevant subsystem document under `docs/`.
3. State the intended behavior, fallback, resource lifetime, synchronization edges, and verification plan before broad renderer edits.
4. Prefer a narrow vertical slice over speculative abstraction or unrelated cleanup.

## Source Map

- `src/core/`: application/window loop, logging, job system, and other engine-core services.
- `src/renderer/Renderer.cpp`: frame synchronization, acquisition, submission, presentation, and recreation.
- `src/renderer/RendererFrame.cpp`: CPU-side scene and frame preparation.
- `src/renderer/RendererRecord.cpp`: Vulkan command recording and explicit/manual barriers.
- `src/renderer/RendererResources.cpp`: renderer-owned resources and descriptor updates.
- `src/renderer/RendererScene.cpp` and `RendererDebugUi.cpp`: scene/material state and ImGui controls.
- `src/renderer/RendererInternal.h`: internal-linkage helpers shared by renderer translation units.
- `src/renderer/RenderGraph.*`: logical pass/resource declarations, liveness/culling, and conservative barrier inference.
- `src/renderer/<Subsystem>.*`: subsystem ownership and feature-local logic.
- `src/rhi/`: low-level RAII Vulkan wrappers and device/resource primitives.
- `src/shaders/`: GLSL; descriptor and binary layouts must stay synchronized with C++.
- `tests/`: headless Catch2 tests for GPU-independent logic.
- `docs/`: architecture, contracts, limitations, measurements, and subsystem explanations.

## Architecture Boundaries

- Keep CPU preparation, command recording, GPU execution, queue submission, and presentation conceptually separate.
- Keep GPU-free math and policy in `VulkanEngineCore`, Vulkan-facing code in `VulkanEngineRuntime`, and `VulkanEngine` as the thin executable. New `.cpp` and test files must be added to the explicit source lists in `CMakeLists.txt` or `tests/CMakeLists.txt`.
- `Renderer` is implemented across lifecycle, resource, scene, frame-prep, recording, and debug-UI translation units. Put behavior in the file matching that responsibility; use `RendererInternal.h` only for TU-local shared helpers.
- `RenderGraphBuilder` declarations describe usage; they do not record the feature's Vulkan work.
- `RenderGraph::beginDeclaredPass()` bridges declarations to conservative Synchronization2 barriers. Physical transient allocation still belongs to `Renderer`.
- The graph is not a production executor, memory-aliasing allocator, or multi-queue scheduler. Async clustered-lighting submission is renderer-owned.
- Preserve explicit barriers for intra-pass sequencing, readback/host visibility, screenshots, and resources not represented by the graph.
- Preserve RAII ownership and complete create/recreate/destroy symmetry, including partial-initialization failure paths.

## Vulkan Change Contract

For every new or changed GPU resource, identify:

- owner and lifetime: persistent, swapchain-sized, frame-indexed, transient, or history;
- creator, resize/recreation path, and destroy path;
- producer and all consumers;
- pipeline stage, access mask, image layout, and queue ownership at each edge;
- descriptor set/binding/type/count and every C++ allocation/write plus GLSL declaration;
- CPU/GLSL struct size, alignment, packing, push constants, and sentinel values;
- capability fallback, feature-off behavior, and temporal-history reset conditions.

Never add device- or queue-idle waits to the steady-state frame path. Fence protects frame-slot reuse, semaphores order queue/presentation operations, and barriers order memory accesses and layouts; do not treat them as interchangeable.

## Implementation Expectations

- Follow `.clang-format`; keep warnings clean under the project's `-Wall -Wextra -Wpedantic` policy.
- Add GPU-free math/state-machine tests when the behavior can be isolated from Vulkan.
- Treat descriptor changes as cross-file API migrations, not local shader edits.
- Keep C++ and GLSL layouts visibly paired and pin C++ offsets/sizes with `static_assert`; validation layers cannot detect a binary-layout mismatch.
- Register new shader entry points and shared GLSL includes in `CMakeLists.txt` so dependency changes rebuild SPIR-V instead of leaving stale binaries.
- A persistent runtime setting requires its field, clamp, JSON read, JSON write, ImGui control, and consuming code to move together.
- Keep debug visibility and a safe fallback for expensive or experimental renderer features.
- Update the relevant `docs/*.md` when pass order, descriptors, ownership, limitations, settings, or measured claims change.
- Do not update committed screenshots or performance numbers without an intentional capture/measurement request.
- Publish nothing about the tooling. The repository is a portfolio: tracked files, paths, commit messages, pull request titles and bodies, review comments, and CI run names must name the renderer and its author, nothing else. `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.agents/`, and `.githooks/` stay untracked through `.git/info/exclude` — never `.gitignore`, which would publish the names it hides. Do not cite `AGENTS.md` by name in tracked content; state the rule directly.
- The guard is `.githooks/hygiene.py`, one pattern table read by `pre-commit`, `commit-msg`, and `pre-push`. It needs `git config core.hooksPath .githooks`, which is per-clone and cannot be committed, so set it first in a fresh clone. Do not bypass it with `--no-verify`, which switches off pre-commit, commit-msg and pre-push in one go; a PreToolUse hook refuses that flag outright, and `HYGIENE_SKIP=1` is the considered exception because it prompts. Run `.githooks/test-hygiene.sh` after changing the table, and use the `repo-hygiene` skill for the GitHub surface the hooks cannot reach.
- Name branches `feature/`, `perf/`, `refactor/`, `docs/`, or `ci/`. A merged pull request's branch name cannot be renamed and the pull request cannot be deleted, so it is the one trace that costs a republished repository to remove.

## Verification

Use the shared entrypoint from the repository root:

- `tools/dev/verify_renderer.sh shaders`: configure if needed and compile all GLSL shaders.
- `tools/dev/verify_renderer.sh tests`: build and run the headless unit tests.
- `tools/dev/verify_renderer.sh fast`: build shaders, renderer, and tests, then run the tests. This is the normal completion gate.
- `tools/dev/verify_renderer.sh full`: run `fast`, ASan/UBSan tests, and a Release renderer build.

GPU execution, validation layers, screenshots, and profiler measurements are separate evidence. Never claim them from a headless build/test result.

Performance claims require a fixed scene/camera/settings, a warm-up period, and multi-frame statistics. Prefer medians over seconds and repeat the control configuration; single-frame timings are not evidence. On MoltenVK, timestamp scopes nested inside Dynamic Rendering may not include deferred fragment work, so measure at pass boundaries.

## Completion Report

Summarize the user-visible result, changed ownership/synchronization/descriptor contracts, exact verification commands and outcomes, and remaining GPU or performance checks. Mention any pre-existing dirty files that were deliberately left untouched.
