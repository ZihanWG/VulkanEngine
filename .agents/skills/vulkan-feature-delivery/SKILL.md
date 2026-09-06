---
name: vulkan-feature-delivery
description: Implement end-to-end VulkanEngine rendering features. Use when adding or changing a render pass, GPU resource, descriptor contract, shader and C++ interface, render setting, or swapchain-dependent renderer subsystem that needs code, tests, docs, and validation.
---

# Vulkan Feature Delivery

Deliver the smallest complete vertical slice while preserving Vulkan ownership, synchronization, and fallback contracts.

## Establish Context

1. Resolve the repository root with `git rev-parse --show-toplevel` and work from there.
2. Read `AGENTS.md` completely and inspect `git status --short`. Preserve all pre-existing changes.
3. Trace the live path before editing. Start from the relevant orchestration files:
   - `src/renderer/Renderer.cpp` for frame lifetime, submission, presentation, and recreation.
   - `src/renderer/RendererFrame.cpp` for CPU-side frame preparation.
   - `src/renderer/RendererRecord.cpp` for command recording and manual synchronization.
   - `src/renderer/RenderGraph.cpp` and `.h` for pass/resource declarations, liveness, and inferred barriers.
4. Read the matching document under `docs/`, but treat current code as authoritative when they differ.

## Define the Contract

Before editing, write a compact implementation contract covering:

- user-visible behavior, enable/disable setting, and fallback behavior;
- resource owner, creation, resize/recreation, per-frame reuse, and destruction;
- producer and every consumer, with stage, access, layout, and queue for each edge;
- CPU/GLSL binary layout, descriptor bindings, push constants, and sentinels;
- validation plan: headless tests, shader compilation, runtime validation, and performance evidence.

For temporal features, explicitly define history validity and reset conditions such as camera cuts, resize, render-scale changes, toggles, and scene discontinuities.

## Implement a Vertical Slice

Work in dependency order:

1. Add or update GPU-free math and settings; add unit tests where practical.
2. Define shared CPU/GLSL data layouts. Keep size/alignment assertions close to C++ structs.
3. Add owned resources and complete create, resize, failure cleanup, and destroy paths.
4. Update descriptor layouts, allocation, writes, pipeline layouts, and all shader declarations as one contract.
5. Declare graph reads/writes with access matching the commands actually recorded.
6. Record commands and retain explicit barriers for intra-pass, host, readback, screenshot, or resources not owned by the graph.
7. Add debug controls, inspection data, and a safe fallback.
8. Update the subsystem doc and any frame-flow or descriptor contract that changed.

Do not move physical resource allocation into `RenderGraph` unless that architectural change is explicitly requested. Do not move async-compute queue scheduling into the graph implicitly.

## Guard Vulkan Invariants

- Never add `vkDeviceWaitIdle` or queue-idle waits to a steady-state frame path.
- Keep frame-indexed resources isolated until their fence proves reuse is safe.
- Match graph declarations to real commands; a declaration is not execution.
- Treat descriptor binding changes as cross-file API changes.
- Keep upload/readback and indirect-command barriers explicit when they occur inside one command block.
- Handle partial Vulkan initialization failures without leaks or double destruction.
- Preserve feature-off and capability fallback paths.

## Verify and Report

Run `tools/dev/verify_renderer.sh fast`. Use `full` for synchronization, lifetime, broad renderer, release-sensitive, or CI-facing changes. If only shaders changed, `shaders` is acceptable during iteration, but finish with `fast` before claiming completion.

Then run `.githooks/hygiene.py range HEAD origin/main`. A feature slice touches docs and comments, which is where a stray reference to the local tooling gets written without anyone noticing. See the `repo-hygiene` skill for what the hooks cannot reach.

Do not claim visual or validation-layer coverage unless the renderer was actually run on a GPU. Report changed contracts, commands executed, results, and any remaining GPU/performance validation separately.
