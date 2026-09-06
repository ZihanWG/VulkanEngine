---
name: renderer-regression
description: Validate VulkanEngine renderer changes with a risk-based build, shader, test, runtime, visual, and performance matrix. Use before declaring renderer work complete, when investigating a regression, or when preparing CI and review evidence.
---

# Renderer Regression

Collect reproducible evidence without confusing headless success with GPU correctness.

## Establish the Change Surface

1. Resolve the repository root with `git rev-parse --show-toplevel` and work from there.
2. Read `AGENTS.md` completely, then record `git status --short` and the relevant diff.
3. Separate pre-existing changes from the changes under validation.
4. Classify risk: CPU-only, shader, descriptor/layout, resource lifetime/recreation, synchronization/queue, render output, or performance.

## Choose the Matrix

- Documentation-only: inspect links and commands; no build claim is required.
- CPU-only logic: `tools/dev/verify_renderer.sh tests`.
- Shader-only iteration: `tools/dev/verify_renderer.sh shaders`, then `fast` before completion.
- Normal renderer work: `tools/dev/verify_renderer.sh fast`.
- Descriptor, lifetime, synchronization, queue, release-sensitive, broad refactor, or CI work: `tools/dev/verify_renderer.sh full`.

If a test is unavailable for changed GPU behavior, add a headless test for any extractable math, packing, indexing, state transition, or policy logic.

Whatever the matrix, run `.githooks/hygiene.py range HEAD origin/main` before the work leaves the machine. A build gate says the change works; that one says it is publishable. The commit hooks cover the same ground, but only for commits made with them installed, and a fresh clone has none.

## GPU Runtime Matrix

Run only when the environment has the required Vulkan SDK, GPU, and display. Report skipped cells honestly.

1. Launch a Debug build with validation enabled and exercise both feature-on and fallback paths.
2. Exercise resize/minimize/recreate, setting toggles, camera cuts, and render-scale changes when relevant.
3. Use a fixed scene, camera, settings file, resolution, warm-up period, and capture point for A/B comparisons.
4. Check for validation messages, NaN/Inf artifacts, stale temporal history, flicker, layout-dependent corruption, and frame-overlap bugs.
5. For performance claims, capture per-pass GPU timestamps over a stable window and report median or another declared statistic with hardware and settings.

Do not update committed portfolio screenshots unless the user explicitly requests a new golden capture.

## Regression Diagnosis

Minimize one dimension at a time: setting, pass, resource path, shader branch, async queue, and fallback. Compare the last known-good revision only if doing so will not overwrite dirty work. Prefer read-only `git show` or a separate worktree.

## Report Evidence

Provide a compact matrix containing check, command/environment, result, and coverage. Separate:

- compilation and shader interface evidence;
- headless unit/sanitizer evidence;
- validation-layer and runtime evidence;
- visual comparison evidence;
- GPU performance evidence.

State residual risk for every skipped category and include exact failures rather than summarizing them away.
