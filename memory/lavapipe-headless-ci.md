---
name: lavapipe-headless-ci
description: "Lavapipe headless CI — MERGED TO MAIN (38c474d); CI renders 30 deterministic frames and gates on validation errors + a golden image; covers indirect-count (MoltenVK can't) but not async compute"
metadata: 
  node_type: memory
  type: project
  originSessionId: 901cbd1c-8080-4c6f-9d21-4f11902d4869
  modified: 2026-08-15T09:43:40.844Z
---

Chosen 2026-08-15 over the asset-pipeline work (see [[asset-pipeline-initiative]],
blocked on content). **COMPLETE AND MERGED TO MAIN as `38c474d` (--no-ff), pushed;
all branches deleted. Linux/Windows/Headless all green on main.** Temp branch
trigger removed. Don't re-offer any of this as future work.

**A Windows-only trap this initiative hit, worth remembering:** a test that
references anything pulling `Application.cpp` or `Window.cpp` gives
`VulkanEngineTests.exe` a startup import on SDL3.dll, which is not beside it in
the build tree. Catch2's `catch_discover_tests` runs the binary at BUILD time, so
it surfaces as a build error `Exit code 0xc0000135` (STATUS_DLL_NOT_FOUND), never
as a test failure, and macOS never reproduces it (dylib rpaths resolve). Fix by
lifting the logic into VulkanEngineCore — NOT by copying DLLs next to the test
target, which would make the whole class of mistake silent. `otool -L` is NOT a
valid check here (macOS lists a dylib merely for being on the link line); use
`nm -u | grep SDL`.

**PHASE 1 SHIPPED AND CI-VERIFIED GREEN** (`e90b8de` + temp-trigger commit, branch
`ci/headless-render`, unmerged; run 31868857377 passed in 4m02s).
`.github/workflows/headless-render.yml` runs 10 frames and fails on validation
errors. `rhi::ValidationTally` counts messenger messages; `--fail-on-validation-error`
exits 2. **The gate was verified by injecting VUID-VkBufferCreateInfo-size-00912
and confirming exit 2, then reverting** — do that again if the gate is ever
reworked; a green CI proves nothing about a gate that cannot fail.
Warnings deliberately do NOT fail (they move with layer versions).
**The workflow must use a Debug-config preset**: `VULKAN_ENGINE_ENABLE_VALIDATION`
is gated on `$<CONFIG:Debug>` in CMakeLists (NOT on NDEBUG — easy to get wrong).
TODO on merge: drop `ci/headless-render` from the workflow's push branches.
Docs: `docs/headless_ci.md`.

Spike history: run 31867892538, workflow `lavapipe-spike.yml`, now deleted.
Stale remote branch `spike/lavapipe` is superseded and can be deleted.

Environment that works: `ubuntu-24.04`, `mesa-vulkan-drivers` 25.2.8 +
`xvfb`, `xvfb-run -a --server-args="-screen 0 1280x720x24"`, ci-debug preset
(leaves NDEBUG undefined so the engine turns validation on itself).
**The ICD is `/usr/share/vulkan/icd.d/lvp_icd.json` — no arch suffix on Mesa 25.x.**
Glob for `lvp_icd*.json` and pin `VK_DRIVER_FILES`; the runner ships 8 other ICDs.

Result: full init + 10 frames + clean shutdown, **zero VUIDs, zero errors, zero
warnings**. Device is llvmpipe, `PHYSICAL_DEVICE_TYPE_CPU`, Vulkan 1.4.318.
Init 4.5 s, first frame 5.1 s, whole job 3m55s.

**Coverage delta vs the user's Mac — this is the payoff:**
- lavapipe **has `drawIndirectCount`**, so CI runs `shadow path: per-cascade
  indirect count`, the compacted path MoltenVK never takes. Real new coverage.
- lavapipe has **no async compute queue**, so ClusterBuild/LightCull stay on the
  graphics queue in CI. That path stays Mac-only — document it, don't pretend.
- Bindless descriptor indexing w/ update-after-bind, clustered lighting, GTAO,
  SSR, volumetric fog, irradiance probes, punctual atlas, GPU culling: all init.

**PHASE 2 SHIPPED** (`92feaac`, CI green run 31873576681). `renderer::FrameClock`
(Core, GPU-free, 8 tests) is now the frame path's ONLY time source;
`--deterministic` = fixed 1/60 s step + dynamic resolution pinned off.
Evidence: 3 runs at `--exit-after-frames 90` are byte-identical (filtering ASLR
pointers + CPU/GPU timings); 3 runs without the flag diverge on luminance/exposure.
**That proves frame INPUTS reproduce, not pixels — pixel proof needs Phase 3 capture.**
Surprise worth remembering: the periodic exposure/GPU-timing debug prints were
gated on a wall-clock second, so two identical runs sampled DIFFERENT FRAME
NUMBERS and the log looked nondeterministic when it wasn't. Both are frame-clock
gated now. If a future "nondeterminism" hunt starts here, check sampling cadence
before suspecting the renderer.

**PHASE 3 SHIPPED** (`c89a064`, CI green run 31874092645).
`--capture-frame N --capture-output PATH` writes ONE named PNG (no timestamp, no
`_latest` alias — that alias is a *tracked* file CI must never clobber). Separate
from the F12 portfolio path, which switches scene presets.
**PIXEL DETERMINISM PROVEN on MoltenVK**: 3 runs of `--deterministic
--capture-frame 60` give byte-identical PNGs; 3 without the flag give 3 different
images. Exit 3 = capture requested but never written (verified by temporarily
zeroing the grace window). Bug found while testing and fixed: an unbounded
capture override made `--capture-frame 999999 --exit-after-frames 5` run ~1M
frames; captureFrame > exitAfterFrames is now rejected at parse time.
**Goldens are driver-specific — a lavapipe capture will NOT match an M3 one.**

**PHASE 4 SHIPPED — INITIATIVE COMPLETE** (`060e248`, CI green run 31875293362).
Golden-image regression: 30 deterministic frames, capture frame 30, compare
against `tests/golden/lavapipe_frame30.png`, fail on ANY difference, upload the
capture + a diff image. `renderer::ImageCompare` (Core, GPU-free, 10 tests) holds
the policy; `tools/compare_images` (links Core ONLY, no Vulkan) applies it.
`PngWriter.cpp` was moved Runtime→Core to make that possible.
~~Zero tolerance is MEASURED~~ — **FALSIFIED, see the flakiness entry below.**
Two runs agreed; a third did not. **Gate verified red** by perturbing the golden by ONE
unit on 1600 px — caught it (max channel delta 1), then reverted (`b33fa1b`).
Golden is re-encoded with real deflate: 3.6 MB → 305 KB (the engine's PNG writer
uses stored/uncompressed deflate blocks). `tests/golden/` is now the 2nd tracked
binary category after `screenshots/` — see [[repo-hygiene-conventions]].
Expected future breakage: a runner Mesa/LLVM bump re-renders the scene; the
failure output prints driverInfo so that's distinguishable from a real
regression, and re-baselining is documented.

**THE GOLDEN GATE WAS FLAKY — corrected 2026-08-16 (`884d32c`).** Same commit
rendered three times gave 0, then 433, then 0 differing pixels, EVERY ONE off by
exactly 1, with byte-identical CPU-side logs and identical llvmpipe version and
vector width. So lavapipe does not rasterize bit-identically run to run. The gate
now uses `--channel-tolerance 1` (differing-pixel budget still 0): a pixel must
be off by 2+ to count, and every real rendering change here has moved pixels by
double digits (the half-res SSR copy moved them by 14). **My earlier "zero
tolerance is measured" claim was wrong — two data points were not enough.**
Blind spot now documented: a regression that only ever moves a pixel by 1.
**Operational trap: `gh run rerun --failed` REPLACES the run's artifacts** —
download the failing capture and diff image BEFORE re-running, or the evidence
is gone.

**Determinism audit (done, for the golden-image phase).** No RNG anywhere;
`parallelFor` partitions by index so there is no float-reduction order hazard;
TAA jitter is frame-indexed Halton and already deterministic. Only three
wall-clock reads need a fixed-step virtual clock:
- `RendererFrame.cpp` `updateFrameData` elapsedSeconds → demo lights, skeletal
  delta, `updateAnimatedTransforms` (the big one)
- `PostProcessStack.cpp` exposure adaptation deltaTime
- `Renderer.cpp` `updateCpuFrameTime` → editor camera
`DynamicResolutionSettings::enabled` defaults false, so it is not a hazard
unless turned on.

**How to apply:** a validation-error gate is worth shipping BEFORE golden images
— validation is clean today, so the gate locks in a real invariant immediately
and costs one workflow. Goldens must be captured ON lavapipe (never cross-vendor
pixel-match against the M3).
