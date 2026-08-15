# Headless Render CI

The Linux and Windows CI jobs compile the renderer and run the headless unit
tests. Neither ever executes a frame. The `Headless render` workflow does: it
runs the real renderer on Mesa's lavapipe software Vulkan driver under a virtual
X server, with the validation layer enabled, and fails if validation reports
anything.

## What it runs

`.github/workflows/headless-render.yml`, on push to `main`, on pull requests,
and on manual dispatch.

```
xvfb-run -a --server-args="-screen 0 1280x720x24" \
  ./build/ci-debug/VulkanEngine \
    --exit-after-frames 10 \
    --fail-on-validation-error \
    --asset-load-stats
```

Because Xvfb provides a real X display, SDL3 creates a real surface and the job
exercises the same swapchain and presentation path a desktop run takes. No
separate offscreen code path exists, so there is nothing that can drift away
from what players run.

## Why it fails, and how

`--fail-on-validation-error` makes the engine exit with code
`Application::kValidationFailureExitCode` (2) when the validation layer reported
any error. The count comes from `rhi::ValidationTally`, which the debug messenger
in `VulkanContext.cpp` increments for every message. The check runs *after*
shutdown, so teardown-time errors count too.

The workflow distinguishes the failure modes rather than collapsing them:

| Exit status | Meaning |
| --- | --- |
| 0 | Completed the requested frames with no validation errors |
| 2 | Validation errors were reported (the log has the `[Error]` lines) |
| 124 | Hit the `timeout` backstop instead of exiting on its own |
| other | Crashed or threw before completing the frames |

A zero exit is additionally checked for a `Validation tally:` line, so a run that
silently did nothing cannot read as a pass.

**Warnings are reported but do not fail.** They move with validation-layer and
loader versions, and a job that goes red because a layer was updated teaches
people to ignore it.

**The build must be a Debug-config preset.** `CMakeLists.txt` gates
`VULKAN_ENGINE_ENABLE_VALIDATION` on `$<CONFIG:Debug>`; a Release build compiles
the layer out, and the gate would then be incapable of ever failing.

## The lavapipe ICD

The ICD filename is not stable across Mesa releases: it is
`/usr/share/vulkan/icd.d/lvp_icd.json` on Mesa 25.x and `lvp_icd.<arch>.json` on
older packages. The workflow globs for `lvp_icd*.json` and pins the result into
`VK_DRIVER_FILES` and `VK_ICD_FILENAMES`.

The pin matters. The runner image ships eight other ICDs (`intel_icd.json`,
`radeon_icd.json`, `nouveau_icd.json`, and so on). An unpinned loader would
enumerate whatever it found and the job would silently be testing a different
driver than it claims to.

## Coverage: what this job does and does not reach

Measured on Mesa 25.2.8 / llvmpipe (LLVM 20.1.2), Vulkan 1.4.318,
`PHYSICAL_DEVICE_TYPE_CPU`.

Reached, and confirmed initializing in the run log: bindless descriptor indexing
with update-after-bind, clustered (Forward+) lighting, CSM with GPU shadow
culling, the punctual shadow atlas, GTAO, SSR, volumetric fog, irradiance
probes, GPU frustum and occlusion culling, and the GPU timestamp profiler.

Two differences from the primary macOS/MoltenVK development machine are worth
stating explicitly, because they cut in opposite directions:

- **lavapipe exposes `drawIndirectCount`; MoltenVK does not.** This job runs the
  compacted `per-cascade indirect count` shadow path, which cannot execute on
  the development machine at all. That is new coverage, and it is the main
  reason this workflow earns its runtime.
- **lavapipe exposes no async compute queue.** ClusterBuild and LightCull stay
  on the graphics queue here, so the async-compute submission path is *not*
  covered by CI and remains verifiable only on the development machine.

## Limitations

- Ten frames at default settings. Swapchain recreation/resize, screenshot
  capture, and the many runtime toggles are not exercised.
- Validation errors, not rendering correctness. Nothing here checks that the
  image looks right; a golden-image comparison is separate future work and would
  need a deterministic frame clock first.
- Software rasterization is slow: roughly 4.5 s of renderer init and 5 s for the
  first frame. These are lavapipe numbers and are not performance data about the
  engine. Never quote them as such.
