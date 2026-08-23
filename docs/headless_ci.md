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

## Deterministic mode

`--deterministic` makes what gets rendered a function of the frame number rather
than of how fast the machine is. It is a prerequisite for any frame-to-frame
image comparison, and it is not enabled by default.

`renderer::FrameClock` (GPU-free, unit tested) is the frame path's only source
of time. Real mode tracks the wall clock; fixed mode advances by a constant
1/60 s per frame and ignores the wall-clock reading entirely. `Renderer` reads
the steady clock once, in `updateCpuFrameTime`, and hands the value to the
clock; every downstream consumer reads the clock instead of taking its own
reading.

What `--deterministic` pins:

- **Animation.** `updateFrameData`'s elapsed seconds drives the demo light orbit,
  the skeletal animation delta, and every animated object transform.
- **Exposure adaptation.** `PostProcessStack` used to take its own
  `steady_clock` reading inside the exposure-reduce recording; it now receives
  frame-clock seconds. That removed a second, independent time source.
- **The editor camera**, via `cpuFrameDeltaMs_`.
- **Dynamic resolution**, forced off. It feeds measured GPU frame time back into
  the render extent, so it would let machine speed change the image even with a
  fixed timestep. It defaults off, but a persisted `config/runtime_settings.json`
  can have enabled it.
- **The periodic diagnostic logs.** The exposure and GPU-timing prints were
  gated on a one-second wall-clock interval, so two otherwise-identical runs
  sampled *different frame numbers* and the log looked nondeterministic when it
  was not. They are frame-clock gated now.

Audited and found already deterministic, so deliberately untouched: there is no
RNG anywhere in the engine; `JobSystem::parallelFor` partitions by index and
performs no float reduction, so worker scheduling cannot change a result; and
the TAA jitter is a frame-indexed Halton sequence.

### Evidence

Three runs of `--deterministic --exit-after-frames 90` produce byte-identical
logs once ASLR pointers and CPU/GPU timings are filtered out. Three runs without
the flag do not: they diverge on scene luminance and the exposure derived from
it, which is exactly the wall-clock-driven animation showing through. The
control is what makes the result meaningful.

## Frame capture

`--capture-frame N --capture-output PATH` captures the swapchain image of frame
N (1-based, matching the frame clock) to exactly one PNG.

```
./build/debug/VulkanEngine --deterministic --capture-frame 60 --capture-output /tmp/frame60.png
```

`--capture-include-ui` takes the copy *after* the ImGui pass instead of before
it. Off by default, because a regression capture wants the rendered frame and not
a debug panel over a third of it — the default path is byte-for-byte what it was.

It exists because the debug panel is otherwise invisible to every scripted run:
it is drawn after the point the capture is taken, so anything it reports and the
log does not — an amber warning, a colour, where a line sits relative to the
numbers it contradicts — had no evidence path at all. That is what left the frame
capacity warning's UI half unverified for months while its log half was A/B
tested; see [engine_upgrade_audit.md](engine_upgrade_audit.md).

Moving the copy is not free of consequence, and validation said so on the first
run: after the overlay the swapchain image is already past the graph's present
transition, so the copy has to put it back in `PRESENT_SRC_KHR` rather than the
colour-attachment layout the pre-overlay copy restores. `recordCopy` takes the
restore layout as a parameter for exactly that reason.

This is deliberately *not* the portfolio screenshot path (F12). That one
switches to the showcase scene preset and writes a timestamped file plus the
tracked `screenshots/..._latest.png` alias. A regression capture wants the scene
as configured, one named file, and no timestamp — a timestamp would defeat
comparing runs, and writing the tracked alias from CI would clobber a committed
screenshot.

The readback lags the recorded frame by the in-flight frame count, so the loop
keeps drawing past frame N until the capture lands rather than exiting at N.
That extension is bounded: `--capture-frame` may not exceed
`--exit-after-frames` (rejected at parse time), and the run gives up after a
16-frame grace window. A capture that was requested but never written exits 3,
checked before the validation tally — a run that did not produce the image it was
asked for has not passed, whatever validation thought of it.

### Pixel determinism

Three runs of `--deterministic --capture-frame 60` produce **byte-identical
PNGs**. Three runs without `--deterministic` produce three different images.

That is the claim Phase 2's log evidence could not make: not just that the frame
*inputs* reproduce, but that the rendered *image* does. It is what makes a
golden-image comparison possible.

Measured on macOS/MoltenVK. Goldens are driver-specific and must be captured on
the driver that will be compared against — a lavapipe run will not match an M3
capture, and should not be expected to.

**This result is driver-specific in a second way, which is easy to over-read.**
Byte-identical repeats hold on MoltenVK. They do not hold on lavapipe, where the
same commit rendered three times produced 0, then 433, then 0 differing pixels,
every one off by a single unit. Determinism here means the engine's inputs are a
function of the frame number; it does not mean every driver rasterizes those
inputs bit-identically every time. See [Why a tolerance of one](#why-a-tolerance-of-one).

## Golden-image regression

The job renders 30 deterministic frames, captures frame 30, and compares it
pixel-for-pixel against `tests/golden/lavapipe_frame30.png`. A mismatch fails
the job and uploads both the new capture and a difference image.

Comparison policy lives in `renderer::ImageCompare` (GPU-free, unit tested);
`tools/compare_images` decodes the PNGs and applies it. The tool links only
`VulkanEngineCore`, so it runs anywhere a downloaded artifact lands:

```
./build/debug/compare_images capture.png tests/golden/lavapipe_frame30.png --diff-output diff.png
```

Exit codes: 0 match, 1 mismatch, 2 usage or IO error.

### Why a tolerance of one

**This corrects an earlier claim.** This page previously said zero tolerance was
measured, on the evidence of two independent runs producing byte-identical
captures. More runs falsified that.

Commit `1e90f23` was rendered three times by this job with no change in between:
0 pixels differing, then 433 differing, then 0 again. Every differing pixel was
off by exactly 1. The CPU-side logs of those runs are byte-identical, and
llvmpipe reports the same version and the same vector width in each, so the
divergence is in rasterization rather than in anything the engine decided.

So the gate was flaky, and a flaky gate is worse than a loose one: it is the
failure mode this page already warns about for warnings, where red stops meaning
anything and people learn to re-run until green.

Tolerating a delta of 1 absorbs exactly that noise and gives up very little. A
pixel now has to be off by 2 or more to count as differing, and the
differing-pixel budget stays at zero, so anything clearing the tolerance still
fails. For calibration, the half-resolution SSR copy — the last real rendering
change to pass through this gate — moved pixels by up to 14, and would still be
caught.

What this does not cover is a real regression that only ever moves a pixel by a
single unit. Nothing seen so far looks like that: rendering changes here have
moved pixels by double digits, and aliasing corruption moves them further still.

### Re-baselining

When the change is intended, or when the runner's Mesa version moved:

1. Take `artifacts/lavapipe_frame30.png` from the failed run's
   `headless-render-logs` artifact.
2. Replace `tests/golden/lavapipe_frame30.png` with it.
3. Say in the commit message which of the two it was.

The failure output prints the driver version to make that distinction possible.

The golden has been re-baselined once, for the skinned shadow caster: the mesh
had thrown no shadow at all, and giving it one moved 811 of 921600 pixels at a
maximum channel delta of 8. Mesa was 25.2.8 on both sides, so it was the change
and not the runner. The difference image is the whole argument: the changed
pixels are the row of perforations the new shadow falls across and the caster's
own silhouette, and nothing else in the frame moved.

The committed golden is re-encoded with real deflate compression (305 KB rather
than the 3.6 MB the engine's stored-block PNG writer emits). Only the encoding
differs; the comparison decodes both sides, so pixels are what is compared.

**Goldens are driver-specific.** This one was captured on lavapipe and is only
meaningful against lavapipe. A capture from the macOS/MoltenVK development
machine will not match it and is not expected to.

Note for repo hygiene: `tests/golden/` is the second category of tracked binary
in this repository, after `screenshots/`. A capture run writes only where
`--capture-output` points, so it cannot clobber either.

## Limitations

- Ten frames at default settings. Swapchain recreation/resize, screenshot
  capture, and the many runtime toggles are not exercised.
- Validation errors, not rendering correctness. Nothing here checks that the
  image looks right; a golden-image comparison is separate future work and would
  need a deterministic frame clock first.
- Software rasterization is slow: roughly 4.5 s of renderer init and 5 s for the
  first frame. These are lavapipe numbers and are not performance data about the
  engine. Never quote them as such.
