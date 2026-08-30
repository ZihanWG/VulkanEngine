# GPU Profiling

Phase 1 adds a small Vulkan timestamp profiler around the existing renderer frame flow. The goal is to expose the GPU cost of major passes without changing rendering behavior, descriptor layout, swapchain ownership, or render graph order.

## Runtime UI

Open the profiler in the ImGui overlay:

1. Open `VulkanEngine Debug`.
2. Expand `Debug Views`.
3. Enable `Show GPU Profiler panel`.
4. Expand `GPU Profiler`.

The panel shows whether timestamp profiling is available, total GPU frame time, CPU frame delta, query usage, and a table of named pass timings. Each row has current, recent average, recent max, and a compact history plot. `Reset averages` clears the CPU-side history buffers.

If timestamps are unavailable, the panel remains visible and reports the unavailable reason instead of crashing or recording invalid queries.

## Implementation

`src/renderer/GpuProfiler.h` and `src/renderer/GpuProfiler.cpp` define the profiler. The renderer initializes it after Vulkan frame resources are created:

- one `VkQueryPool` per frame-in-flight slot
- 256 timestamp queries per frame by default
- two timestamp queries reserved for total frame time
- two timestamp queries per named scope
- timestamp conversion through `VkPhysicalDeviceProperties::limits.timestampPeriod`
- query support checked through the graphics queue family's `timestampValidBits`

The profiler uses `vkCmdWriteTimestamp2` because the renderer already uses Vulkan 1.3 and Synchronization2. Scope begin timestamps are written at `VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT`; scope end timestamps are written at `VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT`.

This makes each timing an inclusive elapsed GPU range around the commands recorded between the markers. Nested scopes, such as `RenderObjects` inside `MainHDRPass`, overlap with their parent and must not be summed with parent scopes. On this project's primary target they are worse than redundant — see the next section before reading any nested number.

## Scopes nested inside a render pass are meaningless on tile-based hardware

**A scope recorded between `vkCmdBeginRendering` and `vkCmdEndRendering` does not measure the work inside it.** On Apple GPUs through MoltenVK, and on tile-based deferred architectures generally, the fragment work for a render pass runs when the pass *resolves*. A timestamp written between draw calls inside a pass therefore captures only command recording and vertex work; the entire fragment cost lands at `vkCmdEndRendering`, outside every nested scope.

Measured in Release on the demo scene, medians over ~12 frames:

```
MainHDRPass       13.512 ms   <- top-level render pass, ACCURATE
  Skybox           0.002 ms
  RenderObjects    0.073 ms   <- these three sum to 0.09 ms
  SkinnedMesh      0.017 ms
                   ^ 13.42 ms belongs to no child scope
```

The 13.42 ms gap is not missing instrumentation, and `MainHDRPass`'s 13.5 ms is not inflated. Both numbers are correct; the children simply cannot see the work.

Practical rules:

- **Trust top-level passes.** `SSRTrace`, `Transparent`, `DepthPyramid`, `CompositePass`, `ClusterBuild`, and the rest each own a pass or a dispatch, and their timings are real.
- **Do not read `Skybox`, `RenderObjects`, or `SkinnedMesh` as a breakdown of `MainHDRPass`.** They read near zero regardless of how much geometry they draw.
- **To attribute cost inside a render pass, split the pass**, or use a GPU capture tool that understands tile-based scheduling. Adding more nested scopes will not help.
- Compute dispatches are not affected. A scope around a `vkCmdDispatch` outside a render pass measures that dispatch.

### Disproved hypothesis — do not re-try

The `TOP_OF_PIPE` (begin) / `BOTTOM_OF_PIPE` (end) pairing was suspected of making each scope absorb preceding in-flight work. Changing the begin marker to `BOTTOM_OF_PIPE` left the medians essentially unchanged (`MainHDRPass` 13.512 → 13.445, `Transparent` 2.452 → 2.360, `SSRTrace` 0.917 → 0.947). The pairing is not the cause; the experiment was reverted.

### Take medians, not single frames

Single-frame numbers on this hardware swing wide enough to invert a comparison. The first frame captured after the marker experiment above looked twice as bad, purely as an outlier. Sample over at least a few seconds and compare medians — the once-per-second `GPU timings:` block in the log is the easiest source.

## Scripted measurement harness

`tools/dev/measure_gpu.py` automates the protocol this document requires for a performance claim, so a pass timing does not depend on remembering the rules by hand. The renderer does parse a command line (`src/core/CommandLine.cpp` — `--scene`, `--vsm`, `--deterministic`, `--exit-after-frames`, `--capture-frame`, and others), but none of those flags reach the toggles an A/B actually varies: there is no general `--set` for a runtime settings key, so SSR, GTAO, fog, render scale and the rest are only reachable through the settings file. That is why the harness patches named keys in `config/runtime_settings.json` and restores the file afterwards. Results come back from the once-per-second `GPU timings:` blocks on stdout, which is the only machine-readable source of per-pass GPU time — the ImGui overlay shows the same numbers but only on screen.

```bash
# One configuration, absolute medians.
python3 tools/dev/measure_gpu.py run --label baseline

# Interleaved A/B/A/B; A is the unchanged persisted config.
python3 tools/dev/measure_gpu.py ab --b-set ssr.enabled=true --repeat 2

# Summarize a log captured by hand.
python3 tools/dev/measure_gpu.py parse build/measurements/fragment-stress.log
```

What the harness enforces:

- **Release only, and not a stale one.** It refuses to run a Debug binary, and aborts when any file under `src/` or `CMakeLists.txt` is newer than the linked binary. This is a hard gate, not a warning: an SSR A/B once ran against a binary 17 commits behind and reported `SSRTrace` at 0.805 ms where the rebuilt binary read 0.158 ms.
- **A settle period after `--build`.** A parallel build leaves the machine hot and the first control run would absorb all of it. The same series drifted 0.41% from a cold start and 28.5% when it began immediately after a build, so `--build` now idles 90 seconds first (`--settle`).
- **A fixed scene and camera.** Both are left at their launch defaults, which is what makes separate launches comparable.
- **A discarded warm-up.** 10 seconds by default, out of a 30-second launch, leaving roughly 20 samples.
- **Medians, with min and max reported** so a delta smaller than the run-to-run spread is visible as such.
- **Per-pass sample coverage.** Every row shows how many sampled frames actually contained that pass. A conditional pass is marked intermittent below 90% coverage, because its median is the cost of the frames that ran it rather than of the configuration. Without this, `DepthPyramid` — built in 1-2 frames out of 29 while occlusion culling is suspended — appeared as a clean `A only` row and read as a pass that one configuration had and the other did not. Coverage is judged only on the sides where the pass runs at all, so a pass genuinely absent from one configuration (`SSRTrace` with SSR off) is not mislabelled as sampling luck.
- **A stability check on one-sided rows.** A pass present in only one configuration has no delta to test its control drift against, so the drift is compared with the pass's own median and the row is marked unstable above a quarter. `SSRTrace` reported 0.158 ms this way while moving 0.137 ms between the two control runs; it reads between 0.106 and 0.458 ms across runs of an identical build, so no single number is its cost. Passes that swing like this need far more than 30 samples.
- **A repeated control, checked per pass.** `ab` runs A/B/A/B rather than AA/BB so a thermal ramp cannot land entirely on one configuration, then compares the first and last A run. Drift above 1% in `Frame total` marks the whole series unusable and exits non-zero. Separately, every row carries its own control drift and an `Attributable` verdict: a pass whose control moved at least as much as the A/B delta is reported as inside the noise floor. Frame-level stability is not enough for a sub-millisecond pass — the composite sharpen filter once read 0.416 vs 0.424 ms at frame level while the pass itself tripled.
- **No implied validation result.** Validation layers are compiled out of Release (`VULKAN_ENGINE_ENABLE_VALIDATION=0`), and the harness only runs Release, so every report says outright that it cannot show validation errors rather than letting silence read as a clean frame.
- **Typed, validated overrides.** An unknown dotted key or a value of the wrong type aborts, because a silently ignored override would measure the baseline twice and read as "no effect".
- **Restoring the settings file.** `config/runtime_settings.json` is per-user state; the harness writes it during a run and restores the original afterwards, including on failure.

Logs and a `summary.json` land in `build/measurements/`. The summary records the full effective settings, not just the overrides, because configuration A is "whatever was persisted that day" and that file lives outside git. Nested scopes are parsed but reported as unusable for attribution, for the reason in the previous section.

Scene presets are not persisted settings, so `--set` cannot reach them. The renderer's own `--scene` flag can, but the harness does not pass arguments to the binary at all yet, so those runs are still captured by hand and fed to `parse`:

```bash
./build/release/VulkanEngine --scene fragment-stress > build/measurements/fragment-stress.log
```

## Frame Latency

The frame loop already waits the fence for `currentFrame_` before reusing that frame slot. The profiler reads timestamp results for that same completed slot immediately after the fence wait and before command-buffer reset:

1. Wait the current frame slot fence.
2. Read query results from that completed slot with `VK_QUERY_RESULT_WITH_AVAILABILITY_BIT`.
3. Skip the update if results are not ready.
4. Reset and record the command buffer for the next use of the slot.
5. Mark the frame slot submitted after `vkQueueSubmit2` succeeds.

The profiler does not add `vkDeviceWaitIdle` to the runtime frame loop and does not use `VK_QUERY_RESULT_WAIT_BIT`. Swapchain recreation does not recreate profiler query pools because they depend only on the device and frame-in-flight count.

## Profiled Ranges

The current frame records timestamp scopes for:

- `CSMShadowPass`
- `ShadowGpuCullingCascade0` through `ShadowGpuCullingCascadeN` when shadow GPU culling is active
- `PunctualShadowAtlas` when spot or point shadows are active
- `MainGpuCullingPass` when main GPU culling is active
- `ClusterBuild` and `LightCull` when clustered lighting is active
- `IrradianceProbeUpdate` and `ProbeCapture` when irradiance probes are active
- `MainHDRPass`
- `Skybox`, `RenderObjects`, and `SkinnedMesh`, recorded inside `MainHDRPass`
- `DepthPyramidMid`, `MainGpuCullingPhase2`, and `MainHDRPhase2` when two-phase occlusion culling is active
- `SSRCopy` and `SSRTrace` when screen-space reflections are enabled
- `GTAO` and `GTAOBlur` when GTAO is enabled
- `VolumetricFog` when volumetric fog is enabled
- `Transparent` when the blend bucket is non-empty
- `DepthPyramid`
- `TAAResolvePass` when TAA is enabled
- `BloomExtractPass`
- `BloomBlurHorizontal`
- `BloomBlurVertical`
- `Bloom Downsample Chain`
- `Bloom Upsample Chain`
- `LuminancePass` when log-average exposure is active
- `Histogram Exposure` when histogram exposure is active
- `CompositePass`
- `ImGuiPass`

`Skybox`, `RenderObjects`, and `SkinnedMesh` are the only entries nested inside another pass rather than owning one, so on tile-based hardware they report near zero for the reason given above. Every other entry is a top-level pass or dispatch and its timing is real.

The same major ranges also use `VK_EXT_debug_utils` labels through the existing optional debug wrapper. If debug utils function pointers are unavailable, labels are no-ops.

## Limitations

- GPU timings are not CPU/GPU calibrated timestamps.
- Parent scopes include child scope work, and on tile-based hardware a scope nested inside a render pass measures almost none of its own work. See the nested-scope section above.
- Top-of-pipe and bottom-of-pipe markers are simple pass-range estimates, not detailed pipeline-stage attribution.
- The profiler has a fixed per-frame query capacity. The UI reports query usage and warns if the frame exceeds the configured capacity.
- Passes that run on the async compute queue are timestamped on that queue, so their rows are not directly comparable with graphics-queue rows on the same timeline. The debug UI notes this next to `ClusterBuild`/`LightCull`.
- Timeline lane visualization and RenderDoc capture automation are future work.
