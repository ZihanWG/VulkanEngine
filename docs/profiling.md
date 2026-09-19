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

This makes each timing an inclusive elapsed GPU range around the commands recorded between the markers. Nested scopes, such as `RenderObjects` inside `MainHDRPass`, overlap with their parent and must not be summed with parent scopes. Whether they mean anything at all depends on the GPU — see the next section before reading any nested number.

## Scopes nested inside a render pass: meaningless on a tiler, real on an immediate-mode GPU

**This is a property of the GPU, not of the profiler**, and this document asserted the tiler answer unconditionally for as long as the development machine was an Apple M3.

**On a tile-based deferred architecture, a scope recorded between `vkCmdBeginRendering` and `vkCmdEndRendering` does not measure the work inside it.** The fragment work for a render pass runs when the pass *resolves*, so a timestamp written between draw calls captures only command recording and vertex work; the entire fragment cost lands at `vkCmdEndRendering`, outside every nested scope.

Measured in Release on an Apple M3 through MoltenVK, demo scene, medians over ~12 frames:

```
MainHDRPass       13.512 ms   <- top-level render pass, ACCURATE
  Skybox           0.002 ms
  RenderObjects    0.073 ms   <- these three sum to 0.09 ms, 0.7%
  SkinnedMesh      0.017 ms
                   ^ 13.42 ms belongs to no child scope
```

The 13.42 ms gap is not missing instrumentation, and `MainHDRPass`'s 13.5 ms is not inflated. Both numbers are correct; the children simply cannot see the work.

**On an immediate-mode GPU the same scopes resolve as recorded.** Measured on an RTX 3080 Ti Laptop, Release, `--scene stress`, medians over ~150 one-second blocks in each of three runs:

```
             run 1    run 2    run 3
MainHDRPass  0.425    0.418    0.433 ms
  Skybox     0.030    0.032    0.033
  RenderObjects
             0.382    0.370    0.383
  SkinnedMesh
             0.001    0.001    0.001   <- no skinned mesh in this scene
children     0.413    0.403    0.417   =  97% / 96% / 96% of the parent
```

The remaining 3-4% is the pass's own begin and end overhead. The two populations are 0.7% and 96%, which is not a difference of degree.

Practical rules:

- **Trust top-level passes everywhere.** `SSRTrace`, `Transparent`, `DepthPyramid`, `CompositePass`, `ClusterBuild`, and the rest each own a pass or a dispatch, and their timings are real on either architecture.
- **Check before reading a nested scope as a breakdown.** `tools/dev/measure_gpu.py` now decides this per report rather than assuming it: `nested_scope_share` sums `Skybox` + `RenderObjects` + `SkinnedMesh` against `MainHDRPass`, and a report annotates the rows with the share it measured or suppresses them outright below `NESTED_SCOPE_RESOLVED_FRACTION`. You do not have to remember which machine you are on; the report says.
- **The decision is made at the parent, never per child.** A child can legitimately read zero because its work is absent from the scene -- `SkinnedMesh` on `--scene stress`, which has no skinned mesh -- and that is not the same thing as a child the hardware cannot see.
- **On a tiler, to attribute cost inside a render pass, split the pass**, or use a capture tool that understands tile-based scheduling. Adding more nested scopes will not help *there*.
- Compute dispatches are not affected on either architecture. A scope around a `vkCmdDispatch` outside a render pass measures that dispatch.

### Disproved hypothesis — do not re-try

The `TOP_OF_PIPE` (begin) / `BOTTOM_OF_PIPE` (end) pairing was suspected of making each scope absorb preceding in-flight work. Changing the begin marker to `BOTTOM_OF_PIPE` left the medians essentially unchanged (`MainHDRPass` 13.512 → 13.445, `Transparent` 2.452 → 2.360, `SSRTrace` 0.917 → 0.947). The pairing is not the cause; the experiment was reverted.

### p10 needs more samples than the default duration gives

`QUOTED_PERCENTILE` is p10 because the median got a delta's sign wrong here (see [Load knobs](#load-knobs-and-why-they-did-not-fix-the-drift-gate)). That choice has a cost the default 30-second duration does not pay for.

At the default duration a run yields about 18 samples, so p10 is the second-smallest of them — which makes it a function of how many anomalously fast blocks a run happened to catch. Measured while re-taking the render-scale table, two runs of the *same* configuration with the clock pinned and no throttle active:

```
1.00a  6.76 6.81 7.09 5.95 6.76 7.13 7.17 6.79 6.78 7.39 6.86 6.82 6.77 6.76 6.86 6.09 6.81 6.72
1.00b  6.78 6.71 7.11 6.82 6.83 7.29 7.60 6.87 6.96 6.83 6.22 6.86 6.90 6.83 6.82 6.76 6.84 6.68
```

The bodies are the same distribution — medians 6.801 and 6.829, 0.4% apart. But `1.00a` caught two low outliers and `1.00b` one, so p10 read 6.087 against 6.679 and the control-drift gate refused the series at **9.7%**. `MainHDRPass` in the same two runs drifted 0.16%, because its distribution has no such tail.

It was not a bimodality worth finding: the low blocks are not frames missing a pass. `DepthPyramid` and `DepthPyramidMid` are absent from *every* block of both runs, the occlusion-yield controller having suspended the pyramid for the whole run on a scene that culls nothing.

Raising the duration to 75 seconds — 63 samples — moved p10 onto the body and the same comparison returned **0.81%**, inside the limit, with the per-scale numbers unchanged. So:

- **A refused frame-level control on a sub-1% real effect is worth diagnosing before believing.** Print the raw blocks; if the medians agree and only p10 disagrees, the gate tripped on the statistic.
- **Do not switch statistics to rescue a series.** Raise the sample count instead. Re-reading the same data with the number that gives the answer you want is how a measurement protocol stops meaning anything.
- Budget roughly 60+ samples when the frame total is the gate. The pass-level control is far more robust at the default and usually passes when the frame level does not.

### Which machine a number came from

Most of the measured numbers in `docs/` predate this project's move from an Apple M3 through MoltenVK to an RTX 3080 Ti Laptop, and a lot of them do not say so. Two of this repository's conclusions have already failed to survive that move — back-face culling, which was measured and rejected on the tiler and is now on by default at −37.4% `MainHDRPass`, and the nested-scope rule above — so "which machine" is not bookkeeping.

The boundary, from history rather than memory: the last commit that labels a measurement as Apple M3 is `e23a63f` (2026-08-11); the first that labels one as RTX 3080 Ti is `9624e05` (2026-09-07). Numbers introduced between those dates carry no label and cannot be assigned by date.

They can usually be assigned by magnitude, because the two machines are an order of magnitude apart on the same scenes:

| | Apple M3 / MoltenVK | RTX 3080 Ti Laptop |
| --- | --- | --- |
| default scene, GPU frame | ~15–16 ms | 1.754 ms |
| `MainHDRPass` | ~10 ms | 0.4–0.9 ms |
| `--scene stress`, GPU frame | — | 1.024 ms |
| `--scene sponza`, GPU frame | — | 5.363 ms at a 1200 MHz pin, 8.254 at 800 |

A frame total in the teens is the M3. A `MainHDRPass` under a millisecond is the RTX. Where neither the label nor the magnitude settles it, the table says the hardware is not recorded rather than guessing — an unattributed number is less misleading than a confidently wrong attribution.

**The rule going forward:** every quoted timing carries hardware, scene, resolution and statistic. Three of those are usually implicit and each has bitten:

- **Hardware**, for the reason above.
- **Scene**, because `--scene stress` is CPU-bound here while `default` and `fragment-stress` are GPU-bound, so the same change reads differently on each.
- **Resolution**, because the frame is fragment-bound and the harness defaults to 1280x720. `--scene gpu-stress` exists precisely because at that resolution every other preset gives a 1–2 ms frame the drift gate cannot resolve.
- **Statistic**, because `QUOTED_PERCENTILE` is p10 and everything older is a median. On this hardware the median got a delta's *sign* wrong where p10 did not, so the two are not interchangeable and a number that does not say which it is cannot be compared with one that does.

### The scene is now stamped on the run, not inferred from it

Of the four stamps below, the scene was the one a log could not be asked for.
`measure_gpu.py` passes `--scene` through to both sides of an A/B and then had no
way to confirm it arrived, and its caveat section said only that the scene was
"at launch defaults" — which stopped being true once a preset could be named on
the command line.

The renderer prints `Scene: <preset> at <WxH>` after the preset is built, so it
reports what is on screen rather than what was asked for. The harness reads it
into the report header and into `summary.json`, and **refuses to summarise an
`ab` series whose runs disagree**. Nothing else in the report could catch that:
the drift gate compares the control against itself, so two configurations
rendering two different scenes can both be perfectly stable and still be
uncomparable. A report that says `Scene: unreported` came from a binary older
than the stamp and should not be quoted.

### `--scene sponza`: the only preset that is content rather than construction

Every other preset is procedural cubes and spheres laid out to provoke a
particular bottleneck. Sponza is 103 primitives and 25 materials of real
authored content, and it is the scene to reach for when the question is about
content: depth complexity, material variety, LOD selection, or object-level
culling on geometry that was not arranged to make the answer come out a
particular way.

It needs `-DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON` and a cook (see
`docs/asset_load_baseline.md`). Naming it on a build without the asset exits
non-zero rather than falling back, so a series cannot quietly measure something
else. It is not in CI: it has no usable pixel gate, and the sweep's admission
rule is that a leg must change the shape of the frame graph rather than the
volume of content.

**Measured, RTX 3080 Ti Laptop, 1280x720, clocks pinned 1200/7001, p10 over 63
samples:** `Frame total` 5.363 ms, `MainHDRPass` 4.270 ms, `RenderObjects` 4.195
ms (99% of its parent, which is the immediate-mode behaviour described above).
That is roughly three times the default scene and five times `--scene stress`,
so the frame is comfortably large enough to measure against — which was the open
question when the scene was added.

#### It needs a much lower clock pin than any other scene: 800 MHz

The gate is reachable here, but only well below the ceiling every other preset
uses. Four `ab --repeat 2 --duration 75 --args --scene sponza --deterministic`
series, identical but for the pin, each started from a 63 °C card:

| graphics pin (memory 7001) | `Frame total` control drift |
| --- | --- |
| 1200 MHz | 5.1% |
| 1000 MHz | 1.7% |
| 900 MHz | 1.1% |
| **800 MHz** | **0.16%** ✅ |

This is the `--scene stress` rule taken further than it had been before: a pin is
a ceiling, heavier load wants a lower one, and Sponza at 5–8 ms is the heaviest
scene in the repository. 1200 was chosen before that was known, and 1100 — the
value that works for `gpu-stress` — was never going to be enough.

**Use `-Mhz 800 -MemMhz 7001` for this scene.** Absolutes move a lot with the
pin (the same frame reads 5.363 ms at 1200 and 8.254 ms at 800), so they are only
comparable within one pin; percentages and deltas are what carry across.

Three other explanations were tested first and all three are wrong, so they do
not need re-testing. The clocks held at their pin throughout with no throttle
reason ever active, so this was never the clock wander the pin exists to remove.
`--deterministic` was tried on the theory that the orbiting demo lights and the
exposure feedback made the content vary across a 75-second window: it tripled the
sample count and improved every per-pass drift by an order of magnitude, and made
frame-total drift **worse**. And it is not a ramp *within* a run — at the 900 pin
each run is flat across its own thirds (7.447 / 7.427 / 7.344) and the whole
drift is a step *between* runs, which is why reducing the heat the series
generates is what fixed it rather than sampling longer.

#### What `MainHDRPass` is made of here: more than half is shadow filtering

`MainHDRPass` is 5.6 ms of a 7.1 ms frame on this scene at the 800 pin, and the
only decomposition of that pass this repository had was taken on an Apple M3
against
the default scene, where it read as roughly two thirds clustered punctual light
loop. Neither the machine nor the scene carries over -- the nested-scope rule and
back-face culling are both conclusions that already failed to survive that move
-- so the pass was ablated again here. Each row removes one contributor through a
runtime setting and measures what the pass loses.

**RTX 3080 Ti Laptop, `--scene sponza` at 1280x720, clocks pinned 800/7001, p10
over ~300 (A) and ~460 (B) samples,
`ab --repeat 2 --duration 75 --args --scene sponza --deterministic`:**

| Removed | `MainHDRPass` | Delta | Share of the pass | Control drift |
| --- | --- | --- | --- | --- |
| nothing (control) | 5.60 ms | -- | -- | -- |
| punctual shadow sampling (`punctualShadows.enabled=false`) | 3.437 ms | **-2.163 ms** | **38.6%** | 0.36% |
| cascade shadow sampling (`csm.shadowDistance=1.0`) | 4.652 ms | **-0.935 ms** | **16.7%** | 0.77% |

Each ablation removes the lookup and nothing else. `punctualShadows.enabled`
false leaves every light in the cluster list and only marks its atlas slot
invalid (`updatePunctualShadowSlots`), so the lights still shade and what goes
away is the atlas fetch inside the loop. A one-metre `shadowDistance` puts every
fragment past the last split, where `selectShadowCascade()` returns -1 and
`sampleShadowFactor()` returns lit without touching `uShadowMapCompare`.

Together, **55% of `MainHDRPass` is shadow filtering.** The remaining ~2.5 ms is
material fetches, the BRDF, the light loop's own arithmetic, and IBL.

**The punctual atlas costs about six times more to read than to write.**
`PunctualShadowAtlasPass` -- every caster drawn into every tile -- is 0.380 ms.
Sampling what it wrote costs 2.163 ms in the main pass. `CSMShadowPass` reads
0.000 ms on this scene because the cascade cache is fully hit under a static
camera, so the sun's shadow cost is *entirely* on the sampling side as well.

That reorders the shadow work worth doing. Drawing fewer casters -- GPU caster
culling, tighter cascade fits, per-tile invalidation -- is aimed at the 0.4 ms
that is already the small half. The 3.1 ms is filter cost: tap count, filter
width, and whether a fragment needs a filtered lookup at all.

Three things this does not say. It does not predict what a cheaper filter would
save -- removing the lookups also unshadows the image, so each row bounds its
contributor rather than costing a replacement for it. It is not a correctness
result: validation layers are compiled out of Release. And in the punctual series
`CompositePass` and `DepthPrepass` also moved, by 0.032 and 0.013 ms, in a
direction the change cannot cause; `CompositePass` read 0.052 ms on that series'
control against 0.020 ms on the other two, so that row is unstable rather than
affected. Both are treated as artifacts and are not reported as effects.

**Quartering the pixels does not quarter the pass.** A third series varied
`renderScale.scale` rather than a feature. It needed a 700 MHz pin: two attempts
at 800 were voided by control drift of 1.1% and 4.3%, because the half-scale side
drives the card at a different rate and the control could not return between
them. Dropping the pin is the remedy the sweep above found, one step further.

**RTX 3080 Ti Laptop, `--scene sponza` at 1280x720, clocks pinned 700/7001, p10
over 267 and 485 samples, control drift 0.44%:** `MainHDRPass` goes 6.354 ->
3.102 ms at a quarter of the pixels. That is **-51.2%, not the -75% a purely
per-pixel pass would give.** Absolutes here are not comparable with the 800-pin
rows above; the percentages are.

Taking those two points as a line, **68% of the pass scales with pixel count and
32% does not** -- about 2.0 ms at this pin. Two points cannot establish that the
relationship is linear, so read that as the split a linear model gives rather
than as a measured constant.

The fixed third is what resolution cannot buy back, and it is the part nothing
here has attributed yet: vertex work for the LOD-selected geometry, per-draw
submission across 103 draw items, and descriptor and state changes. The shadow
filtering above is per-pixel work and sits inside the other two thirds, which is
consistent -- 55% of the pass is less than 68% of it.

### `--overdraw`: how many times the average pixel is shaded

`VK_QUERY_TYPE_PIPELINE_STATISTICS` counting
`FRAGMENT_SHADER_INVOCATIONS`, bracketed around the opaque scene geometry and
printed once a second as its own log line:

```
Overdraw: 2.187 fragment shader invocations per rendered pixel (2015242 invocations over 921600 pixels, opaque scene geometry only)
```

Off unless asked for, and diagnostic only — nothing in the frame path reads it,
and it is verified not to change the frame (0 of 921600 pixels differ with it on
and off). `pipelineStatisticsQuery` is optional in Vulkan; a device without it
loses the line and says so once in the capability report.

**What it counts, and why that is the useful number.** Invocations are what
survives early depth testing, so this is the shading that actually happens rather
than the geometric layer count — which is exactly the work a depth prepass could
remove. The two differ far more than expected here: `gpu-stress` is built from
twenty-four full-frame slabs and shades **1.276** fragments per pixel, because
stacked full-screen quads occlude each other perfectly and early-Z rejects nearly
all of it. `--scene sponza` shades **2.187**, more than any scene in the
repository including the ones built to stress fragment shading. See
`design_decisions.md` on the depth prepass for the full table and what it bounds.

**The denominator is the whole render extent, not the covered area**, so a scene
that does not fill the frame reads below 1.0 (`cornell` is 0.647) and that says
"sky in frame", not "negative overdraw". It still supports a rigorous bound in
the other direction: covered pixels cannot exceed the extent, so
`invocations − extent` is a floor on what a prepass would remove.

Bracketed around the opaque geometry only, deliberately. The skybox writes one
fragment per uncovered pixel and the post-process chain writes several per pixel
regardless of the scene; counting them would add a constant that has nothing to
do with content.

### `DepthPrepass`

Present unless `renderer.enableDepthPrepass` is turned off, and it is **on** by
default.
It replays the opaque and masked buckets depth-only ahead of `MainHDRPass` so
early-Z rejects fragments that pass would otherwise shade and overwrite. On
`--scene sponza` it costs **0.071 ms** and takes **1.221 ms** off `MainHDRPass`;
see `design_decisions.md` for the full A/B and for why a 0.071 ms pass was
predicted to cost 2.9.

It is a runtime setting rather than a build flag precisely so this harness can
A/B it inside one binary. **Set B to `false`**: A is the unchanged persisted
configuration, which now has the prepass on, so asking for `=true` on the B side
compares a configuration against itself and reports a null result as though it
were a measurement.

```bash
python3 tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --b-set renderer.enableDepthPrepass=false --args --scene sponza --deterministic
```

The delta then reads with the sign flipped from the table in
`design_decisions.md`, which quotes off-to-on: B is the slower side here.

### Take medians, not single frames

Single-frame numbers on this hardware swing wide enough to invert a comparison. The first frame captured after the marker experiment above looked twice as bad, purely as an outlier. Sample over at least a few seconds and compare medians — the once-per-second `GPU timings:` block in the log is the easiest source.

## Scripted measurement harness

`tools/dev/measure_gpu.py` automates the protocol this document requires for a performance claim, so a pass timing does not depend on remembering the rules by hand. The renderer does parse a command line (`src/core/CommandLine.cpp` — `--scene`, `--vsm`, `--deterministic`, `--exit-after-frames`, `--capture-frame`, and others), but none of those flags reach the toggles an A/B actually varies: there is no general `--set` for a runtime settings key, so SSR, GTAO, fog, render scale and the rest are only reachable through the settings file. That is why the harness writes a per-label settings file and points the renderer at it with `--settings`. The renderer's own flags are still available through `--args`, which is how a scene preset reaches a measurement. Results come back from the once-per-second `GPU timings:` blocks on stdout, which is the only machine-readable source of per-pass GPU time — the ImGui overlay shows the same numbers but only on screen.

```bash
# One configuration, absolute medians.
python3 tools/dev/measure_gpu.py run --label baseline

# Interleaved A/B/A/B; A is the unchanged persisted config.
python3 tools/dev/measure_gpu.py ab --b-set ssr.enabled=true --repeat 2

# Summarize a log captured by hand.
python3 tools/dev/measure_gpu.py parse build/measurements/fragment-stress.log
```

What the harness enforces:

- **Release only, and not a stale one.** It runs `build/release/VulkanEngine` and nothing else -- it checks that the path exists rather than inspecting the binary's configuration, so pointing that path at a Debug build would defeat it. Staleness is checked per artifact, not with one timestamp: each `.cpp`/`.h` against the binary, each shader against its own `.spv`, a shared `.glsl` against the oldest `.spv`, and `CMakeLists.txt` against `build.ninja`. This is a hard gate, not a warning: an SSR A/B once ran against a binary 17 commits behind and reported `SSRTrace` at 0.805 ms where the rebuilt binary read 0.158 ms.
- **A settle period after `--build`.** A parallel build leaves the machine hot and the first control run would absorb all of it. The same series drifted 0.41% from a cold start and 28.5% when it began immediately after a build, so `--build` now idles 90 seconds first (`--settle`).
- **A fixed scene and camera.** Both are left at their launch defaults, which is what makes separate launches comparable.
- **A discarded warm-up.** 10 seconds by default, out of a 30-second launch, leaving roughly 20 samples.
- **A low percentile, with min and max reported** so a delta smaller than the run-to-run spread is visible as such. `QUOTED_PERCENTILE` is p10, not the median: on this hardware the median got a delta's *sign* wrong where p10 did not (see below). Numbers quoted elsewhere in this repository as medians predate that change and are not comparable with p10 ones.
- **Per-pass sample coverage.** Every row shows how many sampled frames actually contained that pass. A conditional pass is marked intermittent below 90% coverage, because its median is the cost of the frames that ran it rather than of the configuration. Without this, `DepthPyramid` — built in 1-2 frames out of 29 while occlusion culling is suspended — appeared as a clean `A only` row and read as a pass that one configuration had and the other did not. Coverage is judged only on the sides where the pass runs at all, so a pass genuinely absent from one configuration (`SSRTrace` with SSR off) is not mislabelled as sampling luck.
- **A stability check on one-sided rows.** A pass present in only one configuration has no delta to test its control drift against, so the drift is compared with the pass's own median and the row is marked unstable above a quarter. `SSRTrace` reported 0.158 ms this way while moving 0.137 ms between the two control runs; it reads between 0.106 and 0.458 ms across runs of an identical build, so no single number is its cost. Passes that swing like this need far more than 30 samples.
- **A repeated control, checked per pass.** `ab` runs A/B/A/B rather than AA/BB so a thermal ramp cannot land entirely on one configuration, then compares the first and last A run. Drift above 1% in `Frame total` marks the whole series unusable and exits non-zero. Separately, every row carries its own control drift and an `Attributable` verdict: a pass whose control moved at least as much as the A/B delta is reported as inside the noise floor. Frame-level stability is not enough for a sub-millisecond pass — the composite sharpen filter once read 0.416 vs 0.424 ms at frame level while the pass itself tripled.
- **No implied validation result.** Validation layers are compiled out of Release (`VULKAN_ENGINE_ENABLE_VALIDATION=0`), and the harness only runs Release, so every report says outright that it cannot show validation errors rather than letting silence read as a clean frame.
- **Typed, validated overrides.** An unknown dotted key or a value of the wrong type aborts, because a silently ignored override would measure the baseline twice and read as "no effect".
- **Never writing the user's settings file.** `config/runtime_settings.json` is per-user state. The harness writes a separate `<label>.settings.json` under `build/measurements/` and passes it with `--settings`, and raises if the persisted file changed underneath a run rather than assuming it owns it.

Logs and a `summary.json` land in `build/measurements/`. The summary records the full effective settings, not just the overrides, because configuration A is "whatever was persisted that day" and that file lives outside git. Nested scopes are measured rather than assumed: the report sums them against their parent and either annotates the share it found or suppresses them, for the reason in the previous section.

Scene presets are not persisted settings, so `--set` cannot reach them. `--args` passes the renderer's own flags through instead, applied identically to both sides of an A/B so the scene is never the variable:

```bash
python3 tools/dev/measure_gpu.py ab --b-set ssr.enabled=false --args --scene fragment-stress
```

`--args` is an argparse `REMAINDER`, so it swallows everything after it: `--repeat`, `--warmup` and the rest must come before it. A hand-captured log still works and is still fed to `parse`.

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
- `DepthPrepass` when `renderer.enableDepthPrepass` is on
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

## Load knobs, and why they did not fix the drift gate

`tools/dev/measure_gpu.py` refuses a comparison when the repeated control moves
more than 1% across the series. Its own documentation suggests the remedy: *"A
heavier preset pins the clock and the same comparison becomes quotable."* On this
machine -- an RTX 3080 Ti **Laptop** -- that advice is backwards, and it is worth
recording so nobody spends another afternoon on it.

### The knobs

**`--scene gpu-stress`** is `fragment-stress` turned up: 24 overdraw layers
instead of 6, 512 lights instead of 192. Both knobs cost GPU and almost no CPU,
which is the point -- a preset that scaled draw items instead would just move the
bottleneck back onto the CPU, where `stress` already is. The slabs are spread
across a fixed depth span rather than at a fixed spacing, so more layers means
denser overdraw rather than a longer tunnel whose far end shrinks out of frame; at
six layers that arithmetic is identical to what it replaced, so `fragment-stress`
is bit-identical and keeps the measurements taken on it.

Worth knowing what those 24 layers actually buy, now that `--overdraw` can say:
**1.276 shaded fragments per pixel, against `--scene sponza`'s 2.187.** Stacked
full-screen quads occlude each other perfectly, so early-Z rejects nearly all of
the manufactured overdraw before it reaches a fragment shader. The preset still
works as a load knob — it makes the GPU frame large enough for the drift gate,
which is what it was built for — but it is not a scene with high shading
overdraw, and it should not be used as a stand-in for one.

**`--window-size WIDTHxHEIGHT`** is the lever with no ceiling. `renderScale` only
scales *down*, the scene presets are bounded by early-Z and by
`kMaxLightsPerCluster`, and resolution is the one input that multiplies fragment
work while costing nothing on the CPU:

| `--scene gpu-stress` | GPU frame | CPU prep | CPU record |
| --- | --- | --- | --- |
| 1280x720 | 2.45 ms | 0.15 ms | 0.20 ms |
| 1920x1080 | 3.71 ms | 0.19 ms | 0.25 ms |
| 2560x1440 | 6.68 ms | 0.16 ms | 0.23 ms |
| 3840x2160 | **14.54 ms** | 0.18 ms | 0.28 ms |

Six times the GPU work with the CPU flat, ending at 31:1 GPU-bound. As a way to
put load on the card, it works exactly as intended.

### It made the measurement worse, not better

The drift gate was run against `punctualShadows.gpuCasterCulling` at every size.
Best control drift achieved, against a 1% limit:

| load | GPU frame | best drift |
| --- | --- | --- |
| `stress` at 720p | 1.0 ms | **1.4%** |
| `gpu-stress` at 1440p | 7.4 ms | 16.8% |
| `gpu-stress` at 4K | 14.5 ms | 9.6% |

Drift grows with GPU load. Shortening the sample window did not help either -- the
1440p figure above is from the shortest run the harness accepts.

The cause is visible from `nvidia-smi`: this GPU's graphics clock ranges from
**472 MHz idle to 2100 MHz boost**, a 4.4x swing, with no user-settable power
limit. A desktop card under sustained load settles at a steady boost clock, which
is what "a heavier preset pins the clock" assumes. A laptop card under sustained
load throttles instead, and a throttling clock *is* a drifting control.

### What actually fixes it: three separate causes, in order

The gate turned out to be refusing for three unrelated reasons at once. Each was
found only after the one before it was removed, so they are recorded in the order
they surfaced rather than the order they matter.

**1. The clock swings.** Pin it. `tools/dev/gpu_clock.ps1` wraps `nvidia-smi` and
self-elevates, because the device writes need administrator:

```
powershell -File tools/dev/gpu_clock.ps1 lock -Mhz 1100   # graphics and memory
powershell -File tools/dev/gpu_clock.ps1 status           # needs no privileges
powershell -File tools/dev/gpu_clock.ps1 unlock
```

Pinning the graphics clock at 1400 MHz took a `--scene stress` A/B from eight
consecutive refusals, best 1.4%, to 0.30%. The first time the gate had ever
passed on this machine.

**2. A pin is a ceiling, not a floor.** The same A/B on `--scene gpu-stress` was
still refused at 9.0%. Sampling `nvidia-smi` every two seconds during the run,
with the pin in place:

| t | graphics | memory | temp | sw_thermal_slowdown |
| --- | --- | --- | --- | --- |
| 2 s | 1402 MHz | 7001 MHz | 86 C | Not Active |
| 4 s | 1282 MHz | 8001 MHz | 87 C | **Active** |
| 12 s | 1320 MHz | 6001 MHz | 86 C | **Active** |
| 20 s | 1320 MHz | 7001 MHz | 86 C | **Active** |

Two leaks. The card reaches its thermal threshold and slides *below* the pin, and
the memory clock was never pinned at all -- it hops 6001/7001/8001, a 33% swing
in the one resource a bandwidth-bound scene is waiting on. So the script pins
both, and a heavier scene needs a *lower* graphics pin rather than a higher one.
Run `status` during a measurement: if a throttle reason reads Active, the pin is
not holding.

**3. The median was the wrong statistic.** With both clocks pinned at 1100/7001
and no throttle reason active for the whole run, `gpu-stress` drifted *worse*:
15.9%. The pin was holding and the drift got bigger, which falsified the thermal
explanation outright.

The per-sample distribution says why. Every scope has a tight floor and a long
upper tail -- `MainHDRPass` over one launch: p10 1.962, min 1.956, max 2.917.
That is the signature of something else using the GPU, and this laptop has 21
other processes on the device: the compositor, three WebView2 instances, two
NVIDIA overlays, a Steam helper, two chat clients. Contention is **one-sided**.
Another process can only ever add time to a frame, never remove it.

A median rides that tail. A low percentile does not. Same six logs, only the
statistic changed:

| statistic | control drift | A/B delta |
| --- | --- | --- |
| median | 11.8% -- refused | -0.197 ms (-6.5%) |
| p25 | 0.61% -- passes | +0.019 ms |
| p10 | 0.84% -- passes | +0.022 ms |
| min | 0.85% -- passes | +0.021 ms |

The median did not merely have more noise. **It reported the delta with the wrong
sign**, claiming GPU culling made the heavy scene 6.5% faster. The low percentiles
agreed with each other and with the same comparison run on a light scene that had
passed the gate outright.

So `measure_gpu.py` quotes p10 rather than the median; see `QUOTED_PERCENTILE`.
What that gives up is the tail itself -- a regression that shows up only as
occasional long frames, a shader recompile or an allocator stall, is exactly what
p10 discards. Use `run` and read its Max column when that is the question.

### Where that leaves the gate

Both scenes now pass, at the same 1100/7001 pin:

| A/B on `punctualShadows.gpuCasterCulling` | control drift |
| --- | --- |
| `--scene stress` | **0.14%** |
| `--scene gpu-stress` | **0.50%** |

`--scene sponza` needs a lower pin than either, and passes at **800/7001** with
**0.16%** drift. It refuses at 1200 (5.1%), 1000 (1.7%) and 900 (1.1%) — the
scene is heavy enough that the ceiling that works for `gpu-stress` does not work
for it. See the section on that preset above.

Until a comparison passes the gate it is directional at best, and the honest
thing is to report a refused comparison as refused. Absolute numbers, per-pass
breakdowns and anything measured in bytes remain trustworthy; it was only the A/B
deltas that the machine took away.
