---
description: Measure GPU pass cost under the project's evidence protocol (fixed scene/camera, warm-up, medians, A/B/A/B with a control-drift gate).
argument-hint: [what to measure, e.g. "ssr.enabled on vs off" | "baseline" | "renderScale.scale 1.0 vs 0.5"]
allowed-tools: Bash, Read, Edit, Grep, Glob
---

Measure: $ARGUMENTS

`AGENTS.md` requires a fixed scene/camera/settings, a warm-up, and multi-frame
statistics before any performance claim. `tools/dev/measure_gpu.py` implements
that protocol; your job is to pick the right invocation, then interpret the
result honestly.

## 1. Decide what is being compared

Read the request and classify it:

- **A/B of a setting** (the usual case) -> `ab` subcommand. A is the unchanged
  persisted config; B carries the change.
- **Absolute baseline / pass attribution map** -> `run` subcommand.
- **A code change, not a setting** -> the two configurations are two builds.
  Measure the current build with `run --label before`, apply the change, rebuild,
  then `run --label after`. Say explicitly in the report that the control was not
  interleaved, so the comparison is weaker than an `ab` series.

Map the request onto real keys in `config/runtime_settings.json`. Confirm the
dotted key exists before running -- the harness rejects unknown keys, but a
plausible-but-wrong key (`ssao.enabled` when the ask was about SSR) produces a
clean number for the wrong question. If the request does not correspond to a
persisted setting, say so instead of substituting a nearby one.

## 2. Check the ground before spending 2 minutes per run

```bash
git status --short && ls -la build/release/VulkanEngine
```

Debug timings are not evidence -- the harness only runs the Release binary. Pass
`--build` if the binary is stale or missing.

## 3. Run

A/B, interleaved, with the control repeated:

```bash
python3 tools/dev/measure_gpu.py ab --b-set ssr.enabled=true --repeat 2
```

Single configuration:

```bash
python3 tools/dev/measure_gpu.py run --label baseline
```

Each launch costs `--duration` seconds (default 30s: 10s warm-up discarded, ~20
samples). An `ab --repeat 2` series is four launches, so about two minutes. Run
it in the background and tell the user it is running rather than blocking
silently.

The renderer opens a window and takes over focus for the duration. Warn the user
before the first launch so they do not type into it.

## 4. Scenes the harness cannot set up

Scene presets (geometry stress, fragment stress, Cornell box, portfolio
showcase) are ImGui actions and reset on every launch, so they cannot be
scripted. For those, ask the user to do this once per configuration:

1. Launch `build/release/VulkanEngine`, redirecting output:
   `./build/release/VulkanEngine > build/measurements/fragment-stress.log 2>&1`
2. Load the scene from `VulkanEngine Debug`, leave the camera alone, wait ~30s.
3. Quit.

Then summarize it:

```bash
python3 tools/dev/measure_gpu.py parse build/measurements/fragment-stress.log --label fragment-stress
```

Ask for both logs before reporting a comparison. Do not pair a hand-captured
scene run against an automated default-scene run.

## 5. Interpret before reporting

- **A drifted control voids the series.** If the harness reports drift above 1%,
  report that the measurement failed and offer to rerun after the machine
  settles. Do not quote the deltas with a caveat attached -- they are not
  evidence. The harness exits non-zero in this case.
- **Never quote `Skybox`, `RenderObjects`, or `SkinnedMesh` as a breakdown of
  `MainHDRPass`.** On this tile-based target they read near zero regardless of
  their work; see `docs/profiling.md`. To attribute cost inside a render pass the
  pass must be split.
- **Do not sum passes.** Parent scopes contain their children, and async-compute
  rows (`ClusterBuild`, `LightCull`) sit on a different queue timeline.
- **Trust the `Attributable` column over the delta.** Each row carries its own
  control drift; a pass whose control moved at least as much as the A/B delta is
  marked `no` and listed under "Inside the noise floor". Report those as **not
  measured** -- never as "no cost". Frame-level stability does not make a
  sub-millisecond pass stable, which is how two past claims went wrong. If a row
  you care about lands there, raise `--repeat` and rerun.
- **The report cannot tell you the frame was clean.** Validation layers are
  compiled out of Release, so a measurement run is never evidence about
  validation errors. Say so if asked; do not imply otherwise.
- **Ignore intermittent rows.** The `Blocks` column shows how many sampled
  frames contained each pass. A row marked intermittent ran in only some frames,
  so its median is the cost of those frames, and an `A only` label there is
  sampling luck rather than a real presence difference. Do not report it as a
  difference between configurations.
- **Never quote an unstable row.** A one-sided row marked `unstable` moved more
  than a quarter of its own median between the two control runs, so the number
  does not reproduce. Report the pass as not measurable at this sample count and
  fall back to `Frame total`, which is what actually answers "what does this
  feature cost".
- **`Attributable` is a noise-floor check, not a causality check.** A row can
  clear its own control drift and still be an artifact. If a pass moves in a
  direction the change cannot plausibly cause -- especially two passes moving
  opposite ways by similar amounts, which on this tile-based target means work
  shifted across a pass boundary -- say the decomposition is not trustworthy and
  fall back to `Frame total`.
- Absolute numbers late in a session are thermally inflated. Compare within a
  series; do not compare against a number from another session.

## 6. Report

Give the user the harness's markdown table, then add:

- the exact command you ran, so the measurement is reproducible;
- what the numbers mean for the decision at hand, in one or two sentences;
- whether the change is worth keeping, and at what cost -- including "rejected,
  the effect is inside the noise" as a legitimate outcome.

If the result changes a measured claim in `docs/`, update that document as part
of the same change; `AGENTS.md` requires measured claims and their docs to move
together. Do not update committed screenshots or performance numbers that this
measurement did not actually cover.

Finally, if the outcome rejects an approach, note it as a measured-and-rejected
result so it is not re-proposed later.
