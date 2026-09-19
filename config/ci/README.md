# CI configuration sweep

One file per configuration the headless job renders, beyond the default one.

Each file is a **delta**, not a full settings document. `loadRuntimeSettings`
reads the keys that are present and leaves the rest at their defaults, so a leg
says only what it changes:

```json
{
  "renderer": { "enableGpuOcclusionCulling": false }
}
```

That is deliberate. A full dump would be ~140 lines per file, twenty near-identical
copies of the defaults, and a review could not see which line was the point of the
file. It also means these files do not drift when a default changes — which is the
right behaviour, because a leg exists to pin one deviation from the defaults, not
to freeze the defaults themselves.

`schemaVersion` is written by the engine when it saves settings but never read
back, so these files omit it.

## What the legs are for

The renderer's frame graph is declared in one place and recorded across nine
translation units, and the only thing that compares the two is the render graph's
backstop (`RenderGraph::endFrame`). A pass that is declared and then never
recorded leaves the graph's barriers, resource lifetimes and pass culling all
describing work that did not happen — and no validation error is raised, because
every individual Vulkan call is still legal.

The backstop only reports on configurations that actually run. Until this sweep
existed, that was one: the defaults. Two real mismatches were found the first time
the rest were run, both invisible to every other gate in CI:

- `occlusion-off` — `DepthPyramidPass` was declared every frame while the recorder
  skips the build whenever nothing consumes the pyramid.
- `bindless-off` — `TransparentPass` was declared whenever anything was blended,
  while the recorder needs the bindless heap the transparent pipeline binds.

Both of those legs are kept as regression guards. A third arrived the same way
once the sweep's own coverage was measured rather than assumed:

- `gpu-culling-off` — `MainGpuCullingPass` was declared unconditionally while
  `GpuCulling::recordMainCull` returns without recording anything on the CPU
  culling fallback. It had been wrong on every frame of that configuration for
  as long as the graph had sequenced the frame, and no leg turned GPU culling
  off.

## What the sweep is checked to cover

`tools/check_ci_settings_coverage.py` runs as its own job in
`headless-render.yml`, ahead of the render, and fails when:

- a boolean or enum setting is never moved off its default by any leg;
- a leg sets a key to the value it already had, so it renders the defaults
  under another name (`ssr.json` and `taa-off.json` both did, which is why the
  TAA resolve pass had never run in CI at all);
- a leg file and the workflow's leg list disagree about which legs exist;
- a leg sets a key the schema does not have.

It reads `config/runtime_settings.example.json` as the schema — tracked, and
held equal to `RuntimeSettings`' own defaults by `test_runtime_settings.cpp` —
never `config/runtime_settings.json`, which is git-ignored and is whatever the
last local run happened to save.

Scalars are out of scope: they take a range rather than two sides, and a leg per
value would be a sweep of arbitrary numbers. `debugUi.*` is exempt in the
checker, with the reason written there, because the headless sweep renders no
debug UI. The `vsm.*` stage flags are covered through `--vsm`, which the checker
knows about by table rather than by exemption — rename that leg and the keys go
back to being uncovered.

## Adding a leg

1. Add the delta file here.
2. Add a `name|arguments` line to the leg list in
   `.github/workflows/headless-render.yml`.
3. Run `python3 tools/check_ci_settings_coverage.py`. Adding a *setting* without
   adding a leg for it fails there, which is the point: the count is the guard,
   not the leg list.

Prefer a leg that changes the **shape of the frame graph** — a pass appearing or
disappearing. A knob that only moves a number through a shader is already covered
by the default leg, and every leg costs a render.

State the deviation and nothing else. A leg that also restates a default is not
wrong — several spell out the whole configuration they mean, such as
`punctual-gpu-cull` naming `punctualShadows.enabled` alongside the toggle it is
really about — but a leg whose *every* key is already the default is a no-op and
the coverage check rejects it.

Two things a leg cannot express, and take command-line flags instead:

- **VSM stages** (`--vsm mark|render|shadows`). The page pool is allocated in the
  renderer constructor and gated on the marking stage, so it cannot be reached by
  a setting applied afterwards.
- **Scene presets** (`--scene ...`). Not persisted settings at all.

## Running the sweep locally

The same legs, against a real GPU rather than lavapipe:

```bash
./build/release/VulkanEngine --settings config/ci/occlusion-off.json \
    --deterministic --exit-after-frames 10 --fail-on-validation-error --sync-validation
```

`--sync-validation` is what CI runs every leg with, and it needs a Debug build
to do anything -- the validation layer is compiled out otherwise, and asking for
it there is a hard failure rather than a silent skip.

A clean leg prints `Render graph backstop: 0 order violations, 0 unrecorded
passes, 0 declaration issues over 10 frames`. A dirty one names each finding on
its own line underneath.
