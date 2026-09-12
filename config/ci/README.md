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

Both of those legs are kept as regression guards.

## Adding a leg

1. Add the delta file here.
2. Add a `name|arguments` line to the leg list in
   `.github/workflows/headless-render.yml`.

Prefer a leg that changes the **shape of the frame graph** — a pass appearing or
disappearing. A knob that only moves a number through a shader is already covered
by the default leg, and every leg costs a render.

Two things a leg cannot express, and take command-line flags instead:

- **VSM stages** (`--vsm mark|render|shadows`). The page pool is allocated in the
  renderer constructor and gated on the marking stage, so it cannot be reached by
  a setting applied afterwards.
- **Scene presets** (`--scene ...`). Not persisted settings at all.

## Running the sweep locally

The same legs, against a real GPU rather than lavapipe:

```bash
./build/release/VulkanEngine --settings config/ci/occlusion-off.json \
    --deterministic --exit-after-frames 10 --fail-on-validation-error
```

A clean leg prints `Render graph backstop: 0 order violations, 0 unrecorded
passes, 0 declaration issues over 10 frames`. A dirty one names each finding on
its own line underneath.
