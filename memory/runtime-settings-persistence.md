---
name: runtime-settings-persistence
description: "How runtime settings persistence is wired (RuntimeSettings.h is glm-free and Vulkan-free by design), and that GTAO/fog/punctual shadows now persist (merged 1d75562)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-09T10:35:13.105Z
---

**GTAO, volumetric fog, and punctual shadows now persist.** MERGED AND PUSHED
2026-08-06 as `1d75562`; 156 tests (up from 150), 0 VUIDs.

## The layering rule, which is the thing to not violate

`src/renderer/RuntimeSettings.h` is deliberately free of Vulkan **and glm**. It
compiles into `VulkanEngineCore` so the clamping is unit-testable without a
device. That is why:

- `GiSettings` uses `float gridOrigin[3]`, not `glm::vec3`
- `SsaoSettings` was moved here out of the renderer earlier
- `FogSettings` moved here from `VolumetricFogPass.h` in this change, and its
  `glm::vec3 scatteringColor` became `float scatteringColor[3]`

`kDefaultFogMaxDistance` moved with it and `VolumetricFog.h` re-exports it with
`using ve::kDefaultFogMaxDistance;` so the fog math still reads with its own
constants. Types here are `ve::`, not `ve::renderer::` — a member declared as
`renderer::FogSettings` will not compile; declare it unqualified and let
enclosing-namespace lookup find it.

**All renderer toggles are now persisted** (`useClusteredLighting` was the last
holdout, added `d5f1dbe`). It was missed the first time because it lives as a
loose `bool` on `Renderer` rather than inside a settings struct — when auditing
for gaps, compare every `use*_` member on `Renderer` against `RuntimeSettings`,
don't just look at the structs.

**A Renderer member's initializer is not the default.** `loadRuntimeSettingsAtStartup`
calls `applyRuntimeSettings` **unconditionally, before it checks whether the file
loaded**, so a default-constructed `RuntimeSettings` always overwrites those
initializers. `useGpuOcclusionCulling_` read `false` while its setting said
`true` — Renderer.h claimed the opposite of what the engine does. Fixed in
`d5f1dbe`; the whole toggle set was compared and it was the only drift.

## Wiring checklist for adding a persisted setting

Five places, and missing any one fails silently:
1. the struct + a member on `RuntimeSettings`
2. load block in `RuntimeSettings.cpp` (`readBool`/`readFloat`/`readInt`)
3. save block in the same file
4. `clampRuntimeSettings` (both the free function's parameter list and
   `Renderer::clampRuntimeSettings`, which forwards to it)
5. `Renderer::applyRuntimeSettings` **and** `Renderer::captureRuntimeSettings`

Toggles that depend on a subsystem being up follow the culling pattern: assigned
unguarded in the `Startup` branch, because that runs before those subsystems are
created, and gated on an `available()`-style check in the `Runtime` branch.

## The example file drifts, and now a test stops it

`config/runtime_settings.example.json` is the only schema documentation and had
**silently lost its `gi` and `lod` sections entirely** before this change. There
is now a test that loads the example over default-constructed settings and
asserts nothing changed, which catches both a missing section and a value that
drifted from its default. It needs `VULKAN_ENGINE_CONFIG_DIR`, added to
`tests/CMakeLists.txt` (`VULKAN_ENGINE_ASSET_DIR` points at `assets/`, not root).

## Verify persistence end to end, not just in unit tests

A round-trip unit test only proves the struct survives JSON. To prove the value
reaches the renderer, write `config/runtime_settings.json` (gitignored), run, and
look for a *structural* change in the log: setting `punctualShadows.enabled` false
removes the `PunctualShadowAtlas` row from the GPU timings entirely, and
`ssao.enabled` true adds a `GTAO` row. **Delete the file afterwards** — it is the
user's real settings file and would persist into their next run.
