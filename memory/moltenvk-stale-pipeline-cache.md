---
name: moltenvk-stale-pipeline-cache
description: The MoltenVK stale-pipeline-cache startup crash did not reproduce under forced conditions on the M3; the engine-side shader-hash guard shipped anyway
metadata: 
  node_type: memory
  type: project
  originSessionId: 84b1d476-cec7-4c15-9e8b-d154eb735b61
  modified: 2026-08-01T08:55:56.070Z
---

2026-08-01: shipped a shader-SPIR-V digest in the persisted pipeline cache header (commit c506ac9, branch claude/musing-elion-6cfee7) so `~/.cache/VulkanEngine/pipeline_cache.bin` is discarded on any shader change. The guard itself is GPU-verified on the M3: matching cache loads, changed shaders discard it, engine starts clean both ways.

**What did NOT reproduce:** the underlying MoltenVK failure (`VK_ERROR_INITIALIZATION_FAILED` / "Fragment input(s) `user(locn24)` mismatching vertex shader output"). Tried force-feeding a genuine cache saved from the original shaders into a run with a location-24 varying added to simple.vert + simple_bindless.frag (bypassing the new guard by stamping the current hash onto the stale blob) — MoltenVK tolerated it and ran a full 15s. So the crash is conditional on something not captured here: MoltenVK/SDK version, build config, or which pipelines happen to hit a colliding cache key.

**How to apply:** if this crash resurfaces, do not assume the shader-hash guard failed — first check whether the cache was even loaded (the startup log states which of Usable/Missing/Malformed/ShaderMismatch/DeviceMismatch applied). Related: [[vulkanengine-cannot-run-in-sandbox]], [[instrument-before-guessing-runtime-bugs]].
