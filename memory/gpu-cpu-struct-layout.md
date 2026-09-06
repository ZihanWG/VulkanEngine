---
name: gpu-cpu-struct-layout
description: "The CPU/GPU shared-struct contract: layout lives in one shared GLSL header, ObjectFrameData went 688B to 192B over three rounds (dbdd8c2, 19b60b5, 94768a2), and how to verify a layout change when validation cannot"
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-09T15:34:42.328Z
---

## The layout is now declared in exactly one place

`src/shaders/object_frame_data.glsl` holds `ObjectFrameData` + `FrameConstants`
and their `buffer_reference` blocks. **Six** shaders include it — `simple.vert`,
`simple_skinned.vert`, `shadow.vert`, `shadow_masked.vert`,
`shadow_punctual.vert`, `probe_capture.vert` — because the array stride depends
on the whole struct, so even a shader reading only `lightMvp` had to declare
every member. Before `dbdd8c2` all six carried verbatim copies.

`CMakeLists.txt` has `SHADER_INCLUDES` listed in each shader's `DEPENDS`, so
editing the header recompiles everything. Without that a stale `.spv` keeps the
old layout while C++ moves on. **glslc resolves quoted `#include` relative to
the including file and needs no `-I` and no `GL_GOOGLE_include_directive`** —
verified, not assumed.

## Why the layouts agree, precisely

Every member is a multiple of 16 bytes. That is the whole mechanism: GLM is
**not** built with `GLM_FORCE_DEFAULT_ALIGNED_GENTYPES`, so `glm::vec4`'s
alignment is 4, not 16, and `ObjectFrameData`'s C++ alignment is 4. The layouts
match because each member's *size* lands it on a 16-byte boundary, not because
alignments correspond.

**Adding one bare `float`, `vec3`, or `bool` breaks it silently.** C++ would put
the next member at +4 where std430 aligns a vec4/mat4 to +16.

## Sizes after three rounds (`dbdd8c2`, `19b60b5`, `94768a2`)

- `ObjectFrameData` **192** (was 688, **−72%**) — per draw item
- `FrameConstants` **496** — once per frame, own buffer, own device address

What is left per item is genuinely per-object: `model`, `prevMvpNoJitter`, and
four vec4s of material data.
- `PushConstants` 120, `frameConstantsAddress` at offset 112 (limit is 128)
- `ProbeCapturePushConstants` 120, its address replaced two padding words that
  were already an 8-byte hole before the mat4

Design note: a **second buffer_reference, not a UBO**, because the vertex stage
binds no descriptor set at all. A UBO would have forced descriptor sets onto the
shadow pipelines.

**`lightMvp[4]` is gone** (`19b60b5`). It was `cascadeViewProjection[i] * model`,
and only `model` was per-object. The cascade matrices live in `FrameConstants`
now and shaders build the light-space position themselves.

The measurement worth keeping: three shaders (`simple.vert`,
`simple_skinned.vert`, `probe_capture.vert`) already compute `worldPosition`, so
`cascadeViewProjection[i] * worldPosition` costs exactly what
`lightMvp[i] * localPosition` did. **`shadow.vert` and `shadow_masked.vert` do
not** — they transform once and stop — so they gained a `model` multiply per
vertex. That predicted regression measured at **exactly zero**: `CSMShadowPass`
0.854 → 0.854 ms, because the shadow pass is not vertex-ALU bound here and the
cascade matrix is one frame-constant address every vertex hits, so it caches.
Control reproducibility that day was 0.001 ms, so "unchanged" is real rather
than noise-hidden.

**`mvp` and `currMvpNoJitter` are gone too** (`94768a2`) — they were
`jitteredViewProjection * model` and `viewProjection * model`. No shader paid
anything: both consumers already compute `worldPosition`.

**`prevMvpNoJitter` stays, and the reason generalises.** It looks like the same
shape but is `previousViewProjection * *previous* model`, and the previous model
matrix is genuinely per-object. Storing that instead costs the same 64 bytes plus
a multiply — strictly worse. Before assuming a matrix is a foldable product,
check *which* model matrix it uses.

## How to verify a layout change — validation will NOT help

A mismatch produces no VUID. Vulkan sees bytes. **Use scene luminance as the
oracle**: every frame constant feeds the lighting math, so a wrong offset moves
`average luminance` (logged once per second). Observed across the split:

```
main pass              0.3192 -> 0.3194   (0.06%)
probe capture, GI on   0.3292 -> 0.3284   (0.24%)
```

**Off-by-default consumers need their own run.** A default run does not exercise
them, so each round had a blind spot to close:
- probe capture (`gi.enabled`) — reads the frame constants
- **TAA (`taa.enabled`) — reads `vCurrClipPos`, the velocity path.** This one
  matters most: TAA accumulates across frames, so a wrong velocity reprojects
  history from the wrong pixels and compounds over a long run rather than
  staying invisible. Verified 0.3188 to 0.3189.

The skinned path is covered by the main-pass run.

Also force a clean shader rebuild (`rm -rf build/*/shaders`) when verifying, or
the include wiring itself goes untested.

See [[runtime-settings-persistence]] for the temp-settings-file trick used to
turn GI on, and remember to delete `config/runtime_settings.json` afterwards.
