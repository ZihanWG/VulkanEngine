---
name: cpu-frame-prep-cost
description: "MEASURED AND REJECTED: the persistent GPU scene table buys no frame time — CPU frame prep is 9-14% of the GPU frame and cannot become the bottleneck within the engine's own object cap"
metadata: 
  node_type: memory
  type: project
  originSessionId: 16cda6ce-0ffd-440b-8aa7-ade1e10dd7c7
  modified: 2026-08-19T00:00:00.000Z
---

**The persistent-GPU-scene-table premise was falsified before any of it was
built.** The claim was that rebuilding `allDrawItems_` and `ObjectFrameData`
every frame (`RendererFrame.cpp:735` and `:1698`, the latter self-described as
"the heaviest CPU loop of the frame") is a cost worth eliminating.

Release build, geometry stress scene (**2322 draw items**, vs the default
scene's 11), 29 samples after warm-up, medians:

| render scale | GPU frame | CPU frame prep | ratio |
| --- | --- | --- | --- |
| 1.0 | 15.460 ms | 1.436 ms | **9.3%** |
| 0.5 | 9.625 ms | 1.350 ms | **14.0%** |

Prep runs *before* the graphics command buffer is even recorded and the GPU is
the bottleneck, so removing it entirely would save **0 ms of frame time**.
Scale 0.5 was measured specifically because it is the cheapest GPU frame
available — the CPU still does not surface.

**It cannot become the bottleneck at this engine's limits either.** 2322 items /
1.436 ms = **0.62 µs per draw item**. At `kMaxDrawItems = 8192` that is ~5.1 ms,
still under even the 9.6 ms half-scale GPU frame; equalling a full-scale frame
would need ~25,000 items, 3x past the cap. So the table would have to grow past
what the engine can express before the work it saves is exposed.

**Do not re-propose this as a performance change.** It remains defensible purely
as an architectural step toward GPU-side batch generation — a different claim,
with a GPU-side payoff, that would need its own justification.

**Debug timings would have inverted the answer**: the same scene in Debug reads
11.1 ms prep against an 18.4 ms GPU frame (60%). Only the Release number is
evidence — the same rule `/measure` enforces.

**Two structural blockers found while designing it, still true and worth knowing
if this is ever revisited** ([[gpu-cpu-struct-layout]] is the related trap):

1. `ObjectFrameData::prevMvpNoJitter` folds in `previousFrameViewProjection_`, so
   **every object's record changes whenever the camera moves** — dirty tracking
   would degrade to full upload, exactly the trap [[csm-cascade-cache]] hit. The
   fix is already described in `object_frame_data.glsl:34-37`, where it was
   weighed and declined as neutral: store `prevModel` and move `prevViewProjection`
   into `FrameConstants`. Neutral then; enabling under a persistent table.
2. `frameDataIndex = drawIndex` (`RendererFrame.cpp:762`), and
   `sortTransparentDrawItems()` re-sorts the Blend bucket by camera depth every
   frame — so record slots move. Stable per-(object, submesh) slots are required
   first. `kSkinnedObjectFrameSlot` is the existing precedent.

**Shipped from this work regardless** (`5401f71`, **merged to main via PR #15 as
`b13f19e`, pushed, branch deleted**): `--scene stress|fragment-stress|occlusion|cornell|default`
makes every preset scriptable — removing the limitation `tools/agent/measure_gpu.py`
documents — and `Frame prep CPU` now prints beside `Frame total` in the
once-per-second diagnostics block that harness parses. Related:
[[mainhdrpass-attribution]], [[stress-scene]], [[back-to-back-or-dont-claim]].
