---
name: shadow-cascade-cost
description: "CSMShadowPass is per-pass encoder cost, not raster: shadow resolution is not a lever, cascade count is; union cull SHIPPED, multiview layered pass measured slower and ships OFF"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4da1878a-f14b-40e8-b1c0-fc00e696ebfd
  modified: 2026-08-13T13:18:57.126Z
---

Answers the 2026-08-13 question "can the ~4 ms resolution-independent floor be
moved". Merged `d78c31e` (`--no-ff`, pushed), branch deleted. 216 tests, 0
validation errors.

## Shadow map resolution is NOT a lever -- do not re-offer scaling it

Rebuilt at three fixed CSM resolutions, render scale 1.0:

```
2048 (16.8 M texels, 4 cascades)   CSMShadowPass 0.822 ms
1024 ( 4.2 M)                                    0.775
 512 ( 1.0 M)                                    0.812
```

A 16x texel cut moves nothing. Not a light-scene artefact either: in the
geometry stress scene (9288 shadow draws vs 44) it is 1.11 ms at 2048 and 0.98
at 512. **Cascade count is the lever** -- 4 / 2 / 1 cascades = 0.82 / 0.40 /
0.25 ms, dead linear. It is per-render-pass cost (one Metal command encoder per
cascade), not raster and not draw submission.

## What shipped

**Union shadow cull (ON).** One dispatch for all cascades: visible if ANY
cascade frustum wants it; every cascade replays that list. Frusta moved to the
frame-params buffer (`GpuCullFrameParams` 224 -> 608) because four are 384 bytes
against a 128-byte push-constant guarantee. A/B/A/B at scale 0.25: frame 4.13 /
5.12 -> 3.30 / 3.34 ms, CSMShadowPass ~0.9 -> ~0.45. Invisible at scale 1.0
(MainHDRPass swamps it).

**Multiview layered pass (OFF, `enableLayeredCascades`).** All cascades as views
of one pass; `shadow_layered.vert` / `shadow_masked_layered.vert` read
`gl_ViewIndex`. Two shader variants are mandatory -- `gl_ViewIndex` needs the
MultiView SPIR-V capability, invalid in a pipeline without a view mask, so it
cannot be a runtime branch. Output is identical (luminance parity) but
CSMShadowPass is ~20% SLOWER across four measurement windows (0.43/0.51/0.52 vs
0.54/0.59/0.63). Apple amplifies at most 2 views, so 4 is emulated and costs
more than the 3 encoders it removes. **Do not re-offer enabling it here**; kept
because the call inverts on native-multiview hardware.

Also fixed: `VulkanShadowMap` inferred view type from `layerCount`, so
`cascadeCount == 1` gave a 2D view to a `sampler2DArray` binding (22 VUIDs/run).
View kind is now explicit at both call sites; the punctual atlas is the only
legitimate single-layer user.

## Measurement note

Mail.app hit 95% CPU mid-session and made everything ~3x slower; scale-0.25
numbers from that window read 8-10 ms where the same build read 3.3. **Check
`ps -Ao pcpu,comm -r | head` before trusting a run.** Scale 1.0 held its 17.1 ms
baseline all session and was the usable comparison point. See
[[back-to-back-or-dont-claim]] and [[gpu-profiler-nested-scopes]] -- scope times
do not sum to the frame total, and the frame total is the one that decides.

Related: [[render-scale-initiative]], [[punctual-shadows-initiative]].
