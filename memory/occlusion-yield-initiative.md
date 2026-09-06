---
name: occlusion-yield-initiative
description: "Hi-Z occlusion culling now suspends itself when it culls nothing (SHIPPED c50a989): -9.2% on the default scene; the probe bug that stale readback counters caused is the thing to remember"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4da1878a-f14b-40e8-b1c0-fc00e696ebfd
  modified: 2026-08-14T01:46:15.225Z
---

Merged `c50a989` (`--no-ff`, pushed), branch deleted. 227 tests (11 new,
mutation-tested), 0 validation errors. Default scene went 16.5 -> 15.0 ms.

## The finding

On the default portfolio scene, GPU occlusion culling removed **0 draw items in
every sampled frame** while the Hi-Z pyramid it needs cost 1.36 ms -- 8% of the
frame. `DepthPyramid` + `DepthPyramidMid` are the only consumers of the pyramid;
nothing else reads it.

Two changes:

1. **The end-of-frame pyramid build was unconditional** -- it ran even with
   occlusion culling disabled, 0.68 ms/frame feeding nothing. Gated on intent,
   NOT on `depthPyramid_.valid()`: validity is an output of the build, so gating
   on it latches the pyramid off forever after one skip.
2. **`OcclusionYieldController`** (`renderer/OcclusionYield.h`, Vulkan-free,
   unit-tested like [[render-scale-initiative]]'s DynamicResolution): suspend
   after 60 tested frames of zero yield, probe every 180. Occlusion culling then
   switches off as a *consequence* -- the test already requires a valid pyramid,
   so nothing else needed gating.

Safe by construction: skipping occlusion culling can only draw MORE, never less,
so a wrong decision costs frame time and never pixels. That is what justifies
the aggressive policy and `enableAdaptiveOcclusion` defaulting ON.

A/B/A/B at scale 1.0, control back within 0.14%: 16.494/16.517 ->
15.024/14.923 ms (-9.2%), identical visible draw counts and scene luminance.
Stress scene (670 culled) stays Active and never suspends.

## The bug worth remembering

Cull counters come back through a **per-frame-slot readback buffer and lag two
frames**. The first version asked "is occlusion enabled right now?" to decide
whether the counters meant anything. On a probe frame that is true, but the
counters belong to the *suspended* frame that last used the slot, so every probe
read zero yield and re-suspended -- **once suspended it could never recover**,
silently losing occlusion culling on exactly the scenes that need it.

Fix: record the flag per frame SLOT when the cull is issued, read it back
alongside that slot's counters. Unit tests could not catch this (the state
machine was correct; the caller was wrong). Found by forcing the controller to
start Suspended and loading the stress scene -- see
[[instrument-before-guessing-runtime-bugs]]. **Any readback-driven controller in
this engine has this hazard**; the counters describe a frame two back, not now.

Also: `updateOcclusionYield` must run after the fence wait and BEFORE
`resetGpuCullFrameCounters`, which zeroes the slot's readback-ready flag. Called
from frame prep instead, `readMainCounters` always returns false and the
controller sits in Active forever.

Related: [[two-phase-occlusion-initiative]], [[shadow-cascade-cost]],
[[back-to-back-or-dont-claim]].
