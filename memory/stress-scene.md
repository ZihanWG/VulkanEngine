---
name: stress-scene
description: "Two stress scenes: geometry (f3f40c1, loads culling/LOD/CPU prep) and fragment (830b10d, loads shading). Which one to use for what, and the CPU measurements that killed multi-threaded recording."
metadata: 
  node_type: memory
  type: project
  originSessionId: 3a9b7da7-f7d9-4e61-9a3a-3da09dafe927
  modified: 2026-08-11T06:04:10.070Z
---

Shipped 2026-08-10, merged `f3f40c1`, pushed. UI: **advanced mode → `Scene Presets`
→ `Stress scene` → `Load Stress Scene`**. `SceneBuilder::appendStressScene`.

## Why it exists

Eleven draw items could not show a signal from anything this renderer is built
around. Before it: occlusion culling reported **exactly zero** rejections, every
object picked LOD 0, and the parallel frame-prep loops never cleared their
64-item chunk threshold so they ran inline. Several changes before this were
reasoned about against a scene that could not measure them.

## What it moves

| | default | stress |
|---|---|---|
| draw items | 11 | 2322 |
| frustum culled | 0 | 988 (43%) |
| **occlusion culled** | **0** | **670 (29%)** |
| LOD spread | L0 only | L0=336 L2=34 L3=294 |
| punctual cull+record CPU | 5 us | **223 us** |

## The trap — do not read frame times against it naively

**The stress scene is FASTER than the default scene**: ~10.8 ms against ~18 ms,
`MainHDRPass` 3.7 against 10. `MainHDRPass` is fragment-bound (the punctual light
loop is ~64% of it), and 2311 small distant objects cover far fewer pixels than
eleven close ones.

**It loads culling, batching, LOD and CPU frame prep. It is not a fragment-load
test.** That is what the fragment stress scene below is for.

## The companion: fragment stress (`830b10d`)

Same panel, `Load Fragment Stress`. Six full-frame slabs stacked in depth so
every pixel shades several times, viewed close and low, with **192 point lights
packed densely** into the volume they occupy.

**Light density, not light count, is the lever.** The per-fragment punctual loop
scales with lights *per froxel*; the same 192 lights spread over a large scene
would load nothing.

```
                 default   geometry stress   fragment stress
MainHDRPass      10.1 ms        3.7 ms          18.5 ms
Frame total      18.0 ms       10.8 ms          23.5 ms
MainHDRPass %      56%           34%              78%
```

**Use fragment stress as the baseline for any shading change.** Its range is
unusually tight ([18.09, 18.52]) — far better signal-to-noise than the default
scene, where sub-millisecond changes needed back-to-back controls to resolve.

The two are mutually exclusive: loading either removes the other's objects,
since both drive the camera and their own lighting.

## First thing it settled: multi-threaded command recording is NOT worth doing

Measured on the stress scene (2311 objects), CPU cost per frame:

```
frame prep       ~490 us
command record   ~280 us   <- of which 225 us is the punctual shadow caster loop
GPU frame total  10,630 us
```

Recording is **2.6% of the frame**, and the frame is GPU-bound by ~13x — the CPU
finishes long before the GPU, so even perfect parallelisation of recording saves
nothing in wall clock. Secondary command buffers were the big remaining
structural gap on paper; the data says leave it alone.

Two reasons recording is so cheap here, worth remembering:
- **Recording is O(batches), not O(draw items)** — that is what GPU-driven
  indirect drawing buys. 2311 objects record as 5 `vkCmdDrawIndexedIndirect`
  calls.
- **80% of what remains is one loop**, the punctual shadow atlas drawing casters
  per slot (O(slots x draw items), already documented in punctual_shadows.md).
  If recording CPU ever did matter, that is the target — not the main pass.

## Design points worth not re-deriving

- **The occluder slabs matter more than the object count.** A big scene with
  nothing in front of anything still reports zero occlusion rejections.
- **Objects are static on purpose.** An animated transform invalidates the depth
  pyramid every frame, which leaves occlusion culling nothing to test against.
- Cube and sphere alternate so both batches are non-trivial and the LOD chain
  (on the sphere) gets real distance spread.

## Capacity

`kMaxDrawItems`/`kMaxFrameObjects` 1024 -> **8192**. The hard ceiling is **65535**:
`cull.comp` packs the object slot into the low 16 bits of `firstInstance` (high
bits carry the LOD), now pinned by a `static_assert`.

## A "bug" I found and had to retract — check before claiming

I thought shadow culling was silently capped at `kMaxDrawItems / cascadeCount`
because `setShadowCullFrameInfo` passes `drawItems * cascadeCount` while the
input buffer is sized `kMaxDrawItems`. **Wrong.** Shadow culling dispatches
**once per cascade**, each passing `allDrawItems_.size()`; the product is only
used to clamp the *stats counters*, which all four dispatches accumulate into.
The buffer was always correctly sized. Only the stats clamp needed raising
(`kMaxShadowCullStatsDrawItems`), and it never mattered before because 11×4 was
far under 1024.
