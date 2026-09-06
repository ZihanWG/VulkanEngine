---
name: cluster-grid-resolution
description: "Cluster grid is 32x18x24 (merged 12c3a69, was 16x9): -19% MainHDRPass on dense lighting AND a correctness fix, since 160px tiles were silently dropping lights past the 64/cluster cap"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3a9b7da7-f7d9-4e61-9a3a-3da09dafe927
  modified: 2026-08-11T06:36:24.990Z
---

Shipped 2026-08-10, merged `12c3a69`, pushed.

## The result

`16x9x24 -> 32x18x24`. Fragment stress scene (192 dense point lights):

```
MainHDRPass   18.71 -> 15.18 ms   (-19%)
Frame total   23.76 -> 20.31 ms
```

Default scene does **not** regress (MainHDRPass 9.95 -> 9.63, ClusterBuild and
LightCull unchanged), so this is not a trade against sparse scenes.

## Where the knee is, and why — do not re-sweep

| grid | clusters | index list | MainHDRPass | LightCull |
|---|---|---|---|---|
| 16x9x24 | 3,456 | 0.85 MB | 18.71 | 0.081 |
| **32x18x24** | 13,824 | 3.4 MB | **15.18** | 0.089 |
| 48x27x24 | 31,104 | 7.6 MB | 15.26 | 0.161 |
| 64x36x24 | 55,296 | 13.5 MB | 15.37 | 0.260 |

**Past 32x18 there is nothing left to win.** Once a tile is smaller than a
light's screen footprint, refining it stops removing lights from the loop —
while LightCull cost and the (uncompacted) index list keep growing linearly
with cluster count.

## The part that was not the intent: it fixes silently lost light

`light_cull.comp` stops appending at `kMaxLightsPerCluster` (64). No error, no
counter. At 16x9 on a 2560x1440 drawable each XY tile is 160x160 px, so dense
scenes saturated froxels and **rendered darker than the lighting called for**.

Evidence: fragment stress scene luminance **0.0833 -> 0.1023, +23%**, while the
sparse default scene moved 0.9% (inside its own spread). Nothing but saturation
explains a rise confined to the dense-light scene. Inferred from that
signature, not counted directly — if it ever matters, add a saturation counter
to the cull shader.

`clustered_lighting.md` lists the 64-light cap as a known limitation. It was
biting in practice, which nobody had noticed because no scene was dense enough
to show it until the fragment stress scene existed.

## Grid constants are shared now

`src/shaders/cluster_grid.glsl` holds `kClusterGridX/Y/Z`, `kClusterCount`,
`kMaxLightsPerCluster`, included by `simple_bindless.frag`, `fog_inject.comp`
and `light_cull.comp`, and listed in `SHADER_INCLUDES` so a stale `.spv` cannot
outlive an edit. `ClusterGrid.h` is authoritative and documents the mirror.

Before this they were duplicated verbatim in all three. A missed edit is
silent: the main pass and fog derive a cluster index from screen position while
the cull writes the list they read, so a mismatch makes fragments read a
different froxel's light list — reads as lights bleeding across tile edges, not
as a build error. `cluster_build.comp` was already clean (it takes dims from a
UBO).

See [[stress-scene]] — this change was only measurable because the fragment
stress scene exists.
