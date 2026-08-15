# Asset Load Baseline

Phase 0 of the asset cook / async upload initiative. This page records what
startup asset loading costs **today**, so any later claim about a cook step or a
transfer queue has something to be measured against.

It also records the reason the rest of that initiative is currently on hold: on
the scenes this repository ships, startup asset loading costs approximately
nothing.

## How to reproduce

```
cmake --build build/release --target VulkanEngine
./build/release/VulkanEngine --asset-load-stats --exit-after-frames 120
```

`--asset-load-stats` enables the load-time recorder and prints one report after
the run. `--exit-after-frames N` exits after N frames so the run is scriptable
and repeatable. Both default to off; a normal run is unaffected.

Instrumentation lives in `src/renderer/AssetLoadStats.h/.cpp` (GPU-free, unit
tested) with recording hooks in `VulkanTexture::uploadPixels`, timing hooks in
`Renderer`/`RendererScene`, and the wall-clock segments in `Application`.

## Measurements

Release build, default portfolio scene, 1280x720, five consecutive runs. Run 1
is discarded as cold-file-cache warm-up; runs 2-5 span 0.5%, well inside the
project's 1% control-drift gate. Values below are the median of runs 2-5.

| Segment | Median |
| --- | --- |
| renderer init (total) | 191.30 ms |
| — scene create | 168.07 ms |
| — — glTF import | 0.00 ms |
| — — texture decode wait | 0.00 ms |
| — — texture upload | 0.00 ms |
| first frame | 9.60 ms |

| Asset inventory | Value |
| --- | --- |
| textures created | 8 |
| texture device bytes (with mips) | 0.45 MiB |
| block-compressed textures | 0 of 8 |
| largest texture | 0.34 MiB, 256x256, `VK_FORMAT_R8G8B8A8_SRGB` |
| tracked `assets/` directory on disk | 68 KiB |

| Device memory after load (VMA) | Value |
| --- | --- |
| device-local used | 493.68 MiB |
| device-local allocated | 429.93 MiB |
| device-local budget | 13107.20 MiB |
| allocations | 111 |

## What the numbers say

**Startup asset loading is not a cost on this project.** All eight textures
together are 0.45 MiB and upload in less than the timer's resolution. The 493 MiB
of device-local memory is render targets, buffers, and the shadow/probe atlases —
screen-sized and scene-sized resources — not assets.

**The 168 ms in scene create is not asset loading either.** The default scene
finds no HDR environment and generates the skybox, diffuse IBL, and specular IBL
cubemaps plus the BRDF LUT procedurally. That is GPU compute at startup, and it
is what the segment actually measures.

**The glTF import path does not run.** `Renderer::tryLoadGltfScene()` is defined
but has no caller; the default portfolio scene is procedural geometry. The only
glTF touched at startup is `assets/models/skinned_rig.gltf` (3.8 KiB) through
`SkinnedMesh::createFromGltf`, a separate path.

## Consequence for the initiative

The cook + async upload plan targets texture decode cost, VRAM occupancy from
uncompressed RGBA8, runtime mip generation, and the per-texture
`vkQueueWaitIdle` in `VulkanTexture::uploadPixels`. Every one of those is real in
the code. None of them is measurable on the scenes this repository ships, because
there is no content to load.

The blocking dependency is therefore content, not code:

- A production-scale glTF scene (Sponza, Bistro, or similar) with real
  multi-megabyte textures, wired into a scene the renderer actually loads.
- Re-run this baseline against it.

Only if that re-baseline shows meaningful load time and VRAM does Phase 1 have a
target. The instrumentation added here is what makes that re-baseline a
one-command operation.

## Limitations

- `--asset-load-stats` covers `VulkanTexture` only. `VulkanEnvironmentMap`
  cubemaps and the BRDF LUT have their own allocations and their own
  `vkQueueWaitIdle` calls; they are visible in the VMA totals but not itemized.
- "glTF import" times parse and mesh buffer upload together, because
  `Mesh::createFromGltf` does both behind one boundary.
- Wall-clock segments only. No GPU timestamps are involved; these are not
  frame-path measurements and must not be quoted as such.
