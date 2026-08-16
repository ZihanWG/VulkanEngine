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

## With a production-scale scene

`-DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON` downloads Sponza at configure time (see
`cmake/FetchSampleScene.cmake`) and makes it the startup scene. The same one
command then measures a completely different engine:

| | procedural default | Sponza |
| --- | --- | --- |
| renderer init | 191.30 ms | **7105.61 ms** |
| — scene create | 168.07 ms | 7057.06 ms |
| — — glTF import | 0.00 ms | **3944.06 ms** |
| — — texture decode wait | 0.00 ms | 104.72 ms |
| — — texture upload | 0.00 ms | **141.12 ms** |
| textures | 8 | **77** |
| texture device bytes | 0.45 MiB | **365.98 MiB** |
| block compressed | 0 of 8 | **0 of 77** |
| device-local used | 493.68 MiB | **877.68 MiB** |

Every pathology the asset-cook work was proposed to fix is now visible and
measurable:

- **365.98 MiB of uncompressed RGBA8**, none block-compressed. BC7/BC5 would cut
  that by roughly four.
- **3.9 seconds of glTF import**, which is tinygltf parsing plus mesh buffer
  upload behind one boundary. Cooking meshes offline removes most of it.
- **141 ms of texture upload**, serial by construction: every
  `VulkanTexture::uploadPixels` ends in `vkQueueWaitIdle`. This is the transfer
  queue's target.
- **A seven-second startup.**

## Consequence for the initiative

The earlier version of this page concluded that the asset cook and async upload
work was blocked on content rather than code, because on the shipped scenes there
was nothing to optimise. **That blocker is now removable at will**: turning the
fetch on produces a scene where all four costs above are real.

The fetch stays off by default so CI stays fast and the committed golden image
keeps comparing against the small deterministic scene it was captured from. It is
a measurement and screenshot facility, not a change to what the engine renders by
default.

## Limitations

- The fetched scene is one glTF node holding one mesh of 103 primitives, so it
  presents as a **single render object**. It exercises materials, textures, and
  draw submission at scale, but not object-level frustum culling, which has one
  thing to cull. The camera is still the portfolio preset, framed for the sphere
  showcase rather than for Sponza.
- `--asset-load-stats` covers `VulkanTexture` only. `VulkanEnvironmentMap`
  cubemaps and the BRDF LUT have their own allocations and their own
  `vkQueueWaitIdle` calls; they are visible in the VMA totals but not itemized.
- "glTF import" times parse and mesh buffer upload together, because
  `Mesh::createFromGltf` does both behind one boundary.
- Wall-clock segments only. No GPU timestamps are involved; these are not
  frame-path measurements and must not be quoted as such.
