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

## After the texture cook

P1 landed, so the Sponza column above is now the *uncooked* control rather than
the only measurement. Cooking every texture Sponza references
(`tools/cook_textures.py`, 69 cooks, 5.2 s wall) and re-running the same binary
on the same scene:

| Segment | Uncooked | Cooked | Delta |
| --- | --- | --- | --- |
| renderer init (total) | 1379.51 ms | 1300.20 ms | −5.7% |
| — — glTF import | 1037.42 ms | 1053.93 ms | unchanged |
| — — texture decode wait | 58.88 ms | **0.00 ms** | no decode happens |
| — — texture upload | 89.90 ms | **48.70 ms** | −46% |
| texture device bytes | 365.98 MiB | **91.06 MiB** | **−274.92 MiB, 4.02x** |
| block compressed | 0 of 77 | **72 of 77** | |
| device-local used | 877.68 MiB | **621.68 MiB** | −256.00 MiB |

Release build, Apple M3, same scene and camera, A/B/A with the control repeated:
three uncooked runs and two cooked ones. The control returned to 1379.51 ms init
and 92.34 ms upload on its third run, inside the project's 1% / 3% spread, so the
gaps above are the cook and not drift. Byte counts are deterministic and identical
across every run of a configuration.

**4.02x is the honest whole-chain number.** Both columns include mip chains — the
uncooked path generates them with `vkCmdBlitImage`, the cooked path ships them
baked — so this is not the same figure as `vecook`'s per-file "3.00x with mips",
which compares a cooked chain against an uncompressed *base level* only.

Decode wait going to exactly zero is categorical, not a speedup: a cooked texture
is never handed to stb_image, so the JobSystem decode is not dispatched at all.
The 5 of 77 textures that stay uncompressed are the procedurally generated ones
(fallbacks, the backdrop gradient, the cutout lattice), which have no source file
to cook from.

## After batching texture uploads

With glTF import parallelised, texture upload was the next measurable item at
~51 ms. A profile split it as roughly half `vkQueueWaitIdle` — every texture
submitted its own command buffer and then **drained the whole queue** before the
next one started, 69 times — and about a third reading the cooked KTX2 files one
at a time on the device thread.

`rhi::VulkanUploadBatch` records many textures into one command buffer and waits
on a fence, and the file reads moved to the `JobSystem`:

| | serial | batched | |
| --- | --- | --- | --- |
| texture upload | 50.11 / 52.07 ms | **14.42 / 15.37 ms** | **−71%** |
| submits for 69 textures | 69 | **2** | |
| texture device bytes | 91.06 MiB | 91.06 MiB | unchanged |
| device-local used | 621.68 MiB | 621.68 MiB | unchanged |

Release, Apple M3, warm runs; the first run of any series is discarded as
cold-file-cache warm-up (it measures 100-150 ms in both configurations).

The batch flushes when retained staging crosses a 64 MiB budget, so peak staging
does not scale with the scene: Sponza's 91 MiB of cooked texture data uploads in
2 submits with **62 MiB peak staging**, and the uncompressed path would simply
flush more often rather than spike. The prefetch window is bounded by the same
budget, so file bytes and staging together stay near 128 MiB transient host
memory instead of the whole scene.

Re-profiling confirms the mechanism: `vkQueueWaitIdle` no longer appears anywhere
under `createFromKtx2`. The call sites that remain are mesh buffer uploads
(`VulkanBuffer::copyBuffer`), the BRDF LUT, and the non-batched builtin textures —
all deliberately unchanged.

Verified with validation layers **on the Sponza path specifically**, not just the
default scene: batching means many textures share one command buffer and one
fence, and a staging buffer freed before its copy ran is exactly what the layers
catch. 0 errors, 0 warnings.

## After parallel LOD construction

The texture cook did not touch glTF import, which then became the largest startup
cost at ~1035 ms. A sampling profile (`sample -wait`) found that **87% of it was
`meshopt_simplify`** — 823 of 942 main-thread samples inside
`Mesh::createFromGltf` — while tinygltf's JSON parse was **3 samples**. Every LOD
level is simplified from the authored geometry rather than the previous one
(`MeshLod.h`), so Sponza's 103 primitives cost 309 full-geometry simplifications,
all on one thread.

Running them across the `JobSystem`:

| Segment | Serial | Parallel | Delta |
| --- | --- | --- | --- |
| glTF import | 1014.61 ms | **307.39 ms** | **3.30x, −707 ms** |
| renderer init (total) | ~1270 ms | **~534 ms** | −58% |

Release, Apple M3 (7 pool workers), medians of three runs each. The two
configurations differ by one argument at the call site, so this is as close to a
controlled A/B as the code allows; an independent pre-change series measured
1036.92 ms, agreeing with the serial column within 2%.

Re-profiling confirms the mechanism rather than inferring it from the clock:
main-thread `meshopt_simplify` frames went from **823 to 2**, and the seven
workers carry 42–49 frames each — an even spread, which is why the jobs are
enqueued per primitive instead of through `parallelFor`'s equal contiguous
chunks (primitives differ by orders of magnitude in triangle count).

**The geometry is unchanged, and that was checked rather than assumed.** The
serial and parallel paths emit byte-identical LOD chain logs — 84 chains, same
order, same per-level triangle counts — because only the serial append decides
index-buffer layout. A pixel comparison is *not* a usable gate here: this scene
has pre-existing capture nondeterminism of exactly one pixel at (816, 850) with a
channel delta of 37, reproducible between runs of the *same* binary.

**Do not compare these absolute timings against the uncooked column further up.**
That column was captured in an earlier session and reports 7105.61 ms init and
3944.06 ms glTF import for byte-for-byte the same scene — almost certainly a cold
file cache on the run right after the fetch. The texture counts and byte totals
agree exactly, so it is the same content; only the wall clock disagrees. Compare
within a series, never across sessions.

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
  thing to cull.
- The startup camera is framed from the scene's bounds
  (`renderer::framedCamera`), which guarantees the scene is on screen but does
  not compose a shot. For an interior scene that means viewing the building from
  outside. Placing the camera inside from bounds alone was tried and abandoned:
  Sponza's bounds include its own outer walls, so an inset from the edge lands in
  masonry. Use the editor camera to compose a screenshot.
- `--asset-load-stats` covers `VulkanTexture` only. `VulkanEnvironmentMap`
  cubemaps and the BRDF LUT have their own allocations and their own
  `vkQueueWaitIdle` calls; they are visible in the VMA totals but not itemized.
- "glTF import" times parse and mesh buffer upload together, because
  `Mesh::createFromGltf` does both behind one boundary.
- Wall-clock segments only. No GPU timestamps are involved; these are not
  frame-path measurements and must not be quoted as such.
