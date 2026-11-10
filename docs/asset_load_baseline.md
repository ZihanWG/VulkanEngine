# Asset Load Baseline

Phase 0 of the asset cook / async upload initiative. This page records what
startup asset loading costs **today**, so any later claim about a cook step or a
transfer queue has something to be measured against.

It also records why the rest of that initiative was first put on hold -- on the
scenes this repository ships, startup asset loading costs approximately
nothing -- and what happened once an optional production-scale scene removed
that blocker: the texture cook, the mesh cook and the upload work, each with its
before-and-after numbers below.

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
cubemaps plus the BRDF LUT procedurally, and that is what the segment actually
measures.

An earlier version of this page called that "GPU compute at startup". **That was
wrong**, and a profile later corrected it: it is CPU work -- 8.4 million sin/cos/pow
integration steps for the BRDF table alone, on one thread. See
[After parallel IBL precompute](#after-parallel-ibl-precompute) below, which took
it from 166 ms to 60 ms without touching the GPU.

**The glTF import path does not run on the procedural presets.** They are built
from primitive geometry; the only glTF they touch at startup is
`assets/models/skinned_rig.gltf` (3.8 KiB) through `SkinnedMesh::createFromGltf`,
a separate path. Scene import runs only for `--scene sponza`, below.

## With a production-scale scene

`-DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON` downloads Sponza at configure time (see
`cmake/FetchSampleScene.cmake`), and `--scene sponza` selects it (it was the
startup scene when this was first measured; see below). The same one command then
measures a completely different engine:

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

## After the mesh cook

Baking LOD chains and geometry offline (`tools/vemeshcook`, see
[mesh_lod.md](mesh_lod.md)) removes what parallelisation could only spread:

| | uncooked | cooked |
| --- | --- | --- |
| glTF import | 331.49 ms | **15.50 ms (21x)** |
| renderer init (total) | ~466 ms | **~129 ms** |
| scene create | ~441 ms | ~104 ms |

Interleaved A/B, four warm runs each, spreads of 1.3% and 9.4%.

Cooking Sponza takes ~410 ms and writes a 16.54 MiB `.vemesh` -- larger than the
source `.bin` because every simplified LOD level is stored alongside the authored
geometry. Cooked files are gitignored build artifacts, like the `.ktx2` sidecars.

## After parallel IBL precompute

With import and upload dealt with, what was left in scene creation was a fixed
cost every scene pays -- including the empty procedural one. On the default scene,
where `glTF import` and `texture upload` are both 0.00 ms, `scene create` measured
**165.88 ms of pure CPU precompute**, and the Sponza scene paid the same amount on
top of its own work.

A profile attributed it: `VulkanBrdfLut::create` was 75 of the 128 samples in
`createScene` (~59%), and `createProceduralPrefilteredSpecular` another 19 (~15%).
The BRDF table is 256x256 texels at 128 integration samples each -- **8.4 million
steps of sin/cos/pow on one thread** -- and `integrateBrdf` is a pure function of
its two coordinates, so every texel is independent.

Both now run across the `JobSystem`:

| Configuration | scene create | renderer init |
| --- | --- | --- |
| serial (control) | 166.15 ms | ~194 ms |
| BRDF table parallel | 74.37 ms | ~99 ms |
| BRDF + prefiltered specular parallel | **60.38 ms** | **~86 ms** |

Release, Apple M3, medians of three warm runs; the first run of each series is
discarded. **−64% overall.** The control was re-run afterwards and returned
166.15 ms against the original 165.88 ms -- 0.16% drift -- so the gap is the
change, not the machine.

**The output is byte-identical, and that was checked rather than argued.** Unlike
the LOD chains there is no ordering to preserve: each texel writes only its own
bytes and reads nothing another texel wrote. A deterministic capture of the
default scene compares at **0 differing pixels** between the serial and parallel
builds -- and unlike Sponza, that scene really is pixel-reproducible, so the gate
means something here.

The split-sum table depends on nothing but its size and sample count, so it is
identical on every run and every machine. That leaves precomputing it offline as a
future option worth roughly the remaining 6 ms, and makes a GPU compute pass hard
to justify on performance grounds alone.

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

**It is also no longer the startup scene.** It was, whenever the fetch had been
configured on, and that made "the default scene" name two different scenes
depending on a build flag -- which defeats the stamp `docs/profiling.md` requires
every quoted timing to carry. `--scene sponza` selects it; the default scene
renders identically whether or not the fetch ran. Naming the preset on a build
without the asset exits non-zero with the cmake flag in the message, rather than
falling back to a scene that is not the one that was asked for.

The fetched files live in `build/fetched-assets/`, shared by every preset, so
deleting one build directory does not cost a re-download and a re-cook, and the
debug and release builds cannot end up with separately encoded copies of the same
lossy BC7 textures.

## A second imported scene: Bistro

Sponza is one building, lit by one sun and a dozen lamps placed by hand. Amazon
Lumberyard Bistro (CC BY 4.0, from NVIDIA's Open Research Content Archive) is a
street of shopfronts under string lights -- 1296 meshes, 1591 primitives, 2.83
million triangles and 132 materials over 405 textures, against Sponza's 103
primitives and 25 materials.

It is not fetched. It ships only as FBX with DDS textures, so the glTF is a
conversion made on the machine that runs it, by Blender (the only free tool that
reads both):

```sh
blender -b --factory-startup --python tools/convert_bistro.py -- \
    <Bistro_v5_2>/BistroExterior.fbx build/fetched-assets/bistro/Bistro.gltf
./build/release/vemeshcook build/fetched-assets/bistro/Bistro.gltf
python3 tools/cook_textures.py build/fetched-assets/bistro/Bistro.gltf
./build/release/VulkanEngine --scene bistro
```

`--scene bistro` loads it from `VULKAN_ENGINE_BISTRO_SCENE_FILE`, by default the
path above, beside Sponza. A build without the file refuses the preset with a
message naming the path it looked at; the CI sweep does not run it. The conversion
takes about two minutes, the mesh cook three seconds and the texture cook about a
minute, which leaves 1.2 GiB of BC7.

A format conversion alone would not render. What `convert_bistro.py` decides,
rather than copies, is in its header; in short:

- **Materials are rebuilt.** Blender's FBX importer wires the packed "Specular"
  texture into a Principled input glTF cannot express. The scene's README defines
  it as occlusion, roughness and metalness in R, G and B -- glTF's own
  metallic-roughness layout -- so it becomes the metallic-roughness texture
  unchanged. It is not exported as occlusion too: red is zero in every texel of
  all 131, so a viewer reading it as occlusion would light nothing ambiently.
- **Normal maps are DirectX-convention BC5**: green is flipped and Z
  reconstructed.
- **Alpha and sidedness.** Foliage, curtains and lettering are cutouts, glass
  blends, and only foliage stays double-sided: the importer marks every material
  double-sided, which would have put the whole scene on the un-culled pipeline.
- **Lights are derived.** The FBX holds one sun. Its lamps are emissive geometry
  meant for a path tracer, so the conversion places a point light at each emissive
  island -- street lamps, wall spots, shop signs, string-light bulbs merged within
  2.5 m -- coloured by the emissive texture, 128 in all, each with a range written
  into `KHR_lights_punctual`. Their intensities are authored in the script, not
  measured from anything: tuned by eye on the FBX camera's shot, and capped at 20
  for street lamps because most hang off a facade, a few centimetres from a wall
  that inverse-square falloff otherwise blows out.
  Street lamps and wall spots take a warm incandescent colour rather than the
  emissive texture's, whose brightest texels clip to white.
- **The sky comes along.** The HDR the scene was lit under is copied beside the
  glTF as `environment.hdr`.

What the renderer takes from the glTF beyond Sponza's set:

- **`KHR_lights_punctual`.** Point and spot lights go to the clustered path every
  frame, replacing the hand-placed lamps. glTF gives candela, and what that means
  against this renderer's sun is a per-scene choice, so the debug UI scales it
  instead of converting it. A light with no authored range is cut where its
  inverse-square illuminance falls to 0.01, capped at 64 units, since an unbounded
  light would sit in every froxel's list (`importedLightRange`, unit-tested). The
  first directional light sets the sun's direction and colour, not its intensity.
- **An `environment.hdr` beside the glTF** replaces the default environment for
  the skybox and both IBL terms. Sponza ships none and is unchanged. The IBL is
  still precomputed from a 128-texel cube, but the skybox gets its own, a quarter
  of the source's width a face (capped at 1024): Bistro's sky is a 4096x2048
  panorama of a Venetian canal, and wherever the street opens out its buildings
  are the backdrop, which 128 texels a face drew as a smear. One decode feeds
  both cubes, resampled on the job system: 171 ms for the decode and both
  resamples on this machine, and 48 MiB of RGBA16F for the sky.
- **The first perspective camera** frames the startup shot -- Bistro's own street
  view, where the bounds-derived interior camera put the view against a wall.
  Pose and field of view only; the clip planes stay bounds-derived, since
  Blender's 1 mm near plane would waste the depth buffer.

The same contract as Sponza otherwise holds: one RenderObject per primitive, no
portfolio showcase or skinned rig in the shot, and the cooks apply unchanged.

`--scene bistro-night` loads the same files at night. The sun's intensity is
zero, and the sky is the same HDR with its radiance scaled by 0.02 as it is
loaded, so the facades between lamps keep a trace of dusk instead of going black.
That leaves the 128 derived lights and the emissive surfaces to carry the frame,
and auto-exposure settles near 4.6, against 0.89 by day. It is a preset of its own
rather than a toggle because it is a different measurement: the clustered path,
not the sun, decides the frame, and a log line has to be able to say which one it
was. A dark sun skips the cascades outright, and the main pass's sun-shadow
lookup with them (`csm.skipWhenSunDark`; see `docs/design_decisions.md`). Until
it did, night measured the same as day: the cascade cache hid the draws on a
still camera, and the lookup cost what the lamps saved. With every cascade
redrawn, as a moving camera makes them, the skip takes this scene's frame from
4.079 to 2.612 ms at 1280x720; in the README's 1080p series, night now runs
about a fifth cheaper than day.

Measured frame time for all three scenes is in the README's Performance section.
The series was, per scene and run:

```sh
python3 tools/dev/measure_gpu.py run --label bistro-1 --warmup 15 --duration 75     --every-frame --freeze-animation --args --scene bistro --window-size 1920x1080
```

### The interior

`--scene bistro-interior` is the restaurant, from the archive's other FBX
(`BistroInterior_Wine.fbx`, the revision with filled glasses), converted by the
same script to `build/fetched-assets/bistro-interior/BistroInterior.gltf`
(`VULKAN_ENGINE_BISTRO_INTERIOR_SCENE_FILE`) and cooked the same way: 1415 meshes,
74 materials, 1.29M triangles. The script picks the scene from the FBX's file name
and differs in two places:

- **Its own lamp table.** The FBX holds no lights either. Twenty-two are derived:
  eight ceiling fixtures (24 emissive pieces merged within 0.75 m), six wall
  sconces (84 pieces), six wall bulbs, a bare bulb and a wall lamp, all warm, at
  4-15 cd. Auto exposure takes up most of a change in them: doubling or tripling the
  ceiling fixtures barely moved the frame, so the values set the lamps' balance
  against the sky rather than its brightness.
- **The drinks keep their factors.** Wine, beer, water and ice are base colour and
  opacity factors with no texture, and the bare bulb's glow is an emission factor;
  rebuilt from textures alone they came out opaque grey. For the interior the
  script keeps a material's own factors where no texture replaces them, so the
  drinks blend. The exterior's conversion is unchanged.

It is lit at dusk, like the street at night: no sun, and the canal sky dimmed by
the same 0.02, because the IBL is unshadowed and at full strength it floods a room
lit through its windows.

On the FBX camera's shot at 1920x1080, same protocol as above (two runs after a
discarded one): `Frame total` 5.213 / 5.143 ms, of which **`Transparent` 2.447 /
2.456 ms** and `MainHDRPass` 1.683 / 1.628 ms; 1,415 of 2,247 draw items visible,
434,708 triangles drawn, CPU frame preparation 0.99 ms. It is the one scene here
whose transparent pass costs more than its opaque one: some three hundred glasses
and bottles, drawn one at a time back to front (the GPU cull's compaction would
destroy the order, see `transparency.md`), each layer shaded in full. The card
ended the series at 73 °C.

Those figures predate the level selection for blended draws. The pass was
drawing every blended item in the scene at level 0 -- 800 draws, not the three
hundred in view -- and selecting a level for them took 26.1% off it in a pinned
series of its own (`transparency.md`). It still costs more than the opaque pass.

## Limitations

- `--asset-load-stats` covers `VulkanTexture` only. `VulkanEnvironmentMap`
  cubemaps and the BRDF LUT have their own allocations and their own
  `vkQueueWaitIdle` calls; they are visible in the VMA totals but not itemized.
- Its depth complexity is **2.187 shaded fragments per rendered pixel**
  (`--overdraw`), the highest of any preset here. That was the reason the scene
  was wanted; see `design_decisions.md` on the depth prepass.
- "glTF import" times parse and mesh buffer upload together, because
  `Mesh::createFromGltf` does both behind one boundary.
- Wall-clock segments only. No GPU timestamps are involved; these are not
  frame-path measurements and must not be quoted as such.
