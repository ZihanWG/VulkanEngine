# Discrete mesh LOD

Geometry carries a discrete level-of-detail chain, and the level is chosen **per
draw item on the GPU**, inside the cull dispatch that already has the bounds and
the camera. No extra pass, no CPU round-trip.

## Why the level can be a GPU decision

Every level of every primitive lives in the mesh's **single index buffer**, with
simplified levels appended after all authored indices. A level is therefore
addressed purely as a `(firstIndex, indexCount)` pair, and switching level is a
change of two fields in the indirect draw command — never a buffer rebind.

That is the whole trick. A LOD scheme that put each level in its own buffer would
force the selection back onto the CPU, because only the CPU can rebind buffers.

## Building the chains

`buildLodChain` (`renderer/MeshLod.{h,cpp}`) runs at mesh load for the built-in
cube and sphere and for every glTF primitive. It is GPU-free so the index
bookkeeping is unit-testable without a device — the same split already used by
`ClusterGrid.h`, `CascadeMath.h`, and `SkeletalAnimation.h`.

- Each level is simplified with `meshopt_simplify` from the **authored** geometry,
  not from the previous level: chaining simplifications compounds error, and
  build-time simplification is cheap enough that there is no reason to pay that.
- Each level is then vertex-cache optimized.
- Level `n` targets `1/2^n` of the authored index count.
- A level is rejected when the simplifier could not remove at least 15% of the
  previous level's triangles, and the chain stops below 32 triangles. Small
  geometry (the 12-triangle cube) correctly yields a **level-0-only chain** rather
  than levels that cost index memory and a table entry while buying nothing.

The portfolio sphere builds `2208 → 1104 → 552 → 275` triangles.

glTF meshes get one chain per primitive, since primitives are drawn independently
and each needs its own range per level. They share the mesh's flat LOD table,
addressed through `MeshPrimitive::lodBase` / `lodCount`.

## Reaching the GPU

`GpuCullDrawItem` addresses its chain with `lodBase` / `lodCount`. Those two words
are the record's **former padding**, so adding GPU LOD selection did not grow the
64-byte record or change its stride.

The table itself is a new SSBO at cull binding 6, rebuilt each frame next to the
cull input (scene edits add and remove meshes) and deduped by mesh, so a mesh's
chain uploads once no matter how many draw items reference it.
`renderer::MeshLod` uploads unchanged — two tightly packed `uint32`s is already
the std430 layout the shader expects — so there is no GPU mirror type to keep in
sync.

The main and shadow cull dispatches **share** the table: levels belong to the
mesh, only the selection bias differs per pass.

## Selecting

`projectedScreenRadius` / `selectLodIndex` in `renderer/MeshLod.h` are the
unit-tested reference; `cull.comp` mirrors them, the same way `ClusterGrid.h`
mirrors `cluster_build.comp`.

```
projectedRadiusPixels = radius / distance * projScaleY
level                 = log2(referenceRadiusPixels / projectedRadiusPixels) + bias
```

`projScaleY = viewportHeight * 0.5 * |proj[1][1]|` is carried in the previously
unused `viewportAndMipCount.w`. Each level covers one halving of the on-screen
radius, which lines up with the chain halving triangle count per level.

> **The `abs()` is load-bearing.** These projections carry the Vulkan Y-flip, so
> `proj[1][1]` is negative (see the ImGuizmo un-flip in `drawViewportGizmo`).
> Passing it through signed makes the shader's `projScaleY <= 0` guard fire on
> every draw item, silently pinning everything to level 0 — which is
> indistinguishable from correct behaviour in a scene where everything is close
> to the camera. This is exactly what the per-level counters were added to catch.

Shadow dispatches set push-constant bit 2 and add `shadowBias` on top of the
shared bias: shadow-map resolution and PCF hide simplification far better than
the main pass does.

Meshes with no chain (`lodCount == 0`) fall through to the authored range carried
on the draw item, so a missing LOD table degrades to full detail rather than an
out-of-range read.

## Debugging

The **Mesh LOD** panel exposes the selection knobs (all of them are just fields of
`GpuCullFrameParams::lodSettings` uploaded next frame), plus:

- **Selected levels** — emitted draws per level, read back from the cull stats
  block. Meshes without a chain are not counted, so the total can sit below the
  visible draw count.
- **Mesh chains** — what each loaded mesh actually built, including an explicit
  *"too small to simplify"* note. This is usually the answer to "why does my scene
  show no level variety".
- **Color by LOD** — green → yellow → orange → red as detail drops, modulated by
  scene luminance so silhouettes and shading still read through the tint.

Settings persist through `config/runtime_settings.json` under `"lod"`.

### How the level reaches the fragment shader

Only the cull shader knows which level it picked, so it packs the level into the
**high bits of `firstInstance`**:

```glsl
command.firstInstance = objectFrameDataIndex | (selectedLod << 16);
```

Draw-item indices stay below 65536, so the high half is always free. The
guarantee is that ceiling, not the current cap: `kMaxDrawItems` is 8192 today and
a `static_assert` pins it at `<= 0xFFFF`, so raising it stays safe until it would
cross 65536. Every vertex shader that indexes object data masks with `0xFFFF`;
`simple.vert` additionally forwards `gl_InstanceIndex >> 16` as a flat varying.
This costs no extra buffer, descriptor, or push constant — only one `AND` in the
vertex stage.

Draws that never go through the cull pass (the skinned demo, the CPU fallback
path) leave the high bits zero and report level 0, which is accurate: they have no
selected level.

## Build cost, and why it is parallel

Simplification is expensive, and deliberately so: every level is simplified from
the *authored* geometry rather than from the previous level, because chaining
simplifications compounds error. That makes an N-level chain N-1 full-geometry
passes. On Sponza, 103 primitives × 3 levels came to **87% of the entire glTF
import time** — 823 of 942 main-thread profile samples, against 3 samples for
JSON parsing.

Primitives simplify independently, so `buildLodChain` is split in two:

- `buildLodChainDetached()` does the expensive part. It reads the source indices
  and the position stream, writes only into its own result, and touches nothing
  shared — safe on a `JobSystem` worker. It also *composes* its log line instead
  of printing it, because `Logger` has no mutex.
- `appendLodChain()` concatenates a build onto the mesh's shared index buffer and
  rebases the level offsets. Trivial, and stays serial.

`Mesh::createFromGltf` enqueues one job per primitive and appends the results **in
primitive order** after the barrier. That ordering is the correctness argument:
only the append decides layout, so the index buffer is byte-identical to the
serial path no matter how the pool scheduled the work. The serial and parallel
paths emit identical LOD chain logs on Sponza, which is how that is checked.

Jobs are enqueued individually rather than through `JobSystem::parallelFor`
because `parallelFor` splits into equal contiguous chunks, and primitives differ
by orders of magnitude in triangle count — a static split leaves the chunk holding
the heavy primitives straggling. Measured spread across seven workers is 42–49
profile samples each.

Result: glTF import **1014.61 ms → 307.39 ms (3.30x)**, main-thread
`meshopt_simplify` frames 823 → 2. See
[asset_load_baseline.md](asset_load_baseline.md).

## Baking the chains: `vemeshcook`

Parallelising construction spread the cost across cores; it did not remove it. A
direct probe -- forcing `kMaxMeshLods = 1` -- shows how much is left:

| Sponza glTF import | median |
| --- | --- |
| with LOD generation | 349.44 ms |
| with simplification disabled | 53.46 ms |
| **simplification** | **~296 ms, 85% of import** |

So the chains are baked offline instead:

```
tools/vemeshcook <scene.gltf> [--force] [--threads N] [--verify]
```

It writes `Sponza.vemesh` beside the source and links `VulkanEngineCore` alone --
parse, vertex assembly and LOD construction all live in Core
(`renderer/GltfGeometry.h`) precisely so the tool never pulls in Vulkan or SDL3.

**Measured, interleaved A/B, five warm runs each:**

| | median | spread |
| --- | --- | --- |
| uncooked | 331.49 ms | 1.3% |
| cooked | **15.50 ms** | 9.4% |

**21x, −316 ms.** Renderer init on Sponza drops to ~129 ms.

The cooked path validates every primitive range, LOD range and index value
against the buffers they address before uploading -- those become indexed
indirect draws and vertex fetches, and a fallback cannot undo an out-of-bounds
fetch that already happened. The index scan is the ~1.5 ms difference from an
earlier unvalidated measurement, and it is worth it.

### The header is what keeps a stale cook from rendering wrong

A cooked KTX2 states its own format, so a stale one is visible. A cooked mesh is
just bytes: if `Vertex` gains a field or a LOD threshold changes, an old file
still parses cleanly and hands back **wrong geometry from a valid-looking
header**. So `renderer/MeshCache.h` records `sizeof(Vertex)`,
`sizeof(MeshPrimitive)`, `sizeof(MeshLod)`, a fingerprint of the
`LodBuildSettings` that produced it, the source glTF's size and write time, **and
a digest of every external buffer it references**. That last one matters because
an ASCII glTF holds no vertex data of its own -- Sponza's lives in `Sponza.bin`,
and fingerprinting only the `.gltf` would call a cook fresh after its geometry
had been replaced.

`meshCacheStatus()` returns a **reason**, not a bool, so a rejection reads as
"the source glTF changed since the cook. Re-run vemeshcook." rather than as an
unexplained slow startup. Every rejection path falls back to loading the glTF and
is never fatal.

**The glTF is still parsed either way.** Parsing costs ~2 ms, so materials,
textures and node transforms are not cooked -- that would add a large
serialization surface and a second staleness surface for nothing.

### Verification, and why it is not the LOD chain log

The check that verified the parallel LOD change -- comparing LOD chain logs --
**cannot work here**: once geometry is cooked the chains are never rebuilt, so
there is nothing to compare against. `vemeshcook --verify` replaces it, reading
the file back off disk and matching it field by field against what was just
built. It earned its place immediately by catching a bug in the tool: the output
stream was still buffered when verification read the file.

## Limitations

- **Transparent draws bypass LOD selection entirely** (see below).
- **Selection is per draw item, not per cluster.** Large meshes switch as a whole,
  so a big object popping between levels is visible at the silhouette. Meshlet-
  granular selection is the direction modern engines went.
- **No cross-fade or dithered transition.** A switch is a hard pop. Screen-space
  dithering between adjacent levels is the usual cheap fix.
- **No screen-space error metric.** Selection uses projected bounding-sphere
  radius, which ignores how much geometric error a given level actually
  introduced. `meshopt_simplify` reports that error and it is currently discarded.
- **Transparent draws bypass LOD selection entirely.** They are issued as direct
  draws to preserve back-to-front sort order (see
  [docs/transparency.md](transparency.md)), so they never pass through the cull
  shader and always render level 0.

## Meshlets, and why meshlet culling was rejected

Geometry can also be grouped into **meshlets** — clusters of at most 64 vertices
and 124 triangles, each with a bounding sphere and a normal cone. `buildMeshlets`
(`renderer/MeshLod.{h,cpp}`) builds them, and `--meshlet-analysis` reports what
culling them would remove. Nothing in the renderer culls them, and the default
build does not even construct them. This section is why.

### The shape it would have taken

Meshlets here are **contiguous ranges of the existing index buffer**, exactly like
LOD levels. `meshopt_buildMeshlets` natively emits meshlet-local vertex and
micro-index arrays, which would need either a second index buffer or a mesh-shader
path to draw; instead the builder *reorders* each level's triangles so a meshlet
is a `(firstIndex, indexCount)` pair. A surviving meshlet would then become an
ordinary `VkDrawIndexedIndirectCommand` and the existing indirect draw path would
not change at all.

That reordering is a permutation of each level's triangles — the level's own range
never moves, so every primitive and LOD record stays valid. It is safe for the same
reason `meshopt_optimizeVertexCache` already is: opaque geometry is depth-resolved,
and transparency is sorted per object rather than per triangle.

### The measurement that stopped it

`--meshlet-analysis` runs the cull that a GPU pass would run — object frustum test
first, so it only ever counts meshlets the existing pass would have kept — and
reports what it would remove. Over 400 deterministic frames per scene:

| scene | draw items | meshlets | triangles removed | indirect commands |
| --- | --- | --- | --- | --- |
| `stress` | 1334 | 3323 | **0.055%** | 1334 → 3321 |
| `default` | 11 | 100 | 19.0% | 11 → 82 |
| `sunlit` | 8 | 34 | 8.5% | 8 → 32 |
| `occlusion` | 126 | 126 | 0% | 126 → 126 |
| `fragment-stress` | 9 | 9 | 0% | 9 → 9 |
| `cornell` | 7 | 7 | 0% | 7 → 7 |

On `stress` — the only scene here with geometry worth culling — meshlet culling
removes **0.055% of triangles while multiplying the indirect command count by
2.5**. The scenes that show a double-digit percentage are the ones with eleven
draw items, where the absolute saving is a few thousand triangles.

### Why it finds so little: LOD already took it

The obvious suspicion is that the meshlets are poor. They are not. Forcing every
draw item to level 0 on the same scene and camera:

| `--scene stress` | triangles drawn | meshlets | removed by meshlet cull |
| --- | --- | --- | --- |
| LOD as shipped | 190,377 | 3,323 | 0.055% |
| `lod.forcedLod = 0` | 1,471,956 | 17,909 | **23.5%** |

The cone culling works — it finds 23.5% when there is dense geometry in front of
it. But **LOD selection and meshlet cone culling compete for the same win, and LOD
gets there first**: it already cut 1,471,956 triangles to 190,377, an 87%
reduction, and meshlet culling then finds 0.055% of the remainder. LOD is 7.7x
more effective on this content, it costs one `selectLodIndex` per draw item inside
a dispatch that already exists, and it needs no second cull stage, no per-meshlet
command budget and no counter readback.

There is a second, quieter reason the leftover is so small. At the levels this
content actually selects, a whole sphere is down to a handful of meshlets, so each
meshlet spans most of the object and its normal cone is too wide to reject
anything — `meshopt` reports a cutoff of 1, the "never reject" value. Meshlet
culling wants many small clusters on a large, dense, near object. LOD's entire job
is to make sure no such object is ever submitted.

### What would flip this

Content with large contiguous meshes that stay near the camera — Sponza is the
obvious one, and `-DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON` fetches it. A single mesh
spanning the screen cannot be LOD'd away, so the leftover after LOD would be much
larger than 0.055%. The analysis is kept, and the builder with it, so that
measurement is a command-line flag rather than a re-implementation.

The capacity ceiling would also have to move first: `stress` at level 0 wants
13,931 indirect commands against a `kMaxDrawItems` of 8192, so a meshlet path needs
its own budget with over-cap geometry **counted rather than dropped**, the way
`FrameCapacityBudget` already does for draw items.

### What is shipped

- `buildMeshlets` and `meshletConeCulled`, GPU-free and unit-tested, including the
  invariant that meshletizing is a permutation of each level's triangles.
- `--meshlet-analysis`, which turns on meshlet construction and the reporting.

Meshlet construction is **off by default** and startup-only, because the triangle
reorder changes rasterization order: on the default scene it moves 7.45% of pixels
by up to 6/255 through z-fight resolution and the exposure feedback that follows.
That is not a bug, but it is a golden re-baseline, and paying one for data no shader
reads would be backwards. `MeshLod` stays 8 bytes for the same reason — the meshlet
ranges come back from `buildMeshlets` beside the table rather than inside the struct
that gets uploaded to the GPU every frame.

### A measurement trap found on the way

The first attempt to price the `MeshLod` widening reported a **10.7% frame-time
regression**, isolated to the stride change, reproducible at 0.43% control drift.
It was an artifact, and the mechanism is worth knowing because it will catch
anything that A/Bs two binaries built from this tree.

`build/<config>/shaders/*.spv` is **shared**. Building binary B recompiles the
shaders in place, so a later run of binary A loads *B's* shaders. When the two
differ in a GPU/CPU mirrored struct, the older binary reads the LOD table at the
wrong stride and draws different geometry — faster or slower, but not the thing
being measured.

The tell is that the same binary read 1.272 ms in one session and 1.406 ms in
another. Running each binary against the shaders it was built with, every
configuration reads **1.406 ms** and the stride costs nothing:

| binary | with its own shaders | with the other's |
| --- | --- | --- |
| 8-byte `MeshLod` | 1.406 ms | 1.272 ms |
| 16-byte `MeshLod` | 1.407 ms | 1.689 ms |

Rebuild the shaders for whichever binary is about to run, or give each side its own
build tree.
