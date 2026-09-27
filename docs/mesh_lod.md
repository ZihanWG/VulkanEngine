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
- Each level records its **geometric error**: how far its surface strays from the
  authored one, in the mesh's own units. `meshopt_simplify` reports it relative
  to the extent of the position stream it was handed -- for a glTF primitive, the
  whole mesh's shared vertex buffer -- so it is multiplied back by
  `meshopt_simplifyScale` of that same stream. Levels are simplified from the
  authored geometry independently, so nothing forces a coarser level to measure
  worse; the builder raises any level that measured better to its predecessor's
  error, which is what lets selection treat the chain as ordered.

The portfolio sphere (radius 0.5) builds `2208 → 1103 → 552 → 275` triangles at
errors of `0.0059 → 0.0081 → 0.030`, about 1.2%, 1.6% and 6% of its radius. The
chain log prints each level's error beside its triangle count.

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
`renderer::MeshLod` uploads unchanged -- `firstIndex`, `indexCount` and the
level's `error`, three 4-byte scalars, which std430 lays out at a 12-byte stride
because it aligns a struct of scalars to 4 -- so there is no GPU mirror type to
keep in sync. `static_assert`s pin the offsets and the shader interface golden
pins the GLSL side.

The main and shadow cull dispatches **share** the table: levels belong to the
mesh, only the selection bias differs per pass.

## Selecting

Two rules, one setting apart. **By default the level comes from each level's
projected geometric error** ([below](#selecting-by-screen-space-error-the-default));
`lod.screenSpaceError = false` selects by projected radius instead, which is the
rule described first here and the one the engine shipped with.

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

### Selecting by screen-space error (the default)

The radius rule steps one level per halving of the on-screen radius, whatever each
level actually cost in accuracy: a level that barely moved the surface and one
that visibly dented it switch at the same distance. `lod.screenSpaceError` asks the
question directly -- how many pixels would this level be wrong by, from here --
and takes the coarsest level that stays inside `lod.maxErrorPixels` (default 1):

```
worldError  = level.error * maxAxisScale(model)
errorPixels = worldError / distanceToBounds * projScaleY
level       = the last level with errorPixels <= maxErrorPixels * 2^bias
```

- **Scale.** An object-space error becomes a world-space one through the model
  matrix's longest basis column, which bounds the stretch in any direction. It
  rides in `GpuCullDrawItem::boundsMin.w`, which nothing else read, so the record
  keeps its 64 bytes.
- **Distance.** From the camera to the nearest point of the draw item's world
  AABB, not to its centre. Every triangle lies in the box, so no part of the
  object can be nearer, and the error projected there bounds the error anywhere
  on it. Inside the box the distance is 0: only a level with zero error -- one
  whose collapses were coplanar -- is taken, which is right, because it is exact.
- **Bias.** `bias` and `shadowBias` scale the budget by `2^bias` instead of adding
  levels, so one unit still means roughly one level coarser: each level roughly
  doubles its error.
- **One reference, three callers.** `selectLodIndexByError` in
  `renderer/MeshLod.h` is the unit-tested reference; `cull.comp` mirrors it for
  the main and shadow dispatches, and `Renderer::selectedShadowLodLevel` calls it
  directly for the cascade cache's caster hash and the meshlet analysis. With the
  rule on, the default and sunlit scenes render bit-identically with the cascade
  cache on and off, so the CPU hash and the GPU choice agree.

What it does, on the RTX 3080 Ti Laptop at 1280x720, `--deterministic`, 240
frames, counted by the cull pass itself (the `emitted triangles` line):

| scene | radius rule | error ≤ 1 px | |
| --- | --- | --- | --- |
| default | 7,797 | 5,590 | -28% |
| `--scene sunlit` | 2,278 | 1,176 | -48% |
| `--scene stress` | 94,232 | 94,232 | 0 |
| `--scene sponza` | 185,889 | 137,669 | -26% |

On Sponza the budget trades as expected: 157,978 triangles at 0.5 px, 115,090 at
2 px. `stress` does not move: every chained draw it emits sits at level 0 or
level 3 under both rules (336 and 328), and the two rules agree on which.

And in frame time, on Sponza at 1280x720, RTX 3080 Ti Laptop with clocks pinned
at 800/7001 MHz, p10 over 128 samples a side, interleaved A/B/A/B after a
throwaway run:

| `--scene sponza` | radius rule | error ≤ 1 px | |
| --- | --- | --- | --- |
| Frame total | 6.537 ms | 5.528 ms | **-15.4%** |
| `MainHDRPass` | 4.583 ms | 3.868 ms | **-15.6%** |
| `MainGpuCullingPass` | 0.015 ms | 0.017 ms | +0.002 ms |

The control came back within 0.11%. An earlier series that failed the gate at
3.9% -- its first control run read high after the machine had idled -- showed the
same effect, -16.1% on the frame, so the saving reproduces; it is the triangle
count doing it, and the selection loop itself costs the cull pass about 2 µs.

**It is the default, and this is what it changes in the image.** The error is
*geometric*. It bounds where the surface is, not how it shades: normals
interpolated across coarser triangles move a sharp specular highlight further
than the silhouette moves. On the default scene the rule changes 1.7% of pixels
against the radius rule -- thin rings at sphere silhouettes, and the reflected
highlights inside the chrome sphere -- and making it the default re-baselined the
lavapipe golden for the same reason. `lod.screenSpaceError = false` restores the
radius rule, and the `lod-radius-rule` sweep leg keeps it running in CI.

## Cross-faded transitions

A level switch used to be a pop: one frame drew level *n*, the next level *n+1*,
and the silhouette jumped. It now cross-fades over `lod.transitionSeconds`
(default 0.25 s; 0 pops): for those frames **both** levels are drawn, each
discarding a complementary half of its pixels by a per-pixel noise threshold, and
the threshold sweeps from one level to the other. Every surviving fragment is
opaque and depth-tested as usual, so nothing is blended or sorted.

**The state lives on the GPU, per draw item.** Only the cull pass knows which
level it picked, so it keeps one 16-byte record per draw item in a persistent
buffer (cull binding 7) -- current and outgoing level, fade progress, the frame it
was last advanced, and an identity -- and advances it with the reference in
`renderer/LodTransition.h` each time it emits the item. The buffer is one
allocation, not one per frame slot, because a fade is a run of consecutive frames:
each frame's cull reads what the previous one wrote. A compute-to-compute barrier
before each main cull dispatch (and before phase 2, which writes the records of
what it rescues) makes those writes visible; queue submission order already puts
them first. It is zeroed once, before its first use.

Three rules keep the record honest:

- **Identity.** Draw items are renumbered whenever the scene changes, so the record
  carries a hash of the object slot and the authored index range. A mismatch
  snaps: a record must never fade a new object out of somebody else's level.
- **Continuity.** An item culled for a while comes back at whatever level suits it
  now; fading in from a level last seen seconds ago would be a transition nobody
  watched start. A record not advanced on the previous frame snaps.
- **Reversal.** A camera that turns back mid-fade selects the outgoing level
  again; the fade then runs backwards from where it stands rather than restarting,
  so the turn does not pop either. A third level arriving mid-fade fades out of
  whichever level dominates the screen at that moment.

**Two command slots per draw item.** A fading item emits two indirect commands,
so on the GPU-cull path every batch's region in the main indirect buffer is sized
at two slots per draw item (`kLodCommandSlotsPerDrawItem`) and the buffer doubled
to match; the compacted path consumes the second slot only while fading, the
fixed-slot path writes an empty command into it. The CPU fallback still writes
one slot per item, and the shadow dispatches -- which never fade -- keep their
own one-slot layout. The fade reaches the fragment shader in the high half of
`firstInstance`, next to the level: a fading flag, an outgoing flag and a 7-bit
fade. Seven, not eight, because `gl_InstanceIndex` is a signed int and a fade in
bit 31 would turn it negative.

**What the passes do with it.**

- The main fragment shader discards its half first, before any texture work.
  That is safe ahead of implicit-LOD sampling because `discard` compiles to
  `OpDemoteToHelperInvocation` here, so a quad's derivatives survive it.
- The depth prepass leaves fading draws out -- their vertices land outside the
  clip volume -- because a full-coverage prepass depth would let the main pass
  keep neither half where the two levels' outlines disagree. The main pass,
  which tests `LESS_OR_EQUAL` and writes depth, lays down theirs.
- Shadows switch outright. Shadow-map resolution and filtering hide the pop far
  better than the main view does, and fading them would double the casters'
  work for nothing visible.

**What it does.** With a still camera nothing fades: the default scene renders
bit-identically to before the change. Under `--camera-orbit` (below), the
once-a-second report counts the draw items mid-fade -- up to 7 on `--scene stress`
at 0.01-0.03 rad a frame, 14 at 0.05 -- and synchronization validation reports
no hazards with fades in flight. On the default scene orbiting at 0.03 rad a
frame, fading changes 0.69% of frame 30 against the same run with fades off, all
of it on the two spheres that were switching and almost all on their silhouettes;
up close the edge is a stipple of both outlines, and the interior is unbroken.

**What it costs.** A fade only exists while the camera moves, so it is measured
orbiting: RTX 3080 Ti Laptop with clocks pinned at 800/7001 MHz, `--scene sponza
--camera-orbit 0.01` at 1280x720, fades off (`transitionSeconds 0`) against on,
p10 over 128 samples a side, interleaved A/B/A/B after a throwaway run. The orbit
swings through the whole view mix around the preset's target, whose frames run
about 1.5 ms -- cheaper than the preset's own view -- and the once-a-second report
counts how many draw items are mid-fade in each run.

| fade length | draw items mid-fade | `MainHDRPass` | frame total |
| --- | --- | --- | --- |
| 0.25 s (default) | 5.1 on average, 12 at most | +0.004 / +0.005 ms, inside its drift | +0.028 / -0.027 ms |
| 2.0 s | 7.4 on average, 14 at most | **+0.078 / +0.065 ms (+22% / +19%)** | +0.063 ms (+4.1%) |

Each cell is two independent series. **At the default length the cost does not
resolve**: both frame-level series passed the drift gate (0.20%, 0.44%) and still
disagreed in sign, so a moving camera puts the floor around 0.03 ms, and the fade
sits under it. Stretching every fade eightfold does resolve: `MainHDRPass` rose in
both series with its own drift at 0.003 ms or less, and the frame total in the one
series whose control came back (0.96%; the other drifted 4.1% and is not quoted).
The cost is where the design puts it -- a fading draw is drawn twice and its
fragments are not pre-culled by the depth prepass -- and it does not scale from
one row to the other by fade count alone, because long fades on a fast orbit
catch large, near objects that short ones do not.

**Seeing it headlessly.** Live input is dropped in a `--deterministic` run, so
`--camera-orbit R` yaws the camera around its target by `R` radians a frame
instead -- the one camera motion a script can have, reproducible from the frame
number. The `lod-crossfade-orbit` sweep leg runs `--scene stress --camera-orbit
0.05` under validation in CI.

## Debugging

The **Mesh LOD** panel exposes the selection knobs (all of them are just fields of
`GpuCullFrameParams::lodSettings` uploaded next frame), plus:

- **Selected levels** — emitted draws per level, read back from the cull stats
  block. Meshes without a chain are not counted, so the total can sit below the
  visible draw count.
- **Mesh chains** — what each loaded mesh actually built, with each simplified
  level's error in mesh units, and an explicit *"too small to simplify"* note.
  This is usually the answer to "why does my scene show no level variety".
- **Emitted triangles** — in the GPU culling block of the once-a-second report:
  the triangles of every draw the main cull emitted, at the level it chose (both
  levels, for a draw mid cross-fade). It is the number two selection rules are
  compared on, and needs no timer.
- **LOD cross-fades** — in the same block: draw items that were mid-fade, each
  drawn at two levels that frame.
- **Color by LOD** — green → yellow → orange → red as detail drops, modulated by
  scene luminance so silhouettes and shading still read through the tint.

Settings persist through `config/runtime_settings.json` under `"lod"`.

### How the level reaches the fragment shader

Only the cull shader knows which level it picked, so it packs the level into the
**high bits of `firstInstance`**:

```glsl
command.firstInstance = objectFrameDataIndex | (instanceHigh << 16);
```

`instanceHigh` is the level in its low four bits, plus the cross-fade flags and
fade described [above](#cross-faded-transitions); `lod_transition.glsl` lays it
out and the heatmap masks the level back out of it.

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

## What LOD is worth on real content

Measured on an RTX 3080 Ti against `--scene sponza` at 1280x720, clocks pinned
700/7001, p10, forcing every draw to one level so the comparison has no second
variable:

| | forced LOD 0 | forced LOD 3 | delta |
| --- | --- | --- | --- |
| full resolution | 4.920 ms | 2.357 ms | **-2.563 ms (-52.1%)** |
| a sixteenth of the pixels | 1.853 ms | 0.653 ms | -1.200 ms (-64.8%) |

**Level selection is the largest single lever on `MainHDRPass` on this scene**,
and roughly half of what it buys is per-triangle work that no resolution change
can reach. The second row is what separates the two halves: vertex shading and
triangle setup do not care about resolution, and quad overshading cares about
nothing else. See [profiling.md](profiling.md) for the full attribution and for
why the "fixed third" that earlier framing named does not exist.

That is a bound on what a more aggressive `lod.bias` or a looser budget could
win, not a recommendation to spend it: the 2.563 ms is paid for in silhouettes.
Selecting by error at 1 px is what the default spends of it (see above).

## Limitations

- **Transparent draws bypass LOD selection entirely** (see below).
- **Selection is per draw item, not per cluster.** Large meshes switch as a whole,
  so a big object popping between levels is visible at the silhouette. Meshlet-
  granular selection is the direction modern engines went.
- **A cross-fade is a stipple.** The dither is the whole transition: for
  `lod.transitionSeconds` the silhouette shows both outlines as a fine pattern.
  The noise is fixed per pixel, so it does not crawl -- and for the same reason
  TAA does not average it away either.
- **Shadow maps still pop**, and a fading draw pays full main-pass cost for its
  fade, since it draws twice and is left out of the depth prepass. At the default
  length that cost is below what a moving-camera measurement resolves; with
  eightfold fades it is a fifth of `MainHDRPass` (see above).
- **Screen-space error selection is geometric.** It bounds where the surface is,
  not how it shades; an attribute-aware error (`meshopt_simplifyWithAttributes`
  over normals) would bound highlights too, and would change the chains
  themselves.
- **The error projection is conservative.** It uses the object's largest axis
  scale and its nearest bounds point for the whole object, so a long object seen
  end-on keeps the detail its nearest end needs.
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

**That re-measurement is now runnable**: the scene is `--scene sponza`, so
`--meshlet-analysis --scene sponza` is the whole setup. One thing to know before
reading its output. Sponza imports as one object per primitive; it used to import
as one object for the entire file, and because a draw item's LOD is chosen from
its object's projected size, that meant all 103 primitives were measured against
the bounds of the whole building and every one of them selected level 0. An
analysis run before that change would have compared meshlet culling against a
scene with LOD effectively disabled, which is the comparison this page says is
the wrong one — turning LOD off is exactly how the 0.055% became 23.5% on
`stress`.

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
reads would be backwards. `MeshLod` carries no meshlet data for the same reason --
the meshlet ranges come back from `buildMeshlets` beside the table rather than
inside the struct that gets uploaded to the GPU every frame.

### A measurement trap found on the way

(`MeshLod` has since grown to 12 bytes, for the per-level error that screen-space
selection reads. The trap below is unchanged.)

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
