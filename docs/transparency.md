# Transparency: alpha modes and render buckets

The renderer implements the glTF 2.0 alpha model. All three modes are shipped:
`OPAQUE`, `MASK`, and `BLEND` in its own sorted pass. This page covers how each
one reaches the GPU and how the render buckets are laid out.

## Alpha modes

`Material::alphaMode` stores the glTF spelling (`"OPAQUE"` / `"MASK"` /
`"BLEND"`) because that is what the material JSON and glTF import round-trip.
`Material::alphaModeValue()` parses it into the `AlphaMode` enum the render path
actually switches on, and anything unrecognised degrades to `Opaque` rather than
reaching the bucket assignment as garbage.

| Mode | Depth | Bucket | Main pipeline | Shadow pipeline |
| --- | --- | --- | --- | --- |
| `OPAQUE` | write | `Opaque` | `pipeline_` | `shadowPipeline_` (depth-only) |
| `MASK` | write | `Mask` | `pipeline_` (clips) | `maskedShadowPipeline_` (alpha-tested) |
| `BLEND` | test only | `Blend` | `transparentPipeline_` | *(no shadow)* |

## How the alpha test reaches the GPU

There is no dedicated uniform or varying for the cutout. `ObjectFrameData::materialParams.w`
was already reserved for scalar material data and already forwarded to the
fragment stage as the flat `vMaterialParams` varying, so the cutoff rides along
in it:

```
materialParams.w = (alphaMode == MASK) ? alphaCutoff : kNoAlphaTestCutoff   // -1
```

The sign carries the mode. Both fragment shaders then run

```glsl
if (vMaterialParams.w >= 0.0 && alpha < vMaterialParams.w) {
    discard;
}
```

which costs one compare for materials that never clip, adds no bytes to
`ObjectFrameData` (still 688), and adds no varyings. A cutoff of exactly `0.0` is
a legal glTF value and stays distinguishable from "no test at all" — that is why
the sentinel is negative rather than zero (`test_material_alpha.cpp` pins this).

The discard happens immediately after the base-color sample, before any lighting,
shadow, or IBL work, so clipped fragments also stay out of the velocity and thin
G-buffer attachments.

### Vulkan feature requirement

`--target-env=vulkan1.3` emits SPIR-V 1.6, where `OpKill` is deprecated and
glslang lowers GLSL `discard` to `OpDemoteToHelperInvocation`. That capability
needs `VkPhysicalDeviceVulkan13Features::shaderDemoteToHelperInvocation`, so the
device now both requires it in `isDeviceSuitable` and enables it in
`createLogicalDevice`. Without this the validation layer rejects every fragment
shader that clips:

```
vkCreateShaderModule(): SPIR-V Capability DemoteToHelperInvocation was declared,
but one of the following requirements is required (...)
```

## Render buckets

`Renderer::RenderBucket` (`Opaque` / `Mask` / `Blend`) is assigned per draw item
from its material. `buildDrawItems()` sorts by `(bucket, mesh)`:

- **bucket first** so each bucket owns one contiguous range of
  `allDrawItems_` / `visibleDrawItems_`, which reduces the pass and pipeline
  split to a range walk;
- **mesh second** so batching inside a bucket still coalesces vertex/index buffer
  binds exactly as before.

`buildMeshDrawBatchesForItems()` starts a new `MeshDrawBatch` whenever the mesh
*or* the bucket changes, because one batch is one indirect draw with one pipeline
bound. In the default showcase this is visible as the cascade-0 batch count going
from 2 to 3: opaque cubes, opaque spheres, and the masked cutout panel — which
shares the cube mesh with the first batch but cannot merge with it.

Nothing about the *main* pass changes for `MASK`: the alpha test is data-driven,
so opaque and masked geometry share `pipeline_`.

## Alpha-tested shadows

A depth-only shadow pipeline has no fragment stage at all, so a cutout material
throws a solid silhouette — the classic "leaves cast a rectangle" artifact. The
`Mask` bucket therefore swaps to `maskedShadowPipeline_`:

- `shadow_masked.vert` mirrors `shadow.vert` but also forwards UV, the bindless
  base-color index, the cutoff, and `baseColorFactor.a`;
- `shadow_masked.frag` samples the bindless base-color array and discards.

Opaque casters keep the cheaper `shadow.vert`, so the extra varyings are only
paid by materials that need them.

The two pipeline layouts are *not* push-constant compatible (the masked one adds
the bindless descriptor set at set 0), so the recording re-pushes push constants
on every switch. Both the GPU-culled indirect path and the CPU fallback path go
through the same `bindShadowPipelineForBucket` helper.

When the bindless heap is unavailable there is nothing to sample, so `MASK` falls
back to the depth-only pipeline and casts a solid silhouette. The Culling panel
reports this as `Alpha-tested shadow pipeline: unavailable (bindless required)`.

## Demo

`BuiltinTextureFactory::createCutoutLattice` generates a 256×256 sRGB perforated
panel whose alpha channel punches a grid of round holes, with a one-texel ramp at
each hole edge so the cutoff stays meaningful and the mip chain has something sane
to filter. `Portfolio_CutoutLattice` (slot
`kPortfolioCutoutLatticeMaterialIndex`) is `MASK` with cutoff `0.5` and
double-sided, and the showcase places it as **Portfolio Cutout Panel**, angled
into the key light so the perforations read both in the panel and in the shadow it
throws.

The Culling panel shows the bucket split:

```
Render buckets: opaque 8, mask 1, blend 0
Alpha-tested shadow pipeline: active
```

## The transparent pass

`BLEND` geometry is drawn in its own render-graph pass, recorded after SSR and
GTAO — both of which need an opaque-only depth buffer and G-buffer — and before
the TAA resolve, so blended edges still get antialiased.

### Sorting, and why it fights multi-draw indirect

The blend bucket is re-sorted back to front every frame by distance from the
camera to each object's world-bounds centre. Unlike the opaque buckets it cannot
stay grouped by mesh: correct compositing order beats batching, and transparent
draws are few enough for that to be the right trade.

The sort runs in `buildDrawItems()`, on `allDrawItems_`, **not** on
`visibleDrawItems_`. When GPU culling is on, the cull input is marshalled from
`allDrawItems_` while the batches index `visibleDrawItems_`, and the two are only
interchangeable because they share an ordering — sorting one and not the other
silently mismatches every batch's command range. Sorting at the source, before
`frameDataIndex` is handed out, keeps the object-data slots in the final order too.

That leaves the real conflict: **the GPU cull compacts visible commands within a
batch, and compaction is by definition order-destroying.** It would undo the sort
these draws depend on. So the transparent pass issues direct `vkCmdDrawIndexed`
calls instead of replaying the indirect buffer. `firstInstance` still carries the
object-data slot exactly as the indirect path does, so the vertex shader is
unchanged.

### What the cull would have decided is decided on the CPU

Skipping the dispatch's commands means inheriting neither of the two things the
dispatch decides: whether a draw is in view, and which LOD level it draws. For a
long time this page said blended items were still frustum-culled on the CPU. That
was true only of the CPU-culled fallback. With GPU culling on, which is the
default, the visible list is every draw item in the scene, each at its authored
index range -- so the pass drew all of a scene's glass every frame, at full
detail, wherever the camera looked. Nothing showed it until a scene had enough
glass to matter: on Bistro's interior (`--scene bistro-interior`, 1920x1080, the
glTF camera's shot) that was 800 draws and 523,262 triangles a frame, in a pass
that cost more than `MainHDRPass` while shading about half as many fragments.

`Renderer::buildTransparentDrawList` now makes both decisions during frame
preparation, in the order the sort left:

- **Frustum.** The object's world bounds against the camera frustum, the test the
  CPU-culled path already applied. 800 draws become 533 on that shot. It changes
  no pixel, and on its own it moved GPU time by less than a default-clock series
  could resolve -- consistent with the rejected draws having been clipped
  cheaply all along, which is not the same as having measured that.
- **Level.** The rule `cull.comp` applies on the main dispatch, through its C++
  mirror in `renderer/MeshLod.h`, with the same inputs -- render extent, bias,
  error budget, forced level. 333,000 triangles become 194,385. `lod.blended`
  turns it off, which draws every blended item at level 0 as before.

Per frame, on that shot (`--overdraw`, which now reports this pass on its own
line):

| | draws | triangles submitted | fragment shader invocations |
| --- | --- | --- | --- |
| before | 800 | 523,262 | 1,138,121 |
| frustum only (`lod.blended` off) | 533 | 333,000 | 1,138,121 |
| frustum and level | 533 | 194,385 | 1,138,091 |

The fragment count does not move, and the pass gets cheaper anyway: what the
level removes is per-triangle work -- vertex shading, setup, and the quad
overshading of triangles a few pixels across -- which is the same split
[mesh_lod.md](mesh_lod.md#what-lod-is-worth-on-real-content) measured on opaque
geometry. With `lod.blended` off the frame is pixel-identical to the one before
the change; with it on, 1.26% of that frame's pixels differ.

In time, `lod.blended` on against off, interleaved (on, off, on, off) after a
discarded first run -- RTX 3080 Ti Laptop, clocks pinned 800/7001, lights and
animation frozen, every frame sampled for 60 s after a 15 s warm-up, p10:

| | level 0 | level selected | delta |
| --- | --- | --- | --- |
| `Transparent` | 4.094 ms | 3.026 ms | **-1.068 ms (-26.1%)** |
| `Frame total` | 8.662 ms | 7.518 ms | -1.144 ms (-13.2%) |

The repeated control came back within 0.76% on the frame and 0.001 ms on the
pass. `MainHDRPass` read 0.069 ms lower with the selection on, which nothing in
this change reaches -- the opaque pass draws the same commands either way -- so
the frame figure carries that much that is not the change's; the pass figure is
the one to quote. The absolutes are a card held at 800 MHz and are not
comparable with the default-clock tables in the README; the ratios are. Both
sides of the comparison already have the frustum test. Two earlier series at
default clocks had put the saving at the same quarter of the pass and failed the
1% control gate at 3.2% and 6.1%, which is why they are not the ones quoted
([profiling.md](profiling.md)).

Still not decided here, because only depth knows it: occlusion. A glass behind a
wall is submitted, depth-tested against the opaque scene, and rejected fragment
by fragment.

The opaque main pass skips `Blend` batches. The GPU cull still emits commands for
them into the compacted buffer — those slots are simply never replayed.

### Attachments

The pass binds the **same three attachments as the main pass** (scene color,
velocity, thin G-buffer), all loaded rather than cleared. That is deliberate on
two counts:

- The shared fragment shader declares all three outputs. Binding fewer leaves
  declared outputs without attachments, which the validation layer flags.
- Writing velocity is what lets the TAA resolve reproject blended surfaces with
  **their own** motion. Bind only scene color and transparents inherit the
  background's motion vectors and smear whenever they move.

Overwriting the thin G-buffer is harmless: SSR and GTAO are its only readers and
both have already run by this point in the frame, and the main pass rewrites it
next frame.

Only attachment 0 blends. Blending a motion vector or an octahedral normal is
meaningless, so velocity and the G-buffer are plain overwrites. Per-attachment
blend state requires the `independentBlend` device feature, which is now enabled
when supported; without it Vulkan requires every attachment to share one blend
state, so the pipeline falls back to blending all of them — which only degrades
the motion vectors of blended pixels.

### Render-graph modelling

The pass declares scene color as **read-write**, not write-only. The "over" blend
genuinely reads the destination, and declaring it write-only makes the pass culler
treat every earlier write to scene color — the main pass, and SSR's additive blend
— as dead and cull them. That surfaced immediately as
`RenderGraph::beginSsrCopyPass was culled but the renderer attempted to record it`.

`SSRTracePass` had the same modelling gap and now declares `readWriteTexture`
too. Its blend is additive and, since the reflection became an energy-conserving
*correction*, signed -- so it genuinely reads what the main pass wrote rather
than merely landing on top of it.

## Demo

`Portfolio_Glass` (slot `kPortfolioGlassMaterialIndex`) is a `BLEND` material
whose `baseColorFactor.a` is the opacity; the blend pipeline reads it straight out
of the fragment shader's existing alpha output. The showcase places **two**
overlapping panes at different depths on purpose — a single transparent surface
looks correct even with the sort broken, while an overlapping pair immediately
shows whether back-to-front ordering actually holds.

## Known limitations

- **Ambient occlusion bleeds onto transparents.** GTAO is computed from opaque
  depth, and the transparent pass shares the main pass's fragment shader and its
  push constants. So under the default ambient-only path a blended surface
  darkens its own ambient term by the occlusion of whatever opaque surface lies
  behind it, and on the composite reference path the whole blended pixel is
  multiplied after blending. Giving the transparent pass its own AO strength
  would stop the first; it is a look decision and has not been made.
- **No order-independent transparency.** Sorting is per object, not per triangle,
  so intersecting or concave transparent geometry can still composite wrongly.
  Weighted-blended OIT would remove the sort entirely. It is a correctness
  feature rather than a cost one: the pass is bound by its fragments and its
  small triangles, neither of which a different blend removes.
- **A blended draw's level switches outright.** The cross-fade the opaque buckets
  get is a pair of dithered commands written by the cull dispatch, which these
  draws do not use, and a dither is the wrong tool on a surface that is already
  partly transparent. The error budget is the same one pixel, so the switch is
  small; it is not hidden.
- **The two-phase Hi-Z re-test used to submit blended batches, and no longer does.**
  Phase 1 skips them with a comment saying why: the GPU cull emits commands for
  blended geometry into the compacted buffer and those slots are simply never
  replayed, because `recordTransparentPass` draws them later. Phase 2 replays the
  same buffer against the rebuilt pyramid and had no such guard, so a blended
  caster that phase 1 rejected as occluded could be resurrected there -- drawn
  with the opaque main pipeline, depth writes on and no blending, into the scene
  colour the transparent pass is about to composite into.

  It was latent rather than live. Instrumented before the fix, phase 2 was
  submitting 2 blended batches / 146 indirect commands per frame on `--scene
  stress` and 1 batch / 2 commands on `--scene default`, but every one of those
  commands was a zeroed no-op: the phase-2 re-test did not resurrect a blended
  caster at these camera positions. Adding the guard leaves all six presets
  bit-identical, which is what says it was latent. Whether it stays latent is a
  property of the scene and the view, not of the code -- which is the definition
  of the kind of bug worth closing before it is observed.

  Seven of the eight draw loops in `RendererRecord.cpp` already had the guard.
  This was the eighth.

- **The transparent pipeline still forces `VK_CULL_MODE_NONE`, deliberately.**
  `Material::doubleSided` now reaches the cull mode in the main pass -- draw items
  carry the flag, `buildMeshDrawBatches` breaks a run on it, and the recorder binds
  a second pipeline per batch -- so opaque and masked geometry honour it. Blended
  geometry does not: glass and foliage cards are routinely seen from both sides, so
  the transparent pass keeps culling off for every material rather than trusting a
  flag that authoring tools set inconsistently on exactly that kind of surface. The
  shadow pipelines are unchanged for the same reason plus peter-panning.

  Bistro's interior is the measured case for it. None of its blended materials is
  marked double-sided, and culling their back faces halves the pass's fragment
  shader invocations on the startup shot (1,138,121 to 575,147). It also removes
  the body of the wine from every glass, leaving the liquid's top surface
  floating in an outline: the asset relies on both faces. Rejected on the image.
