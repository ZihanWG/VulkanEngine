# Transparency: alpha modes and render buckets

The renderer implements the glTF 2.0 alpha model. This page covers what is
shipped today (`OPAQUE` and `MASK`) and how the bucket infrastructure is laid out
for the `BLEND` pass.

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
unchanged. Blended items are still CPU frustum-culled; they just skip the GPU
culling round trip.

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

Worth noting that `SSRTracePass` still declares its own additive blend into scene
color with `writeTexture`, which has the same modelling gap. It is currently
harmless only because nothing wrote scene color after it until this pass existed.

## Demo

`Portfolio_Glass` (slot `kPortfolioGlassMaterialIndex`) is a `BLEND` material
whose `baseColorFactor.a` is the opacity; the blend pipeline reads it straight out
of the fragment shader's existing alpha output. The showcase places **two**
overlapping panes at different depths on purpose — a single transparent surface
looks correct even with the sort broken, while an overlapping pair immediately
shows whether back-to-front ordering actually holds.

## Known limitations

- **Ambient occlusion bleeds onto transparents.** The composite multiplies the
  GTAO term into scene color after the transparent pass has already blended into
  it, so blended pixels are darkened by the AO of whatever opaque surface is
  behind them. Fixing it means applying AO before the transparent pass, which
  would also change how AO interacts with bloom — a deliberate deferral rather
  than an oversight.
- **No order-independent transparency.** Sorting is per object, not per triangle,
  so intersecting or concave transparent geometry can still composite wrongly.
  Weighted-blended OIT would remove the sort entirely and is the natural follow-up.
- **`Material::doubleSided` is not yet wired to the pipeline cull mode.** Both the
  transparent and masked pipelines currently force `VK_CULL_MODE_NONE`, which is
  right for the demo materials but ignores the flag for single-sided ones. Honouring
  it needs the batch split to also break on double-sidedness.
