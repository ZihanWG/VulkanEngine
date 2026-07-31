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
| `BLEND` | test only | `Blend` | *(pending — see below)* | *(no shadow)* |

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

## What `BLEND` still needs

The bucket, the sort key, the batch split, and the shadow-pipeline switch are all
in place and already treat `Blend` as its own contiguous range. Still outstanding:

1. A per-frame back-to-front depth sort of the blend range (the bucket sort is
   stable, so this slots in as a secondary key).
2. A transparent pass — scene color only, depth test on, depth write off —
   recorded after the GTAO blur and before the TAA resolve.
3. An alpha-blend pipeline (`SRC_ALPHA` / `ONE_MINUS_SRC_ALPHA`).
4. Forcing the blend range onto the **non-compacted** indirect path, so command
   slot order still equals the sorted draw order. Compaction reorders within a
   batch, which would destroy the back-to-front ordering.
5. Wiring `Material::doubleSided` into the pipeline cull mode, which needs the
   batch split to also break on double-sidedness.

Known limitation to document when that lands: transparent fragments do not write
velocity, so TAA reprojects them using the background's motion vectors.
