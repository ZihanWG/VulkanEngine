# Asset Manager And Material Assets

Phase 3 adds a small data-driven asset layer without changing the renderer's
Vulkan ownership model.

## Scope

`src/assets/AssetManager.h/.cpp` provides:

- stable path-derived `AssetHandle` values
- typed `MaterialAssetHandle` and `TextureAssetHandle`
- material JSON load/save
- texture asset path registration as metadata

The asset manager does not own Vulkan images, image views, samplers,
descriptors, mesh buffers, or shader permutations. `Renderer` and
`VulkanTexture` still own runtime GPU resources.

Asset path keys are normalized lexically and stored with generic separators.
The renderer feeds project-resolved paths for material assets and resolved
texture paths for loaded texture metadata. The current `AssetManager` does not
canonicalize symlinks, case-fold Windows paths, or merge arbitrary
relative-vs-absolute spellings of the same file, so callers should pass
project-resolved paths consistently when handle identity matters.

## Material JSON Schema

Material assets live under `assets/materials/` and use the
`.material.json` suffix:

```json
{
  "name": "Portfolio_HeroCeramic",
  "shader": "pbr_opaque",
  "baseColorFactor": [0.66, 0.72, 0.76, 1.0],
  "metallicFactor": 0.0,
  "roughnessFactor": 0.55,
  "textures": {
    "baseColor": "",
    "normal": "",
    "metallicRoughness": ""
  },
  "alphaMode": "OPAQUE",
  "alphaCutoff": 0.5,
  "doubleSided": false
}
```

Supported fields are `name`, `shader`, `baseColorFactor`,
`metallicFactor`, `roughnessFactor`, `textures.baseColor`,
`textures.normal`, `textures.metallicRoughness`, `alphaMode`,
`alphaCutoff`, and `doubleSided`.

`alphaMode` and `alphaCutoff` now drive real rendering: `MASK` materials clip in
the fragment shader and cast alpha-tested shadows, and `BLEND` materials draw in
a separate transparent pass (see [transparency.md](transparency.md)). The cutoff
is data, not a pipeline variant -- it rides in ObjectFrameData -- so there is
still no shader permutation system. `doubleSided` remains metadata only.

## Runtime Mapping

`Renderer::createMaterialFromAsset()` converts a `MaterialAsset` into the
existing `renderer::Material` struct:

- `name` maps to `debugName` and `assetName`.
- `shader` maps to material metadata only.
- `baseColorFactor`, `metallicFactor`, and `roughnessFactor` map to the scalar
  PBR fields uploaded through `ObjectFrameData`.
- texture paths are stored on the runtime material and, when present, loaded
  into renderer-owned `VulkanTexture` objects during material creation.
- missing or failed texture loads use the existing base color, flat normal, and
  neutral metallic-roughness fallbacks before descriptors and bindless indices
  are assigned.

Scalar ImGui edits update the runtime material immediately because every frame
uploads current material data into the object-data buffer. Save writes the
runtime scalar/metadata fields back to the material asset path. Reload updates
scalar and metadata fields only; texture/descriptors are not hot-reloaded.

## glTF Compatibility

glTF loading still uses `Mesh::createFromGltf()`,
`Renderer::createImportedGltfTextures()`, and
`Renderer::createImportedGltfMaterials()`. glTF primitive `materialIndex`
values still resolve through `RenderObject::materialTable`.

JSON material assets do not replace glTF materials. The two paths coexist:
portfolio/procedural materials can come from JSON assets, while imported glTF
materials continue to use the existing runtime material table.

## Missing Textures

If a material asset references a texture path that does not exist or fails to
load, the renderer logs a warning and uses a fallback texture for that slot:

- base color: supplied base-color fallback
- normal: flat normal fallback
- metallic-roughness: neutral metallic-roughness fallback

Fallback selection happens before legacy material descriptor sets and bindless
texture array indices are assigned, so shader-visible indices remain valid.

## Scene Metadata

Scene JSON now saves material asset metadata when a selected object has a
runtime material with a source asset path. Load restores simple
`object.material` assignments by path when the material already exists in the
current runtime scene. Mesh references and glTF material-table assignments
remain metadata-only.

## Texture Cook (`vecook`)

`vecook` is an offline host tool that turns one PNG/JPG into a block-compressed
KTX2 with a baked mip chain. It exists because the asset-load baseline measured
Sponza at 365.98 MiB of textures with **0 of 77 block compressed**
([asset_load_baseline.md](asset_load_baseline.md)).

```
vecook <input.png> -o <output.ktx2> --usage <slot> [--no-mips] [--force] [--threads N] [--verify]
```

It links `VulkanEngineCore` and the vendored encoder, and deliberately does not
link Vulkan: the cook has no device, so it runs on a build machine with no GPU.

### Format policy

Format follows the **material slot**, not the file. The same PNG cooked as a base
colour and as a roughness map gets different formats and different mip filtering.

| Slot | Cooked format | Colour space |
| --- | --- | --- |
| `base-color`, `emissive` | `BC7_SRGB_BLOCK` | sRGB |
| `metallic-roughness`, `occlusion` | `BC7_UNORM_BLOCK` | linear |
| `normal` | `BC5_UNORM_BLOCK` | linear |

BC5 for normals spends the same 16 bytes per block on XY that BC7 would spend on
RGBA, and the shader reconstructs Z. BC7 and BC5 device support is reported
separately (`BlockCompressionCaps`), because a device can have one without the
other; when a slot's format is unsupported, `chooseTextureFormat()` returns
RGBA8 and the caller loads the original image instead of the cooked file. The
RGBA8 fallback keeps the sRGB decode -- losing compression must not also wash out
every base colour.

### Contracts worth knowing

- **Mip levels are averaged in linear light for sRGB textures.** Filtering sRGB
  code values directly darkens every level and the error compounds down the
  chain. Alpha is always linear.
- **Extents need not be multiples of four.** A trailing partial 4x4 block repeats
  its last real row and column. Zero-fill would bleed a dark edge into the
  block's endpoints; rejecting or resizing the image would be worse.
- **KTX2 stores mip data smallest level first**, which is the reverse of the level
  index and of everything Vulkan calls a mip level. Every level starts on a
  16-byte texel-block boundary.
- **Cook caching is mtime-based.** A full Sponza cook is tens of seconds of
  encoding, so re-encoding on every build is not acceptable. `--force` overrides.
- Encoding is spread across a `JobSystem` pool by block row; disjoint row ranges
  write to disjoint bytes, so the chunks need no synchronization.

### Verification

A malformed KTX2 fails *silently* in tools rather than loudly, so the container
is covered three ways:

1. Byte-level unit tests (`tests/test_ktx2.cpp`) read the written header, level
   index and data format descriptor directly rather than trusting a round trip
   to agree with a writer that got the layout wrong.
2. Policy unit tests (`tests/test_texture_cook.cpp`) pin the format choice, the
   linear-light mip filter, the partial-block padding, and the row-major block
   order.
3. `--verify` reads the written file back off disk, decodes **every** level with
   the vendored BC7/BC5 decoders, and reports PSNR against the source mip chain.
   It fails below 20 dB. That floor is a structural-corruption gate, not a
   quality bar: a correct BC7 encode of real content lands around 24-45 dB, while
   a deliberately transposed block order measured 7.5 dB.

Measured on an Apple M3, Release build, on a synthetic 1024x1024 image:
base level exactly **4.00x** (4.00 MiB RGBA8 -> 1.00 MiB BC7), and 3.00x once the
mip chain is included, since baked mips add about a third. A full BC7 chain took
about 100 ms across the default thread pool; BC5 took under 10 ms.

### Not wired to the runtime yet

`VulkanTexture` still decodes RGBA8 and builds mips with `vkCmdBlitImage`.
Consuming cooked files needs a `createFromKtx2()` doing per-mip
`vkCmdCopyBufferToImage` with **no** `generateMipmaps`, and a sampler `maxLod`
that follows the baked `mipLevels`. Dropping the runtime blit is what later makes
a transfer-only queue legal, which is why the cook comes before the async upload
work. Five `static_assert`s in `rhi/VulkanTexture.cpp` already hold the cook's raw
format numbers to the real `VkFormat` enumerators.

## Not Implemented

- asset browser
- mesh/geometry cooker
- runtime consumption of cooked KTX2 textures (see above)
- shader permutation system
- material graph
- full texture or descriptor hot reload
- mesh asset ownership
- ECS/editor scene architecture
