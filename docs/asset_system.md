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

### Cooking a whole scene

```
tools/cook_textures.py <scene.gltf>
```

Runs `vecook` once per (image, material slot). It reads a glTF rather than
walking a directory because the cooked format is decided by the **slot**, and a
directory listing carries no slot information — guessing it from a file name is
the silent-corruption case the pipeline refuses to risk. Sponza is 69 cooks and
takes about 5 s.

## Runtime consumption

`VulkanTexture::createFromKtx2()` uploads a cooked file with one
`vkCmdCopyBufferToImage` carrying one region per level, straight out of a staging
buffer holding the whole file — the level offsets are already texel-block
aligned, so nothing is repacked.

The synchronization is *simpler* than the uncompressed path, not more complex:

| | RGBA8 + blit | cooked KTX2 |
| --- | --- | --- |
| barriers | 2 + 2×(N−1) | **2** |
| layouts | UNDEFINED → TRANSFER_DST → (per level TRANSFER_SRC) → SHADER_READ | UNDEFINED → TRANSFER_DST → SHADER_READ |
| image usage | TRANSFER_DST \| SAMPLED \| **TRANSFER_SRC** | TRANSFER_DST \| SAMPLED |

The blit path needs a barrier per level only because each level is read back as a
blit source; the cooked path never reads the image. Dropping `TRANSFER_SRC` is
the point beyond the memory win: `vkCmdBlitImage` requires a graphics queue and
`vkCmdCopyBufferToImage` does not, so this is what makes an async transfer queue
legal later. Sampler `maxLod` follows the baked `mipLevels`, so a `--no-mips`
file gets a sampler that stops at level 0.

Three load paths consume sidecars, each of which already knows its slot:
`BuiltinTextureFactory`, `Renderer::loadMaterialAssetTextureOrFallback()`, and
the glTF import loop in `RendererScene.cpp` (which skips the JobSystem decode
entirely for a cooked texture, since there is nothing to decode).

### Batched upload

`rhi::VulkanUploadBatch` records many textures into one command buffer and waits
once on a fence. Before it, every texture submitted its own command buffer and
called `vkQueueWaitIdle` — draining the entire queue, 69 times on Sponza, which
was about half of texture upload time.

Two properties are load-bearing:

- **The staging buffers are retained by the batch**, not by the caller's scope. A
  queued copy still reads them after the recording function returns, so freeing
  them early would corrupt texels rather than crash. The batch's destructor
  submits, so an exception mid-load cannot drop them either.
- **The batch is bounded, not unbounded.** It flushes when retained staging
  crosses `kUploadBatchStagingBudgetBytes` (64 MiB), so peak staging stops
  depending on scene size — Sponza uploads in 2 submits at 62 MiB peak. The
  cooked-file prefetch uses the same budget as its window.

A batch is optional: `createFromKtx2()` and `createFromRgba8()` take one or
`nullptr`, and without it they keep their own submit-and-wait. Only the glTF
import loop batches today; `BuiltinTextureFactory`, the material-asset path, the
BRDF LUT, and cubemaps are unchanged.

Batching had to come before the transfer queue below: 69 serialised waits are 69
serialised waits whichever queue they are on.

### Dedicated transfer queue

When the device exposes a **DMA family** -- transfer-capable, neither graphics nor
compute -- the copies run there and each image is handed to the graphics queue
with a queue family ownership transfer. `rhi/TransferQueueSelection.h` holds the
policy as a pure function; there is deliberately no "second queue in the graphics
family" fallback, unlike async compute, because a second graphics queue is still
the graphics ring and would add ownership-transfer machinery for no overlap.

Images stay `VK_SHARING_MODE_EXCLUSIVE`, so this is a real ownership transfer, not
the `VK_SHARING_MODE_CONCURRENT` shortcut the clustered-lighting buffers use
(`async_compute.md`) -- right for buffers, but concurrent sharing can cost image
compression.

| | src family | dst family | oldLayout | newLayout | src stage/access | dst stage/access |
| --- | --- | --- | --- | --- | --- | --- |
| pre-copy (transfer) | IGNORED | IGNORED | UNDEFINED | TRANSFER_DST | NONE / NONE | TRANSFER / TRANSFER_WRITE |
| release (transfer) | transfer | graphics | TRANSFER_DST | SHADER_READ_ONLY | TRANSFER / TRANSFER_WRITE | NONE / NONE |
| acquire (graphics) | transfer | graphics | TRANSFER_DST | SHADER_READ_ONLY | NONE / NONE | FRAGMENT_SHADER / SHADER_SAMPLED_READ |

Three things this depends on:

- **Release and acquire must name identical layouts and identical family
  indices.** They are halves of one operation, and a mismatch is a VUID violation
  that can simply skip the transition. They are built by one helper for that
  reason.
- **The semaphore is not optional.** Ownership transfer needs an execution
  dependency between the two submissions or the acquire may run before the
  release.
- **A texture that generates its own mips cannot use this path**, because
  `vkCmdBlitImage` requires a graphics queue. Those fall back to their own
  submit-and-wait. This is the same constraint that made the texture cook worth
  doing first: cooked textures ship their mips and never blit.

A TRANSFER-only queue does not list `SHADER_READ_ONLY_OPTIMAL` among its
supported layouts, and the release barrier above names it anyway, on the grounds
that the transition belongs to the ownership transfer rather than to the source
queue. **That was verified, not assumed** -- validation layers accept this form on
MoltenVK. The contingency, had they not, was to transfer ownership at
`TRANSFER_DST → TRANSFER_DST` and add a third graphics-side barrier.

**Measured effect on wall clock: none that this machine can resolve.** Interleaved
A/B, seven warm runs each on Sponza: 21.99 ms with the queue off, 21.13 ms with it
on -- against a 42.6% spread within the off configuration alone. The change is a
capability, not a speedup: the serial load flow still waits for the upload before
continuing, so the win would have to come from overlapping upload with mesh
building, which is not done. Do not quote the 0.86 ms.

### Every failure falls back, none is fatal

A cooked file is used only when it exists, is **not older than its source**, and
holds the format that slot expects. Otherwise the uncompressed source loads and a
warning is logged. All three failure modes were exercised on the GPU:

| Broken how | Result |
| --- | --- |
| source edited after the cook | stale, falls back; 3 of 8 block compressed became 2 of 8 |
| BC7 sRGB file planted where the BC5 normal map belongs | refused by format, falls back |
| truncated file | refused by the parser, falls back |

The format is re-checked inside `createFromKtx2()` and not only at the call site,
because a mis-slotted file is a *wrong image that still renders* — worse than one
that does not.

Sidecars are gitignored. They are lossy build artifacts, and committing them
would make the golden image depend on whether a cook had been run.

### Measured

See [asset_load_baseline.md](asset_load_baseline.md): on Sponza, texture memory
drops **365.98 MiB → 91.06 MiB (4.02x)**, 72 of 77 textures become block
compressed, decode wait goes to zero, and upload halves.

### Limitations

- **The transfer queue is inert by default and CI never runs it.** On this
  machine a TRANSFER-only family does not exist unless
  `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1` is set -- MoltenVK otherwise exposes
  four identical GRAPHICS|COMPUTE|TRANSFER families -- which is the same gate the
  async compute path documents. Everything above about ownership transfer is
  therefore exercised only under that environment variable, and was verified by
  hand with validation layers rather than by CI.
- **Upload is not overlapped with anything.** The load flow still waits for the
  copies before continuing, so the transfer queue buys the capability and frees
  the graphics queue without shortening startup.
- No visual A/B has been composed yet. The Sponza startup camera frames the
  building from outside (see the limitations above), and the default portfolio
  scene *loads* the checker textures but does not sample them in the composed
  shot — cooking them changes exactly zero pixels there. That last fact is why
  the committed golden image is unaffected by whether a cook has been run, but it
  also means judging BC7 quality needs the editor camera inside Sponza.
- `VulkanEnvironmentMap` cubemaps and the BRDF LUT keep their own upload paths
  and are not cooked.
- The cook is never run from CMake. A clean configure must not silently spend a
  minute encoding, so `tools/cook_textures.py` is run deliberately.

## Not Implemented

- asset browser
- mesh/geometry cooker
- async transfer-queue upload (now legal, see above, but not done)
- shader permutation system
- material graph
- full texture or descriptor hot reload
- mesh asset ownership
- ECS/editor scene architecture
