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

## Not Implemented

- asset browser
- asset cooker
- texture compression pipeline
- shader permutation system
- material graph
- full texture or descriptor hot reload
- mesh asset ownership
- ECS/editor scene architecture
