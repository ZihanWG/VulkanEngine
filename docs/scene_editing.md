# Editable Scene Workflow

Phase 2 turns the existing debug scene hierarchy into a minimal runtime scene
editing workflow. It stays renderer-side and does not introduce ECS, an asset
browser, scripting, animation authoring, physics, or a material graph.

## Scene Object Representation

Editable objects are the existing `renderer::RenderObject` entries owned by
`Renderer`.

Each object now carries:

- `sceneObjectId`: stable runtime/editor ID, initialized from the existing
  debug ID allocator.
- `debugId`: retained for existing debug labels.
- `debugName`: editor/display name.
- `visible`: renderer visibility toggle.
- `transform`: position/rotation/scale or matrix override.
- `mesh`: runtime mesh pointer.
- `material`: runtime primary material pointer.
- `materialTable/materialCount`: optional glTF primitive material table.

The mesh and material references are still runtime pointers. Scene JSON saves
their debug names and pointer strings for inspection, but load does not rebind
or recreate assets yet.

## Transform Updates

The Scene Hierarchy panel keeps the existing selected-object index and adds an
editable inspector for the selected `RenderObject`.

The inspector exposes:

- position
- rotation in degrees, stored internally as radians
- scale
- visibility

When a transform field changes, the selected object's demo animation flag is
disabled so the next frame does not overwrite the edit. `Renderer::updateFrameData()`
then rebuilds draw items, culling inputs, shadow draw lists, and per-draw
`ObjectFrameData` from the edited transform. The GPU sees the new model, MVP,
shadow MVP, bounds, and culling data through the existing per-frame upload path.

Imported glTF nodes can arrive as `Transform::matrixOverride`. The inspector can
decompose that matrix into editable TRS using GLM. If the matrix cannot be
decomposed safely, the object remains visible but the TRS controls are disabled
for that transform.

## Save And Load

Use the Scene Hierarchy panel:

1. Click `Save Scene` to write `assets/scenes/default.scene.json`.
2. Click `Load Scene` to read that file back into the current runtime scene.

The save path creates `assets/scenes/` if it does not exist.

Load matches objects by `id` first, then by `name` as a fallback. It restores
matching runtime objects in place and preserves their current mesh/material
pointers.

## Serialized Data

The JSON schema is intentionally small:

```json
{
  "schemaVersion": 1,
  "sceneName": "Default Runtime Scene",
  "camera": {
    "position": [0.0, 0.35, 5.5],
    "target": [0.0, 0.1, 0.0],
    "up": [0.0, 1.0, 0.0],
    "verticalFovDegrees": 60.0,
    "nearPlane": 0.1,
    "farPlane": 100.0
  },
  "directionalLight": {
    "direction": [0.35, -0.65, -0.55],
    "color": [0.85, 0.85, 0.85],
    "intensity": 1.0,
    "portfolioPresetActive": false
  },
  "objects": [
    {
      "id": 1,
      "debugId": 1,
      "name": "Center Cube",
      "visible": true,
      "source": "built-in fallback cube",
      "portfolioOnly": false,
      "hideInPortfolio": true,
      "transform": {
        "mode": "trs",
        "position": [0.0, -0.1, 0.0],
        "rotationDegrees": [11.459, 0.0, 0.0],
        "scale": [0.7, 0.7, 0.7]
      },
      "mesh": {
        "name": "Built-in Cube Mesh",
        "pointer": "0000000000000000",
        "submeshCount": 1
      },
      "material": {
        "name": "Checkerboard",
        "primaryLabel": "Checkerboard",
        "pointer": "0000000000000000",
        "slotCount": 0,
        "source": "built-in material"
      },
      "drawItemCount": 1
    }
  ]
}
```

Transforms save in TRS mode for normal edited objects. Matrix-backed imported
objects save `mode: "matrix"` plus a column-major 16-float `matrix` array and a
best-effort decomposed TRS summary.

## Not Serialized Yet

Scene JSON does not currently serialize or restore:

- mesh asset paths or mesh GPU buffers
- material assets, descriptors, texture bindings, or PBR edits
- glTF source asset paths
- object creation/deletion
- hierarchy parenting
- animation state beyond disabling demo animation on edit/load
- camera movement speed, because there is no free-camera controller yet
- ImGuizmo state

## glTF And Portfolio Mode

glTF loading remains unchanged. `Renderer::createScene()` still tries
`assets/models/test_mesh.gltf`, then `.glb`, then the built-in fallback scene.
Scene loading is applied after the runtime scene exists, so it edits matching
objects rather than replacing the glTF or fallback creation path.

Portfolio showcase objects are still appended by
`Renderer::addPortfolioShowcaseObjects()`. The save file may include those
runtime objects if they exist, but the portfolio screenshot scene is not moved
to a separate asset system in this phase.

Portfolio capture mode uses portfolio camera and light presets. Camera edits
are possible while the mode is active, but pressing F12 or clicking the
portfolio showcase button reapplies the portfolio preset before capture. The
directional-light editor controls the non-portfolio directional light; portfolio
lighting remains preset-driven.

## ImGuizmo

ImGuizmo is not currently vendored under `external/`. Phase 2 keeps the ImGui
inspector as the transform-editing path and defers a translate/rotate/scale
gizmo until the dependency can be added without disrupting the build.
