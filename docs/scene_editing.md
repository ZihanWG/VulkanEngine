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

The mesh references are still runtime pointers. Scene JSON saves mesh debug
names and pointer strings for inspection, but load does not recreate mesh
assets. Material asset paths are saved when available; load can restore simple
`object.material` assignments when the path matches a material that is already
loaded by the current runtime scene. glTF `materialTable` assignments remain
runtime data and are not rebuilt from scene JSON.

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

The save path creates `assets/scenes/` and `default.scene.json` if they do not
already exist. A fresh checkout does not track a default scene file by design.
If `Load Scene` is clicked before any scene has been saved, the UI reports
`No saved scene found. Use Save Scene first.` and logs a friendly warning
instead of treating the missing file as a renderer failure.

Load matches objects by `id` first, then by `name` as a fallback. It restores
matching runtime objects in place, preserves current mesh pointers, and only
rebinds simple material pointers when a saved material asset path matches an
already loaded runtime material.

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
        "assetName": "",
        "assetPath": "",
        "shader": "pbr_opaque",
        "primaryLabel": "Checkerboard",
        "pointer": "0000000000000000",
        "slotCount": 0,
        "source": "built-in/procedural material",
        "materialAssetRebinding": "restored by assetPath when available"
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
- glTF material-table rebinding, descriptors, or texture bindings
- glTF source asset paths
- object creation/deletion
- hierarchy parenting
- animation state beyond disabling demo animation on edit/load
- editor camera state (position, orientation, movement speed)
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

Portfolio capture mode uses portfolio showcase object, camera, and light
presets. Object and camera edits are possible while the mode is active, but
pressing F12 or clicking the portfolio showcase button reapplies the portfolio
preset before capture. The directional-light editor controls the non-portfolio
directional light; portfolio lighting remains preset-driven.

## ImGuizmo

ImGuizmo is vendored under `external/imguizmo/` and compiled into the `imgui`
target (`CMakeLists.txt:187`), so it carries no new fetch step and no new
find_package. The viewport gizmo ships: `Renderer::drawViewportGizmo()`
(`src/renderer/Renderer.cpp:783`) draws a translate/rotate/scale manipulator for
the current selection onto the background draw list, with W/E/R switching
operation and X toggling world versus local space (`Renderer.cpp:655-668`); the
same three operations and the space toggle are also radio buttons and a checkbox
in the Selection & Gizmo section of the debug UI (`RendererDebugUi.cpp:1996-2007`).

The gizmo and click-to-pick share the mouse, so picking is gated on
`ImGuizmo::IsOver()` and `ImGuizmo::IsUsing()` (`Renderer.cpp:599-622`) --
without that, releasing a drag on the manipulator would reselect whatever
happened to be under the cursor.

Manipulation writes back through `Transform::fromMatrix` plus
`convertMatrixOverrideToEditableTrs`, so dragging an imported glTF node -- which
arrives as a `matrixOverride` -- converts it to editable TRS rather than
silently discarding the drag.
