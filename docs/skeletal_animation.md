# Skeletal Animation

GPU vertex skinning driven by a CPU-side animation core. The skeleton pose math
(keyframe sampling, hierarchy flatten) runs on the CPU and produces a joint-matrix
palette; the GPU does linear-blend skinning per vertex. It supports both a
procedural bone-chain demo (self-contained, no external asset) and glTF skin +
animation import through the same core; the renderer prefers a rigged glTF if one is
present (`assets/models/skinned_rig.gltf`) and falls back to the procedural chain.

The animation math lives in `src/renderer/SkeletalAnimation.{h,cpp}` (GPU-free, in
`VulkanEngineCore`), the glTF import in `src/renderer/GltfSkinnedImport.{h,cpp}`
(also GPU-free + unit-tested), the GPU mesh + palette in
`src/renderer/SkinnedMesh.{h,cpp}`, and the skinning vertex shader in
`src/shaders/simple_skinned.vert`.

## Animation Core

The core is deliberately GPU-independent so it can be unit-tested without a device
(`tests/test_skeletal_animation.cpp`):

- `JointPose` is a TRS transform; `matrix()` returns `T * R * S`.
- `Skeleton` holds three parallel arrays: `parents` (parent index, -1 for a root),
  `inverseBind` (model space → joint bind space), and `bindPose` (default local TRS).
- `AnimationClip` is a set of `AnimationChannel`s, each a sorted list of keyframe
  times + values for one joint's translation, rotation, or scale.
- `sampleVec3Channel` / `sampleQuatChannel` interpolate a channel at a time value:
  linear `mix` for vec3, `slerp` for rotations, clamping outside the keyframe range.
- `computeJointMatrices` flattens the hierarchy: `global[i] = global[parent[i]] *
  local[i]`, then `jointMatrix[i] = global[i] * inverseBind[i]`. It resolves joints
  iteratively up the parent chain, so arbitrary joint ordering works (glTF does not
  guarantee parents precede children).

A key invariant — verified by a unit test — is that **at the bind pose the joint
matrices are identity**, so an un-animated skinned mesh renders exactly like a
static one. Other tests cover TRS composition, vec3 clamp/lerp, quaternion slerp,
and parent→child propagation.

## GPU Skinning

Skinning is an additive path; the static mesh/pipeline is untouched.

- **Vertex streams.** Binding 0 is the standard geometry `Vertex`; binding 1 is a
  parallel `SkinningVertex` (`uvec4 jointIndices`, `vec4 weights`). A dedicated
  skinning pipeline declares both bindings (seven attributes total) and reuses the
  bindless fragment shader + the main pipeline layout, so the bound descriptor sets
  and push constants stay compatible.
- **Joint palette.** `SkinnedMesh` owns one host-visible, buffer-device-address
  palette buffer per frame-in-flight. Each frame it recomputes the joint matrices
  from the animation core and uploads them; the address is handed to the vertex
  shader through the shared push constant (`PushConstants::jointMatricesAddress`).
- **`simple_skinned.vert`** computes a linear-blend skinning matrix
  `Σ weight[i] * palette[jointIndex[i]]`, applies it to the position and to the
  normal/tangent (via its upper 3×3), then runs the usual model/MVP transforms and
  emits the identical varyings as `simple.vert` — which is why it can share the main
  fragment shader. The skinned mesh draws in the main HDR pass (profiler scope
  `SkinnedMesh`), lit by the scene's directional + clustered lights.

## Procedural Demo

The Phase 1 demo (`SkinnedMesh::create`) builds a six-segment bone chain: each box
segment is rigidly bound to one joint, and joint `i` is offset from its parent so
its bind-space global position is `(0, i·segmentHeight, 0)` (its inverse-bind is the
inverse). `SkinnedMesh::update` applies a phase-shifted sine rotation per joint each
frame, so the chain curls and waves. Pose evaluation goes entirely through the
unit-tested core. The **Skeletal Animation** ImGui panel exposes show/animate
toggles, a 0–4× playback speed (driven by an accumulated, speed-scaled time so
pausing holds the pose), and a reset-to-bind-pose button.

## Casting shadows

The skinned mesh casts into all three shadow paths: the cascades, the punctual
atlas, and the virtual shadow map's page pool. Until it did, it was drawn into
the main pass and lit by the sun while throwing no shadow at all.

It cannot join the casters that were already there. Those go through indirect
draws over one shared vertex buffer, batched by mesh and indexed through
`gl_InstanceIndex`; this mesh has a second vertex binding (joint indices and
weights) and a per-frame palette that no batch carries, and it is not a
`RenderObject`, so it appears in neither `allDrawItems_` nor any batch. It is
drawn directly instead, after the indirect draws, into each rect it reaches.

One shader (`shadow_skinned.vert`) serves all three, because a cascade, an atlas
tile and a clipmap page are the same thing from a caster's point of view: one
rect with one view-projection, pushed. Its skinning loop is a copy of
`simple_skinned.vert`'s deliberately -- a caster that deforms differently from
the surface it casts for produces a shadow subtly detached from its own
geometry, and nothing in the image says which of the two shaders is wrong.

### The bound, and why a skinned caster needs one at all

Every cull and every shadow cache here keys on a transform. A skinned mesh
breaks that: its model matrix holds still while its vertices move. So the mesh
carries two derived values, both computed from the same palette that was
uploaded rather than from a separately evaluated pose:

- **A conservative world bound.** Per joint, the bind-space box of the vertices
  that joint actually influences, transformed by that joint's matrix, unioned.
  Conservative provably: linear-blend skinning places a vertex at
  `sum(w_i * M_i * p)` with non-negative weights summing to one, a convex
  combination of points each inside its own transformed box. Loose -- a rotating
  box's AABB grows -- but never small, which is the direction it has to err in,
  because a bound that is too small is a shadow that silently stops being drawn.
  The tests state that argument and then sweep 24 poses against it.
- **A pose digest.** What the shadow caches key on. Mesh pointer, index range and
  model matrix are all constant across the entire animation, so without it a
  cascade or a tile would hold a shadow of a pose that is long gone.

The digest enters a cascade's or a tile's key only when the caster reaches that
frustum, tested by the same predicate the recorder uses before drawing. The two
must not disagree: a cascade drawn without its pose in its key would cache a
shadow that never updates again.

### What it costs

Measured on the default scene, with the demo rig animating:

| path | effect |
| --- | --- |
| cascades | reaches cascade 0 only; that cascade redraws every frame |
| punctual atlas | 6 of 25 tiles per frame |
| VSM pages | 12 of 99 resident pages invalidated and redrawn per frame |

Animated geometry has no cacheable shadow -- that is what "animated" means for a
depth cache -- so those redraws are the honest price rather than a number to
tune away. Each count is printed in the periodic log block, because
`--capture-frame` excludes the debug UI and there is otherwise no way to tell
from a scripted run whether the caster reached anything.

## Limitations and Future Work

- The procedural demo uses rigid (single-bone) weighting; the glTF path exercises
  smooth multi-influence weights (the shader blends up to four per vertex).
- glTF import (`GltfSkinnedImport`) reads `JOINTS_0`/`WEIGHTS_0`, inverse-bind
  matrices, and animation channels/samplers, feeding the same core. A generator
  (`tools/gen_skinned_rig.py`) emits a minimal embedded test rig
  (`assets/models/skinned_rig.gltf`) that the import is unit-tested against. The
  import currently uses the first skinned mesh + first primitive of a file.
- One skinned instance is drawn with its own object-data slot; a general scene would
  manage many skinned instances with per-instance palette offsets.
- Linear-blend skinning shows the usual volume-loss artifacts at extreme bends; dual
  quaternion skinning would address that.
- **The layered (multiview) cascade path skips the skinned caster.** One draw
  fans out to every cascade layer there and picks its matrix from
  `gl_ViewIndex`, which a pushed matrix cannot answer; drawing it anyway would
  put cascade 0's projection into all four layers. It is skipped and the log
  says so. That path ships off and measured ~20% slower on this driver.
- **The skinned caster is opaque-only.** No alpha-tested variant, so a cutout
  skinned material would throw a solid silhouette. Nothing in the engine has one
  yet.
- **The punctual half is exercised but not visually verified.** In the default
  scene the caster reaches 6 atlas tiles with zero validation errors, and the
  frame changes by at most one quantization step -- the animated light swarm's
  shadows there are too dim, and the mesh too close to the frame edge, to
  compose an A/B from. The cascade and page halves both produce a visible,
  reproducible shadow on the perforated panel.
- **The bound is per joint, not per cluster.** A long bone whose box is mostly
  empty pays for the whole box. Fine for a demo rig; a real character would want
  the bound fitted more tightly than "every vertex this joint touches".
