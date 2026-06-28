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
