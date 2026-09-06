---
name: skeletal-animation-initiative
description: Second interview flagship — GPU skeletal animation on branch feat/skeletal-animation
metadata: 
  node_type: memory
  type: project
  originSessionId: bb7471bd-8ada-486d-90bb-a62898959740
---

## 2026-08-23: the skinned mesh is a caster now (all three shadow paths)

**MERGED to main (`9abe253`, PR #21, `--merge`, branch deleted), all 4 CI pipelines
green.** The lavapipe golden was re-baselined for it — 811/921600 pixels, max delta
8, Mesa 25.2.8 on both sides, and the diff traced exactly the perforations the new
shadow falls across plus the caster's own silhouette.

**Review (Codex) found four real defects, all fixed in `783f84c` — the first is the
one to remember:** `drawFrame` runs `updateVsmResidency` (which decides page
invalidation) BEFORE `updateFrameData`, where the skinned pose used to be advanced.
So invalidation judged the previous pose while the page pass drew the new one, and a
page the caster moved into stayed cached holding depth that omitted it. The pose
advance is now its own step called before residency. Also: the layered cascade path
hashed a pose it never draws (dirtying the whole layered map every frame), and two
telemetry counters were assigned only on their success tails so early returns left
last frame's value standing.

It had been drawn into the main pass and lit by the sun while **throwing no shadow at
all** — it is not a `RenderObject`, so it was in neither `allDrawItems_` nor any batch,
and every shadow path walks one of those.

It cannot join the batched casters (second vertex binding + per-frame palette), so it is
a **direct draw after the indirect ones**, into each rect it reaches. One shader
(`shadow_skinned.vert`) serves cascades, punctual tiles and VSM pages, because all three
are "one rect, one pushed view-projection".

**The part that is easy to get wrong:** every cull and every shadow cache here keys on a
transform, and a skinned mesh deforms while its transform holds still. So `SkinnedMesh`
now derives two things from the *same palette it uploads*: a conservative world bound
(per-joint bind box transformed and unioned — convex-hull argument, 24-pose sweep test)
and a **pose digest** that enters cascade and tile cache keys. The recorder and the key
must use the identical reach predicate, or a cascade caches a shadow that never updates.

**Measured cost on the default scene:** cascade 0 redraws every frame; 6 of 25 punctual
tiles; **12 of 99 VSM pages invalidated and redrawn per frame**. Animated geometry has
no cacheable shadow — that is inherent, not tuning.

**Not done, deliberately:** the layered/multiview cascade path is SKIPPED (one draw fans
out to all layers and picks its matrix from `gl_ViewIndex`, which a pushed matrix cannot
answer; that path ships off and is ~20% slower here). No alpha-tested skinned variant.
**The punctual half is exercised but NOT visually verified** — 6 tiles, zero validation
errors, but the frame moves by at most one quantization step.

Second flagship after [[clustered-lighting-initiative]] (user picked it as the next flagship 2026-06-27). GPU skeletal animation. On branch **feat/skeletal-animation** (off main), NOT pushed/merged, **GPU-unverified** (sandbox can't run GPU — see [[vulkanengine-cannot-run-in-sandbox]]). Phased per [[user-prefers-phased-checkpoints]].

Design choices: additive skinned path (static mesh/pipeline untouched, zero regression risk); procedural-first demo (self-contained, no asset needed) with the GPU-free pose math unit-tested.

Status (5 commits on the branch):
- **SA Phase 1 done**: `SkeletalAnimation.{h,cpp}` (GPU-free core in VulkanEngineCore: JointPose TRS, keyframe sampling w/ slerp, hierarchy flatten → palette, arbitrary joint order) + 5 Catch2 tests incl. the bind-pose→identity invariant (48 tests total). `SkinnedMesh.{h,cpp}` (procedural 6-segment bone chain, parallel joint-index/weight vertex stream, per-frame joint-palette BDA buffers). `simple_skinned.vert` (linear-blend skinning, reuses bindless frag + main pipeline layout). Dedicated skinning pipeline (2 vertex bindings); skinned mesh drawn in main HDR pass via a reserved ObjectFrameData slot; `PushConstants.jointMatricesAddress` added (offset 72). GPU profiler scope `SkinnedMesh`.
- **SA Phase 3 done**: "Skeletal Animation" ImGui panel (show/animate toggles, 0-4x speed via accumulated time, reset-to-bind-pose).
- **SA Phase 4 done (docs)**: `docs/skeletal_animation.md` + README feature/highlights/resume + docs index. **Demo GIF deferred** (needs GPU).
- **SA Phase 2 NOT done**: glTF skin + animation import (JOINTS_0/WEIGHTS_0, inverse-bind, channels). Needs the user to drop a rigged animated glTF (e.g. Khronos CesiumMan/Fox) into assets/models/.

Demo: procedural bone chain stands on the ground at ~x=-2.6 (left of the showcase spheres), bends/waves; lit by directional + clustered lights.

GPU-verified by the user 2026-06-28 (bone chain renders + bends correctly). One bug found + fixed on GPU: abutting box segments shared a cap plane → z-fighting "black flicker" worst when swung sideways; fixed by gapping the segments.

**Shipped** 2026-06-28: merged feat/skeletal-animation → main via --no-ff (756c255), pushed to origin/main, local branch deleted (it was never pushed, so no remote branch). Merged main builds clean + 48/48 tests pass.

**SA Phase 2 done** (glTF import) on branch **feat/skeletal-animation-gltf** (1 commit f962bc9, NOT merged, GPU-unverified for the rig render — but the render path is the GPU-verified procedural one; only the data/clip differ). User chose the self-contained "minimal test rig" route. `GltfSkinnedImport` (GPU-free, in VulkanEngineCore) parses skin/skeleton/JOINTS_0/WEIGHTS_0/inverse-bind/animation channels → Skeleton + AnimationClips. tinygltf IMPLEMENTATION centralized into GltfSkinnedImport.cpp (removed the dup define from Mesh.cpp). `tools/gen_skinned_rig.py` generates `assets/models/skinned_rig.gltf` (3-ring tube, 2 joints, rotation clip); 2 unit tests (50 total). SkinnedMesh.createFromGltf + renderer prefers the rig over procedural; UI shows source/clip.

**SA Phase 2 shipped** 2026-06-28 via **PR #4** (gh pr merge --merge --delete-branch; merge commit 62f0268). Branch deleted local+remote; merged main builds clean + 50/50 tests pass. The whole skeletal-animation initiative (Phases 1-4 + glTF import) is now on main. (Note: gh CLI was installed + authed by the user this session — `gh pr create`/`merge` work now.)

Remaining (optional): GPU-verify the imported rig renders + bends; capture a skinning demo GIF for the README.
