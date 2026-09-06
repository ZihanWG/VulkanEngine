---
name: shadow-and-taa-quality
description: "Phase 1-2 of the improvement plan SHIPPED (6e2231b): Catmull-Rom TAA history, hardware PCF via an immutable compare sampler, CSM normal-offset + cascade blend (both off by default); back-face culling measured and rejected"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4da1878a-f14b-40e8-b1c0-fc00e696ebfd
  modified: 2026-08-14T06:42:58.994Z
---

From the 2026-08-14 improvement plan
(`~/.claude/plans/delightful-questing-otter.md`). Pushed through `6e2231b`,
227 tests, 0 validation errors, 0 compiler warnings.

## Shipped

**Credibility pass (`b3eac19`).** README advertised GTAO as a whole-scene
composite multiply "not restricted to indirect/ambient light" -- the wrong
version fixed back in `e871ea6`. Also: the macOS launcher only searched
`build-mac`, which no preset produces (now searches `build/debug`,
`build/release`, `build-mac`, `build`); six hardcoded `/Users/zihanw` paths in
`docs/build_macos.md`; Windows CI never built or ran the tests (now does); CI was
undersold in README + docs/build.md.

**Catmull-Rom TAA history (`e009e00`).** `taa.catmullRomHistory`, default ON.
Nine bilinear fetches emulating the 4x4 kernel. Costs **+0.20 ms of a 1.26 ms
resolve** (1.4% of frame). **A still screenshot cannot show the difference** --
with a static camera `historyUV == vUV`, so no resampling happens and both paths
are bit-identical. It must be judged in motion, ideally at render scale 0.5.

**Hardware PCF (`e81f936`), default ON.** Makes shadow edges visibly softer.
Cost NOT yet A/B'd on a settled machine.

**CSM normal-offset + cascade blend (`6e2231b`), both default 0.0 = inert.**
`csm.normalBias`, `csm.cascadeBlend`, sliders in the Shadows panel. At 0.004 /
0.15 the cost was inside noise.

## Things worth not rediscovering

**Hardware PCF needs a SECOND sampler and it must be IMMUTABLE.** Four consumers
read the cascade array as a plain `sampler2DArray` for raw depth (`simple.frag`,
`fog_inject.comp`, `probe_capture.frag`, ImGui preview), so `compareEnable`
cannot be flipped on the shared sampler. And a normal descriptor write of a
compare sampler trips
**`VUID-VkDescriptorImageInfo-mutableComparisonSamplers-04450`** on MoltenVK:
`vulkaninfo` reports the feature as supported, but it must be *requested* at
device creation, and `VkPhysicalDevicePortabilitySubsetFeaturesKHR` is behind
`VK_ENABLE_BETA_EXTENSIONS`. `pImmutableSamplers` sidesteps it, which is
presumably why Metal restricts the mutable form at all. The sampler is owned by
`Renderer`, not `VulkanShadowMap`: the layout is created once for the renderer's
lifetime, the shadow map is recreated on cascade-count change.

**Normal-offset belongs in the VERTEX shader.** The fragment shader has no
`frameConstants` in its push-constant block, but the VS already has the cascade
matrices. Zero per-pixel cost, and the scaling is free: `length(cascadeVP[i][0].xyz)`
is `2/orthoWidth`, so its reciprocal is that cascade's world half-extent -- one
unitless bias scales correctly across cascades differing by 10x.

**`simple.vert` and `simple_skinned.vert` both feed `simple_bindless.frag`.**
Adding a varying to one only fails pipeline creation on the SPIR-V interface
check and kills the skinned path. Validation caught it; nothing crashed.

**Back-face culling MEASURED AND REJECTED** (`b7b437f`, documented in
`docs/design_decisions.md`). Every pipeline is `VK_CULL_MODE_NONE`, which reads
as an oversight. On the M3's tiler, HSR already discards occluded back faces
before shading, so there is nothing to save: `MainHDRPass` 9.994/10.216 off vs
10.058/10.098 on, inside the control's own spread. Scoped to this geometry-light
scene. Consequence: `Material::doubleSided` stays metadata-only, and wiring it is
now a pure correctness change with no perf upside. **Don't re-offer it as an
optimisation.**

## Open

- **clang-format gate deliberately NOT added.** 109/150 files would be
  reformatted, but the diff is pure line-wrapping churn -- the tree was formatted
  with an older clang-format. Adding the gate means pinning a version in CI *and*
  locally plus one whitespace-only commit across 109 files. User's call.
- `.clang-tidy` `WarningsAsErrors` still empty; clang-tidy is not installed
  locally so what is clean under the Linux CI toolchain is unknown.
- Phase 3 (SSR energy conservation -- additive blend double-counts specular
  against `specularIbl`), Phase 4 (code health), Phase 5 (LTC area lights) not
  started.

Related: [[back-to-back-or-dont-claim]], [[docs-drift-audit]],
[[shadow-cascade-cost]], [[temporal-upsampling-initiative]].
