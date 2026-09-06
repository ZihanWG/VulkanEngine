---
name: pipeline-key-store
description: PSO-by-state store SHIPPED (4 phases, graphics + compute); membership is decided by LIFETIME not pipeline type; doubleSided is NOT blocked by structure
metadata:
  type: project
---

Pipelines are now looked up by state (`rhi::PipelineKey` +
`rhi::VulkanPipelineStore`), branch `refactor/pipeline-key-store`, four commits,
**MERGED to `main` on 2026-08-31 via PR #8 (merge commit `ec202a9`); all three
CI workflows green on the merge commit, including the lavapipe headless render +
validation gate — an independent pixel check on llvmpipe, not MoltenVK.** Branch
deleted. A follow-up docs fix landed as PR #9 (`245731f`): the section's measured
table had kept the phase-2 numbers after phases 3-4 changed them, which is the
drift shape [[docs-drift-audit]] describes — **a table written mid-initiative goes
stale at the next phase, not at the next release.**
Rationale and the measured table live in `docs/design_decisions.md`; this note is
what that file does not say.

**The collapse, measured, not predicted:** default config 9 requests -> 8
pipelines; `--vsm shadows` 12 -> 9. `PunctualShadowPipeline` = `VsmPagePipeline`,
and the three skinned casters are one object. The default scene alone understates
it because VSM ships off -- **run `--vsm shadows` to exercise the shadow-caster
pipelines at all**.

**I was wrong once here, publicly:** I first justified this work by saying the
pipeline structure blocked `Material::doubleSided`. It does not.
`docs/design_decisions.md` "Back-face culling is off, and that is measured"
records an A/B on this TBDR showing culling buys nothing, so `VK_CULL_MODE_NONE`
everywhere is deliberate and `doubleSided` is intentionally metadata-only. Do not
re-propose wiring it as a cleanup; it is a correctness change that can only make
geometry disappear. See [[back-to-back-or-dont-claim]].

**The normalization rule worth reusing:** over-normalizing a cache key returns an
object built for a different configuration, quietly; under-normalizing costs one
extra object. So normalize only what the creation code *proves* cannot reach the
driver, and keep the rest verbatim.

**The reset rule bit within one commit, and this is the reusable lesson.**
`reset()` requires *every* ref to be reissued by the rebuild that reset the store.
`createProbeCapturePipeline()` was reachable only from the Renderer constructor,
never from `createPipeline()`, so routing it left a dangling ref -- latent, because
`pipelineNeedsRecreate` only fires when a format actually changes. Fixed by
registering it in `createPipeline()`, plus a generation stamp on `PipelineRef` so a
forgotten ref reads as `VK_NULL_HANDLE` rather than freed memory. **Any new
PipelineRef must be reissued from `createPipeline()`.**

**How to exercise the rebuild path** (there is no way to resize the window from a
session): temporarily add a second `createPipeline()` call after the probes are
created, and read the store's request count from the log. 19 requests = correct,
18 = a ref was left dangling. See [[instrument-before-guessing-runtime-bugs]].

**Membership is decided by lifetime, not by pipeline type.** Only a pipeline that
`createPipeline()` rebuilds may live in the store. By that test 22 pipelines are in
(all graphics, plus depth-pyramid + 3 exposure compute, SSR, GTAO x2); the **8
compute pipelines owned by ClusteredLighting, GpuCulling, PunctualShadows,
VolumetricFogPass, VirtualShadowMapPass and the probe volume stay out** -- they are
built inside subsystem `createResources()`, so a store reset would destroy them
with nothing to rebuild them. Do not "finish the job" by routing them.

Store contents: default 21 pipelines / 22 requests; `--vsm shadows` 22 / 25.
Compute and post-process pipelines produced **no** collapse, as expected -- every
`.comp` and post-process fragment shader is distinct.

**Next:** delete `Renderer::pipelineNeedsRecreate` (a 20-clause hand-maintained
format check the key subsumes). Blocked on `reset()` being unconditional: an
unconditional rebuild would recompile everything on every resize, and the obvious
fix (mark-and-sweep over one build generation) does not work while some pipelines
are created outside any build. Needs subsystem-lifetime pipelines scoped
separately, or all pipeline creation funnelled through one place. Shader hot-reload
comes after that. This was step 1 of 4 in an engine-gap review; the others were
hot-reload, a capability-gated meshlet path (RTX-only, unverifiable on this Mac),
and runtime mip streaming.

**GPU-verified 2026-08-30 across 16 configurations** (7 settings x default scene,
6 scene presets, 3 VSM modes), baseline `2171abc` vs `c147e82`, both sides built
from worktrees with byte-identical assets and SPIR-V: **32/32 runs exit 0 under
`--fail-on-validation-error`, all 16 pixel comparisons 0/3686400.** Probe capture
confirmed live (`ProbeCapture` + `IrradianceProbeUpdate` recorded, no "capture
unavailable" warning), which is the path the dangling-ref fix was for. This
refactor must stay pixel-identical, so any diff is a bug, not a tolerance. Read
[[gpu-ab-needs-matched-inputs]] before re-running that matrix.
