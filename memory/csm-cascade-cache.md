---
name: csm-cascade-cache
description: CSM per-cascade content-hash caching shipped; the bounding-sphere fit meant to make it hit under camera motion was MEASURED AND REJECTED for that purpose
metadata: 
  node_type: memory
  type: project
  originSessionId: 16cda6ce-0ffd-440b-8aa7-ade1e10dd7c7
  modified: 2026-08-18T08:04:31.104Z
---

**MERGED to `main` (`411344d`, `--no-ff`)** from `feature/csm-cascade-cache`, 6
commits on top of `3a088dd`. **Pushed**, CI green on all three workflows (the llvmpipe headless gate matched the golden 0/921600 with the cache on, an independent driver from the local MoltenVK checks). Branch deleted.
**373 tests** (361 before). Closes the last shadow path with no caching — the punctual atlas had
hashed its inputs since it shipped, the cascades never did.

**What shipped (on by default):** a cascade is redrawn only when a content hash
of what *it* draws moves. Per cascade, not per pass — each renders into its own
layer view, so skipping one needs none of the LOAD-vs-CLEAR handling the shared
atlas does, and a fully cached frame skips the union caster cull too. The hash
reuses `shadowCascadeDrawItems_[i]`, which `buildShadowDrawItems` already builds
against the very frustum the pass renders with, so there is one cull to keep
correct rather than two that can drift.

**The non-obvious input:** on the GPU-culled path the CPU draw item's index range
is *not* what gets drawn — `cull.comp` picks the LOD level from projected screen
radius and writes it into the indirect command. Hashing `firstIndex/indexCount`
alone would freeze a cascade at the wrong detail when only the level changed. The
key mirrors `selectLodIndex()` per caster. Same class of trap as
[[gpu-cpu-struct-layout]]: the CPU-side record is not the GPU-side truth.

**Phase 2 premise FALSIFIED by measurement — do not re-propose the sphere fit as
a caching fix.** Binary-searched the largest camera change each fit survives
(2048px, 60° fov, 4 cascades):

| fit | yaw tolerance | translate tolerance |
| --- | --- | --- |
| AABB (default) | 0.0000° | 0.0000 units |
| bounding sphere | 0.0025–0.041° | 0.001–0.009 units |

A camera turning at 10°/s covers 0.17°/frame — 4x to 68x past all of them. The
slice centre **orbits the camera**, so rotation translates it by ~(centre
distance × angle), which dwarfs a texel. **Cascade 3 is the worst, not the best**
— its centre is farthest out, cancelling its coarser texel. The fit still shipped
(off by default) for the shimmer it does fix; its visual trade is unverified and
needs eyes. Making cascades cacheable under motion needs the cascade anchored to
the **camera position** rather than the frustum slice — ~1.5x wider coverage,
not attempted.

**Latent hole closed in BOTH caches (`0cf375f`):** casters are keyed by Mesh and
Material *pointer*, and `resetSceneState()` destroys those objects before the
next scene pushes fresh ones into the same vectors — reusing the freed
addresses. Same pointer + same `(firstIndex, indexCount)` + same transform,
different geometry ⇒ hash matches, image doesn't. The punctual atlas had this
since it shipped. Both now invalidate on scene reset.

**Two process traps hit here, both worth repeating:**

1. **`verify_renderer.sh` builds `build/ci-debug`, NOT `build/debug`.** Two
   separate GPU A/B runs were captured from a stale `build/debug/VulkanEngine`
   and read as clean passes. Always `cmake --build build/debug --target
   VulkanEngine` before capturing, and log something that proves the feature was
   live in that run.
2. **A pixel A/B is vacuous unless the feature is visible in the frame.** Probed
   it directly — `shadowDistance` 40 vs 2 changes the image, so CSM shadows do
   contribute in the default scene. Do that probe before trusting a byte compare.

**Correctness matrix, all byte-identical vs the uncached run** (current binary,
validation on, 0 errors/0 warnings throughout):

| case | cascades redrawn | pixels |
| --- | --- | --- |
| static scene, static camera | 4/4 then 0/4 forever | identical |
| moving casters, static camera | 4/4 every frame | identical |
| static casters, camera orbiting 10°/s | 4/4 every frame | identical |
| both moving | 4/4 every frame | identical |

The orbit row is the runtime confirmation of the offline tolerance table above:
**under any camera motion the cache is a strict no-op**, so there is no
stale-shadow regime there by construction. **The default startup scene has no
animated objects and no camera motion** — drive both to exercise invalidation.

**Perf: no quotable frame number.** Two A/B series both failed the 1% control
gate — see [[back-to-back-or-dont-claim]] for why (the Claude desktop app
contends for the GPU). What survives: `ShadowGpuCulling` present in **0/57 frames
cached vs 57/57 uncached** (presence is immune to timing noise), and
`CSMShadowPass` 0.024 vs 0.485 ms reproducing across both series with its own
control drift 200x below the delta. Best case only — the scene was fully static.

**Fixed in passing (`5b9deb9`):** `CsmSettings::normalBias` and `cascadeBlend`
were saved to JSON and read back but never applied — `applyRuntimeSettings`
copied CSM fields by hand and neither was on the list, so tuning them never
survived a restart. Now `applyCsmSettings` in `RuntimeSettings.h`, GPU-free and
tested with a defaulted `operator==` so a future dropped field fails the suite.
**Follow-up recorded, not done:** `kMaxDrawItems`/`kMaxFrameObjects` (both 8192,
`RendererInternal.h:172-173`) truncate **silently** — `buildDrawItems` bare
`return false`, object sweeps `std::min(...)`, no log or UI. Both precedents for
handling it already exist in-repo (`BindlessTextureHeap.cpp:159` throws,
`RendererDebugUi.cpp:2840` warns). Not currently hit (stress scene is 2311
objects). Related: [[runtime-settings-persistence]], [[shadow-cascade-cost]],
[[punctual-shadows-initiative]], [[docs-drift-audit]].
