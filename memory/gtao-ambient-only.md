---
name: gtao-ambient-only
description: "GTAO now modulates only the ambient term via previous-frame reprojection (merged e871ea6, pushed). Visually confirmed 2026-08-10. Exposed a latent descriptor hazard worth knowing."
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-08T06:43:56.166Z
---

Shipped 2026-08-07, merged `e871ea6`, pushed. 180 tests, validation clean in all
three states (GTAO off, ambientOnly on, composite reference path).

## What changed

GTAO was `sceneColor *= ao` in the composite, darkening **direct** lighting too.
`docs/gtao.md` had recorded this as known-wrong. The main pass now samples the AO
target at **set 0 binding 12**, reprojects along the motion vector it already
computes for TAA, and multiplies only into `ambient`.

Reprojection works with no ping-pong because **GTAO runs after the main pass**, so
the persistent AO target still holds the previous frame's result when the main
pass samples it. A depth prepass was rejected: GTAO also needs normals from the
thin G-buffer the main pass writes, so a prepass would submit all geometry twice.

`SsaoSettings::ambientOnly` (default true) selects it; false restores the
composite multiply as an A/B reference.

## VISUALLY CONFIRMED 2026-08-10 — closed

The user did the A/B with manual exposure and GTAO intensity 3. With
`Ambient only` **off**, every object sits in a large black smear on the ground,
including areas in full direct light — exactly the artefact the fix targets, and
worse than "subtle" once intensity is raised. With it **on**, the smears are gone.

Second finding from the same A/B, worth knowing: **with it on, GTAO is barely
visible in the default scene at all.** That is correct, not broken — the demo
scene is dominated by one strong directional light, so the ambient term AO now
modulates is a small fraction of the image. The Cornell box preset (sun off, one
overhead light, probes on) is where ambient-only AO can actually be seen.

Setup that made the A/B readable, reusable for any visual comparison here:
manual exposure (auto-exposure actively compensates the thing you are trying to
see — it moved 0.5080 -> 0.5376 across this toggle), effect strength cranked,
camera still, light animation off.

## Two hazards this exposed — both general, both worth remembering

**1. The material descriptor set had no resize path.** Binding 12 is the *first*
entry in it backed by a swapchain-sized image; shadow map, punctual atlas, fog
volume, and probe atlases are all sized by something else and survive a resize.
So nothing had ever rewritten material sets when post-process targets were
recreated, and the high-DPI startup resize left every material holding a
destroyed view (20 VUIDs).

Fix: `Renderer::refreshMaterialAmbientOcclusionDescriptors()` rewrites **only that
binding**, called at the end of `recreatePostProcessResources()`. **Do not fix
this by re-running `createMaterialDescriptorSet`** — it calls
`vkAllocateDescriptorSets`, so each resize would leak a set out of a fixed pool
(`kMaxMaterialDescriptorSets`) until allocation fails.

**Anything else swapchain-sized added to that set needs the same treatment.**

**2. Three passes share `simple_bindless.frag`**, so all of them sample every
binding it declares: MainHDRPass, MainHDRPhase2, and TransparentPass. Transparent
runs *after* GTAO, when the AO image is back in `COLOR_ATTACHMENT_OPTIMAL`, so it
needed its own `readTexture` declaration to get the layout transitioned back.
The pass already carried a comment about the same problem for `normalRoughness`
("Written because the shared fragment shader emits it") — that comment is the tell
for this class of bug.

First-frame handling mirrors `taaHistoryValid_`:
`PostProcessStack::ambientOcclusionHistoryValid()` is false until GTAO has written
once since creation, and the main pass leaves occlusion at 1.0 until then. Covers
resize as well as startup.

See [[runtime-settings-persistence]] for the 5-place settings wiring this used.
