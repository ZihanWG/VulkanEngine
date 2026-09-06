---
name: realtime-gi-initiative
description: "Real-time GI via irradiance probes — SHIPPED (merge commit f143f08 on main, pushed; off by default); no hardware RT on this device so DDGI is impossible; capture is deterministic so accumulation needs jitter"
metadata: 
  node_type: memory
  type: project
  originSessionId: 30ea2028-e930-4922-9e8f-d02ba071e94d
  modified: 2026-08-06T12:37:40.393Z
---

Real-time global illumination. Started 2026-08-02; picked as the last and largest
item from the 2026-08-02 engine review — lighting is IBL + SSR + GTAO (all
screen-space), so offscreen geometry contributes no indirect light.

**Hard constraint, established by probing the device — do not re-derive.** This
target reports neither `VK_KHR_ray_query` nor `VK_KHR_acceleration_structure`
(131 extensions, none RT). MoltenVK does not expose them. **DDGI as normally
implemented, with ray-traced probe updates, is impossible here.** Technique is
therefore irradiance volumes / probe relighting: gather probe radiance by
*rasterising* the scene from each probe, then convolve.

**Second constraint, found 2026-08-04 — do not re-derive.** The existing IBL
convolution is entirely **CPU-side**: `createProceduralDiffuseIrradiance` and
`createDiffuseIrradianceFromRgba32fFaces` call `makeHdrDiffuseIrradianceFaces` on
CPU pixel spans and upload the result (`VulkanEnvironmentMap.cpp` ~921, ~1102).
So the capture phase can reuse the *math* but not the *pipeline* — feeding a GPU
capture through it would need a readback per probe. Phase 2 must write its own
compute convolution (cubemap → octahedral tile). Bigger than "reuse IBL" implies.

**MERGED AND PUSHED** 2026-08-05: `f143f08` (`--no-ff` into main, pushed to
origin). main == origin/main. Verified on main before pushing: Debug + Release
build, 150/150 both, 0 VUIDs in a validation run.

All 13 feature/claude branches were then deleted (every one fully merged). Only
`main` and `claude/zealous-greider-e81db1` remain — the latter kept because its
worktree has an uncommitted `src/renderer/Renderer.cpp` change. The main repo
directory is now checked out on `main`. `origin/feature/async-compute-clustered`
is merged and could be deleted remotely; the user was asked and had not answered.

Done:
- `7818ace` GPU-free core: octahedral mapping (reusing the shaders' convention,
  pinned by a transcription test), grid index/position round trip, trilinear
  blend weights.
- `b3091fd` Phase 1, probe atlas storage: two atlases (irradiance RGBA16F 8x8
  core, depth RG16F 16x16 core), `IrradianceProbeVolume`, `probe_debug_fill.comp`
  + `probe_border.comp`, render-graph + settings + debug-panel wiring, 5 more
  tests. 139/139. Off by default, GPU-verified (0 VUIDs).
- `0249a90` Phase 2, capture + convolve: `probe_capture.vert/.frag` rasterise six
  16x16 cube faces per probe into a capture atlas, `probe_convolve.comp` turns
  them into the octahedral tiles. Round-robin, 4 probes/frame default. 145/145.
  Debug ~0.45ms capture + 0.23ms convolve at 8 probes/frame.

Two things worth keeping from Phase 1:
- **The octahedral square folds onto its own edge**, mirrored about that edge's
  midpoint — border row = adjacent core row reversed, border corner = diagonally
  opposite core corner. Torus/equirectangular wrapping is the plausible mistake
  and is wrong by most of a hemisphere.
- **Absence of validation errors proved nothing about the ImGui preview**: the
  preview image is clipped, so a deliberately wrong layout claim did not trip
  validation either. Verified the cold-start layout by reading back the layout
  pointer the render graph writes instead. See
  [[instrument-before-guessing-runtime-bugs]] — the negative control is what
  caught it.

Phase 2 decisions worth not re-deriving:
- **Cube texel solid angle is required**, not a refinement: corners subtend ~3x
  less than centres, and omitting the Jacobian biases irradiance smoothly.
- **Capture clears to the ambient/sky term, never black** — outdoors most of a
  probe's hemisphere is sky.
- Capture shading is albedo + lambert + CSM + ambient + emissive only. **Punctual
  lights are genuinely missing** (cluster lists are camera-froxel-based), so a
  room lit only by spots gathers no indirect light.
- Face resolution 16 is set by the convolution's inner loop, not the raster.

- `7e47647` Phase 3, shading lookup: `simple_bindless.frag` blends the 8
  surrounding probes with Chebyshev visibility + wrapped-cosine backface
  rejection, renormalised. Grid params in a UBO at set 0 **binding 11** (push
  constants had only 24 of 32 bytes free; ObjectFrameData was rejected because
  its layout is duplicated in 6 shaders and the array stride depends on it).
  Atlases at bindings 9/10. 148/148.

Phase 3 decisions:
- Probe irradiance **replaces** the constant IBL irradiance, not added — both
  answer the same question.
- `intensity == 0` is the off state (no separate flag), matching `fogMaxDistance`.
- `abs()` on the Chebyshev variance is required: bilinear-filtered moments can
  give meanSq < mean², and a negative variance *subtracts* light.
- The probe volume must be created **before** `createScene()` — material
  descriptor sets bind its atlases.

- `9077862` Phase 4, temporal accumulation + sub-texel capture jitter. 149/149.

Phase 4's key insight (do not re-derive): **this capture is deterministic** —
same 1536 fixed directions every update — so accumulation without jitter is a
no-op. Halton(2,3) sub-texel jitter on the capture, subtracted again in the
convolution. Hysteresis forced to 0 until the first full round-robin cycle
completes, or probes blend against their neutral seed forever. Default 0.7, not
DDGI's ~0.97, because a probe updates once per *cycle* not per frame.
`kProbeDepthLobeExponent` still 20: jitter reduces aliasing but does not
decorrelate captures the way a random rotation would.

- `bc8992a` Punctual lights in the capture. Read as a **flat array**, not via
  cluster lists — those index the camera's froxel grid and a probe has no froxel.
  Punctual shadows included (single tap, bias-before-face-selection). Probe-only
  luminance went 0.1497 static → 0.26-0.28 varying with the orbiting lights.

**Measured cost (Release, demo scene), probes/frame → capture / convolve median:**
4 → 0.006 / 0.263 ms; 16 → 0.333 / 0.468 ms. **The convolution dominates, not
the light loop**, so per-probe light culling is NOT justified — don't add it
without a new measurement. The lever is face resolution (convolution is
quadratic in it).

**GPU timings are identical in Debug and Release** — verified `cmp` says the
SPIR-V is byte-identical, since glslc runs the same regardless of
CMAKE_BUILD_TYPE. The "never trust Debug timings" rule applies to CPU work only.
CPU capture cull+record: 4 probes → Release 45us vs Debug 597us (13x).

- `57cba28` + `b168a01` Debug view bypasses exposure/tone mapping (auto-exposure
  cancels exactly the brightness change a GI debug view exists to show) and no
  longer clips at its own defaults.
- `8fc1f3f` `docs/irradiance_probes.md` (495 lines) + docs/README + root README.

- `4152f9a` Release timings + a CPU cull/record counter for the capture pass.

**The initiative is complete.** Off by default, 149/149 in both builds, 0 VUIDs.
User has visually confirmed: border seams (none), the debug view responds
linearly to intensity, and real indirect structure (soft gradients under spheres,
lit ground).

- `5039568` Multi-bounce: the capture reads the atlas it feeds. Indirect is
  `mix(ambient, probeIrradiance, w)` — **interpolated, not summed** (summing
  double-counts the sky and makes the feedback unbounded). Gain per round is
  `albedo * w`, series settles at `1/(1 - albedo*w)`; `probeBounceAmplification`
  is that, clamped. Weight 0 until the first cycle completes.
  **Measured +2.2% (2.6 sigma) at w=0.95 — stable and correctly signed, but
  barely above noise on this scene.** Not a proven win; documented as such.
  Also fixed: the params UBO was updated *after* the capture read it.

- `70cd478` **The Cornell box preset — DONE, do not re-offer it as future work.**
  `SceneBuilder::appendCornellBox` + `Renderer::loadCornellBoxScene`, reachable
  from the debug UI under **Scene Presets → Cornell box**, documented in
  `docs/irradiance_probes.md` (a full section, ~line 334).

  Loading it does four things beyond placing geometry, and it is useless without
  any of them: sun off (it floods the room through the open side), the orbiting
  swarm replaced by one overhead light (the swarm lights everything directly, and
  the floor between the blocks has to be lit by bounce alone), probe grid fitted
  to the interior and inset by 0.8 so no probe lands inside a wall, probes on.
  White is deliberately below 1.0 — bleeding scales with saturation and a
  perfectly white room is where bounce feedback stops converging.

  It settled the multi-bounce question that the open scene could not:
  colour bleed 34% R/G swing across the room; probe-only luminance 0.4517 →
  0.5920 (+31%) from bounce weight 0 → 0.95, against +2.2% on the demo scene.
  Colour separation widens 1.24x → 1.48x, which one bounce cannot do at all.

**Nothing is outstanding on this initiative.** Both items that used to sit in a
"remaining" list here — the Cornell box preset and multi-bounce — are implemented
and documented above; the stale list caused a later session to offer already-done
work to the user. If asked for GI next steps, the honest answer is that the
subsystem is complete and the open questions are elsewhere (see
[[gpu-profiler-nested-scopes]]: MainHDRPass is now ~57% of the frame).


