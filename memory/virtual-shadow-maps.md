---
name: virtual-shadow-maps
description: "VSM initiative COMPLETE through Phase 3 and merged to main — marking, residency, page rendering, sampling, moving-caster invalidation, cutout casters, debug overlays; five design errors caught by measurement; page LOD measured and rejected; the lit-face gap is now narrowed to what the two paths RECORD, with the cascade side instrumented"
metadata: 
  node_type: memory
  type: project
  originSessionId: d67d313d-f0e1-4bd7-9bba-04e45cc3bfc1
  modified: 2026-08-19T12:26:29.725Z
---

## 2026-08-30: the cascade side is instrumented, and a tint was not evidence

**Review (PR #4) was right about a step I had written as proof.** The doc read
the `enableCascadeDebugColors` tint as "the cascade samples its map under the lit
face". It proves only *selection*: `selectShadowCascade()` picks by `vViewDepth`
alone, and `sampleShadowFactor()` returns 1.0 on the UV/Z bounds exit **without
ever fetching `uShadowMapCompare`**. Never-sampled and sampled-and-found-nothing
are the same tint, the same lit pixel, the same screenshot.

**Built the instrument instead of softening the wording** (`b2f7476`, MERGED to
main as PR #4 / `e5dc08f`, branch deleted;
`VsmSettings::debugCascadeDepthDelta`): the same 16-step bisection the page pool
gets, run against `uShadowMapCompare`, sharing the ramp so one pixel reads in
both views. It separates THREE outcomes, and the third is the one that matters —
**a cleared texel and a texel holding this very surface both bisect to delta 0**,
so "cleared" needs its own colour (near-black) or blue is ambiguous exactly where
the question lives.

**Result at the lit face** (32x32 patch at (1960,1048), cascades 48.98 vs VSM
42.00 — record these coordinates, the doc never had them and finding them again
cost a diff sweep): bounds exit **not** taken, texel **not** cleared, 1024/1024
blue. The cascade holds **the face's own depth**, within 0.01 m; the page pool
holds something 0.25–1 m in front. So the gap is entirely in what the two paths
RECORD. Controls: two captures byte identical (0/3686400); the same view returns a
real occluder at 854/1209 genuine umbra pixels; reading unchanged `--vsm off` vs
`--vsm shadows` (so the cascade array is not stale under VSM).

**New trap, same family as the out-parameter hazard:** the debug flag was first
assembled inside the `isVsmDirectionalShadowActive()` branch, so with VSM off —
the configuration the view exists to interrogate — the word stayed zero and the
capture was **silently the ordinary shaded frame**, which reads exactly like a
debug view finding nothing everywhere. `compare_images` against the untinted
capture caught it (0 differing pixels where a hue-replacing view must differ in
all of them). **Make that the first control on any new debug view here.**

## 2026-08-23: gated, and the gate found two bugs

**The doc's "no pixel gate is possible" was FALSE and had been since before VSM
started.** `--deterministic` (`92feaac`) predates the first VSM commit; the control
that measured "8.5% differ between identical runs" had simply been run without it.
With it, on MoltenVK: **3 runs of cascades and 3 runs of VSM are each byte-identical
(0/3686400)** — including VSM, whose page residency is cross-frame state. Don't
repeat the old claim.

**Bug 1 — every umbra leaked ~15% of the sun.** The sampler subtracted
`CsmSettings::depthBiasConstant` (0.002) from the *page's* normalized depth. That
constant is a fraction of a cascade's ortho box (tens of units); a page's depth axis
is `2*depthRange` = 500 units. Same number, ~250x the world-space bias. Fixed by
expressing it in **texels of the sampled clipmap level** (`VsmSettings::depthBiasTexels`,
default **64**, swept: <32 self-shadows lit surfaces, >=128 starts leaking again).
Umbra 39.40 -> 31.86 against a cascade reference of 31.84.

**Bug 2 — `--vsm mark` produced 11 validation errors a frame**, as old as the pool:
marking allocates the pool, no page pass transitions it, and the material set binds it
anyway, so every draw touched a descriptor promising DEPTH_READ_ONLY on an UNDEFINED
image. Nothing could *select* that configuration until `--vsm` existed. Fixed with a
one-shot explicit barrier (the pool is not a graph resource in that mode).

**Tooling:** `--vsm off|mark|render|shadows` (startup-only marking gates a 64 MiB pool,
so it cannot be a setter) and `tools/agent/vsm_ab.sh`, which runs both controls first
and refuses to report a comparison whose control did not reproduce. **Not a CI job** —
lavapipe is not pixel-deterministic.

**Read the tolerance-3 diff, never tolerance-0**: a shadow change moves scene
luminance, auto-exposure follows, and ~87% of "differing" pixels are a delta of 1 in a
smooth gradient.

**Cutout page shadows DO resolve their holes** (the perforated panel is in the default
startup scene, no ImGui needed): the difference traces one arc per hole and VSM reads
*brighter* there (60.06 vs 59.53), which a solid silhouette could not do.

**`texelsPerPixel` question ANSWERED — not a bug (PR #23).** `vsmSelectLevel` returns
`max(quality, coverage)`, so a finer request only survives where coverage is not
already the larger bound — i.e. **inside level 0's 1.75 m reach** at the defaults
(`(kVsmPagesPerLevelAxis/2 - 1) * pageWorldSize(0)`). The default scene has nothing
that close (`L0=0`, levels touched 1..5), so 0.25 and 1.0 give the identical 99-page
set there; at 1.6 m they genuinely diverge (quality L1 vs L0). Above 1.0 it always
works (2.0 -> 97 pages, 4.0 -> 37, 8.0 -> 18).

**Review caught me over-generalising this**: I first wrote it as "below 1.0 is
clamped", which is a property of THAT SCENE's geometry, not of the setting. Codex
supplied the 1.6 m counter-example and the arithmetic confirmed it. Watch for this
shape of error — a measurement on one scene stated as a semantic.

**The ~6.5/255 lit-face discrepancy is STILL OPEN, but SEVEN hypotheses are dead** —
don't re-run these: VSM bias 2..128 texels (57.09 throughout); VSM level via tpp
0.25..4.0 (57.09); cascade bias /10; cascade resolution 2048/4096/8192; cascade
zPadding x5/x25; PCF radius 0/1/2 on both paths; and **the slope-scaled bias term I
had named as the leading hypothesis — FALSIFIED**: face and umbra move together
(slope 512 -> face 60.43 but umbra already leaking 39.12), so the two surfaces have
similar N·L and no function of surface angle separates them.

Next instrument: **look at what depth the page covering that surface actually holds.**
The residency grid says where pages are, not what is in them. `--capture-include-ui`
now makes the debug panel reachable from a scripted run.

Side fact worth keeping: **shadow LOD moves the cascade umbra** (31.84 -> 35.72 with
`lod.shadowBias=0` or `forcedLod=0`) because pages draw authored geometry while
cascades draw the cull-selected level. Not the cause of the face, but a real
by-construction disagreement between the two paths.

**Scene caveat that limits every visual claim here:** the demo scenes' directional
shadows are low contrast (umbra 31.8 vs lit floor 80 of 255) because ambient and
punctual dominate. Judge by patch means from a reproducible capture, not by eye.


VSM for the directional light, chosen over Nanite/Lumen because it is the only
one of the three whose parts this engine already has and whose platform does not
block it. **MERGED to main and PUSHED (`c6ef279`, `--no-ff`, 6 commits; branch deleted).
All three CI pipelines green on that SHA, step lists read to confirm they really
ran** — Headless render did compile shaders, render, gate on VUIDs and pass the
golden pixel compare (VSM off by default changes no pixel, as predicted);
`Shader constant parity` passed on both Linux and MSVC, which is what would have
caught the `FrameConstants` 512→624 layout change diverging from GLSL.

Marking, residency, page rendering and sampling all work; three nested toggles
(`enableMarking` → `enablePageRendering` → `enableShadows`), all off by default,
cascades still rendering underneath so the A/B is one checkbox.

**`enableMarking` is STARTUP-ONLY**: it gates whether the 64 MiB page pool
(4096² D32) + ~2 MiB of cull buffers are allocated at all. They were briefly
allocated unconditionally — a worse trade than the 17.48 MiB bloom-aliasing
default this project already rejected — and that was caught in the pre-merge
review, not by a test.

**Moving casters FIXED and MERGED (`2394308`).** The first merge was
static-geometry-only. The fix runs from the objects, not the pages: one `ShadowCacheKey` per object (model matrix + each draw
item's mesh/material/range/bucket), and an object whose key moved drops the depth
of every page overlapping **both** its previous and current world bounds. Static
scene = zero work. Load-bearing details: hash the **model matrix not the AABB**
(a rotating symmetric object has a fixed AABB); use **all 8 corners** for the
light-space rect (the basis is a rotation); invalidate **every level** (marking
requests one coarser as a fallback, so a stale coarse copy would survive).

**No scriptable scene animates a caster** — only demo lights move, and the
skinned mesh is not a `RenderObject` so it is not a caster on any shadow path.
Verified by temporarily patching one object to drift (patch not committed):
84 pages invalidated → 84 redrawn → 14 still cached, exact match.

**First page-pass cost, from that pathological run** (one run, n=6, no A/B — an
order of magnitude, not a measurement): `VsmPageMark` 0.32, `VsmPageCull` 0.04,
`VsmPagePass` 0.70 ms for 84 pages, vs `CSMShadowPass` 0.19 ms same frame.

**Spot-checked by eye (2026-08-20) and nothing was wrong** — no holes, seams or
acne on the default scene A/B — but that is one static camera with much of the
shadowed ground behind the debug window, **not a gate**. No pixel gate is
possible on the default scene: two captures of the SAME config at the same frame
differ by 8.5% because `updateDemoLights(elapsedSeconds)` animates the light
swarm on wall-clock time. That **invalidates the control**, so the 53% measured
between CSM and VSM captures means nothing. This qualifies
[[ibl-precompute-cost]]'s "default scene is pixel-reproducible" claim — it is
not, with the animated punctual lights running. A gate needs a fixed timestep or
an unanimated scene.

**Four design errors were caught by measuring before building, which is the whole
reason for the phased order. Phase 1's two:**

1. **`kVsmPagesPerLevelAxis = 8` produced ZERO requested pages** on the geometry
   stress scene at texelsPerPixel 0.25 (38 at 1.0). A finer level has a *smaller*
   window, so every page the request selected fell outside its own level's
   window — in the sampling phase that is no shadow, not a blurry one. The bound
   is `axis >= 2 * projScaleY / kVsmPageSize` (~12 at 1080p) → axis = **16**, and
   `vsmSelectLevel` now returns `max(quality, vsmMinLevelForCoverage)`.
   Consequence: **effective resolution is capped by the axis, not by
   `level0Extent`** — turning level0Extent down past that buys nothing.
2. **The depth pyramid now has two consumers.** The occlusion-yield controller
   suspends the pyramid build when occlusion culls nothing, and the skip calls
   `depthPyramid_.invalidate()`. Marking measured one frame correctly then
   reported zero forever. `isDepthPyramidBuildRequired()` returns true whenever
   marking is active. **Anything else that starts reading the pyramid must do the
   same** — see [[occlusion-yield-initiative]], whose "only consumer is occlusion
   culling" premise is now false.

**Measured (M3, 1080p, validation on, preset cameras, counts identical across
every report block):** default scene **99** pages (L1..L5), geometry stress
**60** (L4..L6) at both tpp 0.25 and 1.0, **25** at tpp 4.0. Page count tracks the
**screen, not the scene** — 2322 draw items ask for fewer pages than 11 do.
`kMaxVsmPagesPerFrame` raised 64 → 128 on the strength of the 99.

Design notes worth keeping: the page grid is **absolute** in light space (not
camera-centred) or nothing caches; slot = absolute coord **mod axis** (toroidal),
or a one-page scroll renumbers all 256. Marking reads the **previous** frame's
depth + the VP it was built with. `sizeof(PushConstants)` is **exactly 128 and
full**, which is why the page-table address went into `FrameConstants` instead.
Global set 0 was used to binding 13; VSM took **14** (the pool as a
`sampler2DShadow`, sharing the cascades' immutable compare sampler).

**Phase 2 findings.** The page table stores each page's **absolute coordinates**,
not just residency — a toroidal slot names a different page after a scroll, so
residency alone would hand back depth rendered elsewhere; `rendered` is tracked
separately because allocation and drawing are different steps. What identity
cannot catch (light moved, clipmap settings, scene switch) residency detects by
**hashing its own inputs**, because the debug UI drags the light direction every
frame and there is no edit site to hook.

Two more measured corrections: (1) the page pipeline was created **before** the
pool existed and was silently skipped — the pool is fixed-size and belongs beside
`createShadowMap()`, ahead of `createPipeline()`; the symptom was invisible until
the page-draw counter was made **cumulative** (a warmed cache draws 0/frame).
(2) `VsmPageMark` cost **0.90 ms** because each thread scanned every pixel of its
block; capping at 2x2 taps gave **0.41 ms at stride 8 with identical page
counts**, so the default stride is now 8.

Two Vulkan traps: binding 14 needs `DEPTH_READ_ONLY_OPTIMAL` (not
`SHADER_READ_ONLY_OPTIMAL`) — 30 validation errors; and its fallback view must be
`shadowMap_.layerImageView(0)`, never the array view, or
VUID-vkCmdDrawIndexed-viewType-07752.

Steady state on the default scene: 99 pages, all resident, **all cached, 0 drawn
per frame** after the cold start. Observed live: **`L0=0`, levels touched 1..5** —
at a 4 m level-0 extent that level's window reaches only 1.75 m, so the coverage
bound rules it out and the advertised 2 mm finest texel is unusable at normal
camera distances. Also: when A/B-ing, the overall tone difference between two
looks is the **animated light swarm**, not the shadows (the sky changes too, and
the sky samples no shadow) — toggle within a second to isolate. `FrameConstants` grew 512 → 624 bytes;
`PushConstants` could not grow at all.

**Phase 3 COMPLETE and fully merged** (`2394308` invalidation, `c7c470a`
overlays + key audit, `17da211` cutout). Cutout casting works by giving each page a region per
CASTER BUCKET (opaque / masked) and drawing each page twice — forced by the lack
of indirect-count, since each draw submits its region's whole stride and a shared
region would run cutout commands through the opaque pipeline. Bucket rides the
existing per-draw-item flag array (`GpuCullDrawItem` is exactly 64 bytes, no
spare field). Unverified visually: the only perforated geometry is the portfolio
showcase panel, an ImGui preset not reachable via `--scene`.

**Page LOD MEASURED AND REJECTED — do not re-propose without a profile.** In the
steady state `VsmPagePass` never appears in the profiler at all (everything
cached, nothing redrawn), so there is no time to save; the only run with work was
the pathological moving-ground-plane experiment. And the saving would land on the
coarse FAR pages, because LOD selection and clipmap level are both driven by
distance to camera — the fine near pages select level 0 either way.

Remaining gaps: bindless path only (`simple.frag` still uses cascades), skinned
geometry casts no shadow on any path, alphaCutoff edited *within* MASK leaves a
stale cutout shadow (bucket changes are hashed, the value is not).

See `docs/virtual_shadow_maps.md`. Related: [[shadow-cascade-cost]],
[[gpu-profiler-nested-scopes]], [[back-to-back-or-dont-claim]].
