---
name: punctual-shadows-initiative
description: "Spot/point shadow atlas initiative — Phase 1 + 2a shipped on branch feature/punctual-shadow-atlas, not yet merged; point-light cube faces still open"
metadata: 
  node_type: memory
  type: project
  originSessionId: df3509ca-ddf6-43e5-9b25-e03a9a8bfe0d
  modified: 2026-08-02T10:17:46.833Z
---

Punctual (spot/point) shadow atlas. Started and finished 2026-08-02: 15 commits
on `feature/punctual-shadow-atlas`, **merged to main as d259bfe (--no-ff), user
confirmed the visuals, pushed to origin/main**. Picked
as the top gap from a 2026-08-02 engine review: clustered lighting could put
hundreds of punctual lights on screen but none of them cast.

Shipped so far (4 commits):
- Phase 1 (`8f209f1`): 4096x4096 atlas, 64 tiles of 512px, spot lights only.
  GPU-free core in `PunctualShadowAtlas.h` + 8 unit tests; `PunctualShadows.*`
  owns the image and per-frame BDA slot buffer; `shadow_punctual.vert` takes the
  slot view-projection as a push constant.
- `ce3841f`: fixed a pre-existing bug where BLEND geometry cast opaque CSM
  shadows, contradicting docs/transparency.md. Skipped at replay, not filtered
  from the caster list (filtering desyncs batch command ranges).
- Phase 2a (`3b753e3`): registered in the render graph; this also fixed a latent
  layout bug (cold start with the toggle off left the atlas UNDEFINED while
  binding 7 claimed DEPTH_READ_ONLY_OPTIMAL). Docs in `docs/punctual_shadows.md`.
- `11ec52f` + `f2a8e49` + `eb7e204`: diagnostics. The atlas preview alone was
  misleading (a plain ImGui::Image with a *multiplicative* tint cannot show
  perspective depth — a correct tile reads solid red, same as an empty one; the
  CSM previews escape this only because ortho depth is linear). Added a
  caster-draw counter and a greyscale shadow-term debug view, which together
  separate "pass broken" / "lookup broken" / "just subtle". `f2a8e49` also fixed
  a real defect found on the way: near plane was a fixed 0.05 vs far=range (320:1),
  crushing receivers to ~0.995; now range-relative at a fixed 50:1.
- `441ce3d`: slope-scaled the punctual bias. It was flat constant while CSM had
  always slope-scaled, so every surface edge-on to a spot had acne (visible as
  combed streaks).
- Phase 2b (`bd288bd`): point-light cube shadows. Six consecutive slots per
  light; face convention pinned on both CPU and GLSL by a unit test because a
  mismatch fails *plausibly*. Spots served first, then point lights
  nearest-camera-first under a visible budget slider (default 4).

- Phase 3 (`da4b2c8`): quadtree allocator with 1024/512/256px tile classes, and
  priority by projected pixel radius (reusing mesh-LOD's projScaleY) replacing
  distance-only ordering. Point lights are demoted one size class — without it
  one point light swallowed six of the largest tiles (demo: 13 slots/81%
  occupancy, only 2 of 4 budgeted lights served; demoted: 25 slots/43%, all 4).

Still open: no caching (every face re-renders every frame, even static light
over static geometry); CPU caster culling rather than the GPU shadow-cull path;
priority ignores occlusion and the view frustum; the atlas preview still cannot
show perspective depth (needs a linearising remap the shared helper has no place
for).

Agreed sequencing after discussing a swap: Phase 3's graph+docs half was pulled
forward into 2a because it was independent; the priority/budget half stays
*after* point lights, because with only spot lights the demo uses 1 of 64 tiles
and a priority system would be untestable and unmotivated. Doing variable tile
sizes before cube faces would also mean rewriting the allocator twice.

Verification: 115/115 tests, validation-clean, 25/64 slots at 43% occupancy,
~0.11ms. User confirmed the final visuals before merging.

Two lessons worth carrying, both reinforcing
[[instrument-before-guessing-runtime-bugs]]:

1. **Instruments lie too.** Twice a diagnostic was wrong before the renderer
   was — the atlas preview (a multiplicative ImGui tint cannot show perspective
   depth, so a correct tile looks identical to an empty one) and the shadow-term
   debug view (it skipped the range/cone gates shading applies, so it min()'d
   the frame to near-black). Adding a diagnostic without validating it just
   relocates the guess.
2. **Measure before building the fix.** I was about to build size-class
   hysteresis for assumed assignment thrash; a churn counter showed 4 events in
   ~800 frames. The measurement instead surfaced a different real defect
   (FLT_MAX collapsing the priority ranking).

Also: a CPU/GPU agreement test that feeds *the same input* to both sides only
validates the convention, never that the two sides receive the same input — the
cube-face seam bug lived precisely in that blind spot.

**Atlas caching (2026-08-02, merged `4fb8ec9`, pushed).** The pass skips
entirely when a content hash of its inputs is unchanged — no clear, no draws, no
transition (the atlas already holds the sampled layout the main pass wants).

Chose a **content hash over dirty flags** deliberately: a stale shadow surfaces
far from its cause, and dirty flags spread that risk over every mutable input
and every mutation site, while a hash concentrates it into one enumeration.
Hashed: per-slot view-projection + atlas rect, per-caster mesh/index range/
bucket/model matrix, raster depth bias. Two things a hash can't express are
handled separately — a recreated atlas image (undefined contents even when the
key matches) and the empty-scene early-return path.

**The bug worth remembering:** computing the hash alongside the slot assignment
reads the *previous* frame's transforms and draw items, because
updateAnimatedTransforms and buildDrawItems run after it — so every change is
noticed one frame late, i.e. a stale shadow frame on every edit. It must run
after both.

Verified both directions: animation frozen -> 293 consecutive cached frames, 0
draws; animation on -> 0 cached, re-renders every frame (correct — the demo's
lights orbit and objects rotate, so the cache never hits by default).

**Per-tile invalidation (merged `894616b`, pushed).** Each tile hashes its own
view-projection, rect, raster bias, and only the casters passing *that tile's*
frustum cull. Attachment loads instead of clearing; dirty tiles get one batched
vkCmdClearAttachments; first frame after image creation still full-clears.

Cache identity is the **tile rect, not the slot index** — slot indices shift
between frames as lights reorder by priority, but a tile is a fixed image
region. Rect packing is exact (not a hash) so tiles can't collide.

Measured: lights orbiting 25/25 tiles; frozen + static 0 (pass skipped); frozen
+ one moving object 9-10/25.

**Testing trap worth remembering:** the portfolio scene sets animateTransform
false on *everything*, so "static lights + animated objects" reads as fully
cached — correct, but it does NOT exercise the partial path. Verifying that
needed a deliberately animated object. Easy to mistake one for the other and
conclude a feature works when it was never exercised.

**GPU caster culling (merged `ea64a90`, pushed) — OFF BY DEFAULT, a
pessimisation at this scene's size.** One dispatch over every (slot, draw item)
pair; forced to be one dispatch because compute can't run inside a
dynamic-rendering scope and the atlas pass is one scope (for the partial clear).

Measured: CPU cull+record ~40us (release) and atlas 0.19ms with CPU culling, vs
atlas 0.33-0.43ms + 0.02-0.08ms cull with GPU. It moves work off the CPU and
makes the GPU *slower*, because MoltenVK lacks vkCmdDrawIndexedIndirectCount so
each slot submits a command per *candidate*, not per survivor. Pays off only
with many more draw items AND indirect-count support.

**Two traps recorded:**
1. **Debug timings are ~10x off.** First measured 350-440us in Debug and nearly
   argued from it; Release was ~40us. Never make an optimisation call on Debug.
2. **Enabling it silently reintroduces the BLEND-casts-shadows bug** (`ce3841f`)
   unless a separate per-draw-item caster mask is passed. The shared cull input
   carries no bucket and GpuCullDrawItem is exactly 64 bytes with no spare
   field; the CSM path filters blend at *replay*, which an indirect per-slot
   draw cannot do since commands are already compacted.

Still open: a tile is all-or-nothing internally (inherent to shadow maps);
frustum membership is conservative (an occluded caster still dirties a tile).
