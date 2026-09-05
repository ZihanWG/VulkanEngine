# Punctual Shadows (Spot / Point Shadow Atlas)

Clustered lighting could put hundreds of dynamic point and spot lights on
screen, but none of them cast: the only shadow source was the directional CSM.
This subsystem gives punctual lights a shared depth atlas so spot lights shadow
the same geometry the sun does.

Both spot and point lights cast. A spot takes one atlas tile; a point light
takes six, one per cube face.

## Why an atlas rather than per-light shadow maps

A shadow map per light means a texture, a view, and a descriptor per light,
which does not survive a light count in the hundreds. One atlas keeps it to a
single image and a single descriptor binding, and turns "which lights cast this
frame" into a tile-allocation problem instead of a resource-allocation problem.

The atlas is 4096x4096, handed out in power-of-two tiles of 1024, 512, or 256px
by a quadtree allocator. A light dominating the view gets four times the
resolution of one barely visible, instead of both getting the same fixed tile.

## Tile allocation

The allocator starts each frame with the atlas cut into largest-class tiles.
Allocating a smaller class splits a free larger tile into four and banks three
for later, so neither class has a reserved region it cannot lend out — a frame
wanting a few big tiles and many small ones packs either way.

Nothing is freed mid-frame and the whole structure is rebuilt every frame, so
there is no coalescing pass and no fragmentation that outlives a frame.

Cube faces reserve **all six or none**, via a snapshot-and-roll-back. A light
that got four of six faces would sample cleared tiles for the other two, which
reads as light leaking through solid geometry — worse than not casting at all.

Because tiles now vary in size, a slot's rect no longer follows from its index.
The rect travels with the slot record and the atlas pass reads it back for the
viewport/scissor. Slot *indices* stay sequential, which is what keeps the float
encoding and the consecutive cube-face property working unchanged.

## Priority

Lights are ranked by **projected pixel radius** — `range / distance *
projScaleY`, using the same `projScaleY` the mesh-LOD selection uses, so "how
big does this look" means one thing across the engine. That single number drives
both who gets a tile and how big a tile they get.

This is what distance-only ordering could not express: a large light far away
can outrank a small one nearby, and it should.

A light whose radius encloses the camera needs separate handling, because
`range / distance` stops describing a footprint there. It ranks by how deeply it
encloses the camera instead, on a scale that starts exactly where the
non-enclosing branch ends — so a light drifting across its own radius does not
pop between size classes.

Saturating all such lights to one value instead looks harmless and is not: they
tie, the tiebreak falls back to light index, and priority silently degenerates
into "whichever lights come first". In a scene of overlapping lights that is
most of them, which makes the whole ranking inert.

Point lights are **demoted one size class**. A point light buys six tiles rather
than one, so ranking it against a spot on the same threshold lets a single point
light swallow six of the largest tiles and starve everything behind it. Each
cube face also only covers 90 degrees of the sphere, so a face genuinely
warrants less resolution than the light's whole footprint implies — the
demotion is a coarse stand-in for that, not purely a budget hack. On the demo
scene it is the difference between 2 point lights at 81% occupancy and 4 at 43%.

The sort orders *indices*, never the light array itself — the cluster
light-culling pass indexes into that array by position, so reordering it would
reassign every froxel's light list. Ties break on light index so equally-ranked
lights do not shuffle between frames and flicker their shadows.

## Point lights: six faces, one budget

A point light is shadowed by six 90-degree faces, one per signed axis, packed
into **consecutive** slots. Consecutive is load-bearing: the light stores only a
base slot, and the shader adds the face index it derives from the
light-to-fragment direction. `allocateRange` therefore fails whole rather than
partially reserving, so a light can never get a truncated cube.

Six tiles per light means the atlas oversubscribes as soon as a swarm shows up,
so how many point lights may cast is capped by a visible **Max shadowed point
lights** slider rather than an implicit limit. Which ones is decided by the
priority ranking above.

### The face convention is the thing to get right

The CPU builds one projection per face and the shader picks a face from the
light-to-fragment direction. If those two disagree, every point light samples
the wrong tile — and that fails *plausibly*, as shadows in roughly the right
place with the wrong contents, rather than as an obvious crash.

So the order (+X, -X, +Y, -Y, +Z, -Z) and the tie-breaking `>=` comparisons are
written identically in `pointShadowFaceIndex` and its GLSL mirror, and a unit
test pins the invariant directly: for directions all over the sphere, the face
selection picks the face whose projection actually contains that direction, and
all six faces are exercised.

That test has a blind spot worth naming, because a real bug lived in it. It
feeds the *same* position to both sides, so it can only catch a disagreement in
the convention itself — never a disagreement in the **input**. The shader
originally picked the face from the unbiased world position but projected the
normal-offset-biased one, and near a boundary the offset pushes a sample into
the neighbouring face's cone. It then projected outside the selected face's
frustum, the bounds test read that as "outside the light", and every face
boundary got a bright fully-lit seam.

The fix is structural rather than a correction: the bias is computed first and
one position drives both the face selection and the projection, so the two
cannot take different inputs. (Reading the normal bias before the face is known
is exact, not an approximation — all six faces of a light carry identical
params.)

## Pass structure

```
CSMShadowPass            directional cascades, unchanged
PunctualShadowAtlasPass  one rendering scope over the whole atlas
MainHDRPass              samples both
```

The atlas pass opens **one** `vkCmdBeginRendering` covering the entire atlas and
isolates each slot with a viewport and scissor. That means:

- the depth clear runs once for the whole image rather than per tile, and
- slots that got no casters stay cleared at the far plane instead of holding a
  previous frame's depth.

Per-slot the pass pushes that slot's light-space view-projection, then draws the
casters that survive a CPU frustum test against the slot's own frustum.

## Why a separate vertex shader

The CSM path precomputes a per-object light MVP for each of its four cascades
and indexes that array in `shadow.vert`. That does not scale to 64 slots — it
would mean 64 matrices per object in `ObjectFrameData`.

`shadow_punctual.vert` inverts the arrangement: the slot's view-projection
arrives as a **push constant**, one push per slot rather than per slot per
object, and is combined with the object's model matrix in the shader. The cost
is one extra matrix multiply per vertex; the saving is an `ObjectFrameData` that
does not grow with the atlas.

The push-constant block carries explicit padding, because GLSL rounds the `mat4`
up to its 16-byte alignment and it therefore lands at offset 16, not 8, after
the buffer-device-address field. `PunctualShadowPushConstants` static-asserts
that offset so the two cannot drift.

## Slot encoding

The slot index rides in `GpuLight::spotScaleOffset.z`, which was unused padding
— the same trick the mesh-LOD work used to fit LOD indices into
`GpuCullDrawItem`. The light record stays 64 bytes and no varying or descriptor
had to grow.

The value is a float, so the encoding is pinned down in one place
(`punctualShadowSlotToFloat` / `punctualShadowSlotFromFloat`):

- a valid slot round-trips exactly — slot counts are far below the 24-bit
  exact-integer range of a float;
- anything else becomes `-1.0`, which the shader tests with a `< 0.0` compare.

**That sentinel is what guards the buffer dereference in the shader.** The CPU
stamps it into every light whenever the atlas is unavailable or the toggle is
off, and the slot-buffer address is only non-zero when at least one light got a
tile. So "this light has a valid slot" always implies "the slot buffer is
valid", and the shader needs no null-address test.

The decoder is written as a negated compare (`!(encoded >= 0.0f)`) so a NaN
decodes to unshadowed rather than sliding through into an out-of-bounds fetch.

## Sampling

`punctualShadowFactor` in `simple_bindless.frag` runs *after* the cheap
rejections in `evaluatePunctualLight` (range, cone, zero attenuation), so fully
attenuated fragments never pay for the atlas fetch.

It offsets the sample position along the surface normal before projecting.
Normal-offset bias fixes acne more cheaply than a large constant depth bias and
does not detach contact shadows the way peter-panning bias does; both biases are
per-slot in `GpuShadowSlot::params` and tunable live from the debug panel.

Both biases are **slope-scaled** by the sine of the angle between the surface
and the light, matching the shape `shadowDepthBias` uses on the CSM path. That
term is not optional: one shadow texel spans the most depth at grazing
incidence, so a flat constant bias leaves acne on every surface edge-on to the
light. Under a downward spot that is the vertical side of every object, and it
reads as combed streaks along the shadow boundaries. Keeping the head-on term
small is what stops the fix from turning into peter panning.

Fragments that project outside the slot's frustum return "lit". For a spot that
is exactly the region outside the lit cone, where the falloff has already
reached zero, so it changes nothing visually.

Filtering is a 3x3 PCF in atlas UV space, with every tap **clamped into the
slot's own tile**.

That clamp is load-bearing rather than defensive. Neighbouring atlas texels
belong to a different tile — for a cube face the adjacent face, and with the
quadtree allocator possibly an unrelated light — so a tap that walks out
compares against unrelated depth. On a spot this only ever happened at the cone
edge, where the falloff is already zero, so it was invisible and the original
code skipped the clamp on exactly that reasoning. On a cube face the tile border
is the *middle* of the lit scene, and the same mismatch draws hard seams along
every face boundary.

## Render graph integration

The atlas is an imported graph texture, and the pass declares a depth-attachment
write on it; `MainHDRPass` declares the matching sampled read. The graph infers
the barriers in both directions, so there are no hand-written image barriers.

One subtlety is load-bearing: the **write** pass is only declared when a light
actually got a tile, but the **read** is declared whenever the atlas exists. A
frame that casts nothing therefore still gets the transition into the layout the
material descriptors record. Without that, a cold start with punctual shadows
disabled would leave the image in `UNDEFINED` while binding 7 claimed
`DEPTH_READ_ONLY_OPTIMAL`.

## GPU caster culling

Optional, off by default, and honestly a pessimisation at this scene's size.

The CPU path frustum-tests every draw item against every slot, which is
`O(slots x draw items)`. The GPU path replaces that with **one** dispatch over
every (slot, draw item) pair, writing each slot's survivors into its own region
of an indirect buffer so the atlas pass issues one draw call per slot.

One dispatch rather than one per slot is forced by where it has to run: compute
cannot be recorded inside a dynamic-rendering scope, and the atlas pass is a
single scope so its cached tiles survive the partial clear. Everything is
therefore culled before the scope opens — and once that is true, batching it
into one dispatch is both less code and avoids per-slot dispatch overhead.

**Measured on the demo scene**, which is why it defaults off:

| | CPU cull + record | Atlas GPU | Cull GPU |
| --- | --- | --- | --- |
| CPU culling | ~40us (release) | 0.19ms | — |
| GPU culling | lower | 0.33-0.43ms | 0.02-0.08ms |

It moves work off the CPU and makes the GPU pass *slower*. The reason is the
missing `vkCmdDrawIndexedIndirectCount` on MoltenVK: without it each slot submits
the full draw-item count as indirect commands and relies on the zeroed ones being
no-ops, so a slot that culls everything still costs a command per draw item. The
CSM path has the same limitation for the same reason.

So it pays off when two things hold, neither true here: enough draw items that
the CPU cost stops being ~40us, and a platform with indirect-count so the draw
side scales with survivors rather than candidates. It is kept as a toggle so the
comparison is something you run rather than something you argue about, and the
panel reports the CPU cost it would remove next to the GPU cost it would add.

**Both of those conditions now hold on the Windows/NVIDIA target, and the CPU
number is much larger than ~40us there.** On `--scene stress` (2322 draw items)
with an RTX 3080 Ti Laptop, which does expose `vkCmdDrawIndexedIndirectCount`,
three runs of each configuration measured the atlas pass's *recording* time:

| | Atlas unit record CPU | Whole-frame record CPU |
| --- | --- | --- |
| CPU culling | 0.313 / 0.335 / 0.340 ms | 0.487 / 0.534 / 0.542 ms |
| GPU culling | 0.019 / 0.019 / 0.022 ms | 0.194 / 0.206 / 0.247 ms |

That is a 94% cut to this pass's recording and a ~60% cut to the frame's total
recording, on a scene where CPU frame work already exceeds the GPU frame. The
`~40us` in the table above was the demo scene; this pass is the single largest
recording cost in the frame once the scene is big enough to matter, which is what
`Record CPU by unit` in the log now shows directly.

**The blended-caster trap described below is already handled, and the note about
"no spare field" is stale.** The mask exists: `punctualShadowCasterFlags_` is built
in `RendererRecord.cpp` (blend, null mesh and empty index range all map to 0), it
travels in its own `CasterFlagBuffer` at set 0 binding 4 rather than inside
`GpuCullDrawItem`, and `punctual_shadow_cull.comp` rejects on it before anything
else. The 64-byte record never had to grow.

**What blocks the default is an unexplained image difference on one scene.** With
`--channel-tolerance 1 --max-differing-fraction 0` at frame 30, GPU culling is
bit-identical to CPU culling on `--scene stress`, `--scene cornell` and
`--scene sunlit`. `--scene default` moves 23715 of 921600 pixels (2.6%) with a
peak channel delta of 27, and the difference is spread in jagged patches wherever
punctual shadows land rather than being one object's silhouette. Four candidate
causes have been ruled out by reading the code:

- the blended-caster mask (present and correct, above);
- a masked/alpha-tested caster pipeline (there is only one punctual shadow
  pipeline, so both paths draw cutouts as solid silhouettes);
- LOD divergence (`punctual_shadow_cull.comp` emits the authored
  `indexCount`/`firstIndex` and does no LOD selection, unlike `cull.comp`);
- draw-item index mismatch (the caster flags, the shadow cull input and the
  shader all index `allDrawItems_` in the same order);
- the frustum test (both are the same positive-vertex test with the same
  `dot(n, v) + d < 0` rejection; the CPU's extra `!bounds.valid()` early accept is
  matched on the GPU side by the builder substituting a `kUnboundedCullExtent` box
  that passes every plane);
- a stale cull input (`buildShadowFrameData` runs and uploads unconditionally in
  prep, not gated on the cascade cache);
- the skinned caster (`recordSkinnedPunctualCasters` runs after the slot loop, so
  both paths draw it);
- non-determinism from the compaction atomics or a `slotCommandStride` overflow
  dropping arbitrary survivors: two GPU-culled runs of `--scene default` in
  `--deterministic` mode are bit-identical to each other and differ from the CPU
  reference by exactly the same 23715 pixels.

### The cause: the indirect path binds one mesh for a whole slot

The culling is not what differs. Replicating `punctual_shadow_cull.comp`'s accept
test on the CPU against the same uploaded inputs and diffing it against the CPU
replay loop's own test, per slot, gives an empty symmetric difference on every
scene: `cpuOnly=0 gpuOnly=0`. Both paths accept exactly the same casters.

The difference is on the draw side, and it is this:

```cpp
// RendererRecord.cpp, the GPU-culled branch of the slot loop
const renderer::Mesh* indirectMesh = allDrawItems_.front().mesh;
...
vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
vkCmdBindIndexBuffer(commandBuffer, indirectMesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
...
vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer, ..., drawCount, ...);
```

One `vkCmdDrawIndexedIndirect` covers every surviving caster in the slot, with the
vertex and index buffers of **the first draw item's mesh** bound for all of them.
Casters belonging to any other mesh are drawn with their own `firstIndex`,
`vertexOffset` and `indexCount` indexing into the wrong buffers, which produces
whatever geometry those offsets happen to name. The CPU replay loop rebinds per
item (`if (boundMesh != drawItem.mesh)`), so it does not have the problem.

This is the same constraint `renderer/DrawItemBatching.h` already states for the
main and CSM paths -- "a batch is one indirect draw with one pipeline bound", and
a run breaks on a change of mesh. The main and CSM indirect paths honour it by
walking `MeshDrawBatch`es. The punctual path does not batch at all: it compacts
per slot, and a slot is not a mesh.

Counting, per slot, the meshes among its accepted casters:

| scene | slots | draw items | distinct meshes | slots whose casters span >1 mesh | image |
| --- | --- | --- | --- | --- | --- |
| `default` | 25 | 11 | 2 | 10 | **differs** |
| `stress` | 25 | 2322 | 2 | 15 | identical |
| `cornell` | 6 | 7 | 1 | 0 | identical |
| `sunlit` | 1 | 8 | 2 | 0 | identical |

Spanning slots are necessary but not sufficient: `stress` has fifteen of them and
still matches, because there the wrong geometry does not land on anything the
camera can see. That makes it a latent bug there rather than an absent one, and it
is exactly the failure mode this repo keeps warning about -- silent, and dependent
on which scene you happen to test.

### The fix: write in place, replay per batch

The fix was an addressing change, not a filter. The shader used to compact
survivors into `slot * slotCommandStride + localIndex` with an atomic, which put
one slot's survivors in arbitrary order in one contiguous run and forced the draw
side to be a single indirect call for the whole slot. It now writes each survivor
**in place**, at `slot * slotCommandStride + item`. Two consequences:

- A mesh batch is a contiguous run of draw items by construction
  (`renderer/DrawItemBatching.h`), so its commands are now the contiguous range
  `[beginDrawItem, beginDrawItem + drawItemCount)` inside the slot's region. The
  replay walks `gpuShadowMeshDrawBatches_` per slot, rebinds when the mesh
  changes, and issues one indirect call per (slot, batch) -- the same shape the
  main and CSM indirect paths already use.
- The per-slot atomic counter is gone, and with it the visible-count buffer, its
  descriptor binding and its per-frame fill. Nothing was lost: with no
  indirect-count on the draw side the full range was always submitted and zeroed
  commands relied on as no-ops, so the total command count is unchanged.

Blended batches are now also skipped at replay. The caster mask already zeroes
their commands, so this is the cheaper of two guards rather than the only one --
batches are homogeneous in bucket, so skipping removes a whole draw call.

**Verified.** With `--channel-tolerance 1 --max-differing-fraction 0` at frame 30,
GPU culling is now bit-identical to CPU culling on all six scene presets
(`default`, `stress`, `cornell`, `sunlit`, `occlusion`, `fragment-stress`), each
with zero validation errors. `default` went from 23715 differing pixels to 0. The
CPU-culling default path is unchanged, byte for byte.

**The CPU win survives the extra draw calls.** Re-measured on `--scene stress`,
Release, three runs each:

| | Atlas unit record CPU | Whole-frame record CPU |
| --- | --- | --- |
| CPU culling | 0.317 / 0.318 / 0.354 ms | 0.492 / 0.495 / 0.546 ms |
| GPU culling | 0.021 / 0.021 / 0.021 ms | 0.201 / 0.204 / 0.208 ms |

One indirect call per (slot, batch) instead of one per slot costs nothing
measurable here, because the batch count is small.

**One latent crash came out with it.** The punctual GPU path borrows two products
of the CSM GPU shadow cull -- the shared cull input buffer, and now
`gpuShadowMeshDrawBatches_`, which the per-batch replay walks. Both are built only
inside `buildShadowFrameData`'s `isGpuShadowCullingActive()` branch, so turning
`punctualShadows.gpuCasterCulling` on while `renderer.useGpuShadowCulling` was off
reached `shadowCullInputBuffer().at()` on an unsized vector and threw "invalid
vector subscript". That combination has always crashed; the per-batch replay just
made the dependency load-bearing enough to notice. `isGpuPunctualShadowCullingActive()`
now requires the CSM path, so the combination falls back to the CPU caster loop and
renders identically instead of dying. Building those two products for whichever
consumer wants them, rather than only for the CSM one, would lift the restriction.

### Compaction granularity, and the two wrong answers before it

Getting this right took three tries, and the granularity is the whole story.

**Per slot** (original): survivors from a whole slot compacted into one run in
atomic order, so the draw side was one indirect call per slot -- and one indirect
call has one mesh bound. Wrong geometry whenever a slot's casters spanned meshes.

**In place** (the correctness fix): each survivor written at its own draw-item
index. That keeps a batch's commands in its own contiguous run so the replay can
rebind per mesh, but it gives up compaction: every slot then submits each batch's
whole range whether its casters survived or not -- on `--scene stress`, 25 slots x
2322 draw items, around 58k mostly-zeroed commands a frame against a few hundred
survivors. Correct, and expensive.

**Per (slot, mesh batch)** (current): survivors compacted into the front of their
own batch's region, counted by their own atomic. A batch's region is a contiguous
run of one mesh's draw items, so the replay still rebinds per mesh, and the
per-(slot, batch) survivor count is exactly what `vkCmdDrawIndexedIndirectCount`
consumes -- so the draw side fetches survivors, not candidates. The counter buffer
is slots x `kMaxGpuCulledBatches` (64 x 256 = 64 KiB); a frame with more batches
than that falls back to CPU caster culling rather than drawing something wrong.

Without indirect-count the replay still submits a batch's whole range and relies
on the zeroed tail, so that path is unchanged and still correct.

**Verified.** Bit-identical to CPU culling on all six scene presets at frame 30,
zero validation errors, and the CPU-culling default path unchanged byte for byte.

**CPU recording, `--scene stress`, Release, three runs each:**

| | Atlas unit record CPU | Whole-frame record CPU |
| --- | --- | --- |
| CPU culling | 0.320 / 0.321 / 0.349 ms | 0.511 / 0.531 / 0.567 ms |
| GPU culling | 0.022 / 0.022 / 0.025 ms | 0.196 / 0.205 / 0.250 ms |

**The GPU side is now measured.** With `tools/dev/gpu_clock.ps1` pinning the
graphics clock at 1400 MHz, an A/B on `--scene stress` passed the control-drift
gate at **0.30%** -- the first gate-passing GPU comparison on this machine, after
eight refusals whose best was 1.4%. Three A/B pairs, medians over 54 and 57
samples:

| Pass | culling off | culling on | delta | attributable |
| --- | --- | --- | --- | --- |
| Frame total | 1.186 ms | 1.207 ms | **+0.021 ms (+1.8%)** | yes |
| PunctualShadowAtlas | 0.035 ms | 0.036 ms | +0.001 ms | yes |
| PunctualShadowGpuCull | - | 0.019 ms | B only | - |

**The feared outcome did not happen.** The worry recorded above was that the
atlas pass would blow up the way it does under MoltenVK. It does not move. The
whole GPU cost is the cull dispatch itself at 0.019 ms, and it lands on the frame
very nearly one-for-one.

So the trade is finally a number instead of a shape: **+0.021 ms of GPU against
~0.31 ms of removed CPU recording**, on a scene whose CPU frame already exceeds
its GPU frame. Net favourable by an order of magnitude, in the direction the CPU
numbers predicted.

The same A/B on `--scene gpu-stress` was still refused, at 9.0% drift: that scene
drives the card to 87 C, `sw_thermal_slowdown` goes Active, and the clock falls
below the pin. See `docs/profiling.md`. So the number above is established for a
CPU-bound scene and **not** for a GPU-bound one.

**This still does not flip the default**, and the reason is no longer
measurement. Two things gate it: `tests/golden/lavapipe_frame30.png` has to be
regenerated on lavapipe, and the correctness trap below has to be closed first.
A favourable performance number is not permission to ship a wrong shadow.

**One trap worth recording.** The cull input buffer is shared with the main and
CSM paths and carries no bucket, and `GpuCullDrawItem` is exactly 64 bytes with
no spare field. The CSM path filters blended geometry at *replay* time, which an
indirect per-slot draw cannot do because the commands are already compacted by
then. Without a separate per-draw-item caster mask, enabling GPU culling
silently reintroduces alpha-blended geometry casting opaque shadows — a bug this
engine already had and fixed once.

## Caching

Every allocated tile used to re-render every frame, including for a static light
over static geometry. The pass now hashes everything it consumes and skips
entirely when that hash is unchanged: no clear, no draws, and no layout
transition, since the atlas keeps the sampled layout the main pass left it in
and the main pass wants that same layout next frame.

Invalidation is **per tile**, not per frame. A tile only cares about the casters
inside its own frustum, so one moving object re-renders the handful of tiles
that can see it rather than the whole atlas.

**Tile rect is the cache identity, not slot index.** Slot indices shift between
frames as lights are reordered by priority, but a tile is a fixed region of the
image. If the tile at rect R holds the render described by hash H, then any slot
this frame wanting rect R with hash H already has its contents there, whichever
light it belongs to. So the resident map is keyed by packed rect, and the
packing is exact rather than itself a hash -- two tiles colliding into one key
would let one tile's contents pass for another's.

Preserving cached tiles means the attachment loads instead of clearing, and the
dirty tiles are cleared individually with one batched `vkCmdClearAttachments`.
The first frame after the atlas image is created still clears everything, since
nothing in it can be trusted then; that also purges resident entries for tiles
the full clear wiped but did not redraw.

**A content hash rather than dirty flags.** The failure mode of caching a shadow
is a stale one, which shows up far from its cause and reads as a rendering bug.
Dirty flags spread the risk across every input that can change and every place
that changes one; hashing the inputs concentrates it into a single enumeration
that is either complete or not. What is hashed:

| Input | Why |
| --- | --- |
| Per-slot view-projection | Carries the light's position, direction, range, cone, and near plane |
| Per-slot atlas rect | Where the tile renders and at what resolution |
| Caster mesh, index range, bucket | Which geometry is drawn |
| Caster model matrix | Where it is |
| Raster depth bias settings | Pipeline state the tiles are rendered with |

Only the casters that survive *that tile's* frustum cull are hashed, and that
cull mirrors the one the recording pass performs. If the two ever diverge the
hash stops describing what actually gets drawn, which is the one way this can
silently go wrong.

The per-slot frustum cull is a pure function of the slot projections and caster
transforms, so its result needs no separate hashing. Floats are hashed by exact
bit pattern, so this is a strict "identical inputs" test and never approximate.

Two things the hash cannot express, handled separately:

- **A recreated atlas image.** Its contents are undefined even when the key
  matches, so `createShadowMap` drops the cache outright.
- **An empty scene.** That path returns before the hash is recomputed, so it
  clears the flag rather than inheriting the previous frame's answer.

**Where the hash is computed matters.** It runs after `updateAnimatedTransforms`
and `buildDrawItems`, not alongside the slot assignment. Hashing at assignment
time would read the *previous* frame's transforms and draw items, so every
change would be noticed exactly one frame late -- a single stale shadow frame on
every edit, which is both visible and hard to attribute.

The debug panel reports how many tiles were redrawn out of how many exist, plus
how many frames were skipped entirely, so the hit rate is measured rather than
assumed. Measured on the demo scene:

| Scene state | Tiles redrawn |
| --- | --- |
| Lights orbiting (the default) | 25 of 25 -- every projection changes, so nothing is reusable |
| Lights frozen, nothing moving | 0, the pass is skipped outright |
| Lights frozen, one object rotating | 9-10 of 25, varying as it moves through frustums |

The middle row is the case the whole mechanism exists for, and the last row is
the one per-tile invalidation adds over a single frame-wide hash.

## Blended geometry does not cast

Alpha-blended draw items are skipped in the atlas pass, matching the rule
[transparency.md](transparency.md) specifies for `BLEND`. They are skipped at
**replay** rather than filtered out of the caster list, the same way the main
HDR pass handles these buckets: the GPU cull still emits commands for those
slots and they are simply never drawn, which keeps every batch's compacted
command range valid. Filtering the lists instead would desync the batch ranges
from the cull input.

## Controls

Debug panel → tick **Advanced mode** → **Scene** tab → **Shadows** → *Punctual
(spot/point)*.
The controls are behind the advanced-mode gate, which is where the cascaded
shadow settings have always lived:

| Control | Effect |
| --- | --- |
| Cast punctual shadows | Master toggle; off stamps the unshadowed sentinel into every light |
| Slots used | Tiles allocated this frame, with atlas occupancy by area — with mixed tile sizes a slot count says nothing about how full the atlas is |
| Depth bias | Constant offset applied to the compared depth |
| Normal bias | World-space offset along the surface normal before projecting |
| Max shadowed point lights | Budget, in lights; each costs 6 tiles. Nearest to the camera are served first |
| Caster draws recorded | Draws the atlas pass issued last frame; zero with slots > 0 means it culled everything |
| Assignment churn | Lights that gained or lost a shadow since last frame; persistently non-zero is what popping looks like |
| Debug: shadow term only | Renders the punctual visibility term as greyscale instead of shaded colour |
| Atlas depth preview | Samples the atlas image directly (see the caveat below) |

**Read the caster-draw count, not the preview image.** The preview is a plain
`ImGui::Image` with a multiplicative tint, so it cannot expand contrast near
1.0 — and a perspective projection puts almost all of its depth range up there.
A correctly rendered tile therefore reads as solid far-plane colour, identical
to an empty one. The CSM cascade previews do not have this problem only because
an orthographic projection gives linear depth.

So the useful signal is `Caster draws recorded`: zero with slots > 0 means the
atlas pass culled or skipped everything; non-zero means the pass ran and any
remaining problem is in the lookup. Making the preview itself readable needs a
linearising remap, which the shared preview helper has no place to put.

That splits the diagnosis cleanly across the two halves:

| Symptom | Meaning |
| --- | --- |
| `Caster draws recorded` is 0 with slots > 0 | The atlas pass culled or skipped everything |
| Draws > 0 but **Debug: shadow term only** is flat white | The atlas is written but never sampled — the lookup is wrong |
| Draws > 0 and the debug view shows structure | Both halves work; the term is simply subtle in the shaded image |

The debug view exists because a single spot among dozens of clustered lights is
easy to lose in the beauty image — "I cannot see a shadow" is not evidence of a
bug on its own.

Read the debug view as a **min across the lights that actually reach each
fragment**, not as a preview of the shaded result. Several casters overlapping
darkens a great deal of the frame there while the shaded image changes far less,
because shading also weighs each light's attenuation. The gate matters:
`evaluatePunctualLight` applies the shadow only after its range and cone
early-outs, so the debug path has to reproduce them — without that it reports
occlusion from lights contributing no light at all, which min()s the frame to
near-black and reads as a catastrophic bug that is not there.

## Testing

`tests/test_punctual_shadow_atlas.cpp` covers the GPU-free core the same way
`ClusterGrid.h` and `CascadeMath.h` are covered:

- slots tile the atlas without overlapping, and every tile lands inside it;
- slot UV rects agree with their pixel rects;
- the allocator hands out each slot once and then degrades to the sentinel
  rather than failing;
- slot indices survive the float round-trip, including the NaN and
  out-of-range cases;
- a spot projection centers its cone axis in the tile, and depth increases with
  distance from the light;
- points outside the cone project off the tile, which is what makes the shader's
  bounds test equivalent to the cone cutoff;
- degenerate inputs (zero direction, a cone at/over 90 degrees, zero range)
  produce finite, non-degenerate matrices.

## Limitations

- The skinned caster here is **verified now**, on `--scene sunlit`. It is drawn
  into every tile its bounds reach (6 of 25 on the default scene, printed beside
  the caster draw count) and its pose digest enters those tiles' cache keys, so
  an animating rig dirties them. The default scene could not show it -- its
  punctual shadows are too dim and the mesh too close to the frame edge, so the
  frame moved by at most one quantization step. On the sunlit yard, where a spot
  sits almost level with the mesh and throws a magnified shadow onto a wall,
  suppressing only the punctual skinned draw moves **20866 pixels at a maximum
  channel delta of 85**. See [skeletal_animation.md](skeletal_animation.md).
- **A tile is all-or-nothing internally.** One moved caster inside a tile's
  frustum redraws that whole tile, including the casters in it that did not
  move. That is inherent to a shadow map: the depth buffer is not separable per
  caster.
- **Invalidation is conservative about frustum membership.** A caster whose
  bounds intersect a tile's frustum dirties it even if it is fully occluded by
  something nearer the light and could not change the result.
- **Tile assignment is recomputed from scratch every frame**, and priority
  depends on the camera, so moving the camera reshuffles which lights hold tiles
  and can flip a light between size classes at a threshold. Both pop visibly.
  Fixing it needs hysteresis on the size class and stable light identity across
  frames, neither of which exists yet.
- **Priority ignores occlusion and the view frustum.** A light directly behind
  the camera ranks by projected size like any other, so it can take a tile that
  a visible light then cannot have.
- **The point-light demotion is a fixed one class**, not derived from the actual
  per-face footprint.
- **GPU caster culling is off by default, and no longer for want of a
  measurement.** It is bit-identical to the CPU path on all six scene presets, it
  removes ~60% of the frame’s CPU recording on `--scene stress`, and a
  gate-passing A/B (0.30% control drift) puts the GPU side at +0.021 ms against
  that ~0.31 ms CPU saving. What still blocks the flip is the missing per-draw-item
  caster mask -- without it, enabling this reintroduces alpha-blended geometry
  casting opaque shadows -- plus `tests/golden/lavapipe_frame30.png` regenerated on
  lavapipe. See [GPU caster culling](#gpu-caster-culling).
- **No filtering beyond 3x3 PCF.** No variance/moment maps, no contact-hardening.
- **No cross-face filtering.** PCF is clamped inside a face rather than
  continuing onto the neighbour, so filtering narrows slightly at face
  boundaries instead of blending across them. Correct, but not seamless; proper
  cross-face taps need neighbour-face lookups.
- **The atlas preview cannot show perspective depth.** See [Controls](#controls);
  the caster-draw counter covers the diagnostic need in the meantime.
