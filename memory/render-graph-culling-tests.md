---
name: render-graph-culling-tests
description: "Render graph CPU logic is now fully covered: pass culling (21b48c3), barrier derivation (efe3e81), barrier necessity (5e875b5). currentTextureLayout deliberately left uncovered. Also: mutation-test new tests."
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-06T14:48:10.647Z
---

**MERGED AND PUSHED** 2026-08-06 as `21b48c3`. 169 tests, up from 156.

## The seam

`RenderGraph::compilePassCulling()` was a private method; its body is now the
free function `cullUnusedPasses(std::vector<RenderPassNode>&, size_t textureCount,
size_t bufferCount)` declared in `RenderGraph.h`, with the method as a one-line
forwarder. It touches no Vulkan state, so this makes it reachable from a test
without a device — the same split `ClusterGrid.h`, `CascadeMath.h`, and
`VolumetricFog.h` already use.

`RenderPassNode`, `RenderResourceUsage`, and `RenderResourceHandle` are all
Vulkan-free (strings, enums, uint32_t), even though `RenderGraph.h` includes
`VulkanCommon.h` for the *other* structs. Tests may include the header freely;
`test_scene_builder` already established that tests link the Vulkan loader
without calling it.

## The culling semantics worth not re-deriving

Backward sweep, last pass first, over two liveness arrays:
- A pass is culled when it has writes and none of them are needed, unless it has
  a side effect. A pass with **no** writes and no side effect is also culled —
  reading without producing cannot affect the frame.
- **A culled pass does not propagate its reads.** This is what makes culling
  transitive back through a chain; without it a dead chain stays alive.
- On a surviving pass, a write **clears** liveness and a read **sets** it, in
  that order. So `ReadWrite` ends up setting — a read-modify-write keeps its
  producer alive. Reversing the two would make it cull its own producer.
- Out-of-range resource indices are ignored, not treated as live.

## Two process lessons

**Mutation-test new tests.** All 13 passed on the first run, which for subtle
logic is a reason for suspicion rather than confidence. Deliberately breaking the
implementation confirmed they bite: dropping the `continue` after culling failed
"Culling is transitive back through a chain", and swapping the write-clear /
read-set order failed "ReadWrite keeps the producer alive".

**Never `git checkout <file>` to revert a mutation when that file holds
uncommitted work** — it reverts to HEAD and takes the real change with it. Doing
this dropped the uncommitted `cullUnusedPasses` definition while the header still
declared it, producing a confusing link error. Copy the file aside and copy it
back instead.

## Barrier derivation — also done (merged `efe3e81`, pushed)

Same seam pattern. `accessStateForTexture`'s only member-state reads were the
resource's aspect and its tracked layout, so both became parameters:
`textureAccessState(VkImageAspectFlags, RGAccess, VkImageLayout currentLayout)`.
`accessStateForBuffer` was already pure → `bufferAccessState(RGAccess)`. The
`TextureAccessState`/`BufferAccessState` structs moved to namespace scope with
`using` aliases left inside `RenderGraph`, so old spellings still compile.
11 cases; 180 tests total.

Behaviours pinned that are easy to break:
- **Shader-read layout is aspect-dependent.** Sampled depth needs
  `DEPTH_READ_ONLY_OPTIMAL`, and stencil presence selects the combined variant.
  Hi-Z, SSR, and GTAO all sample depth, so this is load-bearing.
- A **buffer-shaped access on a texture keeps the tracked layout**, not
  `UNDEFINED` — the latter would discard image contents.
- An **image-shaped access on a buffer resets `declaredAccess` to Unknown**, so
  `transitionBuffer` skips it. Note the asymmetry: the texture mapping does
  *not* reset `declaredAccess` in its fallback. Current behaviour is pinned as
  found; not investigated as a bug.
- Depth attachment writes scope both EARLY and LATE fragment tests.

## Barrier *necessity* — also done (merged `5e875b5`)

`transitionBuffer`/`transitionTexture`'s emit-or-skip predicates became free
`bufferBarrierRequired()` / `textureBarrierRequired()`. 191 tests total.

Why it deserved coverage: **emitting an unneeded barrier only costs
performance; skipping a needed one is a race that often still renders correctly
on the machine it was written on.** Nothing else catches that.

Rules pinned:
- read-after-read emits nothing; a write on either side requires ordering
- **"has been touched" is `usedThisFrame || previousDeclared != Unknown`, an OR
  not an AND** — a resource carried across frames has `usedThisFrame` false
  while still holding a real last access
- a layout change always emits, even read-to-read and even untouched
- same layout falls through to the write rule (what separates two storage-image
  passes both sitting in GENERAL)
- `UNDEFINED` old layout orders against nothing — contents are not preserved

Mutations that must fail: AND-ing the touched halves, and dropping
`previousAccess` from the write check.

**`currentTextureLayout` is deliberately NOT covered.** It reads `frame_`'s
swapchain / shadow map / punctual atlas pointers, so only the
`ExternalLayoutPointer` branch is reachable without a device. Judged not worth
mocking; that decision is made, don't re-open it without a new reason.
