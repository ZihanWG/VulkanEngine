---
name: gltf-import-cost
description: "glTF import cost is LOD simplification, NOT parsing — 87% meshopt_simplify vs 3 ms of JSON. Parallelising it MERGED to main (5b21036, PR #10, CI green): 3.30x, 1014.61 -> 307.39 ms. A mesh cook that skips parsing would have been worthless."
metadata: 
  node_type: memory
  type: project
  originSessionId: 82b33743-1ea4-4a88-8918-779088f596e4
  modified: 2026-08-17T14:49:39.205Z
---

**The premise a mesh cook rests on is wrong here, and it was measured.** A
`sample -wait` profile of Sponza import (942 main-thread samples inside
`Mesh::createFromGltf`):

| stack | samples | share |
| --- | --- | --- |
| `buildLodChain` → `meshopt_simplify` | 823 | **87.3%** |
| `memmove` (vertex/index assembly) | 38 | 4.0% |
| `createDeviceLocal` (buffer upload) | 8 | 0.8% |
| **tinygltf JSON parse** | **3** | **0.3%** |

Parsing Sponza's glTF costs **~3 ms**. A cook that bakes vertices/indices to skip
parsing would save that. Don't propose it on parsing grounds.

The cost follows from a deliberate quality decision in `MeshLod.h`: every level
is simplified from the *authored* geometry, not from the previous level, because
chaining compounds error. So 103 primitives × 3 levels = 309 full-geometry
simplifications.

## What shipped — MERGED (`5b21036`, PR #10, `--no-ff`, all 4 CI checks green)

`buildLodChain` split into `buildLodChainDetached` (pure, worker-safe) +
`appendLodChain` (serial, decides layout). Old entry point kept as a wrapper so
existing tests and the procedural call sites are untouched.

**glTF import 1014.61 → 307.39 ms (3.30x); renderer init ~1270 → ~534 ms.**
Main-thread `meshopt_simplify` frames 823 → 2, spread 42-49 across seven workers.

Facts worth not re-deriving:
- **Enqueue per primitive, NOT `parallelFor`.** `parallelFor` splits into equal
  contiguous chunks (`JobSystem.cpp:63`), and primitives differ by orders of
  magnitude in triangle count, so a static split strands the heavy chunk.
- **`Logger` has NO mutex** — it chains `<<` on `std::cout`/`std::cerr`. A worker
  must never log; compose the message and print it from the serial phase. This
  applies to any future worker-thread code here.
- Ordering the serial appends by primitive is what makes the index buffer
  byte-identical to the serial path. That is the whole safety argument.

## The pixel gate does NOT work on this scene

Sponza at frame 30 has **pre-existing capture nondeterminism: exactly one pixel
at (816, 850), channel delta 37**, reproducible between runs of the *same*
binary (3 captures gave 0, 1, 1 differing pixels). This is distinct from the
[[lavapipe-headless-ci]] golden's off-by-1 noise. Use the **LOD chain log diff**
(84 lines, order and per-level triangle counts) as the geometry-equivalence check
instead.

## What is left

Import is now ~307 ms, of which ~50 ms is assembly and upload rather than
simplification. A real mesh cook (baking chains offline) is the only way to
remove the rest — re-measure before deciding it is worth a container. See
[[asset-pipeline-initiative]] and [[bc7-texture-cook]] for the P1 precedent.
