---
name: mesh-cook
description: "Offline mesh cook MERGED (3a088dd, PR #14): glTF import 331.49 -> 15.50 ms (21x), renderer init ~466 -> ~127 ms. The largest startup win of the series. Review found three real gaps in the staleness guarantee — read them before touching MeshCache."
metadata:
  node_type: memory
  type: project
---

**MERGED `3a088dd`** (PR #14, `--no-ff`, CI green on the fix commit). Four commits:
`c35b480` lift geometry loading to Core, `f0c6e08` container + header, `3481630`
tool + wiring + measurement, `fa874ea` the review fixes.

## Why it was worth doing, and how that was established

Parallelising LOD (see [[gltf-import-cost]]) made import 3.3x faster but only
spread the work. **The decisive probe was forcing `kMaxMeshLods = 1`**: import
349.44 ms → 53.46 ms, so simplification was still **85%** of it. That probe is the
technique to reuse — it measures a floor directly instead of arguing from a sample
tree.

| | uncooked | cooked |
| --- | --- | --- |
| glTF import | 331.49 ms | **15.50 ms (21x)** |
| renderer init | ~466 ms | **~127 ms** |

Cooking Sponza is ~410 ms, producing a 16.54 MiB `.vemesh` (bigger than the source
`.bin` — every LOD level is stored). Gitignored, like the `.ktx2` sidecars.

## Three real gaps review caught — the staleness guarantee was NOT complete

All three were genuine; none was a false positive.

1. **The external `.bin` was outside the fingerprint.** An ASCII glTF holds no
   vertex data — Sponza's is in `Sponza.bin`. Replacing it left the cook "fresh"
   and the runtime would upload old geometry with new scene metadata. The
   expectation now hashes every referenced external buffer's size+mtime; an
   unstat-able buffer hashes to a sentinel so it reads as *changed*, not *absent*.
2. **Skipped meshes crashed the whole import** — a regression from the refactor.
   `createDeviceLocal` throws on an empty span (`VulkanBuffer.cpp:86`), and
   `loadGltfGeometry` leaves unsupported meshes as empty slots so node references
   keep indexing by position. Empty slots must be skipped.
3. **Primitive ranges and index values reached indirect draws unvalidated.** These
   do not stay on the CPU, and a fallback cannot undo an out-of-bounds fetch.

**The lesson: "the header guards staleness" was true for what I thought the inputs
were, and the inputs were wrong.** Enumerate every file the output actually
depends on.

## Facts worth not re-deriving

- The glTF is still parsed on both paths (~2 ms). Materials/textures/nodes are
  deliberately NOT cooked — a large serialization surface and a second staleness
  surface for nothing.
- `loadGltfGeometry` takes an optional cooked-geometry input rather than having a
  second entry point, so node traversal has one code path.
- It checks cooked mesh count against the glTF's: node references index meshes by
  position, so a file cooked from another scene is the one failure that still
  renders.
- **`vemeshcook --verify` was necessary because the LOD chain log comparison
  cannot work here** — once cooked, chains are never rebuilt, so there is nothing
  to compare. It caught a real bug on first run: the output stream was still
  buffered when verify read the file back.
- The index-validation scan costs ~1.5 ms (26x → 21x). The docs were corrected
  from the unvalidated figure rather than left quoting the faster number.
