---
name: ibl-precompute-cost
description: "Scene creation's fixed ~166 ms was CPU IBL precompute, not GPU — 59% the BRDF split-sum table (8.4M sin/cos/pow on one thread). Parallelising it took scene create to 60.38 ms (-64%). MERGED c012758, PR #12."
metadata:
  node_type: memory
  type: project
---

**Every scene, including an empty one, paid ~166 ms of CPU precompute at startup.**
Measured directly on the default procedural scene, where `glTF import` and
`texture upload` are both 0.00 ms, so `scene create` *is* this work. Sponza paid
the same on top of its own.

Profile attribution inside `createScene` (128 samples):

| stack | samples | share |
| --- | --- | --- |
| `VulkanBrdfLut::create` | 75 | **59%** |
| `createProceduralPrefilteredSpecular` | 19 | 15% |
| procedural cube/sphere | 2 | 2% |

The BRDF table is 256x256 texels x 128 integration samples = **8.4 million
sin/cos/pow steps**, single threaded.

## Result — MERGED (`c012758`, PR #12, all 4 CI checks green)

| | scene create | renderer init |
| --- | --- | --- |
| serial | 166.15 ms | ~194 ms |
| BRDF parallel | 74.37 ms | ~99 ms |
| BRDF + prefilter parallel | **60.38 ms** | **~86 ms** |

−64%; control re-run drifted **0.16%**.

## Facts worth not re-deriving

- **`docs/asset_load_baseline.md` used to claim this was "GPU compute at startup".
  That was FALSE** and is corrected in place — it never touches the GPU.
- **No ordering to preserve here**, unlike [[gltf-import-cost]]'s LOD chains: each
  texel writes only its own bytes, so `parallelFor`'s equal contiguous chunks are
  correct and the output cannot depend on scheduling.
- **The default scene IS pixel-reproducible** (0/3686400 differing), unlike Sponza
  which has that (816,850) delta-37 flake. So a zero-diff capture gate is
  meaningful on the default scene and useless on Sponza — pick the scene to match
  the gate.
- The split-sum table depends only on size and sample count, so it is identical on
  every machine. Precomputing it offline is worth only the remaining ~6 ms, and a
  GPU compute pass would still pay pipeline creation — neither is obviously worth
  it now.
- BRDF math lifted to `renderer/BrdfIntegration` in **Core** (testable); the
  prefiltered specular loop was parallelised **in place** because moving the
  cubemap machinery was more churn than its ~14 ms justified — so that half has
  no unit test, only the pixel gate.
