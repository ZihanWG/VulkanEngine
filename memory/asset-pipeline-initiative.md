---
name: asset-pipeline-initiative
description: "Asset cook + async upload initiative — P0 baseline shipped and the CONTENT BLOCKER IS REMOVED (0c726ab, opt-in Sponza fetch). With Sponza: 366 MiB uncompressed textures, 3944 ms glTF import, 141 ms serial upload. P1 has real targets."
metadata: 
  node_type: memory
  type: project
  originSessionId: 901cbd1c-8080-4c6f-9d21-4f11902d4869
  modified: 2026-08-17T11:39:33.709Z
---

Asset cook (KTX2/BC7 + offline mips) and transfer-queue async upload. Six-phase
plan written 2026-08-15, published at
https://claude.ai/code/artifact/781ff702-bd65-4e0a-aa20-a3be01b5f080

**P0 SHIPPED** (`af8ff6d`, branch `measure/asset-load-baseline`, not yet merged):
`renderer/AssetLoadStats` + `--asset-load-stats` / `--exit-after-frames N` flags.
Numbers in `docs/asset_load_baseline.md`.

**BLOCKER REMOVED 2026-08-16** (`0c726ab`): `-DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON`
downloads Sponza at configure time (pinned commit, into the build tree, file list
read out of the glTF with `string(JSON)`, partial download refused). OFF by
default so CI and the golden are untouched. `cmake/FetchSampleScene.cmake`.

**P1 IS COMPLETE end to end** — see [[bc7-texture-cook]]. The Sponza texture
numbers below are now the *uncooked control*: with the cook applied they are
**91.06 MiB and 72 of 77 block compressed (4.02x)**, decode wait is 0.00 ms, and
upload is 48.70 ms. **P2 (async transfer queue) is unblocked**: the cooked upload
path dropped `vkCmdBlitImage`, so texture upload no longer needs a graphics
queue. The remaining big cost is glTF import, which the cook did not touch.

**With Sponza the costs are real and P1 now has a target:**
renderer init 191 ms → **7106 ms**; glTF import 0.00 → **3944 ms**;
texture upload 0.00 → **141 ms**; textures 8 → **77**;
texture bytes 0.45 MiB → **365.98 MiB**, still **0 of 77 block-compressed**;
device-local 494 → 878 MiB. So: BC7/BC5 is worth ~4x on 366 MiB, mesh cook kills
most of 3.9 s, and the transfer queue has a real 141 ms of serial upload to attack.
Numbers in `docs/asset_load_baseline.md`.

Caveats to carry forward: Sponza is ONE node / one mesh / 103 primitives, so it
presents as a **single render object** — it does not exercise object-level frustum
culling. The startup camera is framed from scene bounds (`renderer::framedCamera`,
8 unit tests) which guarantees visibility but not composition; placing it inside
from bounds alone was tried and abandoned (Sponza's bounds include its outer
walls). Compose screenshots with the editor camera.

**The earlier conclusion, now historical:** The baseline killed the premise:
the whole `assets/` dir is 68 KiB, all 8 textures total 0.45 MiB, texture upload
is below timer resolution, and `Renderer::tryLoadGltfScene()` **has no caller** —
the default portfolio scene is entirely procedural. The 168 ms inside scene
create is procedural IBL generation (no `studio.hdr` present), not asset loading.
The 493 MiB device-local is render targets and atlases.

**Why:** every pathology the plan targets is real in the code (`stbi_load` RGBA8,
runtime `vkCmdBlitImage` mips, per-texture `vkQueueWaitIdle` at
`VulkanTexture.cpp`), but none is *measurable* here. Optimizing 0.45 MiB and
0.00 ms would be unfalsifiable work.

**How to apply:** do not start P1 until a production-scale glTF (Sponza/Bistro
class, real multi-MB textures) is in a scene the renderer actually loads, then
re-run the one-command baseline. If the user would rather not add large content
to a portfolio repo, say so plainly and re-rank against the other gap-analysis
candidates instead — lavapipe headless CI + golden images, or the render-graph
transient allocator with aliasing. See [[back-to-back-or-dont-claim]] for the
measurement protocol this baseline follows.
