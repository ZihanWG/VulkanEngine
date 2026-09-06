---
name: texture-upload-batching
description: "Texture upload batching MERGED (02c7f49, PR #11): 50 -> 14 ms (-71%), 69 submits -> 2. Step 1 of the async transfer queue; step 2 (the queue) is NOT started and will be inert by default because MoltenVK hides the TRANSFER-only family."
metadata: 
  node_type: memory
  type: project
  originSessionId: 82b33743-1ea4-4a88-8918-779088f596e4
  modified: 2026-08-17T16:14:04.485Z
---

Step one of the async transfer queue (P2 of [[asset-pipeline-initiative]]).
**MERGED as `02c7f49`** (PR #11, `--no-ff`, all 4 CI checks green). Branch deleted.

## Why batching, not the queue, came first

Profile of the ~51 ms texture phase: **~50% `vkQueueWaitIdle`** (each texture
submitted its own command buffer and drained the whole queue, 69 times), ~36%
reading cooked KTX2 files serially on the device thread, ~14% the rest.

**A transfer queue would not have fixed that** — 69 serialised waits are 69
serialised waits on any queue. Don't re-propose the queue as a fix for the waits.

## Measured (Release, M3, warm; first run of a series is always cold, ~100-150 ms)

| | serial | batched |
| --- | --- | --- |
| texture upload | 50.11 / 52.07 ms | **14.42 / 15.37 ms (−71%)** |
| submits for 69 textures | 69 | **2** |
| device bytes / device-local | 91.06 / 621.68 MiB | unchanged |

Post-merge on main: upload 13.5-14.2 ms, renderer init ~515-567 ms, glTF import
~298 ms.

## `rhi::VulkanUploadBatch` — the two load-bearing properties

- **The batch retains the staging buffers**, not the caller's scope: a queued copy
  still reads them after the recording function returns. Freeing early corrupts
  texels rather than crashing. Destructor submits, so an exception mid-load
  cannot drop them.
- **Bounded at 64 MiB** (`kUploadBatchStagingBudgetBytes`). Peak staging stops
  scaling with the scene — Sponza is 2 submits at 62 MiB peak, not 91 MiB. The
  cooked-file prefetch window uses the same budget.
- Optional: both create paths take a batch or `nullptr`. Only the glTF loop
  batches; `BuiltinTextureFactory`, material assets, BRDF LUT, cubemaps untouched.

## MoltenVK queue families — the fact that shapes step 2

| | default | `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1` |
| --- | --- | --- |
| families | 4, **all** GRAPHICS\|COMPUTE\|TRANSFER | 0: G+C+T, 1: G+T, 2: C+T, **3: T only** |

So a dedicated transfer queue **does not exist by default on this Mac**, same gate
async compute already documents. Step 2 will be inert unless that env var is set,
and CI will not cover it (as with async compute).

## Verification lesson worth keeping

**Run validation layers against the path that actually changed.** The default
procedural scene does not use the batch, so a clean run there proves nothing. It
needed the Sponza build temporarily reconfigured to Debug (validation is compiled
out in Release). Result: 0 errors / 0 warnings, 69 textures in 2 submits.

## CI polling in this repo lies twice over

`gh pr checks --watch` exited **0** on an `HTTP 503` from GitHub's GraphQL API
while a job was still pending, and a naive `until` loop exited on a bad `jq`
expression. Both looked like success. Poll by requiring **all four rows present
AND zero "pending"**, and re-check independently before merging.

## What is left

Step 2: TRANSFER-only family selected like `VulkanDevice::asyncComputeQueue`,
queue family ownership transfer (release on transfer, acquire on graphics), own
command pool, graphics-queue fallback with a log line. Not started.

**CI polling in this repo has now lied three times.** (1) `gh run --watch` exited
0 on a 503. (2) A `gh pr checks` "fail" that was an infrastructure cancel, not a
test failure. (3) A watch loop built on `gh run list --limit 3` announced
ALL-COMPLETE while quoting a SHA that was not even in the last 12 runs on the
branch — the newest runs had not appeared yet and it read whatever came back.
**Filter runs by `git rev-parse HEAD`, never by recency, and read the step list
before believing a conclusion.**
