---
name: async-transfer-queue-upload
description: "Async transfer queue for texture upload MERGED (b376a50, PR #13) and honestly measured as NO FASTER — 21.99 vs 21.13 ms inside a 42.6% spread. It is a capability, inert by default; CI never exercises it."
metadata:
  node_type: memory
  type: project
---

Step 2 of the upload work, after [[texture-upload-batching]]. **MERGED `b376a50`.**

## The result is "no measurable difference", and that is written down as such

Interleaved A/B (to cancel drift), seven warm runs each on Sponza:

| | median | own spread |
| --- | --- | --- |
| transfer queue off | 21.99 ms | **42.6%** |
| transfer queue on | 21.13 ms | 13.5% |

**Do not quote the 0.86 ms** — it is inside the control's own noise. The docs and
the PR say this explicitly. A real win needs upload overlapped with mesh building,
which was deliberately not done.

## Facts worth not re-deriving

- **Verified, not assumed:** a TRANSFER-only queue does not list
  `SHADER_READ_ONLY_OPTIMAL` among its supported layouts, but validation layers
  **accept** it in the release barrier of an ownership transfer on MoltenVK. The
  three-barrier contingency (`TRANSFER_DST → TRANSFER_DST` then a graphics-side
  transition) was not needed.
- **Release and acquire must name identical layouts and family indices** — built
  by one helper for that reason. The semaphore between the two submits is not
  optional.
- **A texture that generates its own mips cannot use the transfer queue at all**,
  because `vkCmdBlitImage` needs graphics. It falls back to its own
  submit-and-wait. Same constraint that made the cook worth doing first.
- Images stay `EXCLUSIVE` (real ownership transfer), unlike the clustered-lighting
  buffers' `CONCURRENT` shortcut — concurrent sharing can cost image compression.
- Selection lives in `rhi/TransferQueueSelection.h` as a pure function with unit
  tests, because on this machine the answer is **none** and the fallback is the
  path that actually runs. No "second queue in the graphics family" fallback:
  that is still the graphics ring.
- **Inert by default; CI never runs it.** Needs
  `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1`, same gate as async compute.

## CI lesson: a red check is not necessarily a test failure

`gh pr checks` reported the lavapipe gate as **fail** on PR #12. The job was
actually **cancelled** after 45m16s stuck in "Install the Vulkan SDK"; every
render and compare step was **skipped**, so the gate never tested the change. A
rerun passed in 20m36s (normal is ~4m37s, so that install step is flaky/slow).

**Read the step list before concluding a gate rejected your change.** Duration is
the tell: a genuine golden-image mismatch fails fast with a pixel count.
