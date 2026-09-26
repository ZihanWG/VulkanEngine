# Async Compute

The clustered-lighting compute passes (`ClusterBuild` + `LightCull`) run on a
dedicated async compute queue, so they can overlap the CSM shadow passes on the
graphics queue. They only depend on CPU-uploaded light data and camera
parameters, and their output has two readers: the main HDR fragment shader,
always, and the volumetric fog injection dispatch when fog is on. That makes
them the textbook async-compute pairing: raster/geometry-bound shadow work on
one queue, ALU-bound light culling on the other.

## Queue selection

`VulkanDevice::createLogicalDevice` picks, in order of preference:

1. a **dedicated compute-only family** (compute without graphics) — runs on the
   GPU's compute ring and overlaps best;
2. a **second queue in the graphics family** (when `queueCount >= 2`) — still
   lets the driver interleave, created at lower priority (0.5) so it never
   starves the frame-critical graphics queue;
3. **unavailable** — the renderer records the cluster passes inline on the
   graphics queue exactly as before.

On MoltenVK the default configuration exposes a single queue, so async compute
reports unavailable; launch with `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1` to
expose a compute-only family backed by its own `MTLCommandQueue`.

## Synchronization model

No graphics command buffer split and no render-graph surgery:

1. Right after CPU frame prep — before the graphics command buffer is even
   recorded — the renderer records ClusterBuild + LightCull into a per-frame
   async command buffer (`rhi::VulkanAsyncCompute`) and submits it to the
   compute queue, signaling a per-frame binary semaphore. The GPU starts light
   culling while the CPU is still recording graphics commands.
2. The frame's single graphics submission waits on that semaphore at **every
   stage that reads the cluster grid / light index buffers** (via buffer device
   address) this frame: `FRAGMENT_SHADER` always, plus `COMPUTE_SHADER` when
   fog is on, because fog injection walks the same per-cluster lists from a
   compute dispatch. Frame prep latches that set once
   (`frameClusterConsumerStages_`), and the same value is the light cull's
   barrier destination when the passes run on the graphics queue instead, so
   the two paths cannot disagree. Fog reads the lists only when the set carries
   the compute stage; otherwise it falls back to its brute-force light loop.
3. Command-buffer and semaphore reuse are guarded transitively by the frame's
   timeline value: the graphics submission waited on the async work, so its
   completion implies the async work finished.

The wait scope is a contract, not a promise of overlap. It used to be
`FRAGMENT_SHADER` alone, which left fog's compute reads unordered against the
cull that writes them -- on both queues, since the graphics-queue barrier named
the same single stage -- for as long as fog and the cluster passes both
existed. No gate here could report it: the reads go through buffer device
addresses, which synchronization validation does not track (it returned 0
hazards on the unfixed code), and lavapipe has no async queue.

Nor could it be made to show on an RTX 3080 Ti Laptop (driver 617.14). With the
async submission delayed by several milliseconds of extra dispatch and the
cluster grid zeroed at its start, no captured frame ever read the zeroed grid --
not even with the wait moved past the fragment stage entirely, where the main
pass itself should have raced. The probe could see a stale read: skipping the
cull changed every pixel. So this driver appears to hold the whole graphics
submission until the semaphore signals, and the overlap it gives is across
frames -- the next frame's cluster work beside this frame's graphics -- not
with this frame's shadow passes. The stage mask matters on drivers that honour
it, and is written for them.

Masked shadow casters and the masked depth prepass run `shadow_masked.frag`, so
under a driver that honours the mask they wait at `FRAGMENT_SHADER` too; the
picture of shadow work running unblocked holds only for opaque casters.

Cross-queue memory: when the async family differs from the graphics family,
every clustered-lighting buffer is created `VK_SHARING_MODE_CONCURRENT` across
both families, so no queue-family ownership transfers are needed; the semaphore
provides the cross-queue execution + memory dependency. The light cull's
trailing barrier to its consumers is skipped on the async queue (`FRAGMENT` is
not a valid stage on a compute-only queue); the intra-buffer build→cull
barrier stays, since both dispatches live in the same async command buffer.

## Controls and observability

- `enableAsyncCompute` (settings, default on) / "Async compute (cluster build +
  light cull)" checkbox in the `Lights (Clustered)` panel, with an
  active/inactive/unavailable status line.
- Debug label `AsyncClusteredLighting` wraps the async command buffer for
  RenderDoc/Instruments captures, which is where to see how much the compute
  queue actually overlaps on a given driver (see the wait-scope note above).
- Known limitation: the GPU profiler's timestamp queries live on the graphics
  command buffer, so the `ClusterBuild`/`LightCull` rows are not captured while
  async compute is active (noted in the panel).
- The cluster passes are not render-graph passes (they manage their own
  barriers), so the graph's pass list is unchanged; multi-queue scheduling
  inside the render graph itself remains future work.
