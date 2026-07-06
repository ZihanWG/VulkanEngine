# Async Compute

The clustered-lighting compute passes (`ClusterBuild` + `LightCull`) run on a
dedicated async compute queue, overlapping the CSM shadow passes on the
graphics queue. They only depend on CPU-uploaded light data and camera
parameters, and nothing reads their output until the main HDR fragment shader —
which makes them the textbook async-compute pairing: raster/geometry-bound
shadow work on one queue, ALU-bound light culling on the other.

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
2. The frame's single graphics submission waits on that semaphore at
   **`FRAGMENT_SHADER`** — the first stage that reads the cluster grid / light
   index buffers (via buffer device address). Shadow passes are depth-only
   (no fragment shader) and the culling dispatches are compute-stage work, so
   everything before the main HDR pass's fragment shading runs unblocked.
3. Command-buffer and semaphore reuse are guarded transitively by the frame
   fence: the graphics submission waited on the async work, so the fence
   implies it finished.

Cross-queue memory: when the async family differs from the graphics family,
every clustered-lighting buffer is created `VK_SHARING_MODE_CONCURRENT` across
both families, so no queue-family ownership transfers are needed; the semaphore
provides the cross-queue execution + memory dependency. The trailing
compute→fragment pipeline barriers are skipped on the async queue (`FRAGMENT`
is not a valid stage on a compute-only queue); the intra-buffer build→cull
barrier stays, since both dispatches live in the same async command buffer.

## Controls and observability

- `enableAsyncCompute` (settings, default on) / "Async compute (cluster build +
  light cull)" checkbox in the `Lights (Clustered)` panel, with an
  active/inactive/unavailable status line.
- Debug label `AsyncClusteredLighting` wraps the async command buffer for
  RenderDoc/Instruments captures — overlap is visible in a GPU trace as the
  compute queue running alongside the shadow passes.
- Known limitation: the GPU profiler's timestamp queries live on the graphics
  command buffer, so the `ClusterBuild`/`LightCull` rows are not captured while
  async compute is active (noted in the panel).
- The cluster passes are not render-graph passes (they manage their own
  barriers), so the graph's pass list is unchanged; multi-queue scheduling
  inside the render graph itself remains future work.
