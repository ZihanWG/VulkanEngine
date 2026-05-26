# GPU Profiling

Phase 1 adds a small Vulkan timestamp profiler around the existing renderer frame flow. The goal is to expose the GPU cost of major passes without changing rendering behavior, descriptor layout, swapchain ownership, or render graph order.

## Runtime UI

Open the profiler in the ImGui overlay:

1. Open `VulkanEngine Debug`.
2. Expand `Debug Views`.
3. Enable `Show GPU Profiler panel`.
4. Expand `GPU Profiler`.

The panel shows whether timestamp profiling is available, total GPU frame time, CPU frame delta, query usage, and a table of named pass timings. Each row has current, recent average, recent max, and a compact history plot. `Reset averages` clears the CPU-side history buffers.

If timestamps are unavailable, the panel remains visible and reports the unavailable reason instead of crashing or recording invalid queries.

## Implementation

`src/renderer/GpuProfiler.h` and `src/renderer/GpuProfiler.cpp` define the profiler. The renderer initializes it after Vulkan frame resources are created:

- one `VkQueryPool` per frame-in-flight slot
- 256 timestamp queries per frame by default
- two timestamp queries reserved for total frame time
- two timestamp queries per named scope
- timestamp conversion through `VkPhysicalDeviceProperties::limits.timestampPeriod`
- query support checked through the graphics queue family's `timestampValidBits`

The profiler uses `vkCmdWriteTimestamp2` because the renderer already uses Vulkan 1.3 and Synchronization2. Scope begin timestamps are written at `VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT`; scope end timestamps are written at `VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT`.

This makes each timing an inclusive elapsed GPU range around the commands recorded between the markers. Nested scopes, such as `RenderObjects` inside `MainHDRPass`, are expected to overlap with their parent and should not be summed with parent scopes.

## Frame Latency

The frame loop already waits the fence for `currentFrame_` before reusing that frame slot. The profiler reads timestamp results for that same completed slot immediately after the fence wait and before command-buffer reset:

1. Wait the current frame slot fence.
2. Read query results from that completed slot with `VK_QUERY_RESULT_WITH_AVAILABILITY_BIT`.
3. Skip the update if results are not ready.
4. Reset and record the command buffer for the next use of the slot.
5. Mark the frame slot submitted after `vkQueueSubmit2` succeeds.

The profiler does not add `vkDeviceWaitIdle` to the runtime frame loop and does not use `VK_QUERY_RESULT_WAIT_BIT`. Swapchain recreation does not recreate profiler query pools because they depend only on the device and frame-in-flight count.

## Profiled Ranges

The current frame records timestamp scopes for:

- `CSMShadowPass`
- `ShadowGpuCullingCascade0` through `ShadowGpuCullingCascadeN` when shadow GPU culling is active
- `MainGpuCullingPass` when main GPU culling is active
- `MainHDRPass`
- `Skybox`
- `RenderObjects`
- `BloomExtractPass`
- `BloomBlurHorizontal`
- `BloomBlurVertical`
- `LuminancePass` when log-average exposure is active
- `HistogramExposurePass` when histogram exposure is active
- `CompositePass`
- `ImGuiPass`

The same major ranges also use `VK_EXT_debug_utils` labels through the existing optional debug wrapper. If debug utils function pointers are unavailable, labels are no-ops.

## Limitations

- GPU timings are not CPU/GPU calibrated timestamps.
- Parent scopes include child scope work.
- Top-of-pipe and bottom-of-pipe markers are simple pass-range estimates, not detailed pipeline-stage attribution.
- The profiler has a fixed per-frame query capacity. The UI reports query usage and warns if the frame exceeds the configured capacity.
- Async compute scheduling, timeline lane visualization, and RenderDoc capture automation are future work.
