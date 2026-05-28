# GPU Culling

Phase 5 keeps the renderer's existing draw-item and indirect-draw model, then adds an optional conservative Hi-Z occlusion test to the GPU culling shader. The goal is to reject only objects that are clearly hidden while preserving the CPU fallback and the existing frustum culling behavior when occlusion is disabled.

## Existing Frustum Culling

`Renderer::updateFrameData()` still builds CPU-side `DrawItem` records from active `RenderObject` instances and mesh primitives. Each draw item carries world-space AABB bounds, material/mesh references, and the object-data index used by the main render pass.

When GPU culling is enabled, `Renderer::updateGpuCullInputBuffer()` uploads those draw bounds into the per-frame cull input buffer. `src/shaders/cull.comp` tests each AABB against the current camera frustum planes, writes visible indirect draw commands, and increments per-batch visible counts. The main pass then reads the indirect command buffer and visible-count buffer as before. If GPU culling is unavailable, the renderer falls back to CPU frustum culling and CPU-written indirect commands.

Shadow culling continues to use the same shared compute shader with occlusion disabled. It tests per-cascade light frustum planes and does not sample the depth pyramid.

## Depth Pyramid Generation

The main depth image is now created with sampled usage only when the selected depth format advertises `VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT`. `VulkanSwapchain::findDepthFormat()` first chooses a format that supports both depth attachment and sampled-image usage. If no sampleable candidate exists, it falls back to an attachment-only depth format, logs a warning, and GPU occlusion remains disabled.

`MainHDRPass` stores the main depth attachment because `DepthPyramidPass` samples that exact image after the pass completes. This changes the previous `STORE_OP_DONT_CARE` behavior to `STORE` for the main depth attachment, which has a bandwidth cost on tile and immediate renderers. The implementation avoids an extra depth copy: the only additional work is storing the depth attachment and sampling it directly into the pyramid.

`DepthPyramidPass` writes `DepthPyramidHiZ`, an `VK_FORMAT_R32_SFLOAT` image with a full mip chain down to 1x1. Mip 0 is generated from main depth; each lower mip samples the previous mip. The compute shader clamps source texel coordinates, so non-power-of-two extents are handled without out-of-bounds fetches. The resource is recreated with the post-process/swapchain resize path and is imported into Render Graph metadata.

## Depth Convention

The renderer uses normal-Z depth:

- clear depth is `1.0`
- nearer depth values are smaller
- depth compare is `LESS`

The Hi-Z pyramid stores the maximum depth in each source footprint. For normal-Z, that is conservative for occlusion because the stored value represents the farthest depth among the sampled occluder texels. An object is culled only when its nearest projected depth is greater than the sampled max depth plus bias.

## Occlusion Test

The main GPU culling shader keeps frustum culling active and optionally adds occlusion:

1. Reject by frustum first.
2. Project the draw item's world-space AABB corners with the previous valid depth-pyramid view-projection matrix.
3. Build a screen-space rectangle from the projected corners.
4. Choose a mip from the projected rectangle size.
5. Sample the pyramid at the rectangle corners and center.
6. Cull only when `hizMaxDepth + depthBias < objectNearestDepth`.

The cull shader still writes the same indirect draw command and visible-count outputs. Occlusion only adds another rejection path and a separate counter.

## Conservative Fallbacks

The shader and renderer keep objects visible when the test is uncertain. Occlusion is skipped for:

- invalid, NaN, infinite, negative, or tiny bounds
- objects too near the camera
- objects intersecting or crossing the near plane
- objects partly outside the screen
- objects covering more than the configured screen-coverage threshold
- very small projected rectangles
- stale or invalid depth-pyramid state
- unavailable sampled depth or depth-pyramid resources

Previous-frame depth is used. The renderer invalidates the pyramid after swapchain resize/resource recreation, scene loads, portfolio camera/preset changes, camera resets, UI transform/visibility edits, and animated transform updates. Camera matrix and camera-position thresholds also disable occlusion for the current cull dispatch if the previous pyramid was built from a different view. These fallbacks intentionally allow false negatives. False positives are avoided to prevent visible popping.

## Debug Counters and UI

The GPU culling visible-count buffer now reserves four counter slots after the per-batch counts:

- total draw items
- visible draw items
- frustum-culled draw items
- occlusion-culled draw items

The renderer copies this buffer to the existing per-frame readback buffer and reads it only after the frame slot fence has completed. No current-frame stall is introduced.

ImGui exposes:

- `GPU occlusion culling enabled`
- depth bias
- near-object skip distance
- maximum screen coverage
- minimum projected size
- total, visible, frustum-culled, and occlusion-culled draw counts
- depth pyramid valid/unavailable state
- depth pyramid mip count
- depth pyramid mip preview in the render-target debug panel

The GPU profiler adds a `DepthPyramid` timestamp scope. Main GPU culling timing remains under `MainGpuCullingPass`, and the render graph panel lists `DepthPyramidPass` with its main-depth read and pyramid write metadata when resources are available.

## Known Limitations

- Occlusion is disabled by default.
- Occlusion uses previous-frame depth, not a current-frame prepass.
- Animated transforms invalidate the pyramid, so moving scenes may get few or no occlusion rejections.
- There is no HLOD, meshlet culling, mesh shader path, software occlusion rasterizer, ray tracing, or full GPU-built draw-list rewrite.
- The test is AABB based and intentionally biased toward false negatives.
- Main depth must be stored after `MainHDRPass`, which costs bandwidth compared with the previous discardable depth attachment.
- Shadow caster culling does not use Hi-Z occlusion.
