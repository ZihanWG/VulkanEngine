# GPU Culling

Phase 5 kept the renderer's existing draw-item and indirect-draw model and added an optional conservative Hi-Z occlusion test to the GPU culling shader. The two-phase upgrade later made occlusion safe to enable by default: phase 1 culls against the previous frame's pyramid without requiring a still camera, and a phase-2 re-test rescues disoccluded objects the same frame, so occlusion never drops a visible draw.

## Two-Phase Occlusion (default on)

1. **Cull phase 1** — frustum test + Hi-Z test against the *previous frame's* pyramid, projected with the view-projection stored at that pyramid's build (correct under camera motion). Visible items emit compacted indirect commands; in-frustum-but-occluded items are marked as candidates in a per-item phase-result buffer (`cull.comp` binding 5).
2. **Main HDR pass (phase 1)** draws as usual.
3. **Mid-frame pyramid rebuild** (`DepthPyramidMidPass`) downsamples the phase-1 depth.
4. **Cull phase 2** (`MainGpuCullingPhase2`) resets only the per-batch visible counts (the stats counters persist), re-tests the candidates against the mid-frame pyramid with the *current* view-projection, and emits commands for anything no longer occluded. A `rescued` stats counter records them.
5. **Main HDR phase 2** (`MainHDRPhase2`) replays the per-batch indirect-count draws with LOAD attachments (color, velocity, depth), so rescued objects composite into the existing frame.
6. The end-of-frame pyramid rebuild then includes the rescued draws for next frame's phase 1.

Correctness: phase-1 false negatives (disocclusions) are exactly the phase-2 candidates, and the phase-2 pyramid only contains real geometry from this frame, so the max-depth test stays conservative — an object behind a not-yet-drawn occluder simply draws (overdraw, never a hole).

Requirements: the bindless multi-draw-indirect path and a valid previous-frame pyramid. With the indirect-count path (`vkCmdDrawIndexedIndirectCount`), phase 2 resets the per-batch counts and re-compacts into the batch regions the phase-1 draws already consumed; without it (e.g. MoltenVK), phase 2 instead rewrites every fixed per-item command slot — rescued items get real commands, everything else is zeroed — and the replayed multi-draw skips the zeroes. When the requirements are missing, the frame falls back to the single-phase behavior below, which only trusts the pyramid while the camera holds still. `enableTwoPhaseOcclusion` (settings / GPU Culling panel) toggles the mode for A/B comparison.

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

## Occlusion Test Scene

`VulkanEngine Debug` -> `Scene` tab -> `Scene Presets` includes `Load Occlusion Test Scene`. The preset is procedural and uses the existing cube mesh and runtime materials, so it does not add large assets or replace the default glTF/fallback scene or portfolio showcase scene.

The layout is intentionally static:

- 1 ground plane for orientation
- 5 large opaque foreground occluder walls
- 120 smaller cube objects behind the occluders, arranged in rows and columns
- side and top "visible-edge" cubes left outside the main occluder coverage so the view is not fully hidden

The preset also applies an occlusion-test camera that looks through the wall group toward the dense cube grid. Static transforms are important because the implementation uses previous-frame depth; animated objects would repeatedly invalidate the pyramid and make rejection hard to measure.

To exercise the path:

1. Run the engine with the normal default scene first if you want a baseline with occlusion disabled.
2. Click `Load Occlusion Test Scene`.
3. Click `Enable Occlusion Test Settings` in the `GPU Culling` panel.
4. Let one or two frames pass so the previous-frame depth pyramid is valid.
5. Read the `Culling` panel counters and the `GPU Profiler` / `Render Graph` panels.

This scene is a validation and visual-debug preset, not a benchmark. It is designed to make non-zero occlusion rejection easy to inspect; it should not be used as a representative scene-performance number.

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

- `Load Occlusion Test Scene`
- `Reset Occlusion Test Camera`
- `Enable Occlusion Test Settings`
- `GPU occlusion culling enabled`
- depth bias
- near-object skip distance
- maximum screen coverage
- minimum projected size
- occlusion test scene active/inactive state
- total object and draw-item counts
- visible-after-culling, frustum-culled, and occlusion-culled draw counts
- occlusion rejection percentage
- depth pyramid valid/unavailable state
- previous-frame depth valid/invalid state
- depth pyramid mip count
- depth pyramid mip preview in the render-target debug panel

The GPU profiler adds a `DepthPyramid` timestamp scope. Main GPU culling timing remains under `MainGpuCullingPass`, and the render graph panel lists `DepthPyramidPass` with its main-depth read and pyramid write metadata when resources are available.

## Known Limitations

- Single-phase mode (two-phase disabled) only runs occlusion while the camera holds still.
- Phase 1 tests against the previous frame's depth. Phase 2 re-tests the rejects against a pyramid rebuilt from *this* frame's phase-1 depth, so the frame is not purely previous-frame -- but neither phase uses a dedicated depth prepass. With two-phase disabled, only the previous-frame test remains.
- Animated transforms invalidate the pyramid, so moving scenes may get few or no occlusion rejections.
- There is no HLOD, meshlet culling, mesh shader path, software occlusion rasterizer, ray tracing, or full GPU-built draw-list rewrite.
- The test is AABB based and intentionally biased toward false negatives.
- The default glTF/fallback scene may not show high occlusion rejection because it has too few draw items and little deliberate occluder coverage.
- Main depth must be stored after `MainHDRPass`, which costs bandwidth compared with the previous discardable depth attachment.
- Shadow caster culling does not use Hi-Z occlusion.
