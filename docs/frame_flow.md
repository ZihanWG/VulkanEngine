# One-Frame Rendering Flow and Descriptor Contract

_Detailed per-frame pass ordering and the descriptor-set layout contract, moved out of the README for length._

## One-Frame Rendering Flow

1. Wait for the current frame fence, read previous completed exposure debug state and GPU timestamp results when available, update CPU-side debug history without a same-frame GPU/CPU stall, and acquire the next swapchain image.
2. Reset the fence and command buffer, update transforms, and build all `DrawItem` records from render objects and mesh primitives.
3. Extract the camera frustum from `projection * view`, compute CSM split depths, and build one texel-snapped directional light view-projection matrix per cascade.
4. Build CPU fallback shadow draw items/batches for each cascade, and build main-pass mesh-compatible draw batches for GPU culling or the CPU fallback.
5. Upload per-object MVP/model/light/material data into the current frame's Buffer Device Address object-data buffer.
6. Begin `RenderGraph` recording, import swapchain/depth/shadow resources, register transient scene/bloom targets, and declare pass read/write usage.
7. If virtual-shadow-map page marking is enabled, record `VsmPageMarkPass`: a
   compute dispatch reading the **previous** frame's Hi-Z depth pyramid, paired
   with the view-projection stored when that pyramid was built, which works out
   which clipmap pages this frame's visible surfaces would need and atomicOr's a
   page-request bitmask copied to a host-readable buffer. Recorded first because
   it reads what the previous frame left behind, and it changes nothing that is
   drawn -- see [virtual_shadow_maps.md](virtual_shadow_maps.md). Off by default.
   With page rendering also enabled, this is followed by `VsmPageCull` (one
   dispatch over every (dirty page, draw item) pair, compacting per-page indirect
   commands) and `VsmPagePass` (one rendering scope over the whole page pool, with
   a clear rect, a viewport, a scissor and one indirect draw per dirty page). The
   pool is still sampled by nothing.
8. Reset shadow batch counts and shadow indirect commands, dispatch **one** GPU shadow cull for all cascades (an object survives if any cascade's light frustum wants it; the planes come from the frame-params buffer, since four frusta do not fit in a push-constant block), and barrier its writes for indirect/count reads.
9. Let the graph transition the cascaded shadow-map array, then for each cascade begin depth-only Dynamic Rendering against that layer's view and replay the shared caster list. With `enableLayeredCascades` the loop collapses into one multiview pass over the whole array view, and the shader reads `gl_ViewIndex` instead of a pushed cascade index -- including the skinned caster, which takes `shadow_skinned_layered.vert` and indexes `cascadeViewProjection[]` the same way. Verified image-preserving on `--scene sunlit` (RTX 3080 Ti, `--deterministic --capture-frame 30`): 0 of 921600 pixels differ against the per-cascade path, with three byte-identical repeats per configuration. Off by default on measured cost, not on correctness -- it is ~20% slower on the shadow pass on both MoltenVK and native-multiview NVIDIA, because it cannot keep the per-cascade cache (see [render_scale.md](render_scale.md)).


   That equality is worth re-checking after any change to this path, because the way it broke before was not visible in timings. The skinned caster used to push a per-cascade matrix, which `gl_ViewIndex` cannot answer, so it was skipped here entirely -- and since its pose then had to stay out of the cascade cache key as well, nothing dirtied the cascades. `CSMShadowPass` read 0.000 ms with 3263 consecutive cached frames, which looks like multiview paying off and was a frozen shadow map missing the caster's whole ground shadow (11491 pixels, every one brighter).
10. If any punctual light was assigned an atlas tile, run `PunctualShadowAtlasPass`: let the graph transition the punctual shadow atlas, open one depth-only Dynamic Rendering scope over the whole atlas, and draw each slot's casters under that slot's viewport/scissor.
11. Reset the main-pass batch visible-count buffer, dispatch the camera-frustum compute culling pass, optionally sample the previous completed Hi-Z depth pyramid for conservative occlusion, manually barrier visible counts for the immediate readback copy, and let the graph barrier culling outputs for later indirect/count reads in `MainHDRPass`.
12. Let the graph transition the HDR scene color image, cascaded shadow map, punctual shadow atlas, and main depth image for `MainHDRPass`.
13. Begin `MainHDRPass`, draw the skybox, bind global and bindless material descriptors when available, and issue indirect indexed mesh draws into `sceneColor_`.
14. Run `DepthPyramidPass` to sample the stored normal-Z main depth image and write the max-depth Hi-Z pyramid for later-frame culling.
15. If TAA is enabled, run `TAAResolvePass` to resolve jittered `sceneColor_` into the current HDR history target; otherwise keep `sceneColor_` as the active post-process source.
16. Run the legacy bloom extract/blur fallback into `BloomPong`.
17. Run the mip-chain bloom downsample passes at 1/2, 1/4, 1/8, and 1/16 resolution when practical, then progressively upsample into the final mip-chain bloom target.
18. Run `LuminancePass` to reduce log luminance from the active HDR scene source into per-frame GPU storage.
19. Run `HistogramExposurePass` to bin HDR scene luminance, reduce the selected exposure mode into the GPU exposure state buffer, manually preserve host readback visibility, and let the graph make the exposure buffer visible to `CompositePass`.
20. Run `CompositePass` to combine active HDR scene color + selected bloom * intensity, apply manual or GPU exposure, apply Reinhard or ACES tone mapping, and write the final color to the swapchain. This is where a reduced render scale is upscaled: every step above runs at the internal render extent, and the composite is the first pass whose viewport is the swapchain (see [render_scale.md](render_scale.md)).
21. If a portfolio screenshot was requested, transition the composited swapchain image to transfer source, copy it into a per-frame readback buffer, then return it to color-attachment layout.
22. Run `ImGuiPass` to load the composited swapchain image as a color attachment and draw the debug UI overlay.
23. Let the graph transition the swapchain image to present, submit with `vkQueueSubmit2`, and present.
24. Recreate the swapchain, post-process images, TAA history, depth pyramid resources, and ImGui swapchain-dependent backend state if presentation reports an out-of-date or resized surface.

## Current Descriptor Contract

Bindless main-pass global resource descriptor set 0:

- binding 1 = cascaded shadow map combined image sampler, sampled in shaders as `sampler2DArray`
- binding 4 = diffuse irradiance cubemap combined image sampler
- binding 5 = prefiltered specular cubemap combined image sampler
- binding 6 = BRDF LUT combined image sampler
- binding 7 = punctual (spot/point) shadow atlas combined image sampler, sampled as `sampler2D`
- binding 13 = cascaded shadow map compare sampler, sampled as `sampler2DArrayShadow`
- binding 14 = virtual shadow map page pool, sampled as `sampler2DShadow` through
  the page table rather than with a direct UV; shares binding 13's immutable
  compare sampler, and is always bound (the cascade array's *layer 0* view stands
  in when the pool does not exist, because an array view under a non-array
  sampler is VUID-vkCmdDrawIndexed-viewType-07752)

Bindless material texture descriptor set 1:

- binding 0 = base color combined image sampler runtime array
- binding 1 = normal map combined image sampler runtime array
- binding 2 = metallic-roughness combined image sampler runtime array

Skybox descriptor set:

- set 0 binding 0 = visible environment cubemap combined image sampler

Post-process descriptor sets, separate from material/bindless descriptors:

- TAA resolve set binding 0 = current jittered HDR scene color combined image sampler
- TAA resolve set binding 1 = previous HDR history combined image sampler
- bloom extract/blur set binding 0 = one combined image sampler for the current post-process input
- bloom upsample set binding 0 = current bloom mip combined image sampler
- bloom upsample set binding 1 = lower accumulated bloom combined image sampler
- composite set binding 0 = active HDR scene color combined image sampler, either `SceneColorHDR` or resolved TAA history
- composite set binding 1 = legacy blurred bloom combined image sampler
- composite set binding 2 = mip-chain bloom combined image sampler
- composite set binding 3 = per-frame exposure state storage buffer
- luminance compute set binding 0 = HDR scene color combined image sampler
- luminance compute set binding 1 = per-frame luminance partial-sum storage buffer
- histogram compute set binding 0 = HDR scene color combined image sampler
- histogram compute set binding 1 = per-frame 256-bin histogram storage buffer
- exposure reduce set binding 0 = per-frame luminance partial-sum storage buffer
- exposure reduce set binding 1 = per-frame histogram storage buffer
- exposure reduce set binding 2 = per-frame exposure state storage buffer

Legacy fallback material descriptor set 0, used when descriptor indexing is unavailable:

- binding 0 = base color combined image sampler
- binding 1 = cascaded shadow map combined image sampler, sampled in shaders as `sampler2DArray`
- binding 2 = normal map combined image sampler
- binding 3 = metallic-roughness combined image sampler
- binding 4 = diffuse irradiance cubemap combined image sampler
- binding 5 = prefiltered specular cubemap combined image sampler
- binding 6 = BRDF LUT combined image sampler
- binding 7 = punctual (spot/point) shadow atlas combined image sampler, sampled as `sampler2D`

GPU culling compute descriptor set:

- binding 0 = per-frame culling input storage buffer
- binding 1 = per-frame indirect command output storage buffer
- binding 2 = per-frame batch visible draw count storage buffer

Shadow GPU culling compute descriptor set:

- binding 0 = per-frame shadow culling input storage buffer
- binding 1 = per-frame shadow compacted indirect command output storage buffer
- binding 2 = per-frame shadow batch visible draw count storage buffer

Object and material scalar data still use Buffer Device Address plus a vertex-stage push constant. On the bindless main multi-draw path and the shadow indirect path, the pushed address is the base of the current frame's `ObjectFrameData` array, and indirect `firstInstance` selects the object-data entry. The shadow pass also pushes the current cascade index. Fallback paths still push one per-draw object-data address with `firstInstance = 0`.

Virtual shadow map page-marking compute descriptor set (see
[virtual_shadow_maps.md](virtual_shadow_maps.md)):

- binding 0 = Hi-Z depth pyramid combined image sampler (mip 0 is read)
- binding 1 = page-request bitmask storage buffer
- binding 2 = per-frame marking parameter storage buffer

Virtual shadow map page-cull compute descriptor set:

- binding 0 = shared per-frame cull input storage buffer
- binding 1 = per-page compacted indirect command storage buffer
- binding 2 = per-page visible counts plus a trailing over-cap counter
- binding 3 = six frustum planes per dirty page
- binding 4 = per-draw-item caster flags

The virtual shadow page pass reuses `shadow_punctual.vert` and its push block:
one depth-only draw of a rect with that rect's own projection pushed is the same
operation for an atlas tile and for a clipmap page.

ImGui uses its own descriptor pool and backend-owned descriptor layouts. It does not change the material, bindless texture, post-process, shadow, IBL, BRDF LUT, or ObjectFrameData descriptor contracts above.
