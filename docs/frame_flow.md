# One-Frame Rendering Flow and Descriptor Contract

_Detailed per-frame pass ordering and the descriptor-set layout contract, moved out of the README for length._

## One-Frame Rendering Flow

1. Wait for the current frame fence, read previous completed exposure debug state and GPU timestamp results when available, update CPU-side debug history without a same-frame GPU/CPU stall, and acquire the next swapchain image.
2. Reset the fence and command buffer, update transforms, and build all `DrawItem` records from render objects and mesh primitives.
3. Extract the camera frustum from `projection * view`, compute CSM split depths, and build one texel-snapped directional light view-projection matrix per cascade.
4. Build CPU fallback shadow draw items/batches for each cascade, and build main-pass mesh-compatible draw batches for GPU culling or the CPU fallback.
5. Upload per-object MVP/model/light/material data into the current frame's Buffer Device Address object-data buffer.
6. Begin `RenderGraph` recording, import swapchain/depth/shadow resources, register transient scene/bloom targets, and declare pass read/write usage.
7. For each cascade, optionally reset shadow batch counts and shadow indirect commands, dispatch the GPU shadow cull with that cascade's light-frustum planes, and barrier its writes for indirect/count reads.
8. Let the graph transition the cascaded shadow-map array, begin depth-only Dynamic Rendering against the current layer view, and draw shadow casters for that cascade.
9. If any punctual light was assigned an atlas tile, run `PunctualShadowAtlasPass`: let the graph transition the punctual shadow atlas, open one depth-only Dynamic Rendering scope over the whole atlas, and draw each slot's casters under that slot's viewport/scissor.
10. Reset the main-pass batch visible-count buffer, dispatch the camera-frustum compute culling pass, optionally sample the previous completed Hi-Z depth pyramid for conservative occlusion, manually barrier visible counts for the immediate readback copy, and let the graph barrier culling outputs for later indirect/count reads in `MainHDRPass`.
11. Let the graph transition the HDR scene color image, cascaded shadow map, punctual shadow atlas, and main depth image for `MainHDRPass`.
12. Begin `MainHDRPass`, draw the skybox, bind global and bindless material descriptors when available, and issue indirect indexed mesh draws into `sceneColor_`.
13. Run `DepthPyramidPass` to sample the stored normal-Z main depth image and write the max-depth Hi-Z pyramid for later-frame culling.
14. If TAA is enabled, run `TAAResolvePass` to resolve jittered `sceneColor_` into the current HDR history target; otherwise keep `sceneColor_` as the active post-process source.
15. Run the legacy bloom extract/blur fallback into `BloomPong`.
16. Run the mip-chain bloom downsample passes at 1/2, 1/4, 1/8, and 1/16 resolution when practical, then progressively upsample into the final mip-chain bloom target.
17. Run `LuminancePass` to reduce log luminance from the active HDR scene source into per-frame GPU storage.
18. Run `HistogramExposurePass` to bin HDR scene luminance, reduce the selected exposure mode into the GPU exposure state buffer, manually preserve host readback visibility, and let the graph make the exposure buffer visible to `CompositePass`.
19. Run `CompositePass` to combine active HDR scene color + selected bloom * intensity, apply manual or GPU exposure, apply Reinhard or ACES tone mapping, and write the final color to the swapchain.
20. If a portfolio screenshot was requested, transition the composited swapchain image to transfer source, copy it into a per-frame readback buffer, then return it to color-attachment layout.
21. Run `ImGuiPass` to load the composited swapchain image as a color attachment and draw the debug UI overlay.
22. Let the graph transition the swapchain image to present, submit with `vkQueueSubmit2`, and present.
23. Recreate the swapchain, post-process images, TAA history, depth pyramid resources, and ImGui swapchain-dependent backend state if presentation reports an out-of-date or resized surface.

## Current Descriptor Contract

Bindless main-pass global resource descriptor set 0:

- binding 1 = cascaded shadow map combined image sampler, sampled in shaders as `sampler2DArray`
- binding 4 = diffuse irradiance cubemap combined image sampler
- binding 5 = prefiltered specular cubemap combined image sampler
- binding 6 = BRDF LUT combined image sampler
- binding 7 = punctual (spot/point) shadow atlas combined image sampler, sampled as `sampler2D`

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

ImGui uses its own descriptor pool and backend-owned descriptor layouts. It does not change the material, bindless texture, post-process, shadow, IBL, BRDF LUT, or ObjectFrameData descriptor contracts above.
