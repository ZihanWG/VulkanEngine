# One-Frame Rendering Flow and Descriptor Contract

_Detailed per-frame pass ordering and the descriptor-set layout contract, moved out of the README for length._

## One-Frame Rendering Flow

The order below is the order the render graph schedules and verifies each frame
(see [render_graph.md](render_graph.md)); a step whose feature is off is not
declared, so it is not recorded either.

1. Wait on the device timeline semaphore for this frame slot's previous submission, read back the GPU timestamps, exposure state, culling counters and VSM page requests that submission produced, update CPU-side debug history without a same-frame GPU/CPU stall, advance the skinned pose and VSM page residency, and acquire the next swapchain image (waiting on the timeline again if that image still belongs to another slot).
2. Reset the command buffer, build the debug UI, update the object transform cache, and build all `DrawItem` records from render objects and mesh primitives.
3. Extract the camera frustum from `projection * view`, compute CSM split depths, and build one texel-snapped directional light view-projection matrix per cascade; decide per cascade whether its content hash moved, and per punctual tile whether its casters or light did.
4. Build CPU fallback shadow draw items/batches, and build main-pass mesh-compatible draw batches for GPU culling or the CPU fallback.
5. Upload the frame-wide `FrameConstants` (view-projections, cascade matrices, lighting, shadow settings) and the per-draw-item `ObjectFrameData` array (model matrices, material data) into the current frame's Buffer Device Address buffers, and the light list for clustered shading.
6. If async compute is active, record the cluster build and light cull into the frame's compute command buffer and submit it now, before graphics recording starts (see [async_compute.md](async_compute.md)).
7. Begin `RenderGraph` recording, import swapchain/depth/shadow resources, register transient scene/bloom targets, and declare pass read/write usage.
8. If virtual-shadow-map page marking is enabled, record `VsmPageMarkPass`: a
   compute dispatch reading the **previous** frame's Hi-Z depth pyramid, paired
   with the view-projection stored when that pyramid was built, which works out
   which clipmap pages this frame's visible surfaces would need and atomicOr's a
   page-request bitmask copied to a host-readable buffer. Recorded first because
   it reads what the previous frame left behind -- this frame's depth does not
   exist yet -- and it changes nothing that is drawn; see
   [virtual_shadow_maps.md](virtual_shadow_maps.md). Off by default.
   With page rendering also enabled, this is followed by `VsmPageCull` (one
   dispatch over every (dirty page, draw item) pair, compacting per-page indirect
   commands) and `VsmPagePass` (one rendering scope over the whole page pool, with
   a clear rect, a viewport, a scissor and one indirect draw per dirty page). The
   main pass samples the pool instead of the cascades only when
   `vsm.enableShadows` is on as well.
9. If any cascade needs redrawing, dispatch **one** GPU shadow cull for all cascades as `ShadowGpuCullingPass` (an object survives if any cascade's light frustum wants it; the planes come from the frame-params buffer, since four frusta do not fit in a push-constant block). The graph orders its indirect commands and counts for `CSMShadowPass`.
10. Let the graph transition the cascaded shadow-map array, then for each cascade whose hash moved begin depth-only Dynamic Rendering against that layer's view and replay the shared caster list; a cached cascade is skipped outright. With `enableLayeredCascades` the loop collapses into one multiview pass over the whole array view, and the shader reads `gl_ViewIndex` instead of a pushed cascade index -- including the skinned caster, which takes `shadow_skinned_layered.vert` and indexes `cascadeViewProjection[]` the same way. Verified image-preserving on `--scene sunlit` (RTX 3080 Ti, `--deterministic --capture-frame 30`): 0 of 921600 pixels differ against the per-cascade path, with three byte-identical repeats per configuration. Off by default on measured cost, not on correctness -- it is ~20% slower on the shadow pass on both MoltenVK and native-multiview NVIDIA, because it cannot keep the per-cascade cache (see [render_scale.md](render_scale.md)).


   That equality is worth re-checking after any change to this path, because the way it broke before was not visible in timings. The skinned caster used to push a per-cascade matrix, which `gl_ViewIndex` cannot answer, so it was skipped here entirely -- and since its pose then had to stay out of the cascade cache key as well, nothing dirtied the cascades. `CSMShadowPass` read 0.000 ms with 3263 consecutive cached frames, which looks like multiview paying off and was a frozen shadow map missing the caster's whole ground shadow (11491 pixels, every one brighter).
11. If any punctual light was assigned an atlas tile, run `PunctualShadowAtlasPass` (preceded by `PunctualShadowCullPass` when GPU caster culling is on): let the graph transition the punctual shadow atlas, open one depth-only Dynamic Rendering scope over the whole atlas, and draw each dirty slot's casters under that slot's viewport/scissor.
12. Reset the main-pass batch visible-count buffer, dispatch `MainGpuCullingPass` -- frustum culling, per-draw-item LOD selection, and phase-1 occlusion against the previous frame's Hi-Z pyramid unless the occlusion-yield controller has suspended it -- manually barrier visible counts for the immediate readback copy, and let the graph order the culling outputs for indirect/count reads in `MainHDRPass`. On the CPU culling fallback the pass is not declared and the host uploads the same commands.
13. If async compute is not active, record the cluster build and light cull inline on the graphics queue.
14. If volumetric fog is on, run `VolumetricFogPass`: inject, light and temporally filter the froxel volume, then integrate it front to back. Injection walks the per-cluster light lists, which is why it follows the cluster passes.
15. If irradiance-probe GI is on and probes are due this frame, run `ProbeCapture` and the probe convolution.
16. With `renderer.enableDepthPrepass` (on by default, and wherever the multi-draw indirect path exists), replay the opaque and masked buckets depth-only into the main depth image as `DepthPrepass`, so the pass below loads that depth instead of clearing it and its `LESS_OR_EQUAL` test lets early-Z reject fragments it would otherwise shade and overwrite. Masked geometry runs the same cutout test the shadow path uses; blended geometry is never prepassed. See `design_decisions.md`.
17. Let the graph transition the HDR scene color image, the thin G-buffer, the velocity buffer, the shadow images and main depth for `MainHDRPass`; begin it, draw the skybox, bind global and bindless material descriptors when available, and issue indirect indexed mesh draws into `sceneColor_`.
18. With two-phase occlusion active, rebuild the pyramid from this frame's depth (`DepthPyramidMidPass`), re-test phase 1's rejects (`MainGpuCullingPhase2`), and draw the rescued items in a load-op pass (`MainHDRPhase2`).
19. If SSR is active, copy scene colour at half resolution and trace (`SSRCopyPass`, `SSRTracePass`); if GTAO is on, trace and upsample it (`GTAOPass`, `GTAOBlurPass`). Then draw the blended bucket, sorted back to front, in `TransparentPass`.
20. Run `DepthPyramidPass` to write the max-depth Hi-Z pyramid for the next frame's culling and page marking, unless nothing will read it.
21. If TAA is enabled, run `TAAResolvePass`, which reconstructs at presentation resolution from the jittered render-resolution samples and so also performs the upscale; otherwise keep `sceneColor_` as the active post-process source.
22. Run bloom: the mip-chain downsample and upsample passes by default, or the legacy extract and separable blur when the mip chain is off.
23. Run the exposure reduction for the selected mode: `HistogramExposurePass` bins HDR luminance and reduces it into the GPU exposure state buffer (the default), or `LuminancePass` reduces log-average luminance; the reduce stage manually preserves host readback visibility, and the graph makes the exposure buffer visible to `CompositePass`.
24. Run `CompositePass` to combine the active HDR source + selected bloom * intensity, apply manual or GPU exposure, apply Reinhard or ACES tone mapping and, when the frame is upscaled, contrast-adaptive sharpening, and write the final color to the swapchain. With TAA off this is where a reduced render scale is upscaled (see [render_scale.md](render_scale.md)).
25. If a portfolio screenshot was requested, transition the composited swapchain image to transfer source, copy it into a per-frame readback buffer, then return it to color-attachment layout.
26. Run `ImGuiPass` to load the composited swapchain image as a color attachment and draw the debug UI overlay.
27. Let the graph transition the swapchain image to present, submit with `vkQueueSubmit2` -- waiting on the async compute semaphore at every stage that reads the cluster lists, and signalling the frame timeline -- and present.
28. Recreate the swapchain, post-process images, TAA history, depth pyramid resources, and ImGui swapchain-dependent backend state if presentation reports an out-of-date or resized surface.

## Current Descriptor Contract

Main-pass descriptor set 0. One layout (`Renderer::createMaterialDescriptorSetLayout`)
serves the bindless path, which allocates a single shared set, and the legacy
fallback, which allocates one per material; each shader declares the bindings it
reads:

- binding 0 = base color combined image sampler (legacy `simple.frag` only)
- binding 1 = cascaded shadow map combined image sampler, sampled in shaders as `sampler2DArray`
- binding 2 = normal map combined image sampler (legacy only)
- binding 3 = metallic-roughness combined image sampler (legacy only)
- binding 4 = diffuse irradiance cubemap combined image sampler
- binding 5 = prefiltered specular cubemap combined image sampler
- binding 6 = BRDF LUT combined image sampler
- binding 7 = punctual (spot/point) shadow atlas combined image sampler, sampled as `sampler2D` -- stored depth, read by probe capture
- binding 8 = integrated volumetric fog volume, sampled as `sampler3D`; bound whether or not fog runs
- binding 9 = irradiance-probe irradiance atlas combined image sampler
- binding 10 = irradiance-probe depth-moment atlas combined image sampler
- binding 11 = irradiance-probe shading parameters uniform buffer
- binding 12 = ambient occlusion from the previous frame, sampled at the reprojected position for ambient-only GTAO
- binding 13 = cascaded shadow map compare sampler, sampled as `sampler2DArrayShadow`
- binding 14 = virtual shadow map page pool, sampled as `sampler2DShadow` through
  the page table rather than with a direct UV; shares binding 13's immutable
  compare sampler, and is always bound (the cascade array's *layer 0* view stands
  in when the pool does not exist, because an array view under a non-array
  sampler is VUID-vkCmdDrawIndexed-viewType-07752)
- binding 15 = punctual shadow atlas again, through the same immutable compare
  sampler, sampled as `sampler2DShadow` for the hardware-filtered lookup

Bindings 13-15 carry the immutable compare sampler in the layout itself rather
than in the descriptor write: MoltenVK allows a mutable comparison sampler only
behind a portability feature, and the cascades' sampler must outlive a shadow map
that is recreated when the cascade count changes.

Bindless material texture descriptor set 1:

- binding 0 = base color combined image sampler runtime array
- binding 1 = normal map combined image sampler runtime array
- binding 2 = metallic-roughness combined image sampler runtime array

Skybox descriptor set:

- set 0 binding 0 = visible environment cubemap combined image sampler

Post-process descriptor sets, separate from material/bindless descriptors:

- TAA resolve set binding 0 = current jittered HDR scene color combined image sampler
- TAA resolve set binding 1 = previous HDR history combined image sampler
- TAA resolve set binding 2 = velocity buffer combined image sampler
- TAA resolve set binding 3 = main depth combined image sampler, for closest-depth velocity dilation
- bloom extract/blur set binding 0 = one combined image sampler for the current post-process input
- bloom upsample set binding 0 = current bloom mip combined image sampler
- bloom upsample set binding 1 = lower accumulated bloom combined image sampler
- composite set binding 0 = active HDR scene color combined image sampler, either `SceneColorHDR` or resolved TAA history
- composite set binding 1 = legacy blurred bloom combined image sampler
- composite set binding 2 = mip-chain bloom combined image sampler
- composite set binding 3 = per-frame exposure state storage buffer
- composite set binding 4 = GTAO visibility combined image sampler, multiplied in only on the non-default whole-scene A/B path (`ambientOnly` off)
- luminance compute set binding 0 = HDR scene color combined image sampler
- luminance compute set binding 1 = per-frame luminance partial-sum storage buffer
- histogram compute set binding 0 = HDR scene color combined image sampler
- histogram compute set binding 1 = per-frame 256-bin histogram storage buffer
- exposure reduce set binding 0 = per-frame luminance partial-sum storage buffer
- exposure reduce set binding 1 = per-frame histogram storage buffer
- exposure reduce set binding 2 = per-frame exposure state storage buffer

GPU culling compute descriptor set:

- binding 0 = per-frame culling input storage buffer
- binding 1 = per-frame indirect command output storage buffer
- binding 2 = per-frame batch visible draw count storage buffer
- binding 3 = Hi-Z depth pyramid combined image sampler, for occlusion
- binding 4 = per-frame cull parameters (viewport, occlusion and LOD settings, and the cascade planes for the shadow dispatch)
- binding 5 = per-draw-item phase result, which phase 2 re-tests
- binding 6 = flat per-mesh LOD table the dispatch selects levels from

The shadow cull reuses `cull.comp` and this layout, with its own per-frame input,
compacted command and visible-count buffers at bindings 0-2; it culls against the
union of every cascade's frustum from the planes in binding 4.

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
