# Milestone History

_Moved out of the top-level README to keep it scannable. These notes preserve the incremental build history and design decisions behind the current renderer._

_Timings in Milestones 85 and later were taken on an RTX 3080 Ti Laptop; in 84 and earlier, on an Apple M3 through MoltenVK, unless an entry says otherwise. Each figure's full conditions -- scene, resolution, clock pin, statistic -- are in the subsystem document it came from, and `docs/profiling.md` records how the figures whose machine was not written down at the time were attributed._

## Milestone 93: Audit Fixes -- Queue Ordering, Settings Round Trip, glTF Conformance

An audit of the engine against its own documents found three defects, fixed here, and a documentation sweep that followed them.

Fog injection reads the per-cluster light lists from a compute dispatch, but the light cull was ordered before its readers at the fragment stage only -- both the async-compute semaphore wait and, when the cluster passes run on the graphics queue, the barrier after the cull. The frame now latches every stage that reads the lists (fragment always, compute when fog is on) and uses that one value for both, and fog reads the lists only when it carries the compute stage. No gate here could have seen it: the reads go through buffer device addresses, which synchronization validation does not track, and lavapipe has no async queue. A probe that delayed the async work and zeroed the grid could not make it show on the RTX either, which appears to hold the whole graphics submission until the semaphore signals; `async_compute.md` records that and what it means for overlap.

Save Settings wrote the defaults of `framesInFlight` and `enableShaderHotReload` over the file on every save, because capture never copied them. `tools/check_settings_capture.py` now fails the Linux job when any `RuntimeSettings` field is missing from capture or from apply. The cascades' receiver depth bias became a clamped, persisted setting, and four settings that were saved but had no control got one. The skinned glTF import now decodes normalized integer accessors -- quantized weights had been read as 0-255 -- honours STEP and CUBICSPLINE sampling, and skips and reports channels it cannot honour instead of misreading them.

## Milestone 92: Punctual Caster Lists, Skinned Motion Vectors, and Explicit LOD

The punctual atlas culled every slot against every draw item twice a frame -- once to build each tile's cache key and again, on one thread, to decide what to draw -- and the two had to be kept identical by hand. The key build now keeps each slot's caster list and the atlas draws from it. On `--scene stress` the atlas unit's recording fell from 0.34-0.43 ms to about 0.02 ms, and the hash and the draws can no longer disagree.

Skinned motion vectors projected this frame's skinned position through the previous MVP, so a bending arm under a still camera reported zero velocity. Each per-frame palette buffer now holds the previous frame's palette as well, inside the buffer its own frame slot guards; a velocity probe went from zero on all 38733 mesh-body pixels to non-zero on 38701.

An Intel UHD 770 lost the device on its first frame that sampled the punctual atlas. The cause was implicit-LOD `texture()` in non-uniform control flow, where the derivatives it needs are undefined; NVIDIA tolerated it and Intel faulted. Every fetch in divergent code now takes an explicit LOD. All 38 CI configurations run on the Intel part, and 75 RTX captures are bit-identical before and after.

## Milestone 91: Correctness Found by Widening the Gates

`--scene sponza` under synchronization validation reported the texture upload's queue-family ownership transfer as unordered. Both barrier halves were right; the semaphore between them signalled at `ALL_TRANSFER`, and the release barrier's layout transition -- whose second scope is legitimately `NONE` -- sat in no narrower stage, so nothing ordered it. The signal is `ALL_COMMANDS` now. CI cannot see this path (lavapipe has no transfer-only family and Sponza is not swept), so `tools/dev/verify_renderer.sh sync` runs it on a machine that can, and was checked against the bug: exit 2 with it, 0 without.

Nothing had ever counted which settings the configuration sweep exercised. Counted against the schema, it reached 19 of the 49 settings that select a code path, and one of the other 30 was live: with `renderer.useGpuCulling = false`, `MainGpuCullingPass` was declared every frame and never recorded. Two legs set a key to its default, so the TAA resolve had never run in CI at all. The main cull is now declared from the predicate its recorder returns on, `tools/check_ci_settings_coverage.py` counts instead of trusting, and the sweep grew to cover every path-selecting setting.

## Milestone 90: What the Main Pass Is Made Of on Real Content

The only decomposition of `MainHDRPass` came from the M3 against the default scene, so it was ablated again on Sponza: 55% of the pass is shadow filtering -- 38.6% punctual, 16.7% cascades -- and the punctual atlas costs about six times more to sample (2.163 ms) than to fill (0.380 ms). Caster-side work was aimed at the small half.

The filter half was then taken. The atlas was filtered with nine manual depth fetches; it is now four hardware comparison fetches through a second descriptor on the same image, reusing the cascades' immutable compare sampler. `MainHDRPass` fell 21.9% on Sponza. That reverses the M3's conclusion that the taps were not the cost -- the second such reversal after back-face culling, for the same reason: the tiler hid a real cost. The tile clamp had to inset a whole texel, because a linear compare tap gathers its own 2x2 and would otherwise reach the neighbouring cube face.

A two-point resolution fit had called about a third of the pass fixed cost. A third scale point falsified it -- cost per pixel rises as pixels get fewer, which is quad overshading -- and it was retracted. What survives is a bound: at a sixteenth of the pixels and the lowest LOD the pass costs 15% of full resolution, which caps everything that is neither per-pixel nor per-triangle. The lever left on this scene is triangle count.

## Milestone 89: Synchronization Validation and Graph-Owned Barriers

The render graph infers every barrier in the frame and nothing had checked the result: unit tests cover the derivation functions, the golden compares pixels, and core validation cannot see a missing dependency. `--sync-validation` enables synchronization validation through `VK_EXT_layer_settings`, and a self-test records two unordered writes and exits 5 if the layer stays silent. Its first run found thirty hazards in four classes on the default scene, every one a barrier naming a narrower stage than the command it ordered -- `COPY` does not cover a blit, and fills and updates run in the clear stage. The graph now maps transfer access to `ALL_TRANSFER`. Every sweep leg runs under it, and no pixel changed, which is why the class had survived.

A barrier belongs to the graph when it sits between two passes and to the subsystem when it sits between two commands inside one. Three subsystems wrote boundary barriers by hand; the froxel volumes and both shadow culls' indirect outputs are graph resources now. Per-subresource tracking was surveyed across every swept configuration and not built: the only multi-subresource images are always declared whole.

## Milestone 88: Sponza as a Measurement Scene, and the Depth Prepass

Sponza had been a compile-time startup hijack; it is `--scene sponza` now, one render object per glTF primitive. That split found two live bugs: every primitive had measured LOD against the bounds of the whole building (`L0=103` before, `L0=46 L1=9 L2=8 L3=4` after), and the transform cache kept handing culling those same bounds, so nothing was ever rejected. `--overdraw` counts fragment invocations per pixel and falsified the scene inventory: Sponza shades 2.187 per pixel, while the scene built with twenty-four overdraw layers shades 1.276, because early-Z already rejects stacked full-screen quads.

That was the number a depth prepass had been waiting for. Built over the opaque and masked buckets, it takes 13.0% off the Sponza frame, at a cost of 0.071 ms, and 4.2% off `--scene stress`; it is on by default. It is not pixel-neutral -- the main pass tests `LESS_OR_EQUAL` -- but only scenes with authored contact surfaces move, and the golden's scene is not one of them. Review then tied the compare op to whether a prepass can actually run, and the measured-and-rejected decisions were collected into a page for readers.

## Milestone 87: CPU Frame Attribution, Frame Prep, and Guards That Can Fire

The CPU frame gained seventeen named scopes. They name `updatePunctualShadowCacheState` as 30% of frame preparation, and on `--scene stress` the CPU, not the GPU, is the ceiling. The same work found that scopes nested inside a render pass read near zero on the M3's tiler but 96-97% of the parent on the RTX, so the profiler rule became per report rather than per project, and every quoted timing now carries its machine, scene, resolution and statistic. Frame preparation then fell 19% by caching each object's model matrix once per frame and building the punctual cache keys in parallel, with a chunk size derived from total work because one slot per chunk won on one scene and lost on another. Shadow-slot churn was damped with hysteresis.

Two proposals were measured and rejected. Specialization constants for the uber-shader have a negative ceiling: removing the default-off code makes `MainHDRPass` 4.4% slower. Meshlet culling finds 0.055% of the triangles LOD leaves behind, because LOD already removes 87% of them.

Four guards were made to work: the settings-schema test could not fire (the example file was missing seventeen keys and the test passed with a whole section deleted), every shader's descriptor and push-constant interface is pinned against a golden, the Windows leg builds with warnings as errors, and anisotropic filtering -- off since Milestone 9 behind a stale comment -- is on for material samplers.

## Milestone 86: The Configuration Sweep, a Froxel Producer Fix, and Back-Face Culling

The render graph's backstop reported only through the ImGui panel and only for the default configuration. It now logs a line, `--settings` lets a run name its settings file, and CI renders a list of configurations past it. The first sweep found two passes declared every frame and never recorded -- the depth pyramid with occlusion culling off, and the transparent pass with bindless off.

`cluster_build.comp` still built a 16x9 grid long after every consumer read 32x18x24, so 10368 of 13824 cluster bounds were never written. Clustered shading now matches the brute-force reference exactly, where it had differed on 15.8% of the default scene's pixels. It was found by making the constant checker read declarations rather than comments: the comment-based version had found two of five mirror groups and reported success.

Back-face culling, measured and rejected on the M3, was re-measured on the RTX: `MainHDRPass` -37.4% and frame total -14.9% on `--scene stress`. It is on by default.

## Milestone 85: Measurement on a Pinned Clock, Tooling Hygiene, and Two VSM Fixes

Frame pacing moved from a fence per frame slot to one device timeline semaphore, optional device extensions got a selection step and a startup capability report, and shader hot reload and a frames-in-flight setting landed. GPU clocks can be pinned for measurement and `measure_gpu.py` quotes p10, because on this hardware the median got the sign of a delta wrong where p10 did not. The PNG writer gained a deflate encoder (12.78x smaller, verified lossless), all first-party translation units compile clean at `/W4`, and the formatting config is enforced in CI.

The virtual shadow map's long-open lit-face discrepancy was not a bias or a filter: the page pass drew every caster from the first draw item's mesh buffers, so spheres were drawn as their bounding boxes. Each page's casters now draw from their own mesh, and the false shadow fell from 48482 pixels to 879 -- the floor two correct shadow paths already differ by. Separately, the clipmap's pages-per-level axis went from 16 to 32, because at 16 a 4K frame was capped at 0.55 shadow texels per pixel and its quality setting did nothing.

## Milestone 84: Pipelines Looked Up by State

Graphics and compute pipelines are looked up through a hashable `PipelineKey` instead of held in named members, so identical state compiles once: five named shadow-caster pipelines turned out to be two objects, and format-change detection stopped being hand-maintained. Membership is decided by lifetime rather than pipeline type -- only what the pipeline rebuild recreates may live in the store -- and references are generation-guarded, because that invariant broke within one commit of being written. The change is required to be pixel-identical and is, across sixteen configurations.

## Milestone 83: Skinned Shadow Casters, and a Scene Built to Show Shadows

The skinned mesh was lit by the sun while throwing no shadow at all. It now casts into the cascades, the punctual atlas and the virtual shadow map's pages. Every shadow cache and cull here keys on a transform, and a skinned mesh deforms while its transform holds still, so it carries a pose digest and a conservative per-joint world bound taken from the palette it uploads. `--capture-include-ui` lets a scripted run capture the debug overlay, which the first check of a UI-only warning needed.

Every preset was ambient-dominated, with directional shadows at about 2.5:1 against the lit floor. `--scene sunlit` measures 5.92:1, and it immediately verified the skinned caster's punctual shadow, which the default scene could not show. A depth readback of the page pool then moved the VSM lit-face question from the sampler to what gets drawn into a page, which is where Milestone 85 found it.

## Milestone 82: The Render Graph Drives the Frame

The graph described the frame and nothing checked the description. It now has a declaration validator, versioned handles, a dependency graph derived from the declarations, and an end-of-frame backstop comparing the recorded order against it -- and then the whole frame was handed to the checked declarations: `recordRenderCommands` went from 979 lines to 162. The checks found six defects with no symptom, among them a stale handle that shortened the derived dependency chain from 17 passes to 9, passes declared and never recorded, fog declared ahead of the cluster build, and a bloom chain computed and discarded every frame. No reordering is applied; the scheduled order is verified equal to the recorded one every frame.

## Milestone 81: Virtual Shadow Maps

A clipmap of 128-texel pages for the directional light on an absolute grid, so a page's world rect never moves with the camera and a warm clipmap on a static scene draws no pages at all. It was built in four measured stages -- marking, residency, page rendering, sampling -- each off by default, and the staging surfaced five design errors before anything depended on them. Moving casters dirty the pages under their old and new bounds; cutout casters draw through the alpha-tested pipeline; the page pool got a depth bias in its own units after reusing the cascades' constant, which is scaled to a cascade's depth box rather than a page's, lifted every umbra by about 15% of the sun. Page LOD was deliberately not built: the steady state draws nothing, so there is no time for it to save.

## Milestone 80: Scriptable Scene Presets and Frame Capacity Counting

`--scene` selects any preset from the command line, and CPU frame preparation is printed beside the GPU frame total. A persistent GPU scene table was measured and rejected as a performance change: frame preparation was 9-14% of the GPU frame on `--scene stress`. Geometry past the frame's object and draw-item caps is now counted and reported rather than silently dropped.

## Milestone 79: Per-Cascade Shadow Caching

The cascades were the one shadow path that redrew unconditionally. Each cascade now hashes what it draws -- its fitted matrix, the raster depth bias, and every caster's geometry, selected LOD level, transform and alpha cutoff -- and is skipped outright when the hash still matches. A bounding-sphere fit meant to make the cache hit under camera motion was built and measured not to: it survives 0.0025-0.041 degrees of yaw, against 0.17 per frame for a camera turning at ten degrees a second. It ships off, for the rotation shimmer it does remove. Two older bugs came out of it: the cascades' normal bias and blend band were saved and then never applied, and both shadow caches keyed casters by pointers a scene switch could reuse.

## Milestone 78: The Asset Pipeline

An optional configure-time fetch of Sponza gave the asset work something to measure: 366 MiB of uncompressed textures, 0 of 77 block-compressed, and four seconds of glTF import. An offline BC7/BC5 KTX2 cook took textures to 91.06 MiB (4.02x) and removed runtime decoding and mip generation. Import turned out to be 87% mesh simplification, so the LOD build went parallel (3.30x) and then offline in a mesh cook (glTF import 331.49 ms to 15.50 ms, 21x), with a header that refuses a stale cook. Texture upload was batched (69 submits to 2, -71%, staging bounded at 64 MiB), a dedicated transfer queue was added and documented as a capability rather than a speedup, and the IBL precompute -- found to be CPU work, not GPU as the docs had said -- went parallel (-64%). `docs/asset_load_baseline.md` carries each measurement.

## Milestone 77: Transient Memory, Measured, Built, and Left Off

The graph now knows what its transient resources cost and where they could share memory. The measurement found the cheapest win first: the SSR scene-colour copy was full resolution for a single point sample, and halving it saves 21.13 MiB with no aliasing at all. Aliasing itself was then built and verified -- a probe writes through one alias and reads through the other -- and measured: 17.48 MiB saved for 1.2% of frame time, on a device with memory to spare. It ships off. The golden gate was found to be flaky at zero tolerance and now allows a delta of 1.

## Milestone 76: Headless Rendering in CI

CI runs the renderer instead of only compiling it: on Mesa's lavapipe under Xvfb it renders 30 deterministic frames, fails on any validation error, and pixel-compares the captured frame against a committed golden. That needed a frame clock with a fixed-timestep mode (`--deterministic`), a capture path that never touches the portfolio screenshots, an image-comparison tool, and command-line parsing moved out of the SDL-dependent code. Each gate was shown able to fail -- an injected validation error, a zeroed readback window, a perturbed golden -- before it was trusted to pass.

## Milestone 75: A Scripted Measurement Protocol

`tools/dev/measure_gpu.py` enforces the rules a performance claim here has to meet: a fixed scene and camera, a warm-up, and A/B/A/B with the control repeated and a drift gate. Each reported pass carries its own control drift and an attributable verdict, a stale Release binary is a hard abort, and a pass that runs in only one configuration is judged against its own spread rather than skipped. Every one of those came out of an A/B that the missing rule had answered wrongly.

## Milestone 74: Tested Seams and Constant Parity

Arithmetic whose failure mode is silence moved to where it can be tested. Draw-item batching -- every indirect command offset the cull shader, the indirect draws and the readback share -- became a Vulkan-free unit with tests checked against deliberate mutations, one of which exposed a dead loop. The punctual shadow assignment policy was lifted out of the Vulkan work the same way. `tools/check_shader_constants.py` compares the constants mirrored by hand between C++ and GLSL, since a mismatch neither fails the build nor trips validation. The build went to zero warnings, the Windows CI job started running the tests, and the README and docs were corrected where the day's work had made them false.

## Milestone 73: Shadow, TAA and SSR Quality

TAA history is resampled with Catmull-Rom instead of one bilinear tap, which had been re-blurring the history every frame. The cascades are filtered through a hardware comparison sampler, which needed a second, immutable sampler in the descriptor layout because MoltenVK only permits a mutable one behind a portability feature. Normal-offset bias and cascade blending were added to the CSM path, both defaulting to zero. SSR stopped double-counting specular energy by emitting a signed correction toward the traced colour rather than adding on top of the IBL, and its march went from fixed view-space steps to uniform screen-space steps, which closed the holes near the camera. Back-face culling was measured and rejected on the M3's tiler, which already discards those fragments; Milestone 86 reversed that on an immediate-mode GPU.

## Milestone 72: Occlusion Yield and a Single Exposure Scan

The depth pyramid now builds only when something will read it, and Hi-Z occlusion suspends itself while it culls nothing, probing every 180 frames to notice when it starts paying again -- 9.2% off the default scene with identical draw counts. Histogram exposure no longer runs the log-average pass; it derives the geometric mean from the bins it already walks. Culling lights at an "effective" radius was measured and rejected: the image darkens faster than the pass shortens, and even a 50% radius only takes 13% off the pass.

## Milestone 71: Shadow Cascade Cost

Shadow-map resolution turned out not to be a lever -- 2048, 1024 and 512 all cost the same -- while cascade count was, because the cost is per-pass encoder work. One GPU caster cull now serves every cascade, halving `CSMShadowPass`. A multiview pass rendering all cascades at once is implemented and ships off: pixel-identical and about 20% slower under MoltenVK. A single-cascade configuration also produced a wrongly-typed image view and 22 validation errors per run; the view type is now explicit.

## Milestone 70: Sub-Rect Rendering and Temporal Upsampling

Every screen-space target is allocated at the maximum render resolution and written only in a sub-rect, with each consumer scaling its UVs into the written region. A render-scale change used to cost a ~15 ms rebuild hitch, most of it an unavoidable idle wait; it now costs 0.008 ms, because nothing is reallocated. The TAA resolve then became the upsampler: it runs at presentation resolution and reconstructs each output pixel from the jittered low-resolution samples nearest to it. Its two stricter ghosting guards ship off, since they were built for a report that was retracted. The debug panel was reorganised into task tabs under a status strip.

## Milestone 69: Render Scale, Dynamic Resolution and Sharpening

The scene shades at a fraction of the window and the composite upscales, trading fragment cost for sharpness. A GPU-free, tested controller drives the scale from measured GPU frame time; its first version ping-ponged between two steps forever, and a raise is now vetoed when the predicted cost would be over budget. A contrast-adaptive sharpen runs only when the frame is upscaled. The frame-cost series was later re-taken in Release on the RTX at 2560x1440, clocks pinned, 63 samples a point: 6.733 ms at full scale, 2.101 ms at half, 1.076 ms at a quarter. A floor-contact z-fight that TAA jitter had turned into flicker was fixed on the way.

## Milestone 68: Stress Scenes and a Finer Cluster Grid

Eleven draw items could not show occlusion culling, LOD or the parallel frame-prep loops, so a 2311-object geometry stress scene was added (occlusion culling goes from 0 to 670 rejections) and the draw-item cap rose from 1024 to 8192. A fragment stress scene with 192 densely packed lights loads the opposite axis. On it the cluster grid went from 16x9x24 to 32x18x24, -19% on `MainHDRPass`, which was also a correctness fix: 160-pixel tiles were hitting the per-cluster light cap and silently dropping lights. The grid constants moved into a shared shader header.

## Milestone 67: Leaner Per-Object Data

The punctual shadow lookup read the 96-byte slot record twice per light per fragment, the first time only to reach a normal bias that is one global value; it now rides in the light record, -7% on `MainHDRPass`. The per-draw-item `ObjectFrameData` went from 688 bytes to 192 over three rounds by moving everything identical across the frame -- view-projections and cascade matrices -- into a `FrameConstants` record written once, after its six verbatim shader copies were deduplicated into one header. Two toggles whose initializers and persistence disagreed with their settings were brought into line.

## Milestone 66: Ambient-Only Ambient Occlusion

GTAO had been multiplied into the whole composited image, darkening direct light too. The main pass now samples the previous frame's AO, reprojected along the motion vector TAA already computes, and applies it to the ambient term alone; the composite multiply stays as an A/B reference. It exposed a latent hazard: the AO target was the first swapchain-sized image in the material descriptor set, and nothing had ever rewritten those sets after a resize.

## Milestone 65: Render Graph Test Coverage

The render graph's two pieces of non-trivial CPU logic gained unit tests. Both were reachable only from private methods, so each had its body lifted into a pure free function with the method left as a one-line forwarder — the split the GPU-free cores (`ClusterGrid.h`, `CascadeMath.h`, `VolumetricFog.h`) already used.

`compilePassCulling` became `cullUnusedPasses(passes, textureCount, bufferCount)`. The backward liveness sweep decides which declared passes actually run, and a mistake in it fails silently in both directions: dropping work that was needed, or keeping work that was not, with no validation error either way. Thirteen cases cover the parts that are subtle — a culled pass must not propagate its reads or a dead chain stays alive, a write must clear liveness so a value overwritten before any read culls its producer, `ReadWrite` must clear then set or a read-modify-write culls its own producer, and texture and buffer indices both start at zero and must not alias.

`accessStateForTexture` became `textureAccessState(aspectMask, access, currentLayout)` and `accessStateForBuffer` became `bufferAccessState(access)`. This mapping is what every image transition and buffer barrier in the frame is built from. Eleven cases pin the aspect-dependent layouts — a sampled depth image needs a depth read-only layout and stencil presence selects the combined variant, which matters because Hi-Z, SSR, and GTAO all sample depth — along with both fragment-test stages being in scope for depth attachment writes, storage access being `GENERAL` regardless of aspect, present carrying a layout but no scopes, and a buffer-shaped access on a texture keeping the tracked layout rather than transitioning to `UNDEFINED` and discarding the image.

Both sets were checked against deliberate mutations rather than trusted for passing on the first run. Suite total went from 150 to 180. The extractions are behaviour-preserving: the frame's profiler pass set is identical and validation runs stay clean.

This milestone does not cover `currentTextureLayout`, which is owner-dependent, or the skip logic in `transitionTexture`/`transitionBuffer` that consumes these states.

## Milestone 64: Runtime Settings Coverage for GTAO, Fog, and Punctual Shadows

Tone mapping, bloom, TAA, SSR, CSM, LOD, GI, and the culling toggles all survived a restart. GTAO, volumetric fog, and punctual shadows did not — three shipped subsystems whose settings were lost every time the engine closed, including whether they were enabled at all.

`FogSettings` moved from `VolumetricFogPass.h` to `RuntimeSettings.h`, the move `SsaoSettings` had already made, so serialization and clamping can see it without dragging Vulkan into the settings layer. Its `glm::vec3` scattering colour became three floats for the same reason `GiSettings` stores its grid origin that way, and `kDefaultFogMaxDistance` followed it with `VolumetricFog.h` re-exporting the name. Punctual shadows had no settings struct at all, only three loose booleans on `Renderer`, so `PunctualShadowSettings` is new; its GPU caster-culling toggle follows the culling flags' pattern of being assigned unguarded at startup and gated on availability at runtime.

The clamps are load-bearing rather than cosmetic. Fog `maxDistance` is a divisor in the froxel slice distribution and must stay past the volume's near plane, anisotropy must stay inside the open interval where the Henyey-Greenstein denominator is non-zero, and a temporal blend of 1.0 would keep the history forever. GTAO's slice and step counts bound nested shader loops, so a bad stored value is a performance cliff rather than a visual artefact.

The example settings file turned out to have silently lost its `gi` and `lod` sections entirely. Both were restored, and a test now loads the example over default-constructed settings and asserts nothing changes, which catches a missing section and a drifted value alike.

## Milestone 63: Frame Cost Reduction in the Exposure and Shading Paths

Three GPU costs turned out to be work that was computed and then discarded, rather than algorithms that needed improving. Together they took the frame from roughly 28 ms to 19.4 ms on the demo scene.

`luminance_histogram.comp` ran one invocation per pixel, each doing a global `atomicAdd` into one of 256 bins — millions of atomics contending on 256 addresses. Staging the tally in workgroup-shared memory and flushing only the non-empty bins took the pass from 6.07 ms to 1.96 ms. `exposure_reduce.comp` was then the larger half of what remained: declared `local_size 1,1,1` and dispatched `(1,1,1)`, a single GPU thread walking one luminance partial per 16x16 tile, which at this drawable is 14,400 dependent global reads with nothing to hide the latency behind. Reducing them across a 256-thread workgroup through a shared-memory tree took it to 0.33 ms. The percentile walk stays serial, being inherently sequential over bins that are in shared memory by then.

In the main shading pass, every fragment ran the punctual shadow atlas PCF twice per light: once inside `evaluatePunctualLight` for shading, and once through `punctualShadowDebugFactor` to accumulate a visibility term read only by a debug overlay that is off by default. Gating the accumulation on the flag that consumes it took `MainHDRPass` from 14.07 ms to 11.20 ms and `Transparent`, which reuses the same fragment shader, from 2.44 ms to 1.93 ms.

`docs/profiling.md` was corrected alongside this. It had said only that parent scopes include child scope work, which invites reading `Skybox`/`RenderObjects`/`SkinnedMesh` as a breakdown of `MainHDRPass`. On tile-based deferred hardware the fragment work resolves at `vkCmdEndRendering`, so a scope recorded between draw calls inside a render pass measures command recording and vertex work only and reads near zero whatever it contains.

## Milestone 62: Irradiance-Probe Global Illumination

Lighting was IBL, SSR, and GTAO — all screen-space — so offscreen geometry contributed no indirect light. This milestone adds a grid of irradiance probes storing incoming radiance in small octahedral tiles, plus the distance to what it came from so a probe behind a wall can be rejected rather than lighting through it.

The target exposes neither `VK_KHR_ray_query` nor `VK_KHR_acceleration_structure`, so probe radiance is gathered by rasterising the scene from each probe rather than by tracing rays. Six small cube faces per probe are captured into an atlas and convolved into octahedral tiles by a compute pass, round-robin at a configurable number of probes per frame. The main shading pass blends the eight surrounding probes with Chebyshev visibility and wrapped-cosine backface rejection, replacing the constant IBL irradiance rather than adding to it, since both answer the same question.

Captures are accumulated over time with Halton sub-texel jitter, which is required rather than a refinement: the capture is deterministic, so accumulation without jitter is a no-op. Multi-bounce feeds the previous atlas back into the capture, interpolating between ambient and probe irradiance rather than summing, which would double-count the sky and leave the feedback unbounded.

A Cornell box preset ships with it, because the open demo scene cannot show indirect light well enough to judge the implementation. Loading it disables the sun, replaces the orbiting lights with a single overhead one, and fits the probe grid to the room's interior. It measures a 34% swing in the red/green ratio across the room from walls the probes never see directly, and settles the multi-bounce question the open scene could not: probe-only luminance rises 31% between bounce weights of 0 and 0.95, against 2.2% on the demo scene.

The subsystem is off by default. It does not add ray-traced probe updates, probe relocation, or runtime grid resizing.

## Milestone 61: Renderer.cpp Split Into Focused Translation Units

`Renderer.cpp` had reached 6,368 lines and 168 member definitions. It was split into five translation units — lifecycle and frame loop, resources, scene and materials, per-frame CPU preparation, and command recording — alongside the existing debug-UI unit.

The split moves definitions only. `Renderer.h` is untouched and the class is still one large type. Two habits made a change of this size safe, given that the unit tests barely reach `Renderer` and "it compiles and runs" is otherwise the only signal: the file was partitioned into top-level definitions and reassembly was verified to reproduce the original byte for byte before anything moved, and every definition was verified to still exist exactly once afterwards.

## Milestone 60: Volumetric Fog

A froxel volume over the view frustum is filled with in-scattered light and integrated front to back, then applied in the main shading pass where the view depth needed to find the froxel is already a varying. The grid is 160x90x64 with an exponential depth distribution and its own near and far planes, deliberately not the clustered lighting grid, which is far too coarse for fog. What is shared is the addressing scheme, so a fog froxel can find the light cluster covering it and reuse its light list.

Light shafts come from the same shadow data the shading pass uses: the cascaded shadow map for the sun and the punctual shadow atlas for spots and points. Temporal reprojection follows, because one sample per froxel per frame aliases badly under a high-frequency shadow; jittering the sample and blending against the reprojected previous volume converges it instead.

A per-light importance cull bounds what each light could still contribute to a froxel — brightest channel times intensity times attenuation times the phase peak — and skips it before the shadow fetch when even that bound falls below a threshold. Zero disables the cull, which is the reference the culled result is judged against.

Fog is off by default. The froxel distribution's round trip against the injection pass is pinned by a unit test.

## Milestone 59: Punctual Shadows

Spot and point lights cast shadows through a quadtree-packed atlas. Tiles are sized by projected screen size so a nearby light gets more resolution than a distant one, and lights are ranked so the ranking does not collapse when many lights compete for the atlas.

Two bugs found here are worth preserving. PCF taps must be clamped inside their own atlas tile: neighbouring texels belong to a different tile, which for a cube face is the adjacent face and with the quadtree allocator can be an unrelated light, so a tap that walks out compares against unrelated depth. On a spot that only happened at the cone edge where falloff was already zero, which is why it went unnoticed; on a cube face the tile border is the middle of the lit scene and the mismatch draws hard seams. Separately, the cube face must be selected from the same biased position the projection uses — selecting from the unbiased direction lets the normal offset push a sample across a face boundary into a face whose frustum no longer contains it, and the bounds test then reports "outside the light" and returns fully lit, drawing a bright seam along every face boundary.

The atlas is cached across frames with per-tile invalidation, so a static light's tile is not redrawn, and assignment churn is measured so shadows popping in and out is visible rather than inferred. Optional GPU caster culling exists and is off by default: on this scene it is a net loss, trading CPU frustum tests for a dispatch and its barriers, but the CPU cost scales with slots times draw items.

A shadow-term-only debug view renders punctual visibility on its own, gated on lights actually reaching each fragment so out-of-range casters do not report occlusion they never contribute.

## Milestone 58: Alpha Transparency

glTF `MASK` materials cut out in the fragment shader against a per-material cutoff, before any lighting, shadow, or IBL work, which also keeps clipped fragments off the velocity and G-buffer attachments. The cutoff is carried as a negative value for `OPAQUE` and `BLEND`, so materials that never clip pay one comparison.

Shadow casters are alpha-tested too, so a cutout material casts the shadow of its visible silhouette rather than of its quad.

`BLEND` materials draw in a separate transparent pass after the opaque pass and the G-buffer writes, and before the TAA resolve so blended edges are still antialiased. It is recorded inline, reusing the main pass's push constants, descriptor sets, and viewport rather than rebuilding them, and reuses the same fragment shader.

## Milestone 57: Mesh Level of Detail

Discrete LOD chains are built at load time with meshoptimizer. Level selection runs per draw item inside the GPU cull dispatch, from the projected sphere radius in pixels, with each halving of on-screen radius stepping one level down. The chosen level travels to the fragment stage in the high bits of `gl_InstanceIndex`.

Shadow-cascade dispatches take an additional bias on top of the main one, since shadows tolerate simplification far better than the main pass. A forced-level override pins every draw item to one level for comparison, with -1 kept as the select-by-distance sentinel, and a color-by-LOD debug view keeps the lighting term as luminance so silhouettes still read through the tint.

## Milestone 56: Ground-Truth Ambient Occlusion

The inline depth-only SSAO was replaced with a dedicated GTAO pass (Jimenez et al. 2016) that consumes the main depth buffer and the thin G-buffer normal and writes a visibility texture the composite pass multiplies into scene color.

It arrived in three parts: the horizon-search trace itself, a depth-aware bilateral denoise with an AO debug preview, and finally a half-resolution trace with a joint-bilateral upsample. GTAO is off by default and its toggle is honoured only when the depth image supports sampling.

## Milestone 55: Screen-Space Reflections and the Thin G-Buffer

A view-space linear march with binary refinement against the main depth buffer, blended into scene color before TAA. Surfaces rougher than a threshold trace nothing, fading mirror to glossy, and reflections fade toward the screen edge where the march runs out of information.

This milestone also introduced the thin G-buffer that later work depends on: a second attachment carrying octahedral-encoded normal plus roughness and metallic, written by the main pass. GTAO reads the same attachment.

SSR is on by default and requires a samplable main depth image, the same gate SSAO uses.

## Milestone 54: Async Compute for Clustered Lighting

The `ClusterBuild` and `LightCull` compute passes move to a dedicated compute queue, overlapping the shadow passes rather than serializing behind them. Queue ownership and timeline synchronization are handled by an async-compute subsystem; when the device exposes no async-capable queue the passes fall back to the graphics queue and the toggle is ignored.

On MoltenVK a separate compute queue family is only exposed with `MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1`, so the fallback is the default path there unless that is set.

A related fix landed alongside: the bindless material texture heap needed update-after-bind to satisfy MoltenVK's validation, which had been reporting descriptor-limit errors.

## Milestone 53: Two-Phase Hi-Z Occlusion Culling

Hi-Z occlusion culling became the default by removing the reason it had been opt-in. Phase one tests against the previous frame's depth pyramid, which is fast but wrong wherever the camera moved; phase two rebuilds the pyramid mid-frame and re-tests only the candidates phase one rejected, so disocclusion false negatives are corrected within the same frame rather than popping in the next one.

Turning the toggle off falls back to the conservative single-phase test that only runs while the camera holds still. MoltenVK does not expose `vkCmdDrawIndexedIndirectCount`, so the non-compacted fallback path is what runs on this target.

## Milestone 52: Task-Parallel Frame Preparation

CPU frame preparation moved onto the `JobSystem`. A `parallelFor` primitive was added, and the independent per-frame work — culling inputs, draw-item construction, shadow draw lists, and per-draw object data — now runs as parallel tasks rather than sequentially on the render thread.

## Milestone 51: Motion-Vector TAA

TAA gained real reprojection. The main pass writes a velocity buffer as a second render target from the unjittered current and previous clip positions, and the resolve reprojects the history sample along it, so object and camera motion no longer smear. Same-UV history sampling remains available as an A/B comparison, along with the neighborhood clamp.

The render graph debug tables were also fixed here: they had been collapsing to a bare scrollbar, and their column sizing was wrong under horizontal scrolling.

## Milestone 50: Disk-Backed Pipeline Cache and Built-In Texture Factory

A `VkPipelineCache` is now persisted to disk between runs, cutting pipeline creation time on startup. The blob is keyed on a hash of the compiled SPIR-V, so a shader change invalidates it rather than feeding a stale cache to the driver — MoltenVK is unforgiving about that.

Separately, the procedural built-in textures (checkerboard, flat normal, white, and the portfolio base colour) moved out of `Renderer` into a `BuiltinTextureFactory`.

## Milestone 49: Subsystem Extraction and the Runtime-Library Test Seam

Scene construction, the Hi-Z depth pyramid, and GPU-driven visibility culling were extracted from `Renderer` into `SceneBuilder`, `DepthPyramid`, and `GpuCulling`. Each subsystem owns its own GPU resources but borrows services and settings by reference under the same member names, so relocated bodies compile unchanged and the many call sites elsewhere stay untouched.

Main and shadow culling could not be split from each other — they share one pipeline, one descriptor set layout, and a creation path where the shadow resources are built inside the main ones — so they were extracted as a single `GpuCulling` class. The extraction used a duplicate-then-switch sequence: one commit builds the new class as standalone compiling code, a second rewires `Renderer` and deletes the originals, so both commits are independently green despite the shared descriptor pool forcing the switch itself to be atomic.

The more consequential change is that engine code now builds as a static library, `VulkanEngineRuntime`, which both the executable and the test binary link. That is what makes renderer-side code testable headlessly at all; every test added since depends on it.

## Milestone 48: Render Target Debug Views and CSM Cascade Visualization

The ImGui debug UI now includes read-only `Render Target Debug Views`. HDR scene color, bloom extract/ping/pong targets, the BRDF LUT, the cascaded shadow map array, swapchain composite metadata, and major global cubemap resources expose debug name, dimensions, format, mip count, layer count, intended usage, previewability, and sampled-image type.

Preview descriptors are cached separately from material texture previews through the ImGui Vulkan backend. They reuse existing renderer image views/samplers, including the existing CSM per-layer 2D image views, and are invalidated when post-process, swapchain-dependent, or shadow debug resources are recreated.

CSM cascades can be inspected by selected cascade index, split depth/range, shadow-map layer, resolution, texel snapping state, estimated coverage, visible shadow draw count, and shadow batch count. Cascade depth layers can be visualized as raw sampled depth grayscale/debug previews through the per-layer views.

The BRDF LUT is previewable as a 2D linear data texture for split-sum IBL validation. This milestone is debug visualization only; it does not add render-target editing, asset browsing, material editing, or a full texture viewer/editor.

## Milestone 47: Material Inspector and Texture Debug Views

Milestone 47 added a read-only `Material Inspector` for the selected `RenderObject`. Phase 3 later extends that inspector with material asset scalar editing, save, and reload for JSON material assets. The inspector shows PBR factors, multi-scatter strength, bindless texture indices, material source, and whether the renderer is currently using bindless material textures or the legacy descriptor fallback path.

Selected material texture metadata is visible for base color, normal, and metallic-roughness slots, including debug name, bindless index, dimensions, mip levels, Vulkan format, source, color-space/semantic intent, and whether a fallback texture is used.

Basic texture previews are available for the selected material's base color, normal, and metallic-roughness textures. Preview descriptors are cached through the ImGui Vulkan backend and reuse existing texture image views/samplers without changing renderer material descriptor layouts or bindless material texture bindings.

`Texture Debug Views` also lists metadata for global/post-process resources such as the cascaded shadow map array, irradiance and prefiltered cubemaps, BRDF LUT, scene color, and bloom targets. This milestone is read-only inspection; it does not add material editing, texture import UI, an asset browser, a material graph, or advanced render-target preview tooling.

## Milestone 46: ImGui Scene Hierarchy Viewer

The ImGui debug UI now includes a read-only `Scene Hierarchy` panel. It lists the active renderer `RenderObject` entries from the CPU-side scene data, including stable debug IDs, source type, mesh/material labels, submesh count, bounds, draw-item counts, and available culling/debug status.

Objects can be selected in the hierarchy without changing rendering behavior. The selected-object inspector shows the object name, object index, debug ID, source type, mesh pointer/name, material summary, submesh count, draw-item count, object-data index when available, transform summary or world matrix, local/world bounds, and main/shadow culling metadata when that data is reliable.

When GPU culling is active, the UI does not pretend to know per-object GPU visibility unless that data is actually available; it reports that only aggregate/per-object-readback-unavailable culling data exists. The panel is for inspection only. It does not add transform editing, gizmos, object picking, scene serialization, asset browser, material editing, ECS, or editor architecture.

## Milestone 2: Triangle Rendering

`src/shaders/simple.vert` and `src/shaders/simple.frag` are compiled by CMake into SPIR-V files under the build directory. `VulkanPipeline` loads those `.spv` files, creates shader modules, creates a pipeline layout, and builds a graphics pipeline with `VkPipelineRenderingCreateInfo`.

The pipeline layout still matters because Vulkan pipelines always need a layout describing descriptor sets and push constants. At this stage, the renderer used a vertex-stage push constant for MVP data, while descriptor sets were intentionally left for later texture and sampler work.

Dynamic Rendering does not use a legacy `VkRenderPass`, so the pipeline declares compatible color and optional depth formats through `VkPipelineRenderingCreateInfo`. Viewport and scissor are dynamic states so resizing the window does not require rebuilding the pipeline when only the extent changes.

## Milestone 3: Vertex/Index Buffer Rendering

`VulkanBuffer` is now the RAII owner for buffer handles and VMA allocations. CPU-visible buffers can be filled through `upload`, while GPU-local buffers use a temporary staging buffer and a one-time `vkCmdCopyBuffer` submission. The copy command records a Synchronization2 buffer barrier so transfer writes are visible to vertex and index fetch.

The renderer uses an explicit `Vertex` layout with position and color, device-local vertex and index buffers, and `vkCmdDrawIndexed`. The pipeline receives explicit vertex binding and attribute descriptions, and `simple.vert` reads locations 0 and 1 instead of generating positions from `gl_VertexIndex`.

## Milestone 4: Depth And MVP

Dynamic Rendering now binds both color and depth attachments. The swapchain depth image is transitioned with Synchronization2 into `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL` before rendering, and the graphics pipeline enables depth testing with the swapchain depth format.

Milestone 4 introduced a colored cube, depth testing, and per-frame MVP data. Each frame wrote an MVP matrix to that frame's CPU-visible storage buffer. Those buffers were created with Buffer Device Address support, so the renderer could query each `VkDeviceAddress`.

The vertex shader reads the MVP through `GL_EXT_buffer_reference`. A small vertex-stage push constant carries only the `VkDeviceAddress` of the MVP data, so no descriptor set is used for MVP data in this milestone.

## Milestone 5: Scene Abstractions

The hard-coded cube vertex and index data has moved out of `Renderer` and into `Mesh::createCube()`. `Mesh` owns the GPU-local vertex and index buffers for that built-in cube.

`Renderer` now owns a `Camera`, one cube `Mesh`, and a list of `RenderObject` entries. Each `RenderObject` references a `Mesh` and owns its own `Transform`, giving the renderer a simple draw list instead of direct single-cube draw state.

The MVP is generated from `Camera + Transform`, then uploaded through the existing Buffer Device Address storage-buffer path. A vertex-stage push constant passes the MVP data address to the shader, which reads the MVP through `GL_EXT_buffer_reference`. Descriptor sets are not used for MVP data.

## Milestone 6: Basic Texture Descriptor

Milestone 6 is implemented and introduces descriptor sets only for texture sampling. MVP still uses the existing Buffer Device Address storage-buffer path, with a vertex-stage push constant carrying the current MVP data address. The vertex shader still reads MVP through `GL_EXT_buffer_reference`; it has not moved to a uniform buffer descriptor.

The texture binding contract is:

- set 0, binding 0
- `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`
- fragment shader visibility

At Milestone 6, the texture was still a CPU-generated RGBA8 checkerboard. No image files were loaded at that stage, and no `stb_image` dependency was used yet.

Texture upload uses a CPU-visible staging buffer, a GPU-local `VkImage`, `vkCmdCopyBufferToImage`, and Synchronization2 image barriers:

- `VK_IMAGE_LAYOUT_UNDEFINED` to `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`
- `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL` to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`

The frame binding flow is now:

1. Bind pipeline.
2. Bind texture descriptor set 0.
3. Push object MVP buffer device address.
4. Bind vertex and index buffers.
5. Draw indexed.

Milestone 9 later adds stb_image-based file texture loading and GPU mipmap generation. At Milestone 6, bindless descriptors, lighting, model loading, and render graph work were still future milestones.

## Milestone 7: Basic Material Abstraction

`Material` is now the minimal link between a render object and texture sampling state. It stores a debug name, references a base color `VulkanTexture`, and stores the descriptor set used by the fragment shader's texture binding.

`Renderer` still owns the actual checkerboard `VulkanTexture`, the checkerboard `Material`, the cube `Mesh`, the `Camera`, and the `RenderObject` list. `RenderObject` now references both `Mesh` and `Material`, while continuing to own its `Transform` and debug name.

The descriptor contract is unchanged from Milestone 6:

- set 0, binding 0
- `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`
- fragment shader visibility

MVP data still uses Buffer Device Address plus a vertex-stage push constant. The vertex shader still reads the MVP through `GL_EXT_buffer_reference`, and MVP data has not moved into descriptor uniform buffers.

The per-object draw flow is now:

1. Bind pipeline.
2. For each `RenderObject`, bind its material descriptor set.
3. Push the object MVP buffer device address.
4. Bind the object's mesh vertex and index buffers.
5. Draw indexed.

This milestone does not add PBR, lighting, bindless descriptors, descriptor indexing texture arrays, file texture loading, or model loading.

## Milestone 8: Multi-Object Scene

Milestone 8 is implemented. `Renderer` now draws a small scene with several cube `RenderObject` entries: center, left, right, and elevated cubes. Each render object references the shared cube `Mesh`, references the shared checkerboard `Material`, owns its own `Transform`, and carries a debug name.

MVP data is now per object. Each frame owns one CPU-visible storage buffer large enough for multiple `ObjectFrameData` entries. `updateFrameData()` animates object transforms independently, computes `projection * view * model` for each object, and uploads the resulting MVP matrices into that frame's object-data buffer.

The shader still uses `GL_EXT_buffer_reference`. For each draw, the renderer pushes the Buffer Device Address of the current object's `ObjectFrameData` entry to the vertex stage. MVP data still does not use uniform buffer descriptors.

The texture path is unchanged in Milestone 8: texture/sampler data still uses descriptor set 0 binding 0 as a combined image sampler visible to the fragment shader.

This milestone does not add lighting, PBR, bindless descriptors, file texture loading, model loading, ECS, ImGui, or a render graph.

## Milestone 9: File Texture Loading and Mipmaps

Milestone 9 adds stb_image-based file texture loading and GPU mipmap generation while keeping the existing renderer contracts intact. `VulkanTexture::createFromFile()` loads image data from disk with stb_image, forces RGBA8 pixels, uploads through a CPU-visible staging buffer, and stores the result in a GPU-local `VkImage` allocated with VMA.

When mipmap generation is requested, the texture computes `floor(log2(max(width, height))) + 1` mip levels, creates the image with transfer source, transfer destination, and sampled usage, then generates the mip chain on the GPU with `vkCmdBlitImage`. Synchronization2 image barriers transition all levels from `UNDEFINED` to `TRANSFER_DST_OPTIMAL`, copy level 0, move each previous level to `TRANSFER_SRC_OPTIMAL`, blit into the next level, and finally transition every level to `SHADER_READ_ONLY_OPTIMAL`.

If the format does not support the blit path needed for this simple GPU mip generation, the texture falls back to one mip level. The sampler uses linear min/mag filtering, linear mip filtering, repeat addressing, and a `maxLod` matching the texture mip count. Anisotropy stays disabled for now because the current device wrapper does not explicitly expose and enable `samplerAnisotropy`.

The texture/sampler descriptor contract remains set 0, binding 0 as a combined image sampler. MVP data still uses Buffer Device Address plus a vertex-stage push constant, and the vertex shader still reads per-object MVP data through `GL_EXT_buffer_reference`; MVP data has not moved into uniform buffer descriptors.

`Renderer` tries to load `assets/textures/checker.png` into the existing `checkerboardMaterial_`. If the asset is absent or stb_image fails to decode it, the procedural checkerboard path remains as the fallback. `Material` remains minimal: debug name, base color texture pointer, and descriptor set. This milestone does not add PBR parameters, normal maps, material asset files, bindless descriptors, model loading, lighting, ECS, ImGui, or a render graph.

## Milestone 10: Basic Directional Lighting

Milestone 10 is implemented and adds minimal, non-PBR directional lighting to the textured cube scene. Mesh vertices now contain position, color, UV, and normal attributes. The built-in cube still uses duplicated vertices per face so each face has clean flat normals and UVs.

The vertex shader keeps the existing `GL_EXT_buffer_reference` path. A vertex-stage push constant still carries the Buffer Device Address of the current object's `ObjectFrameData` entry. That entry now contains MVP, model, light direction, light color, and ambient color values. MVP/object data has not moved to uniform-buffer descriptors or any other descriptor set.

Normals are transformed to world space in the vertex shader with `transpose(inverse(mat3(model)))`, then passed to the fragment shader. The fragment shader keeps the texture/sampler at descriptor set 0 binding 0, samples the base color texture, and applies a simple Lambert diffuse term with a small ambient contribution:

```glsl
baseColor * vertexColor * (ambient + diffuse)
```

`Material` remains minimal: debug name, base color texture pointer, and descriptor set. PBR, specular BRDFs, normal maps, IBL/image-based lighting, material parameter buffers, bindless descriptors, model loading, ImGui, ECS, and render graph work remain future milestones.

## Milestone 11: Basic Shadow Mapping

Milestone 11 adds a minimal directional shadow map for the existing directional light. A fixed 2048x2048 sampled depth image is rendered first with a depth-only Dynamic Rendering pass from a simple orthographic light camera covering the cube scene. The shadow pass uses a vertex-only pipeline, depth writes, and static depth bias to reduce acne.

The main pass samples that depth image in the fragment shader and performs one manual depth comparison. The base color texture remains descriptor set 0 binding 0, and the shadow map is descriptor set 0 binding 1. One material descriptor set is still used for now; there are no descriptor arrays or bindless resources.

Object data still uses Buffer Device Address plus a vertex-stage push constant. `ObjectFrameData` now contains `mvp`, `model`, `lightMvp`, light direction, light color, and ambient color. MVP and lighting data have not moved to UBO descriptors.

The Milestone 11 shader/resource contract is:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- object data = Buffer Device Address plus a vertex-stage push constant
- shadow pass = depth-only Dynamic Rendering from the directional light
- main pass = fragment shader samples the shadow map and applies a single depth comparison

This was intentionally not a cascaded shadow implementation. Later milestones add PCF, cascaded shadow maps, basic texel snapping, PBR, normal maps, model loading, and a render graph; multiple lights, ECS, and ImGui remain out of scope for now.

## Milestone 12: Shadow Quality Improvements

Milestone 12 improves the existing directional shadow map without changing the renderer structure. The shadow pass is still one depth-only Dynamic Rendering pass, and the main graphics pipeline still samples the shadow map from descriptor set 0 binding 1.

The fragment shader now uses simple manual 3x3 PCF by averaging neighboring shadow-map depth comparisons. This softens jagged shadow edges compared with the Milestone 11 single-sample comparison while keeping sampler compare mode disabled for now.

`Renderer` now owns tunable shadow settings for shadow-map resolution, small shader-side constant/slope bias values, PCF enable/radius, and static rasterizer depth-bias factors. The static rasterizer depth bias remains on the shadow pipeline to reduce acne, while the shader-side bias stays small to avoid obvious peter panning.

The directional light projection now comes from a documented fixed bounding sphere that covers the current rotating cube demo. This keeps the orthographic near/far planes stable and gives the fixed 2048 shadow map a tighter useful area. This is acceptable for the current static demo scene, but it is still not cascaded shadow mapping, camera-frustum fitting, or texel snapping.

The Milestone 12 resource contract remains:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- object data = Buffer Device Address plus a vertex-stage push constant
- shadow pass = depth-only Dynamic Rendering from the directional light
- main pass = fragment shader shadow-map sampling with simple 3x3 manual PCF
- no PBR, normal maps, bindless descriptors, model loading, ECS, ImGui, or render graph

Later shadow and lighting milestones add cascaded shadow maps, basic texel snapping, PBR, IBL, and a render graph. Variance or EVSM shadows remain unscheduled.

## Milestone 13: Basic PBR Material Parameters

Milestone 13 adds minimal PBR-style material parameters without changing the descriptor layout. `Material` now stores `baseColorFactor`, `metallic`, and `roughness` in addition to its debug name, base color texture pointer, and descriptor set.

Material parameters are passed through the existing Buffer Device Address object-data path. Each `ObjectFrameData` entry now includes `baseColorFactor`, `materialParams`, and `cameraPosition`; `materialParams.x` is metallic, `materialParams.y` is roughness, and `materialParams.zw` are reserved.

The fragment shader still samples the base color texture from descriptor set 0 binding 0 and the shadow map from descriptor set 0 binding 1. It multiplies the texture by `baseColorFactor`, then applies a simple non-IBL diffuse plus Blinn-style specular approximation controlled by roughness and metallic.

This was not full PBR yet. At Milestone 13 there was still no BRDF LUT, IBL, Kulla-Conty multi-scattering compensation, normal maps, metallic/roughness texture maps, bindless material descriptors, model loading, ECS, ImGui, or render graph.

Cook-Torrance GGX, normal mapping, and metallic-roughness texture support are now covered by later milestones. Remaining material and lighting work is tracked in the Next Milestones section.

## Milestone 14: Cook-Torrance GGX Direct Lighting

Milestone 14 replaces the Milestone 13 Blinn-style specular approximation with a direct-light Cook-Torrance GGX BRDF in the fragment shader. The renderer still samples the base color texture from descriptor set 0 binding 0 and the shadow map from descriptor set 0 binding 1, with material values coming through the existing Buffer Device Address object-data path.

The shader computes base color from the texture multiplied by `baseColorFactor`, reads metallic from `materialParams.x`, and reads roughness from `materialParams.y`. Roughness is clamped to `[0.04, 1.0]` to avoid unstable highlights. The direct light BRDF now uses the GGX / Trowbridge-Reitz normal distribution function, Smith geometry function, and Schlick Fresnel approximation. Metallic and roughness now affect the diffuse/specular energy split, `F0`, highlight width, and specular intensity.

Lighting was still direct lighting only at Milestone 14. The PCF-filtered directional shadow factor still modulated the direct light, and ambient remained a simple unshadowed term. There was still no IBL, split-sum BRDF LUT, Kulla-Conty multi-scattering compensation, normal maps, metallic/roughness textures, bindless descriptors, model loading, ECS, ImGui, or render graph.

Normal mapping and metallic-roughness maps are now implemented in Milestones 15 and 16. Remaining material and lighting work is tracked in the Next Milestones section.

## Milestone 15: Basic Normal Mapping

Milestone 15 adds basic tangent-space normal mapping while keeping the renderer architecture simple. Mesh vertices now contain position, color, UV, normal, and tangent attributes. The built-in cube still uses duplicated vertices per face, and its tangents are hardcoded per face; general tangent generation for imported meshes is future work.

`Material` can now reference a normal map in addition to its base color texture. Descriptor set 0 keeps the existing bindings and adds one new sampler:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- set 0 binding 2 = normal map combined image sampler

The vertex shader reads the tangent at location 4, transforms the normal and tangent to world space, computes the bitangent from `cross(normal, tangent) * tangent.w`, and passes the TBN basis to the fragment shader. The fragment shader samples the normal map, decodes the tangent-space normal from `[0, 1]` to `[-1, 1]`, transforms it through TBN, and uses that world-space normal for Cook-Torrance GGX direct lighting and the shadow bias path.

`Renderer` loads `assets/textures/checker_normal.png` when present. If that file is missing or cannot be decoded, it creates a small procedural flat normal texture with RGBA `(128, 128, 255, 255)` and still binds it at descriptor set 0 binding 2. This keeps materials descriptor-complete without adding shader branching or dynamic descriptor behavior.

Object data still uses Buffer Device Address plus a vertex-stage push constant. Normal map state stays in the material descriptor set; `ObjectFrameData` is unchanged. This milestone is still not IBL, a BRDF LUT, Kulla-Conty multi-scattering compensation, bindless descriptors, model loading, glTF, ECS, ImGui, or a render graph.

## Milestone 16: Metallic-Roughness Texture Map

Milestone 16 adds a basic metallic-roughness texture map while keeping the same simple material and object-data architecture. `Material` can now reference a metallic-roughness texture in addition to its base color and normal textures. `Renderer` loads `assets/textures/checker_mr.png` when available; if the file is missing or cannot be decoded, it creates a small procedural fallback texture instead.

Object data still uses Buffer Device Address plus a vertex-stage push constant, and `materialParams.xy` remain the scalar metallic and roughness factors.

Descriptor set 0 is still the material/shadow texture set:

- set 0 binding 0 = base color combined image sampler
- set 0 binding 1 = shadow map combined image sampler
- set 0 binding 2 = normal map combined image sampler
- set 0 binding 3 = metallic-roughness combined image sampler

The metallic-roughness texture uses the R channel as the metallic factor and the G channel as the roughness factor. B and A are unused by the shader. The fragment shader multiplies the sampled texture values by the scalar material factors:

```glsl
metallic = clamp(materialMetallic * textureMetallic, 0.0, 1.0);
roughness = clamp(materialRoughness * textureRoughness, 0.04, 1.0);
```

The procedural fallback uses neutral R/G factors so the existing scalar `Material::metallic` and `Material::roughness` values remain the visible fallback behavior.

GGX direct lighting uses the resulting metallic and roughness values together with the existing normal map and PCF shadow paths. This is still direct lighting only: no IBL, no split-sum BRDF LUT, no Kulla-Conty multi-scattering compensation, and not full glTF material support.

## Milestone 17: IBL Preparation and Environment Texture Infrastructure

Milestone 17 prepares the renderer for image-based lighting without changing the lighting model yet. A new `VulkanEnvironmentMap` wrapper owns a cube-compatible `VkImage`, VMA allocation, `VK_IMAGE_VIEW_TYPE_CUBE` view, and clamp-to-edge sampler. The renderer creates a small generated six-face RGBA8 cubemap during scene setup so later skybox and image-based-lighting milestones have a real GPU cubemap resource to build from.

The environment map upload path uses the same explicit staging-buffer and Synchronization2 style as the existing texture code: all six cube faces are copied into array layers 0 through 5, then transitioned to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`. At Milestone 17, the shader pipeline did not sample this cubemap, so descriptor set 0 remained unchanged and no environment descriptor binding was added in that milestone.

This is intentionally infrastructure only. There is still no split-sum BRDF LUT, no Kulla-Conty multi-scattering compensation, no bindless descriptors, no skybox draw, no environment prefiltering, and no model loading.

## Milestone 18: Skybox Rendering

Milestone 18 renders the procedural environment cubemap from Milestone 17 as a skybox background. The skybox uses a fullscreen triangle, a separate graphics pipeline, and a separate descriptor set layout where skybox set 0 binding 0 is the visible environment cubemap combined image sampler.

At Milestone 18, material descriptor set 0 still contained only the mesh texture and shadow bindings from 0 through 3. Object data continued to use Buffer Device Address plus the existing vertex-stage push constant. The skybox has its own descriptor set and its own vertex-stage push constant containing the inverse view-projection matrix with camera translation removed.

The main Dynamic Rendering pass clears color/depth, draws the skybox first with depth writes disabled, then draws the normal `RenderObject` meshes as before. The shadow pass is unchanged and still runs before the main pass.

Milestone 19 kept the visible skybox cubemap and diffuse irradiance cubemap as separate resources. The skybox cubemap remains the background source, while mesh materials sample the diffuse irradiance cubemap for ambient/environment diffuse lighting.

Later environment work can still add Kulla-Conty multi-scattering compensation, HDR environment loading, bindless descriptors, model loading, and a render graph.

## Milestone 19: Diffuse IBL Irradiance

Milestone 17 created the reusable environment cubemap resource, and Milestone 18 rendered that cubemap as a visible skybox. Milestone 19 adds simple diffuse image-based lighting while keeping the visible skybox path unchanged. The renderer now owns both `environmentMap_` for the skybox and `diffuseIrradianceMap_` for mesh materials.

The diffuse irradiance cubemap is generated procedurally on the CPU from the same six environment face colors, stored as a small low-frequency RGBA8 cubemap, uploaded through the existing `VulkanEnvironmentMap` staging-buffer path, and sampled as a cube image.

Mesh material descriptor set 0 now adds one fragment-stage binding:

- set 0 binding 0 = base color texture
- set 0 binding 1 = shadow map
- set 0 binding 2 = normal map
- set 0 binding 3 = metallic-roughness map
- set 0 binding 4 = diffuse irradiance cubemap
- skybox set 0 binding 0 = visible environment cubemap

The fragment shader samples `uDiffuseIrradianceMap` with the current world-space normal after tangent-space normal mapping. Diffuse IBL contributes `irradiance * baseColor * (1.0 - metallic)` as the ambient/environment diffuse term, with the old ambient color retained only as a small fallback. Direct Cook-Torrance GGX lighting, Schlick Fresnel, Smith geometry, the GGX NDF, metallic-roughness sampling, normal mapping, and PCF shadow filtering remain unchanged; the shadow factor still affects direct lighting only.

This milestone is diffuse IBL only. There is still no prefiltered specular environment map, split-sum BRDF LUT, Kulla-Conty multi-scattering compensation, HDR environment loading, bindless descriptors, model loading, ECS, ImGui, or render graph.

## Milestone 20: Specular IBL and BRDF LUT

Milestone 20 is implemented and adds basic split-sum specular image-based lighting while keeping the skybox descriptor set separate and keeping object/material scalar data on the Buffer Device Address plus vertex-stage push-constant path.

The renderer now owns `environmentMap_` for the visible skybox, `diffuseIrradianceMap_` for diffuse IBL, `prefilteredEnvironmentMap_` for specular IBL, and `brdfLutTexture_` for the split-sum BRDF lookup. The prefiltered specular cubemap is generated on the CPU from the existing procedural environment colors as a mip chain: low roughness mips preserve the face gradients, and higher roughness mips blend toward low-frequency face/global colors. This is a readable approximation, not full importance-sampled environment prefiltering.

The BRDF LUT is a generated 256x256 `VK_FORMAT_R8G8_UNORM` 2D texture. It stores the split-sum scale/bias terms from a small CPU-side Hammersley/GGX integration.

Material descriptor set 0 now contains:

- set 0 binding 0 = base color texture
- set 0 binding 1 = shadow map
- set 0 binding 2 = normal map
- set 0 binding 3 = metallic-roughness map
- set 0 binding 4 = diffuse irradiance cubemap
- set 0 binding 5 = prefiltered specular cubemap
- set 0 binding 6 = BRDF LUT

The fragment shader combines direct Cook-Torrance GGX lighting, PCF shadows on direct light only, diffuse IBL from the irradiance cubemap, and specular IBL from the prefiltered environment plus BRDF LUT. At Milestone 20, this was still not Kulla-Conty, HDR environment loading, bindless rendering, descriptor indexing arrays, model loading, or a render graph.

## Milestone 21: Kulla-Conty-Style Multi-Scattering Compensation

Milestone 21 is implemented and adds a compact multi-scattering compensation
approximation for rough metallic/specular materials. The goal is to reduce the
energy loss that single-scatter GGX can show as roughness increases, especially
on high-metallic materials.

`Material` now has `multiScatterStrength`. The value is passed through
`ObjectFrameData::materialParams.z`, with `materialParams.x` still metallic,
`materialParams.y` still roughness, and `materialParams.w` reserved.
Object/material scalar data remains on the Buffer Device Address plus
vertex-stage push-constant path.

The descriptor layout remains unchanged:

- binding 0 = base color texture
- binding 1 = shadow map
- binding 2 = normal map
- binding 3 = metallic-roughness map
- binding 4 = diffuse irradiance cubemap
- binding 5 = prefiltered specular cubemap
- binding 6 = BRDF LUT

No descriptor layout change was required for this milestone.

The fragment shader keeps the existing direct GGX, PCF shadow, diffuse IBL,
prefiltered specular IBL, BRDF LUT, and normal-map paths. It estimates average
Schlick Fresnel, uses the existing BRDF LUT scale/bias to estimate remaining
single-scatter specular energy, and adds a bounded roughness-squared, mostly
metallic-weighted, and `multiScatterStrength` scaled term to specular IBL.

This is an educational approximation, not a full production Kulla-Conty LUT
implementation.

## Milestone 22: Minimal Render Graph

Milestone 22 adds a small `RenderGraph` layer without changing the renderer's
visual output. The current frame is represented as two pass nodes:

- `ShadowPass` writes the directional shadow map depth image.
- `MainPass` reads the shadow map, writes the swapchain color image, writes the
  main depth image, reads the material textures, and reads the IBL resources.

`MainPass` still draws the skybox first and then the mesh `RenderObject`s. The
material descriptor set remains set 0 bindings 0 through 6, the skybox
descriptor set remains separate, and object/material data still uses Buffer
Device Address plus the vertex-stage push constant.

The graph centralizes the existing Synchronization2 transitions for the shadow
map, swapchain color image, main depth image, and present transition. Dynamic
Rendering is still used for both the depth-only shadow pass and the main
color/depth pass.

This is not a full production render graph yet. It does not perform automatic
dependency inference, transient resource allocation, attachment aliasing, async
compute scheduling, pass culling, or render graph visualization.

## Milestone 23: GPU Debug Labels and Timestamp Profiling

Milestone 23 adds lightweight GPU inspection and profiling support without
changing visual output, descriptor layouts, the Buffer Device Address object-data
path, or render graph pass order.

`VulkanDebugUtils` wraps `VK_EXT_debug_utils` object names and command-buffer
labels. If the extension or function pointers are unavailable, the helpers are
safe no-ops. When available, major Vulkan objects get readable names, including
swapchain images and image views, the main depth image, shadow map resources,
material textures, IBL cubemaps, the BRDF LUT, pipelines, descriptor set
layouts, and pipeline layouts.

Frame command recording now emits debug labels around:

- `Frame`
- `ShadowPass`
- `MainPass`
- `Skybox`
- `RenderObjects`

These labels are intended to show up in RenderDoc and NSight captures.

The Phase 1 engine-upgrade work supersedes the original fixed
`VulkanTimestampQuery` ranges with `GpuProfiler`. It owns one timestamp query
pool per frame-in-flight slot, records named scopes dynamically, and reads the
completed frame slot after the existing fence wait. Elapsed GPU time is
computed from the device timestamp period as `(end - begin) * timestampPeriod /
1e6`.

Timing output is throttled to about once per second to avoid console spam:

```text
GPU timings:
  Frame total: X ms
  timestamp queries: N/256
  CSMShadowPass: A ms
  MainHDRPass: B ms
  CompositePass: C ms
```

If timestamp queries are not supported, the engine prints one warning and leaves
profiling disabled.

Future profiling/debugging work can add more detailed per-material or per-draw
profiling, CPU/GPU timestamp calibration, and a documented RenderDoc capture
workflow.

## Milestone 24: Static glTF Mesh Loading

Milestone 24 added static glTF geometry only. Milestone 25 extends this with basic glTF material and texture loading.

The renderer uses tinygltf to load the first glTF mesh and merges supported triangle primitives into one `Mesh`. Geometry is converted into the existing `Vertex` format:

- `POSITION` -> location 0 `vec3 position`, required
- missing vertex color -> location 1 `vec3 color = vec3(1.0)`
- `TEXCOORD_0` -> location 2 `vec2 uv`, fallback `vec2(0.0)`
- `NORMAL` -> location 3 `vec3 normal`, fallback `vec3(0.0, 1.0, 0.0)`
- `TANGENT` -> location 4 `vec4 tangent`, fallback `vec4(1.0, 0.0, 0.0, 1.0)`

Indices are converted to `uint32_t`; unsigned byte, unsigned short, and unsigned int index accessors are supported. If a primitive has no index accessor, the loader generates sequential indices. Non-triangle primitives are skipped with a warning.

Loaded vertices and indices are uploaded through the existing staging-buffer path into GPU-local vertex and index buffers. At Milestone 24, imported geometry used existing engine `Material` objects and the existing descriptor set layout. The renderer tries `assets/models/test_mesh.gltf` and then `assets/models/test_mesh.glb`; if loading fails or assets are missing, the built-in cube scene remains the fallback and useful test geometry.

At Milestone 24, glTF material and texture loading were still future work; Milestone 25 adds the first material and texture loading path, and Milestone 26 adds static scene node traversal. glTF positions and node transforms are currently preserved as authored; no handedness or up-axis conversion is applied yet. Proper tangent generation for meshes without tangents is also future work.

## Milestone 25: glTF Material and Texture Loading

Milestone 25 reads glTF primitive material indices and stores them as `MeshPrimitive` ranges with `firstIndex`, `indexCount`, and `materialIndex`. Imported geometry can remain in one uploaded `Mesh`, while the renderer draws each submesh range with the assigned engine `Material`.

glTF `baseColorFactor`, `metallicFactor`, and `roughnessFactor` are mapped into engine material scalar data. glTF base color, normal, and metallic-roughness texture references are read from the material's PBR fields, and external image URIs are resolved relative to the source `.gltf` file before loading through `VulkanTexture::createFromFile()`. Embedded/data-URI image bytes are also accepted when tinygltf exposes them as encoded PNG/JPEG data.

Missing or failed material textures use descriptor-complete fallbacks: base color uses the existing checker/base texture, normal uses a flat procedural normal texture, and metallic-roughness uses a neutral procedural MR texture. The material descriptor set layout remains unchanged:

- binding 0 = material base color texture
- binding 1 = global shadow map
- binding 2 = material normal map
- binding 3 = material metallic-roughness map
- binding 4 = global diffuse irradiance cubemap
- binding 5 = global prefiltered specular cubemap
- binding 6 = global BRDF LUT

Each imported glTF material gets its own descriptor set, while global shadow and IBL resources are shared. Object/material scalar data continues to use Buffer Device Address plus the existing vertex-stage push constant.

This milestone does not add animation, skinning, alpha blending, emissive textures, occlusion textures, bindless descriptors, descriptor indexing arrays, render graph scheduling changes, or HDR environment loading.

## Milestone 26: glTF Scene Node Hierarchy

Milestone 26 traverses the default glTF scene when one is present, or scene 0 otherwise. Root `scene.nodes` are visited recursively, each node's local transform is computed, and parent/child transforms are accumulated into a world matrix. Nodes with a mesh create renderer `RenderObject`s using that world transform.

Both glTF node transform forms are supported. If `node.matrix` is authored, the loader uses the 4x4 matrix directly. Otherwise it builds `translation * rotation * scale` from TRS fields, with glTF quaternions interpreted as `[x, y, z, w]`. The accumulated world matrix is stored through `Transform::fromMatrix()`, so static imported objects can preserve hierarchy results that do not map cleanly to the engine's Euler TRS fields.

Imported `Mesh` objects are stored by glTF mesh index in renderer-owned mesh slots. Multiple glTF nodes referencing the same mesh create multiple `RenderObject`s that point at the same uploaded mesh buffers. Meshes still merge supported triangle primitives into one `Mesh`, and `MeshPrimitive` material assignment continues to control submesh material binding.

Current coordinate assumptions are intentionally simple: glTF's right-handed authoring convention is used as-is. There is no handedness, up-axis, unit, or scene-scale conversion yet.

This is static hierarchy support only. It does not add animation, skinning, morph targets, glTF cameras, glTF lights, ECS, bindless descriptors, or material/shader binding changes. If glTF loading fails, or if no supported glTF asset exists, the built-in cube fallback scene remains available.

## Milestone 27: Scene Bounds and Frustum Culling

Milestone 27 adds CPU-side static bounds and basic camera frustum culling for the main pass. `Mesh` now stores a local-space AABB, computed from the built-in cube vertices or from glTF `POSITION` data. The glTF loader uses accessor min/max metadata when available and falls back to expanding bounds from decoded vertex positions otherwise.

Each `RenderObject` can compute a world-space AABB by transforming its mesh-local bounds with `transform.modelMatrix()`. This keeps imported scene nodes using `Transform::fromMatrix()` on the same model-matrix path as TRS-based objects.

The camera frustum is extracted from `projection * view`. The extraction code rebuilds GLM matrix rows explicitly because GLM stores matrices by column, then uses Vulkan clip-space rules: `x` and `y` are in `[-w, w]`, while `z` is in `[0, w]`. Planes are normalized before AABB tests.

Main-pass drawing now skips objects whose world-space AABB is outside the camera frustum. The shadow pass still draws all objects for now, so shadow behavior stays simple and unchanged. Object/material scalar data is still uploaded through the existing Buffer Device Address path; culling only skips main-pass draw calls.

Culling statistics are logged with the throttled GPU timing output:

```text
Culling: total=N visible=M culled=K
```

This is CPU frustum culling only. It does not add occlusion culling, GPU culling, indirect drawing, a BVH, an octree, LOD, ECS, animation, skinning, bindless descriptors, shader binding changes, descriptor layout changes, or render graph scheduling changes.

Future culling and scene-management work can add GPU shadow caster culling, aggregate scene bounds, spatial partitioning, BVH or octree acceleration, compute-built indirect commands, occlusion culling, and LOD.

## Milestone 28: Indirect Draw Preparation

Milestone 28 introduces a compact `DrawItem` record as the renderer-side bridge between scene objects and Vulkan draw commands. Each draw item stores the mesh, resolved material, render object index, submesh index, index range, and vertex offset. The renderer builds draw items from `RenderObject`s and `MeshPrimitive` submeshes, so imported glTF submesh material assignments decide which material is used for the draw.

CPU frustum culling now produces a separate visible main-pass draw item list. The culling test still runs per `RenderObject` against its world-space AABB, and the shadow pass is intentionally left on the simpler all-objects path for now.

Visible main-pass draws are also mirrored into a CPU-visible per-frame indirect command buffer. Each visible draw item writes one `VkDrawIndexedIndirectCommand` with `indexCount`, `instanceCount = 1`, `firstIndex`, `vertexOffset`, and `firstInstance = 0`.

The main pass now uses `vkCmdDrawIndexedIndirect` for mesh draws. At Milestone 28, material descriptors were still bound per draw and object/material scalar data still used the Buffer Device Address plus vertex-stage push constant path. Each draw therefore bound descriptor set 0, pushed the selected object's BDA address, bound the mesh vertex/index buffers when needed, and then issued a one-command indirect indexed draw.

This was preparation for GPU culling and later bindless rendering. It did not add GPU culling, compute-built command generation, multi-draw indirect count, descriptor indexing arrays, ECS, ImGui, animation, skinning, occlusion culling, BVH or octree acceleration, shader resource model changes, descriptor layout changes, or render graph scheduling changes.

Milestone 29 adds GPU culling and compute-built indirect commands. Milestone 30 adds the bindless material texture path.

## Milestone 29: Compute-Based GPU Frustum Culling

Milestone 29 moves main-pass frustum culling and indirect command generation onto the GPU while keeping the existing material descriptor contract and Buffer Device Address object-data path. The CPU still builds the full `DrawItem` list each frame, but when GPU culling is active it uploads all draw item bounds and draw parameters into a per-frame CPU-visible storage buffer instead of compacting a visible-only list.

The culling input record mirrors `src/shaders/cull.comp` in std430 layout:

```cpp
struct GpuCullDrawItem {
    vec4 boundsMin;      // xyz = world-space AABB min
    vec4 boundsMax;      // xyz = world-space AABB max
    uint indexCount;
    uint firstIndex;
    int  vertexOffset;
    uint objectIndex;
};
```

The two `vec4` bounds are 16-byte aligned, and the four scalar draw fields occupy the next 16 bytes, so each runtime-array element has a 48-byte stride. The compute push constant stores six `vec4` frustum planes plus a small `uvec4` parameter block where `x` is the draw item count.

Each frame also owns an indirect command output buffer with both `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` and `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`. The compute shader reads one culling input record per invocation, tests the world-space AABB against the camera frustum, and writes one `VkDrawIndexedIndirectCommand`-compatible record. Visible draw items receive the real `indexCount`, `instanceCount = 1`, `firstIndex`, and `vertexOffset`; culled draw items write zero-count commands with `instanceCount = 0`.

The GPU culling descriptor set is separate from material set 0:

- set 0 binding 0 = cull input storage buffer
- set 0 binding 1 = indirect output storage buffer

After the shadow pass, the renderer records the `GpuCulling` / `ComputeCullDispatch` debug labels, binds the compute pipeline, binds the current frame's cull descriptor set, pushes the frustum planes, dispatches `ceil(drawItemCount / 64)` workgroups, and inserts a Synchronization2 buffer barrier from compute shader storage writes to draw-indirect command reads. The main pass then continues to use `vkCmdDrawIndexedIndirect`.

At Milestone 29, the CPU still looped over main-pass draw items because material descriptors were still bound per draw and object/material scalar data still used the existing BDA plus vertex-stage push constant path. Each draw still bound material descriptor set 0, pushed the selected object's object-data address, bound the mesh buffers as needed, and issued one indirect indexed draw. GPU culling simply made culled commands do zero work. Milestone 30 adds the bindless material texture path so per-material descriptor binding is no longer needed on devices that support descriptor indexing.

CPU frustum culling remains the fallback if `useGpuCulling_` is false or GPU culling resource creation fails. At Milestone 29 the GPU path did not read the visible count back yet; Milestone 32 later adds that count without changing the zero-count indirect command fallback. The shadow pass remains direct `vkCmdDrawIndexed` over all draw items for now.

Future GPU-driven rendering work can add multi-draw indirect count, compact object/material buffers, GPU-driven material indexing, occlusion culling, BVH / spatial partitioning, and LOD. Milestone 30 adds the first bindless material descriptor path.

## Milestone 30: Bindless Material Descriptors

Milestone 30 adds a simple fixed-size bindless texture heap for material sampling while keeping object and material scalar data on the existing Buffer Device Address path. `BindlessTextureHeap` owns one descriptor set layout, one descriptor pool, and one descriptor set. The bindless heap uses descriptor set 1 with 256 slots per material texture class:

- set 1 binding 0 = base color combined image sampler runtime array
- set 1 binding 1 = normal combined image sampler runtime array
- set 1 binding 2 = metallic-roughness combined image sampler runtime array

`VulkanDevice` enables the descriptor indexing features needed by this path when supported: runtime descriptor arrays, partially bound descriptor bindings, and non-uniform sampled image array indexing. The heap does not use variable descriptor count or update-after-bind. If the required descriptor indexing features are unavailable, the renderer logs a warning and keeps the Milestone 29 per-material descriptor set path.

Each `Material` stores `baseColorTextureIndex`, `normalTextureIndex`, and `metallicRoughnessTextureIndex`. Materials receive those indices when their textures are registered into the bindless heap. The first registered entries are fallbacks: checker/base color, flat normal, and neutral metallic-roughness.

`ObjectFrameData` now includes a `uvec4 textureIndices` field. Texture indices travel through the existing BDA object-data path. The CPU still loops over `DrawItem`s and pushes a BDA address per draw, but the address points at draw-specific frame data so submesh material indices can feed the shader without moving to a full material buffer yet.

`simple_bindless.frag` samples material textures with descriptor indexing and `nonuniformEXT`:

- `uBaseColorTextures[nonuniformEXT(vTextureIndices.x)]`
- `uNormalTextures[nonuniformEXT(vTextureIndices.y)]`
- `uMetallicRoughnessTextures[nonuniformEXT(vTextureIndices.z)]`

Global resources remain fixed in the transitional set 0 layout: shadow map at binding 1, diffuse irradiance at binding 4, prefiltered specular at binding 5, and BRDF LUT at binding 6. The main pass binds set 0 and set 1 once on the bindless path, then draws with indirect indexed commands without binding a unique material descriptor set per draw. The skybox descriptor set, shadow pass, compute culling pipeline, lighting math, normal mapping, IBL, BRDF LUT, Kulla-Conty-style compensation, and render graph pass order are unchanged.

Future work can replace draw-specific frame material data with compact material buffers, make material indexing fully GPU-driven, add multi-draw indirect count, support bindless samplers or separate image/sampler descriptors, add texture streaming, resize descriptor heaps, and improve model loading.

## Milestone 31: Multi-Draw Indirect and Object-Data Array Indexing

Milestone 31 moves the bindless main-pass path closer to a GPU-driven renderer. Instead of pushing a different `ObjectFrameData` device address before each bindless draw, the renderer pushes the base address of the current frame's `ObjectFrameData` array. Indirect commands use `firstInstance` as the object-data index, and `simple.vert` uses `gl_InstanceIndex` to read `ObjectFrameData` from that array.

Main-pass `DrawItem`s are ordered into mesh-compatible ranges and then grouped into mesh batches. Each batch stores the mesh, the first indirect command, and the command count. The renderer binds the mesh vertex and index buffers once per batch and submits the range with one `vkCmdDrawIndexedIndirect` call, so compatible batches can use `drawCount > 1`.

Compute culling still writes one indirect command per draw item. Visible draw items receive their real index range, `instanceCount = 1`, and `firstInstance = objectFrameDataIndex`; culled draw items write zero-count commands. Milestone 31 intentionally keeps zero-count commands instead of compacting the visible list, so `vkCmdDrawIndexedIndirectCount` and per-batch count buffers are left for later.

Material textures continue to be sampled through the bindless descriptor arrays:

- set 1 binding 0 = base color texture array
- set 1 binding 1 = normal texture array
- set 1 binding 2 = metallic-roughness texture array

The shadow pass remains the simpler direct path for now. It still draws all shadow-casting draw items directly and pushes the per-object BDA address, which keeps shadow rendering independent from the new main-pass batching work.

The renderer checks and enables `multiDrawIndirect` and `drawIndirectFirstInstance` when the device supports them. If descriptor indexing or the required indirect features are unavailable, the main pass falls back to the Milestone 29 style: CPU loop over draw items, one indirect command per draw, per-draw BDA push constants, and per-material descriptor binding when bindless textures are unavailable.

Milestone 31 is a bridge toward full GPU-driven rendering. Future work can add `vkCmdDrawIndexedIndirectCount`, compacted visible command buffers, fully GPU-built draw batches, bindless object/material buffers, shadow pass indirect drawing, occlusion culling, BVH or other spatial partitioning, and LOD.

## Milestone 32: GPU Visible Count and Indirect Count Preparation

Milestone 32 adds a per-frame GPU visible draw count to the main-pass compute culling path. Each frame owns a 32-bit visible count buffer with storage, indirect, transfer-destination, and transfer-source usage. Before the culling dispatch, the renderer resets that count with `vkCmdFillBuffer`, then uses a Synchronization2 buffer barrier so the compute shader's atomic increment path sees the cleared value.

The GPU culling compute descriptor set now has three storage-buffer bindings:

- binding 0 = per-frame culling input storage buffer
- binding 1 = per-frame indirect command output storage buffer
- binding 2 = per-frame visible draw count storage buffer

`cull.comp` now increments the visible count for each draw item whose world-space AABB passes the frustum test. The active path still writes one indirect command per draw item: visible items keep `indexCount`, `instanceCount = 1`, `firstIndex`, `vertexOffset`, and `firstInstance = objectFrameDataIndex`, while culled items write zero-count commands. The shader also has a compact-output mode for future work, but the renderer leaves it disabled in this milestone so mesh-compatible batch ranges stay valid.

After compute culling, the graph barriers the indirect command buffer and visible count buffer for draw-indirect reads in `MainHDRPass`, while the renderer keeps a manual transfer-read barrier for the immediate visible-count copy. It then copies the count into a small CPU-visible readback buffer and reads it after the existing frame fence. The throttled GPU timing log now includes:

```text
GPU culling:
  total draw items: N
  visible draw items: M
  culled draw items: N - M
```

`VulkanDevice` also queries and logs `vkCmdDrawIndexedIndirectCount` availability and `maxDrawIndirectCount`. The renderer does not use `vkCmdDrawIndexedIndirectCount` yet, because the current compacted command stream would need per-batch visible ranges or per-batch count buffers to preserve mesh-compatible binding. The fallback remains the Milestone 31 zero-count indirect command buffer, and devices without the required indirect features continue to use the existing CPU/per-draw indirect path.

Bindless material descriptors, the `ObjectFrameData` Buffer Device Address array, `firstInstance` object-data indexing, timestamp profiling, render graph pass order, shadow mapping, IBL, the BRDF LUT, and Kulla-Conty-style compensation are unchanged. The shadow pass remains direct draw over all draw items.

Milestone 33 follows by adding compacted visible command buffers, per-batch indirect count buffers, and `vkCmdDrawIndexedIndirectCount` for each mesh batch.

## Milestone 33: Per-Batch Indirect Count and Command Compaction

Milestone 33 replaces the active GPU culling draw path's zero-count command stream with compacted visible commands per mesh-compatible batch when the device supports `vkCmdDrawIndexedIndirectCount`.

The renderer still builds mesh-compatible batches on the CPU. Each batch stores its mesh pointer, begin draw-item index, draw-item count, compacted indirect command offset, and offset into a per-frame batch visible-count buffer. The count buffer is one GPU buffer per frame with one `uint` entry per batch, using storage, indirect, transfer-source, and transfer-destination usage.

The per-frame indirect command buffer is now also the compacted output buffer. Each batch owns a fixed region sized to that batch's draw-item count. During compute culling, visible draw items atomically append to `batchVisibleCounts[batchIndex]`, write into `batchOutputBase + localVisibleIndex`, and emit a `VkDrawIndexedIndirectCommand` with `instanceCount = 1` and `firstInstance = objectFrameDataIndex`. Culled draw items do not write compacted commands.

Before dispatch, the renderer clears the batch count buffer with `vkCmdFillBuffer` and uses Synchronization2 so the compute shader sees zeroed counts. After dispatch, Synchronization2 barriers make the compacted command buffer and batch count buffer visible to indirect drawing, and the count buffer is copied to a CPU-visible readback buffer for throttled logging.

On the bindless multi-draw path, the main pass binds global set 0 and bindless material texture set 1, pushes the current frame's `ObjectFrameData` base address, binds each batch's mesh buffers, and calls `vkCmdDrawIndexedIndirectCount` with:

- indirect buffer offset = `batch.compactedCommandOffset * sizeof(VkDrawIndexedIndirectCommand)`
- count buffer offset = `batch.visibleCountOffset`
- max draw count = `batch.drawItemCount`
- stride = `sizeof(VkDrawIndexedIndirectCommand)`

`firstInstance` still selects the `ObjectFrameData` entry. Bindless material textures remain unchanged. The shadow pass remains direct `vkCmdDrawIndexed`. If GPU culling, bindless multi-draw indirect, indirect-count support, or the compacted count resources are unavailable, the renderer keeps the old zero-count indirect path or CPU-visible draw-list fallback.

The throttled log now reports total draw items, visible draw items, culled draw items, batch count, and whether the indirect-count path was enabled.

Future GPU-driven work can add fully GPU-built mesh batches, GPU-driven material/object buffers, GPU shadow caster culling, occlusion culling, BVH or other spatial partitioning, LOD, and mesh/task shaders in a later renderer branch.

## Milestone 34: Shadow Pass Indirect Drawing and Shadow Caster Culling

Milestone 34 moves the shadow pass off the old direct all-draw loop. The renderer now keeps an explicit all-submesh draw item list, builds a separate shadow draw item list from `RenderObject`s and `MeshPrimitive` submeshes, and treats every opaque renderable as a shadow caster. Alpha-tested shadows are still out of scope.

Shadow casters are culled on the CPU against the directional light frustum from the existing light view-projection. Each render object computes its world-space AABB from mesh-local bounds and its model matrix; if the AABB intersects the light frustum, all of that object's submesh draw items are included in the shadow draw list.

Visible shadow draw items are grouped into mesh-compatible batches using the same mesh pointer and vertex/index buffers. Each frame owns a shadow indirect command buffer with `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` and `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. For each visible shadow draw item, the renderer writes one `VkDrawIndexedIndirectCommand` with the draw item's index range, `instanceCount = 1`, and `firstInstance = objectFrameDataIndex`.

The shadow vertex shader now matches the main-pass object-data array model. The shadow indirect path pushes the current frame's `ObjectFrameData` base address once, and indirect `firstInstance` selects the object-data entry through `gl_InstanceIndex`. If shadow indirect drawing is unavailable, the renderer falls back to direct `vkCmdDrawIndexed` calls and pushes the per-draw object-data address.

The shadow pass remains material-independent. It does not bind material descriptor sets, does not sample material textures, and does not change material or global descriptor layouts. Main-pass GPU culling, bindless material descriptors, per-batch indirect-count drawing, skybox rendering, IBL, the BRDF LUT, Kulla-Conty-style compensation, synchronization, and render graph pass order are unchanged.

The throttled timing log now also reports:

```text
Shadow culling:
  total shadow draw items: N
  visible shadow draw items: M
  culled shadow draw items: N - M
  shadow batches: B
```

Future shadow and GPU-driven work can add GPU-built shadow batches, cascaded shadow maps, alpha-tested shadow casters, shadow LOD, stable crop matrices, cascade blending, shadow caster culling acceleration structures, occlusion culling, and mesh/task shaders.

## Milestone 35: GPU Shadow Culling Preparation

Milestone 35 adds an optional GPU culling path to the shadow pass while keeping the Milestone 34 CPU shadow culling and direct draw fallbacks. The CPU still builds the draw item list and mesh-compatible shadow batches, but each frame now uploads GPU shadow culling input records containing world-space AABB min/max, indexed draw parameters, the `ObjectFrameData` index, the shadow batch index, and the compacted batch output base. The record mirrors `src/shaders/cull.comp` in std430 layout, with two 16-byte AABB vectors followed by packed 32-bit draw and batch fields for a 64-byte stride.

The shadow path reuses `cull.comp` by pushing the directional light frustum planes instead of the camera frustum planes. The shader tests each shadow draw item AABB, atomically appends visible items into that item's mesh-compatible shadow batch, writes `VkDrawIndexedIndirectCommand`-compatible commands, and uses `firstInstance = objectFrameDataIndex` so the shadow vertex shader continues to index the per-frame `ObjectFrameData` array.

Each frame now has a separate shadow culling descriptor set, independent of material descriptors:

- binding 0 = shadow cull input storage buffer
- binding 1 = shadow compacted indirect output storage buffer
- binding 2 = shadow batch visible-count storage buffer

Before shadow rendering, the command buffer records `GpuShadowCulling` and `ShadowCullDispatch`, clears per-batch visible counts and the shadow indirect command buffer, dispatches the compute cull, then barriers shadow indirect/count writes for indirect drawing and readback. The shadow pass then records `ShadowIndirectDrawBatches` and draws each shadow mesh batch with `vkCmdDrawIndexedIndirectCount` when available, or `vkCmdDrawIndexedIndirect` over zero-cleared fallback command slots otherwise.

If GPU shadow culling setup is unavailable or `useGpuShadowCulling_` is disabled, the renderer keeps using CPU light-frustum shadow culling, the existing shadow indirect path, and the direct `vkCmdDrawIndexed` fallback when shadow indirect drawing is unavailable. Main-pass GPU culling, bindless material rendering, material descriptor layouts, the ObjectFrameData BDA array path, skybox rendering, IBL, the BRDF LUT, Kulla-Conty-style compensation, swapchain synchronization, and render graph pass order remain unchanged apart from the pre-shadow shadow-cull compute barriers.

This milestone is still not cascaded shadow mapping, alpha-tested shadows, occlusion culling, BVH or octree culling, LOD, ECS, animation, skinning, ImGui, or a new shading feature.

Future work:

- cascaded shadow maps
- alpha-tested shadow casters
- GPU-built shadow batches
- spatial partitioning / BVH
- occlusion culling
- LOD
- mesh/task shaders

## Milestone 36: Cascaded Shadow Maps

Milestone 36 replaces the single directional shadow map with a minimal cascaded shadow map for the camera view. The renderer owns simple CSM settings for cascade count, practical-split lambda, near/far depth, shadow distance, and shader depth bias. The default path uses four cascades.

The shadow resource is now one 2D array depth image. Array layers equal the active cascade count, the sampled descriptor remains set 0 binding 1, and shaders sample it as `sampler2DArray`. The array image has one sampling view for the main pass plus one 2D attachment view per layer so Dynamic Rendering can render each cascade separately without geometry-shader layered rendering.

Each frame computes cascade split depths between the camera near plane and `shadowDistance` with the practical split scheme:

```text
uniformSplit = near + (shadowDistance - near) * cascadeRatio
logSplit = near * pow(shadowDistance / near, cascadeRatio)
split = mix(uniformSplit, logSplit, lambda)
```

For each cascade, the renderer builds the camera frustum-slice corners in world space, transforms them into a directional-light view, fits orthographic bounds around those corners, and stores the resulting light view-projection matrix. The per-draw `ObjectFrameData` now stores four `lightMvp` matrices, `cascadeSplits`, shadow settings, and camera-forward data. This keeps the existing BDA plus vertex-stage push-constant path and the indirect `firstInstance` object-data indexing model. The larger object-data stride is accepted for this educational milestone; future work can move scene/light data into a separate buffer.

The shadow pass records a `CSMShadowPass` label and one `ShadowCascadeN` label per cascade. When GPU shadow culling is active, the existing compute culling path is reused per cascade by pushing that cascade's light-frustum planes, rebuilding the compacted shadow indirect commands, and drawing through the shadow indirect-count path when available. If GPU shadow culling or shadow indirect drawing is unavailable, the renderer uses CPU per-cascade shadow-caster culling and direct shadow draws.

The main vertex shader outputs light-space positions for the four cascades plus the fragment view depth. The fragment shader selects the cascade by comparing view depth against `cascadeSplits`, samples the matching layer of the shadow-map array, and reuses the existing 3x3 PCF depth comparisons for that layer. Main-pass GPU culling, bindless material descriptors, indirect-count drawing, skybox rendering, IBL, the BRDF LUT, and Kulla-Conty-style compensation are unchanged.

Current limitations:

- no alpha-tested shadow casters
- no GPU-built cascade batch system yet
- no VSM or EVSM
- basic texel snapping and cascade debug tinting are covered by Milestone 37

Future CSM and shadow work:

- better cascade split tuning
- stable crop matrices
- cascade blending
- per-cascade resolution control
- alpha-tested shadows
- GPU-built cascade batches
- shadow LOD
- VSM or EVSM
- occlusion culling
- BVH or spatial partitioning

## Milestone 37: CSM Stabilization and Cascade Debug Visualization

Milestone 37 keeps the Milestone 36 CSM resource model and adds basic CSM texel snapping plus optional cascade debug tinting. `CsmSettings` exposes the active cascade count, texel-snapping toggle, and debug-color toggle with conservative hardcoded defaults. The shadow resource remains a 2D array depth image, and the main shader still samples it as `sampler2DArray` from descriptor binding 1.

For each cascade, the renderer still computes practical split depths and fits light-space orthographic bounds around the camera frustum slice. When texel snapping is enabled, it derives:

```text
worldUnitsPerTexel = orthoExtent / shadowResolution
```

The directional-light view center is then snapped to that increment in the light right/up axes, with a one-texel guard band on the fitted orthographic bounds. This reduces shadow shimmering caused by sub-texel camera motion. It is intentionally a basic CSM stabilization method, not a production-grade solution with stable crop matrices, cascade blending, or per-cascade resolution control.

The main fragment shader can optionally tint shaded pixels by the selected cascade index: red, green, blue, and yellow for cascades 0 through 3. The tint is mixed subtly over the final lighting result so it can diagnose cascade selection without replacing material shading. The debug flag is carried through the existing `ObjectFrameData` BDA path by using `cameraPosition.w`; no descriptor set, UBO, material layout, bindless texture set, or push-constant contract was added.

The throttled timing/culling log now reports the active cascade count plus texel snapping and debug color states.

Main-pass GPU culling, bindless material descriptors, indirect-count drawing, GPU shadow culling, the shadow indirect path, IBL, the BRDF LUT, Kulla-Conty-style compensation, render graph pass order, and swapchain synchronization are unchanged.

Future CSM and shadow work:

- better cascade split tuning
- stable crop matrices
- cascade blending
- per-cascade resolution control
- alpha-tested shadow casters
- VSM/EVSM
- shadow debug UI
- ImGui controls

## Milestone 38: Texture Color Space and sRGB Correctness

Milestone 38 separates texture color-space intent in the existing material texture pipeline. `VulkanTexture` now accepts an explicit `TextureColorSpace` for file and encoded-byte uploads, while `createFromRgba8()` still accepts an explicit `VkFormat` for procedural/data paths.

Base color textures now use `VK_FORMAT_R8G8B8A8_SRGB`. Vulkan sampler hardware converts those sRGB texels to linear values during sampling, so the fragment shaders continue to treat sampled base color as linear and do not manually apply a `pow()` decode.

Normal maps and metallic-roughness maps remain `VK_FORMAT_R8G8B8A8_UNORM` data textures. glTF `baseColorTexture` references are uploaded through the sRGB path, while `normalTexture` and `metallicRoughnessTexture` references are uploaded through the linear path for both external URI images and embedded/data-URI images. The renderer keeps separate internal glTF texture caches per material semantic so a reused image can be uploaded with the correct format for each slot.

Procedural fallbacks follow the same rule: checker/base-color fallback is sRGB, flat normal fallback is linear UNORM, and neutral metallic-roughness fallback is linear UNORM. Procedural environment cubemaps and the split-sum BRDF LUT remain linear/data resources.

Descriptor bindings, the bindless material texture set layout, shader resource layout, ObjectFrameData BDA path, main-pass GPU culling, indirect-count drawing, CSM, IBL bindings, BRDF LUT binding, Kulla-Conty-style compensation, render graph pass order, and swapchain synchronization are unchanged.

Future color and material work can add additional glTF texture semantics such as occlusion and emissive plus a broader color-management policy.

## Milestone 39: HDR Environment Loading and Tone Mapping

Milestone 39 adds an optional HDR environment path without changing material descriptors, bindless texture descriptors, ObjectFrameData, GPU culling, CSM, BRDF LUT, Kulla-Conty-style compensation, render graph pass order, or swapchain synchronization.

At startup, the renderer attempts to load `assets/environments/studio.hdr`. Large HDR assets are intentionally not committed; place a local Radiance `.hdr` file at that path to exercise the HDR path. The image is decoded with `stb_image` float loading (`stbi_loadf`) and converted to RGBA float data when needed.

Environment resources remain cubemap-based. The HDR equirectangular source is sampled on the CPU into a visible cubemap, then the existing educational approximate paths generate diffuse irradiance and prefiltered specular cubemaps from that HDR-derived source. The preferred upload format is `VK_FORMAT_R16G16B16A16_SFLOAT`, with `VK_FORMAT_R32G32B32A32_SFLOAT` as a fallback when needed. If the file is missing, decoding fails, or no supported sampled float cubemap format is available, the renderer keeps the procedural environment fallback.

In Milestone 39, exposure and tone mapping were introduced in the skybox and material fragment shaders. The renderer added a simple `ToneMappingSettings` value with `exposure = 1.0f` and `operatorType = 0`; operator `0` is Reinhard, and operator `1` is a compact ACES fitted approximation. Milestone 40 moves that tone-mapping work into the final composite pass.

Base color sRGB handling from Milestone 38 remains unchanged: base color textures use sRGB image formats, while normal and metallic-roughness textures remain linear UNORM data textures.

Limitations at the end of Milestone 39: equirectangular-to-cubemap conversion was approximate, there was no auto-exposure, no HDR swapchain, no ImGui control, and no production-quality environment prefiltering yet. Milestone 40 later added the basic bloom and post-process composite path.

Future work after Milestone 39 included better environment prefiltering, HDR skybox asset curation, ImGui controls, and a post-process path. The basic post-process path is now covered by Milestone 40, and basic auto exposure is covered by Milestone 41.

## Milestone 40: Post-Process Pass and Bloom

Milestone 40 moves the renderer to a minimal post-process path. Skybox and mesh lighting now render into a renderer-owned HDR offscreen scene color target using `VK_FORMAT_R16G16B16A16_SFLOAT`, while the existing main depth image remains attached to the main scene pass.

Tone mapping moved out of the skybox/material fragment shaders and into the final composite pass. Scene shaders now output linear HDR color. The composite pass samples scene color and blurred bloom, applies `sceneColor + bloom * intensity`, applies exposure, then applies the existing Reinhard or optional ACES fitted tone mapper before writing to the swapchain. The shader still does not add manual gamma correction; the current swapchain path follows the selected surface format behavior.

Bloom is intentionally simple and educational. A bright-pass extraction shader samples scene color, computes luminance, and keeps pixels above a hardcoded threshold. The extracted highlights are written to half-resolution bloom images, then a separable blur runs as one horizontal pass and one vertical pass using ping/pong bloom targets. There is no mip-chain bloom yet.

Post-process descriptors are separate from material, bindless material texture, skybox, and compute culling descriptors. The bloom extract/blur pipelines use a one-image sampled descriptor set, while the composite pipeline uses a separate descriptor set for scene color and blurred bloom.

The minimal `RenderGraph` now manually tracks `CSMShadowPass`, `MainHDRPass`, `BloomExtractPass`, `BloomBlurPass`, and `CompositePass`. It centralizes explicit transitions for scene color, bloom images, main depth, swapchain color attachment use, and swapchain presentation. This is still a manual post-process path, not a production render graph with automatic dependency inference, scheduling, aliasing, or transient resource allocation.

Swapchain resize recreates scene color and bloom images, resets their tracked layouts, and rebuilds the post-process descriptor sets that point at the resized image views. Shadow maps, environment resources, material descriptors, bindless texture sets, ObjectFrameData BDA data, GPU culling, indirect drawing, CSM, IBL, BRDF LUT, and Kulla-Conty-style compensation are unchanged.

Known limitations at the end of Milestone 40: bloom was simple and not mip-chain based yet, HDR swapchain output was not implemented, ImGui controls were not implemented, temporal effects were not implemented, and render graph scheduling/aliasing remained manual rather than production grade. Milestone 41 later adds the minimal average/log-average auto-exposure path.

## Milestone 41: Auto Exposure and Average Luminance

Milestone 41 adds a minimal automatic exposure path without changing material descriptors, bindless texture sets, ObjectFrameData BDA data, GPU culling, indirect drawing, CSM, IBL, BRDF LUT, Kulla-Conty-style compensation, or glTF loading.

The renderer computes log-average luminance from the HDR `sceneColor_` target in a compute pass. `src/shaders/luminance.comp` samples scene color, converts RGB to luminance with Rec. 709 weights, accumulates `log(max(luminance, 0.0001))` per workgroup, and writes partial sums plus sample counts into per-frame storage buffers. Those buffers are copied to CPU-readable readback buffers and reduced on the CPU after the existing frame fence completes.

Automatic exposure adapts toward `targetLuminance / max(avgLum, epsilon)`, clamped by the configured minimum and maximum exposure. The current exposure is smoothed with `1 - exp(-adaptationRate * deltaTime)` so brightness changes do not jump abruptly. The composite pass receives the current exposure through its existing push constants, applies exposure before Reinhard or ACES tone mapping, and continues to combine scene color with the existing simple bloom result.

The implementation intentionally uses the previous completed frame's luminance readback to avoid a CPU/GPU stall in the current frame. Manual exposure remains the fallback: if auto exposure is disabled or the luminance compute pipeline/resources fail to initialize, the renderer logs one warning, disables auto exposure, and uses `manualExposure`.

The minimal `RenderGraph` now tracks `CSMShadowPass`, `MainHDRPass`, `BloomExtractPass`, `BloomBlurPass`, `LuminancePass`, and `CompositePass`. `LuminancePass` reads `sceneColor_` and writes luminance partial data; it is still part of the manual graph rather than a production scheduler with aliasing or automatic dependency inference.

Known limitations after this milestone: there is no local exposure, no eye adaptation curve UI, no ImGui controls, no exposure debug visualization, no HDR swapchain output, no temporal AA, and bloom is still a simple half-resolution extract plus separable blur. Milestone 42 adds histogram-based exposure and percentile luminance clipping.

Milestone 42 later covers histogram-based auto exposure and percentile luminance clipping. Remaining future work after Milestone 41 includes local exposure, ImGui controls, exposure debug visualization, HDR swapchain output, better bloom quality, and temporal effects.

## Milestone 42: Histogram-Based Auto Exposure

Milestone 42 upgrades automatic exposure while keeping the existing HDR `sceneColor_`, post-process composite, and Milestone 41 log-average path. A new compute shader, `src/shaders/luminance_histogram.comp`, samples the HDR scene color target, computes Rec. 709 luminance, maps `log2(max(luminance, epsilon))` into 256 histogram bins between the configured minimum and maximum log luminance, and atomically increments a storage-buffer bin count.

The renderer allocates one device-local histogram buffer and one CPU-visible readback buffer per frame in flight. Each histogram buffer is a 256-entry `uint32_t` storage buffer with transfer source and destination usage. Before dispatch, the frame's histogram buffer is reset with `vkCmdFillBuffer`, then a Synchronization2 buffer barrier makes the transfer write visible to compute shader storage reads/writes. After `HistogramCompute`, another barrier makes compute shader writes visible to transfer, the histogram is copied into the frame's readback buffer, and a final barrier makes the copy visible to the host. The CPU reads only a previous completed frame after the existing fence wait, so the current frame is not stalled for exposure.

CPU exposure now supports three modes: manual, log-average luminance, and histogram percentile. Histogram mode is the default preferred mode. The CPU sums the readback histogram, finds the configured low/high percentile cut points, and computes a weighted average luminance from bin centers inside that clipped percentile range. Bin centers are converted back from log2 luminance with `exp2`. Percentile clipping keeps extreme dark or bright pixels from dominating exposure.

Exposure still targets `targetLuminance / max(luminance, epsilon)`, clamps between `minExposure` and `maxExposure`, and smooths with `currentExposure += (targetExposure - currentExposure) * (1 - exp(-adaptationRate * deltaTime))`. The composite pass remains the only tone-mapping location: it samples scene color and bloom, applies `currentToneMappingExposure()`, then applies Reinhard or ACES tone mapping.

The minimal render graph now documents `CSMShadowPass`, `MainHDRPass`, `BloomExtractPass`, `BloomBlurPass`, `LuminancePass`, `HistogramExposurePass`, and `CompositePass`. The implementation intentionally keeps `LuminancePass` active alongside `HistogramExposurePass` while auto exposure is enabled so log-average fallback data and once-per-second exposure logging remain available; only the selected exposure mode's result drives `currentToneMappingExposure()`.

Fallback behavior is conservative. Manual mode always returns `manualExposure`. Log-average mode uses the Milestone 41 luminance path when available and falls back to manual exposure otherwise. Histogram mode uses histogram percentile exposure when the histogram resources and pipeline are available, falls back to log-average exposure if needed, and falls back to manual exposure if no automatic path is available. If histogram resource, descriptor, or pipeline creation fails, the renderer logs a warning and keeps the log-average path when possible.

Runtime logging now prints once per second:

```text
Exposure:
  mode: manual / log-average / histogram
  average luminance: X
  histogram clipped luminance: Y
  exposure: Z
  low percentile: A
  high percentile: B
```

Timestamp/debug capture labels now include `HistogramExposurePass` and `HistogramCompute`, and timestamp results include a separate `HistogramExposure` range while keeping the existing `AutoExposure` / `Luminance` timing.

At the end of Milestone 42, histogram reduction still read back to the CPU, there was no GPU-only exposure chain yet, no local exposure, no eye adaptation curve UI, no ImGui controls, no HDR swapchain output, no temporal AA, and bloom remained the existing simple half-resolution extract plus separable blur. Later phases changed several of those items; the current status is summarized at the top of this README.

At that point, planned work included GPU-side histogram reduction, exposure debug visualization, ImGui runtime controls, local exposure, HDR swapchain output, improved bloom, and temporal effects.

## Milestone 43: ImGui Debug UI and Runtime Render Settings

Milestone 43 integrates standard Dear ImGui with the SDL3 platform backend and Vulkan renderer backend. The engine owns this through `src/ui/ImGuiLayer`, which creates the ImGui context, initializes the SDL3/Vulkan backends, owns a dedicated ImGui descriptor pool, starts and ends ImGui frames, renders draw data, and shuts the backend down without changing existing renderer descriptor layouts.

The Vulkan backend uses the Dynamic Rendering path. `ImGuiLayer` initializes `ImGui_ImplVulkan_InitInfo` with `UseDynamicRendering = true` and a `VkPipelineRenderingCreateInfoKHR` matching the swapchain color format, so no compatibility render pass is needed for the overlay.

`RenderGraph` now includes `ImGuiPass` after `CompositePass`. The composite pass writes the exposed and tone-mapped image to the swapchain, then `ImGuiPass` loads that same swapchain color attachment and draws the overlay before the graph transitions the image to `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR`. Existing HDR scene color, bloom, luminance, histogram exposure, CSM, IBL, BRDF LUT, bindless material textures, GPU culling, and indirect drawing paths remain unchanged.

The `VulkanEngine Debug` panel exposes runtime controls for tone mapping, manual/log-average/histogram exposure, target luminance, exposure clamps, adaptation rate, histogram percentile clipping, bloom enable/threshold/intensity, CSM lambda and shadow distance, texel snapping, cascade debug colors, and main/shadow GPU culling toggles. It also displays bindless material texture state, indirect-count fallback state, HDR environment versus procedural fallback state, and the current tone-mapping exposure value.

Profiling and runtime stats are visible in the same debug panel. GPU timing readouts include Shadow/CSM, Main, Bloom, Composite, AutoExposure, HistogramExposure, Skybox, and RenderObjects. Culling stats include total/visible/culled draw items, shadow draw items, visible shadow draw items, shadow batches, and main/shadow GPU culling state. Exposure stats include current exposure, log-average luminance, histogram clipped luminance, and the active exposure mode.

Console logging remains available and unchanged. The ImGui display is additive and intended only as a debug UI, not a full editor.

Resize handling keeps the ImGui context and SDL3 backend alive. When the swapchain image count or color format changes, the renderer recreates the ImGui Vulkan backend state and descriptor pool after the device is idle, then lets the backend rebuild font texture state as needed.

Known limitations after Milestone 43: docking/editor layout, asset browser, material inspector, persistent settings files, scene inspection UI, and GPU capture automation were still future work.

Milestone 44 later adds the render graph visualization and GPU timing history graphs, Milestone 45 adds persistent settings serialization, Milestone 46 adds scene hierarchy inspection, and Milestone 47 adds read-only material and selected texture inspection. Remaining future work includes editable material workflows, advanced render-target debug views, CSM cascade visualization panel, and GPU capture workflow improvements.

## Milestone 44: Render Graph Visualization and GPU Timing Graphs

`RenderGraph` now exposes read-only debug pass metadata through `debugPasses()`. The ImGui debug UI uses that data to visualize the manual render graph pass order in a `Render Graph` table. The table lists `CSMShadowPass`, `MainHDRPass`, `BloomExtractPass`, `BloomBlurPass`, `LuminancePass`, `HistogramExposurePass`, `CompositePass`, and `ImGuiPass` in order, and each row includes the pass type, graphics/compute execution class, major resource reads/writes, and transition notes.

GPU timestamp results are stored in short CPU-side history buffers. The Phase 1 `GPU Profiler` section shows current, recent average, and recent max timings for the total GPU frame, CSM shadows, shadow GPU culling cascades, main GPU culling, main HDR, bloom extract/blur, luminance, histogram exposure, composite, ImGui, skybox, and object drawing scopes. Each timing range gets a compact ImGui line plot using the completed frame slot's timestamp query results.

Culling and exposure stats remain visible in the debug UI. Main and shadow visible/culled draw item counts have short history plots, and exposure, log-average luminance, and histogram clipped luminance have small trend plots. The UI also has simple checkboxes for showing the render graph panel, GPU timing graphs, culling stats, and exposure graphs. These toggles are runtime-only and are not serialized.

This milestone is a debug visualization layer only. It does not add a render graph node editor, docking/editor layout, production render graph scheduler, automatic pass scheduling, resource aliasing, transient resource allocation, or persistent settings serialization, and it does not change descriptor layouts, render pass order, culling, indirect drawing, CSM, IBL, bloom, histogram exposure, or swapchain synchronization.

Known limitations after Milestone 44: docking/editor layout, persistent settings serialization, scene inspection UI, material inspector, asset browser, and render graph node editing were still future work. The render graph was still manual and not automatically scheduled, with no production dependency inference, aliasing, or transient resource allocation.

Milestone 45 later covers persistent settings serialization, Milestone 46 adds scene hierarchy inspection, and Milestone 47 adds read-only material and selected texture inspection. Remaining future work: render graph node view, GPU capture workflow panel, editable material inspector, advanced render-target debug views, CSM cascade visualization panel, in-engine profiler UI improvements, and render graph scheduling, aliasing, and transient resources.

## Milestone 45: Persistent Runtime Settings Serialization

Milestone 45 adds a small persistent runtime settings layer for the existing ImGui-edited render settings. `RuntimeSettings` groups tone mapping and exposure, bloom, CSM stability/debug options, renderer toggles, and debug panel visibility, then saves and loads them as human-readable JSON.

The renderer loads `config/runtime_settings.json` during startup. Missing files are not errors and fall back to defaults. Malformed files log a warning and also fall back to defaults. The ImGui debug UI now includes `Save Settings`, `Reload Settings`, and `Reset to Defaults` buttons plus the settings file path, last load/save status, and simple warning text for missing or malformed files.

`Save Settings` writes the current in-memory settings to `config/runtime_settings.json`, creating `config/` if necessary. `Reload Settings` reads the JSON again and applies runtime-safe fields. `Reset to Defaults` restores runtime-safe defaults in memory and does not overwrite the file unless `Save Settings` is pressed afterward. `config/runtime_settings.json` is git-ignored, while `config/runtime_settings.example.json` is tracked as the format reference.

Runtime-safe settings include exposure values, exposure mode, tone mapper, bloom enable/threshold/intensity, CSM lambda, shadow distance, texel snapping, cascade debug colors, GPU culling toggles when their resources were created at startup, and ImGui panel visibility. Startup-applied settings include CSM cascade count, bindless material texture heap enablement, and culling resources that were disabled before initialization; these are loaded before Vulkan resource creation and are shown as read-only or resource-dependent in the UI.

This milestone does not add a full editor, docking layout, scene hierarchy editing, asset browser, material inspector, per-project profiles, scene-specific settings, or hot-reload for settings requiring GPU resource recreation. It does not change descriptor layouts, ObjectFrameData BDA usage, GPU culling algorithms, indirect-count drawing, CSM resource layout, shadow bindings, IBL/BRDF LUT bindings, glTF loading, bloom extraction/blur, histogram exposure, render graph behavior, or swapchain synchronization.

At that point, planned work included profile presets, per-scene settings, editable material inspector, transform editing, object visibility toggles, render graph node view, GPU capture workflow panel, render-target debug views, CSM cascade visualization, and broader editor settings management.
