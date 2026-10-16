# Ray-Traced Sun Shadows

The main pass can ask the scene's acceleration structure whether the sun is
visible instead of reading the cascades or the virtual shadow map: one ray per
shaded pixel, built with `VK_KHR_ray_query`, against a top-level structure built
from the draw items that cast and rebuilt only on frames where one of them
changed. The ray is traced from the fragment shader, or optionally ahead of
the main pass, once for every pixel the depth prepass drew
([the screen-space mask](#the-screen-space-mask)), which is off by default
because a hard shadow gains nothing from it. The shadow maps stay
as the fallback, and the cascades keep rendering whenever something besides the
main pass reads them.

**Status: correct and off by default.** Everything that casts into the
cascades casts here: static geometry, opaque and alpha-tested, and the skinned
mesh, refitted whenever its pose moves. On a frame that traces, the cascades are not drawn
unless something else reads them (below).

`rayTracedShadows.enabled` in the runtime settings, or the *Ray-traced sun
shadows* checkbox under *Shadows*. It needs a device with ray query (an RTX
3080 Ti Laptop has it; the Intel UHD 770 beside it and MoltenVK do not) and the
bindless shading path; without either the setting is kept, ignored, and the
panel says which of the reasons applies.

## When it runs

Three conditions, each decided once and kept apart so a bug report can say
which one failed:

| Condition | Decided by | Off means |
| --- | --- | --- |
| The device has ray query | `VulkanDevice` from the optional-extension table (`rayQueryExtensionRequests`, [design_decisions.md](design_decisions.md#optional-device-extensions-are-a-table-not-an-if)); `--disable-ray-query` declines it | no acceleration-structure usage bits, no binding 16, the plain fragment shader |
| The pipelines can trace | `Renderer::rayQueryShadingAvailable`: the TLAS was created and bindless shading is active | the plain `simple_bindless.frag` |
| This frame traces | `frameRayTracedShadowsActive_`, latched in frame prep: the setting, the line above, the BLAS built, and a non-empty draw list | the FrameConstants flag word is zero and no build pass is declared |

The ray-query variant of the fragment shader is used whenever the pipelines can
trace, whether or not a frame does. The branch that traces is uniform on a
FrameConstants flag, so the setting flips at runtime without rebuilding a
pipeline, and with the flag clear the variant renders the same image as the
plain shader -- pinned below.

## Skipping the cascades

On a frame that traces, the main pass, the phase-two re-draw, the transparent
pass and the skinned draw all run the ray-query shader and never sample the
cascades. Five things still can, and any one of them keeps the cascades drawn
(`Renderer::cascadeShadowMapReadBeyondMainPass`):

- volumetric fog, whose injection shadows the froxels from the cascades;
- probe capture, on a frame that captures;
- the ray-traced comparison view, which samples them to compare against;
- the virtual shadow map's cascade-depth view;
- `rayTracedShadows.renderCascades`, which keeps them on regardless, for the
  A/B that separates what tracing costs from what skipping saves.

Otherwise `CSMShadowPass` and the GPU caster cull ahead of it are neither
declared nor recorded. The cascade cache's resident keys are committed only when
a cascade is actually drawn, so a skipped frame leaves them describing what the
image still holds: turning tracing off redraws exactly the cascades whose content
changed meanwhile, and none of the others. The CPU half -- cascade fitting, the
per-cascade caster lists and their keys -- still runs, because those keys are
what that comparison needs.

Skipping changes no pixel, because nothing that was drawn reads what was
skipped: with tracing on, the frame with the cascades skipped and the frame with
`renderCascades` forcing them match at 0/921600 pixels on `--scene default`,
`sunlit` and `stress`. The ImGui cascade preview keeps showing the last cascades
drawn while they are skipped, and the panel says so.

## Keeping the top-level structure

A frame that traces rebuilds the TLAS only when what it would build differs
from what the structure already holds (`RayTracingTopLevelTracker` in
`renderer/RayTracedShadows.h`). The comparison is on exactly what the build
reads: this frame's `VkAccelerationStructureInstanceKHR` records, byte for byte
and in order -- every instance's transform, custom index, mask, flags and BLAS
address -- staged in cached memory before anything is copied into the slot's
write-combined buffer. A moved object, an instance added or dropped (the caster
isolation, the skinned mesh hidden), a flag change and a reordered draw list all
differ. The last one matters because the custom index is the draw item's
`frameDataIndex`: two frames can hold the same objects in the same places and
still need a rebuild, because the index that reaches each one's material moved.
Opaque and masked items are ordered by batch key, which does not depend on the
camera, and only blended items -- which cast nothing -- are re-sorted by depth,
so a camera moving through a still scene keeps the structure.

The skinned mesh's instance record does not move with its pose, so the pose is
tracked on its own: `SkinnedMesh::poseHash`, the digest of the palette uploaded
this frame, which the cascade cache already keys on. The BLAS is pre-skinned and
refitted only when the pose differs from the one it was last refitted to, and a
refit always rebuilds the TLAS, which copied that BLAS's bounds into its
instance when it was built.

Three things go stale separately and are tracked separately:

| State | Brought up to date when |
| --- | --- |
| the TLAS | the instance records differ from the ones it was last built from, or the skinned BLAS refits |
| the skinned BLAS | the pose differs from the one it was last refitted to |
| a frame slot's instance and geometry-record buffers | the slot holds older records than the frame traces -- on a frame that keeps the TLAS too, because the rays read this slot's geometry records through the custom index |

A build counts only once it is recorded: frame prep plans and
`recordTopLevelBuild` commits, so a frame that throws between the two leaves the
next one building.

On a frame that keeps the structure, `RayTracingSceneBuild` is neither declared
nor recorded: no entry barrier, no build, no timestamp scope. The traced passes
still declare their reads, and with no write before them in the frame the graph
emits no barrier for the TLAS; the barrier recorded after the last real build
orders them (Synchronization, below).

`rayTracedShadows.rebuildEveryFrame` turns the detection off: every traced
frame rewrites its slot, refits and rebuilds, which is what every frame did
before it existed. It is the A side of the cost series below and the control the
detection is checked against.

## Cost

RTX 3080 Ti Laptop, clocks pinned 800/7001, `--scene sponza` with its animation
frozen, 1280x720, p10 over every frame of A/B/A/B series with the control
repeated (`docs/profiling.md` explains both flags).

**The first two series are re-measured against fixed cascades.** They were
first taken while the cascade fit clamped its near plane (see Verification
below). On Sponza the cascades then drew 144 of the 293 caster draws they
should have, and receivers outside their depth box returned lit before any
shadow fetch. Fixed cascades cost more to draw and to read
([design_decisions.md](design_decisions.md#the-cascades-depth-box-reaches-past-the-slice-toward-the-sun)),
so tracing saves more than first measured. The numbers below are the re-runs,
with the first measurement given for each.

**Tracing against the cascades.** Control drift 0.46%, 20552 and 26099 samples:

```
python tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --every-frame --freeze-animation --a-set rayTracedShadows.enabled=false --b-set rayTracedShadows.enabled=true --args --scene sponza
```

| Pass | cascades | traced | Delta |
| --- | --- | --- | --- |
| `Frame total`, p10 | 6.020 | 4.391 | −1.629, not like for like (below) |
| `Frame total`, p5 / median | 5.595 / 6.237 | 4.364 / 4.997 | **−1.231 / −1.240 (about −20%)** |
| `MainHDRPass` | 4.588 | 3.410 | −1.178 (−25.7%) |

The p10 frame-total delta is not the number. The fixed cascades run this scene
at about 160 fps, the line below which fewer than 10% of frames escape another
process's time slice
([profiling.md](profiling.md#two-mainhdrpass-modes-on-identical-work-the-gpu-is-shared)).
So the cascades' p10 is a sliced frame, while tracing's, at about 200 fps, is a
clean one. Like for like -- p5, clean on both sides, or the median, sliced on
both -- tracing saves 1.23 to 1.24 ms. `MainHDRPass` p10 is clean on both sides
and agrees. The first measurement read −0.691 ms (5.316 to 4.625), with the
traced side then also rebuilding the TLAS on every frame. It no longer does on
a still frame (the last series below), and `RayTracingSceneBuild` is absent
from every sampled frame of B here.

The camera is still, so the cascade cache hits on every frame of A and
`CSMShadowPass` draws nothing on either side (its scope still opens, around
nothing, as on any cache hit). What this row compares is therefore *reading* the
shadow: one ray against the cascade lookup and its filtering, which
`docs/profiling.md` found to be more than half of `MainHDRPass` on this scene.

**Skipping the cascades when they would redraw.** Both sides trace. A keeps the
cascades (`renderCascades`) with the cascade cache off, so all four redraw every
frame, as they do on any frame the camera moves; B skips them. Control drift
0.42%, 23795 and 25991 samples. A first run, started after 25 minutes of
continuous load, was refused at 2.0% and repeated after the card had cooled:

```
python tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --every-frame --freeze-animation --a-set rayTracedShadows.enabled=true --a-set rayTracedShadows.renderCascades=true --a-set csm.enableCascadeCache=false --b-set rayTracedShadows.enabled=true --b-set csm.enableCascadeCache=false --args --scene sponza
```

| Pass | traced + cascades | traced | Delta |
| --- | --- | --- | --- |
| `Frame total` | 5.198 | 4.410 | **−0.788 (−15.2%)** |
| `CSMShadowPass` | 0.829 | 0.000 | −0.829 |
| `ShadowGpuCulling` | 0.027 | — | A only |
| `MainHDRPass` | 3.371 | 3.425 | +0.054 (+1.6%) |

Both sides' p10 are clean frames, and the medians agree (−0.807). `MainHDRPass`
moved about 0.05 ms the wrong way, as it did in the first measurement (+0.051),
although both sides run the same shader. Nothing in the change explains it, so
`Frame total` is the number: skipping saves about 0.8 ms on every frame whose
cascades would have redrawn. The first measurement read −0.513 ms, with a
`CSMShadowPass` of 0.564 for the clamped cascades. The two series were measured
separately and do not add exactly. Together they put tracing on a moving camera
at roughly 2 ms cheaper than the cascades on this scene, against roughly 1.2 ms
when the cascades were missing casters.

A first attempt at this series moved the camera with `--camera-orbit` instead.
The orbit carried it outside the building, where the frame costs under a third
as much, and the control drifted 2.2%; holding the view and turning the cache off
measures the same redraw without changing what is on screen.

**Keeping the TLAS on a still frame.** Both sides trace, on the same binary with
one setting flipped -- not a comparison of two builds. A sets
`rayTracedShadows.rebuildEveryFrame`, which rebuilds the TLAS on every traced
frame as every frame did before the change detection above; B leaves it off and
keeps the structure. Control drift 0.09%, 25099 and 27993 samples:

```
python tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --every-frame --freeze-animation --a-set rayTracedShadows.enabled=true --a-set rayTracedShadows.rebuildEveryFrame=true --b-set rayTracedShadows.enabled=true --a-label rebuild --b-label keep --args --scene sponza
```

| Pass | rebuild every frame | keep | Delta |
| --- | --- | --- | --- |
| `Frame total` | 4.640 | 4.417 | **−0.223 (−4.8%)** |
| `MainHDRPass` | 3.421 | 3.410 | −0.011 (−0.3%) |
| `RayTracingSceneBuild` | 0.233 | — | on 0 of 27993 frames of B |

The frame total moves by what the build cost, and the build's scope is absent
from every sampled frame of B. Nothing that casts moves in Sponza -- its lights
animate, but a light is not an instance -- so with or without
`--freeze-animation` every frame after the first keeps the structure. A scene
where something casting moves pays the A column on every frame; the skinned
demo mesh is enough, so the default, sunlit and stress scenes rebuild every
frame unless their animation is frozen.

**The screen-space mask, against tracing inline.** Both sides trace and keep
the TLAS, on the same binary with `screenSpaceMask` the only change. Control
drift 0.74%, 25731 and 22793 samples, after a throwaway warm-up run; the pin
held at 802/7001 with no throttle reason:

```
python tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --every-frame --freeze-animation --a-set rayTracedShadows.enabled=true --a-set rayTracedShadows.screenSpaceMask=false --b-set rayTracedShadows.enabled=true --b-set rayTracedShadows.screenSpaceMask=true --a-label inline --b-label mask --args --scene sponza
```

| Pass | inline | mask | Delta |
| --- | --- | --- | --- |
| `Frame total` | 4.446 | 5.089 | **+0.643 (+14.5%)** |
| `RayTracedShadowMask` | — | 0.554 | mask only |
| `MainHDRPass` | 3.444 | 3.418 | inside the noise floor: not measured |

The mask answers 99% of Sponza's lit fragments, so the main pass stopped
tracing almost entirely -- and got no faster. Its rays were not what it was
waiting on. The likely reading, not isolated here: [profiling.md](profiling.md)
found much of this pass's cost on Sponza to be per-triangle work and quad
overshading, and the rays ran alongside it. Moved into a pass of their own they cost
0.554 ms, plus roughly 0.09 ms more of frame total for the depth transitions and
the serialisation around a compute pass. For a hard shadow the mask is pure
cost, which is why it is off by default; it is there for a soft one.

## What casts

The cascades' rules, applied to the same draw-item list (`docs/transparency.md`):

- **Opaque** draw items become instances with `FORCE_OPAQUE`: the ray commits on
  them without asking.
- **Mask** items become `FORCE_NO_OPAQUE` instances, and the shader runs the
  cascades' masked-shadow test on each candidate triangle: the same three
  `ObjectFrameData` fields (`textureIndices.x`, `baseColorFactor.a`,
  `materialParams.w`) and the same comparison as `shadow_masked.frag`. The one
  difference is the texture level: a ray has no footprint, so it samples level 0
  where the raster pass takes the mip its derivatives select.
- **Blend** items cast nothing. They still *receive*: the transparent pass runs
  the same fragment shader.
- `csm.debugOnlyShadowCasterObject` narrows the instances exactly as it narrows
  the cascades' casters, which is what makes it a negative control for this
  path too.
- **The skinned mesh** casts under the cascades' own condition for it
  (`skinnedCasterActive`: shown, with a pose and bounds), as an opaque instance
  -- its material has no alpha test, like its cascade caster -- whose custom
  index is the reserved `kSkinnedObjectFrameSlot`. Like its cascade caster, it is
  not narrowed by the caster isolation, which selects render objects.

An instance's custom index is its draw item's `frameDataIndex`. That one number
reaches both the draw item's `ObjectFrameData` (its material) and a per-frame
`RayTracingGeometryRecord` (where its vertices and first index are), so the
alpha test needs no table of its own beyond the geometry addresses.

Instance facing culling is disabled: a shadow comes from whichever side of a
caster faces the sun.

## Resources

| Resource | Lifetime | Created | Destroyed |
| --- | --- | --- | --- |
| BLAS, one per (mesh, primitive) at LOD 0, each with its own storage buffer | renderer | `buildRayTracingBottomLevel`, after `createScene` | `RayTracingScene` destructor |
| TLAS and storage buffer | renderer, **one** structure | `createRayTracingScene`, before the material set layout | same |
| Scratch | renderer, shared by every build | with the TLAS; grown once for the BLAS batch | same |
| Instance buffer (`VkAccelerationStructureInstanceKHR` x 8192) | per frame slot, host-visible | with the TLAS | same |
| Geometry records (`RayTracingGeometryRecord` x 8192) | per frame slot, host-visible | with the TLAS | same |
| Skinned positions (3 floats per vertex) | renderer, one buffer | `createRayTracingSkinnedCaster`, after the skinned mesh | `RayTracingScene::destroy` |
| Skinned BLAS (`ALLOW_UPDATE`) | renderer, one structure | storage with the positions; built by the first frame that traces with the mesh shown, refitted by every later one whose pose moved | same |
| Pre-skin pipeline (`skin_positions.comp`, push constants only) | renderer | with the positions | same |
| Set 0 binding 16, `ACCELERATION_STRUCTURE_KHR` | written once into the shared bindless set | `createMaterialDescriptorSet` | with the pool |
| Shadow mask image, `R32G32_UINT` at the render-resolution allocation extent | renderer, one image, written and read within a frame; its layout is tracked across frames | `RayTracedShadowMask::createResources`, from `recreatePostProcessResources`, whenever the TLAS exists | every resize, and the destructor |
| Its nearest sampler and the pass's descriptor set (depth, TLAS, mask) | same | same | same |
| Set 0 binding 17, `COMBINED_IMAGE_SAMPLER` (`usampler2D`) | the shared bindless set, whenever the TLAS exists | `createMaterialDescriptorSet`; rewritten by `refreshMaterialRayTracedShadowMaskDescriptors` on resize | with the pool |
| Mask pipeline (`ray_traced_shadow_mask.comp`; set 0 its own, set 1 the bindless heap) | renderer | `createRayTracedShadowMaskPipeline`, on every pipeline rebuild; its set layout once | the pipeline store |

The mesh set is fixed after construction -- every scene preset draws the
built-in cube and sphere, and the sample scene is imported at startup only -- so
nothing rebuilds a BLAS. The BLAS build is one submission waited on with a
fence, at startup.

The skinned caster follows the TLAS: one positions buffer and one BLAS,
rewritten in place. A refit keeps the hierarchy built for the first pose and
moves its boxes, which is what makes a per-frame update cheap and what makes its
quality drift with how far the mesh bends from that pose; the demo rig bends
about 22 degrees a joint. The pre-skin is the same linear blend
`simple_skinned.vert` runs, from the same half of the same palette, so a ray
finds the pose the main pass draws.

There is one TLAS rather than one per frame slot because binding 16 lives in the
environment set every material shares, and a set in use by a frame in flight
cannot be rewritten. A per-slot copy would need a descriptor per slot. The
cascaded shadow map makes the same trade: one image, redrawn every frame and
read by the frame before.

Measured at startup on the RTX 3080 Ti Laptop: the TLAS for 8192 instances is
2413 KiB with 877 KiB of scratch; the default scene's two meshes build into 159
KiB of BLAS, Sponza's 105 primitives (264487 triangles) into 18350 KiB, in
6.5 ms.

## Synchronization

| Edge | Ordered by |
| --- | --- |
| CPU writes this slot's instances -> this frame's TLAS build | the slot's fence, then the submit's host-write guarantee |
| earlier frames' ray queries (fragment and compute, AS read) and builds (TLAS and scratch writes) -> this frame's build | a global barrier `RayTracingScene::recordTopLevelBuild` records first |
| this frame's build -> its ray queries | the render graph: `RayTracingSceneBuild` declares `AccelerationStructureBuild` on the TLAS storage buffer; `RayTracedShadowMask` declares `AccelerationStructureComputeRead`, whose scope names the compute and the fragment stage; `MainHDRPass`, `MainHDRPhase2` and `TransparentPass` declare `AccelerationStructureRead` |
| an earlier frame's build -> the ray queries of a frame that keeps the TLAS | the graph's barrier on the frame that built it, whose second scope is the stages it names in every later command: the fragment stage, and the compute stage when the mask traced that frame. The frame the mask turns on rebuilds (`rayTracedShadowMaskNeedsFreshBuild`), so a keeping frame never reads in a stage its build's barrier did not name. The keeping frame declares the reads alone, which as the structure's first use in the frame emit nothing, and the next build's entry barrier waits for them |
| depth prepass writes -> the mask's depth reads -> the main pass's depth test | the graph: `DepthPrepass` writes main depth, `RayTracedShadowMask` reads it (`DEPTH_READ_ONLY_OPTIMAL`), `MainHDRPass` takes it back as an attachment |
| the mask's storage writes -> the main shader's reads | the graph: `GENERAL` to `SHADER_READ_ONLY_OPTIMAL`, compute to fragment |
| the previous frame's mask reads -> this frame's mask writes | the graph's first-use barrier, whose source is the layout the image ended the previous frame in |
| mesh upload (transfer) -> BLAS build | a barrier at the top of the startup build, whose first scope reaches back over the earlier upload submissions on the same queue |
| earlier frames' refit reads of the skinned positions -> this frame's pre-skin writes | the same entry barrier, which names the compute stage as a destination |
| pre-skin writes -> refit reads | a barrier inside `RayTracingSkinnedCaster::record` |
| refit -> TLAS build, and -> the rays | a global barrier after the refit, naming the compute and fragment stages |
| startup BLAS builds -> the rays | the entry barrier, which names the compute and fragment stages as destinations |

The second edge is explicit because the graph rebuilds buffer state every frame
and so has nothing to order a frame's first write against -- the same situation
as the probe shading-parameter buffer in [render_graph.md](render_graph.md).

The last two exist because the graph's barrier names one buffer, the TLAS's,
while a ray reads every BLAS the TLAS references. Bottom-level writes are made
visible to the fragment stage by global barriers instead -- the entry barrier for
the structures built at startup, the post-refit barrier for the one rewritten
inside the frame. The first version of this path had only the graph's barrier,
which left the startup BLASes formally unordered against the rays.

Adding the third edge found a hole in the graph. Its write mask had no
acceleration-structure write bit, so a build followed by a trace read as a read
after a read and emitted nothing. The bit is in the mask now, and
`tests/test_render_graph.cpp` pins that a build-then-trace pair needs a barrier.

**Synchronization validation does not see these edges.** Each was removed in
turn, rebuilt, and run under `--sync-validation` on the RTX: zero hazards
either way, while the layer's own self-test fired in the same build. So a clean
sync-validation run is not evidence for this path, and the barriers are pinned
by what was recorded instead. An `api_dump` of a frame shows the entry barrier
(`FRAGMENT_SHADER | ACCELERATION_STRUCTURE_BUILD` to `ACCELERATION_STRUCTURE_BUILD`,
AS read and write) immediately before the build, and the graph's buffer barrier
on `SceneTopLevelAccelerationStructureStorage` (build to `FRAGMENT_SHADER`,
`ACCELERATION_STRUCTURE_READ`) at the start of `MainHDRPass`, before its first
traced draw. A frame that keeps the structure shows none of it -- no
`RayTracingSceneBuild` label, no build, no acceleration-structure barrier and
no barrier on the TLAS buffer -- while `MainHDRPass`, `MainHDRPhase2` and the
skinned draw run as before: four frames of `--scene default
--freeze-animation` record one build, then three frames without one.

**The mask found a second hole, and this one is in how the graph orders reads.**
A read after a read gets no barrier, whatever stage it is in, so only the first
reader after a write is ordered against it. Until the mask, every ray was in the
fragment stage and that never mattered. With the mask first, the graph's barrier
after the build named the compute stage alone -- an `api_dump` of the first
version shows `ACCELERATION_STRUCTURE_BUILD` to `COMPUTE_SHADER` in front of the
mask and nothing in front of `MainHDRPass` -- which left the main pass's own rays
unordered against the build. The mask's access names the fragment stage as well,
since on a frame it traces it is always that first reader, and the same dump now
shows `ACCELERATION_STRUCTURE_BUILD` to `FRAGMENT_SHADER | COMPUTE_SHADER`.
`tests/test_render_graph.cpp` pins the scope. The general fix, a graph that
tracks the stages each write has been made visible to, is not made here; see
[render_graph.md](render_graph.md).

The same dump shows the edges the mask adds and nothing else changed: the entry
barrier's first scope and both destinations now include `COMPUTE_SHADER`, the
mask image goes `UNDEFINED` to `GENERAL` on its first frame and
`SHADER_READ_ONLY_OPTIMAL` to `GENERAL` on every later one, and four frames of
`--scene sunlit --freeze-animation` record one build and then three frames that
dispatch the mask with no acceleration-structure barrier at all.

Everything runs on the graphics queue. The asynchronous cluster passes never
read the TLAS, so no queue-family ownership is involved.

## Binary contracts

- **FrameConstants** gains two vec4s at the end: `rayTracedShadow` (geometry
  record address as a uvec2, then the `kRtShadowFlag*` word) at offset 624 and
  `rayTracedShadowParams` (normal offset, offset per unit of distance, ray
  length) at 640. Pinned by `static_assert` in `RendererInternal.h`.
- **The flag word** is `kRtShadowFlagEnabled`, `kRtShadowFlagDebugVisibility`,
  `kRtShadowFlagDebugCompareCascades`, `kRtShadowFlagScreenMask` and
  `kRtShadowFlagDebugMaskSource`, mirrored in `object_frame_data.glsl` and
  pinned by `tools/check_shader_constants.py`.
- **The shadow mask** is set 0 binding 17 in the ray-query variant, a
  `usampler2D` read with `texelFetch`: x is the visibility's float bits, y the
  depth bits it was traced at. `kRtMaskNotTraced` (-1) marks a texel with no
  surface to trace from, and `kRtMaskBackFacingBias` (+2) one traced from a
  surface facing away from the sun; both live in `ray_traced_shadows.glsl`, which
  both shaders include and nothing on the CPU reads. The dispatch's workgroup,
  `kRtMaskLocalSize`, is paired with C++ by the checker.
- **The mask dispatch** pushes two addresses, the sub-rect, its reciprocal and
  the inverse jittered view-projection (`RayTracedShadowMaskPushConstants`, 96
  bytes, offsets pinned by `static_assert` and by the shader interface golden).
- **The trace itself** lives once, in `ray_traced_shadow_trace.glsl`, and takes
  the frame-constants and object-data addresses as arguments, so the mask and the
  inline fallback cannot drift apart.
- **The vertex layout** the alpha test reads by address, `kRtVertexStrideFloats`
  and `kRtVertexUvOffsetFloats`, is pinned against `renderer::Vertex` by
  `static_assert` and against the shader by the same checker.
- **The pre-skin dispatch** pushes four addresses and a count
  (`RayTracingPreSkinPushConstants`, offsets pinned by `static_assert` and by the
  shader interface golden). Its constants -- the source vertex stride, the
  three-float position stride, the 64-thread workgroup -- live in
  `ray_traced_shadows.glsl`, shared with the fragment shader's alpha test.
- **The ray-query variant** is `simple_bindless.frag` compiled again with
  `VE_RAY_QUERY` (`SHADER_VARIANTS` in `CMakeLists.txt`). It also declares the
  push block's leading `objectFrameData` address, which the plain variant does
  not read. Its interface is in `tests/golden/shader_interface.txt`.
- **The instance transform** is a 3x4 row-major matrix where glm is
  column-major; `rayTracingInstanceTransform` converts it, and a test moves
  points through both.

## The shadow ray

The origin moves off the surface along its interpolated normal, flipped to the
side facing the sun so a two-sided caster seen from behind does not start inside
itself, by `normalOffset` plus a term proportional to the distance from the
camera. That second term is what covers LOD: rays see every mesh at level 0
while the raster pass drew whichever level the cull chose, and under
screen-space error selection that level is at most `lod.maxErrorPixels` off level
0 on screen. The renderer adds that budget to `pixelFootprintOffset` and turns
the sum into world units per unit of distance
(`rayTracedShadowOffsetPerDistance`). In the fragment shader, where nothing
faces the sun the ray is skipped, because the term it produces multiplies zero;
the mask traces those pixels anyway (below).

The ray is as long as the diagonal of the world bounds of everything drawn this
frame (`rayTracedShadowRayLength`). The first version stopped it at the camera's
far plane, which is the wrong bound: the far plane limits which receivers are
drawn, not how far away their occluders are, and the TLAS holds every caster,
in view or not. A tall caster further than the far plane along the sun direction
from a visible receiver would have been missed.

The shadow is hard: one ray toward a light with no extent.

## The screen-space mask

`RayTracedShadowMask` is a compute pass between `DepthPrepass` and `MainHDRPass`
that traces one ray for every pixel the prepass drew and writes the answer to an
image the main pass reads. It is what a soft shadow needs: a filter has to see
the visibility before anything is shaded with it. For the hard shadow traced
today it only costs (below), so `rayTracedShadows.screenSpaceMask` is off by
default.

It runs when the frame traces, the prepass ran, the mask is available -- its
pipeline exists, main depth can be sampled and is 32-bit float, and the
bindless heap is there for the cutout test -- and
`rayTracedShadows.screenSpaceMask` is on (`rayTracedShadowMaskActive`).
Otherwise every fragment traces inline, as before the mask.

**What it traces from.** Each pixel's position is the prepass depth unprojected
through the inverse of the jittered view-projection the prepass rasterised with,
so it is the surface point the main pass shades at that pixel. Its normal comes
from depth too, because the prepass has no fragment shader and no colour target
by design ([design_decisions.md](design_decisions.md)): along each axis the
position difference towards whichever neighbour is closer in depth, so a
silhouette takes its tangent from its own surface, crossed and turned towards the
camera. The ray is then exactly the inline one -- the same function, the same
offsets -- from that position and normal.

**Which fragments may read it.** A texel holds the visibility and the depth bits
it was traced at. The main pass uses it only when its fragment's depth is
exactly those bits. The prepass drew opaque and cutout geometry, and the main
pass tests `LESS_OR_EQUAL` against it with the same positions, so a fragment of
a surface the prepass drew has exactly the depth it wrote, and everything else
-- the skinned mesh, phase-two rescues, blended surfaces, cutouts when the heap
is missing -- is strictly in front of it and traces inline. No draw needs a flag.

**Surfaces facing away from the sun.** The mask traces them too. The first
version left them untraced, reasoning that they are dark, and on Sponza that
sent 35.9% of the lit fragments to the inline fallback: a normal map tilts the
shading normal of a wall just past the sun's horizon back towards it, so the
main pass still needs the term. A ray from such a surface starts on its far side
and usually stops on the first triangle it tests, its own. But one case is not
the mask's to answer: the terminator of a smooth-shaded surface, where the face
faces away while the vertex normal still faces the sun. There the inline ray,
offset along the vertex normal, is the one that matches the shading, and the
mask's answer differed on a band as bright as delta 159 on `--scene sunlit`. So
those texels are marked, and a fragment uses one only when its own vertex
normal faces away as well.

**How much it covers**, by the mask-source debug view (Debug, frame 30 of a
fixed-step run, 1280x720, exposure pinned):

| Scene | Fragments the mask answers | Lit fragments still traced inline |
| --- | --- | --- |
| default | 88.9% of the frame | 3.78% -- the skinned tube and the two blended panels |
| stress | 67.5% (the rest is sky) | 0.20% |
| Sponza | 99.0% | 0.10% |

## Debug views and settings

| Setting | Default | |
| --- | --- | --- |
| `rayTracedShadows.enabled` | false | |
| `rayTracedShadows.renderCascades` | false | keep drawing the cascades while tracing |
| `rayTracedShadows.rebuildEveryFrame` | false | rebuild the TLAS and refit the skinned BLAS on every traced frame, as before change detection; the A/B control and the image control for it |
| `rayTracedShadows.screenSpaceMask` | false | trace the mask ahead of the main pass; off, every fragment traces inline. Off by default on measured cost |
| `rayTracedShadows.normalOffset` | 0.02 | world units, clamped to [0, 1] |
| `rayTracedShadows.pixelFootprintOffset` | 1.0 | pixels, clamped to [0, 16] |
| `rayTracedShadows.debugVisibility` | false | the traced term alone, white lit |
| `rayTracedShadows.debugCompareCascades` | false | red where only the ray finds an occluder, blue where only the cascades do; wins over the visibility view |
| `rayTracedShadows.debugMaskSource` | false | green where the mask answered and an inline ray from the fragment agrees, magenta where it does not, blue where the fragment traces inline, grey where it faces away and traces nothing; wins over both views above |

The CI sweep renders it enabled (cascades skipped), enabled with both debug
views, `renderCascades` and `rebuildEveryFrame`, enabled on a still scene
(`--freeze-animation`, which keeps the TLAS), enabled with the mask and its
source view on, and enabled with `--disable-ray-query` (`config/ci/`). Mesa's lavapipe exposes all three extensions, so the first two
run the traced path in CI, under validation.

## Verification

RTX 3080 Ti Laptop, Debug build with validation, `--deterministic`, frame 30 at
1280x720. Every run below had zero validation errors and a clean render graph
backstop.

| Check | Result |
| --- | --- |
| ray query off, against the build before this change | 0/921600 pixels differ |
| enabled with `--disable-ray-query`, against disabled | 0/921600 pixels differ |
| enabled, no casters but the skinned mesh (`csm.debugOnlyShadowCasterObject` = 99999), exposure pinned, against the cascades with the same isolation | 2981 pixels differ, all in the skinned mesh's shadow; against the virtual shadow map with the same isolation, 1572 (the cascades: 4078) |
| enabled, against disabled, default scene | 7026 pixels differ before the skinned caster, max channel delta 10 |
| enabled, keeping the TLAS, against `rebuildEveryFrame` and against the build before change detection: `--scene default`, `sunlit` and `stress`, animated and frozen, and Sponza (Release) animated and frozen | 0/921600 pixels differ in every one; the frozen scenes and both Sponza runs keep the structure on 39 of 40 frames, the animated scenes rebuild all 40 |
| the same animated scenes with the detection broken to build and refit only once | 3903 (default), 10847 (sunlit) and 206 (stress) pixels differ: a stale structure shows |

The mask against the inline path it replaces, each traced with the other's
setting the only change (Debug with validation and `--sync-validation`, exposure
pinned; Sponza from `build/debug`, which has the asset). Zero validation errors,
zero sync hazards and a clean backstop in all twelve runs:

| Scene | Pixels differing | by more than 2 | by more than 8 | Max delta |
| --- | --- | --- | --- | --- |
| default | 47 | 47 | 25 | 11 |
| sunlit | 1084 | 41 | 37 | 52 |
| stress | 171 | 129 | 66 | 21 |
| Sponza | 43 | 12 | 0 | 6 |

Sunlit's small differences are bloom spreading its 41 real ones. Where the real
ones sit is what the reconstructed normal predicts: silhouettes, contact points
where a sphere meets the floor, the one-pixel bars of the cutout lattice -- each
a depth discontinuity the normal is taken across -- and the last pixels of a
terminator.

Unpinned, the no-caster comparison also differed by single steps scattered over
the frame: auto-exposure follows total scene luminance, and any change in shadow
area moves it. Pinning exposure removed every one of them.

In the skinned-only comparison all three regions that differ -- the tube's own
upper side, the tip of its shadow on the floor, the top row of the lattice --
read the same under the ray and the virtual shadow map and different under the
cascades (the floor region: 114.64 and 114.69 mean luminance against 115.73), so
the residue is the cascades again. The capture is frame 30 of a fixed-step run,
so the pose has moved every frame and the BLAS has been refitted 29 times. An
`api_dump` of the first four frames shows the order: a full build on the first
traced frame and `MODE_UPDATE` after it, each preceded by the pre-skin dispatch
and its barrier and followed by the TLAS build.

**The cascades were the outlier on the two scenes with a strong sun, and they
were wrong.** On `--scene sunlit` the traced image showed two shadows that the
cascades did not draw and the virtual shadow map did: a long shadow across the
right of the yard and a second sphere shadow. On Sponza, against the virtual
shadow map (Debug build, auto exposure):

| Path | Pixels differing from VSM | Mean absolute luminance difference |
| --- | --- | --- |
| ray-traced | 15062 (1.6%), max delta 7 | 0.006 |
| cascades | 374721 (40.7%), max delta 11 | 0.397 |

Two independent methods agreed and the cascades disagreed with both. The cause
was the cascade fit's near plane, clamped to one unit sunward of each slice's
centre. It clipped and culled every caster beyond that, and it predated this
path; see [design_decisions.md](design_decisions.md#the-cascades-depth-box-reaches-past-the-slice-toward-the-sun)
for how it was isolated. With it fixed, a Release build with exposure pinned
gives these differences against the virtual shadow map:

| Scene | traced | cascades before | cascades after |
| --- | --- | --- | --- |
| sunlit | 5.6% | 18.2% | 2.2% |
| Sponza | 0.074% | 29.5% | 0.045% |

Most of the skinned-only residue above was the same clamp. Fixed, the cascades
under that isolation differ from the traced image on 1000 pixels rather than
2981, and from the virtual shadow map on 2172 rather than 4078.

## Limitations

- **The skinned BLAS is refitted, never rebuilt** after its first build, so its
  hierarchy quality depends on the pose it was first built in. Fine for the demo
  rig; a mesh that deforms far from its first pose would want a periodic rebuild.
- **Hard shadows only.** No sun angular radius and no denoising.
- **The cascades come back with fog or probe capture,** which read them, so on
  those frames tracing adds its cost without removing theirs.
- **Virtual shadow map pages still render** when VSM is enabled alongside
  tracing, though nothing samples them.
- **One moving caster rebuilds the whole TLAS.** Keeping the structure is all
  or nothing, so a scene with anything casting in motion -- the skinned demo
  mesh is enough -- pays the full build (0.233 ms on Sponza at the 800 MHz pin)
  on every frame. Updating only the moved instances would need `ALLOW_UPDATE` on
  the TLAS and a fixed instance count.
- **The skinned pose is compared by digest.** `SkinnedMesh::poseHash` is 64 bits
  over the palette and the model matrix, the key the cascade cache already uses;
  a collision would keep a stale skinned BLAS until the pose moved again.
- **Shadow terminator on coarse spheres.** The comparison view shows a red rim
  where the ray crosses the faceted geometry just past the smooth terminator.
  It sits where `N.L` is near zero, so it is barely visible in the shaded image
  on these scenes.
- **Alpha tests at texture level 0,** where the cascades sample a mip.
- **The mask's normal is the face normal from depth.** Where depth is
  discontinuous within a pixel's neighbourhood -- silhouettes, contact points,
  one-pixel cutout bars -- it is approximate, and the mask and the inline ray
  can disagree there (the table above). The prepass could write the shading
  normal instead, at the price of the fragment work it exists to avoid.
- **The mask needs a 32-bit float depth format,** the depth prepass and the
  bindless heap; without any of them every fragment traces inline.
- **Synchronization validation is blind to it,** as above.
