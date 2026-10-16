# Ray-Traced Sun Shadows

The main pass can ask the scene's acceleration structure whether the sun is
visible instead of reading the cascades or the virtual shadow map: one ray per
shaded pixel, built with `VK_KHR_ray_query` from the fragment shader, against a
top-level structure rebuilt every frame from the draw items that cast. The
shadow maps stay as the fallback, and the cascades keep rendering whenever
something besides the main pass reads them.

**Status: correct and off by default.** Everything that casts into the
cascades casts here: static geometry, opaque and alpha-tested, and the skinned
mesh, refitted every frame. On a frame that traces, the cascades are not drawn
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

## Cost

RTX 3080 Ti Laptop, clocks pinned 800/7001, `--scene sponza` with its animation
frozen, 1280x720, p10 over every frame of A/B/A/B series with the control
repeated (`docs/profiling.md` explains both flags).

**Tracing against the cascades.** Control drift 0.04%, 21744 and 24795 samples:

```
python tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --every-frame --freeze-animation --a-set rayTracedShadows.enabled=false --b-set rayTracedShadows.enabled=true --args --scene sponza
```

| Pass | cascades | traced | Delta |
| --- | --- | --- | --- |
| `Frame total` | 5.316 | 4.625 | **−0.691 (−13.0%)** |
| `MainHDRPass` | 4.290 | 3.401 | −0.889 (−20.7%) |
| `RayTracingSceneBuild` | — | 0.233 | traced only |

The camera is still, so the cascade cache hits on every frame of A and
`CSMShadowPass` draws nothing on either side (its scope still opens, around
nothing, as on any cache hit). What this row compares is therefore *reading* the
shadow: one ray against the cascade lookup and its filtering, which
`docs/profiling.md` found to be more than half of `MainHDRPass` on this scene.
The ray is 0.889 ms cheaper there and still 0.691 ms cheaper after paying for
the TLAS.

**Skipping the cascades when they would redraw.** Both sides trace. A keeps the
cascades (`renderCascades`) with the cascade cache off, so all four redraw every
frame, as they do on any frame the camera moves; B skips them. Control drift
0.33%, 22126 and 24481 samples:

```
python tools/dev/measure_gpu.py ab --repeat 2 --duration 75 --every-frame --freeze-animation --a-set rayTracedShadows.enabled=true --a-set rayTracedShadows.renderCascades=true --a-set csm.enableCascadeCache=false --b-set rayTracedShadows.enabled=true --b-set csm.enableCascadeCache=false --args --scene sponza
```

| Pass | traced + cascades | traced | Delta |
| --- | --- | --- | --- |
| `Frame total` | 5.177 | 4.664 | **−0.513 (−9.9%)** |
| `CSMShadowPass` | 0.564 | 0.000 | −0.564 |
| `ShadowGpuCulling` | 0.029 | — | A only |
| `MainHDRPass` | 3.372 | 3.423 | +0.051 (+1.5%) |

`MainHDRPass` moved 0.051 ms the wrong way although both sides run the same
shader; it clears its own control drift, but nothing in the change explains it,
so `Frame total` is the number: skipping saves about half a millisecond on every
frame whose cascades would have redrawn. The two series were measured separately
and do not add exactly, but together they put tracing on a moving camera at
roughly 1.2 ms cheaper than the cascades on this scene.

A first attempt at this series moved the camera with `--camera-orbit` instead.
The orbit carried it outside the building, where the frame costs under a third
as much, and the control drifted 2.2%; holding the view and turning the cache off
measures the same redraw without changing what is on screen.

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
| Skinned BLAS (`ALLOW_UPDATE`) | renderer, one structure | storage with the positions; built by the first frame that traces with the mesh shown, refitted by every later one | same |
| Pre-skin pipeline (`skin_positions.comp`, push constants only) | renderer | with the positions | same |
| Set 0 binding 16, `ACCELERATION_STRUCTURE_KHR` | written once into the shared bindless set | `createMaterialDescriptorSet` | with the pool |

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
| previous frame's ray queries (fragment, AS read) and previous builds (TLAS and scratch writes) -> this frame's build | a global barrier `RayTracingScene::recordTopLevelBuild` records first |
| this frame's build -> its ray queries | the render graph: `RayTracingSceneBuild` declares `AccelerationStructureBuild` on the TLAS storage buffer, and `MainHDRPass`, `MainHDRPhase2` and `TransparentPass` declare `AccelerationStructureRead` |
| mesh upload (transfer) -> BLAS build | a barrier at the top of the startup build, whose first scope reaches back over the earlier upload submissions on the same queue |
| previous frame's refit reads of the skinned positions -> this frame's pre-skin writes | the same entry barrier, which names the compute stage as a destination |
| pre-skin writes -> refit reads | a barrier inside `RayTracingSkinnedCaster::record` |
| refit -> TLAS build, and -> the rays | a global barrier after the refit |
| startup BLAS builds -> the rays | the entry barrier, which names the fragment stage as a destination |

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
traced draw.

Everything runs on the graphics queue. The asynchronous cluster passes never
read the TLAS, so no queue-family ownership is involved.

## Binary contracts

- **FrameConstants** gains two vec4s at the end: `rayTracedShadow` (geometry
  record address as a uvec2, then the `kRtShadowFlag*` word) at offset 624 and
  `rayTracedShadowParams` (normal offset, offset per unit of distance, ray
  length) at 640. Pinned by `static_assert` in `RendererInternal.h`.
- **The flag word** is `kRtShadowFlagEnabled`, `kRtShadowFlagDebugVisibility`
  and `kRtShadowFlagDebugCompareCascades`, mirrored in
  `object_frame_data.glsl` and pinned by `tools/check_shader_constants.py`.
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
(`rayTracedShadowOffsetPerDistance`). Where nothing faces the sun the ray is
skipped, because the term it produces multiplies zero.

The ray is as long as the diagonal of the world bounds of everything drawn this
frame (`rayTracedShadowRayLength`). The first version stopped it at the camera's
far plane, which is the wrong bound: the far plane limits which receivers are
drawn, not how far away their occluders are, and the TLAS holds every caster,
in view or not. A tall caster further than the far plane along the sun direction
from a visible receiver would have been missed.

The shadow is hard: one ray toward a light with no extent.

## Debug views and settings

| Setting | Default | |
| --- | --- | --- |
| `rayTracedShadows.enabled` | false | |
| `rayTracedShadows.renderCascades` | false | keep drawing the cascades while tracing |
| `rayTracedShadows.normalOffset` | 0.02 | world units, clamped to [0, 1] |
| `rayTracedShadows.pixelFootprintOffset` | 1.0 | pixels, clamped to [0, 16] |
| `rayTracedShadows.debugVisibility` | false | the traced term alone, white lit |
| `rayTracedShadows.debugCompareCascades` | false | red where only the ray finds an occluder, blue where only the cascades do; wins over the visibility view |

The CI sweep renders it enabled (cascades skipped), enabled with both debug
views and `renderCascades`, and enabled with `--disable-ray-query`
(`config/ci/`). Mesa's lavapipe exposes all three extensions, so the first two
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

**The cascades are the outlier on the two scenes with a strong sun.** On
`--scene sunlit` the traced image shows a long shadow across the right of the
yard and a second sphere shadow that the cascades do not draw; the virtual
shadow map draws both. On Sponza, against the virtual shadow map:

| Path | Pixels differing from VSM | Mean absolute luminance difference |
| --- | --- | --- |
| ray-traced | 15062 (1.6%), max delta 7 | 0.006 |
| cascades | 374721 (40.7%), max delta 11 | 0.397 |

Two independent methods agree and the cascades disagree with both, so the
cascades are missing casters there -- most likely casters outside a cascade's
light-space depth range, though that has not been isolated. It predates this
change and is recorded here because this path is what exposed it.

## Limitations

- **The skinned BLAS is refitted, never rebuilt** after its first build, so its
  hierarchy quality depends on the pose it was first built in. Fine for the demo
  rig; a mesh that deforms far from its first pose would want a periodic rebuild.
- **Hard shadows only.** No sun angular radius and no denoising.
- **The cascades come back with fog or probe capture,** which read them, so on
  those frames tracing adds its cost without removing theirs.
- **Virtual shadow map pages still render** when VSM is enabled alongside
  tracing, though nothing samples them.
- **The TLAS is rebuilt every frame that traces,** static scene or not: 0.233
  ms on Sponza at the 800 MHz pin, a third of what tracing saves there.
- **Shadow terminator on coarse spheres.** The comparison view shows a red rim
  where the ray crosses the faceted geometry just past the smooth terminator.
  It sits where `N.L` is near zero, so it is barely visible in the shaded image
  on these scenes.
- **Alpha tests at texture level 0,** where the cascades sample a mip.
- **Synchronization validation is blind to it,** as above.
