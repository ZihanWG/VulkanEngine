# Clustered (Forward+) Lighting

The renderer evaluates many dynamic point and spot lights by assigning lights to
a view-space froxel grid in compute, then shading each fragment against only the
lights in its froxel. This keeps a forward HDR pipeline while scaling to hundreds
of lights, and reuses the engine's existing GPU-driven approach (bindless,
indirect, buffer-device-address). The directional sun and image-based lighting are
unchanged; clustered lighting is additive on top of them.

The subsystem lives in `src/renderer/ClusteredLighting.{h,cpp}`. The froxel math is
factored into a GPU-independent header, `src/renderer/ClusterGrid.h`, which is the
single source of truth for the grid dimensions and is unit-tested in
`tests/test_cluster_grid.cpp`. The shaders are `src/shaders/cluster_build.comp`,
`src/shaders/light_cull.comp`, and the shading path in
`src/shaders/simple_bindless.frag`.

## Grid

The grid is fixed at `kClusterGridX × kClusterGridY × kClusterGridZ` = 16×9×24 =
3456 froxels. Tiles divide the screen in normalized space (`tile / dim`), and depth
is sliced exponentially:

```
sliceDepth(k) = -zNear * (zFar / zNear)^(k / gridZ)
```

Exponential slicing matches the perspective depth distribution, so froxels stay
roughly cube-shaped through the frustum instead of becoming long slabs near the far
plane. The grid count is resolution-independent; only the per-tile pixel size
changes with the swapchain extent.

## Froxel Construction

`cluster_build.comp` runs one invocation per froxel. It rebuilds the froxel's
view-space AABB from the inverse projection: it unprojects the four tile corners to
get frustum-edge rays from the eye (the view-space origin), then intersects each ray
with the slice's near and far constant-depth planes and takes the min/max of the
resulting eight points.

Because the froxel-to-NDC mapping (`ndc = 2 · tile / dim − 1`) and the fragment's
tile lookup (`tile = floor(gl_FragCoord / tileSize)`) both go through
`uv = fragCoord / screen`, the grid stays consistent with the rendered image
regardless of the projection's y-flip. `ClusterGrid.h` carries a CPU mirror of this
math (`viewSpaceClusterBounds`, `clusterIndex`); a unit test projects a froxel's
centre and asserts it maps back to that froxel, cross-checking the build shader
against the fragment lookup.

The froxels depend only on the projection, so this pass is a candidate for
rebuilding only on projection change. It currently runs each frame for simplicity;
the cost is ~3456 invocations of cheap arithmetic.

## Light Assignment

`light_cull.comp` runs one invocation per froxel. It transforms each light into
view space and tests the light's bounding sphere (`position`, `range`) against the
froxel AABB with the standard squared-distance test (zero when the sphere centre is
inside the box). Surviving light indices are written to a fixed per-froxel region of
the light index list:

```
offset = clusterIndex * kMaxLightsPerCluster   // kMaxLightsPerCluster = 64
cell    = { offset, visibleCount }
```

Using a fixed region per froxel (rather than a globally compacted list with an
atomic allocator) trades index-list memory for zero atomic contention, and keeps the
shader branch-light. A froxel saturates at 64 lights; excess lights in a single
froxel are dropped.

## Shading

In `simple_bindless.frag`, when clustered lighting is active the fragment derives its
froxel from `gl_FragCoord.xy` (tile) and `vViewDepth` (slice), reads the
`{ offset, count }` cell, and loops only that froxel's lights. Each light uses the
same Cook-Torrance GGX / Smith / Fresnel terms as the directional light, with
inverse-square falloff, a smooth range cutoff, and a spot-cone term for spot lights.

The fragment reaches the light buffer, the froxel grid, and the index list through
**buffer-device-address** pointers passed in the push constant, not descriptor sets.
Per-frame buffers therefore need no per-frame descriptor sets and no descriptor
updates, which avoids descriptor hazards across frames in flight (the same reason the
object-data buffer uses BDA). When the clustered path is inactive — a missing grid
address or the runtime toggle — the shader falls back to looping all `lightCount`
lights, which doubles as the brute-force comparison path.

## Buffers and Ownership

`ClusteredLighting` owns the CPU light list, one light buffer per frame-in-flight, the
froxel grid resources, the two compute pipelines, their shared descriptor set layout,
and per-frame descriptor sets. The compute passes bind a single descriptor set (the
light buffer, froxel AABBs, grid cells, index list, and a params buffer); the
fragment reads grid + index list via BDA. The light buffer is bound both as a compute
SSBO and read by the fragment via BDA.

| Buffer | Per-frame | Memory | Access |
| --- | --- | --- | --- |
| Light list (`GpuLight[]`, 64 B each) | yes | host-visible, BDA | CPU upload; compute SSBO; fragment BDA |
| Froxel AABBs (`ClusterAabb[]`) | yes | device-local | `cluster_build` write, `light_cull` read |
| Grid cells (`{offset,count}[]`) | yes | device-local, BDA | `light_cull` write; fragment BDA |
| Light index list (`uint[]`) | yes | device-local, BDA | `light_cull` write; fragment BDA |
| Cluster params | yes | host-visible | CPU upload; compute SSBO |

Per-frame copies (one per frame-in-flight) prevent a frame from writing a buffer that
an in-flight frame is still reading.

## Synchronization

The passes are recorded before `MainHDRPass`, each in its own GPU profiler scope
(`ClusterBuild`, `LightCull`). `cluster_build` ends with a `COMPUTE → COMPUTE` buffer
barrier so `light_cull` reads finished froxel AABBs. `light_cull` ends with a
`COMPUTE → FRAGMENT` buffer barrier on the grid cells and index list so the main pass
reads finished light assignments. These barriers are explicit Synchronization2
`VkBufferMemoryBarrier2` records, consistent with the other manual buffer barriers in
the renderer.

## Debug and Tooling

- **Cluster heatmap:** a push-constant flag makes the main pass output a per-froxel
  light-count color ramp (blue → red) instead of shaded color, which visualizes
  froxel occupancy directly.
- **Lights (Clustered) panel:** toggles the clustered path on/off (the brute-force
  comparison) and the heatmap, drives the demo light count (0–512), and animates the
  swarm. The count slider is the clustered-path stress test.
- **GPU profiler:** `ClusterBuild` and `LightCull` appear as per-pass timing rows.

## Limitations and Future Work

- Point and spot lights are culled as bounding spheres; spot cones are not tightened
  to a cone test, so a spot is slightly over-included.
- A froxel saturates at 64 lights; the index list is preallocated at full size rather
  than globally compacted.
- No shadow-casting punctual lights (the directional light still owns the cascaded
  shadow maps).
- Froxel AABBs are rebuilt every frame; they could be rebuilt only on projection
  change.
- Lights are world-space spheres; tighter culling (oriented bounds, cone tests, or a
  BVH) would reduce per-froxel light counts in dense scenes.
