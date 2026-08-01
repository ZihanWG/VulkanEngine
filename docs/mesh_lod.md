# Discrete mesh LOD

Geometry carries a discrete level-of-detail chain, and the level is chosen **per
draw item on the GPU**, inside the cull dispatch that already has the bounds and
the camera. No extra pass, no CPU round-trip.

## Why the level can be a GPU decision

Every level of every primitive lives in the mesh's **single index buffer**, with
simplified levels appended after all authored indices. A level is therefore
addressed purely as a `(firstIndex, indexCount)` pair, and switching level is a
change of two fields in the indirect draw command — never a buffer rebind.

That is the whole trick. A LOD scheme that put each level in its own buffer would
force the selection back onto the CPU, because only the CPU can rebind buffers.

## Building the chains

`buildLodChain` (`renderer/MeshLod.{h,cpp}`) runs at mesh load for the built-in
cube and sphere and for every glTF primitive. It is GPU-free so the index
bookkeeping is unit-testable without a device — the same split already used by
`ClusterGrid.h`, `CascadeMath.h`, and `SkeletalAnimation.h`.

- Each level is simplified with `meshopt_simplify` from the **authored** geometry,
  not from the previous level: chaining simplifications compounds error, and
  build-time simplification is cheap enough that there is no reason to pay that.
- Each level is then vertex-cache optimized.
- Level `n` targets `1/2^n` of the authored index count.
- A level is rejected when the simplifier could not remove at least 15% of the
  previous level's triangles, and the chain stops below 32 triangles. Small
  geometry (the 12-triangle cube) correctly yields a **level-0-only chain** rather
  than levels that cost index memory and a table entry while buying nothing.

The portfolio sphere builds `2208 → 1104 → 552 → 275` triangles.

glTF meshes get one chain per primitive, since primitives are drawn independently
and each needs its own range per level. They share the mesh's flat LOD table,
addressed through `MeshPrimitive::lodBase` / `lodCount`.

## Reaching the GPU

`GpuCullDrawItem` addresses its chain with `lodBase` / `lodCount`. Those two words
are the record's **former padding**, so adding GPU LOD selection did not grow the
64-byte record or change its stride.

The table itself is a new SSBO at cull binding 6, rebuilt each frame next to the
cull input (scene edits add and remove meshes) and deduped by mesh, so a mesh's
chain uploads once no matter how many draw items reference it.
`renderer::MeshLod` uploads unchanged — two tightly packed `uint32`s is already
the std430 layout the shader expects — so there is no GPU mirror type to keep in
sync.

The main and shadow cull dispatches **share** the table: levels belong to the
mesh, only the selection bias differs per pass.

## Selecting

`projectedScreenRadius` / `selectLodIndex` in `renderer/MeshLod.h` are the
unit-tested reference; `cull.comp` mirrors them, the same way `ClusterGrid.h`
mirrors `cluster_build.comp`.

```
projectedRadiusPixels = radius / distance * projScaleY
level                 = log2(referenceRadiusPixels / projectedRadiusPixels) + bias
```

`projScaleY = viewportHeight * 0.5 * |proj[1][1]|` is carried in the previously
unused `viewportAndMipCount.w`. Each level covers one halving of the on-screen
radius, which lines up with the chain halving triangle count per level.

> **The `abs()` is load-bearing.** These projections carry the Vulkan Y-flip, so
> `proj[1][1]` is negative (see the ImGuizmo un-flip in `drawViewportGizmo`).
> Passing it through signed makes the shader's `projScaleY <= 0` guard fire on
> every draw item, silently pinning everything to level 0 — which is
> indistinguishable from correct behaviour in a scene where everything is close
> to the camera. This is exactly what the per-level counters were added to catch.

Shadow dispatches set push-constant bit 2 and add `shadowBias` on top of the
shared bias: shadow-map resolution and PCF hide simplification far better than
the main pass does.

Meshes with no chain (`lodCount == 0`) fall through to the authored range carried
on the draw item, so a missing LOD table degrades to full detail rather than an
out-of-range read.

## Debugging

The **Mesh LOD** panel exposes the selection knobs (all of them are just fields of
`GpuCullFrameParams::lodSettings` uploaded next frame), plus:

- **Selected levels** — emitted draws per level, read back from the cull stats
  block. Meshes without a chain are not counted, so the total can sit below the
  visible draw count.
- **Mesh chains** — what each loaded mesh actually built, including an explicit
  *"too small to simplify"* note. This is usually the answer to "why does my scene
  show no level variety".
- **Color by LOD** — green → yellow → orange → red as detail drops, modulated by
  scene luminance so silhouettes and shading still read through the tint.

Settings persist through `config/runtime_settings.json` under `"lod"`.

### How the level reaches the fragment shader

Only the cull shader knows which level it picked, so it packs the level into the
**high bits of `firstInstance`**:

```glsl
command.firstInstance = objectFrameDataIndex | (selectedLod << 16);
```

Draw-item indices are bounded by `kMaxDrawItems` (1024), so the high half is
always free. Every vertex shader that indexes object data masks with `0xFFFF`;
`simple.vert` additionally forwards `gl_InstanceIndex >> 16` as a flat varying.
This costs no extra buffer, descriptor, or push constant — only one `AND` in the
vertex stage.

Draws that never go through the cull pass (the skinned demo, the CPU fallback
path) leave the high bits zero and report level 0, which is accurate: they have no
selected level.

## Limitations

- **Selection is per draw item, not per cluster.** Large meshes switch as a whole,
  so a big object popping between levels is visible at the silhouette. Meshlet-
  granular selection is the direction modern engines went.
- **No cross-fade or dithered transition.** A switch is a hard pop. Screen-space
  dithering between adjacent levels is the usual cheap fix.
- **No screen-space error metric.** Selection uses projected bounding-sphere
  radius, which ignores how much geometric error a given level actually
  introduced. `meshopt_simplify` reports that error and it is currently discarded.
- **Transparent draws bypass LOD selection entirely.** They are issued as direct
  draws to preserve back-to-front sort order (see
  [docs/transparency.md](transparency.md)), so they never pass through the cull
  shader and always render level 0.
