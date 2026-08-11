// Clustered (Forward+) froxel grid dimensions, mirrored from
// ve::renderer::kClusterGrid* in src/renderer/ClusterGrid.h.
//
// These were duplicated verbatim in three shaders. Changing the grid means
// changing every consumer at once: the main pass and the fog injection both
// derive a cluster index from screen position, and the light cull writes the
// list they read. Missing one silently mismatches the addressing -- fragments
// read a different froxel's light list than the one they belong to, which looks
// like lights bleeding across tile boundaries rather than like a build error.
//
// The C++ side is authoritative; a static_assert there pins these values.
//
// Grid resolution is the main lever on the per-fragment light loop, which
// ablation put at ~64% of MainHDRPass. Each XY tile covers
// (drawableWidth / kClusterGridX) x (drawableHeight / kClusterGridY) pixels, and
// every fragment in a tile iterates every light assigned to it. Finer is
// cheaper per fragment but costs cluster memory: the light index list is
// preallocated at kClusterCount * kMaxLightsPerCluster entries rather than
// compacted, so it grows linearly with the cluster count.

#ifndef VE_CLUSTER_GRID_GLSL
#define VE_CLUSTER_GRID_GLSL

const uint kClusterGridX = 32u;
const uint kClusterGridY = 18u;
const uint kClusterGridZ = 24u;
const uint kClusterCount = kClusterGridX * kClusterGridY * kClusterGridZ;
const uint kMaxLightsPerCluster = 64u;

#endif // VE_CLUSTER_GRID_GLSL
