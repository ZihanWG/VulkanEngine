#pragma once

// Discrete level-of-detail chain construction, kept GPU-free so the index
// bookkeeping is unit-testable without a Vulkan device — the same split used by
// ClusterGrid.h, CascadeMath.h, and SkeletalAnimation.h.
//
// The renderer never rebinds buffers to switch LOD: every level of every
// primitive lives in the mesh's single index buffer, and a level is addressed
// purely as a (firstIndex, indexCount) pair. That is what lets the GPU cull pass
// pick a level per draw item and write it straight into the indirect command.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ve::renderer {

// One discrete level of detail: a triangle range inside a mesh's index buffer.
// Level 0 is the authored geometry; simplified levels are appended after every
// authored index in the same buffer.
struct MeshLod {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
};

// Upper bound on the chain length, and the floor below which simplifying stops
// paying for itself (32 triangles).
inline constexpr uint32_t kMaxMeshLods = 4;
inline constexpr size_t kMinLodIndexCount = 96;
// Relative to the mesh extents, as meshopt_simplify defines it.
inline constexpr float kLodTargetError = 0.05f;
// Each level targets this fraction of the *authored* index count, raised to the
// level number: 1/2, 1/4, 1/8.
inline constexpr double kLodIndexRatio = 0.5;
// A level is rejected outright unless the simplifier removed at least this
// fraction of the previous level's triangles.
inline constexpr double kLodMinReduction = 0.15;

struct LodBuildSettings {
    uint32_t maxLods = kMaxMeshLods;
    size_t minIndexCount = kMinLodIndexCount;
    float targetError = kLodTargetError;
    double indexRatio = kLodIndexRatio;
    double minReduction = kLodMinReduction;
};

// Builds the LOD chain for the [firstIndex, firstIndex + indexCount) range of
// `indices`, appending each simplified level to the end of `indices` and
// returning the LOD records with level 0 first. Level 0 always exists, so the
// result is never empty for a non-empty range.
//
// Every level is simplified from the authored geometry rather than from the
// previous level: chaining simplifications compounds error, and simplifying is
// cheap enough at build time that there is no reason to pay that quality cost.
//
// vertexPositions/vertexCount/vertexStride describe the position stream the
// simplifier reads; the caller keeps ownership. When debugName is non-empty a
// multi-level chain is logged.
[[nodiscard]] std::vector<MeshLod> buildLodChain(std::vector<uint32_t>& indices,
                                                 uint32_t firstIndex,
                                                 uint32_t indexCount,
                                                 const float* vertexPositions,
                                                 size_t vertexCount,
                                                 size_t vertexStride,
                                                 std::string_view debugName = {},
                                                 const LodBuildSettings& settings = {});

// --- Parallel construction ----------------------------------------------
//
// Simplification dominates glTF import -- on Sponza it is 87% of it, since every
// level is simplified from the authored geometry and a scene is hundreds of
// independent primitives. The two functions below are the same algorithm split
// so those primitives can be simplified concurrently: the expensive half touches
// nothing shared, and the half that mutates the mesh's index buffer is trivial
// and stays serial.

// One primitive's simplified levels, built without touching the mesh's shared
// index buffer.
struct LodChainBuild {
    // Simplified levels only. Level 0 is the authored range, which already lives
    // in the shared buffer and is re-derived by appendLodChain. Each firstIndex
    // here is relative to `simplifiedIndices`, not to any mesh buffer.
    std::vector<MeshLod> simplifiedLods;
    std::vector<uint32_t> simplifiedIndices;
    // Zero means the source range was empty, so there is no level 0 either.
    uint32_t sourceIndexCount = 0;
    // Composed here and printed by the serial caller. Logger has no mutex, so a
    // worker thread must not log: concurrent calls interleave characters.
    std::string logMessage;
};

// Pure and thread-safe: reads `sourceIndices` and the position stream, writes
// only into the returned value. Safe to run on a JobSystem worker.
[[nodiscard]] LodChainBuild buildLodChainDetached(std::span<const uint32_t> sourceIndices,
                                                  const float* vertexPositions,
                                                  size_t vertexCount,
                                                  size_t vertexStride,
                                                  std::string_view debugName = {},
                                                  const LodBuildSettings& settings = {});

// Concatenates a build's simplified indices onto `indices` and returns the full
// LOD table, level 0 first, rebased onto that buffer. `firstIndex` is where the
// primitive's authored range already sits.
//
// Calling this in a fixed order over the primitives is what makes the parallel
// path produce a byte-identical index buffer to the serial one: the expensive
// work is order-independent, and only this append decides layout.
[[nodiscard]] std::vector<MeshLod> appendLodChain(std::vector<uint32_t>& indices,
                                                  uint32_t firstIndex,
                                                  const LodChainBuild& build);

// --- LOD selection -------------------------------------------------------
//
// Mirrored by selectLodIndex() in src/shaders/cull.comp, which is where it
// actually runs: selection happens per draw item inside the cull dispatch that
// already has the bounds and the camera, so picking a level costs no extra pass
// and no CPU round-trip. This C++ copy is the unit-tested reference — keep the
// two in sync, the same way ClusterGrid.h mirrors cluster_build.comp.

struct LodSelectionSettings {
    // Projected sphere radius, in pixels, at which level 0 is still the right
    // choice. Each halving of the on-screen radius steps one level down, which
    // lines up with the chain halving triangle count per level.
    float referenceRadiusPixels = 220.0f;
    // Positive biases toward *lower* detail. Shadow passes push a bias here
    // because shadow-map error hides simplification far better than the main
    // pass does.
    float bias = 0.0f;
    // >= 0 pins every draw item to that level, for the debug view.
    int32_t forcedLod = -1;
};

// Radius in pixels that a bounding sphere of `radius` at `distance` from the
// camera covers. projScaleY = viewportHeight * 0.5 * projection[1][1], i.e. the
// vertical pixels per unit of projected height at unit distance.
[[nodiscard]] float projectedScreenRadius(float radius, float distance, float projScaleY);

// Level for a given projected radius, clamped into [0, lodCount - 1]. Returns 0
// when the mesh has no chain, so a missing LOD table degrades to the authored
// geometry rather than to an out-of-range read.
[[nodiscard]] uint32_t selectLodIndex(float projectedRadiusPixels,
                                      uint32_t lodCount,
                                      const LodSelectionSettings& settings = {});

} // namespace ve::renderer
