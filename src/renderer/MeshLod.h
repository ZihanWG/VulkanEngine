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

} // namespace ve::renderer
