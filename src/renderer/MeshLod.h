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
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ve::renderer {

// One discrete level of detail: a triangle range inside a mesh's index buffer,
// and how far that range strays from the authored surface. Level 0 is the
// authored geometry; simplified levels are appended after every authored index
// in the same buffer.
//
// This struct is uploaded verbatim into the per-frame GPU LOD table, so its
// stride IS the table's stride: every field here is one the cull shader reads.
// That is why meshlet ranges are NOT among them -- nothing on the GPU reads a
// meshlet (meshlet culling was measured and rejected, docs/mesh_lod.md), so
// buildMeshlets returns the ranges alongside the table instead. The error earns
// its place because screen-space selection projects it per draw item.
//
// Twelve bytes, not sixteen: std430 aligns a struct of scalars to 4, so the
// GLSL array stride is 12 as well. A 16-byte stride was tried and measured as
// free on frame time; the point is not to ship a dead word per entry.
struct MeshLod {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    // The farthest this level's surface strays from the authored one, in the
    // mesh's own units -- meshopt_simplify's error, turned from relative into
    // absolute. Zero for level 0, and never smaller than the previous level's,
    // so the coarsest level within an error budget is a prefix search.
    float error = 0.0f;
};

// One meshlet: a contiguous triangle range inside a mesh's index buffer, plus
// what a cull test needs to reject it.
//
// Contiguity is the load-bearing choice. meshopt_buildMeshlets natively produces
// meshlet-local vertex and micro-index arrays, which would need either a second
// index buffer or a mesh-shader path to draw. Instead the builder REORDERS each
// LOD level's triangles so a meshlet is just a (firstIndex, indexCount) pair --
// the identical addressing MeshLod already uses -- which means a surviving
// meshlet becomes an ordinary VkDrawIndexedIndirectCommand and the existing
// indirect draw path needs no change at all.
//
// Reordering triangles within a level is safe here for the same reason
// meshopt_optimizeVertexCache (already called on every simplified level) is:
// opaque geometry is depth-resolved, and transparency is sorted per object, not
// per triangle.
//
// Laid out for std430: two vec4s then four uints, 48 bytes, no padding holes.
struct Meshlet {
    // Bounding sphere in mesh-local space. xyz centre, w radius.
    glm::vec4 centerRadius{0.0f, 0.0f, 0.0f, 0.0f};
    // Normal cone for backface rejection, as meshopt_computeClusterBounds
    // defines it: xyz axis, w cos(angle/2). A cutoff of 1 means the triangles
    // face too many directions for a cone to reject anything, which is the
    // value the builder leaves when meshopt reports no usable cone -- so a
    // consumer that ignores the distinction simply never culls it.
    glm::vec4 coneAxisCutoff{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

static_assert(sizeof(Meshlet) == 48, "Meshlet is mirrored in GLSL as a 48-byte std430 struct.");
static_assert(sizeof(MeshLod) == 12, "MeshLod is mirrored in GLSL as a 12-byte std430 struct.");

// Meshlet sizes. 64/124 is the conventional pairing: meshopt caps vertices at
// 255 and triangles at 512-divisible-by-4, and these are what its own samples
// and the NVIDIA mesh-shader guidance use.
inline constexpr uint32_t kMeshletMaxVertices = 64;
inline constexpr uint32_t kMeshletMaxTriangles = 124;
// Trades meshlet compactness against normal-cone tightness. 0 ignores normals
// entirely and packs purely for locality; 1 packs purely for cone tightness and
// produces more, smaller meshlets. 0.5 is meshopt's suggested middle.
inline constexpr float kMeshletConeWeight = 0.5f;

struct MeshletBuildSettings {
    uint32_t maxVertices = kMeshletMaxVertices;
    uint32_t maxTriangles = kMeshletMaxTriangles;
    float coneWeight = kMeshletConeWeight;
};

// A mesh's meshlets, and which of them belong to each LOD level.
//
// The two are returned together rather than stored apart so they cannot desync:
// the only thing that produces either is buildMeshlets, and it produces both.
struct MeshletBuild {
    std::vector<Meshlet> meshlets;
    // One (base, count) into `meshlets` per level, parallel to the `lods` span
    // that was passed in. A count of 0 means that level was not meshletized and
    // must be drawn whole -- never "drawn not at all".
    std::vector<glm::uvec2> rangesPerLod;
};

// Groups every level in `lods` into meshlets.
//
// Rewrites each level's own range of `indices` in place so its meshlets are
// contiguous. Level ranges themselves do not move: only the order of triangles
// inside a range changes, so every existing (firstIndex, indexCount) stays valid
// and the caller's primitives keep pointing at the same geometry. `lods` is read,
// never written -- the per-level ranges come back in the result instead, which is
// what keeps meshlet data out of the GPU table's stride.
//
// An empty result means nothing could be meshletized (no positions, or no level
// with triangles).
[[nodiscard]] MeshletBuild buildMeshlets(std::vector<uint32_t>& indices,
                                         std::span<const MeshLod> lods,
                                         const float* vertexPositions,
                                         size_t vertexCount,
                                         size_t vertexStride,
                                         const MeshletBuildSettings& settings = {});

// True when a meshlet's normal cone proves every one of its triangles faces away
// from `viewPosition`, so the whole meshlet can be rejected before rasterization.
//
// This is the conservative *centre* form of meshopt's test rather than the apex
// form. The apex version is tighter, but it needs the cone apex stored per
// meshlet -- another 16 bytes -- and its extra reach is worth nothing unless a
// measurement says cone culling pays at all. The `radius / distance` term is what
// makes the centre form safe: it widens the cone by the angle the bounding sphere
// itself subtends, so this can only ever under-cull.
//
// All arguments are in the same space; the caller transforms the meshlet's
// centre, radius and axis into it. A cutoff of 1 never rejects, which is what
// the builder leaves when meshopt finds no usable cone.
[[nodiscard]] bool meshletConeCulled(
    const glm::vec3& center, float radius, const glm::vec3& coneAxis, float coneCutoff, const glm::vec3& viewPosition);

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
// How much a bent vertex normal costs the simplifier next to a moved position,
// per unit of normal. meshopt's own demos use 0.5.
//
// Without it the simplifier sees only positions, so it flattens curved,
// normal-mapped surfaces -- Sponza's curtains -- whose silhouettes barely move
// while their shading does: at a 1 px budget that left 11.5% of Sponza's frame
// off the full-detail image. With it, a collapse that bends normals costs error
// like one that moves the surface, the chains keep that detail, and the error
// recorded for each level counts the bend too (docs/mesh_lod.md).
inline constexpr float kLodNormalWeight = 0.5f;

struct LodBuildSettings {
    uint32_t maxLods = kMaxMeshLods;
    size_t minIndexCount = kMinLodIndexCount;
    float targetError = kLodTargetError;
    double indexRatio = kLodIndexRatio;
    double minReduction = kLodMinReduction;
    // 0 simplifies by position alone, as the chains were built before normals
    // counted; so does a build given no normal stream.
    float normalWeight = kLodNormalWeight;
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
// simplifier reads; vertexNormals, when not null, is a unit normal per vertex in
// the same interleaved layout (same stride), weighed by settings.normalWeight.
// The caller keeps ownership of both. When debugName is non-empty a multi-level
// chain is logged.
[[nodiscard]] std::vector<MeshLod> buildLodChain(std::vector<uint32_t>& indices,
                                                 uint32_t firstIndex,
                                                 uint32_t indexCount,
                                                 const float* vertexPositions,
                                                 size_t vertexCount,
                                                 size_t vertexStride,
                                                 const float* vertexNormals,
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

// Pure and thread-safe: reads `sourceIndices` and the vertex streams, writes
// only into the returned value. Safe to run on a JobSystem worker.
[[nodiscard]] LodChainBuild buildLodChainDetached(std::span<const uint32_t> sourceIndices,
                                                  const float* vertexPositions,
                                                  size_t vertexCount,
                                                  size_t vertexStride,
                                                  const float* vertexNormals,
                                                  std::string_view debugName = {},
                                                  const LodBuildSettings& settings = {});

// Concatenates a build's simplified indices onto `indices` and returns the full
// LOD table, level 0 first, rebased onto that buffer. `firstIndex` is where the
// primitive's authored range already sits.
//
// Calling this in a fixed order over the primitives is what makes the parallel
// path produce a byte-identical index buffer to the serial one: the expensive
// work is order-independent, and only this append decides layout.
[[nodiscard]] std::vector<MeshLod>
appendLodChain(std::vector<uint32_t>& indices, uint32_t firstIndex, const LodChainBuild& build);

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
    // Select by each level's projected geometric error (selectLodIndexByError)
    // rather than by projected radius. referenceRadiusPixels is then unused.
    bool screenSpaceError = false;
    // The most a selected level may stray from the authored surface on screen,
    // in pixels, before bias. Bias scales it by 2^bias, so one unit of bias
    // still means "one level coarser" in the sense the radius rule gave it:
    // each level roughly doubles the error it is allowed.
    float maxErrorPixels = 1.0f;
};

// Radius in pixels that a bounding sphere of `radius` at `distance` from the
// camera covers. projScaleY = viewportHeight * 0.5 * projection[1][1], i.e. the
// vertical pixels per unit of projected height at unit distance.
[[nodiscard]] float projectedScreenRadius(float radius, float distance, float projScaleY);

// Level for a given projected radius, clamped into [0, lodCount - 1]. Returns 0
// when the mesh has no chain, so a missing LOD table degrades to the authored
// geometry rather than to an out-of-range read.
[[nodiscard]] uint32_t
selectLodIndex(float projectedRadiusPixels, uint32_t lodCount, const LodSelectionSettings& settings = {});

// --- Screen-space error selection ----------------------------------------
//
// The radius rule above steps one level per halving of the on-screen radius,
// whatever each level actually cost in accuracy: a level that barely moved the
// surface and one that visibly dented it switch at the same distance. Selecting
// by error asks the question directly -- how many pixels would this level be
// wrong by, from here -- and takes the coarsest level that stays inside a pixel
// budget. Mirrored by selectLodIndex() in cull.comp's error branch.

// How much the model matrix can stretch a length: its longest basis column. An
// object-space error times this bounds the world-space error in any direction.
[[nodiscard]] float maxAxisScale(const glm::mat4& model);

// Distance from `point` to the nearest point of an axis-aligned box; zero
// inside it. Every triangle of the draw lies in the box, so no part of it can be
// closer than this, and an error projected at this distance bounds the error
// anywhere on the object.
[[nodiscard]] float distanceToAabb(const glm::vec3& point, const glm::vec3& boundsMin, const glm::vec3& boundsMax);

// Pixels a world-space length covers at `distance`, with the same projScaleY
// the radius rule uses. A camera at distance 0 (inside the bounds) reports
// infinity, so any non-zero error is too much there.
[[nodiscard]] float projectedErrorPixels(float worldError, float distance, float projScaleY);

// The coarsest level of `lods` whose error, scaled by `worldScale` and projected
// at `distance`, stays within settings.maxErrorPixels * 2^settings.bias. Level 0
// when the chain has one level, the inputs are degenerate, or nothing coarser
// fits; forcedLod overrides as it does for the radius rule. Relies on errors
// never decreasing along the chain, which buildLodChainDetached guarantees.
[[nodiscard]] uint32_t selectLodIndexByError(std::span<const MeshLod> lods,
                                             float worldScale,
                                             float distance,
                                             float projScaleY,
                                             const LodSelectionSettings& settings);

} // namespace ve::renderer
