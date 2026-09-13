#include "renderer/MeshLod.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using ve::renderer::appendLodChain;
using ve::renderer::buildLodChain;
using ve::renderer::buildLodChainDetached;
using ve::renderer::buildMeshlets;
using ve::renderer::MeshletBuild;
using ve::renderer::Meshlet;
using ve::renderer::kMaxMeshLods;
using ve::renderer::kMinLodIndexCount;
using ve::renderer::LodBuildSettings;
using ve::renderer::LodChainBuild;
using ve::renderer::LodSelectionSettings;
using ve::renderer::MeshLod;
using ve::renderer::projectedScreenRadius;
using ve::renderer::selectLodIndex;

namespace {

struct Grid {
    std::vector<float> positions;
    std::vector<uint32_t> indices;
};

// A flat NxN triangulated grid: dense enough to simplify, and every vertex is
// distinct so the simplifier has real work to do.
Grid makeGrid(uint32_t cells)
{
    Grid grid;
    const uint32_t verticesPerSide = cells + 1;
    for (uint32_t y = 0; y < verticesPerSide; ++y) {
        for (uint32_t x = 0; x < verticesPerSide; ++x) {
            grid.positions.push_back(static_cast<float>(x));
            grid.positions.push_back(static_cast<float>(y));
            // A gentle bump keeps the surface non-planar; a perfectly flat grid
            // collapses to almost nothing and would not exercise the ratios.
            grid.positions.push_back(static_cast<float>((x * y) % 7) * 0.25f);
        }
    }

    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            const uint32_t topLeft = y * verticesPerSide + x;
            const uint32_t topRight = topLeft + 1;
            const uint32_t bottomLeft = topLeft + verticesPerSide;
            const uint32_t bottomRight = bottomLeft + 1;
            grid.indices.insert(grid.indices.end(), {topLeft, bottomLeft, topRight});
            grid.indices.insert(grid.indices.end(), {topRight, bottomLeft, bottomRight});
        }
    }
    return grid;
}

std::vector<MeshLod> buildFor(Grid& grid, const LodBuildSettings& settings = {})
{
    return buildLodChain(grid.indices,
                         0,
                         static_cast<uint32_t>(grid.indices.size()),
                         grid.positions.data(),
                         grid.positions.size() / 3,
                         3 * sizeof(float),
                         /*debugName=*/{},
                         settings);
}

} // namespace

TEST_CASE("An empty index range produces no LOD chain", "[mesh][lod]")
{
    std::vector<uint32_t> indices;
    const std::vector<MeshLod> lods = buildLodChain(indices, 0, 0, nullptr, 0, 0);
    CHECK(lods.empty());
}

TEST_CASE("Level 0 always describes the authored range", "[mesh][lod]")
{
    Grid grid = makeGrid(16);
    const uint32_t authoredCount = static_cast<uint32_t>(grid.indices.size());

    const std::vector<MeshLod> lods = buildFor(grid);

    REQUIRE(!lods.empty());
    CHECK(lods[0].firstIndex == 0);
    CHECK(lods[0].indexCount == authoredCount);
}

TEST_CASE("Geometry below the minimum index count gets a level-0-only chain", "[mesh][lod]")
{
    // A cube is 36 indices, well under kMinLodIndexCount. Simplifying it would
    // cost a LOD-table entry and buy nothing.
    Grid grid = makeGrid(2); // 8 triangles = 24 indices
    REQUIRE(grid.indices.size() < kMinLodIndexCount);
    const size_t indicesBefore = grid.indices.size();

    const std::vector<MeshLod> lods = buildFor(grid);

    CHECK(lods.size() == 1);
    // Nothing was appended, so the index buffer is untouched.
    CHECK(grid.indices.size() == indicesBefore);
}

TEST_CASE("Simplified levels are appended after the authored indices", "[mesh][lod]")
{
    Grid grid = makeGrid(16);
    const size_t authoredCount = grid.indices.size();

    const std::vector<MeshLod> lods = buildFor(grid);

    REQUIRE(lods.size() > 1);
    // Level 0 keeps its original placement; every simplified level lives past the
    // authored data, which is what lets a LOD switch be a pure index-range change.
    for (size_t level = 1; level < lods.size(); ++level) {
        CHECK(lods[level].firstIndex >= authoredCount);
        CHECK(static_cast<size_t>(lods[level].firstIndex) + lods[level].indexCount <= grid.indices.size());
    }
}

TEST_CASE("LOD levels shrink monotonically", "[mesh][lod]")
{
    Grid grid = makeGrid(16);

    const std::vector<MeshLod> lods = buildFor(grid);

    REQUIRE(lods.size() > 1);
    for (size_t level = 1; level < lods.size(); ++level) {
        CHECK(lods[level].indexCount < lods[level - 1].indexCount);
    }
}

TEST_CASE("Every LOD range holds whole triangles and valid vertex indices", "[mesh][lod]")
{
    Grid grid = makeGrid(16);
    const uint32_t vertexCount = static_cast<uint32_t>(grid.positions.size() / 3);

    const std::vector<MeshLod> lods = buildFor(grid);

    REQUIRE(!lods.empty());
    for (const MeshLod& lod : lods) {
        CHECK(lod.indexCount % 3 == 0);
        for (uint32_t offset = 0; offset < lod.indexCount; ++offset) {
            CHECK(grid.indices[lod.firstIndex + offset] < vertexCount);
        }
    }
}

TEST_CASE("The chain never exceeds the configured level cap", "[mesh][lod]")
{
    Grid grid = makeGrid(32);

    const std::vector<MeshLod> lods = buildFor(grid);

    CHECK(lods.size() <= kMaxMeshLods);
}

TEST_CASE("A lower level cap truncates the chain", "[mesh][lod]")
{
    Grid grid = makeGrid(32);
    LodBuildSettings settings{};
    settings.maxLods = 2;

    const std::vector<MeshLod> lods = buildFor(grid, settings);

    CHECK(lods.size() <= 2);
}

TEST_CASE("A chain built at a non-zero offset preserves level 0's placement", "[mesh][lod]")
{
    // glTF meshes build one chain per primitive, so ranges routinely start part
    // way into a shared index buffer.
    Grid grid = makeGrid(16);
    const uint32_t primitiveCount = static_cast<uint32_t>(grid.indices.size());

    // Prepend a second primitive's worth of indices so the range under test does
    // not start at zero.
    std::vector<uint32_t> combined(grid.indices.begin(), grid.indices.end());
    combined.insert(combined.end(), grid.indices.begin(), grid.indices.end());

    const std::vector<MeshLod> lods = buildLodChain(
        combined, primitiveCount, primitiveCount, grid.positions.data(), grid.positions.size() / 3, 3 * sizeof(float));

    REQUIRE(!lods.empty());
    CHECK(lods[0].firstIndex == primitiveCount);
    CHECK(lods[0].indexCount == primitiveCount);
    for (size_t level = 1; level < lods.size(); ++level) {
        CHECK(lods[level].firstIndex >= 2 * primitiveCount);
    }
}

TEST_CASE("An out-of-range request degrades to a level-0-only chain", "[mesh][lod]")
{
    // Defensive: a caller passing a range past the end must not read out of
    // bounds, it must just skip simplification.
    Grid grid = makeGrid(16);
    const uint32_t tooMany = static_cast<uint32_t>(grid.indices.size()) + 300;

    const std::vector<MeshLod> lods =
        buildLodChain(grid.indices, 0, tooMany, grid.positions.data(), grid.positions.size() / 3, 3 * sizeof(float));

    CHECK(lods.size() == 1);
    CHECK(lods[0].indexCount == tooMany);
}

TEST_CASE("A null position stream degrades to a level-0-only chain", "[mesh][lod]")
{
    Grid grid = makeGrid(16);
    const size_t indicesBefore = grid.indices.size();

    const std::vector<MeshLod> lods =
        buildLodChain(grid.indices, 0, static_cast<uint32_t>(grid.indices.size()), nullptr, 0, 0);

    CHECK(lods.size() == 1);
    CHECK(grid.indices.size() == indicesBefore);
}

// --- detached construction -----------------------------------------------
// These are the safety argument for building chains on worker threads: the
// detached path must produce exactly what the serial path does, and the layout
// must depend only on the order the appends happen in, never on when the
// expensive half ran.

TEST_CASE("Detached build plus append equals the serial chain", "[mesh][lod]")
{
    Grid serialGrid = makeGrid(24);
    Grid detachedGrid = makeGrid(24);
    const uint32_t indexCount = static_cast<uint32_t>(serialGrid.indices.size());

    const std::vector<MeshLod> serialLods = buildLodChain(serialGrid.indices,
                                                          0,
                                                          indexCount,
                                                          serialGrid.positions.data(),
                                                          serialGrid.positions.size() / 3,
                                                          3 * sizeof(float));

    const LodChainBuild build =
        buildLodChainDetached(std::span<const uint32_t>(detachedGrid.indices.data(), indexCount),
                              detachedGrid.positions.data(),
                              detachedGrid.positions.size() / 3,
                              3 * sizeof(float));
    const std::vector<MeshLod> detachedLods = appendLodChain(detachedGrid.indices, 0, build);

    REQUIRE(serialLods.size() == detachedLods.size());
    REQUIRE(serialLods.size() > 1); // otherwise this proves nothing
    for (size_t level = 0; level < serialLods.size(); ++level) {
        CHECK(serialLods[level].firstIndex == detachedLods[level].firstIndex);
        CHECK(serialLods[level].indexCount == detachedLods[level].indexCount);
    }

    // The index buffer itself must come out byte-identical, not merely the same
    // size: a rebasing bug would keep the ranges plausible and scramble geometry.
    CHECK(serialGrid.indices == detachedGrid.indices);
}

TEST_CASE("Appending in a fixed order is what fixes the layout", "[mesh][lod]")
{
    // Simulates what the parallel path does: build every primitive's chain first,
    // in whatever order the pool happened to run them, then append in primitive
    // order. The result must match building and appending one at a time.
    Grid grid = makeGrid(20);
    const uint32_t primitiveIndexCount = static_cast<uint32_t>(grid.indices.size());
    constexpr uint32_t kPrimitives = 3;

    std::vector<uint32_t> serialIndices;
    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
        serialIndices.insert(serialIndices.end(), grid.indices.begin(), grid.indices.end());
    }
    std::vector<uint32_t> batchedIndices = serialIndices;

    std::vector<std::vector<MeshLod>> serialLods;
    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
        serialLods.push_back(buildLodChain(serialIndices,
                                           primitive * primitiveIndexCount,
                                           primitiveIndexCount,
                                           grid.positions.data(),
                                           grid.positions.size() / 3,
                                           3 * sizeof(float)));
    }

    // Build all of them before appending any -- the ordering the pool imposes.
    std::vector<LodChainBuild> builds;
    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
        builds.push_back(buildLodChainDetached(
            std::span<const uint32_t>(batchedIndices.data() + primitive * primitiveIndexCount, primitiveIndexCount),
            grid.positions.data(),
            grid.positions.size() / 3,
            3 * sizeof(float)));
    }

    std::vector<std::vector<MeshLod>> batchedLods;
    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
        batchedLods.push_back(appendLodChain(batchedIndices, primitive * primitiveIndexCount, builds[primitive]));
    }

    CHECK(serialIndices == batchedIndices);
    REQUIRE(serialLods.size() == batchedLods.size());
    for (size_t primitive = 0; primitive < serialLods.size(); ++primitive) {
        REQUIRE(serialLods[primitive].size() == batchedLods[primitive].size());
        for (size_t level = 0; level < serialLods[primitive].size(); ++level) {
            CHECK(serialLods[primitive][level].firstIndex == batchedLods[primitive][level].firstIndex);
            CHECK(serialLods[primitive][level].indexCount == batchedLods[primitive][level].indexCount);
        }
    }
}

TEST_CASE("A detached build carries its own relative offsets", "[mesh][lod]")
{
    Grid grid = makeGrid(24);
    const LodChainBuild build =
        buildLodChainDetached(std::span<const uint32_t>(grid.indices.data(), grid.indices.size()),
                              grid.positions.data(),
                              grid.positions.size() / 3,
                              3 * sizeof(float));

    REQUIRE(!build.simplifiedLods.empty());
    CHECK(build.sourceIndexCount == static_cast<uint32_t>(grid.indices.size()));

    // Levels are packed back to back inside the build's own buffer, so the first
    // starts at zero regardless of where the primitive lives in a mesh.
    uint32_t expectedOffset = 0;
    for (const MeshLod& level : build.simplifiedLods) {
        CHECK(level.firstIndex == expectedOffset);
        expectedOffset += level.indexCount;
    }
    CHECK(expectedOffset == static_cast<uint32_t>(build.simplifiedIndices.size()));

    // Appending into a buffer that already holds something must offset by exactly
    // that much and nothing more.
    std::vector<uint32_t> destination(1000, 7);
    const std::vector<MeshLod> lods = appendLodChain(destination, 0, build);
    REQUIRE(lods.size() == build.simplifiedLods.size() + 1);
    for (size_t level = 1; level < lods.size(); ++level) {
        CHECK(lods[level].firstIndex == 1000 + build.simplifiedLods[level - 1].firstIndex);
    }
    CHECK(destination.size() == 1000 + build.simplifiedIndices.size());
}

TEST_CASE("A detached build never logs from the worker", "[mesh][lod]")
{
    // Logger has no mutex, so the message is composed and handed back rather than
    // printed. An empty debug name still means no message at all.
    Grid grid = makeGrid(24);
    const std::span<const uint32_t> source(grid.indices.data(), grid.indices.size());
    const size_t vertexCount = grid.positions.size() / 3;

    const LodChainBuild unnamed = buildLodChainDetached(source, grid.positions.data(), vertexCount, 3 * sizeof(float));
    CHECK(unnamed.logMessage.empty());

    const LodChainBuild named =
        buildLodChainDetached(source, grid.positions.data(), vertexCount, 3 * sizeof(float), "GridMesh");
    REQUIRE(!named.simplifiedLods.empty());
    CHECK(named.logMessage.find("GridMesh") != std::string::npos);
    CHECK(named.logMessage.find("L0=") != std::string::npos);
    CHECK(named.logMessage.find("L1=") != std::string::npos);

    // A chain with nothing to simplify has nothing worth saying.
    std::vector<uint32_t> tiny(3, 0);
    const std::vector<float> position(9, 0.0f);
    const LodChainBuild trivial = buildLodChainDetached(
        std::span<const uint32_t>(tiny.data(), tiny.size()), position.data(), 3, 3 * sizeof(float), "Tiny");
    CHECK(trivial.logMessage.empty());
    CHECK(trivial.simplifiedLods.empty());
    CHECK(trivial.sourceIndexCount == 3);
}

TEST_CASE("An empty source range produces no chain at all", "[mesh][lod]")
{
    const LodChainBuild build = buildLodChainDetached(std::span<const uint32_t>{}, nullptr, 0, 0);
    CHECK(build.sourceIndexCount == 0);
    CHECK(build.simplifiedLods.empty());

    // No level 0 either, matching buildLodChain's zero-count behaviour.
    std::vector<uint32_t> indices(10, 1);
    const std::vector<MeshLod> lods = appendLodChain(indices, 0, build);
    CHECK(lods.empty());
    CHECK(indices.size() == 10);
}

// --- selection -----------------------------------------------------------
// These pin the reference implementation that selectLodIndex() in
// src/shaders/cull.comp mirrors.

TEST_CASE("Projected radius falls off with distance", "[mesh][lod]")
{
    const float projScaleY = 1000.0f;
    const float near = projectedScreenRadius(1.0f, 10.0f, projScaleY);
    const float far = projectedScreenRadius(1.0f, 20.0f, projScaleY);

    CHECK(near == Catch::Approx(100.0f));
    // Twice the distance, half the projected radius.
    CHECK(far == Catch::Approx(near * 0.5f));
}

TEST_CASE("Projected radius rejects degenerate inputs", "[mesh][lod]")
{
    CHECK(projectedScreenRadius(0.0f, 10.0f, 1000.0f) == Catch::Approx(0.0f));
    // A negative projScaleY is the Vulkan-Y-flip mistake: the caller must pass
    // |proj[1][1]|, and anything else is treated as unusable rather than
    // silently producing a negative radius.
    CHECK(projectedScreenRadius(1.0f, 10.0f, -1000.0f) == Catch::Approx(0.0f));
}

TEST_CASE("A camera inside the bounding sphere stays at full detail", "[mesh][lod]")
{
    const float radius = projectedScreenRadius(1.0f, 0.0f, 1000.0f);
    CHECK(selectLodIndex(radius, 4) == 0);
}

TEST_CASE("A mesh without a chain always selects level 0", "[mesh][lod]")
{
    LodSelectionSettings settings{};
    settings.forcedLod = 3;

    CHECK(selectLodIndex(1.0f, 0, settings) == 0);
    CHECK(selectLodIndex(1.0f, 1, settings) == 0);
}

TEST_CASE("Each halving of projected radius steps one level down", "[mesh][lod]")
{
    LodSelectionSettings settings{};
    settings.referenceRadiusPixels = 200.0f;

    CHECK(selectLodIndex(200.0f, 4, settings) == 0);
    CHECK(selectLodIndex(100.0f, 4, settings) == 1);
    CHECK(selectLodIndex(50.0f, 4, settings) == 2);
    CHECK(selectLodIndex(25.0f, 4, settings) == 3);
}

TEST_CASE("Selection clamps to the chain length", "[mesh][lod]")
{
    LodSelectionSettings settings{};
    settings.referenceRadiusPixels = 200.0f;

    // Far past the last level: clamp rather than read off the end of the table.
    CHECK(selectLodIndex(0.01f, 3, settings) == 2);
    CHECK(selectLodIndex(0.01f, 2, settings) == 1);
    // Larger on screen than the reference never goes below level 0.
    CHECK(selectLodIndex(10000.0f, 4, settings) == 0);
}

TEST_CASE("Bias shifts selection toward lower detail", "[mesh][lod]")
{
    LodSelectionSettings settings{};
    settings.referenceRadiusPixels = 200.0f;

    CHECK(selectLodIndex(200.0f, 4, settings) == 0);

    // This is what the shadow dispatch does on top of the shared bias.
    settings.bias = 2.0f;
    CHECK(selectLodIndex(200.0f, 4, settings) == 2);

    settings.bias = -1.0f;
    CHECK(selectLodIndex(100.0f, 4, settings) == 0);
}

TEST_CASE("A forced level overrides distance", "[mesh][lod]")
{
    LodSelectionSettings settings{};
    settings.forcedLod = 2;

    CHECK(selectLodIndex(10000.0f, 4, settings) == 2);
    CHECK(selectLodIndex(0.01f, 4, settings) == 2);
    // Still clamped to what the chain actually has.
    CHECK(selectLodIndex(10000.0f, 2, settings) == 1);
}

TEST_CASE("A zero or negative projected radius takes the cheapest level", "[mesh][lod]")
{
    CHECK(selectLodIndex(0.0f, 4) == 3);
    CHECK(selectLodIndex(-1.0f, 4) == 3);
}

// --- Meshlet construction -------------------------------------------------
//
// The load-bearing invariant is that meshletizing is a PERMUTATION of each
// level's triangles, not a rewrite of them. Every level keeps its
// (firstIndex, indexCount) range, so any primitive or LOD record pointing into
// the buffer stays valid; only the order inside a range changes. If that ever
// stopped holding, every later level would shift and the whole buffer would be
// silently wrong -- which is exactly the failure a cooked mesh cannot show.

namespace {

// Multiset of triangles in a range, each sorted internally so that a change in
// winding order would be caught by a separate test rather than read here as a
// different triangle.
std::vector<std::array<uint32_t, 3>>
trianglesIn(const std::vector<uint32_t>& indices, uint32_t firstIndex, uint32_t indexCount)
{
    std::vector<std::array<uint32_t, 3>> triangles;
    for (uint32_t i = 0; i < indexCount; i += 3) {
        std::array<uint32_t, 3> triangle{
            indices[firstIndex + i], indices[firstIndex + i + 1], indices[firstIndex + i + 2]};
        std::sort(triangle.begin(), triangle.end());
        triangles.push_back(triangle);
    }
    std::sort(triangles.begin(), triangles.end());
    return triangles;
}

// A grid with its LOD chain already built: the state every meshlet test starts
// from.
struct LoddedGrid {
    Grid grid;
    std::vector<uint32_t> indices;
    std::vector<MeshLod> lods;

    [[nodiscard]] size_t vertexCount() const
    {
        return grid.positions.size() / 3;
    }
    [[nodiscard]] const float* positions() const
    {
        return grid.positions.data();
    }
};

LoddedGrid makeLoddedGrid(uint32_t cells)
{
    LoddedGrid lodded;
    lodded.grid = makeGrid(cells);
    lodded.indices = lodded.grid.indices;
    lodded.lods = buildLodChain(lodded.indices,
                                0,
                                static_cast<uint32_t>(lodded.grid.indices.size()),
                                lodded.grid.positions.data(),
                                lodded.grid.positions.size() / 3,
                                sizeof(float) * 3);
    return lodded;
}

} // namespace

TEST_CASE("Meshletizing permutes each level's triangles and moves no range")
{
    LoddedGrid lodded = makeLoddedGrid(24);
    REQUIRE(lodded.lods.size() > 1);

    const std::vector<uint32_t> beforeIndices = lodded.indices;
    const std::vector<MeshLod> beforeLods = lodded.lods;

    const MeshletBuild build = buildMeshlets(lodded.indices,
                                                        std::span<const MeshLod>(lodded.lods),
                                                        lodded.positions(),
                                                        lodded.vertexCount(),
                                                        sizeof(float) * 3);

    REQUIRE_FALSE(build.meshlets.empty());
    // A permutation of every range leaves the total untouched. If this grew or
    // shrank, every level after the first would already be pointing at the
    // wrong triangles.
    REQUIRE(lodded.indices.size() == beforeIndices.size());

    for (size_t level = 0; level < lodded.lods.size(); ++level) {
        INFO("level " << level);
        CHECK(lodded.lods[level].firstIndex == beforeLods[level].firstIndex);
        CHECK(lodded.lods[level].indexCount == beforeLods[level].indexCount);
        CHECK(trianglesIn(lodded.indices, lodded.lods[level].firstIndex, lodded.lods[level].indexCount) ==
              trianglesIn(beforeIndices, beforeLods[level].firstIndex, beforeLods[level].indexCount));
    }
}

TEST_CASE("A level's meshlets tile its range exactly")
{
    LoddedGrid lodded = makeLoddedGrid(24);
    const MeshletBuild build = buildMeshlets(lodded.indices,
                                                        std::span<const MeshLod>(lodded.lods),
                                                        lodded.positions(),
                                                        lodded.vertexCount(),
                                                        sizeof(float) * 3);
    REQUIRE_FALSE(build.meshlets.empty());

    REQUIRE(build.rangesPerLod.size() == lodded.lods.size());
    for (size_t levelIndex = 0; levelIndex < lodded.lods.size(); ++levelIndex) {
        const MeshLod& level = lodded.lods[levelIndex];
        const glm::uvec2 range = build.rangesPerLod[levelIndex];
        INFO("level at " << level.firstIndex);
        REQUIRE(range.y > 0);
        REQUIRE(static_cast<size_t>(range.x) + range.y <= build.meshlets.size());

        // Contiguous from the level's start, no gaps and no overlaps, ending
        // exactly at the level's end. A gap drops triangles the level is meant
        // to draw; an overlap draws them twice.
        uint32_t cursor = level.firstIndex;
        for (uint32_t i = 0; i < range.y; ++i) {
            const Meshlet& meshlet = build.meshlets[range.x + i];
            CHECK(meshlet.firstIndex == cursor);
            CHECK(meshlet.indexCount > 0);
            CHECK(meshlet.indexCount % 3 == 0);
            CHECK(meshlet.indexCount <= ve::renderer::kMeshletMaxTriangles * 3U);
            cursor += meshlet.indexCount;
        }
        CHECK(cursor == level.firstIndex + level.indexCount);
    }

    // Every meshlet belongs to exactly one level: a meshlet no level points at
    // would never be drawn, and the table would be quietly larger than the
    // geometry it describes.
    size_t covered = 0;
    for (const glm::uvec2& range : build.rangesPerLod) {
        covered += range.y;
    }
    CHECK(covered == build.meshlets.size());
}

TEST_CASE("A meshlet's bounding sphere contains its own triangles")
{
    // The sphere is what the frustum and occlusion tests use, so one that does
    // not contain its geometry culls triangles that are visible.
    LoddedGrid lodded = makeLoddedGrid(24);
    const MeshletBuild build = buildMeshlets(lodded.indices,
                                                        std::span<const MeshLod>(lodded.lods),
                                                        lodded.positions(),
                                                        lodded.vertexCount(),
                                                        sizeof(float) * 3);
    REQUIRE_FALSE(build.meshlets.empty());

    for (const Meshlet& meshlet : build.meshlets) {
        for (uint32_t i = 0; i < meshlet.indexCount; ++i) {
            const uint32_t vertex = lodded.indices[meshlet.firstIndex + i];
            const float dx = lodded.grid.positions[vertex * 3 + 0] - meshlet.centerRadius.x;
            const float dy = lodded.grid.positions[vertex * 3 + 1] - meshlet.centerRadius.y;
            const float dz = lodded.grid.positions[vertex * 3 + 2] - meshlet.centerRadius.z;
            const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            // A small tolerance: the radius comes back in float, and an
            // exact-fit sphere is allowed to touch its extreme vertex.
            CHECK(distance <= meshlet.centerRadius.w + 1.0e-3f);
        }
        // Above 1 the GPU cone test stops being merely permissive and becomes
        // nonsensical; 1 itself is the documented "never reject" value.
        CHECK(meshlet.coneAxisCutoff.w <= 1.0f);
    }
}

TEST_CASE("Meshletizing without positions yields no meshlets, not missing geometry")
{
    // The consumer contract is that meshletCount == 0 means "draw the level
    // whole". Returning an empty table while leaving a stale non-zero count
    // would read as "draw nothing" and lose the mesh silently.
    LoddedGrid lodded = makeLoddedGrid(24);
    const std::vector<uint32_t> before = lodded.indices;

    const MeshletBuild build =
        buildMeshlets(lodded.indices, std::span<const MeshLod>(lodded.lods), nullptr, 0, sizeof(float) * 3);

    CHECK(build.meshlets.empty());
    CHECK(lodded.indices == before);
    // No ranges at all, rather than ranges of zero pointing into an empty
    // table: the caller must be able to tell "not meshletized" from "meshletized
    // into nothing".
    CHECK(build.rangesPerLod.empty());
}

TEST_CASE("A triangle cap not divisible by four is rounded down rather than trusted")
{
    // meshopt requires max_triangles divisible by 4 and leaves a violation
    // undefined rather than diagnosing it, so the builder normalises instead of
    // passing the caller's number through.
    LoddedGrid lodded = makeLoddedGrid(16);
    ve::renderer::MeshletBuildSettings settings{};
    settings.maxTriangles = 126;

    const MeshletBuild build = buildMeshlets(lodded.indices,
                                                        std::span<const MeshLod>(lodded.lods),
                                                        lodded.positions(),
                                                        lodded.vertexCount(),
                                                        sizeof(float) * 3,
                                                        settings);
    REQUIRE_FALSE(build.meshlets.empty());
    for (const Meshlet& meshlet : build.meshlets) {
        CHECK(meshlet.indexCount <= 124U * 3U);
    }
}
