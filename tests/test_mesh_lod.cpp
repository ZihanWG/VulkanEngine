#include "renderer/MeshLod.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

using ve::renderer::buildLodChain;
using ve::renderer::kMaxMeshLods;
using ve::renderer::kMinLodIndexCount;
using ve::renderer::LodBuildSettings;
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

    const std::vector<MeshLod> lods = buildLodChain(combined,
                                                    primitiveCount,
                                                    primitiveCount,
                                                    grid.positions.data(),
                                                    grid.positions.size() / 3,
                                                    3 * sizeof(float));

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

    const std::vector<MeshLod> lods = buildLodChain(grid.indices,
                                                    0,
                                                    tooMany,
                                                    grid.positions.data(),
                                                    grid.positions.size() / 3,
                                                    3 * sizeof(float));

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
