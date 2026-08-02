#include "renderer/PunctualShadowAtlas.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>

#include <cmath>
#include <limits>
#include <set>
#include <utility>

using Catch::Approx;
using ve::renderer::computeSpotShadowFrustum;
using ve::renderer::computeSpotShadowViewProjection;
using ve::renderer::GpuShadowSlot;
using ve::renderer::kInvalidPunctualShadowSlot;
using ve::renderer::kMaxPunctualShadowSlots;
using ve::renderer::kPunctualShadowAtlasSize;
using ve::renderer::kPunctualShadowTilesPerSide;
using ve::renderer::kPunctualShadowTileSize;
using ve::renderer::punctualShadowSlotFromFloat;
using ve::renderer::punctualShadowSlotRect;
using ve::renderer::punctualShadowSlotToFloat;
using ve::renderer::PunctualShadowAtlasAllocator;
using ve::renderer::punctualShadowSlotUvOffsetScale;
using ve::renderer::ShadowAtlasRect;

namespace {

// Projects a world point through a shadow view-projection and maps it the way
// the fragment shader does: perspective divide, then NDC xy -> [0,1] UV.
glm::vec3 projectToShadowUv(const glm::mat4& viewProjection, const glm::vec3& worldPosition)
{
    const glm::vec4 clip = viewProjection * glm::vec4(worldPosition, 1.0f);
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    return glm::vec3(ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f, ndc.z);
}

} // namespace

TEST_CASE("Atlas slots tile the texture without overlapping", "[shadowatlas]")
{
    std::set<std::pair<uint32_t, uint32_t>> origins;

    for (uint32_t slot = 0; slot < kMaxPunctualShadowSlots; ++slot) {
        const ShadowAtlasRect rect = punctualShadowSlotRect(slot);

        CHECK(rect.size == kPunctualShadowTileSize);
        // Every tile lands fully inside the atlas.
        CHECK(rect.x + rect.size <= kPunctualShadowAtlasSize);
        CHECK(rect.y + rect.size <= kPunctualShadowAtlasSize);
        // Tiles are grid-aligned.
        CHECK(rect.x % kPunctualShadowTileSize == 0u);
        CHECK(rect.y % kPunctualShadowTileSize == 0u);

        // No two slots share an origin, so with uniform sizes none overlap.
        CHECK(origins.insert({rect.x, rect.y}).second);
    }

    CHECK(origins.size() == kMaxPunctualShadowSlots);
    // The grid is exactly covered -- no wasted tiles, no slot off the end.
    CHECK(kMaxPunctualShadowSlots == kPunctualShadowTilesPerSide * kPunctualShadowTilesPerSide);
}

TEST_CASE("Out-of-range slots produce an empty rect", "[shadowatlas]")
{
    CHECK(punctualShadowSlotRect(kMaxPunctualShadowSlots).size == 0u);
    CHECK(punctualShadowSlotRect(kInvalidPunctualShadowSlot).size == 0u);
    CHECK(punctualShadowSlotUvOffsetScale(kMaxPunctualShadowSlots) == glm::vec4(0.0f));
}

TEST_CASE("Slot UV rects match their pixel rects", "[shadowatlas]")
{
    constexpr float kAtlasSize = static_cast<float>(kPunctualShadowAtlasSize);

    for (uint32_t slot = 0; slot < kMaxPunctualShadowSlots; ++slot) {
        const ShadowAtlasRect rect = punctualShadowSlotRect(slot);
        const glm::vec4 uv = punctualShadowSlotUvOffsetScale(slot);

        CHECK(uv.x == Approx(static_cast<float>(rect.x) / kAtlasSize));
        CHECK(uv.y == Approx(static_cast<float>(rect.y) / kAtlasSize));
        CHECK(uv.z == Approx(static_cast<float>(rect.size) / kAtlasSize));
        CHECK(uv.w == Approx(uv.z));

        // A tile's UV window stays within the atlas.
        CHECK(uv.x + uv.z <= 1.0f + 1.0e-6f);
        CHECK(uv.y + uv.w <= 1.0f + 1.0e-6f);
    }
}

TEST_CASE("The allocator hands out every slot once and then degrades", "[shadowatlas]")
{
    PunctualShadowAtlasAllocator allocator;
    std::set<uint32_t> handedOut;

    for (uint32_t i = 0; i < kMaxPunctualShadowSlots; ++i) {
        const uint32_t slot = allocator.allocate();
        REQUIRE(slot != kInvalidPunctualShadowSlot);
        CHECK(slot < kMaxPunctualShadowSlots);
        CHECK(handedOut.insert(slot).second);
    }

    CHECK(allocator.allocatedCount() == kMaxPunctualShadowSlots);
    CHECK(allocator.full());

    // Over budget is a graceful "unshadowed this frame", not an error.
    CHECK(allocator.allocate() == kInvalidPunctualShadowSlot);
    CHECK(allocator.allocate() == kInvalidPunctualShadowSlot);
    CHECK(allocator.allocatedCount() == kMaxPunctualShadowSlots);

    allocator.reset();
    CHECK(allocator.allocatedCount() == 0u);
    CHECK_FALSE(allocator.full());
    CHECK(allocator.allocate() == 0u);
}

TEST_CASE("Slot indices survive the float round-trip through GpuLight", "[shadowatlas]")
{
    for (uint32_t slot = 0; slot < kMaxPunctualShadowSlots; ++slot) {
        CHECK(punctualShadowSlotFromFloat(punctualShadowSlotToFloat(slot)) == slot);
    }

    // The sentinel encodes negative so the shader's `< 0.0` test catches it.
    CHECK(punctualShadowSlotToFloat(kInvalidPunctualShadowSlot) < 0.0f);
    CHECK(punctualShadowSlotFromFloat(punctualShadowSlotToFloat(kInvalidPunctualShadowSlot)) ==
          kInvalidPunctualShadowSlot);

    // Anything out of range or not a number decodes to unshadowed rather than
    // sliding through into an out-of-bounds slot fetch.
    CHECK(punctualShadowSlotFromFloat(-1.0f) == kInvalidPunctualShadowSlot);
    CHECK(punctualShadowSlotFromFloat(static_cast<float>(kMaxPunctualShadowSlots)) == kInvalidPunctualShadowSlot);
    CHECK(punctualShadowSlotFromFloat(1.0e9f) == kInvalidPunctualShadowSlot);
    CHECK(punctualShadowSlotFromFloat(std::numeric_limits<float>::quiet_NaN()) == kInvalidPunctualShadowSlot);
}

TEST_CASE("A spot shadow projection centers its cone axis in the tile", "[shadowatlas]")
{
    const glm::vec3 position{0.0f, 5.0f, 0.0f};
    const glm::vec3 direction{0.0f, -1.0f, 0.0f}; // straight down: the lookAt degenerate case
    constexpr float outerAngle = 0.6f;
    constexpr float range = 20.0f;

    const glm::mat4 viewProjection = computeSpotShadowViewProjection(position, direction, outerAngle, range);

    // A point on the cone axis lands dead center of the tile.
    const glm::vec3 onAxis = projectToShadowUv(viewProjection, position + direction * 10.0f);
    CHECK(onAxis.x == Approx(0.5f).margin(1.0e-5f));
    CHECK(onAxis.y == Approx(0.5f).margin(1.0e-5f));
    // ...and inside the [0,1] depth range that GLM_FORCE_DEPTH_ZERO_TO_ONE gives.
    CHECK(onAxis.z > 0.0f);
    CHECK(onAxis.z < 1.0f);

    // Depth increases with distance from the light.
    const glm::vec3 near = projectToShadowUv(viewProjection, position + direction * 2.0f);
    const glm::vec3 far = projectToShadowUv(viewProjection, position + direction * 18.0f);
    CHECK(near.z < far.z);
}

TEST_CASE("A spot shadow frustum inscribes the lit cone", "[shadowatlas]")
{
    const glm::vec3 position{1.0f, 4.0f, -2.0f};
    const glm::vec3 direction = glm::normalize(glm::vec3{0.3f, -1.0f, 0.2f});
    constexpr float outerAngle = 0.5f;
    constexpr float range = 15.0f;

    const glm::mat4 viewProjection = computeSpotShadowViewProjection(position, direction, outerAngle, range);

    // Build an orthonormal basis around the cone axis so we can step off-axis by
    // an exact angle.
    const glm::vec3 side = glm::normalize(glm::cross(direction, glm::vec3{0.0f, 1.0f, 0.0f}));

    constexpr float distance = 8.0f;
    // Just inside the cone: still on the tile.
    const glm::vec3 insideDirection = glm::normalize(direction + side * std::tan(outerAngle * 0.9f));
    const glm::vec3 inside = projectToShadowUv(viewProjection, position + insideDirection * distance);
    CHECK(inside.x >= 0.0f);
    CHECK(inside.x <= 1.0f);
    CHECK(inside.y >= 0.0f);
    CHECK(inside.y <= 1.0f);

    // Outside the cone: off the tile, which is what makes the shader's bounds
    // test equivalent to the spot cone cutoff.
    const glm::vec3 outsideDirection = glm::normalize(direction + side * std::tan(outerAngle * 1.6f));
    const glm::vec3 outside = projectToShadowUv(viewProjection, position + outsideDirection * distance);
    CHECK((outside.x < 0.0f || outside.x > 1.0f));

    // Past the range: beyond the far plane, so the depth leaves [0,1].
    const glm::vec3 beyondRange = projectToShadowUv(viewProjection, position + direction * (range + 5.0f));
    CHECK(beyondRange.z > 1.0f);
}

TEST_CASE("Spot shadow projections stay finite at degenerate inputs", "[shadowatlas]")
{
    const glm::vec3 position{0.0f, 2.0f, 0.0f};

    // A zero direction falls back to straight down rather than producing NaNs.
    const glm::mat4 zeroDirection = computeSpotShadowViewProjection(position, glm::vec3{0.0f}, 0.5f, 10.0f);
    // A cone at/over 90 degrees clamps under 180 degrees of FOV.
    const glm::mat4 wideCone =
        computeSpotShadowViewProjection(position, glm::vec3{0.0f, -1.0f, 0.0f}, 3.0f, 10.0f);
    // A range collapsed onto the near plane still leaves a usable depth range.
    const glm::mat4 zeroRange =
        computeSpotShadowViewProjection(position, glm::vec3{0.0f, -1.0f, 0.0f}, 0.5f, 0.0f);

    for (const glm::mat4& matrix : {zeroDirection, wideCone, zeroRange}) {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                CHECK(std::isfinite(matrix[column][row]));
            }
        }
        // A finite matrix that collapsed to rank-deficient would still be useless.
        CHECK(std::abs(glm::determinant(matrix)) > 0.0f);
    }

    // The frustum extractor tolerates those matrices too.
    const ve::renderer::Frustum frustum = computeSpotShadowFrustum(wideCone);
    for (const auto& plane : frustum.planes) {
        CHECK(std::isfinite(plane.distance));
        CHECK(std::isfinite(plane.normal.x));
    }
}

TEST_CASE("The GPU slot record keeps its std430 layout", "[shadowatlas]")
{
    // The fragment shader indexes these through a buffer-device-address pointer,
    // so a stride change here is a silent miscompare on the GPU.
    STATIC_REQUIRE(sizeof(GpuShadowSlot) == 96);
    STATIC_REQUIRE(offsetof(GpuShadowSlot, viewProjection) == 0);
    STATIC_REQUIRE(offsetof(GpuShadowSlot, atlasUvOffsetScale) == 64);
    STATIC_REQUIRE(offsetof(GpuShadowSlot, params) == 80);
}
