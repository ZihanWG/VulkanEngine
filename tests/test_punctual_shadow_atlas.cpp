#include "renderer/PunctualShadowAtlas.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <limits>
#include <set>
#include <vector>
#include <utility>

using Catch::Approx;
using ve::renderer::computeSpotShadowFrustum;
using ve::renderer::computeSpotShadowViewProjection;
using ve::renderer::GpuShadowSlot;
using ve::renderer::kInvalidPunctualShadowSlot;
using ve::renderer::kMaxPunctualShadowSlots;
using ve::renderer::kPunctualShadowAtlasSize;
using ve::renderer::kPunctualShadowMaxTileSize;
using ve::renderer::kPunctualShadowMinTileSize;
using ve::renderer::kPunctualShadowSizeClassCount;
using ve::renderer::punctualShadowSizeClassForRadius;
using ve::renderer::punctualShadowProjectedRadius;
using ve::renderer::punctualShadowSlotFromFloat;
using ve::renderer::punctualShadowSlotToFloat;
using ve::renderer::punctualShadowTileSizeForClass;
using ve::renderer::PunctualShadowAtlasAllocator;
using ve::renderer::shadowAtlasRectUvOffsetScale;
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

namespace {

// Do two tiles share any texel?
bool rectsOverlap(const ShadowAtlasRect& a, const ShadowAtlasRect& b)
{
    return a.x < b.x + b.size && b.x < a.x + a.size && a.y < b.y + b.size && b.y < a.y + a.size;
}

} // namespace

TEST_CASE("The allocator never hands out overlapping tiles", "[shadowatlas]")
{
    PunctualShadowAtlasAllocator allocator;
    std::vector<ShadowAtlasRect> handedOut;

    // Deliberately mixed classes, so splitting is exercised rather than a single
    // uniform size that could pass by accident.
    for (int round = 0; round < 40; ++round) {
        const uint32_t sizeClass = static_cast<uint32_t>(round % kPunctualShadowSizeClassCount);
        ShadowAtlasRect rect{};
        if (!allocator.allocate(sizeClass, rect)) {
            break;
        }

        CHECK(rect.size == punctualShadowTileSizeForClass(sizeClass));
        // Inside the atlas, and aligned to its own size -- a quadtree tile can
        // only ever sit on a multiple of its edge.
        CHECK(rect.x + rect.size <= kPunctualShadowAtlasSize);
        CHECK(rect.y + rect.size <= kPunctualShadowAtlasSize);
        CHECK(rect.x % rect.size == 0u);
        CHECK(rect.y % rect.size == 0u);

        for (const ShadowAtlasRect& existing : handedOut) {
            CHECK_FALSE(rectsOverlap(existing, rect));
        }
        handedOut.push_back(rect);
    }

    // The loop stops when the atlas cannot fit the next tile, not at a fixed
    // count -- a mixed-size run exhausts on area, so the useful assertions are
    // that it got a meaningful way in and never exceeded the atlas.
    CHECK(handedOut.size() > 20u);
    CHECK(allocator.allocatedCount() == handedOut.size());
    // Occupancy tracks area, not slot count.
    CHECK(allocator.occupancy() > 0.5f);
    CHECK(allocator.occupancy() <= 1.0f);

    // The area handed out equals the sum of the tiles, i.e. nothing was
    // double-counted and nothing leaked.
    uint64_t summedArea = 0;
    for (const ShadowAtlasRect& rect : handedOut) {
        summedArea += static_cast<uint64_t>(rect.size) * rect.size;
    }
    const auto atlasArea =
        static_cast<double>(kPunctualShadowAtlasSize) * static_cast<double>(kPunctualShadowAtlasSize);
    CHECK(allocator.occupancy() == Approx(static_cast<float>(static_cast<double>(summedArea) / atlasArea)));
}

TEST_CASE("The allocator fills the atlas exactly and then degrades", "[shadowatlas]")
{
    PunctualShadowAtlasAllocator allocator;

    // Smallest class only: the atlas should divide into exactly kMax tiles.
    const uint32_t smallest = kPunctualShadowSizeClassCount - 1;
    uint32_t granted = 0;
    ShadowAtlasRect rect{};
    while (allocator.allocate(smallest, rect)) {
        ++granted;
        REQUIRE(granted <= kMaxPunctualShadowSlots);
    }

    CHECK(granted == kMaxPunctualShadowSlots);
    CHECK(allocator.occupancy() == Approx(1.0f));
    // Exhaustion is a graceful "unshadowed this frame", not an error.
    CHECK_FALSE(allocator.allocate(smallest, rect));

    allocator.reset();
    CHECK(allocator.allocatedCount() == 0u);
    CHECK(allocator.occupancy() == Approx(0.0f));
    CHECK(allocator.allocate(0, rect));
}

TEST_CASE("A big tile consumes the area of many small ones", "[shadowatlas]")
{
    // The whole point of size classes: one class-0 tile costs what four class-1
    // tiles cost, so a frame of big tiles runs out sooner.
    PunctualShadowAtlasAllocator big;
    uint32_t bigCount = 0;
    ShadowAtlasRect rect{};
    while (big.allocate(0, rect)) {
        ++bigCount;
    }

    PunctualShadowAtlasAllocator small;
    uint32_t smallCount = 0;
    while (small.allocate(1, rect)) {
        ++smallCount;
    }

    CHECK(smallCount == bigCount * 4);
    CHECK(big.occupancy() == Approx(1.0f));
    CHECK(small.occupancy() == Approx(1.0f));
}

TEST_CASE("Tile rects map to the UV rects the shader reads", "[shadowatlas]")
{
    constexpr float kAtlasSize = static_cast<float>(kPunctualShadowAtlasSize);
    PunctualShadowAtlasAllocator allocator;

    for (uint32_t sizeClass = 0; sizeClass < kPunctualShadowSizeClassCount; ++sizeClass) {
        ShadowAtlasRect rect{};
        REQUIRE(allocator.allocate(sizeClass, rect));

        const glm::vec4 uv = shadowAtlasRectUvOffsetScale(rect);
        CHECK(uv.x == Approx(static_cast<float>(rect.x) / kAtlasSize));
        CHECK(uv.y == Approx(static_cast<float>(rect.y) / kAtlasSize));
        CHECK(uv.z == Approx(static_cast<float>(rect.size) / kAtlasSize));
        CHECK(uv.w == Approx(uv.z));
        CHECK(uv.x + uv.z <= 1.0f + 1.0e-6f);
        CHECK(uv.y + uv.w <= 1.0f + 1.0e-6f);
    }

    // An empty rect yields an empty window rather than sampling the whole atlas.
    CHECK(shadowAtlasRectUvOffsetScale(ShadowAtlasRect{}) == glm::vec4(0.0f));
}

TEST_CASE("Tile size follows a light's projected size", "[shadowatlas]")
{
    constexpr float projScaleY = 540.0f; // 1080p half-height at unit projection

    const glm::vec3 camera{0.0f, 0.0f, 0.0f};
    // Same range, increasing distance -> shrinking projected radius -> smaller
    // tiles. This is the behaviour distance-only ordering could not express.
    const float near = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 10.0f}, 8.0f, camera, projScaleY);
    const float mid = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 40.0f}, 8.0f, camera, projScaleY);
    const float far = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 400.0f}, 8.0f, camera, projScaleY);

    CHECK(near > mid);
    CHECK(mid > far);
    CHECK(punctualShadowSizeClassForRadius(near) <= punctualShadowSizeClassForRadius(mid));
    CHECK(punctualShadowSizeClassForRadius(mid) <= punctualShadowSizeClassForRadius(far));

    // A bigger light at the same distance outranks a smaller one, which pure
    // distance ordering treated as equal.
    const float small = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 40.0f}, 2.0f, camera, projScaleY);
    CHECK(mid > small);

    // Class thresholds are the tile sizes themselves.
    CHECK(punctualShadowSizeClassForRadius(static_cast<float>(kPunctualShadowMaxTileSize) + 1.0f) == 0u);
    CHECK(punctualShadowSizeClassForRadius(0.0f) == kPunctualShadowSizeClassCount - 1);
    for (uint32_t sizeClass = 0; sizeClass < kPunctualShadowSizeClassCount; ++sizeClass) {
        const uint32_t chosen =
            punctualShadowSizeClassForRadius(static_cast<float>(punctualShadowTileSizeForClass(sizeClass)));
        CHECK(chosen <= sizeClass);
    }

    // A point light is demoted one class: it buys six tiles, not one, so ranking
    // it against a spot on the same threshold lets one point light swallow six
    // of the largest tiles and starve everything behind it.
    for (float radius : {4000.0f, 900.0f, 400.0f, 10.0f}) {
        const uint32_t spotClass = punctualShadowSizeClassForRadius(radius, false);
        const uint32_t pointClass = punctualShadowSizeClassForRadius(radius, true);
        CHECK(pointClass >= spotClass);
        CHECK(pointClass <= kPunctualShadowSizeClassCount - 1);
        if (spotClass < kPunctualShadowSizeClassCount - 1) {
            CHECK(pointClass == spotClass + 1);
        }
    }
    // The demotion still clamps at the smallest class rather than running off.
    CHECK(punctualShadowSizeClassForRadius(0.0f, true) == kPunctualShadowSizeClassCount - 1);

    // Lights enclosing the camera outrank every light that does not...
    const float enclosing = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 1.0f}, 8.0f, camera, projScaleY);
    CHECK(enclosing > near);
    // Never a worse class than a light it outranks. Not pinned to class 0
    // outright: the class follows the projected radius, and this case's
    // deliberately small projScaleY keeps even a fully enclosing light under the
    // largest class's threshold. Real projScaleY values clear it comfortably.
    CHECK(punctualShadowSizeClassForRadius(enclosing) <= punctualShadowSizeClassForRadius(near));

    // ...but they still rank against *each other*. Saturating them all to one
    // value collapses the ordering exactly when a scene has many overlapping
    // lights: they tie, the tiebreak falls back to light index, and priority
    // degenerates into "whichever lights come first".
    const float deep = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 0.5f}, 8.0f, camera, projScaleY);
    const float shallow = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 7.0f}, 8.0f, camera, projScaleY);
    CHECK(deep > shallow);
    CHECK(shallow > punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 9.0f}, 8.0f, camera, projScaleY));
    CHECK(std::isfinite(deep));

    // Continuous where the two branches meet, so a light drifting across its own
    // radius does not pop between size classes.
    const float justInside = punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 8.0f}, 8.0f, camera, projScaleY);
    const float justOutside =
        punctualShadowProjectedRadius(glm::vec3{0.0f, 0.0f, 8.001f}, 8.0f, camera, projScaleY);
    CHECK(justInside == Approx(justOutside).epsilon(0.001));

    CHECK(punctualShadowTileSizeForClass(0) == kPunctualShadowMaxTileSize);
    CHECK(punctualShadowTileSizeForClass(kPunctualShadowSizeClassCount - 1) == kPunctualShadowMinTileSize);
    // Out-of-range classes clamp rather than shifting off the end.
    CHECK(punctualShadowTileSizeForClass(99) == kPunctualShadowMinTileSize);
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

TEST_CASE("The near plane scales with range to hold depth precision", "[shadowatlas]")
{
    using ve::renderer::kMinPunctualShadowNearPlane;
    using ve::renderer::punctualShadowNearPlane;

    // Never below the floor, however small the range.
    CHECK(punctualShadowNearPlane(0.0f) == Approx(kMinPunctualShadowNearPlane));
    CHECK(punctualShadowNearPlane(-5.0f) == Approx(kMinPunctualShadowNearPlane));
    CHECK(punctualShadowNearPlane(0.1f) == Approx(kMinPunctualShadowNearPlane));

    // Past the floor it tracks the range, holding the near:far ratio constant so
    // depth precision does not degrade as ranges grow.
    CHECK(punctualShadowNearPlane(16.0f) == Approx(0.32f));
    CHECK(punctualShadowNearPlane(100.0f) == Approx(2.0f));
    CHECK(punctualShadowNearPlane(200.0f) / 200.0f == Approx(punctualShadowNearPlane(100.0f) / 100.0f));

    // The practical payoff: a receiver partway down a spot's range lands well
    // inside the depth range instead of being crushed against the far plane.
    // With a fixed 0.05 near and a range of 16, a receiver at 6 units sat at
    // ~0.995 -- indistinguishable from the far-plane clear.
    const glm::vec3 position{0.0f, 6.0f, 0.0f};
    const glm::vec3 direction{0.0f, -1.0f, 0.0f};
    const glm::mat4 derived = computeSpotShadowViewProjection(position, direction, 0.5f, 16.0f);
    const float derivedDepth = projectToShadowUv(derived, glm::vec3{0.0f, 0.0f, 0.0f}).z;

    const glm::mat4 fixedNear =
        computeSpotShadowViewProjection(position, direction, 0.5f, 16.0f, 0.05f);
    const float fixedDepth = projectToShadowUv(fixedNear, glm::vec3{0.0f, 0.0f, 0.0f}).z;

    CHECK(derivedDepth < fixedDepth);
    CHECK(fixedDepth > 0.99f);  // the old behaviour: crushed against the clear
    CHECK(derivedDepth < 0.98f);
    CHECK(derivedDepth > 0.0f);

    // An explicit positive near plane still overrides the derivation.
    CHECK(fixedDepth == Approx(projectToShadowUv(
                                   computeSpotShadowViewProjection(position, direction, 0.5f, 16.0f, 0.05f),
                                   glm::vec3{0.0f, 0.0f, 0.0f})
                                   .z));
}

TEST_CASE("Cube face selection and face projections agree", "[shadowatlas]")
{
    using ve::renderer::computePointShadowFaceViewProjection;
    using ve::renderer::kPointShadowFaceCount;
    using ve::renderer::pointShadowFaceDirection;
    using ve::renderer::pointShadowFaceIndex;

    const glm::vec3 position{1.5f, 3.0f, -2.0f};
    constexpr float range = 12.0f;

    // Each face axis selects its own face.
    for (uint32_t face = 0; face < kPointShadowFaceCount; ++face) {
        CHECK(pointShadowFaceIndex(pointShadowFaceDirection(face)) == face);
    }

    // The invariant that matters: for directions all over the sphere, the face
    // the shader would pick is the face whose projection actually contains that
    // direction. A convention mismatch between the two sides samples the wrong
    // tile, which looks like plausible-but-wrong shadows rather than a crash.
    uint32_t facesExercised = 0;
    std::set<uint32_t> seenFaces;
    for (int xStep = -3; xStep <= 3; ++xStep) {
        for (int yStep = -3; yStep <= 3; ++yStep) {
            for (int zStep = -3; zStep <= 3; ++zStep) {
                if (xStep == 0 && yStep == 0 && zStep == 0) {
                    continue;
                }

                const glm::vec3 direction = glm::normalize(
                    glm::vec3(static_cast<float>(xStep), static_cast<float>(yStep), static_cast<float>(zStep)));
                const uint32_t face = pointShadowFaceIndex(direction);
                REQUIRE(face < kPointShadowFaceCount);
                seenFaces.insert(face);

                const glm::mat4 viewProjection =
                    computePointShadowFaceViewProjection(position, face, range);
                const glm::vec3 target = position + direction * 4.0f;
                const glm::vec4 clip = viewProjection * glm::vec4(target, 1.0f);

                // In front of that face's camera...
                REQUIRE(clip.w > 0.0f);
                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                // ...and inside its 90-degree frustum.
                CHECK(std::abs(ndc.x) <= 1.0f + 1.0e-4f);
                CHECK(std::abs(ndc.y) <= 1.0f + 1.0e-4f);
                CHECK(ndc.z >= 0.0f);
                CHECK(ndc.z <= 1.0f);
                ++facesExercised;
            }
        }
    }

    CHECK(facesExercised > 300);
    // All six faces were actually reached, so this is not passing by only ever
    // testing one of them.
    CHECK(seenFaces.size() == kPointShadowFaceCount);
}

TEST_CASE("Cube faces reserve all six tiles or none", "[shadowatlas]")
{
    using ve::renderer::kPointShadowFaceCount;

    PunctualShadowAtlasAllocator allocator;
    std::vector<ShadowAtlasRect> faces;

    REQUIRE(allocator.allocateRange(1, kPointShadowFaceCount, faces));
    CHECK(faces.size() == kPointShadowFaceCount);
    for (size_t i = 0; i < faces.size(); ++i) {
        for (size_t j = i + 1; j < faces.size(); ++j) {
            CHECK_FALSE(rectsOverlap(faces[i], faces[j]));
        }
    }

    // Fill the atlas with the largest class, then confirm a six-face request
    // fails whole. A partially reserved cube would have the light sampling
    // cleared tiles for its missing faces, which reads as light leaking through
    // solid geometry -- worse than not casting at all.
    PunctualShadowAtlasAllocator full;
    ShadowAtlasRect rect{};
    while (full.allocate(0, rect)) {
    }
    const uint32_t consumedBefore = full.allocatedCount();
    const float occupancyBefore = full.occupancy();

    std::vector<ShadowAtlasRect> denied{ShadowAtlasRect{1, 2, 3}};
    CHECK_FALSE(full.allocateRange(0, kPointShadowFaceCount, denied));
    CHECK(denied.empty());
    // Nothing was consumed by the failed attempt.
    CHECK(full.allocatedCount() == consumedBefore);
    CHECK(full.occupancy() == Approx(occupancyBefore));

    // A partial fit still fails whole: room for some faces is not room for six.
    PunctualShadowAtlasAllocator partial;
    while (partial.allocatedCount() + 3 < kMaxPunctualShadowSlots) {
        REQUIRE(partial.allocate(kPunctualShadowSizeClassCount - 1, rect));
    }
    const uint32_t partialBefore = partial.allocatedCount();
    CHECK_FALSE(partial.allocateRange(kPunctualShadowSizeClassCount - 1, kPointShadowFaceCount, faces));
    CHECK(partial.allocatedCount() == partialBefore);

    PunctualShadowAtlasAllocator edge;
    CHECK_FALSE(edge.allocateRange(0, 0, faces));
    CHECK(edge.allocatedCount() == 0u);
}

TEST_CASE("The cache key moves when any input moves", "[shadowatlas]")
{
    using ve::renderer::PunctualShadowCacheKey;

    // A cached shadow tile is only safe if nothing that could change its
    // contents has changed, so what matters is that every kind of input the
    // atlas pass consumes actually perturbs the key. A stale shadow shows up far
    // from its cause, so this is the test that keeps caching honest.
    const auto keyFor = [](const glm::mat4& viewProjection,
                           const ShadowAtlasRect& rect,
                           const glm::mat4& model,
                           const void* mesh,
                           uint32_t firstIndex,
                           float bias) {
        PunctualShadowCacheKey key;
        key.reset();
        key.add(viewProjection);
        key.add(rect);
        key.add(model);
        key.add(mesh);
        key.add(firstIndex);
        key.add(bias);
        return key.value();
    };

    const glm::mat4 viewProjection = glm::perspective(1.0f, 1.0f, 0.1f, 50.0f);
    const ShadowAtlasRect rect{512, 256, 512};
    const glm::mat4 model = glm::translate(glm::mat4{1.0f}, glm::vec3{1.0f, 2.0f, 3.0f});
    int meshA = 0;
    int meshB = 0;

    const uint64_t base = keyFor(viewProjection, rect, model, &meshA, 12u, 0.0015f);

    // Identical inputs hash identically -- otherwise nothing would ever cache.
    CHECK(keyFor(viewProjection, rect, model, &meshA, 12u, 0.0015f) == base);

    // The light moved, or its cone/range changed: all of that lands in the
    // projection.
    CHECK(keyFor(glm::perspective(1.01f, 1.0f, 0.1f, 50.0f), rect, model, &meshA, 12u, 0.0015f) != base);
    // The tile moved or changed size class.
    CHECK(keyFor(viewProjection, ShadowAtlasRect{512, 256, 256}, model, &meshA, 12u, 0.0015f) != base);
    CHECK(keyFor(viewProjection, ShadowAtlasRect{0, 256, 512}, model, &meshA, 12u, 0.0015f) != base);
    // A caster moved.
    CHECK(keyFor(viewProjection,
                 rect,
                 glm::translate(glm::mat4{1.0f}, glm::vec3{1.0f, 2.0f, 3.001f}),
                 &meshA,
                 12u,
                 0.0015f) != base);
    // A caster swapped mesh, or swapped LOD level within one mesh.
    CHECK(keyFor(viewProjection, rect, model, &meshB, 12u, 0.0015f) != base);
    CHECK(keyFor(viewProjection, rect, model, &meshA, 13u, 0.0015f) != base);
    // The raster depth bias was retuned, which changes what gets written.
    CHECK(keyFor(viewProjection, rect, model, &meshA, 12u, 0.0016f) != base);

    // Tiny float deltas count: the hash is over exact bit patterns, so this is
    // never an approximate comparison that could call a changed scene unchanged.
    glm::mat4 nudged = model;
    nudged[3][0] = std::nextafter(nudged[3][0], 100.0f);
    CHECK(keyFor(viewProjection, rect, nudged, &meshA, 12u, 0.0015f) != base);

    // Order matters, so two casters swapping places is not mistaken for no
    // change -- they can occlude each other differently.
    PunctualShadowCacheKey forward;
    forward.reset();
    forward.add(1.0f);
    forward.add(2.0f);
    PunctualShadowCacheKey reversed;
    reversed.reset();
    reversed.add(2.0f);
    reversed.add(1.0f);
    CHECK(forward.value() != reversed.value());

    // reset() really returns to the starting state, so frame N+1 does not
    // inherit frame N.
    PunctualShadowCacheKey reused;
    reused.reset();
    reused.add(7.0f);
    reused.reset();
    PunctualShadowCacheKey fresh;
    fresh.reset();
    CHECK(reused.value() == fresh.value());

    // An empty frame (no slots, no casters) is a stable key rather than a
    // coincidence of zero.
    CHECK(fresh.value() != 0u);
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
