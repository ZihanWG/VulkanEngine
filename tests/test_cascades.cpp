#include "renderer/CascadeMath.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/glm.hpp>

using ve::renderer::CascadeBuildInput;
using ve::renderer::CascadeBuildOutput;
using ve::renderer::computeShadowCascades;
using ve::renderer::Frustum;
using ve::renderer::kMaxShadowCascades;

namespace {

// A reasonable default camera/light setup; individual tests tweak fields.
CascadeBuildInput defaultInput()
{
    CascadeBuildInput input;
    input.requestedCascadeCount = 4;
    input.nearPlane = 0.1f;
    input.farPlane = 100.0f;
    input.shadowDistance = 40.0f;
    input.lambda = 0.5f;
    input.enableTexelSnapping = true;
    input.shadowResolution = 2048;
    input.cameraPosition = glm::vec3(0.0f, 2.0f, 5.0f);
    input.cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    input.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    input.cameraVerticalFovRadians = glm::radians(60.0f);
    input.lightDirection = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f));
    input.aspectRatio = 16.0f / 9.0f;
    return input;
}

bool pointInsideFrustum(const Frustum& frustum, const glm::vec3& point, float epsilon)
{
    for (const auto& plane : frustum.planes) {
        if (plane.signedDistance(point) < -epsilon) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("Cascade split distances are increasing and bounded by the shadow distance", "[cascades]")
{
    const CascadeBuildOutput output = computeShadowCascades(defaultInput());

    // Strictly increasing across the four active cascades.
    CHECK(output.splitDistances.x < output.splitDistances.y);
    CHECK(output.splitDistances.y < output.splitDistances.z);
    CHECK(output.splitDistances.z < output.splitDistances.w);

    // The first split is past the near plane; the last reaches the shadow distance.
    CHECK(output.splitDistances.x > 0.1f);
    CHECK(output.splitDistances.w == Catch::Approx(40.0f));
}

TEST_CASE("Shadow distance is clamped into the camera depth range", "[cascades]")
{
    CascadeBuildInput input = defaultInput();
    input.shadowDistance = 1000.0f; // beyond farPlane
    input.farPlane = 100.0f;

    const CascadeBuildOutput output = computeShadowCascades(input);
    CHECK(output.splitDistances.w == Catch::Approx(100.0f));
}

TEST_CASE("Requested cascade count is clamped to the supported range", "[cascades]")
{
    SECTION("zero collapses to a single cascade repeated across all slots")
    {
        CascadeBuildInput input = defaultInput();
        input.requestedCascadeCount = 0;
        const CascadeBuildOutput output = computeShadowCascades(input);

        for (uint32_t i = 1; i < kMaxShadowCascades; ++i) {
            CHECK(output.splitDistances[i] == Catch::Approx(output.splitDistances[0]));
            CHECK(output.cascades[i].lightViewProjection == output.cascades[0].lightViewProjection);
        }
        // With one cascade, its far split is the (clamped) shadow distance.
        CHECK(output.splitDistances[0] == Catch::Approx(40.0f));
    }

    SECTION("overflow is capped at kMaxShadowCascades distinct cascades")
    {
        CascadeBuildInput input = defaultInput();
        input.requestedCascadeCount = 99;
        const CascadeBuildOutput output = computeShadowCascades(input);
        // Four active cascades remain strictly ordered (no division by an
        // over-large count, no out-of-bounds writes).
        CHECK(output.splitDistances.x < output.splitDistances.w);
    }
}

TEST_CASE("Inactive cascades repeat the last active cascade", "[cascades]")
{
    CascadeBuildInput input = defaultInput();
    input.requestedCascadeCount = 2;
    const CascadeBuildOutput output = computeShadowCascades(input);

    for (uint32_t i = 2; i < kMaxShadowCascades; ++i) {
        CHECK(output.cascades[i].lightViewProjection == output.cascades[1].lightViewProjection);
        CHECK(output.cascades[i].splitDepth == Catch::Approx(output.cascades[1].splitDepth));
    }
}

TEST_CASE("Extracted cascade frustum planes are normalized", "[cascades]")
{
    const CascadeBuildOutput output = computeShadowCascades(defaultInput());

    for (uint32_t c = 0; c < kMaxShadowCascades; ++c) {
        for (const auto& plane : output.cascades[c].lightFrustum.planes) {
            CHECK(glm::length(plane.normal) == Catch::Approx(1.0f).margin(1e-4f));
        }
    }
}

TEST_CASE("Each cascade frustum contains its own camera-slice center", "[cascades]")
{
    const CascadeBuildInput input = defaultInput();
    const CascadeBuildOutput output = computeShadowCascades(input);

    const glm::vec3 forward = glm::normalize(input.cameraTarget - input.cameraPosition);
    const uint32_t activeCount = 4;
    for (uint32_t c = 0; c < activeCount; ++c) {
        // The 8 slice corners are symmetric about their per-depth centers, so the
        // slice centroid is just the midpoint of the near/far depth centers.
        const float midDepth = 0.5f * (output.cascades[c].nearDepth + output.cascades[c].farDepth);
        const glm::vec3 sliceCenter = input.cameraPosition + forward * midDepth;
        CHECK(pointInsideFrustum(output.cascades[c].lightFrustum, sliceCenter, 1e-2f));
    }
}

TEST_CASE("computeShadowCascades is deterministic", "[cascades]")
{
    const CascadeBuildInput input = defaultInput();
    const CascadeBuildOutput a = computeShadowCascades(input);
    const CascadeBuildOutput b = computeShadowCascades(input);

    CHECK(a.splitDistances == b.splitDistances);
    for (uint32_t c = 0; c < kMaxShadowCascades; ++c) {
        CHECK(a.cascades[c].lightViewProjection == b.cascades[c].lightViewProjection);
    }
}

// --- Stable (bounding-sphere) fit -----------------------------------------
//
// The reason this fit exists is that the AABB fit's matrix moves whenever the
// camera does, which makes the per-cascade shadow cache unable to ever hit while
// the camera is moving. So the cases below assert the two invariants that make
// caching possible at all -- a matrix that is bit-identical under rotation, and
// one that only changes in whole-texel steps under translation -- plus the
// coverage property that must survive both.

namespace {

ve::renderer::CascadeBuildInput stableFitInput()
{
    ve::renderer::CascadeBuildInput input{};
    input.requestedCascadeCount = 4;
    input.nearPlane = 0.1f;
    input.farPlane = 100.0f;
    input.shadowDistance = 40.0f;
    input.lambda = 0.5f;
    input.enableStableFit = true;
    input.shadowResolution = 2048;
    input.cameraPosition = glm::vec3(3.0f, 2.0f, 7.0f);
    input.cameraTarget = input.cameraPosition + glm::vec3(0.0f, 0.0f, -1.0f);
    input.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    input.cameraVerticalFovRadians = glm::radians(60.0f);
    input.lightDirection = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));
    input.aspectRatio = 16.0f / 9.0f;
    return input;
}

// Bit-for-bit, not approximately: the cache keys off the exact bit pattern of
// this matrix, so "close enough" is a cache miss.
bool bitIdentical(const glm::mat4& a, const glm::mat4& b)
{
    return std::memcmp(&a, &b, sizeof(glm::mat4)) == 0;
}

// The reciprocal ortho half-extent, read straight off the matrix: the rows of
// an ortho projection times a view are the light basis vectors scaled by it.
float extentScale(const glm::mat4& viewProjection)
{
    return glm::length(glm::vec3(viewProjection[0][0], viewProjection[1][0], viewProjection[2][0]));
}

// Points the camera along `forward`, keeping its position.
void aimCamera(ve::renderer::CascadeBuildInput& input, const glm::vec3& forward)
{
    input.cameraTarget = input.cameraPosition + glm::normalize(forward);
}

} // namespace

// The extent is what the stable fit actually makes rotation-invariant, and it is
// what the AABB fit gets wrong. The rows of a light view-projection are the
// light basis scaled by the reciprocal ortho half-extents, so comparing their
// bit patterns compares the fitted extents exactly.
TEST_CASE("The stable fit's extent is invariant under camera rotation", "[cascades][stable-fit]")
{
    const ve::renderer::CascadeBuildOutput reference = ve::renderer::computeShadowCascades(stableFitInput());

    const std::array<glm::vec3, 4> directions{glm::vec3(1.0f, 0.0f, 0.0f),
                                              glm::vec3(0.0f, 0.0f, 1.0f),
                                              glm::vec3(-0.3f, 0.5f, -0.8f),
                                              glm::vec3(0.7f, -0.2f, 0.4f)};

    for (const glm::vec3& direction : directions) {
        ve::renderer::CascadeBuildInput rotated = stableFitInput();
        aimCamera(rotated, direction);
        const ve::renderer::CascadeBuildOutput output = ve::renderer::computeShadowCascades(rotated);

        for (uint32_t cascadeIndex = 0; cascadeIndex < 4; ++cascadeIndex) {
            REQUIRE(extentScale(output.cascades[cascadeIndex].lightViewProjection) ==
                    Catch::Approx(extentScale(reference.cascades[cascadeIndex].lightViewProjection)));
        }
    }
}

TEST_CASE("The AABB fit's extent breathes as the camera turns", "[cascades][stable-fit]")
{
    // The negative control, and the reason the stable fit exists: a breathing
    // extent means a breathing world-units-per-texel, so the snapping grid
    // itself never lands on the same lattice twice.
    ve::renderer::CascadeBuildInput input = stableFitInput();
    input.enableStableFit = false;
    const ve::renderer::CascadeBuildOutput reference = ve::renderer::computeShadowCascades(input);

    ve::renderer::CascadeBuildInput rotated = input;
    aimCamera(rotated, glm::vec3(1.0f, 0.0f, 0.0f));
    const ve::renderer::CascadeBuildOutput output = ve::renderer::computeShadowCascades(rotated);

    REQUIRE(extentScale(output.cascades[0].lightViewProjection) !=
            Catch::Approx(extentScale(reference.cascades[0].lightViewProjection)));
}

// MEASURED, and the reason the cascade cache is not helped by this fit: the
// slice's centre orbits the camera, so a rotation translates it by roughly the
// centre distance times the angle -- which dwarfs a texel long before the angle
// is visible. Tolerances measured at 2048px, 60 degree fov, 4 cascades were
// 0.0025 to 0.041 degrees of yaw. A camera turning at 10 degrees per second
// covers 0.17 degrees in a frame, so every cascade dirties on any real rotation.
//
// This case pins that as a known property rather than leaving it to be
// rediscovered: the matrix survives a rotation far below one texel of centre
// movement, and does not survive one merely small.
TEST_CASE("The stable fit does not survive a visible camera rotation", "[cascades][stable-fit]")
{
    const ve::renderer::CascadeBuildOutput reference = ve::renderer::computeShadowCascades(stableFitInput());

    ve::renderer::CascadeBuildInput imperceptible = stableFitInput();
    aimCamera(imperceptible, glm::vec3(std::sin(glm::radians(0.0005f)), 0.0f, -std::cos(glm::radians(0.0005f))));
    const ve::renderer::CascadeBuildOutput held = ve::renderer::computeShadowCascades(imperceptible);
    REQUIRE(bitIdentical(held.cascades[0].lightViewProjection, reference.cascades[0].lightViewProjection));

    ve::renderer::CascadeBuildInput slow = stableFitInput();
    aimCamera(slow, glm::vec3(std::sin(glm::radians(0.2f)), 0.0f, -std::cos(glm::radians(0.2f))));
    const ve::renderer::CascadeBuildOutput moved = ve::renderer::computeShadowCascades(slow);
    REQUIRE_FALSE(bitIdentical(moved.cascades[0].lightViewProjection, reference.cascades[0].lightViewProjection));
}

TEST_CASE("The stable fit quantizes camera translation", "[cascades][stable-fit]")
{
    ve::renderer::CascadeBuildInput input = stableFitInput();
    const ve::renderer::CascadeBuildOutput reference = ve::renderer::computeShadowCascades(input);

    // The largest cascade's texel is the coarsest, so a step far below it must
    // leave every cascade's matrix untouched.
    ve::renderer::CascadeBuildInput nudged = stableFitInput();
    nudged.cameraPosition += glm::vec3(1.0e-5f, 0.0f, 0.0f);
    nudged.cameraTarget += glm::vec3(1.0e-5f, 0.0f, 0.0f);
    const ve::renderer::CascadeBuildOutput nudgedOutput = ve::renderer::computeShadowCascades(nudged);

    for (uint32_t cascadeIndex = 0; cascadeIndex < 4; ++cascadeIndex) {
        REQUIRE(bitIdentical(nudgedOutput.cascades[cascadeIndex].lightViewProjection,
                             reference.cascades[cascadeIndex].lightViewProjection));
    }

    // A large move must still move it, or "quantized" would just mean "frozen".
    ve::renderer::CascadeBuildInput moved = stableFitInput();
    moved.cameraPosition += glm::vec3(25.0f, 0.0f, 0.0f);
    moved.cameraTarget += glm::vec3(25.0f, 0.0f, 0.0f);
    const ve::renderer::CascadeBuildOutput movedOutput = ve::renderer::computeShadowCascades(moved);

    REQUIRE_FALSE(bitIdentical(movedOutput.cascades[0].lightViewProjection,
                               reference.cascades[0].lightViewProjection));
}

TEST_CASE("The stable fit still covers its slice", "[cascades][stable-fit]")
{
    // Invariance is worthless if the cascade stops containing what it is meant
    // to shadow, and a snap that rounds the centre away is exactly how that
    // would happen. Every corner of every slice must land inside that cascade's
    // own frustum, from several camera orientations.
    const std::array<glm::vec3, 3> directions{glm::vec3(0.0f, 0.0f, -1.0f),
                                              glm::vec3(1.0f, 0.0f, 0.0f),
                                              glm::vec3(-0.3f, 0.5f, -0.8f)};

    for (const glm::vec3& direction : directions) {
        ve::renderer::CascadeBuildInput input = stableFitInput();
        aimCamera(input, direction);
        const ve::renderer::CascadeBuildOutput output = ve::renderer::computeShadowCascades(input);

        const glm::vec3 forward = glm::normalize(input.cameraTarget - input.cameraPosition);
        const glm::vec3 right = glm::normalize(glm::cross(forward, input.cameraUp));
        const glm::vec3 up = glm::normalize(glm::cross(right, forward));
        const float tanHalfFov = std::tan(input.cameraVerticalFovRadians * 0.5f);

        float cascadeNear = input.nearPlane;
        for (uint32_t cascadeIndex = 0; cascadeIndex < 4; ++cascadeIndex) {
            const float cascadeFar = output.cascades[cascadeIndex].splitDepth;
            const glm::mat4& viewProjection = output.cascades[cascadeIndex].lightViewProjection;

            for (const float depth : {cascadeNear, cascadeFar}) {
                const float halfHeight = tanHalfFov * depth;
                const float halfWidth = halfHeight * input.aspectRatio;
                const glm::vec3 center = input.cameraPosition + forward * depth;

                for (const float x : {-halfWidth, halfWidth}) {
                    for (const float y : {-halfHeight, halfHeight}) {
                        const glm::vec3 corner = center + right * x + up * y;
                        const glm::vec4 clip = viewProjection * glm::vec4(corner, 1.0f);
                        REQUIRE(clip.w > 0.0f);
                        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                        REQUIRE(std::abs(ndc.x) <= 1.0f);
                        REQUIRE(std::abs(ndc.y) <= 1.0f);
                        REQUIRE(ndc.z >= 0.0f);
                        REQUIRE(ndc.z <= 1.0f);
                    }
                }
            }

            cascadeNear = cascadeFar;
        }
    }
}
