#include "renderer/CascadeMath.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/geometric.hpp>
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
