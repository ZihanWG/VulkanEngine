#include "renderer/VolumetricFog.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>

#include <cmath>
#include <vector>

using Catch::Approx;
using ve::renderer::FogFroxelSample;
using ve::renderer::fogHeightDensity;
using ve::renderer::FogIntegratedSample;
using ve::renderer::fogSliceThickness;
using ve::renderer::fogSliceViewDepth;
using ve::renderer::fogViewDepthToSlice;
using ve::renderer::henyeyGreensteinPhase;
using ve::renderer::kDefaultFogMaxDistance;
using ve::renderer::kFogGridZ;
using ve::renderer::kFogNearPlane;
using ve::renderer::integrateFogSlice;

TEST_CASE("Fog slice depth and its inverse round-trip", "[fog]")
{
    constexpr float maxDistance = kDefaultFogMaxDistance;

    // The injection pass turns a slice into a world position with one of these
    // and the apply pass turns a fragment's depth back into a slice with the
    // other. If they disagree the fog slides relative to the geometry it should
    // be sitting in front of.
    for (uint32_t slice = 0; slice <= kFogGridZ; ++slice) {
        const float depth = fogSliceViewDepth(static_cast<float>(slice), maxDistance);
        const float recovered = fogViewDepthToSlice(depth, maxDistance);
        CHECK(recovered == Approx(static_cast<float>(slice)).margin(1.0e-3f));
    }

    // Endpoints land exactly on the volume bounds.
    CHECK(fogSliceViewDepth(0.0f, maxDistance) == Approx(kFogNearPlane));
    CHECK(fogSliceViewDepth(static_cast<float>(kFogGridZ), maxDistance) == Approx(maxDistance));

    // Depth outside the volume clamps rather than extrapolating off the ends.
    CHECK(fogViewDepthToSlice(0.0f, maxDistance) == Approx(0.0f));
    CHECK(fogViewDepthToSlice(maxDistance * 10.0f, maxDistance) == Approx(static_cast<float>(kFogGridZ)));
}

TEST_CASE("Fog slices grow with depth and tile the volume", "[fog]")
{
    constexpr float maxDistance = kDefaultFogMaxDistance;

    float summedThickness = 0.0f;
    float previousThickness = 0.0f;
    for (uint32_t slice = 0; slice < kFogGridZ; ++slice) {
        const float thickness = fogSliceThickness(slice, maxDistance);
        CHECK(thickness > 0.0f);
        // Exponential distribution: every slice is at least as thick as the one
        // before it, which is what puts resolution near the camera.
        CHECK(thickness >= previousThickness);
        previousThickness = thickness;
        summedThickness += thickness;
    }

    // The slices exactly cover the volume, with no gap or overlap.
    CHECK(summedThickness == Approx(maxDistance - kFogNearPlane).epsilon(0.001));
    // Resolution really is front-loaded, not merely non-decreasing.
    CHECK(fogSliceThickness(kFogGridZ - 1, maxDistance) > fogSliceThickness(0, maxDistance) * 10.0f);
}

TEST_CASE("Empty fog leaves the background untouched", "[fog]")
{
    FogIntegratedSample accumulated{};

    for (uint32_t slice = 0; slice < kFogGridZ; ++slice) {
        accumulated = integrateFogSlice(accumulated, FogFroxelSample{}, fogSliceThickness(slice, 64.0f));
    }

    // No medium means nothing scattered and nothing absorbed, so the apply pass
    // multiplies the scene by 1 and adds 0.
    CHECK(accumulated.transmittance == Approx(1.0f));
    CHECK(accumulated.inScattering.r == Approx(0.0f));
}

TEST_CASE("Transmittance decays monotonically and stays bounded", "[fog]")
{
    FogFroxelSample medium{};
    medium.inScattering = glm::vec3{0.4f, 0.5f, 0.7f};
    medium.extinction = 0.15f;

    FogIntegratedSample accumulated{};
    float previousTransmittance = accumulated.transmittance;
    glm::vec3 previousScatter = accumulated.inScattering;

    for (uint32_t slice = 0; slice < kFogGridZ; ++slice) {
        accumulated = integrateFogSlice(accumulated, medium, fogSliceThickness(slice, 64.0f));

        // Transmittance only ever decreases, and never leaves [0, 1] -- a value
        // above 1 would brighten the background instead of fogging it.
        CHECK(accumulated.transmittance <= previousTransmittance);
        CHECK(accumulated.transmittance >= 0.0f);
        CHECK(accumulated.transmittance <= 1.0f);
        // Scattered light only accumulates.
        CHECK(accumulated.inScattering.r >= previousScatter.r);

        previousTransmittance = accumulated.transmittance;
        previousScatter = accumulated.inScattering;
    }

    // Over 64 units at this density the far background is almost entirely gone.
    CHECK(accumulated.transmittance < 0.05f);
}

TEST_CASE("Dense fog converges instead of blowing out", "[fog]")
{
    // This is what the analytic slab integral buys. A naive
    // inScattering * thickness keeps adding light as density rises and drives
    // the result past the medium's own colour; the correct integral saturates
    // at it, because a fully opaque medium can only show its own scattering.
    const auto integrateAtDensity = [](float extinction) {
        FogFroxelSample medium{};
        medium.inScattering = glm::vec3{1.0f} * extinction; // scattering albedo of 1
        medium.extinction = extinction;

        FogIntegratedSample accumulated{};
        for (uint32_t slice = 0; slice < kFogGridZ; ++slice) {
            accumulated = integrateFogSlice(accumulated, medium, fogSliceThickness(slice, 64.0f));
        }
        return accumulated;
    };

    const FogIntegratedSample thin = integrateAtDensity(0.05f);
    const FogIntegratedSample thick = integrateAtDensity(5.0f);
    const FogIntegratedSample extreme = integrateAtDensity(500.0f);

    CHECK(thick.inScattering.r > thin.inScattering.r);
    // Saturates at the albedo rather than growing without bound.
    CHECK(extreme.inScattering.r <= 1.0f + 1.0e-3f);
    CHECK(extreme.inScattering.r == Approx(1.0f).margin(0.01f));
    CHECK(extreme.transmittance == Approx(0.0f).margin(1.0e-6f));
}

TEST_CASE("A near slice occludes the ones behind it", "[fog]")
{
    // Light from a far slice has to travel back through everything nearer, so
    // an opaque slice in front must suppress it. Getting this weighting wrong
    // makes fog glow through walls of denser fog.
    FogFroxelSample opaque{};
    opaque.extinction = 50.0f;

    FogFroxelSample emissive{};
    emissive.inScattering = glm::vec3{10.0f};
    emissive.extinction = 0.01f;

    const float thickness = fogSliceThickness(0, 64.0f);

    FogIntegratedSample blocked{};
    blocked = integrateFogSlice(blocked, opaque, thickness);
    blocked = integrateFogSlice(blocked, emissive, thickness);

    FogIntegratedSample unblocked{};
    unblocked = integrateFogSlice(unblocked, emissive, thickness);

    CHECK(blocked.inScattering.r < unblocked.inScattering.r * 0.5f);
}

TEST_CASE("The phase function conserves energy and points forward", "[fog]")
{
    const glm::vec3 view{0.0f, 0.0f, 1.0f};

    // Isotropic scattering is uniform in every direction.
    const float isotropicForward = henyeyGreensteinPhase(view, view, 0.0f);
    const float isotropicBack = henyeyGreensteinPhase(view, -view, 0.0f);
    CHECK(isotropicForward == Approx(isotropicBack));
    // ...and equals the 1/(4pi) normalisation.
    CHECK(isotropicForward == Approx(0.0795774715f).epsilon(0.001));

    // Forward anisotropy concentrates light along the view direction, which is
    // what gives the bright halo when looking toward a light through fog.
    const float forward = henyeyGreensteinPhase(view, view, 0.7f);
    const float sideways = henyeyGreensteinPhase(view, glm::vec3{1.0f, 0.0f, 0.0f}, 0.7f);
    const float backward = henyeyGreensteinPhase(view, -view, 0.7f);
    CHECK(forward > sideways);
    CHECK(sideways > backward);

    // Backward anisotropy mirrors it.
    CHECK(henyeyGreensteinPhase(view, -view, -0.7f) > henyeyGreensteinPhase(view, view, -0.7f));

    // Stays finite at the extremes, where the denominator would otherwise
    // collapse for one scattering angle.
    for (float g : {-1.0f, -0.999f, 0.999f, 1.0f, 5.0f, -5.0f}) {
        CHECK(std::isfinite(henyeyGreensteinPhase(view, view, g)));
        CHECK(std::isfinite(henyeyGreensteinPhase(view, -view, g)));
        CHECK(henyeyGreensteinPhase(view, view, g) >= 0.0f);
    }
}

TEST_CASE("Height fog falls off above its base", "[fog]")
{
    // A falloff of zero is uniform density at every height.
    CHECK(fogHeightDensity(0.0f, 0.0f, 0.0f) == Approx(1.0f));
    CHECK(fogHeightDensity(1000.0f, 0.0f, 0.0f) == Approx(1.0f));

    // At and below the base the medium is at full density, so a camera inside
    // the fog layer does not see it thin out downward.
    CHECK(fogHeightDensity(0.0f, 0.0f, 0.5f) == Approx(1.0f));
    CHECK(fogHeightDensity(-50.0f, 0.0f, 0.5f) == Approx(1.0f));

    // Above it, strictly decreasing and bounded.
    float previous = 1.0f;
    for (float height = 0.0f; height < 40.0f; height += 2.0f) {
        const float density = fogHeightDensity(height, 0.0f, 0.25f);
        CHECK(density <= previous);
        CHECK(density >= 0.0f);
        CHECK(density <= 1.0f);
        previous = density;
    }
    CHECK(previous < 0.05f);

    // The base height shifts the layer rather than reshaping it.
    CHECK(fogHeightDensity(12.0f, 10.0f, 0.3f) == Approx(fogHeightDensity(2.0f, 0.0f, 0.3f)));
}
