#include "renderer/ClusterGrid.h"
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

TEST_CASE("The phase bound is never exceeded by the phase itself", "[fog]")
{
    using ve::renderer::maxHenyeyGreensteinPhase;

    // This is the property the fog's per-light culling rests on. The cull skips
    // a light when colour * intensity * attenuation * bound is negligible, so if
    // the bound could ever understate the real phase, a visible light would be
    // dropped -- and it would be dropped only at certain viewing angles, which
    // is the kind of bug that looks like flickering rather than like a cull.
    const glm::vec3 view{0.0f, 0.0f, 1.0f};

    for (float g = -0.98f; g <= 0.98f; g += 0.04f) {
        const float bound = maxHenyeyGreensteinPhase(g);
        CHECK(std::isfinite(bound));
        CHECK(bound > 0.0f);

        // Sweep the full scattering sphere, not just the axes: the peak of a
        // strongly anisotropic phase is narrow, so a coarse check could miss it.
        for (int step = 0; step <= 360; ++step) {
            const float theta = static_cast<float>(step) * 3.14159265f / 180.0f;
            const glm::vec3 lightDirection{std::sin(theta), 0.0f, std::cos(theta)};
            const float phase = henyeyGreensteinPhase(view, lightDirection, g);
            CHECK(phase <= bound * (1.0f + 1.0e-4f));
        }

        // And it is tight, not just safe: an infinite bound would make the cull
        // never fire, which is a silent performance regression rather than a
        // visual bug.
        const glm::vec3 peakDirection = g >= 0.0f ? view : -view;
        CHECK(henyeyGreensteinPhase(view, peakDirection, g) == Approx(bound).epsilon(0.001));
    }

    // Isotropic scattering bounds to the uniform value.
    CHECK(maxHenyeyGreensteinPhase(0.0f) == Approx(0.0795774715f).epsilon(0.001));
    // Out-of-range anisotropy clamps the same way the phase function does, so
    // the two cannot disagree about which anisotropy is in effect.
    CHECK(maxHenyeyGreensteinPhase(5.0f) == Approx(maxHenyeyGreensteinPhase(0.99f)));
    CHECK(maxHenyeyGreensteinPhase(-5.0f) == Approx(maxHenyeyGreensteinPhase(-0.99f)));
    CHECK(std::isfinite(maxHenyeyGreensteinPhase(1.0f)));
    CHECK(std::isfinite(maxHenyeyGreensteinPhase(-1.0f)));
}

TEST_CASE("A fog froxel maps to the cluster a fragment there would use", "[fog]")
{
    using ve::renderer::clusterIndex;
    using ve::renderer::fogFroxelClusterIndex;
    using ve::renderer::kClusterCount;
    using ve::renderer::kFogGridX;
    using ve::renderer::kFogGridY;

    constexpr float fogMaxDistance = 64.0f;
    constexpr float zNear = 0.1f;
    constexpr float zFar = 100.0f;

    // The invariant that matters: the fog volume and the light grid are
    // different resolutions with different depth ranges, and the only thing
    // making a shared light list correct is that a froxel resolves to the same
    // cluster a fragment at that screen position and depth would.
    size_t checked = 0;
    for (uint32_t x = 0; x < kFogGridX; x += 7) {
        for (uint32_t y = 0; y < kFogGridY; y += 5) {
            for (uint32_t z = 0; z < kFogGridZ; z += 3) {
                const uint32_t fogCluster =
                    fogFroxelClusterIndex(x, y, z, fogMaxDistance, zNear, zFar);
                REQUIRE(fogCluster < kClusterCount);

                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kFogGridX);
                const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kFogGridY);
                const float viewDepth = fogSliceViewDepth(static_cast<float>(z) + 0.5f, fogMaxDistance);

                // Same query, expressed the way the fragment shader asks it.
                CHECK(fogCluster == clusterIndex(u, v, viewDepth, 1.0f, 1.0f, zNear, zFar));
                ++checked;
            }
        }
    }
    CHECK(checked > 3000);

    // Neighbouring froxels stay in the same cluster or an adjacent one -- the
    // fog volume is finer than the light grid in every axis, so the mapping
    // must never skip clusters.
    for (uint32_t x = 0; x + 1 < kFogGridX; ++x) {
        const uint32_t left = fogFroxelClusterIndex(x, 40, 20, fogMaxDistance, zNear, zFar);
        const uint32_t right = fogFroxelClusterIndex(x + 1, 40, 20, fogMaxDistance, zNear, zFar);
        CHECK(right >= left);
        CHECK(right - left <= 1u);
    }

    // Corners land inside the grid rather than off either end.
    CHECK(fogFroxelClusterIndex(0, 0, 0, fogMaxDistance, zNear, zFar) < kClusterCount);
    CHECK(fogFroxelClusterIndex(kFogGridX - 1, kFogGridY - 1, kFogGridZ - 1, fogMaxDistance, zNear, zFar) <
          kClusterCount);
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
