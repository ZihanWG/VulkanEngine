#include "renderer/VolumetricFog.h"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

namespace ve::renderer {

namespace {

// Below this the medium is optically thin enough that the analytic slab
// integral degenerates into scatter * thickness, and dividing by it would lose
// precision rather than gain it.
constexpr float kMinExtinction = 1.0e-4f;

[[nodiscard]] float safeMaxDistance(float maxDistance)
{
    return std::max(maxDistance, kFogNearPlane + 1.0e-3f);
}

} // namespace

float fogSliceViewDepth(float slice, float maxDistance)
{
    // Exponential distribution, matching the clustered light grid's shape so a
    // fog froxel and the light cluster covering it agree on where depth sits.
    const float far = safeMaxDistance(maxDistance);
    const float normalized = std::clamp(slice / static_cast<float>(kFogGridZ), 0.0f, 1.0f);
    return kFogNearPlane * std::pow(far / kFogNearPlane, normalized);
}

float fogViewDepthToSlice(float viewDepth, float maxDistance)
{
    const float far = safeMaxDistance(maxDistance);
    const float depth = std::clamp(viewDepth, kFogNearPlane, far);
    const float normalized = std::log(depth / kFogNearPlane) / std::log(far / kFogNearPlane);
    return std::clamp(normalized, 0.0f, 1.0f) * static_cast<float>(kFogGridZ);
}

float fogSliceThickness(uint32_t slice, float maxDistance)
{
    const float nearDepth = fogSliceViewDepth(static_cast<float>(slice), maxDistance);
    const float farDepth = fogSliceViewDepth(static_cast<float>(slice + 1), maxDistance);
    return std::max(farDepth - nearDepth, 0.0f);
}

FogIntegratedSample integrateFogSlice(const FogIntegratedSample& accumulated,
                                      const FogFroxelSample& slice,
                                      float sliceThickness)
{
    const float thickness = std::max(sliceThickness, 0.0f);
    const float extinction = std::max(slice.extinction, 0.0f);

    const float sliceTransmittance = std::exp(-extinction * thickness);

    // Analytic integral of in-scattering across a homogeneous slab. The naive
    // `inScattering * thickness` ignores that light scattered in near the back
    // of the slice is partly absorbed before it leaves, and overestimates badly
    // once the slice is optically thick.
    const glm::vec3 sliceIntegral = extinction > kMinExtinction
                                        ? slice.inScattering * ((1.0f - sliceTransmittance) / extinction)
                                        : slice.inScattering * thickness;

    FogIntegratedSample result{};
    // Weighted by the transmittance accumulated *before* this slice: light from
    // here still has to travel back through everything nearer the eye.
    result.inScattering = accumulated.inScattering + accumulated.transmittance * sliceIntegral;
    result.transmittance = accumulated.transmittance * sliceTransmittance;
    return result;
}

float henyeyGreensteinPhase(const glm::vec3& viewDirection, const glm::vec3& lightDirection, float anisotropy)
{
    // Clamped strictly inside (-1, 1): at either endpoint the denominator
    // collapses to zero for one scattering angle and the phase blows up.
    const float g = std::clamp(anisotropy, -0.99f, 0.99f);
    const float cosTheta = glm::dot(viewDirection, lightDirection);

    const float gSquared = g * g;
    const float denominator = 1.0f + gSquared - 2.0f * g * cosTheta;
    // The 1/(4pi) normalisation makes the phase integrate to 1 over the sphere,
    // so changing anisotropy redistributes light rather than adding it.
    constexpr float kInverseFourPi = 0.0795774715f;

    return kInverseFourPi * (1.0f - gSquared) / std::max(std::pow(denominator, 1.5f), 1.0e-6f);
}

float fogHeightDensity(float worldHeight, float baseHeight, float falloff)
{
    if (falloff <= 0.0f) {
        return 1.0f;
    }

    // Full density at and below the base height, decaying exponentially above.
    return std::exp(-std::max(worldHeight - baseHeight, 0.0f) * falloff);
}

} // namespace ve::renderer
