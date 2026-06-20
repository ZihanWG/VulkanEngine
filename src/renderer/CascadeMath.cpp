#include "renderer/CascadeMath.h"

#include <algorithm>
#include <cmath>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <limits>

namespace ve::renderer {

CascadeBuildOutput computeShadowCascades(const CascadeBuildInput& input)
{
    CascadeBuildOutput output{};

    const uint32_t cascadeCount = std::clamp(input.requestedCascadeCount, 1U, kMaxShadowCascades);
    const float nearPlane = std::max(0.001f, input.nearPlane);
    const float cameraFarPlane = std::max(nearPlane + 0.001f, input.farPlane);
    const float shadowFarPlane = std::clamp(input.shadowDistance, nearPlane + 0.001f, cameraFarPlane);
    const float lambda = std::clamp(input.lambda, 0.0f, 1.0f);

    const glm::vec3 cameraPosition = input.cameraPosition;
    const glm::vec3 cameraForward = glm::normalize(input.cameraTarget - input.cameraPosition);
    const glm::vec3 cameraRight = glm::normalize(glm::cross(cameraForward, input.cameraUp));
    const glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, cameraForward));
    const float tanHalfFov = std::tan(input.cameraVerticalFovRadians * 0.5f);

    const glm::vec3 lightDirection = glm::normalize(input.lightDirection);
    const glm::vec3 lightUp = std::abs(glm::dot(lightDirection, glm::vec3{0.0f, 1.0f, 0.0f})) > 0.95f
                                  ? glm::vec3{0.0f, 0.0f, 1.0f}
                                  : glm::vec3{0.0f, 1.0f, 0.0f};
    const glm::vec3 lightRight = glm::normalize(glm::cross(lightDirection, lightUp));
    const glm::vec3 lightBasisUp = glm::normalize(glm::cross(lightRight, lightDirection));

    output.splitDistances = glm::vec4(shadowFarPlane);

    float cascadeNear = nearPlane;
    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        const float splitRatio = static_cast<float>(cascadeIndex + 1) / static_cast<float>(cascadeCount);
        const float uniformSplit = nearPlane + (shadowFarPlane - nearPlane) * splitRatio;
        const float logSplit = nearPlane * std::pow(shadowFarPlane / nearPlane, splitRatio);
        const float cascadeFar = glm::mix(uniformSplit, logSplit, lambda);
        output.splitDistances[cascadeIndex] = cascadeFar;

        std::array<glm::vec3, 8> corners{};
        const auto writeDepthCorners = [&](float depth, uint32_t baseIndex) {
            const float halfHeight = tanHalfFov * depth;
            const float halfWidth = halfHeight * input.aspectRatio;
            const glm::vec3 center = cameraPosition + cameraForward * depth;

            corners[baseIndex + 0] = center - cameraRight * halfWidth - cameraUp * halfHeight;
            corners[baseIndex + 1] = center + cameraRight * halfWidth - cameraUp * halfHeight;
            corners[baseIndex + 2] = center - cameraRight * halfWidth + cameraUp * halfHeight;
            corners[baseIndex + 3] = center + cameraRight * halfWidth + cameraUp * halfHeight;
        };

        // Each cascade still starts from the readable fitted bounds of the
        // camera-frustum slice between cascadeNear and cascadeFar.
        writeDepthCorners(cascadeNear, 0);
        writeDepthCorners(cascadeFar, 4);

        glm::vec3 cascadeCenter{0.0f};
        for (const glm::vec3& corner : corners) {
            cascadeCenter += corner;
        }
        cascadeCenter /= static_cast<float>(corners.size());

        const glm::mat4 fitLightView = glm::lookAt(cascadeCenter - lightDirection, cascadeCenter, lightUp);
        glm::vec3 minBounds{std::numeric_limits<float>::infinity()};
        glm::vec3 maxBounds{-std::numeric_limits<float>::infinity()};
        for (const glm::vec3& corner : corners) {
            const glm::vec3 lightSpaceCorner = glm::vec3(fitLightView * glm::vec4(corner, 1.0f));
            minBounds = glm::min(minBounds, lightSpaceCorner);
            maxBounds = glm::max(maxBounds, lightSpaceCorner);
        }

        const float orthoWidth = std::max(maxBounds.x - minBounds.x, 0.001f);
        const float orthoHeight = std::max(maxBounds.y - minBounds.y, 0.001f);
        const float orthoExtent = std::max(orthoWidth, orthoHeight);
        const float shadowResolution = static_cast<float>(std::max(input.shadowResolution, 1U));
        const float worldUnitsPerTexel = orthoExtent / shadowResolution;

        glm::vec3 lightViewCenter = cascadeCenter;
        if (input.enableTexelSnapping && worldUnitsPerTexel > std::numeric_limits<float>::epsilon()) {
            // CSM shimmering happens when the camera moves by a sub-texel amount
            // in the light projection: static receivers then sample a slightly
            // different part of the shadow map every frame. Snapping the
            // light-view center to worldUnitsPerTexel increments keeps the
            // shadow texel grid from sliding continuously with the camera. This
            // is a basic stabilization step, not a full production CSM solution
            // with stable crop matrices, cascade blending, or per-cascade tuning.
            const float centerX = glm::dot(lightViewCenter, lightRight);
            const float centerY = glm::dot(lightViewCenter, lightBasisUp);
            const float snappedCenterX = std::round(centerX / worldUnitsPerTexel) * worldUnitsPerTexel;
            const float snappedCenterY = std::round(centerY / worldUnitsPerTexel) * worldUnitsPerTexel;
            lightViewCenter += lightRight * (snappedCenterX - centerX) + lightBasisUp * (snappedCenterY - centerY);

            // The orthographic bounds were fitted before moving the light view
            // by less than one shadow texel, so add a one-texel guard band to
            // preserve coverage after the quantized center shift.
            minBounds.x -= worldUnitsPerTexel;
            minBounds.y -= worldUnitsPerTexel;
            maxBounds.x += worldUnitsPerTexel;
            maxBounds.y += worldUnitsPerTexel;
        }

        const glm::mat4 lightView = glm::lookAt(lightViewCenter - lightDirection, lightViewCenter, lightUp);
        const float depthRange = std::max(cascadeFar - cascadeNear, 1.0f);
        const float zPadding = std::max(depthRange * 2.0f, 10.0f);
        const float orthoNear = std::max(0.001f, -maxBounds.z - zPadding);
        const float orthoFar = std::max(orthoNear + 0.001f, -minBounds.z + zPadding);

        glm::mat4 lightProjection =
            glm::ortho(minBounds.x, maxBounds.x, minBounds.y, maxBounds.y, orthoNear, orthoFar);
        lightProjection[1][1] *= -1.0f;

        ShadowCascade& cascade = output.cascades[cascadeIndex];
        cascade.lightViewProjection = lightProjection * lightView;
        cascade.lightFrustum = Frustum::fromViewProjection(cascade.lightViewProjection);
        cascade.splitDepth = cascadeFar;
        cascade.nearDepth = cascadeNear;
        cascade.farDepth = cascadeFar;

        cascadeNear = cascadeFar;
    }

    for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
        output.cascades[cascadeIndex] = output.cascades[cascadeCount - 1];
        output.splitDistances[cascadeIndex] = shadowFarPlane;
    }

    return output;
}

} // namespace ve::renderer
