#include "renderer/CascadeMath.h"

#include <algorithm>
#include <cmath>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <limits>

namespace ve::renderer {

namespace {

// Fits one cascade to the bounding sphere of its frustum slice.
//
// Three things have to be invariant for a cascade matrix to reproduce bit for
// bit across frames, and the AABB fit gets none of them right:
//
//  * the ortho extent. Here it is the sphere radius, which depends only on the
//    slice's near/far and the camera's fov/aspect -- not on where the camera is
//    pointing. The AABB fit measures rotated corners, so it breathes as the
//    camera turns.
//  * the snapping grid. It follows from the extent, so a breathing extent means
//    a grid that never lands on the same lattice twice.
//  * the depth range. Deriving it from rotated bounds makes it breathe too, so
//    it is pinned to the radius instead, which leaves the projection matrix
//    fully constant for a given cascade. Only the view translates, and it does
//    so in whole-texel steps because the centre is snapped on all three light
//    axes rather than only the two lateral ones.
//
// The radius is measured from the corners rather than derived in closed form:
// the closed form needs a separate case for a centre that falls beyond the far
// plane (reachable at wide fields of view), and this is called four times a
// frame.
ShadowCascade fitStableCascade(const std::array<glm::vec3, 8>& corners,
                               const glm::vec3& sliceCenter,
                               const glm::vec3& lightDirection,
                               const glm::vec3& lightUp,
                               const glm::vec3& lightRight,
                               const glm::vec3& lightBasisUp,
                               uint32_t shadowResolution,
                               float cascadeNear,
                               float cascadeFar)
{
    float radius = 0.0f;
    for (const glm::vec3& corner : corners) {
        radius = std::max(radius, glm::length(corner - sliceCenter));
    }
    // A zero radius would divide by zero below and produce a degenerate
    // projection; a hair of extent costs nothing and keeps the matrix finite.
    radius = std::max(radius, 0.001f);

    const float resolution = static_cast<float>(std::max(shadowResolution, 1U));
    const float worldUnitsPerTexel = (radius * 2.0f) / resolution;

    // Snapped on all three light axes. The lateral two are the classic
    // anti-shimmer snap; the third is what keeps the near/far planes -- and so
    // the whole view matrix -- from sliding continuously as the camera moves
    // toward or away from the light.
    const float centerX = std::round(glm::dot(sliceCenter, lightRight) / worldUnitsPerTexel) * worldUnitsPerTexel;
    const float centerY = std::round(glm::dot(sliceCenter, lightBasisUp) / worldUnitsPerTexel) * worldUnitsPerTexel;
    const float centerZ = std::round(glm::dot(sliceCenter, lightDirection) / worldUnitsPerTexel) * worldUnitsPerTexel;
    const glm::vec3 snappedCenter = lightRight * centerX + lightBasisUp * centerY + lightDirection * centerZ;

    // How far back along the light the view sits, and therefore how much space
    // there is for casters between the light and the sphere. Constant for a
    // given cascade, because it is a function of the radius alone -- a
    // scene-dependent term here would put the projection back in play.
    const float casterExtent = std::max(radius * 4.0f, 10.0f);

    const glm::mat4 lightView =
        glm::lookAt(snappedCenter - lightDirection * casterExtent, snappedCenter, lightUp);

    glm::mat4 lightProjection = glm::ortho(-radius, radius, -radius, radius, 0.0f, casterExtent + radius);
    lightProjection[1][1] *= -1.0f;

    ShadowCascade cascade{};
    cascade.lightViewProjection = lightProjection * lightView;
    cascade.lightFrustum = Frustum::fromViewProjection(cascade.lightViewProjection);
    cascade.splitDepth = cascadeFar;
    cascade.nearDepth = cascadeNear;
    cascade.farDepth = cascadeFar;
    return cascade;
}

} // namespace

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

        if (input.enableStableFit) {
            output.cascades[cascadeIndex] = fitStableCascade(corners,
                                                             cascadeCenter,
                                                             lightDirection,
                                                             lightUp,
                                                             lightRight,
                                                             lightBasisUp,
                                                             input.shadowResolution,
                                                             cascadeNear,
                                                             cascadeFar);
            cascadeNear = cascadeFar;
            continue;
        }

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
