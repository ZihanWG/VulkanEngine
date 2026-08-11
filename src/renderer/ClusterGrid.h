#pragma once

// GPU-independent froxel cluster math. These helpers are the testable reference
// for the clustered (Forward+) lighting compute/fragment shaders: cluster_build.comp
// builds the per-cluster AABBs, light_cull.comp assigns lights with the sphere/AABB
// test, and simple_bindless.frag resolves a pixel's cluster with clusterIndex().
// The grid constants below must stay in sync with the kCluster* constants declared
// in those GLSL sources (the shaders cannot include this header).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace ve::renderer {

inline constexpr uint32_t kClusterGridX = 32;
inline constexpr uint32_t kClusterGridY = 18;
inline constexpr uint32_t kClusterGridZ = 24;
inline constexpr uint32_t kClusterCount = kClusterGridX * kClusterGridY * kClusterGridZ;
inline constexpr uint32_t kMaxLightsPerCluster = 64;

// These four are mirrored in src/shaders/cluster_grid.glsl, which every shader
// that addresses the grid includes. Nothing checks the two files agree, so they
// have to be edited together -- a mismatch is silent, and shows up as fragments
// reading a different froxel's light list than they belong to.
//
// Grid resolution is the main lever on the per-fragment light loop (~64% of
// MainHDRPass by ablation): each XY tile covers drawable/kClusterGrid* pixels
// and every fragment in it iterates every light assigned to the tile. Finer is
// cheaper per fragment, but the light index list is preallocated at
// kClusterCount * kMaxLightsPerCluster entries rather than compacted, so memory
// grows linearly with the cluster count.

// View-space depth slice for a positive linear depth, using the same exponential
// distribution as the shaders: slice = log(z/zNear) / log(zFar/zNear) * gridZ.
[[nodiscard]] inline uint32_t clusterDepthSlice(float viewDepth, float zNear, float zFar)
{
    const float safeNear = std::max(zNear, 1.0e-4f);
    const float safeFar = std::max(zFar, safeNear + 1.0e-4f);
    const float depth = std::max(viewDepth, safeNear);
    float slice = std::log(depth / safeNear) / std::log(safeFar / safeNear) * static_cast<float>(kClusterGridZ);
    slice = std::clamp(slice, 0.0f, static_cast<float>(kClusterGridZ - 1));
    return static_cast<uint32_t>(slice);
}

// Linear cluster index for a framebuffer pixel + view depth (mirrors the fragment
// shader). screenWidth/Height are the swapchain extent in pixels.
[[nodiscard]] inline uint32_t clusterIndex(
    float fragX, float fragY, float viewDepth, float screenWidth, float screenHeight, float zNear, float zFar)
{
    const float tileSizeX = screenWidth / static_cast<float>(kClusterGridX);
    const float tileSizeY = screenHeight / static_cast<float>(kClusterGridY);
    const uint32_t tileX = std::min(static_cast<uint32_t>(fragX / tileSizeX), kClusterGridX - 1);
    const uint32_t tileY = std::min(static_cast<uint32_t>(fragY / tileSizeY), kClusterGridY - 1);
    const uint32_t slice = clusterDepthSlice(viewDepth, zNear, zFar);
    return tileX + tileY * kClusterGridX + slice * kClusterGridX * kClusterGridY;
}

// Squared distance from a point to an AABB (zero when inside), mirroring the
// light_cull.comp sphere/AABB test.
[[nodiscard]] inline float squaredDistancePointAabb(const glm::vec3& point,
                                                    const glm::vec3& boundsMin,
                                                    const glm::vec3& boundsMax)
{
    const glm::vec3 closest = glm::clamp(point, boundsMin, boundsMax);
    const glm::vec3 delta = point - closest;
    return glm::dot(delta, delta);
}

[[nodiscard]] inline bool sphereIntersectsAabb(const glm::vec3& center,
                                               float radius,
                                               const glm::vec3& boundsMin,
                                               const glm::vec3& boundsMax)
{
    return squaredDistancePointAabb(center, boundsMin, boundsMax) <= radius * radius;
}

// View-space AABB of one froxel, mirroring cluster_build.comp: intersect the tile
// corner rays (from the eye at the view-space origin) with the slice depth planes.
struct ClusterBounds {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
};

[[nodiscard]] inline ClusterBounds viewSpaceClusterBounds(
    const glm::mat4& inverseProjection, uint32_t tileX, uint32_t tileY, uint32_t slice, float zNear, float zFar)
{
    const auto unprojectToView = [&inverseProjection](float ndcX, float ndcY) {
        const glm::vec4 clip{ndcX, ndcY, 1.0f, 1.0f};
        const glm::vec4 view = inverseProjection * clip;
        return glm::vec3(view) / view.w;
    };

    const glm::vec2 ndcMin = glm::vec2(static_cast<float>(tileX) / static_cast<float>(kClusterGridX),
                                       static_cast<float>(tileY) / static_cast<float>(kClusterGridY)) *
                                 2.0f -
                             1.0f;
    const glm::vec2 ndcMax = glm::vec2(static_cast<float>(tileX + 1) / static_cast<float>(kClusterGridX),
                                       static_cast<float>(tileY + 1) / static_cast<float>(kClusterGridY)) *
                                 2.0f -
                             1.0f;

    const float safeNear = std::max(zNear, 1.0e-4f);
    const float safeFar = std::max(zFar, safeNear + 1.0e-4f);
    const float ratio = safeFar / safeNear;
    const float sliceNear = -safeNear * std::pow(ratio, static_cast<float>(slice) / static_cast<float>(kClusterGridZ));
    const float sliceFar = -safeNear * std::pow(ratio, static_cast<float>(slice + 1) / static_cast<float>(kClusterGridZ));

    const glm::vec2 corners[4] = {
        {ndcMin.x, ndcMin.y}, {ndcMax.x, ndcMin.y}, {ndcMin.x, ndcMax.y}, {ndcMax.x, ndcMax.y}};

    ClusterBounds bounds;
    bounds.min = glm::vec3(1.0e30f);
    bounds.max = glm::vec3(-1.0e30f);
    for (const glm::vec2& corner : corners) {
        const glm::vec3 direction = unprojectToView(corner.x, corner.y);
        if (std::abs(direction.z) < 1.0e-6f) {
            continue;
        }
        const glm::vec3 pointNear = direction * (sliceNear / direction.z);
        const glm::vec3 pointFar = direction * (sliceFar / direction.z);
        bounds.min = glm::min(bounds.min, glm::min(pointNear, pointFar));
        bounds.max = glm::max(bounds.max, glm::max(pointNear, pointFar));
    }
    return bounds;
}

} // namespace ve::renderer
