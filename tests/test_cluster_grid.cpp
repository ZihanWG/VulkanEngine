#include "renderer/ClusterGrid.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using ve::renderer::clusterDepthSlice;
using ve::renderer::clusterIndex;
using ve::renderer::kClusterCount;
using ve::renderer::kClusterGridX;
using ve::renderer::kClusterGridY;
using ve::renderer::kClusterGridZ;
using ve::renderer::sphereIntersectsAabb;
using ve::renderer::viewSpaceClusterBounds;

namespace {

// Mirrors ve::renderer::Camera::projectionMatrix (GLM ZO depth + Vulkan y-flip).
glm::mat4 vulkanPerspective(float fovRadians, float aspect, float zNear, float zFar)
{
    glm::mat4 projection = glm::perspective(fovRadians, aspect, zNear, zFar);
    projection[1][1] *= -1.0f;
    return projection;
}

} // namespace

TEST_CASE("clusterDepthSlice spans the grid across the depth range", "[cluster]")
{
    constexpr float zNear = 0.1f;
    constexpr float zFar = 100.0f;

    CHECK(clusterDepthSlice(zNear, zNear, zFar) == 0u);
    CHECK(clusterDepthSlice(zNear * 0.5f, zNear, zFar) == 0u); // clamped below near
    CHECK(clusterDepthSlice(zFar, zNear, zFar) == kClusterGridZ - 1);
    CHECK(clusterDepthSlice(zFar * 2.0f, zNear, zFar) == kClusterGridZ - 1); // clamped past far

    // The exponential slicing is monotonically non-decreasing with depth.
    uint32_t previous = 0;
    for (float depth = zNear; depth <= zFar; depth *= 1.2f) {
        const uint32_t slice = clusterDepthSlice(depth, zNear, zFar);
        CHECK(slice >= previous);
        CHECK(slice < kClusterGridZ);
        previous = slice;
    }
}

TEST_CASE("clusterIndex maps screen tiles into the grid", "[cluster]")
{
    constexpr float screenW = 1600.0f;
    constexpr float screenH = 900.0f;
    constexpr float zNear = 0.1f;
    constexpr float zFar = 100.0f;

    // Top-left pixel at the near plane is cluster 0.
    CHECK(clusterIndex(0.0f, 0.0f, zNear, screenW, screenH, zNear, zFar) == 0u);

    // Every pixel/depth maps to a valid cluster, including the far bottom-right edge.
    for (float fx = 0.0f; fx <= screenW; fx += screenW / 4.0f) {
        for (float fy = 0.0f; fy <= screenH; fy += screenH / 4.0f) {
            for (float depth = zNear; depth <= zFar; depth *= 4.0f) {
                const uint32_t index = clusterIndex(fx, fy, depth, screenW, screenH, zNear, zFar);
                CHECK(index < kClusterCount);
            }
        }
    }

    // Pixels in the next horizontal tile advance the index by exactly one.
    const float tileSizeX = screenW / static_cast<float>(kClusterGridX);
    const uint32_t firstTile = clusterIndex(tileSizeX * 0.5f, 10.0f, zNear, screenW, screenH, zNear, zFar);
    const uint32_t secondTile = clusterIndex(tileSizeX * 1.5f, 10.0f, zNear, screenW, screenH, zNear, zFar);
    CHECK(secondTile == firstTile + 1u);
}

TEST_CASE("sphereIntersectsAabb matches the light-cull overlap test", "[cluster]")
{
    const glm::vec3 boundsMin{-1.0f, -1.0f, -1.0f};
    const glm::vec3 boundsMax{1.0f, 1.0f, 1.0f};

    CHECK(sphereIntersectsAabb(glm::vec3{0.0f}, 0.1f, boundsMin, boundsMax));         // center inside
    CHECK(sphereIntersectsAabb(glm::vec3{2.0f, 0.0f, 0.0f}, 1.5f, boundsMin, boundsMax)); // reaches the face
    CHECK_FALSE(sphereIntersectsAabb(glm::vec3{3.0f, 0.0f, 0.0f}, 1.5f, boundsMin, boundsMax)); // just short
    CHECK_FALSE(sphereIntersectsAabb(glm::vec3{5.0f, 5.0f, 5.0f}, 1.0f, boundsMin, boundsMax)); // far corner
    // Touching a corner counts as intersecting (distance == radius).
    CHECK(sphereIntersectsAabb(glm::vec3{2.0f, 2.0f, 2.0f}, std::sqrt(3.0f) + 1.0e-3f, boundsMin, boundsMax));
}

TEST_CASE("viewSpaceClusterBounds are ordered and grow with depth", "[cluster]")
{
    constexpr float zNear = 0.1f;
    constexpr float zFar = 100.0f;
    const glm::mat4 invProj = glm::inverse(vulkanPerspective(glm::radians(60.0f), 16.0f / 9.0f, zNear, zFar));

    const auto nearSlice = viewSpaceClusterBounds(invProj, 8, 4, 0, zNear, zFar);
    const auto farSlice = viewSpaceClusterBounds(invProj, 8, 4, kClusterGridZ - 1, zNear, zFar);

    CHECK(nearSlice.min.x <= nearSlice.max.x);
    CHECK(nearSlice.min.y <= nearSlice.max.y);
    CHECK(nearSlice.min.z <= nearSlice.max.z);

    // View-space z is negative; deeper slices sit further from the camera.
    CHECK(farSlice.max.z < nearSlice.min.z);
}

TEST_CASE("a froxel center round-trips back to its own cluster", "[cluster]")
{
    constexpr float screenW = 1600.0f;
    constexpr float screenH = 900.0f;
    constexpr float zNear = 0.1f;
    constexpr float zFar = 100.0f;
    const glm::mat4 projection = vulkanPerspective(glm::radians(60.0f), 16.0f / 9.0f, zNear, zFar);
    const glm::mat4 invProj = glm::inverse(projection);

    // For interior froxels, the AABB centre projects back to the same cluster.
    for (uint32_t tileX : {3u, 8u, 12u}) {
        for (uint32_t tileY : {2u, 4u, 6u}) {
            for (uint32_t slice : {2u, 10u, 18u}) {
                const auto bounds = viewSpaceClusterBounds(invProj, tileX, tileY, slice, zNear, zFar);
                const glm::vec3 center = 0.5f * (bounds.min + bounds.max);

                const glm::vec4 clip = projection * glm::vec4(center, 1.0f);
                REQUIRE(clip.w > 0.0f);
                const glm::vec3 ndc = glm::vec3(clip) / clip.w;
                const float fragX = (ndc.x * 0.5f + 0.5f) * screenW;
                const float fragY = (ndc.y * 0.5f + 0.5f) * screenH;
                const float viewDepth = -center.z;

                const uint32_t expected = tileX + tileY * kClusterGridX + slice * kClusterGridX * kClusterGridY;
                CHECK(clusterIndex(fragX, fragY, viewDepth, screenW, screenH, zNear, zFar) == expected);
            }
        }
    }
}
