#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <limits>
#include <utility>

namespace ve::renderer {

struct Ray {
    glm::vec3 origin{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f}; // expected to be normalized
};

struct Aabb {
    glm::vec3 min{std::numeric_limits<float>::infinity()};
    glm::vec3 max{-std::numeric_limits<float>::infinity()};

    [[nodiscard]] bool valid() const
    {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }

    void expand(const glm::vec3& point)
    {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }

    void merge(const Aabb& other)
    {
        if (!other.valid()) {
            return;
        }

        expand(other.min);
        expand(other.max);
    }

    [[nodiscard]] Aabb transform(const glm::mat4& matrix) const
    {
        Aabb result{};
        if (!valid()) {
            return result;
        }

        const std::array<glm::vec3, 8> corners = {{
            {min.x, min.y, min.z},
            {max.x, min.y, min.z},
            {min.x, max.y, min.z},
            {max.x, max.y, min.z},
            {min.x, min.y, max.z},
            {max.x, min.y, max.z},
            {min.x, max.y, max.z},
            {max.x, max.y, max.z},
        }};

        for (const glm::vec3& corner : corners) {
            const glm::vec4 transformed = matrix * glm::vec4(corner, 1.0f);
            result.expand(glm::vec3(transformed));
        }

        return result;
    }

    // Slab-method ray/AABB intersection. On a hit, outDistance is the distance
    // along the ray to the entry point (0 if the ray starts inside the box).
    // The ray direction is assumed normalized; hits behind the origin are ignored.
    [[nodiscard]] bool intersectRay(const Ray& ray, float& outDistance) const
    {
        if (!valid()) {
            return false;
        }

        float tMin = -std::numeric_limits<float>::infinity();
        float tMax = std::numeric_limits<float>::infinity();

        for (int axis = 0; axis < 3; ++axis) {
            const float origin = ray.origin[axis];
            const float direction = ray.direction[axis];
            const float slabMin = min[axis];
            const float slabMax = max[axis];

            if (std::abs(direction) < 1e-8f) {
                // Ray is parallel to this slab; it misses unless the origin is inside it.
                if (origin < slabMin || origin > slabMax) {
                    return false;
                }
                continue;
            }

            const float inverse = 1.0f / direction;
            float tEnter = (slabMin - origin) * inverse;
            float tExit = (slabMax - origin) * inverse;
            if (tEnter > tExit) {
                std::swap(tEnter, tExit);
            }

            tMin = tEnter > tMin ? tEnter : tMin;
            tMax = tExit < tMax ? tExit : tMax;
            if (tMin > tMax) {
                return false;
            }
        }

        // The box is entirely behind the ray origin.
        if (tMax < 0.0f) {
            return false;
        }

        outDistance = tMin >= 0.0f ? tMin : 0.0f;
        return true;
    }
};

struct Sphere {
    glm::vec3 center{0.0f};
    float radius = 0.0f;
};

struct FrustumPlane {
    glm::vec3 normal{0.0f};
    float distance = 0.0f;

    [[nodiscard]] float signedDistance(const glm::vec3& point) const
    {
        return glm::dot(normal, point) + distance;
    }

    void normalize()
    {
        const float length = glm::length(normal);
        if (length <= 0.0f || !std::isfinite(length)) {
            return;
        }

        normal /= length;
        distance /= length;
    }
};

struct Frustum {
    enum PlaneIndex : size_t {
        Left = 0,
        Right,
        Bottom,
        Top,
        Near,
        Far,
    };

    std::array<FrustumPlane, 6> planes{};

    [[nodiscard]] static Frustum fromViewProjection(const glm::mat4& viewProjection)
    {
        const auto makePlane = [](const glm::vec4& equation) {
            FrustumPlane plane{};
            plane.normal = glm::vec3(equation);
            plane.distance = equation.w;
            plane.normalize();
            return plane;
        };

        // GLM stores matrices by column and indexes as matrix[column][row].
        // Frustum extraction is easier to read as row combinations, so rebuild
        // the four rows explicitly from the projection * view matrix.
        const glm::vec4 row0{
            viewProjection[0][0], viewProjection[1][0], viewProjection[2][0], viewProjection[3][0]};
        const glm::vec4 row1{
            viewProjection[0][1], viewProjection[1][1], viewProjection[2][1], viewProjection[3][1]};
        const glm::vec4 row2{
            viewProjection[0][2], viewProjection[1][2], viewProjection[2][2], viewProjection[3][2]};
        const glm::vec4 row3{
            viewProjection[0][3], viewProjection[1][3], viewProjection[2][3], viewProjection[3][3]};

        Frustum frustum{};
        // Vulkan clip space with GLM_FORCE_DEPTH_ZERO_TO_ONE is:
        // -w <= x <= w, -w <= y <= w, and 0 <= z <= w.
        // Each plane is oriented so points inside the frustum have
        // dot(normal, point) + distance >= 0.
        frustum.planes[Left] = makePlane(row3 + row0);
        frustum.planes[Right] = makePlane(row3 - row0);
        frustum.planes[Bottom] = makePlane(row3 + row1);
        frustum.planes[Top] = makePlane(row3 - row1);
        frustum.planes[Near] = makePlane(row2);
        frustum.planes[Far] = makePlane(row3 - row2);
        return frustum;
    }

    [[nodiscard]] bool testAabb(const Aabb& bounds) const
    {
        if (!bounds.valid()) {
            return true;
        }

        for (const FrustumPlane& plane : planes) {
            const glm::vec3 positiveVertex{
                plane.normal.x >= 0.0f ? bounds.max.x : bounds.min.x,
                plane.normal.y >= 0.0f ? bounds.max.y : bounds.min.y,
                plane.normal.z >= 0.0f ? bounds.max.z : bounds.min.z,
            };

            if (plane.signedDistance(positiveVertex) < 0.0f) {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] bool testSphere(const Sphere& sphere) const
    {
        for (const FrustumPlane& plane : planes) {
            if (plane.signedDistance(sphere.center) < -sphere.radius) {
                return false;
            }
        }

        return true;
    }
};

} // namespace ve::renderer
