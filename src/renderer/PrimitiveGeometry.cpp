#include "renderer/PrimitiveGeometry.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace ve::renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;

const std::array<Vertex, 24> kCubeVertices = {{
    // Front (+Z)
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},

    // Back (-Z)
    {{0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},

    // Left (-X)
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 0.95f, 1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.95f, 0.95f, 1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.95f, 0.95f, 1.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.95f, 0.95f, 1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},

    // Right (+X)
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.9f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.9f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.9f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 0.9f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},

    // Top (+Y)
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 1.0f}, {0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 1.0f}, {1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.95f, 1.0f}, {1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.95f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},

    // Bottom (-Y)
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.95f, 1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.95f, 1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}
}};

const std::array<uint32_t, 36> kCubeIndices = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
};

} // namespace

PrimitiveGeometry buildCubeGeometry()
{
    PrimitiveGeometry geometry;
    geometry.vertices.assign(kCubeVertices.begin(), kCubeVertices.end());
    geometry.indices.assign(kCubeIndices.begin(), kCubeIndices.end());
    return geometry;
}

PrimitiveGeometry buildUvSphereGeometry(uint32_t segments, uint32_t rings)
{
    segments = std::max(segments, 8U);
    rings = std::max(rings, 4U);

    PrimitiveGeometry geometry;
    geometry.vertices.reserve(static_cast<size_t>(segments + 1U) * static_cast<size_t>(rings + 1U));
    geometry.indices.reserve(static_cast<size_t>(segments) * static_cast<size_t>(rings - 1U) * 6U);

    for (uint32_t ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / static_cast<float>(rings);
        const float phi = v * kPi;
        const float y = std::cos(phi) * 0.5f;
        const float radius = std::sin(phi) * 0.5f;

        for (uint32_t segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / static_cast<float>(segments);
            const float theta = u * kPi * 2.0f;
            const glm::vec3 normal = glm::normalize(glm::vec3{
                std::cos(theta) * radius,
                y,
                std::sin(theta) * radius,
            });
            const glm::vec3 tangent = glm::normalize(glm::vec3{-std::sin(theta), 0.0f, std::cos(theta)});

            Vertex vertex{};
            vertex.position = normal * 0.5f;
            vertex.color = glm::vec3(1.0f);
            vertex.uv = {u, 1.0f - v};
            vertex.normal = normal;
            vertex.tangent = {tangent, 1.0f};
            geometry.vertices.push_back(vertex);
        }
    }

    const uint32_t rowStride = segments + 1U;
    for (uint32_t ring = 0; ring < rings; ++ring) {
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const uint32_t a = ring * rowStride + segment;
            const uint32_t b = (ring + 1U) * rowStride + segment;
            const uint32_t c = b + 1U;
            const uint32_t d = a + 1U;

            // Winding is (a, d, b) and (d, c, b), not (a, b, d) and (d, b, c).
            //
            // Ring index increases from the north pole downward, so b sits
            // *below* a, while d sits one segment around from a in the +theta
            // direction. Near the equator at theta = 0 the outward direction is
            // +X, and cross(b - a, d - a) = cross(-dY, +dZ) points along -X --
            // into the sphere. Swapping the last two indices flips both
            // triangles to face out, which is what
            // VK_FRONT_FACE_COUNTER_CLOCKWISE requires.
            //
            // The original order was inward, and VK_CULL_MODE_NONE hid it until
            // back-face culling was measured on an immediate-mode GPU. The
            // regression guard is the normal-agreement check in
            // tests/test_primitive_geometry.cpp, not this comment.
            if (ring > 0) {
                geometry.indices.push_back(a);
                geometry.indices.push_back(d);
                geometry.indices.push_back(b);
            }
            if (ring + 1U < rings) {
                geometry.indices.push_back(d);
                geometry.indices.push_back(c);
                geometry.indices.push_back(b);
            }
        }
    }

    return geometry;
}

} // namespace ve::renderer
