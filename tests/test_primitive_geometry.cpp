#include "renderer/PrimitiveGeometry.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

using ve::renderer::buildCubeGeometry;
using ve::renderer::buildUvSphereGeometry;
using ve::renderer::PrimitiveGeometry;
using ve::renderer::Vertex;

namespace {

// Signed volume by the divergence theorem: for a closed surface the sum of
// dot(v0, cross(v1, v2)) / 6 is the enclosed volume, positive when the winding
// is counter-clockwise seen from outside and negative when every triangle is
// reversed. It is the whole-mesh check -- one flipped triangle barely moves it,
// which is why the per-triangle test below exists as well.
[[nodiscard]] double signedVolume(const PrimitiveGeometry& geometry)
{
    double volume = 0.0;
    for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
        const glm::vec3& v0 = geometry.vertices[geometry.indices[i]].position;
        const glm::vec3& v1 = geometry.vertices[geometry.indices[i + 1]].position;
        const glm::vec3& v2 = geometry.vertices[geometry.indices[i + 2]].position;
        volume += static_cast<double>(glm::dot(v0, glm::cross(v1, v2)));
    }
    return volume / 6.0;
}

// Triangles whose geometric normal disagrees with the authored vertex normals.
//
// This is the check that actually pins the convention: the vertex normals say
// which way the surface faces, independently of the index order, so a triangle
// wound the wrong way is one where cross(v1 - v0, v2 - v0) points against them.
// Degenerate triangles -- the sphere's pole fans produce none, but a future
// primitive might -- are skipped rather than counted as failures.
[[nodiscard]] size_t trianglesFacingInward(const PrimitiveGeometry& geometry)
{
    size_t inward = 0;
    for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
        const Vertex& a = geometry.vertices[geometry.indices[i]];
        const Vertex& b = geometry.vertices[geometry.indices[i + 1]];
        const Vertex& c = geometry.vertices[geometry.indices[i + 2]];

        const glm::vec3 geometric = glm::cross(b.position - a.position, c.position - a.position);
        if (glm::length(geometric) < 1e-12f) {
            continue;
        }

        const glm::vec3 authored = a.normal + b.normal + c.normal;
        if (glm::dot(geometric, authored) <= 0.0f) {
            ++inward;
        }
    }
    return inward;
}

} // namespace

// The regression guard for the bug that motivated this file: every pipeline
// rasterised with VK_CULL_MODE_NONE, so an inverted mesh drew identically to a
// correct one and nothing could tell them apart. It only surfaced when
// back-face culling was measured on an immediate-mode GPU and the spheres
// rendered their own interiors.
TEST_CASE("built-in primitives are wound counter-clockwise when seen from outside",
          "[geometry][winding]")
{
    SECTION("cube")
    {
        const PrimitiveGeometry cube = buildCubeGeometry();
        REQUIRE(cube.indices.size() == 36);
        CHECK(trianglesFacingInward(cube) == 0);
        // A unit cube encloses exactly 1.
        CHECK_THAT(signedVolume(cube), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("uv sphere")
    {
        const PrimitiveGeometry sphere = buildUvSphereGeometry(48, 24);
        REQUIRE(!sphere.indices.empty());
        CHECK(trianglesFacingInward(sphere) == 0);

        // Diameter 1, so the exact volume is pi/6 ~= 0.5236. An inscribed
        // polyhedron under-reports it, and 48x24 lands within about 0.5%; the
        // point of the bound is the sign and the magnitude, since a reversed
        // winding gives the same number negated.
        const double volume = signedVolume(sphere);
        CHECK(volume > 0.0);
        CHECK_THAT(volume, Catch::Matchers::WithinRel(3.14159265358979 / 6.0, 0.01));
    }
}

// Guards the guard: if trianglesFacingInward could not see a reversal, the test
// above would pass on a broken mesh. Reversing every triangle must flip both
// signals, which is exactly the state the sphere shipped in.
TEST_CASE("the winding check detects a reversed mesh", "[geometry][winding]")
{
    PrimitiveGeometry reversed = buildUvSphereGeometry(16, 8);
    const size_t triangleCount = reversed.indices.size() / 3;
    REQUIRE(triangleCount > 0);
    REQUIRE(trianglesFacingInward(reversed) == 0);

    for (size_t i = 0; i + 2 < reversed.indices.size(); i += 3) {
        std::swap(reversed.indices[i + 1], reversed.indices[i + 2]);
    }

    CHECK(trianglesFacingInward(reversed) == triangleCount);
    CHECK(signedVolume(reversed) < 0.0);
}

TEST_CASE("uv sphere clamps degenerate segment and ring counts", "[geometry]")
{
    // Below the clamp the surface would not close, and signedVolume would stop
    // meaning anything.
    const PrimitiveGeometry tiny = buildUvSphereGeometry(0, 0);
    CHECK(tiny.vertices.size() == static_cast<size_t>(8 + 1) * static_cast<size_t>(4 + 1));
    CHECK(trianglesFacingInward(tiny) == 0);
    CHECK(signedVolume(tiny) > 0.0);
}
