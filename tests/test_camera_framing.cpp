// Framing a camera on scene bounds. A fetched scene has no idea what the shipped
// portfolio preset was positioned for, so this is what stops a real asset from
// rendering as the inside of a column.

#include "renderer/Camera.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using ve::renderer::Aabb;
using ve::renderer::Camera;
using ve::renderer::framedCamera;

namespace {

Aabb boxAround(glm::vec3 center, float halfExtent)
{
    Aabb bounds{};
    bounds.expand(center - glm::vec3(halfExtent));
    bounds.expand(center + glm::vec3(halfExtent));
    return bounds;
}

// True when every corner of the box projects inside the clip volume.
bool boxFitsInFrame(const Camera& camera, const Aabb& bounds, float aspectRatio)
{
    const glm::mat4 viewProjection = camera.viewProjectionMatrix(aspectRatio);
    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 point{
            (corner & 1) ? bounds.max.x : bounds.min.x,
            (corner & 2) ? bounds.max.y : bounds.min.y,
            (corner & 4) ? bounds.max.z : bounds.min.z,
        };
        const glm::vec4 clip = viewProjection * glm::vec4(point, 1.0f);
        if (clip.w <= 0.0f) {
            return false;
        }
        if (std::abs(clip.x) > clip.w || std::abs(clip.y) > clip.w) {
            return false;
        }
        if (clip.z < 0.0f || clip.z > clip.w) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("An invalid bounds yields a usable camera rather than NaNs")
{
    const Camera camera = framedCamera(Aabb{}, 16.0f / 9.0f, glm::vec3(0.0f, 0.0f, -1.0f));

    REQUIRE(std::isfinite(camera.position.x));
    REQUIRE(std::isfinite(camera.position.y));
    REQUIRE(std::isfinite(camera.position.z));
    REQUIRE(camera.nearPlane > 0.0f);
    REQUIRE(camera.farPlane > camera.nearPlane);
}

TEST_CASE("A degenerate zero-size bounds does not divide by zero")
{
    const Aabb point = boxAround(glm::vec3(3.0f, 4.0f, 5.0f), 0.0f);

    const Camera camera = framedCamera(point, 16.0f / 9.0f, glm::vec3(0.0f, 0.0f, -1.0f));

    REQUIRE(std::isfinite(camera.position.x));
    REQUIRE(camera.target == glm::vec3(3.0f, 4.0f, 5.0f));
    REQUIRE(camera.nearPlane > 0.0f);
}

TEST_CASE("The camera looks at the centre of the bounds")
{
    const Aabb bounds = boxAround(glm::vec3(10.0f, -2.0f, 7.0f), 4.0f);

    const Camera camera = framedCamera(bounds, 16.0f / 9.0f, glm::vec3(1.0f, 0.0f, 0.0f));

    REQUIRE(camera.target.x == Approx(10.0f));
    REQUIRE(camera.target.y == Approx(-2.0f));
    REQUIRE(camera.target.z == Approx(7.0f));
    // Looking along +X means standing on the -X side of the centre.
    REQUIRE(camera.position.x < camera.target.x);
}

TEST_CASE("The whole box lands inside the frame")
{
    const Aabb bounds = boxAround(glm::vec3(0.0f), 5.0f);

    for (const float aspect : {16.0f / 9.0f, 1.0f, 0.5f}) {
        const Camera camera = framedCamera(bounds, aspect, glm::vec3(0.3f, -0.4f, -1.0f));
        INFO("aspect " << aspect);
        REQUIRE(boxFitsInFrame(camera, bounds, aspect));
    }
}

TEST_CASE("A narrow window pulls the camera back rather than cropping")
{
    const Aabb bounds = boxAround(glm::vec3(0.0f), 5.0f);
    const glm::vec3 direction{0.0f, 0.0f, -1.0f};

    const Camera wide = framedCamera(bounds, 16.0f / 9.0f, direction);
    const Camera narrow = framedCamera(bounds, 0.5f, direction);

    // Horizontal field of view is the tighter axis on a tall window, so the
    // camera has to retreat further. Ignoring that crops the sides.
    REQUIRE(glm::length(narrow.position) > glm::length(wide.position));
    REQUIRE(boxFitsInFrame(narrow, bounds, 0.5f));
}

TEST_CASE("Distance scales with the size of the scene")
{
    const glm::vec3 direction{0.0f, 0.0f, -1.0f};
    const Camera small = framedCamera(boxAround(glm::vec3(0.0f), 1.0f), 16.0f / 9.0f, direction);
    const Camera large = framedCamera(boxAround(glm::vec3(0.0f), 100.0f), 16.0f / 9.0f, direction);

    REQUIRE(glm::length(large.position) == Approx(glm::length(small.position) * 100.0f));

    // Clip planes have to follow, or a scene orders of magnitude bigger than the
    // sphere showcase is entirely beyond the far plane.
    REQUIRE(large.farPlane > small.farPlane);
    REQUIRE(large.nearPlane > small.nearPlane);
}

TEST_CASE("A lower fill fraction leaves margin around the scene")
{
    const Aabb bounds = boxAround(glm::vec3(0.0f), 5.0f);
    const glm::vec3 direction{0.0f, 0.0f, -1.0f};

    const Camera tight = framedCamera(bounds, 16.0f / 9.0f, direction, glm::radians(60.0f), 1.0f);
    const Camera loose = framedCamera(bounds, 16.0f / 9.0f, direction, glm::radians(60.0f), 0.5f);

    REQUIRE(glm::length(loose.position) > glm::length(tight.position));
}

TEST_CASE("An unnormalised direction is handled")
{
    const Aabb bounds = boxAround(glm::vec3(0.0f), 5.0f);

    const Camera unit = framedCamera(bounds, 16.0f / 9.0f, glm::vec3(0.0f, 0.0f, -1.0f));
    const Camera scaled = framedCamera(bounds, 16.0f / 9.0f, glm::vec3(0.0f, 0.0f, -37.0f));

    REQUIRE(scaled.position.z == Approx(unit.position.z));
}
