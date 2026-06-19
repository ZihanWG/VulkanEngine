#include "renderer/Bounds.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using ve::renderer::Aabb;
using ve::renderer::Frustum;
using ve::renderer::Sphere;

TEST_CASE("Aabb::transform translates the box", "[bounds]")
{
    Aabb box;
    box.min = glm::vec3(-1.0f);
    box.max = glm::vec3(1.0f);

    const Aabb moved = box.transform(glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 0.0f, 0.0f)));

    CHECK(moved.min.x == Catch::Approx(9.0f));
    CHECK(moved.max.x == Catch::Approx(11.0f));
    CHECK(moved.min.y == Catch::Approx(-1.0f));
    CHECK(moved.max.y == Catch::Approx(1.0f));
}

TEST_CASE("Aabb::transform keeps a symmetric cube symmetric under rotation", "[bounds]")
{
    Aabb box;
    box.min = glm::vec3(-1.0f);
    box.max = glm::vec3(1.0f);

    const glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const Aabb rotated = box.transform(rotation);

    // A unit cube centred on the origin maps onto itself (up to floating point).
    CHECK(rotated.min.x == Catch::Approx(-1.0f).margin(1e-5));
    CHECK(rotated.max.x == Catch::Approx(1.0f).margin(1e-5));
    CHECK(rotated.min.z == Catch::Approx(-1.0f).margin(1e-5));
    CHECK(rotated.max.z == Catch::Approx(1.0f).margin(1e-5));
}

TEST_CASE("Invalid (default) Aabb is treated as always visible", "[bounds]")
{
    const Aabb invalid; // min = +inf, max = -inf
    CHECK_FALSE(invalid.valid());

    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    const Frustum frustum = Frustum::fromViewProjection(proj * view);

    // Conservative behaviour: an unbounded object should never be culled.
    CHECK(frustum.testAabb(invalid));
}

TEST_CASE("Frustum culls spheres outside the view volume", "[bounds][frustum]")
{
    // Camera at +Z looking toward the origin (down -Z), Vulkan 0..1 depth.
    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    const Frustum frustum = Frustum::fromViewProjection(proj * view);

    SECTION("sphere at the origin is visible")
    {
        CHECK(frustum.testSphere(Sphere{glm::vec3(0.0f), 1.0f}));
    }

    SECTION("sphere behind the camera is culled")
    {
        // +Z is behind the camera, so this is outside the near/far volume.
        CHECK_FALSE(frustum.testSphere(Sphere{glm::vec3(0.0f, 0.0f, 50.0f), 1.0f}));
    }

    SECTION("sphere far to the side is culled")
    {
        CHECK_FALSE(frustum.testSphere(Sphere{glm::vec3(100.0f, 0.0f, 0.0f), 1.0f}));
    }

    SECTION("large sphere straddling the frustum edge stays visible")
    {
        // Centre is off to the side but the radius reaches back into the volume.
        CHECK(frustum.testSphere(Sphere{glm::vec3(3.0f, 0.0f, 0.0f), 5.0f}));
    }
}

TEST_CASE("Aabb frustum test matches sphere expectations", "[bounds][frustum]")
{
    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    const Frustum frustum = Frustum::fromViewProjection(proj * view);

    Aabb inside;
    inside.min = glm::vec3(-1.0f);
    inside.max = glm::vec3(1.0f);
    CHECK(frustum.testAabb(inside));

    Aabb behind;
    behind.min = glm::vec3(-1.0f, -1.0f, 49.0f);
    behind.max = glm::vec3(1.0f, 1.0f, 51.0f);
    CHECK_FALSE(frustum.testAabb(behind));
}
