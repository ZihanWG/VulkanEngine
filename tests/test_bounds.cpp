#include "renderer/Bounds.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using ve::renderer::Aabb;
using ve::renderer::clipVolumeBounds;
using ve::renderer::Frustum;
using ve::renderer::Ray;
using ve::renderer::Sphere;

namespace {
Aabb unitBox()
{
    Aabb box;
    box.min = glm::vec3(-1.0f);
    box.max = glm::vec3(1.0f);
    return box;
}
} // namespace

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

TEST_CASE("Ray hits an AABB it points at and reports entry distance", "[bounds][ray]")
{
    const Aabb box = unitBox();

    SECTION("ray from outside toward the box hits at the near face")
    {
        Ray ray;
        ray.origin = glm::vec3(0.0f, 0.0f, 5.0f);
        ray.direction = glm::vec3(0.0f, 0.0f, -1.0f);
        float distance = -1.0f;
        REQUIRE(box.intersectRay(ray, distance));
        CHECK(distance == Catch::Approx(4.0f)); // 5 -> near face at z=1
    }

    SECTION("ray starting inside the box reports zero distance")
    {
        Ray ray;
        ray.origin = glm::vec3(0.0f);
        ray.direction = glm::vec3(1.0f, 0.0f, 0.0f);
        float distance = -1.0f;
        REQUIRE(box.intersectRay(ray, distance));
        CHECK(distance == Catch::Approx(0.0f));
    }

    SECTION("diagonal ray still hits")
    {
        Ray ray;
        ray.origin = glm::vec3(5.0f, 5.0f, 5.0f);
        ray.direction = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
        float distance = -1.0f;
        CHECK(box.intersectRay(ray, distance));
    }
}

TEST_CASE("Ray misses an AABB it points away from or beside", "[bounds][ray]")
{
    const Aabb box = unitBox();

    SECTION("ray pointing away from the box misses")
    {
        Ray ray;
        ray.origin = glm::vec3(0.0f, 0.0f, 5.0f);
        ray.direction = glm::vec3(0.0f, 0.0f, 1.0f); // away from the box
        float distance = -1.0f;
        CHECK_FALSE(box.intersectRay(ray, distance));
    }

    SECTION("parallel ray offset beside the box misses")
    {
        Ray ray;
        ray.origin = glm::vec3(5.0f, 5.0f, 5.0f);
        ray.direction = glm::vec3(0.0f, 0.0f, -1.0f); // parallel to Z slab but x,y outside
        float distance = -1.0f;
        CHECK_FALSE(box.intersectRay(ray, distance));
    }
}

TEST_CASE("Ray picks the nearest of two boxes by entry distance", "[bounds][ray]")
{
    Ray ray;
    ray.origin = glm::vec3(0.0f, 0.0f, 10.0f);
    ray.direction = glm::vec3(0.0f, 0.0f, -1.0f);

    Aabb near;
    near.min = glm::vec3(-1.0f, -1.0f, 2.0f);
    near.max = glm::vec3(1.0f, 1.0f, 4.0f);
    Aabb far;
    far.min = glm::vec3(-1.0f, -1.0f, -4.0f);
    far.max = glm::vec3(1.0f, 1.0f, -2.0f);

    float nearDistance = -1.0f;
    float farDistance = -1.0f;
    REQUIRE(near.intersectRay(ray, nearDistance));
    REQUIRE(far.intersectRay(ray, farDistance));
    CHECK(nearDistance < farDistance); // the closer box is selected when picking
    CHECK(nearDistance == Catch::Approx(6.0f));
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

namespace {

// A spot-light-like frustum: 70 degree cone looking down -Z from (2, 1, 3),
// near plane at a fiftieth of an 8-unit range, as the atlas builds them.
glm::mat4 spotViewProjection()
{
    const glm::vec3 position(2.0f, 1.0f, 3.0f);
    const glm::mat4 view = glm::lookAt(position, position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1.0f, 8.0f / 50.0f, 8.0f);
    return projection * view;
}

Aabb boxAt(const glm::vec3& center, float halfExtent)
{
    Aabb box;
    box.min = center - glm::vec3(halfExtent);
    box.max = center + glm::vec3(halfExtent);
    return box;
}

} // namespace

TEST_CASE("Aabb::overlaps is a closed-interval test on every axis", "[bounds]")
{
    const Aabb box = unitBox();
    CHECK(box.overlaps(boxAt(glm::vec3(0.5f), 1.0f)));
    CHECK(box.overlaps(boxAt(glm::vec3(2.0f, 0.0f, 0.0f), 1.0f))); // touching faces
    CHECK_FALSE(box.overlaps(boxAt(glm::vec3(2.5f, 0.0f, 0.0f), 1.0f)));
    CHECK_FALSE(box.overlaps(boxAt(glm::vec3(0.0f, -2.5f, 0.0f), 1.0f)));
    CHECK_FALSE(box.overlaps(boxAt(glm::vec3(0.0f, 0.0f, 2.5f), 1.0f)));
}

TEST_CASE("clipVolumeBounds contains every point the frustum sees", "[bounds]")
{
    // The pre-cull built on it may only reject what the frustum cannot see, so
    // sample the clip volume densely -- corners and faces included -- and
    // require every point back in world space inside the box.
    const glm::mat4 viewProjection = spotViewProjection();
    const glm::mat4 inverse = glm::inverse(viewProjection);
    const Aabb bounds = clipVolumeBounds(viewProjection, 1.0e-3f);
    REQUIRE(bounds.valid());

    constexpr int kSteps = 8;
    for (int ix = 0; ix <= kSteps; ++ix) {
        for (int iy = 0; iy <= kSteps; ++iy) {
            for (int iz = 0; iz <= kSteps; ++iz) {
                const glm::vec4 clip(-1.0f + 2.0f * static_cast<float>(ix) / kSteps,
                                     -1.0f + 2.0f * static_cast<float>(iy) / kSteps,
                                     static_cast<float>(iz) / kSteps,
                                     1.0f);
                const glm::vec4 world = inverse * clip;
                const glm::vec3 point = glm::vec3(world) / world.w;
                REQUIRE(bounds.overlaps(boxAt(point, 0.0f)));
            }
        }
    }

    // And it is a useful box, not an unbounded one: the far plane is 8 units
    // out, so nothing on the volume is much farther than that from the apex.
    CHECK(bounds.max.z == Catch::Approx(3.0f).margin(0.2f));
    CHECK(bounds.min.z > 3.0f - 8.0f - 0.2f);
}

TEST_CASE("Boxes clipVolumeBounds rejects are outside the frustum too", "[bounds]")
{
    const glm::mat4 viewProjection = spotViewProjection();
    const Aabb bounds = clipVolumeBounds(viewProjection, 1.0e-3f);
    const Frustum frustum = Frustum::fromViewProjection(viewProjection);

    // Behind the light and past its range: both rejected by the box, and the
    // plane test agrees, so the pre-cull never removes a visible caster.
    for (const glm::vec3& center :
         {glm::vec3(2.0f, 1.0f, 6.0f), glm::vec3(2.0f, 1.0f, -9.0f), glm::vec3(30.0f, 1.0f, 0.0f)}) {
        const Aabb caster = boxAt(center, 0.5f);
        CHECK_FALSE(bounds.overlaps(caster));
        CHECK_FALSE(frustum.testAabb(caster));
    }
    // In front of the light, inside the range: kept by both.
    const Aabb visible = boxAt(glm::vec3(2.0f, 1.0f, -1.0f), 0.5f);
    CHECK(bounds.overlaps(visible));
    CHECK(frustum.testAabb(visible));
}

TEST_CASE("clipVolumeBounds is empty for a matrix it cannot invert", "[bounds]")
{
    // Empty means "no information" to the caller, which then falls back to the
    // plane test alone.
    CHECK_FALSE(clipVolumeBounds(glm::mat4(0.0f), 1.0e-3f).valid());
}
