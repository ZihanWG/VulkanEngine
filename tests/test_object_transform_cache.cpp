#include "renderer/ObjectTransformCache.h"

#include <catch2/catch_test_macros.hpp>

#include <cstring>

using ve::renderer::Aabb;
using ve::renderer::CachedObjectTransform;
using ve::renderer::refreshCachedObjectTransform;
using ve::renderer::Transform;

namespace {

Transform sampleTransform()
{
    Transform transform{};
    transform.position = {1.5f, -2.0f, 3.25f};
    transform.rotationRadians = {0.3f, -1.1f, 2.4f};
    transform.scale = {2.0f, 0.5f, 1.25f};
    return transform;
}

Aabb sampleBounds()
{
    Aabb bounds{};
    bounds.min = {-1.0f, -0.5f, -2.0f};
    bounds.max = {1.0f, 0.75f, 2.0f};
    return bounds;
}

template <typename T> bool sameBits(const T& lhs, const T& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

// The contract: whatever the cache holds is bit-for-bit what composing afresh
// gives, which is what keeps the renderer's output byte-identical.
void requireFresh(const CachedObjectTransform& entry, const Transform& transform, const Aabb& localBounds)
{
    const glm::mat4 model = transform.modelMatrix();
    const Aabb worldBounds = localBounds.transform(model);
    REQUIRE(sameBits(entry.model, model));
    REQUIRE(sameBits(entry.worldBounds.min, worldBounds.min));
    REQUIRE(sameBits(entry.worldBounds.max, worldBounds.max));
}

} // namespace

TEST_CASE("Object transform cache computes on first use and reuses unchanged inputs", "[transform-cache]")
{
    CachedObjectTransform entry{};
    const Transform transform = sampleTransform();
    const Aabb bounds = sampleBounds();

    CHECK(refreshCachedObjectTransform(entry, transform, bounds));
    requireFresh(entry, transform, bounds);

    CHECK_FALSE(refreshCachedObjectTransform(entry, transform, bounds));
    requireFresh(entry, transform, bounds);
}

TEST_CASE("Object transform cache recomputes when any input changes", "[transform-cache]")
{
    const Transform base = sampleTransform();
    const Aabb bounds = sampleBounds();

    const auto changedAfter = [&](const Transform& transform, const Aabb& localBounds) {
        CachedObjectTransform entry{};
        REQUIRE(refreshCachedObjectTransform(entry, base, bounds));
        const bool recomputed = refreshCachedObjectTransform(entry, transform, localBounds);
        requireFresh(entry, transform, localBounds);
        return recomputed;
    };

    Transform moved = base;
    moved.position.y += 0.25f;
    CHECK(changedAfter(moved, bounds));

    Transform rotated = base;
    rotated.rotationRadians.z -= 0.5f;
    CHECK(changedAfter(rotated, bounds));

    Transform scaled = base;
    scaled.scale.x = 3.0f;
    CHECK(changedAfter(scaled, bounds));

    Transform overridden = base;
    overridden.useMatrixOverride = true;
    overridden.matrixOverride[3] = glm::vec4(7.0f, 8.0f, 9.0f, 1.0f);
    CHECK(changedAfter(overridden, bounds));

    Aabb grown = bounds;
    grown.max.z = 5.0f;
    CHECK(changedAfter(base, grown));

    // An object whose mesh went away reports empty local bounds, and its world
    // bounds must become empty too rather than keep the old box.
    CHECK(changedAfter(base, Aabb{}));
}

TEST_CASE("Object transform cache compares bits, so -0 and +0 are different keys", "[transform-cache]")
{
    // Both zeros compare equal with ==, but they can compose to matrices that
    // differ in a sign bit, so reusing one for the other would break the
    // bit-identical contract above.
    Transform positiveZero = sampleTransform();
    positiveZero.rotationRadians.x = 0.0f;
    Transform negativeZero = positiveZero;
    negativeZero.rotationRadians.x = -0.0f;

    CachedObjectTransform entry{};
    REQUIRE(refreshCachedObjectTransform(entry, positiveZero, sampleBounds()));
    CHECK(refreshCachedObjectTransform(entry, negativeZero, sampleBounds()));
    requireFresh(entry, negativeZero, sampleBounds());
}
