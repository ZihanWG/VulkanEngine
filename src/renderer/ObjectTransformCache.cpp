#include "renderer/ObjectTransformCache.h"

#include <cstring>

namespace ve::renderer {

namespace {

// glm's vec3 and mat4 are tightly packed floats, so their bytes are their value.
static_assert(sizeof(glm::vec3) == 3 * sizeof(float));
static_assert(sizeof(glm::mat4) == 16 * sizeof(float));

template <typename T> bool sameBits(const T& lhs, const T& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

// Field by field: Transform has padding after its bool, and padding bytes are
// not part of the value.
bool sameTransform(const Transform& lhs, const Transform& rhs)
{
    return lhs.useMatrixOverride == rhs.useMatrixOverride && sameBits(lhs.position, rhs.position) &&
           sameBits(lhs.rotationRadians, rhs.rotationRadians) && sameBits(lhs.scale, rhs.scale) &&
           sameBits(lhs.matrixOverride, rhs.matrixOverride);
}

bool sameBounds(const Aabb& lhs, const Aabb& rhs)
{
    return sameBits(lhs.min, rhs.min) && sameBits(lhs.max, rhs.max);
}

} // namespace

bool refreshCachedObjectTransform(CachedObjectTransform& entry, const Transform& transform, const Aabb& localBounds)
{
    if (entry.valid && sameTransform(entry.transform, transform) && sameBounds(entry.localBounds, localBounds)) {
        return false;
    }

    entry.transform = transform;
    entry.localBounds = localBounds;
    entry.model = transform.modelMatrix();
    entry.worldBounds = localBounds.transform(entry.model);
    entry.valid = true;
    return true;
}

} // namespace ve::renderer
