#include "renderer/PunctualShadowAtlas.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace ve::renderer {

namespace {

// lookAt degenerates when the view direction is parallel to the up vector, which
// is exactly the common case for a spot light aimed straight down. Swap to a
// different reference axis when the direction gets close to vertical.
[[nodiscard]] glm::vec3 shadowUpVector(const glm::vec3& normalizedDirection)
{
    constexpr float kParallelThreshold = 0.999f;
    if (std::abs(normalizedDirection.y) > kParallelThreshold) {
        return glm::vec3{0.0f, 0.0f, 1.0f};
    }

    return glm::vec3{0.0f, 1.0f, 0.0f};
}

} // namespace

ShadowAtlasRect punctualShadowSlotRect(uint32_t slot)
{
    if (slot >= kMaxPunctualShadowSlots) {
        return {};
    }

    ShadowAtlasRect rect{};
    rect.x = (slot % kPunctualShadowTilesPerSide) * kPunctualShadowTileSize;
    rect.y = (slot / kPunctualShadowTilesPerSide) * kPunctualShadowTileSize;
    rect.size = kPunctualShadowTileSize;
    return rect;
}

glm::vec4 punctualShadowSlotUvOffsetScale(uint32_t slot)
{
    const ShadowAtlasRect rect = punctualShadowSlotRect(slot);
    if (rect.size == 0) {
        return glm::vec4{0.0f};
    }

    constexpr float kAtlasSize = static_cast<float>(kPunctualShadowAtlasSize);
    const float extent = static_cast<float>(rect.size) / kAtlasSize;
    return glm::vec4{static_cast<float>(rect.x) / kAtlasSize,
                     static_cast<float>(rect.y) / kAtlasSize,
                     extent,
                     extent};
}

float punctualShadowSlotToFloat(uint32_t slot)
{
    if (slot >= kMaxPunctualShadowSlots) {
        return -1.0f;
    }

    return static_cast<float>(slot);
}

uint32_t punctualShadowSlotFromFloat(float encoded)
{
    // Written as a negated compare so a NaN slot decodes to "unshadowed" rather
    // than sliding through into an out-of-bounds slot fetch.
    if (!(encoded >= 0.0f)) {
        return kInvalidPunctualShadowSlot;
    }

    const auto slot = static_cast<uint32_t>(std::lround(encoded));
    if (slot >= kMaxPunctualShadowSlots) {
        return kInvalidPunctualShadowSlot;
    }

    return slot;
}

float punctualShadowNearPlane(float range)
{
    return std::max(kMinPunctualShadowNearPlane, std::max(range, 0.0f) * kPunctualShadowNearPlaneRangeFraction);
}

uint32_t PunctualShadowAtlasAllocator::allocate()
{
    return allocateRange(1);
}

uint32_t PunctualShadowAtlasAllocator::allocateRange(uint32_t count)
{
    if (count == 0 || count > kMaxPunctualShadowSlots || nextSlot_ + count > kMaxPunctualShadowSlots) {
        return kInvalidPunctualShadowSlot;
    }

    const uint32_t first = nextSlot_;
    nextSlot_ += count;
    return first;
}

glm::vec3 pointShadowFaceDirection(uint32_t face)
{
    switch (face) {
    case 0:
        return glm::vec3{1.0f, 0.0f, 0.0f};
    case 1:
        return glm::vec3{-1.0f, 0.0f, 0.0f};
    case 2:
        return glm::vec3{0.0f, 1.0f, 0.0f};
    case 3:
        return glm::vec3{0.0f, -1.0f, 0.0f};
    case 4:
        return glm::vec3{0.0f, 0.0f, 1.0f};
    default:
        return glm::vec3{0.0f, 0.0f, -1.0f};
    }
}

uint32_t pointShadowFaceIndex(const glm::vec3& direction)
{
    // Ties are broken toward the earlier axis, and the shader's mirror of this
    // uses the same >= comparisons so a direction exactly on a face boundary
    // resolves identically on both sides.
    const glm::vec3 magnitude = glm::abs(direction);
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
        return direction.x >= 0.0f ? 0u : 1u;
    }
    if (magnitude.y >= magnitude.z) {
        return direction.y >= 0.0f ? 2u : 3u;
    }

    return direction.z >= 0.0f ? 4u : 5u;
}

glm::mat4 computePointShadowFaceViewProjection(const glm::vec3& position,
                                               uint32_t face,
                                               float range,
                                               float nearPlane)
{
    const glm::vec3 faceDirection = pointShadowFaceDirection(face % kPointShadowFaceCount);

    const float requestedNear = nearPlane > 0.0f ? nearPlane : punctualShadowNearPlane(range);
    const float clampedNear = std::max(requestedNear, 1.0e-3f);
    const float clampedFar = std::max(range, clampedNear + 1.0e-3f);

    const glm::mat4 view = glm::lookAt(position, position + faceDirection, shadowUpVector(faceDirection));
    // Exactly 90 degrees, so the six faces tile the sphere without gaps.
    const glm::mat4 projection =
        glm::perspective(glm::radians(90.0f), 1.0f, clampedNear, clampedFar);

    return projection * view;
}

glm::mat4 computeSpotShadowViewProjection(const glm::vec3& position,
                                          const glm::vec3& direction,
                                          float outerAngleRadians,
                                          float range,
                                          float nearPlane)
{
    const float directionLength = glm::length(direction);
    const glm::vec3 normalizedDirection =
        directionLength > 0.0f ? direction / directionLength : glm::vec3{0.0f, -1.0f, 0.0f};

    // A non-positive nearPlane means "derive it from the range".
    const float requestedNear = nearPlane > 0.0f ? nearPlane : punctualShadowNearPlane(range);
    const float clampedNear = std::max(requestedNear, 1.0e-3f);
    const float clampedFar = std::max(range, clampedNear + 1.0e-3f);
    const float halfAngle = std::clamp(outerAngleRadians, 1.0e-3f, kMaxSpotShadowHalfAngleRadians);

    const glm::mat4 view =
        glm::lookAt(position, position + normalizedDirection, shadowUpVector(normalizedDirection));
    // Square tile, so the aspect ratio is always 1.
    const glm::mat4 projection = glm::perspective(halfAngle * 2.0f, 1.0f, clampedNear, clampedFar);

    return projection * view;
}

Frustum computeSpotShadowFrustum(const glm::mat4& viewProjection)
{
    return Frustum::fromViewProjection(viewProjection);
}

} // namespace ve::renderer
