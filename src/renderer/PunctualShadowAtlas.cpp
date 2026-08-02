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
    if (full()) {
        return kInvalidPunctualShadowSlot;
    }

    return nextSlot_++;
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
