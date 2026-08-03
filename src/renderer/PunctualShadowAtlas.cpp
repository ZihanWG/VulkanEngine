#include "renderer/PunctualShadowAtlas.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>

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

uint64_t packShadowAtlasRect(const ShadowAtlasRect& rect)
{
    return (static_cast<uint64_t>(rect.x) << 32) | (static_cast<uint64_t>(rect.y) << 16) |
           static_cast<uint64_t>(rect.size);
}

void PunctualShadowCacheKey::addBytes(const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash_ ^= bytes[i];
        // FNV-1a 64-bit prime.
        hash_ *= 1099511628211ULL;
    }
}

uint32_t punctualShadowTileSizeForClass(uint32_t sizeClass)
{
    const uint32_t clamped = std::min(sizeClass, kPunctualShadowSizeClassCount - 1);
    return kPunctualShadowMaxTileSize >> clamped;
}

glm::vec4 shadowAtlasRectUvOffsetScale(const ShadowAtlasRect& rect)
{
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

float punctualShadowProjectedRadius(const glm::vec3& lightPosition,
                                    float range,
                                    const glm::vec3& cameraPosition,
                                    float projScaleY)
{
    const float distance = glm::length(lightPosition - cameraPosition);
    const float clampedRange = std::max(range, 0.0f);

    const float scale = std::max(projScaleY, 0.0f);

    // Inside its own radius the light wraps the camera, and range/distance stops
    // describing a footprint. Returning one saturated value for all of them
    // collapses the ranking exactly when a scene has many overlapping lights --
    // they tie, the tiebreak falls back to light index, and priority silently
    // degenerates into "whichever lights come first".
    //
    // Instead keep ranking them, by how deeply they enclose the camera. This is
    // continuous with the branch below (both give `scale` at distance == range),
    // so a light drifting across its own radius does not pop between classes.
    if (distance <= clampedRange) {
        const float enclosure = clampedRange > 0.0f ? (clampedRange - distance) / clampedRange : 1.0f;
        return scale * (1.0f + enclosure);
    }

    return clampedRange / distance * scale;
}

uint32_t punctualShadowSizeClassForRadius(float projectedRadius, bool isPoint)
{
    // A tile is enough when its edge covers the projected footprint, so each
    // class's threshold is its own size. Walk from largest down and take the
    // first that is not oversized for this light.
    uint32_t chosen = kPunctualShadowSizeClassCount - 1;
    for (uint32_t sizeClass = 0; sizeClass + 1 < kPunctualShadowSizeClassCount; ++sizeClass) {
        const auto tileSize = static_cast<float>(punctualShadowTileSizeForClass(sizeClass));
        if (projectedRadius >= tileSize) {
            chosen = sizeClass;
            break;
        }
    }

    if (isPoint) {
        chosen = std::min(chosen + 1, kPunctualShadowSizeClassCount - 1);
    }

    return chosen;
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

void PunctualShadowAtlasAllocator::reset()
{
    for (auto& tiles : freeTiles_) {
        tiles.clear();
    }
    allocatedCount_ = 0;
    allocatedArea_ = 0;

    // Seed with the atlas cut into largest-class tiles; every smaller tile is
    // produced by splitting one of these on demand.
    const uint32_t tileSize = punctualShadowTileSizeForClass(0);
    for (uint32_t y = 0; y < kPunctualShadowAtlasSize; y += tileSize) {
        for (uint32_t x = 0; x < kPunctualShadowAtlasSize; x += tileSize) {
            freeTiles_[0].push_back(ShadowAtlasRect{x, y, tileSize});
        }
    }
}

bool PunctualShadowAtlasAllocator::allocateNode(uint32_t sizeClass, ShadowAtlasRect& outRect)
{
    if (sizeClass >= kPunctualShadowSizeClassCount) {
        return false;
    }

    std::vector<ShadowAtlasRect>& tiles = freeTiles_[sizeClass];
    if (!tiles.empty()) {
        outRect = tiles.back();
        tiles.pop_back();
        return true;
    }

    // Nothing free at this size: split one class up. Recursion bottoms out at
    // class 0, which is only ever seeded by reset().
    if (sizeClass == 0) {
        return false;
    }

    ShadowAtlasRect parent{};
    if (!allocateNode(sizeClass - 1, parent)) {
        return false;
    }

    const uint32_t half = parent.size / 2;
    // Keep one quadrant, bank the other three for later allocations.
    freeTiles_[sizeClass].push_back(ShadowAtlasRect{parent.x + half, parent.y, half});
    freeTiles_[sizeClass].push_back(ShadowAtlasRect{parent.x, parent.y + half, half});
    freeTiles_[sizeClass].push_back(ShadowAtlasRect{parent.x + half, parent.y + half, half});

    outRect = ShadowAtlasRect{parent.x, parent.y, half};
    return true;
}

bool PunctualShadowAtlasAllocator::allocate(uint32_t sizeClass, ShadowAtlasRect& outRect)
{
    const uint32_t clamped = std::min(sizeClass, kPunctualShadowSizeClassCount - 1);
    if (!allocateNode(clamped, outRect)) {
        return false;
    }

    ++allocatedCount_;
    allocatedArea_ += static_cast<uint64_t>(outRect.size) * outRect.size;
    return true;
}

bool PunctualShadowAtlasAllocator::allocateRange(uint32_t sizeClass,
                                                 uint32_t count,
                                                 std::vector<ShadowAtlasRect>& outRects)
{
    outRects.clear();
    if (count == 0) {
        return false;
    }

    // All-or-nothing via a snapshot: a partially reserved cube would have the
    // light sampling cleared tiles for its missing faces, which reads as light
    // leaking through solid geometry -- worse than not casting at all.
    const auto savedTiles = freeTiles_;
    const uint32_t savedCount = allocatedCount_;
    const uint64_t savedArea = allocatedArea_;

    outRects.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        ShadowAtlasRect rect{};
        if (!allocate(sizeClass, rect)) {
            freeTiles_ = savedTiles;
            allocatedCount_ = savedCount;
            allocatedArea_ = savedArea;
            outRects.clear();
            return false;
        }
        outRects.push_back(rect);
    }

    return true;
}

float PunctualShadowAtlasAllocator::occupancy() const
{
    constexpr auto kAtlasArea =
        static_cast<double>(kPunctualShadowAtlasSize) * static_cast<double>(kPunctualShadowAtlasSize);
    return static_cast<float>(static_cast<double>(allocatedArea_) / kAtlasArea);
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
