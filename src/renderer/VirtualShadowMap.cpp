#include "renderer/VirtualShadowMap.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

namespace ve::renderer {
namespace {

// Floor division that stays correct for negative operands. The absolute page
// grid runs both ways from the light-space origin, and C++ integer division
// truncates toward zero, which would fold the pages on either side of zero onto
// the same coordinate.
[[nodiscard]] int32_t floorDiv(float value, float divisor)
{
    return static_cast<int32_t>(std::floor(value / divisor));
}

[[nodiscard]] uint32_t wrapToAxis(int32_t coordinate)
{
    const int32_t axis = static_cast<int32_t>(kVsmPagesPerLevelAxis);
    return static_cast<uint32_t>(((coordinate % axis) + axis) % axis);
}

} // namespace

VsmClipmapSettings clampVsmClipmapSettings(const VsmClipmapSettings& settings)
{
    VsmClipmapSettings clamped = settings;
    clamped.levelCount = std::clamp(settings.levelCount, 1u, kVsmMaxClipmapLevels);
    // The lower bound keeps the finest texel above float precision at world
    // scale; the upper bound is where level 0 is already coarser than a cascade.
    clamped.level0Extent = std::clamp(settings.level0Extent, 0.25f, 1024.0f);
    clamped.texelsPerPixel = std::clamp(settings.texelsPerPixel, 0.25f, 8.0f);
    clamped.depthRange = std::clamp(settings.depthRange, 1.0f, 100000.0f);
    if (!std::isfinite(clamped.level0Extent)) {
        clamped.level0Extent = 4.0f;
    }
    if (!std::isfinite(clamped.texelsPerPixel)) {
        clamped.texelsPerPixel = 1.0f;
    }
    if (!std::isfinite(clamped.depthRange)) {
        clamped.depthRange = 250.0f;
    }
    return clamped;
}

glm::mat4 vsmLightView(const glm::vec3& lightDirection)
{
    glm::vec3 direction = lightDirection;
    const float lengthSquared = glm::dot(direction, direction);
    if (!(lengthSquared > 1.0e-12f) || !std::isfinite(lengthSquared)) {
        direction = glm::vec3{0.0f, -1.0f, 0.0f};
    } else {
        direction = glm::normalize(direction);
    }

    // Same 0.95 threshold and same fallback axis as computeShadowCascades, so a
    // near-vertical sun produces the same basis for both shadow paths.
    const glm::vec3 up = std::abs(glm::dot(direction, glm::vec3{0.0f, 1.0f, 0.0f})) > 0.95f
                             ? glm::vec3{0.0f, 0.0f, 1.0f}
                             : glm::vec3{0.0f, 1.0f, 0.0f};

    // Eye at the world origin: the page grid is absolute, so the basis must not
    // translate with anything.
    return glm::lookAt(glm::vec3{0.0f}, direction, up);
}

float vsmPageWorldSize(const VsmClipmapSettings& settings, uint32_t level)
{
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    const uint32_t effectiveLevel = std::min(level, kVsmMaxClipmapLevels - 1u);
    const float levelExtent = clamped.level0Extent * static_cast<float>(1u << effectiveLevel);
    return levelExtent / static_cast<float>(kVsmPagesPerLevelAxis);
}

float vsmTexelWorldSize(const VsmClipmapSettings& settings, uint32_t level)
{
    return vsmPageWorldSize(settings, level) / static_cast<float>(kVsmPageSize);
}

uint32_t vsmMinLevelForCoverage(const VsmClipmapSettings& settings, float distanceToCamera)
{
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    const uint32_t maxLevel = clamped.levelCount - 1u;

    if (!(distanceToCamera > 0.0f) || !std::isfinite(distanceToCamera)) {
        return 0u;
    }

    // One page short of half the window: the camera sits somewhere inside its own
    // page rather than at its centre, so half the window is not all reachable.
    const float reachablePages = static_cast<float>(kVsmPagesPerLevelAxis / 2u) - 1.0f;
    const float level0Reach = reachablePages * vsmPageWorldSize(clamped, 0u);
    if (!(level0Reach > 0.0f) || distanceToCamera <= level0Reach) {
        return 0u;
    }

    // Each level doubles the reach, so the first level that reaches is the
    // ceiling of the log2 ratio -- the same relation the quality side uses.
    const float levels = std::ceil(std::log2(distanceToCamera / level0Reach));
    if (!(levels > 0.0f)) {
        return 0u;
    }
    if (levels >= static_cast<float>(maxLevel)) {
        return maxLevel;
    }
    return std::min(static_cast<uint32_t>(levels), maxLevel);
}

uint32_t vsmSelectLevel(const VsmClipmapSettings& settings, float distanceToCamera, float projScaleY)
{
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    const uint32_t maxLevel = clamped.levelCount - 1u;

    // Coverage is checked even when the quality side bails out: a degenerate
    // projScaleY must not silently return level 0 for a point kilometres away,
    // because level 0 has no slot that can hold it.
    const uint32_t coverageLevel = vsmMinLevelForCoverage(clamped, distanceToCamera);

    if (!(projScaleY > 0.0f) || !std::isfinite(projScaleY)) {
        return coverageLevel;
    }
    if (!(distanceToCamera > 0.0f) || !std::isfinite(distanceToCamera)) {
        return coverageLevel;
    }

    // World footprint of one screen pixel at this distance -- the same
    // radius/distance*projScaleY relation mesh LOD selection inverts.
    const float desiredTexelSize = distanceToCamera / projScaleY * clamped.texelsPerPixel;
    const float level0TexelSize = vsmTexelWorldSize(clamped, 0u);
    if (!(desiredTexelSize > level0TexelSize)) {
        return coverageLevel;
    }

    // Each level doubles the texel, so the level whose texel first covers the
    // request is the ceiling of the log2 ratio.
    const float levels = std::ceil(std::log2(desiredTexelSize / level0TexelSize));
    if (!(levels > 0.0f)) {
        return coverageLevel;
    }
    const uint32_t qualityLevel =
        levels >= static_cast<float>(maxLevel) ? maxLevel : std::min(static_cast<uint32_t>(levels), maxLevel);

    // A level finer than coverage allows has no slot for this point at all, so
    // quality can only ever ask for something coarser, never finer.
    return std::max(qualityLevel, coverageLevel);
}

glm::ivec2 vsmAbsolutePageCoords(const VsmClipmapSettings& settings, uint32_t level, const glm::vec2& lightSpaceXy)
{
    const float pageWorldSize = vsmPageWorldSize(settings, level);
    if (!std::isfinite(lightSpaceXy.x) || !std::isfinite(lightSpaceXy.y)) {
        return glm::ivec2{0, 0};
    }
    return glm::ivec2{floorDiv(lightSpaceXy.x, pageWorldSize), floorDiv(lightSpaceXy.y, pageWorldSize)};
}

glm::ivec2 vsmWindowOrigin(const VsmClipmapSettings& settings, uint32_t level, const glm::vec2& cameraLightSpaceXy)
{
    const glm::ivec2 cameraPage = vsmAbsolutePageCoords(settings, level, cameraLightSpaceXy);
    const int32_t halfAxis = static_cast<int32_t>(kVsmPagesPerLevelAxis / 2u);
    return cameraPage - glm::ivec2{halfAxis, halfAxis};
}

bool vsmPageInWindow(const glm::ivec2& absolutePage, const glm::ivec2& windowOrigin)
{
    const glm::ivec2 offset = absolutePage - windowOrigin;
    const int32_t axis = static_cast<int32_t>(kVsmPagesPerLevelAxis);
    return offset.x >= 0 && offset.x < axis && offset.y >= 0 && offset.y < axis;
}

uint32_t vsmSlotIndex(const glm::ivec2& absolutePage)
{
    const uint32_t x = wrapToAxis(absolutePage.x);
    const uint32_t y = wrapToAxis(absolutePage.y);
    return y * kVsmPagesPerLevelAxis + x;
}

uint32_t vsmPageId(uint32_t level, uint32_t slot)
{
    const uint32_t effectiveLevel = std::min(level, kVsmMaxClipmapLevels - 1u);
    const uint32_t effectiveSlot = std::min(slot, kVsmPagesPerLevel - 1u);
    return effectiveLevel * kVsmPagesPerLevel + effectiveSlot;
}

uint32_t vsmPageLevel(uint32_t pageId)
{
    return std::min(pageId, kVsmMaxVirtualPages - 1u) / kVsmPagesPerLevel;
}

uint32_t vsmPageSlot(uint32_t pageId)
{
    return std::min(pageId, kVsmMaxVirtualPages - 1u) % kVsmPagesPerLevel;
}

VsmPageRect vsmPagePoolRect(uint32_t physicalPage)
{
    const uint32_t page = std::min(physicalPage, kVsmPagePoolPageCount - 1u);
    VsmPageRect rect{};
    rect.x = (page % kVsmPagePoolPagesPerAxis) * kVsmPageSize;
    rect.y = (page / kVsmPagePoolPagesPerAxis) * kVsmPageSize;
    rect.size = kVsmPageSize;
    return rect;
}

glm::vec4 vsmPagePoolUvOffsetScale(const VsmPageRect& rect)
{
    const float poolSize = static_cast<float>(kVsmPagePoolSize);
    return glm::vec4{static_cast<float>(rect.x) / poolSize,
                     static_cast<float>(rect.y) / poolSize,
                     static_cast<float>(rect.size) / poolSize,
                     static_cast<float>(rect.size) / poolSize};
}

glm::mat4 vsmPageViewProjection(const VsmClipmapSettings& settings,
                                const glm::mat4& lightView,
                                uint32_t level,
                                const glm::ivec2& absolutePage)
{
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    const float pageWorldSize = vsmPageWorldSize(clamped, level);

    const float minX = static_cast<float>(absolutePage.x) * pageWorldSize;
    const float minY = static_cast<float>(absolutePage.y) * pageWorldSize;

    // zNear negative and zFar positive centres the depth range on the light-space
    // origin. With GLM_FORCE_DEPTH_ZERO_TO_ONE that maps the end of the range
    // nearest the light to 0 and the far end to 1, which is the normal-Z
    // convention the depth compare (LESS) and every other shadow projection use.
    glm::mat4 projection = glm::ortho(minX,
                                      minX + pageWorldSize,
                                      minY,
                                      minY + pageWorldSize,
                                      -clamped.depthRange,
                                      clamped.depthRange);
    // Vulkan's clip space has Y pointing the other way. The whole second row is
    // negated, not just the diagonal element: a page's rect is asymmetric about
    // the light-space origin, so glm::ortho leaves a non-zero translation in
    // [3][1] and flipping only [1][1] would offset the page by twice its centre
    // instead of mirroring it. (The cascades get away with flipping one element
    // because they use the same matrix to render and to look up, so any
    // consistent mapping works there.)
    projection[1][1] *= -1.0f;
    projection[3][1] *= -1.0f;

    return projection * lightView;
}

uint32_t vsmRequestWordIndex(uint32_t pageId)
{
    return std::min(pageId, kVsmMaxVirtualPages - 1u) / 32u;
}

uint32_t vsmRequestBitMask(uint32_t pageId)
{
    return 1u << (std::min(pageId, kVsmMaxVirtualPages - 1u) % 32u);
}

VsmPageRequestStats vsmDecodeRequestStats(const uint32_t* words, uint32_t levelCount)
{
    VsmPageRequestStats stats{};
    stats.lowestRequestedLevel = kVsmMaxClipmapLevels;
    if (words == nullptr) {
        stats.lowestRequestedLevel = 0;
        return stats;
    }

    const uint32_t activeLevels = std::clamp(levelCount, 1u, kVsmMaxClipmapLevels);
    const uint32_t activePages = activeLevels * kVsmPagesPerLevel;

    for (uint32_t pageId = 0; pageId < activePages; ++pageId) {
        if ((words[vsmRequestWordIndex(pageId)] & vsmRequestBitMask(pageId)) == 0u) {
            continue;
        }
        const uint32_t level = vsmPageLevel(pageId);
        ++stats.requestedPages;
        ++stats.requestedPerLevel[level];
        stats.highestRequestedLevel = std::max(stats.highestRequestedLevel, level);
        stats.lowestRequestedLevel = std::min(stats.lowestRequestedLevel, level);
    }

    if (stats.requestedPages == 0) {
        stats.lowestRequestedLevel = 0;
    }
    return stats;
}

} // namespace ve::renderer
