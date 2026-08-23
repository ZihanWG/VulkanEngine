#include "renderer/VirtualShadowMap.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

glm::ivec2 vsmAbsolutePageForSlot(const glm::ivec2& windowOrigin, uint32_t slot)
{
    const int32_t axis = static_cast<int32_t>(kVsmPagesPerLevelAxis);
    const int32_t slotX = static_cast<int32_t>(slot % kVsmPagesPerLevelAxis);
    const int32_t slotY = static_cast<int32_t>((slot / kVsmPagesPerLevelAxis) % kVsmPagesPerLevelAxis);

    // The unique page in [origin, origin + axis) congruent to the slot mod axis.
    const int32_t offsetX = ((slotX - windowOrigin.x) % axis + axis) % axis;
    const int32_t offsetY = ((slotY - windowOrigin.y) % axis + axis) % axis;
    return windowOrigin + glm::ivec2{offsetX, offsetY};
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

VsmPageAllocator::VsmPageAllocator()
{
    entries_.assign(kVsmMaxVirtualPages, VsmPageTableEntry{});
    physicalOwner_.assign(kVsmPagePoolPageCount, kVsmInvalidPhysicalPage);
    physicalLastUsed_.assign(kVsmPagePoolPageCount, 0);
}

void VsmPageAllocator::reset()
{
    entries_.assign(kVsmMaxVirtualPages, VsmPageTableEntry{});
    physicalOwner_.assign(kVsmPagePoolPageCount, kVsmInvalidPhysicalPage);
    physicalLastUsed_.assign(kVsmPagePoolPageCount, 0);
    frameCounter_ = 0;
    residentPages_ = 0;
    evictionsThisFrame_ = 0;
}

void VsmPageAllocator::beginFrame(uint64_t frameCounter)
{
    frameCounter_ = frameCounter;
    evictionsThisFrame_ = 0;
}

uint32_t VsmPageAllocator::allocatePhysicalPage()
{
    // Free page first. Scanning is fine: the pool is 1024 entries and this runs
    // only for pages that were not already resident, which the measurement puts
    // at a handful per frame once the clipmap has warmed up.
    for (uint32_t physical = 0; physical < physicalOwner_.size(); ++physical) {
        if (physicalOwner_[physical] == kVsmInvalidPhysicalPage) {
            return physical;
        }
    }

    // Full: evict the least recently used page that was not acquired this frame.
    // Excluding this frame's own pages is what stops a frame whose request set
    // exceeds the pool from evicting pages it is still using and thrashing.
    uint32_t victim = kVsmInvalidPhysicalPage;
    uint64_t oldest = 0;
    for (uint32_t physical = 0; physical < physicalOwner_.size(); ++physical) {
        if (physicalLastUsed_[physical] >= frameCounter_) {
            continue;
        }
        if (victim == kVsmInvalidPhysicalPage || physicalLastUsed_[physical] < oldest) {
            victim = physical;
            oldest = physicalLastUsed_[physical];
        }
    }
    return victim;
}

VsmPageAcquireResult VsmPageAllocator::acquire(uint32_t pageId, const glm::ivec2& absolutePage)
{
    VsmPageAcquireResult result{};
    if (pageId >= entries_.size()) {
        return result;
    }

    VsmPageTableEntry& entry = entries_[pageId];

    // Already bound. Two cases: same absolute page (a real hit), or the window
    // scrolled and this slot now names somewhere else. The second keeps the
    // physical page -- it belongs to this slot either way -- but its depth is
    // for the old location, so it drops back to unrendered.
    if (entry.physicalPage != kVsmInvalidPhysicalPage && entry.physicalPage < physicalOwner_.size() &&
        physicalOwner_[entry.physicalPage] == pageId) {
        physicalLastUsed_[entry.physicalPage] = frameCounter_;
        if (entry.absoluteX == absolutePage.x && entry.absoluteY == absolutePage.y) {
            result.physicalPage = entry.physicalPage;
            result.alreadyRendered = entry.rendered != 0;
            return result;
        }

        entry.absoluteX = absolutePage.x;
        entry.absoluteY = absolutePage.y;
        entry.rendered = 0;
        result.physicalPage = entry.physicalPage;
        return result;
    }

    const uint32_t physical = allocatePhysicalPage();
    if (physical == kVsmInvalidPhysicalPage) {
        // Nothing evictable: every physical page is spoken for by this same
        // frame. The caller sees an invalid page and leaves this one to the
        // coarser-level fallback rather than stealing depth still in use.
        return result;
    }

    const uint32_t previousOwner = physicalOwner_[physical];
    if (previousOwner != kVsmInvalidPhysicalPage && previousOwner < entries_.size()) {
        entries_[previousOwner] = VsmPageTableEntry{};
        ++evictionsThisFrame_;
        result.evicted = true;
        --residentPages_;
    }

    physicalOwner_[physical] = pageId;
    physicalLastUsed_[physical] = frameCounter_;
    entry.physicalPage = physical;
    entry.absoluteX = absolutePage.x;
    entry.absoluteY = absolutePage.y;
    entry.rendered = 0;
    ++residentPages_;

    result.physicalPage = physical;
    return result;
}

bool VsmPageAllocator::invalidate(uint32_t pageId)
{
    if (pageId >= entries_.size()) {
        return false;
    }

    VsmPageTableEntry& entry = entries_[pageId];
    // Only pages that actually hold depth count. An unrendered page is already
    // queued, and a page owning no physical space has nothing to drop -- both
    // would otherwise inflate the invalidation counter with no-ops.
    if (entry.physicalPage == kVsmInvalidPhysicalPage || entry.rendered == 0) {
        return false;
    }

    entry.rendered = 0;
    return true;
}

void VsmPageAllocator::markRendered(uint32_t pageId)
{
    if (pageId >= entries_.size()) {
        return;
    }
    if (entries_[pageId].physicalPage == kVsmInvalidPhysicalPage) {
        return;
    }
    entries_[pageId].rendered = 1;
}

bool vsmLightSpaceBoundsXy(const glm::mat4& lightView,
                           const glm::vec3& boundsMin,
                           const glm::vec3& boundsMax,
                           glm::vec2& outMin,
                           glm::vec2& outMax)
{
    if (!glm::all(glm::lessThanEqual(boundsMin, boundsMax))) {
        return false;
    }

    outMin = glm::vec2(std::numeric_limits<float>::max());
    outMax = glm::vec2(std::numeric_limits<float>::lowest());

    // All eight corners: the light basis is a rotation, so an axis-aligned box in
    // world space is not axis-aligned in light space and its extent cannot be
    // taken from two corners.
    for (uint32_t cornerIndex = 0; cornerIndex < 8u; ++cornerIndex) {
        const glm::vec3 corner{(cornerIndex & 1u) != 0u ? boundsMax.x : boundsMin.x,
                               (cornerIndex & 2u) != 0u ? boundsMax.y : boundsMin.y,
                               (cornerIndex & 4u) != 0u ? boundsMax.z : boundsMin.z};
        const glm::vec2 lightSpace = glm::vec2(lightView * glm::vec4(corner, 1.0f));
        if (!std::isfinite(lightSpace.x) || !std::isfinite(lightSpace.y)) {
            return false;
        }
        outMin = glm::min(outMin, lightSpace);
        outMax = glm::max(outMax, lightSpace);
    }

    return true;
}

bool vsmPageOverlapsLightSpaceBounds(const VsmClipmapSettings& settings,
                                     uint32_t level,
                                     const glm::ivec2& absolutePage,
                                     const glm::vec2& lightMin,
                                     const glm::vec2& lightMax)
{
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    if (level >= clamped.levelCount) {
        return false;
    }

    const float pageWorldSize = vsmPageWorldSize(clamped, level);
    const glm::vec2 pageMin = glm::vec2(absolutePage) * pageWorldSize;
    const glm::vec2 pageMax = pageMin + glm::vec2(pageWorldSize);

    return lightMax.x >= pageMin.x && lightMin.x <= pageMax.x && lightMax.y >= pageMin.y &&
           lightMin.y <= pageMax.y;
}

void vsmPagesOverlappingBounds(const VsmClipmapSettings& settings,
                               const glm::mat4& lightView,
                               uint32_t level,
                               const glm::ivec2& windowOrigin,
                               const glm::vec3& boundsMin,
                               const glm::vec3& boundsMax,
                               std::vector<uint32_t>& pageIds)
{
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    if (level >= clamped.levelCount) {
        return;
    }

    glm::vec2 lightMin{0.0f};
    glm::vec2 lightMax{0.0f};
    if (!vsmLightSpaceBoundsXy(lightView, boundsMin, boundsMax, lightMin, lightMax)) {
        return;
    }

    const glm::ivec2 firstPage = vsmAbsolutePageCoords(clamped, level, lightMin);
    const glm::ivec2 lastPage = vsmAbsolutePageCoords(clamped, level, lightMax);

    // Clipped to the window before iterating, not after: a caster far outside it
    // would otherwise walk an unbounded rect to produce nothing.
    const int32_t axis = static_cast<int32_t>(kVsmPagesPerLevelAxis);
    const int32_t beginX = std::max(firstPage.x, windowOrigin.x);
    const int32_t beginY = std::max(firstPage.y, windowOrigin.y);
    const int32_t endX = std::min(lastPage.x, windowOrigin.x + axis - 1);
    const int32_t endY = std::min(lastPage.y, windowOrigin.y + axis - 1);

    for (int32_t y = beginY; y <= endY; ++y) {
        for (int32_t x = beginX; x <= endX; ++x) {
            pageIds.push_back(vsmPageId(level, vsmSlotIndex(glm::ivec2{x, y})));
        }
    }
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
