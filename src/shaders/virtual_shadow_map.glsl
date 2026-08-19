#ifndef VE_VIRTUAL_SHADOW_MAP_GLSL
#define VE_VIRTUAL_SHADOW_MAP_GLSL

// GLSL mirror of renderer/VirtualShadowMap.h.
//
// That header is the unit-tested reference copy; this is the shader-side
// duplicate, the same arrangement ClusterGrid.h/cluster_build.comp and
// MeshLod.h/cull.comp already use. Keep the two in sync: a divergence here
// marks or samples the wrong page, which reads as a plausible-but-wrong shadow
// rather than as an obvious failure.

const uint kVsmPageSize = 128u;
const uint kVsmPagesPerLevelAxis = 16u;
const uint kVsmPagesPerLevel = kVsmPagesPerLevelAxis * kVsmPagesPerLevelAxis;
const uint kVsmLevelResolution = kVsmPagesPerLevelAxis * kVsmPageSize;
const uint kVsmMaxClipmapLevels = 12u;
const uint kVsmMaxVirtualPages = kVsmMaxClipmapLevels * kVsmPagesPerLevel;
const uint kVsmPagePoolSize = 4096u;
const uint kVsmPagePoolPagesPerAxis = kVsmPagePoolSize / kVsmPageSize;

// World units across one page at this level. Level L doubles level L-1.
float vsmPageWorldSize(float level0Extent, uint level)
{
    float levelExtent = level0Extent * float(1u << min(level, kVsmMaxClipmapLevels - 1u));
    return levelExtent / float(kVsmPagesPerLevelAxis);
}

float vsmTexelWorldSize(float level0Extent, uint level)
{
    return vsmPageWorldSize(level0Extent, level) / float(kVsmPageSize);
}

// Coarsest level whose addressable window still reaches a point this far from
// the camera. One page short of half the window, because the camera sits
// somewhere inside its own page rather than at its centre.
uint vsmMinLevelForCoverage(float level0Extent, uint levelCount, float distanceToCamera)
{
    uint maxLevel = max(levelCount, 1u) - 1u;
    if (!(distanceToCamera > 0.0)) {
        return 0u;
    }

    float level0Reach = (float(kVsmPagesPerLevelAxis / 2u) - 1.0) * vsmPageWorldSize(level0Extent, 0u);
    if (!(level0Reach > 0.0) || distanceToCamera <= level0Reach) {
        return 0u;
    }

    float levels = ceil(log2(distanceToCamera / level0Reach));
    if (!(levels > 0.0)) {
        return 0u;
    }
    if (levels >= float(maxLevel)) {
        return maxLevel;
    }
    return min(uint(levels), maxLevel);
}

// Finest level that both resolves and reaches a point at this distance.
//
// Quality: the finest level whose texel is no smaller than the world footprint
// of one screen pixel. projScaleY is viewportHeight * 0.5 * abs(proj[1][1]), the
// same quantity cull.comp uses for LOD selection -- and the abs() matters for the
// same reason: these projections carry the Vulkan Y-flip.
//
// Coverage: a level finer than its window can reach has no slot for the point at
// all, so the maximum of the two is what is actually addressable.
uint vsmSelectLevel(float level0Extent, float texelsPerPixel, uint levelCount, float distanceToCamera, float projScaleY)
{
    uint maxLevel = max(levelCount, 1u) - 1u;
    uint coverageLevel = vsmMinLevelForCoverage(level0Extent, levelCount, distanceToCamera);
    if (!(projScaleY > 0.0) || !(distanceToCamera > 0.0)) {
        return coverageLevel;
    }

    float desiredTexelSize = distanceToCamera / projScaleY * texelsPerPixel;
    float level0TexelSize = vsmTexelWorldSize(level0Extent, 0u);
    if (!(desiredTexelSize > level0TexelSize)) {
        return coverageLevel;
    }

    float levels = ceil(log2(desiredTexelSize / level0TexelSize));
    if (!(levels > 0.0)) {
        return coverageLevel;
    }
    uint qualityLevel = levels >= float(maxLevel) ? maxLevel : min(uint(levels), maxLevel);
    return max(qualityLevel, coverageLevel);
}

// Page coordinates in the level's absolute, camera-independent grid. Flooring
// (not truncation) is load-bearing: the grid runs both ways from the light-space
// origin, and truncating toward zero folds the pages either side of it together.
ivec2 vsmAbsolutePageCoords(float level0Extent, uint level, vec2 lightSpaceXy)
{
    float pageWorldSize = vsmPageWorldSize(level0Extent, level);
    return ivec2(floor(lightSpaceXy / pageWorldSize));
}

// Minimum corner of the addressable window: the kVsmPagesPerLevelAxis square of
// absolute pages centred on the camera.
ivec2 vsmWindowOrigin(float level0Extent, uint level, vec2 cameraLightSpaceXy)
{
    return vsmAbsolutePageCoords(level0Extent, level, cameraLightSpaceXy) - ivec2(kVsmPagesPerLevelAxis / 2u);
}

bool vsmPageInWindow(ivec2 absolutePage, ivec2 windowOrigin)
{
    ivec2 offset = absolutePage - windowOrigin;
    return all(greaterThanEqual(offset, ivec2(0))) && all(lessThan(offset, ivec2(kVsmPagesPerLevelAxis)));
}

// Toroidal wrap rather than a window-relative offset, so scrolling the window by
// one page renumbers only the row or column that actually changed identity.
uint vsmSlotIndex(ivec2 absolutePage)
{
    int axis = int(kVsmPagesPerLevelAxis);
    uint x = uint(((absolutePage.x % axis) + axis) % axis);
    uint y = uint(((absolutePage.y % axis) + axis) % axis);
    return y * kVsmPagesPerLevelAxis + x;
}

uint vsmPageId(uint level, uint slot)
{
    return min(level, kVsmMaxClipmapLevels - 1u) * kVsmPagesPerLevel + min(slot, kVsmPagesPerLevel - 1u);
}

uint vsmPageLevel(uint pageId)
{
    return min(pageId, kVsmMaxVirtualPages - 1u) / kVsmPagesPerLevel;
}

uint vsmPageSlot(uint pageId)
{
    return min(pageId, kVsmMaxVirtualPages - 1u) % kVsmPagesPerLevel;
}

uint vsmRequestWordIndex(uint pageId)
{
    return min(pageId, kVsmMaxVirtualPages - 1u) / 32u;
}

uint vsmRequestBitMask(uint pageId)
{
    return 1u << (min(pageId, kVsmMaxVirtualPages - 1u) % 32u);
}

#endif // VE_VIRTUAL_SHADOW_MAP_GLSL
