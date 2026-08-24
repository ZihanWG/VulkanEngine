#ifndef VE_VIRTUAL_SHADOW_MAP_GLSL
#define VE_VIRTUAL_SHADOW_MAP_GLSL

// The page table is reached by device address, and the sampling path takes a
// FrameConstants by value. Both are declared here rather than left to the
// including shader, so a shader that includes this header compiles whether or
// not it happened to need either on its own.
#extension GL_EXT_buffer_reference : require
// The page-table address travels in a uvec4 field, so the reference is built
// from a uvec2 rather than declared as one.
#extension GL_EXT_buffer_reference_uvec2 : require
#include "object_frame_data.glsl"

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

// --- Page table and sampling ---------------------------------------------

const uint kVsmInvalidPhysicalPage = 0xFFFFFFFFu;

// Mirrors ve::renderer::VsmPageTableEntry (16-byte std430 stride).
//
// The absolute coordinates are checked, not just the residency flag: a slot is a
// toroidal wrap, so a scrolled window makes the same slot name a different page,
// and trusting residency alone would sample depth rendered for somewhere else.
struct VsmPageTableEntry {
    uint physicalPage;
    int absoluteX;
    int absoluteY;
    uint rendered;
};

layout(buffer_reference, std430) readonly buffer VsmPageTableBuffer {
    VsmPageTableEntry entries[];
};

// UV origin and extent of a physical page inside the pool.
vec4 vsmPagePoolUvRect(uint physicalPage)
{
    uint page = min(physicalPage, kVsmPagePoolPagesPerAxis * kVsmPagePoolPagesPerAxis - 1u);
    float poolSize = float(kVsmPagePoolSize);
    float pageSize = float(kVsmPageSize);
    vec2 origin = vec2(float((page % kVsmPagePoolPagesPerAxis) * kVsmPageSize),
                       float((page / kVsmPagePoolPagesPerAxis) * kVsmPageSize)) /
                  poolSize;
    return vec4(origin, pageSize / poolSize, pageSize / poolSize);
}

// Page-local UV of a light-space position inside its page, matching the Y flip
// vsmPageViewProjection bakes in: that projection negates the whole second row,
// so light-space +Y maps to UV 0 rather than 1.
vec2 vsmPageLocalUv(float level0Extent, uint level, ivec2 absolutePage, vec2 lightSpaceXy)
{
    float pageWorldSize = vsmPageWorldSize(level0Extent, level);
    vec2 pageMin = vec2(absolutePage) * pageWorldSize;
    vec2 local = (lightSpaceXy - pageMin) / pageWorldSize;
    return vec2(local.x, 1.0 - local.y);
}

// Light-space Z to the page's [0,1] depth, from the same ortho mapping:
// zNear = -depthRange and zFar = +depthRange, so the end nearest the light is 0.
float vsmPageDepth(float depthRange, float lightSpaceZ)
{
    return 0.5 - lightSpaceZ / (2.0 * max(depthRange, 1e-4));
}

// Visibility of a world position, walking from the level its resolution wants up
// to the coarsest, and sampling the first level that is both addressable and
// actually rendered.
//
// The walk IS the fallback that makes the whole design survivable: the page
// request set lags the camera by a frame or two, so a fine page may not be
// resident yet -- but clipmap levels nest, so a coarser one covering the same
// world is. The result degrades to a blurrier shadow rather than a hole.
//
// Returns 1.0 (fully lit) when no level is resident at all, which is the same
// thing the cascades do outside their range.
// Flag word in vsmPageTable.w. Bit 0 is "the pool is sampleable at all", which
// is what the whole path is gated on; bit 1 asks the sampler to also report the
// level it used, for the debug view. A bitfield rather than a second slot
// because every other component of every VSM vector is already carrying a value.
const uint kVsmFlagEnabled = 1u;
const uint kVsmFlagDebugLevels = 2u;
const uint kVsmFlagDebugDepthDelta = 4u;

// What vsmShadowFactorLevel reports when the walk found nothing resident and the
// lookup fell through to "lit". Distinct from level 0, which is a real answer.
const uint kVsmNoResidentLevel = 0xFFFFFFFFu;

// Recovers the depth a page stores at one texel, by bisecting the comparison.
//
// The pool is bound as a sampler2DShadow, so it cannot return a raw depth: every
// fetch is a comparison against a reference this code supplies. But a comparison
// is a predicate on the stored value, and sixteen of them bisect [0, 1] to about
// 1/65536 of the page's depth range -- 0.0076 world units at the default 250.
//
// The compare op is LESS_OR_EQUAL, so a lit result means the reference is at or
// in front of what is stored, and "still lit" is the half to keep. The compare
// filter is LINEAR, so each fetch is a bilinear blend of four comparisons and
// the threshold converges on their median rather than on one texel's value --
// the right answer for "what is this lookup comparing against", the wrong one
// for "what is in texel (i, j)".
float vsmProbeStoredDepth(sampler2DShadow pagePool, vec2 poolUv)
{
    float lo = 0.0;
    float hi = 1.0;
    for (int iteration = 0; iteration < 16; ++iteration) {
        float mid = 0.5 * (lo + hi);
        if (texture(pagePool, vec3(poolUv, mid)) >= 0.5) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// The shadow factor, and the clipmap level the answer came from.
//
// The level is the diagnostic that the residency grid cannot give: the grid says
// which pages exist, not which one a given surface ended up sampling after the
// walk up from the coverage bound. A surface that looks wrong and a surface that
// sampled a coarser level than expected are the same picture until this is
// visible.
float vsmShadowFactorLevel(FrameConstants frame,
                           sampler2DShadow pagePool,
                           vec3 worldPosition,
                           vec3 normal,
                           int pcfRadius,
                           out uint sampledLevel)
{
    sampledLevel = kVsmNoResidentLevel;
    if ((frame.vsmPageTable.w & kVsmFlagEnabled) == 0u) {
        return 1.0;
    }

    uint levelCount = max(frame.vsmPageTable.z, 1u);
    float level0Extent = frame.vsmParams.x;
    float depthRange = frame.vsmParams.z;
    float poolTexel = frame.vsmParams.w;
    vec2 cameraLightSpaceXy = frame.vsmCamera.xy;
    float normalBias = frame.vsmCamera.z;
    // In TEXELS of the level that ends up being sampled, not in normalized page
    // depth. A page's depth axis spans 2*depthRange world units -- 500 by
    // default -- while a cascade's spans its own ortho box, so the cascades'
    // normalized constant means something entirely different here: 0.002 of a
    // cascade is centimetres, 0.002 of a page is a whole world unit. Passing it
    // straight through lifted every umbra by about 15% of the sun (measured:
    // docs/virtual_shadow_maps.md).
    //
    // Texels rather than world units because a clipmap has no single texel size:
    // level L's is 2^L times level 0's, so the depth error a texel can hide
    // doubles with it. One bias in texel units is right at every level; one in
    // world units is wrong at all but one of them.
    float depthBiasTexels = frame.vsmCamera.w;

    // Offset along the normal before projecting, same trick and same units the
    // cascades and the punctual atlas use: it lets the depth bias stay small,
    // and depth bias is what buys peter-panning.
    vec3 biased = worldPosition + normal * normalBias;
    vec4 lightSpace = frame.vsmLightView * vec4(biased, 1.0);

    float distanceToCamera = length(biased - frame.cameraPosition.xyz);
    // Coverage bound rather than the full selection: this only has to be a lower
    // bound on where to start looking, because the walk below finds the first
    // level that is actually resident -- which is the level the marking pass
    // chose. A finer start costs a few failed table reads, never a wrong answer,
    // and it saves carrying projScaleY into the fragment stage.
    uint startLevel = vsmMinLevelForCoverage(level0Extent, levelCount, distanceToCamera);

    // Unbiased: the range test asks whether the point is inside the page's depth
    // slab at all, which the bias must not be able to answer differently.
    float pageDepth = vsmPageDepth(depthRange, lightSpace.z);
    if (pageDepth <= 0.0 || pageDepth >= 1.0) {
        return 1.0;
    }

    VsmPageTableBuffer table = VsmPageTableBuffer(frame.vsmPageTable.xy);

    for (uint level = startLevel; level < levelCount; ++level) {
        ivec2 absolutePage = vsmAbsolutePageCoords(level0Extent, level, lightSpace.xy);
        ivec2 windowOrigin = vsmWindowOrigin(level0Extent, level, cameraLightSpaceXy);
        if (!vsmPageInWindow(absolutePage, windowOrigin)) {
            continue;
        }

        VsmPageTableEntry entry = table.entries[vsmPageId(level, vsmSlotIndex(absolutePage))];
        if (entry.rendered == 0u || entry.physicalPage == kVsmInvalidPhysicalPage) {
            continue;
        }
        // Identity check: this slot may have scrolled to a different page since
        // its depth was drawn.
        if (entry.absoluteX != absolutePage.x || entry.absoluteY != absolutePage.y) {
            continue;
        }

        vec4 poolRect = vsmPagePoolUvRect(entry.physicalPage);
        vec2 pageUv = vsmPageLocalUv(level0Extent, level, absolutePage, lightSpace.xy);

        // Now that the level is known, so is the texel it will be compared
        // against. World units first, then into this page's depth normalization.
        float worldBias = depthBiasTexels * vsmTexelWorldSize(level0Extent, level);
        float depth = pageDepth - worldBias / (2.0 * max(depthRange, 1e-4));

        // Every tap is clamped inside this page. Neighbouring pool texels belong
        // to a different page -- a different world location entirely, possibly a
        // different clipmap level -- so a tap that walks out compares against
        // unrelated depth. The punctual atlas learned this the same way.
        float halfTexel = 0.5 * poolTexel / poolRect.z;
        int radius = clamp(pcfRadius, 0, 4);
        float litSamples = 0.0;
        float sampleCount = 0.0;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                vec2 offset = vec2(float(x), float(y)) * (poolTexel / poolRect.z);
                vec2 clamped = clamp(pageUv + offset, vec2(halfTexel), vec2(1.0 - halfTexel));
                vec2 poolUv = poolRect.xy + clamped * poolRect.zw;
                litSamples += texture(pagePool, vec3(poolUv, depth));
                sampleCount += 1.0;
            }
        }
        sampledLevel = level;
        return sampleCount > 0.0 ? litSamples / sampleCount : 1.0;
    }

    return 1.0;
}

// The stored depth in front of a surface, in world units, for the debug view.
//
// A separate walk rather than another out-parameter on vsmShadowFactorLevel, and
// that is not a style choice. Adding a second `out` and *reading* it in the
// fragment shader made the first one -- the sampled level -- come back as its
// "nothing resident" sentinel on this driver, even though an instrumented probe
// three lines earlier proved it valid. Every piece worked alone: one extra
// unused out param, the bool, the guarded block, the runtime flag. Only reading
// both outputs failed, and collapsing the two debug branches into one did not
// help. A function with a single return value has no such problem.
//
// The duplication is the price, and it is bounded: this runs only when the debug
// view is on, and it is the same walk with the same terminating conditions.
// Returning a negative value means the walk found nothing resident.
float vsmDebugStoredDepthDelta(FrameConstants frame,
                               sampler2DShadow pagePool,
                               vec3 worldPosition,
                               vec3 normal)
{
    if ((frame.vsmPageTable.w & kVsmFlagEnabled) == 0u) {
        return -1.0;
    }

    uint levelCount = max(frame.vsmPageTable.z, 1u);
    float level0Extent = frame.vsmParams.x;
    float depthRange = frame.vsmParams.z;
    float poolTexel = frame.vsmParams.w;
    vec2 cameraLightSpaceXy = frame.vsmCamera.xy;
    float normalBias = frame.vsmCamera.z;

    vec3 biased = worldPosition + normal * normalBias;
    vec4 lightSpace = frame.vsmLightView * vec4(biased, 1.0);
    float distanceToCamera = length(biased - frame.cameraPosition.xyz);
    uint startLevel = vsmMinLevelForCoverage(level0Extent, levelCount, distanceToCamera);

    float pageDepth = vsmPageDepth(depthRange, lightSpace.z);
    if (pageDepth <= 0.0 || pageDepth >= 1.0) {
        return -1.0;
    }

    VsmPageTableBuffer table = VsmPageTableBuffer(frame.vsmPageTable.xy);

    for (uint level = startLevel; level < levelCount; ++level) {
        ivec2 absolutePage = vsmAbsolutePageCoords(level0Extent, level, lightSpace.xy);
        ivec2 windowOrigin = vsmWindowOrigin(level0Extent, level, cameraLightSpaceXy);
        if (!vsmPageInWindow(absolutePage, windowOrigin)) {
            continue;
        }

        VsmPageTableEntry entry = table.entries[vsmPageId(level, vsmSlotIndex(absolutePage))];
        if (entry.rendered == 0u || entry.physicalPage == kVsmInvalidPhysicalPage) {
            continue;
        }
        if (entry.absoluteX != absolutePage.x || entry.absoluteY != absolutePage.y) {
            continue;
        }

        vec4 poolRect = vsmPagePoolUvRect(entry.physicalPage);
        vec2 pageUv = vsmPageLocalUv(level0Extent, level, absolutePage, lightSpace.xy);
        float halfTexel = 0.5 * poolTexel / poolRect.z;
        vec2 centreUv = poolRect.xy + clamp(pageUv, vec2(halfTexel), vec2(1.0 - halfTexel)) * poolRect.zw;
        float stored = vsmProbeStoredDepth(pagePool, centreUv);
        // Signed, then clamped at zero -- NOT abs(). The compare is
        // LESS_OR_EQUAL, so only pageDepth > stored means the stored value is in
        // FRONT of this surface. A texel that is merely far away, a cleared one
        // above all, reconstructs to nearly 1.0 and would come back through
        // abs() as a large warm "occluder" on a surface the same lookup calls
        // lit. Zero therefore reads as "nothing in front of it", which covers
        // both a surface comparing against itself and one comparing against
        // something behind it; the negative sentinel above stays reserved for
        // "no resident page".
        return max(pageDepth - stored, 0.0) * 2.0 * max(depthRange, 1e-4);
    }

    return -1.0;
}

// The plain form, for every caller that does not want the diagnostic.
float vsmShadowFactor(FrameConstants frame,
                      sampler2DShadow pagePool,
                      vec3 worldPosition,
                      vec3 normal,
                      int pcfRadius)
{
    uint ignored = kVsmNoResidentLevel;
    return vsmShadowFactorLevel(frame, pagePool, worldPosition, normal, pcfRadius, ignored);
}

// One distinct colour per clipmap level, magenta where the walk found nothing
// resident at all -- a different failure from "sampled a coarse page", and one
// that has to be told apart at a glance.
//
// Sized by kVsmMaxClipmapLevels rather than by however many levels the default
// settings use: the UI goes to 12, and a palette that ran out would map every
// coarse level to the same colour, which is the one thing this view exists to
// prevent. Sizing the array by the constant makes a future change to it a
// compile error rather than a silently degraded view.
//
// The first five are a warm ramp because those are the levels a normal scene
// actually lands on; past that the requirement is only that neighbours are
// telling apart. Nothing in the ramp reuses the magenta.
// How far in front of a surface the page's stored depth sits, in world units.
// Negative means the walk found nothing resident. Blue is "nothing in front of
// it" -- either the surface is comparing against itself, or what it compares
// against is behind it; both mean any darkening there is self-shadowing or bias
// rather than an occluder.
vec3 vsmDepthDeltaDebugColor(float deltaWorld)
{
    if (deltaWorld < 0.0) {
        return vec3(1.0, 0.0, 0.8);
    }
    if (deltaWorld < 0.01) {
        return vec3(0.20, 0.35, 1.00);
    }
    if (deltaWorld < 0.05) {
        return vec3(0.20, 0.80, 0.90);
    }
    if (deltaWorld < 0.25) {
        return vec3(0.15, 0.85, 0.25);
    }
    if (deltaWorld < 1.0) {
        return vec3(0.95, 0.85, 0.15);
    }
    if (deltaWorld < 4.0) {
        return vec3(1.00, 0.55, 0.10);
    }
    return vec3(0.95, 0.25, 0.25);
}

vec3 vsmLevelDebugColor(uint level)
{
    if (level == kVsmNoResidentLevel) {
        return vec3(1.0, 0.0, 0.8);
    }
    const vec3 palette[kVsmMaxClipmapLevels] = vec3[kVsmMaxClipmapLevels](vec3(0.15, 0.85, 0.25),  // L0 green
                                                                          vec3(0.65, 0.90, 0.15),  // L1 yellow-green
                                                                          vec3(0.95, 0.85, 0.15),  // L2 yellow
                                                                          vec3(1.00, 0.55, 0.10),  // L3 orange
                                                                          vec3(0.95, 0.25, 0.25),  // L4 red
                                                                          vec3(0.55, 0.32, 0.12),  // L5 brown
                                                                          vec3(0.35, 0.45, 1.00),  // L6 blue
                                                                          vec3(0.20, 0.80, 0.90),  // L7 cyan
                                                                          vec3(0.10, 0.55, 0.50),  // L8 teal
                                                                          vec3(0.45, 0.55, 0.75),  // L9 slate
                                                                          vec3(0.70, 0.70, 0.70),  // L10 grey
                                                                          vec3(0.10, 0.15, 0.45)); // L11 navy
    return palette[min(level, kVsmMaxClipmapLevels - 1u)];
}

#endif // VE_VIRTUAL_SHADOW_MAP_GLSL
