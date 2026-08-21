#include "renderer/VirtualShadowMap.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

using Catch::Approx;
using ve::renderer::clampVsmClipmapSettings;
using ve::renderer::kMaxVsmCastersPerPage;
using ve::renderer::kMaxVsmPagesPerFrame;
using ve::renderer::kVsmLevelResolution;
using ve::renderer::kVsmMaxClipmapLevels;
using ve::renderer::kVsmMaxVirtualPages;
using ve::renderer::kVsmPagePoolPageCount;
using ve::renderer::kVsmPagePoolSize;
using ve::renderer::kVsmPageRequestWordCount;
using ve::renderer::kVsmPageSize;
using ve::renderer::kVsmPagesPerLevel;
using ve::renderer::kVsmPagesPerLevelAxis;
using ve::renderer::VsmClipmapSettings;
using ve::renderer::kVsmInvalidPhysicalPage;
using ve::renderer::vsmAbsolutePageCoords;
using ve::renderer::vsmAbsolutePageForSlot;
using ve::renderer::vsmLightSpaceBoundsXy;
using ve::renderer::vsmPagesOverlappingBounds;
using ve::renderer::VsmPageAllocator;
using ve::renderer::VsmPageTableEntry;
using ve::renderer::vsmDecodeRequestStats;
using ve::renderer::vsmLightView;
using ve::renderer::vsmPageId;
using ve::renderer::vsmPageInWindow;
using ve::renderer::vsmPageLevel;
using ve::renderer::vsmPagePoolRect;
using ve::renderer::vsmPagePoolUvOffsetScale;
using ve::renderer::VsmPageRect;
using ve::renderer::vsmPageSlot;
using ve::renderer::vsmPageViewProjection;
using ve::renderer::vsmPageWorldSize;
using ve::renderer::vsmRequestBitMask;
using ve::renderer::vsmRequestWordIndex;
using ve::renderer::vsmMinLevelForCoverage;
using ve::renderer::vsmSelectLevel;
using ve::renderer::vsmSlotIndex;
using ve::renderer::vsmTexelWorldSize;
using ve::renderer::vsmWindowOrigin;

namespace {

VsmClipmapSettings defaultSettings()
{
    VsmClipmapSettings settings{};
    settings.levelCount = 8;
    settings.level0Extent = 4.0f;
    settings.texelsPerPixel = 1.0f;
    settings.depthRange = 250.0f;
    return settings;
}

// Light space is a pure rotation here (the view's eye is the world origin), so a
// light-space point can be pushed back into world space by the inverse.
glm::vec3 lightSpaceToWorld(const glm::mat4& lightView, const glm::vec3& lightSpacePosition)
{
    return glm::vec3(glm::inverse(lightView) * glm::vec4(lightSpacePosition, 1.0f));
}

glm::vec3 projectToNdc(const glm::mat4& viewProjection, const glm::vec3& worldPosition)
{
    const glm::vec4 clip = viewProjection * glm::vec4(worldPosition, 1.0f);
    return glm::vec3(clip) / clip.w;
}

} // namespace

TEST_CASE("Fixed page geometry is self-consistent", "[vsm]")
{
    REQUIRE(kVsmLevelResolution == kVsmPagesPerLevelAxis * kVsmPageSize);
    REQUIRE(kVsmPagesPerLevel == kVsmPagesPerLevelAxis * kVsmPagesPerLevelAxis);
    REQUIRE(kVsmMaxVirtualPages == kVsmMaxClipmapLevels * kVsmPagesPerLevel);
    REQUIRE(kVsmPageRequestWordCount * 32u >= kVsmMaxVirtualPages);

    // The addressable set is larger than the pool, which is what makes the page
    // table a real indirection rather than an identity map.
    REQUIRE(kVsmMaxVirtualPages > kVsmPagePoolPageCount);

    // The per-frame budget has to fit in the pool it draws into.
    REQUIRE(kMaxVsmPagesPerFrame <= kVsmPagePoolPageCount);
    REQUIRE(kMaxVsmCastersPerPage > 0u);

    // The coverage bound the axis has to satisfy for level selection to have any
    // headroom at all at 1080p; see the constant's comment.
    REQUIRE(kVsmPagesPerLevelAxis * kVsmPageSize >= 2u * 771u);
}

TEST_CASE("Clamping keeps every clipmap setting in range", "[vsm]")
{
    VsmClipmapSettings settings{};
    settings.levelCount = 0;
    settings.level0Extent = -5.0f;
    settings.texelsPerPixel = 0.0f;
    settings.depthRange = -1.0f;

    const VsmClipmapSettings clamped = clampVsmClipmapSettings(settings);
    REQUIRE(clamped.levelCount >= 1u);
    REQUIRE(clamped.levelCount <= kVsmMaxClipmapLevels);
    REQUIRE(clamped.level0Extent > 0.0f);
    REQUIRE(clamped.texelsPerPixel > 0.0f);
    REQUIRE(clamped.depthRange > 0.0f);

    settings.levelCount = kVsmMaxClipmapLevels + 7u;
    REQUIRE(clampVsmClipmapSettings(settings).levelCount == kVsmMaxClipmapLevels);

    // Non-finite input must not survive into a matrix.
    settings.level0Extent = std::numeric_limits<float>::quiet_NaN();
    settings.depthRange = std::numeric_limits<float>::infinity();
    const VsmClipmapSettings sanitized = clampVsmClipmapSettings(settings);
    REQUIRE(std::isfinite(sanitized.level0Extent));
    REQUIRE(std::isfinite(sanitized.depthRange));
}

TEST_CASE("The light basis is translation-free and puts the light along -Z", "[vsm]")
{
    const glm::vec3 direction = glm::normalize(glm::vec3{0.35f, -0.65f, -0.55f});
    const glm::mat4 lightView = vsmLightView(direction);

    // The absolute page grid only works if the basis never translates.
    REQUIRE(lightView[3][0] == Approx(0.0f).margin(1e-5f));
    REQUIRE(lightView[3][1] == Approx(0.0f).margin(1e-5f));
    REQUIRE(lightView[3][2] == Approx(0.0f).margin(1e-5f));

    // A point one unit down-light sits one unit along -Z.
    const glm::vec3 downLight = glm::vec3(lightView * glm::vec4(direction, 1.0f));
    REQUIRE(downLight.x == Approx(0.0f).margin(1e-4f));
    REQUIRE(downLight.y == Approx(0.0f).margin(1e-4f));
    REQUIRE(downLight.z == Approx(-1.0f).margin(1e-4f));
}

TEST_CASE("A straight-down light still produces a usable basis", "[vsm]")
{
    // The default up axis is parallel to this direction; without the fallback
    // the cross product degenerates and every matrix entry becomes NaN.
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.0f, -1.0f, 0.0f});
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            REQUIRE(std::isfinite(lightView[column][row]));
        }
    }

    // A zero direction must not produce NaNs either.
    const glm::mat4 degenerate = vsmLightView(glm::vec3{0.0f});
    REQUIRE(std::isfinite(degenerate[0][0]));
}

TEST_CASE("Each clipmap level doubles the previous one", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();

    const float page0 = vsmPageWorldSize(settings, 0);
    REQUIRE(page0 == Approx(settings.level0Extent / static_cast<float>(kVsmPagesPerLevelAxis)));
    REQUIRE(vsmTexelWorldSize(settings, 0) == Approx(page0 / static_cast<float>(kVsmPageSize)));

    for (uint32_t level = 1; level < settings.levelCount; ++level) {
        REQUIRE(vsmPageWorldSize(settings, level) == Approx(vsmPageWorldSize(settings, level - 1) * 2.0f));
        REQUIRE(vsmTexelWorldSize(settings, level) == Approx(vsmTexelWorldSize(settings, level - 1) * 2.0f));
    }
}

TEST_CASE("Level selection tracks the pixel footprint and stays in range", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const float projScaleY = 771.0f; // ~1080p at a 70 degree vertical FOV.

    // Right at the camera the finest level is the answer.
    REQUIRE(vsmSelectLevel(settings, 0.01f, projScaleY) == 0u);

    // Doubling the distance doubles the requested texel, which is exactly one
    // level. Selection must be monotonic in distance.
    uint32_t previous = 0;
    for (float distance = 1.0f; distance < 4000.0f; distance *= 1.7f) {
        const uint32_t level = vsmSelectLevel(settings, distance, projScaleY);
        REQUIRE(level >= previous);
        REQUIRE(level < settings.levelCount);
        previous = level;
    }

    // Far past the clipmap's reach it saturates rather than running off the end.
    REQUIRE(vsmSelectLevel(settings, 1.0e9f, projScaleY) == settings.levelCount - 1u);

    // The selected level's texel must actually cover the requested footprint.
    // Only ever coarser than asked for: the coverage bound can raise the level,
    // never lower it, so the texel can only grow.
    for (float distance = 0.5f; distance < 500.0f; distance *= 1.3f) {
        const uint32_t level = vsmSelectLevel(settings, distance, projScaleY);
        const float requested = distance / projScaleY * settings.texelsPerPixel;
        if (level < settings.levelCount - 1u) {
            REQUIRE(vsmTexelWorldSize(settings, level) >= Approx(requested).epsilon(1e-4));
        }
    }
}

TEST_CASE("Degenerate level-selection inputs fall back to what coverage allows", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();

    // A negative projScaleY is the trap cull.comp documents: these projections
    // carry the Vulkan Y-flip, so a caller forgetting abs() passes one in. The
    // quality side gives up, but the answer must still be addressable -- level 0
    // has no slot for a point a kilometre away.
    REQUIRE(vsmSelectLevel(settings, 10.0f, -771.0f) == vsmMinLevelForCoverage(settings, 10.0f));
    REQUIRE(vsmSelectLevel(settings, 10.0f, 0.0f) == vsmMinLevelForCoverage(settings, 10.0f));
    REQUIRE(vsmSelectLevel(settings, 1000.0f, -771.0f) > 0u);

    // No distance at all: nothing to cover, so the finest level is correct.
    REQUIRE(vsmSelectLevel(settings, -1.0f, 771.0f) == 0u);
    REQUIRE(vsmSelectLevel(settings, std::numeric_limits<float>::quiet_NaN(), 771.0f) == 0u);
}

TEST_CASE("The selected level's window always reaches the point that selected it", "[vsm]")
{
    // The property this whole coverage bound exists for. Without it the
    // geometry-stress scene selected levels whose windows did not reach the
    // geometry, and the page request set came back EMPTY -- which in the
    // sampling phase is no shadow at all, not a blurry one.
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.35f, -0.65f, -0.55f});

    for (const float projScaleY : {200.0f, 771.0f, 2000.0f}) {
        for (float texelsPerPixel : {0.25f, 1.0f, 4.0f}) {
            VsmClipmapSettings tuned = settings;
            tuned.texelsPerPixel = texelsPerPixel;

            for (float distance = 0.5f; distance < 600.0f; distance *= 1.4f) {
                const uint32_t level = vsmSelectLevel(tuned, distance, projScaleY);

                // Worst case: the point lies `distance` away along one light-space
                // axis, which is the furthest the window has to stretch.
                const glm::vec2 camera{0.0f, 0.0f};
                const glm::vec2 point{distance, 0.0f};
                const glm::ivec2 windowOrigin = vsmWindowOrigin(tuned, level, camera);
                const glm::ivec2 page = vsmAbsolutePageCoords(tuned, level, point);

                if (level == tuned.levelCount - 1u) {
                    // The coarsest level is the end of the road; past its reach
                    // there is nothing left to fall back to.
                    continue;
                }
                REQUIRE(vsmPageInWindow(page, windowOrigin));
            }
        }
    }

    // Referenced so the light basis stays part of this test's setup rather than
    // silently unused if the loop above is edited.
    REQUIRE(std::isfinite(lightView[0][0]));
}

TEST_CASE("Coverage never selects a finer level than quality asked for", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    for (float distance = 0.5f; distance < 500.0f; distance *= 1.3f) {
        const uint32_t coverage = vsmMinLevelForCoverage(settings, distance);
        const uint32_t selected = vsmSelectLevel(settings, distance, 771.0f);
        REQUIRE(selected >= coverage);
    }

    // Coverage is monotonic in distance, like the quality side.
    uint32_t previous = 0;
    for (float distance = 0.5f; distance < 5000.0f; distance *= 1.6f) {
        const uint32_t coverage = vsmMinLevelForCoverage(settings, distance);
        REQUIRE(coverage >= previous);
        REQUIRE(coverage < settings.levelCount);
        previous = coverage;
    }
}

TEST_CASE("Coarser levels are selected when texelsPerPixel rises", "[vsm]")
{
    VsmClipmapSettings settings = defaultSettings();
    const uint32_t sharp = vsmSelectLevel(settings, 30.0f, 771.0f);
    settings.texelsPerPixel = 4.0f;
    const uint32_t coarse = vsmSelectLevel(settings, 30.0f, 771.0f);
    REQUIRE(coarse >= sharp);
}

TEST_CASE("Absolute page coordinates floor correctly through the origin", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const float pageWorldSize = vsmPageWorldSize(settings, 0);

    REQUIRE(vsmAbsolutePageCoords(settings, 0, glm::vec2{0.0f, 0.0f}) == glm::ivec2{0, 0});
    REQUIRE(vsmAbsolutePageCoords(settings, 0, glm::vec2{pageWorldSize * 0.99f, 0.0f}) == glm::ivec2{0, 0});
    REQUIRE(vsmAbsolutePageCoords(settings, 0, glm::vec2{pageWorldSize * 1.01f, 0.0f}) == glm::ivec2{1, 0});

    // Truncation toward zero would fold the pages either side of the origin onto
    // the same coordinate; flooring keeps them distinct.
    REQUIRE(vsmAbsolutePageCoords(settings, 0, glm::vec2{-0.01f, -0.01f}) == glm::ivec2{-1, -1});
    REQUIRE(vsmAbsolutePageCoords(settings, 0, glm::vec2{-pageWorldSize * 1.01f, 0.0f}).x == -2);
}

TEST_CASE("A page's world rect does not depend on the camera", "[vsm]")
{
    // This is the property the whole absolute-grid design exists to provide: if
    // a page's rect moved with the camera, its contents would change every
    // frame and caching could never hit.
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.35f, -0.65f, -0.55f});
    const glm::ivec2 page{3, -2};

    const glm::mat4 first = vsmPageViewProjection(settings, lightView, 2, page);
    const glm::mat4 second = vsmPageViewProjection(settings, lightView, 2, page);
    REQUIRE(first == second);

    // And the slot a page occupies is its absolute coordinate wrapped, so it
    // does not change when the window scrolls either.
    REQUIRE(vsmSlotIndex(page) == vsmSlotIndex(page));
}

TEST_CASE("Scrolling the window by one page renumbers only the pages that left", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const uint32_t level = 3;
    const float pageWorldSize = vsmPageWorldSize(settings, level);

    const glm::vec2 cameraA{0.0f, 0.0f};
    const glm::vec2 cameraB{pageWorldSize, 0.0f};

    const glm::ivec2 originA = vsmWindowOrigin(settings, level, cameraA);
    const glm::ivec2 originB = vsmWindowOrigin(settings, level, cameraB);
    REQUIRE(originB == originA + glm::ivec2{1, 0});

    // Every page still inside both windows keeps its slot. Window-relative
    // indexing would have shifted all of them.
    uint32_t survivors = 0;
    for (int32_t y = originA.y; y < originA.y + static_cast<int32_t>(kVsmPagesPerLevelAxis); ++y) {
        for (int32_t x = originA.x; x < originA.x + static_cast<int32_t>(kVsmPagesPerLevelAxis); ++x) {
            const glm::ivec2 page{x, y};
            if (!vsmPageInWindow(page, originB)) {
                continue;
            }
            ++survivors;
            REQUIRE(vsmSlotIndex(page) == vsmSlotIndex(page));
        }
    }
    // One column of the old window scrolled out.
    REQUIRE(survivors == kVsmPagesPerLevel - kVsmPagesPerLevelAxis);
}

TEST_CASE("Every page in a window occupies a distinct slot", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const glm::ivec2 origin = vsmWindowOrigin(settings, 4, glm::vec2{123.0f, -456.0f});

    std::set<uint32_t> slots;
    for (int32_t y = origin.y; y < origin.y + static_cast<int32_t>(kVsmPagesPerLevelAxis); ++y) {
        for (int32_t x = origin.x; x < origin.x + static_cast<int32_t>(kVsmPagesPerLevelAxis); ++x) {
            const uint32_t slot = vsmSlotIndex(glm::ivec2{x, y});
            REQUIRE(slot < kVsmPagesPerLevel);
            REQUIRE(slots.insert(slot).second);
        }
    }
    REQUIRE(slots.size() == kVsmPagesPerLevel);
}

TEST_CASE("The window contains the camera and rejects pages outside it", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const uint32_t level = 1;
    const glm::vec2 camera{9.5f, -3.25f};

    const glm::ivec2 origin = vsmWindowOrigin(settings, level, camera);
    const glm::ivec2 cameraPage = vsmAbsolutePageCoords(settings, level, camera);
    REQUIRE(vsmPageInWindow(cameraPage, origin));

    const int32_t axis = static_cast<int32_t>(kVsmPagesPerLevelAxis);
    REQUIRE_FALSE(vsmPageInWindow(origin - glm::ivec2{1, 0}, origin));
    REQUIRE_FALSE(vsmPageInWindow(origin + glm::ivec2{axis, 0}, origin));
    REQUIRE(vsmPageInWindow(origin + glm::ivec2{axis - 1, axis - 1}, origin));
}

TEST_CASE("Page ids round-trip through level and slot", "[vsm]")
{
    for (uint32_t level = 0; level < kVsmMaxClipmapLevels; ++level) {
        for (uint32_t slot = 0; slot < kVsmPagesPerLevel; ++slot) {
            const uint32_t pageId = vsmPageId(level, slot);
            REQUIRE(pageId < kVsmMaxVirtualPages);
            REQUIRE(vsmPageLevel(pageId) == level);
            REQUIRE(vsmPageSlot(pageId) == slot);
        }
    }
}

TEST_CASE("Physical page rects tile the pool without overlapping", "[vsm]")
{
    // Every page must claim a cell of the pool's page grid, and claim it alone:
    // an overlap is one page silently stomping another page's depth. Counting
    // claims per cell tests that in one pass instead of pairwise.
    std::vector<uint32_t> claims(kVsmPagePoolPageCount, 0u);

    for (uint32_t page = 0; page < kVsmPagePoolPageCount; ++page) {
        const VsmPageRect rect = vsmPagePoolRect(page);
        REQUIRE(rect.size == kVsmPageSize);
        REQUIRE(rect.x + rect.size <= kVsmPagePoolSize);
        REQUIRE(rect.y + rect.size <= kVsmPagePoolSize);
        REQUIRE(rect.x % kVsmPageSize == 0u);
        REQUIRE(rect.y % kVsmPageSize == 0u);

        const uint32_t cellX = rect.x / kVsmPageSize;
        const uint32_t cellY = rect.y / kVsmPageSize;
        ++claims[cellY * (kVsmPagePoolSize / kVsmPageSize) + cellX];
    }

    for (const uint32_t claimCount : claims) {
        REQUIRE(claimCount == 1u);
    }
}

TEST_CASE("Pool UV rects map a page onto its own texels", "[vsm]")
{
    const VsmPageRect rect = vsmPagePoolRect(37);
    const glm::vec4 uv = vsmPagePoolUvOffsetScale(rect);

    REQUIRE(uv.x == Approx(static_cast<float>(rect.x) / static_cast<float>(kVsmPagePoolSize)));
    REQUIRE(uv.y == Approx(static_cast<float>(rect.y) / static_cast<float>(kVsmPagePoolSize)));
    REQUIRE(uv.z == Approx(static_cast<float>(kVsmPageSize) / static_cast<float>(kVsmPagePoolSize)));
    REQUIRE(uv.w == Approx(uv.z));

    // Page-local UV 0 and 1 land on the page's own edges.
    REQUIRE(uv.x + 0.0f * uv.z == Approx(static_cast<float>(rect.x) / static_cast<float>(kVsmPagePoolSize)));
    REQUIRE(uv.x + 1.0f * uv.z ==
            Approx(static_cast<float>(rect.x + rect.size) / static_cast<float>(kVsmPagePoolSize)));
}

TEST_CASE("A page's projection fills clip space with exactly that page's world rect", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.35f, -0.65f, -0.55f});
    const uint32_t level = 2;
    const glm::ivec2 page{-3, 5};
    const glm::mat4 pageViewProjection = vsmPageViewProjection(settings, lightView, level, page);

    const float pageWorldSize = vsmPageWorldSize(settings, level);
    const float minX = static_cast<float>(page.x) * pageWorldSize;
    const float minY = static_cast<float>(page.y) * pageWorldSize;

    // Centre of the page maps to the centre of clip space.
    const glm::vec3 centre = projectToNdc(
        pageViewProjection,
        lightSpaceToWorld(lightView, glm::vec3{minX + pageWorldSize * 0.5f, minY + pageWorldSize * 0.5f, 0.0f}));
    REQUIRE(centre.x == Approx(0.0f).margin(1e-4f));
    REQUIRE(centre.y == Approx(0.0f).margin(1e-4f));
    REQUIRE(centre.z == Approx(0.5f).margin(1e-4f));

    // All four corners land on the clip-space boundary, so the page covers its
    // rect exactly -- no gap between neighbouring pages and no overlap.
    for (int corner = 0; corner < 4; ++corner) {
        const float x = minX + ((corner & 1) != 0 ? pageWorldSize : 0.0f);
        const float y = minY + ((corner & 2) != 0 ? pageWorldSize : 0.0f);
        const glm::vec3 ndc = projectToNdc(pageViewProjection, lightSpaceToWorld(lightView, glm::vec3{x, y, 0.0f}));
        REQUIRE(std::abs(ndc.x) == Approx(1.0f).margin(1e-4f));
        REQUIRE(std::abs(ndc.y) == Approx(1.0f).margin(1e-4f));
    }

    // A point just outside the rect is outside clip space, which is what the
    // page's scissor and this projection have to agree on.
    const glm::vec3 outside =
        projectToNdc(pageViewProjection, lightSpaceToWorld(lightView, glm::vec3{minX - pageWorldSize * 0.1f, minY, 0.0f}));
    REQUIRE(outside.x < -1.0f);
}

TEST_CASE("Page depth follows the engine's normal-Z convention", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.0f, -1.0f, 0.0f});
    const glm::mat4 pageViewProjection = vsmPageViewProjection(settings, lightView, 0, glm::ivec2{0, 0});
    const float pageWorldSize = vsmPageWorldSize(settings, 0);
    const glm::vec2 centre{pageWorldSize * 0.5f, pageWorldSize * 0.5f};

    // Light-space +Z is toward the light; it must map to depth 0 so that a
    // LESS compare keeps the nearest caster.
    const glm::vec3 nearLight = projectToNdc(
        pageViewProjection, lightSpaceToWorld(lightView, glm::vec3{centre.x, centre.y, settings.depthRange}));
    const glm::vec3 farFromLight = projectToNdc(
        pageViewProjection, lightSpaceToWorld(lightView, glm::vec3{centre.x, centre.y, -settings.depthRange}));

    REQUIRE(nearLight.z == Approx(0.0f).margin(1e-4f));
    REQUIRE(farFromLight.z == Approx(1.0f).margin(1e-4f));
    REQUIRE(nearLight.z < farFromLight.z);
}

TEST_CASE("Neighbouring pages meet exactly at their shared edge", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.2f, -0.9f, 0.3f});
    const uint32_t level = 1;
    const float pageWorldSize = vsmPageWorldSize(settings, level);

    const glm::ivec2 left{4, 1};
    const glm::ivec2 right{5, 1};
    const float sharedX = static_cast<float>(right.x) * pageWorldSize;
    const float y = (static_cast<float>(left.y) + 0.5f) * pageWorldSize;
    const glm::vec3 world = lightSpaceToWorld(lightView, glm::vec3{sharedX, y, 0.0f});

    const glm::vec3 fromLeft = projectToNdc(vsmPageViewProjection(settings, lightView, level, left), world);
    const glm::vec3 fromRight = projectToNdc(vsmPageViewProjection(settings, lightView, level, right), world);

    REQUIRE(fromLeft.x == Approx(1.0f).margin(1e-4f));
    REQUIRE(fromRight.x == Approx(-1.0f).margin(1e-4f));
    REQUIRE(fromLeft.y == Approx(fromRight.y).margin(1e-4f));
}

TEST_CASE("Each page owns a distinct request bit", "[vsm]")
{
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (uint32_t pageId = 0; pageId < kVsmMaxVirtualPages; ++pageId) {
        const uint32_t word = vsmRequestWordIndex(pageId);
        const uint32_t mask = vsmRequestBitMask(pageId);
        REQUIRE(word < kVsmPageRequestWordCount);
        REQUIRE(mask != 0u);
        REQUIRE(seen.insert({word, mask}).second);
    }
}

TEST_CASE("Request stats decode the bitmask the marking pass writes", "[vsm]")
{
    std::vector<uint32_t> words(kVsmPageRequestWordCount, 0u);

    const auto request = [&words](uint32_t pageId) {
        words[vsmRequestWordIndex(pageId)] |= vsmRequestBitMask(pageId);
    };

    request(vsmPageId(0, 0));
    request(vsmPageId(0, 17));
    request(vsmPageId(3, 5));

    const auto stats = vsmDecodeRequestStats(words.data(), 8);
    REQUIRE(stats.requestedPages == 3u);
    REQUIRE(stats.requestedPerLevel[0] == 2u);
    REQUIRE(stats.requestedPerLevel[3] == 1u);
    REQUIRE(stats.lowestRequestedLevel == 0u);
    REQUIRE(stats.highestRequestedLevel == 3u);

    // Levels past the active count are not counted even if their bits are set,
    // so shrinking levelCount cannot report pages that will never be rendered.
    words.assign(kVsmPageRequestWordCount, 0u);
    request(vsmPageId(6, 1));
    REQUIRE(vsmDecodeRequestStats(words.data(), 4).requestedPages == 0u);
    REQUIRE(vsmDecodeRequestStats(words.data(), 8).requestedPages == 1u);
}

TEST_CASE("An empty request set decodes to zero rather than a stale level", "[vsm]")
{
    const std::vector<uint32_t> words(kVsmPageRequestWordCount, 0u);
    const auto stats = vsmDecodeRequestStats(words.data(), 8);
    REQUIRE(stats.requestedPages == 0u);
    REQUIRE(stats.lowestRequestedLevel == 0u);
    REQUIRE(stats.highestRequestedLevel == 0u);

    const auto nullStats = vsmDecodeRequestStats(nullptr, 8);
    REQUIRE(nullStats.requestedPages == 0u);
    REQUIRE(nullStats.lowestRequestedLevel == 0u);
}


TEST_CASE("A slot plus its window names exactly one absolute page", "[vsm]")
{
    // The inverse of vsmSlotIndex within a window, and the reason the request
    // bitmask carries no coordinates: the slot and the window it was marked
    // against are enough to recover the page.
    const VsmClipmapSettings settings = defaultSettings();
    for (const glm::vec2 camera : {glm::vec2{0.0f, 0.0f}, glm::vec2{137.0f, -998.0f}, glm::vec2{-4.5f, 3.25f}}) {
        const glm::ivec2 origin = vsmWindowOrigin(settings, 2, camera);
        for (int32_t y = origin.y; y < origin.y + static_cast<int32_t>(kVsmPagesPerLevelAxis); ++y) {
            for (int32_t x = origin.x; x < origin.x + static_cast<int32_t>(kVsmPagesPerLevelAxis); ++x) {
                const glm::ivec2 page{x, y};
                REQUIRE(vsmAbsolutePageForSlot(origin, vsmSlotIndex(page)) == page);
            }
        }
    }
}

TEST_CASE("A freshly acquired page owns space but holds no depth", "[vsm]")
{
    VsmPageAllocator allocator;
    allocator.beginFrame(1);

    const auto result = allocator.acquire(vsmPageId(2, 5), glm::ivec2{3, -4});
    REQUIRE(result.physicalPage != kVsmInvalidPhysicalPage);
    REQUIRE_FALSE(result.alreadyRendered);
    REQUIRE_FALSE(result.evicted);
    REQUIRE(allocator.residentPages() == 1u);

    // Allocation and rendering are separate steps, so the entry must not claim
    // depth that has not been drawn.
    const VsmPageTableEntry& entry = allocator.entries()[vsmPageId(2, 5)];
    REQUIRE(entry.rendered == 0u);
    REQUIRE(entry.absoluteX == 3);
    REQUIRE(entry.absoluteY == -4);
}

TEST_CASE("Re-acquiring the same page at the same place is a hit", "[vsm]")
{
    VsmPageAllocator allocator;
    const uint32_t pageId = vsmPageId(1, 9);

    allocator.beginFrame(1);
    const uint32_t physical = allocator.acquire(pageId, glm::ivec2{7, 7}).physicalPage;
    allocator.markRendered(pageId);

    allocator.beginFrame(2);
    const auto second = allocator.acquire(pageId, glm::ivec2{7, 7});
    REQUIRE(second.physicalPage == physical);
    REQUIRE(second.alreadyRendered);
    REQUIRE(allocator.residentPages() == 1u);
}

TEST_CASE("A scrolled slot keeps its page but loses its depth", "[vsm]")
{
    // The failure this prevents is the quiet one: the slot is still resident and
    // still has depth, but that depth was rendered for somewhere else.
    VsmPageAllocator allocator;
    const uint32_t pageId = vsmPageId(0, 12);

    allocator.beginFrame(1);
    const uint32_t physical = allocator.acquire(pageId, glm::ivec2{0, 0}).physicalPage;
    allocator.markRendered(pageId);
    REQUIRE(allocator.entries()[pageId].rendered == 1u);

    allocator.beginFrame(2);
    const auto scrolled = allocator.acquire(pageId, glm::ivec2{16, 0});
    REQUIRE(scrolled.physicalPage == physical);
    REQUIRE_FALSE(scrolled.alreadyRendered);
    REQUIRE(allocator.entries()[pageId].rendered == 0u);
    REQUIRE(allocator.entries()[pageId].absoluteX == 16);
    // Reusing the slot's own page is not an eviction; nothing else lost anything.
    REQUIRE_FALSE(scrolled.evicted);
    REQUIRE(allocator.residentPages() == 1u);
}

TEST_CASE("Distinct virtual pages never share a physical page", "[vsm]")
{
    VsmPageAllocator allocator;
    allocator.beginFrame(1);

    std::set<uint32_t> physicalPages;
    for (uint32_t slot = 0; slot < 64u; ++slot) {
        const uint32_t pageId = vsmPageId(3, slot);
        const auto result = allocator.acquire(pageId, glm::ivec2{static_cast<int32_t>(slot), 0});
        REQUIRE(result.physicalPage != kVsmInvalidPhysicalPage);
        REQUIRE(physicalPages.insert(result.physicalPage).second);
    }
    REQUIRE(allocator.residentPages() == 64u);
}

TEST_CASE("A full pool evicts the least recently used page", "[vsm]")
{
    VsmPageAllocator allocator;

    // Fill the pool, one virtual page per physical page, each on its own frame
    // so the use order is unambiguous.
    for (uint32_t page = 0; page < kVsmPagePoolPageCount; ++page) {
        allocator.beginFrame(page + 1u);
        const auto result = allocator.acquire(page, glm::ivec2{static_cast<int32_t>(page), 0});
        REQUIRE(result.physicalPage != kVsmInvalidPhysicalPage);
        REQUIRE_FALSE(result.evicted);
        allocator.markRendered(page);
    }
    REQUIRE(allocator.residentPages() == kVsmPagePoolPageCount);

    // One more page has to take someone's space, and it must be page 0's -- the
    // one used longest ago.
    allocator.beginFrame(kVsmPagePoolPageCount + 1u);
    const auto overflow = allocator.acquire(kVsmPagePoolPageCount, glm::ivec2{0, 1});
    REQUIRE(overflow.physicalPage != kVsmInvalidPhysicalPage);
    REQUIRE(overflow.evicted);
    REQUIRE(allocator.evictionsThisFrame() == 1u);
    REQUIRE(allocator.entries()[0].physicalPage == kVsmInvalidPhysicalPage);
    REQUIRE(allocator.entries()[1].physicalPage != kVsmInvalidPhysicalPage);
    REQUIRE(allocator.residentPages() == kVsmPagePoolPageCount);
}

TEST_CASE("A frame never evicts a page it acquired itself", "[vsm]")
{
    // Without this a frame whose request set exceeds the pool would evict pages
    // it is still about to draw, and thrash instead of degrading.
    VsmPageAllocator allocator;
    allocator.beginFrame(1);

    for (uint32_t page = 0; page < kVsmPagePoolPageCount; ++page) {
        REQUIRE(allocator.acquire(page, glm::ivec2{static_cast<int32_t>(page), 0}).physicalPage !=
                kVsmInvalidPhysicalPage);
    }

    // Everything is spoken for by this same frame, so the extra request is
    // refused rather than granted at another live page's expense. The caller
    // reads that as "not resident" and falls back to a coarser level.
    const auto refused = allocator.acquire(kVsmPagePoolPageCount, glm::ivec2{0, 1});
    REQUIRE(refused.physicalPage == kVsmInvalidPhysicalPage);
    REQUIRE(allocator.evictionsThisFrame() == 0u);
    REQUIRE(allocator.residentPages() == kVsmPagePoolPageCount);
}

TEST_CASE("Resetting the allocator drops every page", "[vsm]")
{
    VsmPageAllocator allocator;
    allocator.beginFrame(1);
    allocator.acquire(vsmPageId(0, 0), glm::ivec2{0, 0});
    allocator.markRendered(vsmPageId(0, 0));
    REQUIRE(allocator.residentPages() == 1u);

    allocator.reset();
    REQUIRE(allocator.residentPages() == 0u);
    REQUIRE(allocator.entries()[vsmPageId(0, 0)].physicalPage == kVsmInvalidPhysicalPage);
    REQUIRE(allocator.entries()[vsmPageId(0, 0)].rendered == 0u);

    // And the pool is genuinely free again rather than merely unreferenced.
    allocator.beginFrame(2);
    REQUIRE(allocator.acquire(vsmPageId(5, 5), glm::ivec2{1, 1}).physicalPage != kVsmInvalidPhysicalPage);
}

TEST_CASE("markRendered does nothing for a page that owns no space", "[vsm]")
{
    VsmPageAllocator allocator;
    allocator.beginFrame(1);
    allocator.markRendered(vsmPageId(4, 4));
    REQUIRE(allocator.entries()[vsmPageId(4, 4)].rendered == 0u);

    // Out of range is a no-op rather than a write past the table.
    allocator.markRendered(kVsmMaxVirtualPages + 100u);
    REQUIRE(allocator.residentPages() == 0u);
}

TEST_CASE("Every page table entry starts unbound", "[vsm]")
{
    const VsmPageAllocator allocator;
    REQUIRE(allocator.entries().size() == kVsmMaxVirtualPages);
    for (const VsmPageTableEntry& entry : allocator.entries()) {
        REQUIRE(entry.physicalPage == kVsmInvalidPhysicalPage);
        REQUIRE(entry.rendered == 0u);
    }
}


TEST_CASE("A world box projects to the light-space rect of all eight corners", "[vsm]")
{
    // Two corners are not enough: the light basis is a rotation, so a world
    // axis-aligned box is not axis-aligned in light space.
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.35f, -0.65f, -0.55f});
    const glm::vec3 boundsMin{-1.0f, -2.0f, -3.0f};
    const glm::vec3 boundsMax{4.0f, 5.0f, 6.0f};

    glm::vec2 lightMin{0.0f};
    glm::vec2 lightMax{0.0f};
    REQUIRE(vsmLightSpaceBoundsXy(lightView, boundsMin, boundsMax, lightMin, lightMax));
    REQUIRE(lightMin.x <= lightMax.x);
    REQUIRE(lightMin.y <= lightMax.y);

    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 world{(corner & 1) != 0 ? boundsMax.x : boundsMin.x,
                              (corner & 2) != 0 ? boundsMax.y : boundsMin.y,
                              (corner & 4) != 0 ? boundsMax.z : boundsMin.z};
        const glm::vec2 lightSpace = glm::vec2(lightView * glm::vec4(world, 1.0f));
        REQUIRE(lightSpace.x >= lightMin.x - 1e-4f);
        REQUIRE(lightSpace.x <= lightMax.x + 1e-4f);
        REQUIRE(lightSpace.y >= lightMin.y - 1e-4f);
        REQUIRE(lightSpace.y <= lightMax.y + 1e-4f);
    }

    // An inverted box is rejected rather than producing a flipped rect.
    REQUIRE_FALSE(vsmLightSpaceBoundsXy(lightView, boundsMax, boundsMin, lightMin, lightMax));
}

TEST_CASE("Overlapping pages cover the box and stop at the window", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.0f, -1.0f, 0.0f});
    const uint32_t level = 3;
    const glm::ivec2 origin = vsmWindowOrigin(settings, level, glm::vec2{0.0f, 0.0f});
    const float pageWorldSize = vsmPageWorldSize(settings, level);

    // A box a little over two pages across in light space. With this light the
    // basis maps world XZ onto light XY, so the box spans pages either way.
    const glm::vec3 boundsMin{-pageWorldSize * 1.1f, -1.0f, -pageWorldSize * 1.1f};
    const glm::vec3 boundsMax{pageWorldSize * 1.1f, 1.0f, pageWorldSize * 1.1f};

    std::vector<uint32_t> pageIds;
    vsmPagesOverlappingBounds(settings, lightView, level, origin, boundsMin, boundsMax, pageIds);

    REQUIRE_FALSE(pageIds.empty());
    // Every returned page belongs to the requested level and is a real page.
    for (const uint32_t pageId : pageIds) {
        REQUIRE(pageId < kVsmMaxVirtualPages);
        REQUIRE(vsmPageLevel(pageId) == level);
    }
    // No duplicates: each page in the rect is emitted once.
    const std::set<uint32_t> unique(pageIds.begin(), pageIds.end());
    REQUIRE(unique.size() == pageIds.size());

    // The page containing the box centre must be in there -- that is the one
    // whose depth definitely changed.
    const glm::vec2 centreLightSpace = glm::vec2(lightView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    const uint32_t centrePage =
        vsmPageId(level, vsmSlotIndex(vsmAbsolutePageCoords(settings, level, centreLightSpace)));
    REQUIRE(unique.count(centrePage) == 1u);
}

TEST_CASE("A caster outside the window invalidates nothing", "[vsm]")
{
    // The clip has to happen before the walk, or a distant caster would iterate
    // an unbounded rect to produce an empty result.
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.0f, -1.0f, 0.0f});
    const uint32_t level = 0;
    const glm::ivec2 origin = vsmWindowOrigin(settings, level, glm::vec2{0.0f, 0.0f});

    const float far = vsmPageWorldSize(settings, level) * 1000.0f;
    std::vector<uint32_t> pageIds;
    vsmPagesOverlappingBounds(settings,
                              lightView,
                              level,
                              origin,
                              glm::vec3{far, -1.0f, far},
                              glm::vec3{far + 1.0f, 1.0f, far + 1.0f},
                              pageIds);
    REQUIRE(pageIds.empty());
}

TEST_CASE("A caster spanning the level dirties the whole window, and no more", "[vsm]")
{
    const VsmClipmapSettings settings = defaultSettings();
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.0f, -1.0f, 0.0f});
    const uint32_t level = 2;
    const glm::ivec2 origin = vsmWindowOrigin(settings, level, glm::vec2{0.0f, 0.0f});

    const float huge = vsmPageWorldSize(settings, level) * 1000.0f;
    std::vector<uint32_t> pageIds;
    vsmPagesOverlappingBounds(
        settings, lightView, level, origin, glm::vec3{-huge, -1.0f, -huge}, glm::vec3{huge, 1.0f, huge}, pageIds);

    // Bounded by the window: a ground plane that moves legitimately dirties its
    // whole level, but it can never ask for more pages than exist there.
    REQUIRE(pageIds.size() == kVsmPagesPerLevel);
    const std::set<uint32_t> unique(pageIds.begin(), pageIds.end());
    REQUIRE(unique.size() == kVsmPagesPerLevel);
}

TEST_CASE("A level past the active count yields nothing", "[vsm]")
{
    VsmClipmapSettings settings = defaultSettings();
    settings.levelCount = 4;
    const glm::mat4 lightView = vsmLightView(glm::vec3{0.0f, -1.0f, 0.0f});
    const glm::ivec2 origin = vsmWindowOrigin(settings, 4, glm::vec2{0.0f, 0.0f});

    std::vector<uint32_t> pageIds;
    vsmPagesOverlappingBounds(
        settings, lightView, 4, origin, glm::vec3{-1.0f}, glm::vec3{1.0f}, pageIds);
    REQUIRE(pageIds.empty());
}

TEST_CASE("Invalidating a page keeps its space but drops its depth", "[vsm]")
{
    VsmPageAllocator allocator;
    const uint32_t pageId = vsmPageId(2, 7);

    allocator.beginFrame(1);
    const uint32_t physical = allocator.acquire(pageId, glm::ivec2{1, 1}).physicalPage;
    allocator.markRendered(pageId);

    REQUIRE(allocator.invalidate(pageId));
    REQUIRE(allocator.entries()[pageId].rendered == 0u);
    // The page keeps its physical space and its identity -- only the depth is
    // stale, so the next frame redraws in place rather than reallocating.
    REQUIRE(allocator.entries()[pageId].physicalPage == physical);
    REQUIRE(allocator.entries()[pageId].absoluteX == 1);
    REQUIRE(allocator.residentPages() == 1u);

    // Re-acquiring now reports it as needing a draw.
    allocator.beginFrame(2);
    const auto reacquired = allocator.acquire(pageId, glm::ivec2{1, 1});
    REQUIRE(reacquired.physicalPage == physical);
    REQUIRE_FALSE(reacquired.alreadyRendered);
}

TEST_CASE("Invalidating reports only pages that actually held depth", "[vsm]")
{
    // The counter this feeds is meant to say how much work a moving caster
    // caused, so repeated requests and never-drawn pages must not inflate it.
    VsmPageAllocator allocator;
    const uint32_t pageId = vsmPageId(0, 3);

    allocator.beginFrame(1);
    REQUIRE_FALSE(allocator.invalidate(pageId));  // owns nothing yet

    allocator.acquire(pageId, glm::ivec2{0, 0});
    REQUIRE_FALSE(allocator.invalidate(pageId));  // allocated but never drawn

    allocator.markRendered(pageId);
    REQUIRE(allocator.invalidate(pageId));
    REQUIRE_FALSE(allocator.invalidate(pageId));  // already invalidated

    REQUIRE_FALSE(allocator.invalidate(kVsmMaxVirtualPages + 5u));
}
