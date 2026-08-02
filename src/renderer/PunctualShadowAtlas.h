#pragma once

// GPU-free core for the punctual (point/spot) shadow atlas.
//
// Two pieces live here, both free of Vulkan state so they can be unit tested on
// the CPU the same way ClusterGrid.h and CascadeMath.h are:
//   * the tile allocator that hands out atlas slots each frame, and
//   * the light-space projection an allocated slot renders its casters with.
//
// The atlas is a single depth texture split into a uniform grid of square
// tiles. A spot light takes one tile. Point lights take six (one per cube face)
// and are not wired up yet -- the slot record and the allocator are already
// shaped for them so that step does not have to reopen this layout.

#include "renderer/Bounds.h"

#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace ve::renderer {

// Atlas geometry. Tiles come in power-of-two size classes rather than one fixed
// size, so a light dominating the screen can get four times the resolution of
// one barely visible instead of both getting the same 512px. The shader is
// unaffected either way -- it only ever reads the per-slot UV rect out of
// GpuShadowSlot::atlasUvOffsetScale, so these constants stay CPU-side.
inline constexpr uint32_t kPunctualShadowAtlasSize = 4096;
inline constexpr uint32_t kPunctualShadowMaxTileSize = 1024;
inline constexpr uint32_t kPunctualShadowMinTileSize = 256;
// 1024, 512, 256.
inline constexpr uint32_t kPunctualShadowSizeClassCount = 3;

// Upper bound on slots, reached only if every tile is the smallest class.
inline constexpr uint32_t kMaxPunctualShadowSlots =
    (kPunctualShadowAtlasSize / kPunctualShadowMinTileSize) * (kPunctualShadowAtlasSize / kPunctualShadowMinTileSize);

// Edge length of a size class, class 0 being the largest.
[[nodiscard]] uint32_t punctualShadowTileSizeForClass(uint32_t sizeClass);

// Projected pixel radius of a light's sphere of influence -- the priority
// signal. Uses the same projScaleY (viewportHeight * 0.5 * proj[1][1]) the
// mesh-LOD selection uses, so "how big does this look" means the same thing
// across the engine. Lights at or behind the eye clamp to a huge radius rather
// than going negative: something wrapped around the camera is not a candidate
// for demotion.
[[nodiscard]] float punctualShadowProjectedRadius(const glm::vec3& lightPosition,
                                                  float range,
                                                  const glm::vec3& cameraPosition,
                                                  float projScaleY);

// Size class for a light of the given projected radius. A tile is "enough" when
// its edge covers the light's projected footprint, so the thresholds are the
// tile sizes themselves.
//
// isPoint demotes the result one class. A point light pays for six tiles rather
// than one, so ranking it against a spot on the same threshold lets a single
// point light swallow six of the largest tiles and starve everything behind it.
// Each cube face also only covers 90 degrees of the sphere, so a face genuinely
// warrants less resolution than the light's whole projected footprint implies --
// the demotion is a coarse stand-in for that, not just a budget hack.
[[nodiscard]] uint32_t punctualShadowSizeClassForRadius(float projectedRadius, bool isPoint = false);

// Floor for the punctual shadow near plane. Small enough that geometry hugging
// the bulb still rasterizes; see punctualShadowNearPlane for why the plane is
// usually pushed well past this.
inline constexpr float kMinPunctualShadowNearPlane = 0.05f;

// Fraction of a light's range used as its shadow near plane.
//
// A perspective projection spends most of its [0,1] depth range close to the
// near plane, so a fixed 0.05 near against a range of 16 (a 320:1 ratio) leaves
// everything past a couple of units crammed into the last ~2% of the range --
// which is exactly where the receiving geometry actually is. Scaling the near
// plane with the range keeps that ratio fixed at 50:1 no matter how far the
// light reaches, so precision does not silently degrade as ranges grow.
inline constexpr float kPunctualShadowNearPlaneRangeFraction = 0.02f;

// Near plane for a light of the given range.
[[nodiscard]] float punctualShadowNearPlane(float range);

// Widest half-angle a spot shadow will project with. The perspective FOV is
// twice this, so the clamp keeps the frustum strictly under 180 degrees where
// the projection would otherwise blow up.
inline constexpr float kMaxSpotShadowHalfAngleRadians = 1.5533f; // ~89 degrees

// Written into a light's shadow-slot field when it gets no tile this frame (not
// a caster, culled, or out of budget). The shader reads it as "fully lit" and
// skips the atlas fetch entirely, so the sentinel has to survive the float
// round-trip through GpuLight -- see punctualShadowSlotToFloat below.
inline constexpr uint32_t kInvalidPunctualShadowSlot = 0xFFFFFFFFu;

// A tile's pixel rect inside the atlas, which is what the shadow pass turns
// into a viewport and scissor.
struct ShadowAtlasRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t size = 0;
};

// Per-slot record the fragment shader reads through a buffer-device-address
// pointer, mirroring the GpuShadowSlot block in the GLSL. std430 puts the mat4
// at 0 and each vec4 on a 16-byte boundary, giving a 96-byte stride.
struct GpuShadowSlot {
    glm::mat4 viewProjection{1.0f};
    // xy = the tile's UV origin in the atlas, zw = its UV extent. Shading maps
    // light-space NDC into [0,1] and then through this into the tile.
    glm::vec4 atlasUvOffsetScale{0.0f};
    // x = constant depth bias, y = normal-offset bias in world units,
    // z = one tile texel in atlas UV (the PCF step), w unused.
    glm::vec4 params{0.0f};
};

static_assert(sizeof(GpuShadowSlot) == 96);
static_assert(offsetof(GpuShadowSlot, viewProjection) == 0);
static_assert(offsetof(GpuShadowSlot, atlasUvOffsetScale) == 64);
static_assert(offsetof(GpuShadowSlot, params) == 80);

// A tile rect expressed as the UV offset/scale pair the shader needs. Tiles no
// longer have a fixed size, so a slot's rect cannot be derived from its index
// and travels with the slot record instead.
[[nodiscard]] glm::vec4 shadowAtlasRectUvOffsetScale(const ShadowAtlasRect& rect);

// GpuLight carries the slot index in a float field, so the sentinel has to
// become a value the shader can test without an exact-equality trap. These two
// pin that encoding down in one place: a valid slot round-trips exactly (slot
// counts are far below the 24-bit exact-integer range of a float), and the
// invalid slot becomes -1.0, which the shader tests with a < 0.0 compare.
[[nodiscard]] float punctualShadowSlotToFloat(uint32_t slot);
[[nodiscard]] uint32_t punctualShadowSlotFromFloat(float encoded);

// A point light is shadowed by six 90-degree faces, one per signed axis, packed
// into six consecutive atlas slots. Six tiles per light against 64 total is what
// makes a budget necessary rather than optional.
inline constexpr uint32_t kPointShadowFaceCount = 6;

// Face order is +X, -X, +Y, -Y, +Z, -Z.
//
// This is the one convention that has to hold on both sides: the CPU builds a
// projection per face, and the shader picks a face from the light-to-fragment
// direction. If the two disagree, every point light samples the wrong tile --
// which looks like plausible-but-wrong shadows rather than an obvious failure.
// pointShadowFaceDirection and pointShadowFaceIndex are inverses of each other,
// and a unit test pins that they agree for directions all over the sphere.
[[nodiscard]] glm::vec3 pointShadowFaceDirection(uint32_t face);

// Face whose frustum contains this direction: the axis with the largest
// magnitude, resolved by sign. Mirrored verbatim in simple_bindless.frag.
[[nodiscard]] uint32_t pointShadowFaceIndex(const glm::vec3& direction);

// Light-space view-projection for one cube face. The FOV is exactly 90 degrees
// so the six faces tile the sphere with no gaps and no overlap.
[[nodiscard]] glm::mat4 computePointShadowFaceViewProjection(const glm::vec3& position,
                                                             uint32_t face,
                                                             float range,
                                                             float nearPlane = 0.0f);

// Quadtree tile allocator, reset once per frame and filled in priority order.
//
// The atlas starts as a grid of largest-class tiles. Allocating a smaller class
// splits a larger free tile into four and keeps three for later, so a frame that
// wants a few big tiles and many small ones packs without either class having a
// reserved region it cannot lend out. Nothing is freed mid-frame and the whole
// structure is rebuilt each frame, so there is no coalescing pass and no
// fragmentation that outlives a frame.
class PunctualShadowAtlasAllocator final {
public:
    PunctualShadowAtlasAllocator()
    {
        reset();
    }

    void reset();

    // Reserves one tile of the given size class. Returns false when the atlas
    // cannot fit it; callers treat that as "this light is unshadowed this
    // frame" rather than an error, so an over-budget scene degrades rather than
    // failing.
    [[nodiscard]] bool allocate(uint32_t sizeClass, ShadowAtlasRect& outRect);

    // Reserves `count` tiles of one size class, all or nothing. Cube faces need
    // this: a light that got four of six faces would sample cleared tiles for
    // the other two, which is worse than not casting at all. On failure nothing
    // is consumed.
    [[nodiscard]] bool allocateRange(uint32_t sizeClass, uint32_t count, std::vector<ShadowAtlasRect>& outRects);

    [[nodiscard]] uint32_t allocatedCount() const
    {
        return allocatedCount_;
    }

    // Atlas area handed out so far, as a fraction of the whole. Reported in the
    // debug panel because with mixed tile sizes a slot count no longer says
    // anything about how full the atlas is.
    [[nodiscard]] float occupancy() const;

private:
    [[nodiscard]] bool allocateNode(uint32_t sizeClass, ShadowAtlasRect& outRect);

    // Free tiles per size class, largest class first.
    std::array<std::vector<ShadowAtlasRect>, kPunctualShadowSizeClassCount> freeTiles_{};
    uint32_t allocatedCount_ = 0;
    uint64_t allocatedArea_ = 0;
};

// Light-space view-projection for one spot light. The perspective FOV is the
// full cone angle (2 * outer) so the lit cone inscribes the tile, and the far
// plane is the light's range -- exactly where the shading falloff reaches zero,
// so nothing that could receive light falls outside the depth range.
//
// A nearPlane of 0 (the default) derives one from the range via
// punctualShadowNearPlane; pass a positive value to override it.
[[nodiscard]] glm::mat4 computeSpotShadowViewProjection(const glm::vec3& position,
                                                        const glm::vec3& direction,
                                                        float outerAngleRadians,
                                                        float range,
                                                        float nearPlane = 0.0f);

// Frustum of a spot shadow projection, used to cull casters down to the ones
// that can actually write into the slot.
[[nodiscard]] Frustum computeSpotShadowFrustum(const glm::mat4& viewProjection);

} // namespace ve::renderer
