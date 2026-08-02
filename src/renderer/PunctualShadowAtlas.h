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

#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace ve::renderer {

// Atlas geometry. 4096/512 gives an 8x8 grid = 64 slots, which covers the demo
// scene's shadow-casting spots with headroom for point-light faces later. The
// shader only needs the tile UV extent, and that arrives per slot in
// GpuShadowSlot::atlasUvOffsetScale, so these constants stay CPU-side.
inline constexpr uint32_t kPunctualShadowAtlasSize = 4096;
inline constexpr uint32_t kPunctualShadowTileSize = 512;
inline constexpr uint32_t kPunctualShadowTilesPerSide = kPunctualShadowAtlasSize / kPunctualShadowTileSize;
inline constexpr uint32_t kMaxPunctualShadowSlots = kPunctualShadowTilesPerSide * kPunctualShadowTilesPerSide;

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

// Pixel rect of a slot. Slots fill the grid in row-major order.
[[nodiscard]] ShadowAtlasRect punctualShadowSlotRect(uint32_t slot);

// The same rect expressed as the UV offset/scale pair the shader needs.
[[nodiscard]] glm::vec4 punctualShadowSlotUvOffsetScale(uint32_t slot);

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

// Fixed-grid slot allocator, reset once per frame and filled in whatever
// priority order the caller walks its lights in. Uniform tiles make allocation
// a bump counter; variable tile sizes (so a near light gets a sharper map) are
// deliberately left out until there is a priority pass to drive them.
class PunctualShadowAtlasAllocator final {
public:
    void reset()
    {
        nextSlot_ = 0;
    }

    // Returns the next free slot, or kInvalidPunctualShadowSlot once the grid is
    // exhausted. Callers treat exhaustion as "this light is unshadowed this
    // frame" rather than an error, so an over-budget scene degrades instead of
    // failing.
    [[nodiscard]] uint32_t allocate();

    // Reserves `count` consecutive slots and returns the first, or
    // kInvalidPunctualShadowSlot when that many do not remain. Consecutive
    // matters for cube faces: the light stores only a base slot and the shader
    // adds the face index to it, so the six have to be adjacent.
    [[nodiscard]] uint32_t allocateRange(uint32_t count);

    [[nodiscard]] uint32_t allocatedCount() const
    {
        return nextSlot_;
    }

    [[nodiscard]] bool full() const
    {
        return nextSlot_ >= kMaxPunctualShadowSlots;
    }

private:
    uint32_t nextSlot_ = 0;
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
