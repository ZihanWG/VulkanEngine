#pragma once

// The GPU-free half of ray-traced sun shadows: which draw items cast, how an
// instance's transform is laid out, how far a shadow ray starts off its
// surface, when the top-level structure has to be rebuilt, and the binary
// contracts the fragment shader reads. Everything that
// touches Vulkan -- acceleration structures, the build pass, descriptors --
// lives in renderer/RayTracingScene.h.
//
// Kept apart for the same reason as renderer/DrawItemBatching.h: the failure
// mode of every function here is a plausible wrong picture rather than an
// error, so each one is pinned by tests/test_ray_traced_shadows.cpp instead of
// being inferred from a capture. See docs/ray_traced_shadows.md.

#include "renderer/Bounds.h"
#include "renderer/DrawItemBatching.h"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ve::renderer {

// Bits of FrameConstants::rayTracedShadow.z. Mirrored by kRtShadowFlag* in
// src/shaders/object_frame_data.glsl. Zero means the main pass takes the
// virtual shadow map or the cascades exactly as it would without ray query.
inline constexpr uint32_t kRtShadowFlagEnabled = 1u;
// Writes the ray-traced visibility term on its own instead of shading.
inline constexpr uint32_t kRtShadowFlagDebugVisibility = 2u;
// Colours each pixel by whether the ray and the cascades agree. Wins over the
// visibility view when both are set.
inline constexpr uint32_t kRtShadowFlagDebugCompareCascades = 4u;
// The main pass reads the ray-traced shadow mask where it covers the fragment
// and traces inline only where it does not. See RayTracedShadowMask.h.
inline constexpr uint32_t kRtShadowFlagScreenMask = 8u;
// Colours each pixel by where its visibility came from: the mask, checked
// against an inline ray, or the inline fallback. Wins over both views above.
inline constexpr uint32_t kRtShadowFlagDebugMaskSource = 16u;

// Where a mesh vertex keeps what the alpha test needs, in floats. The fragment
// shader reads candidate triangles' UVs straight out of the vertex buffer by
// device address, so these restate renderer::Vertex for GLSL; the .cpp pins
// both against the struct with static_asserts, and
// tools/check_shader_constants.py pins the GLSL copies in
// src/shaders/ray_traced_shadows.glsl against these.
inline constexpr uint32_t kRtVertexStrideFloats = 15;
inline constexpr uint32_t kRtVertexUvOffsetFloats = 6;

// The skinned caster. Its BLAS is built from positions a compute pass skins out
// on every frame that traces with the pose moved (skin_positions.comp), three
// floats per vertex, in workgroups of this many threads.
inline constexpr uint32_t kRtSkinnedPositionStrideFloats = 3;
inline constexpr uint32_t kRtPreSkinLocalSize = 64;

// Workgroups that cover vertexCount with kRtPreSkinLocalSize threads each; the
// shader returns early past the end.
[[nodiscard]] uint32_t rayTracingPreSkinGroupCount(uint32_t vertexCount);

// The ray-traced shadow mask's square workgroup (ray_traced_shadow_mask.comp).
inline constexpr uint32_t kRtMaskLocalSize = 8;

// Workgroups along one axis that cover `pixels` with kRtMaskLocalSize each; the
// shader returns early outside the sub-rect.
[[nodiscard]] uint32_t rayTracedShadowMaskGroupCount(uint32_t pixels);

// Whether this frame traces the mask ahead of the main pass. The mask is only
// as wide as the depth prepass: without one there is no depth to trace from,
// and every fragment traces inline exactly as before the mask existed.
[[nodiscard]] constexpr bool
rayTracedShadowMaskActive(bool tracing, bool depthPrepass, bool maskAvailable, bool enabled)
{
    return tracing && depthPrepass && maskAvailable && enabled;
}

// Whether the frame the mask turns on must rebuild the top-level structure
// whether or not anything moved. A frame that keeps the structure relies on the
// barrier the building frame recorded after its build, and that barrier names
// only the stages that read the structure on that frame. Until the mask, the
// fragment stage was the only one; the mask adds the compute stage, which a
// structure built while the mask was off was never made visible to.
[[nodiscard]] constexpr bool rayTracedShadowMaskNeedsFreshBuild(bool activeNow, bool activeWhenLastBuilt)
{
    return activeNow && !activeWhenLastBuilt;
}

// The mask dispatch's push block. Mirrors MaskPushConstants in
// src/shaders/ray_traced_shadow_mask.comp, whose reflected offsets
// tests/golden/shader_interface.txt pins.
struct RayTracedShadowMaskPushConstants {
    // FrameConstants and ObjectFrameData for this frame slot, by address.
    uint64_t frameConstants = 0;
    uint64_t objectFrameData = 0;
    // The render-resolution sub-rect the depth prepass drew, and 1 / it.
    glm::uvec2 extent{0u};
    glm::vec2 inverseExtent{0.0f};
    // Inverse of the jittered view-projection the prepass rasterised with.
    glm::mat4 inverseViewProjection{1.0f};
};
static_assert(offsetof(RayTracedShadowMaskPushConstants, objectFrameData) == 8);
static_assert(offsetof(RayTracedShadowMaskPushConstants, extent) == 16);
static_assert(offsetof(RayTracedShadowMaskPushConstants, inverseExtent) == 24);
static_assert(offsetof(RayTracedShadowMaskPushConstants, inverseViewProjection) == 32);
static_assert(sizeof(RayTracedShadowMaskPushConstants) == 96);

// The pre-skin dispatch's push block, all device addresses. Mirrors
// PreSkinPushConstants in src/shaders/skin_positions.comp, whose reflected
// offsets tests/golden/shader_interface.txt pins.
struct RayTracingPreSkinPushConstants {
    // renderer::Vertex stream, bind pose; only the position is read.
    uint64_t sourceVertices = 0;
    // renderer::SkinningVertex stream, parallel to the vertices.
    uint64_t skinning = 0;
    // This frame slot's joint palette; the shader reads the current half.
    uint64_t jointPalette = 0;
    uint64_t skinnedPositions = 0;
    uint32_t vertexCount = 0;
};
static_assert(offsetof(RayTracingPreSkinPushConstants, skinning) == 8);
static_assert(offsetof(RayTracingPreSkinPushConstants, jointPalette) == 16);
static_assert(offsetof(RayTracingPreSkinPushConstants, skinnedPositions) == 24);
static_assert(offsetof(RayTracingPreSkinPushConstants, vertexCount) == 32);
static_assert(sizeof(RayTracingPreSkinPushConstants) == 40);

// One row per draw item, indexed by the instance's custom index -- which is the
// draw item's frameDataIndex, so the same index also reaches its
// ObjectFrameData and therefore the alpha cutoff, base colour texture and base
// colour alpha the cascades' masked shadow pass reads. Only the geometry is
// new: where its vertices are, and where its first index is.
//
// Mirrors RtGeometryRecord in src/shaders/ray_traced_shadow_trace.glsl (two uvec2).
struct RayTracingGeometryRecord {
    uint64_t vertexAddress = 0;
    // Already advanced to the draw item's firstIndex, so the shader indexes it
    // with the primitive index the ray query reports, which is relative to the
    // BLAS geometry -- and the geometry starts at firstIndex.
    uint64_t indexAddress = 0;
};
static_assert(sizeof(RayTracingGeometryRecord) == 16);

// What a draw item contributes to the scene the rays see. Matches the cascades
// on purpose (docs/transparency.md): opaque casts, a cutout casts what its
// alpha test keeps, and blended geometry casts nothing.
enum class RayTracedCasterKind : uint8_t {
    None,
    Opaque,
    AlphaTested,
};

[[nodiscard]] RayTracedCasterKind rayTracedCasterKind(RenderBucket bucket);

// glm::mat4 is column-major; VkTransformMatrixKHR is a 3x4 ROW-major matrix.
// Copying the first twelve floats of the glm matrix compiles, builds, traces,
// and puts every instance's translation in the wrong place, which is why this
// is a function with a test rather than a memcpy at the call site.
[[nodiscard]] std::array<float, 12> rayTracingInstanceTransform(const glm::mat4& model);

// How far a shadow ray's origin moves along the surface normal per unit of
// distance from the camera, in world units.
//
// The rays are traced against level 0 of every mesh, while the raster pass
// drew whichever level the cull chose. Under screen-space error selection that
// level is at most lodErrorPixels off level 0 on screen, so offsetting by that
// many pixels' world footprint keeps a coarse level's surface from shadowing
// itself against the fine one. pixelFootprints adds a margin on top for
// floating-point precision, which degrades with distance the same way.
//
// One pixel's footprint at distance d is d * 2 tan(fov/2) / height; this
// returns the coefficient of d. Zero height returns zero rather than infinity.
[[nodiscard]] float rayTracedShadowOffsetPerDistance(float pixelFootprints,
                                                     float lodErrorPixels,
                                                     float verticalFovRadians,
                                                     uint32_t renderHeight);

// How far a sun ray has to travel to find any occluder: the diagonal of the
// world bounds of everything drawn this frame, with a margin. Every receiver is
// one of those draw items and so is every caster, so no occluder of any shaded
// pixel can be further away than that. The camera's far plane is the wrong bound
// -- it limits which receivers are drawn, not how far their occluders are -- and
// the TLAS holds casters well outside the view. Invalid bounds give a large
// fallback rather than zero, which would light every pixel.
inline constexpr float kRtFallbackRayLength = 1.0e6f;
[[nodiscard]] float rayTracedShadowRayLength(const Aabb& sceneBounds);

// Rounds value up to a multiple of alignment, which must be a power of two.
// Scratch addresses for acceleration-structure builds must be aligned to
// minAccelerationStructureScratchOffsetAlignment, and a buffer's own allocation
// alignment does not imply it.
[[nodiscard]] uint64_t alignUpPowerOfTwo(uint64_t value, uint64_t alignment);

// What one frame that traces has to do for the top-level structure.
struct RayTracingTopLevelPlan {
    // Record the TLAS build and declare its pass. False when the structure
    // already holds exactly what this frame would build into it.
    bool build = false;
    // Pre-skin and refit the skinned BLAS ahead of the build. Never set without
    // build: a refit moves the bounds the TLAS cached for that instance.
    bool refitSkinned = false;
    // Write this frame slot's instance and geometry-record buffers, which hold
    // an older generation than the one this frame traces. Independent of build,
    // because the fragment shader reads this slot's geometry records through
    // the custom index whether or not the structure was rebuilt.
    bool writeSlot = false;
};

// Decides RayTracingTopLevelPlan from exactly what the build would read: the
// instance records, byte for byte, and the skinned caster's pose. The bytes are
// every instance's transform, custom index, mask, flags and BLAS address in
// order, so a moved object, a reordered draw list (the custom index is the draw
// item's frameDataIndex), an instance added or dropped and a mode change all
// differ; the pose is SkinnedMesh::poseHash, because the skinned BLAS changes
// under an instance record that does not.
//
// Three things are tracked apart, because they go stale apart: the content the
// structure was last built from, the pose the skinned BLAS was last refitted
// to, and per frame slot, the content that slot's buffers hold. A build is
// committed only when it is recorded (commitBuild), so a frame that planned one
// and never recorded it leaves the next frame building.
class RayTracingTopLevelTracker {
public:
    // Forgets everything: the next plan builds, refits and writes.
    void reset(uint32_t frameSlots);

    // skinnedPose is 0 when the instances hold no skinned caster, and never 0
    // when they do. rebuildEveryFrame builds, refits and writes regardless,
    // which is what every frame did before this existed.
    [[nodiscard]] RayTracingTopLevelPlan
    plan(uint32_t frameSlot, std::span<const std::byte> instances, uint64_t skinnedPose, bool rebuildEveryFrame);

    // The build the last plan asked for was recorded. Does nothing when that
    // plan built nothing.
    void commitBuild();

    // Bumped by every plan whose instance records differ from the ones before.
    [[nodiscard]] uint64_t generation() const
    {
        return generation_;
    }

private:
    std::vector<std::byte> content_;
    bool hasContent_ = false;
    uint64_t generation_ = 0;
    // Zero until something this tracker planned has been built.
    uint64_t builtGeneration_ = 0;
    uint64_t refittedPose_ = 0;
    std::vector<uint64_t> slotGenerations_;
    RayTracingTopLevelPlan planned_{};
    uint64_t plannedGeneration_ = 0;
    uint64_t plannedPose_ = 0;
};

} // namespace ve::renderer
