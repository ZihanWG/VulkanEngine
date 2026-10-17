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
#include <glm/vec3.hpp>

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
// The mask's resolve writes how far each ray went instead of whether it reached
// the sun, and the main pass shows that term on its own.
inline constexpr uint32_t kRtShadowFlagDebugHitDistance = 32u;
// The mask's resolve writes the soft shadow filter's penumbra radius instead of
// the visibility, and the main pass shows that term on its own.
inline constexpr uint32_t kRtShadowFlagDebugPenumbraRadius = 64u;

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

// The ray-traced shadow mask's square workgroup, shared by its trace and
// resolve dispatches.
inline constexpr uint32_t kRtMaskLocalSize = 8;

// Workgroups along one axis that cover `pixels` with kRtMaskLocalSize each; the
// shader returns early outside the sub-rect.
[[nodiscard]] uint32_t rayTracedShadowMaskGroupCount(uint32_t pixels);

// Added to a mask visibility traced from a surface facing away from the sun.
// Mirrors kRtMaskBackFacingBias in ray_traced_shadows.glsl; the CPU reads it
// only to strip it from the debug readback.
inline constexpr float kRtMaskBackFacingBias = 2.0f;

// Most rays a pixel traces in the reference mode, and the clamp on
// RayTracedShadowSettings::debugReferenceSamples. Mirrored in
// ray_traced_shadows.glsl.
inline constexpr uint32_t kRtReferenceMaxSamples = 64;

// Full-resolution pixels per trace texel along each axis: 2 when the mask
// traces at half resolution, 1 otherwise -- and always 1 in the reference mode,
// which is meant to be the converged answer for every pixel.
[[nodiscard]] constexpr uint32_t rayTracedShadowTraceScale(bool halfResolution, uint32_t referenceSamples)
{
    return referenceSamples > 0 ? 1u : (halfResolution ? 2u : 1u);
}

// Trace texels along one axis: enough that texel t, standing for full-resolution
// pixel t * scale, reaches the last pixel of the sub-rect.
[[nodiscard]] constexpr uint32_t rayTracedShadowTraceExtent(uint32_t fullPixels, uint32_t scale)
{
    return scale == 0 ? 0u : (fullPixels + scale - 1u) / scale;
}

// Whether anything asks for the mask. A soft shadow does by itself: its filter
// has to see the visibility before anything is shaded with it, which only the
// mask provides, and so does the reference mode. A hard one only asks through
// screenSpaceMask, which measured as a cost (docs/ray_traced_shadows.md#cost).
[[nodiscard]] constexpr bool
rayTracedShadowMaskRequested(bool screenSpaceMask, float sunAngularDiameterDegrees, uint32_t referenceSamples)
{
    return screenSpaceMask || sunAngularDiameterDegrees > 0.0f || referenceSamples > 0;
}

// Whether this frame traces the mask ahead of the main pass. The mask is only
// as wide as the depth prepass: without one there is no depth to trace from,
// and every fragment traces inline exactly as before the mask existed.
[[nodiscard]] constexpr bool
rayTracedShadowMaskActive(bool tracing, bool depthPrepass, bool maskAvailable, bool enabled)
{
    return tracing && depthPrepass && maskAvailable && enabled;
}

// --- The soft shadow filter ---------------------------------------------------
//
// Between the trace and the resolve, at the trace's resolution: a radius pass
// that turns the hit distances around each texel into a penumbra radius, then a
// separable bilateral blur over that radius, one axis at a time. See
// docs/ray_traced_shadows.md#the-filter.

// The filter's three dispatches, in recording order. The radius pass and the
// vertical axis write the filter image; the horizontal axis writes the trace
// image back, which is the ping-pong that lets two images serve three passes.
enum class RayTracedShadowFilterStage : uint32_t {
    Radius = 0,
    BlurHorizontal = 1,
    BlurVertical = 2,
};
inline constexpr uint32_t kRayTracedShadowFilterStageCount = 3;

// Whether this frame filters the mask. Only a frame that traces the mask has
// anything to filter, and only a soft shadow needs it: a hard shadow's single
// ray is the answer. Never the reference mode, which is the converged answer the
// filter is judged against, and never under the hit-distance view, which shows
// the rays as traced: the radius pass writes over the hit distance it reads.
[[nodiscard]] constexpr bool rayTracedShadowFilterActive(
    bool maskActive, bool filter, float sunAngularDiameterDegrees, uint32_t referenceSamples, bool debugHitDistance)
{
    return maskActive && filter && sunAngularDiameterDegrees > 0.0f && referenceSamples == 0 && !debugHitDistance;
}

// Most trace texels the blur reaches either side of a texel, and the clamp on
// the radius. Mirrored in ray_traced_shadows.glsl.
inline constexpr float kRtFilterMaxRadius = 16.0f;
// How much surface one trace texel spans along a screen axis, in world units:
// the width a texel subtends at cameraDistance, combined with distanceStep, how
// far the surface recedes from one texel to the next along that axis. Mirrors
// rtWorldPerTexel in ray_traced_shadows.glsl.
[[nodiscard]] float rayTracedShadowWorldPerTexel(float cameraDistance, float texelsPerUnit, float distanceStep);

// Half the narrowest width the penumbra of an occluder `occluderDistance` along
// the sun ray can have on the receiver -- 2 * distance * tanSunRadius across --
// in texels that each span worldPerTexel, clamped to kRtFilterMaxRadius. Zero
// without an occluder. The narrowest because a kernel wider than the penumbra
// bends its ends towards its middle, where a narrower one only leaves noise.
// Mirrors rtPenumbraRadius in ray_traced_shadows.glsl.
[[nodiscard]] float rayTracedShadowPenumbraRadius(float occluderDistance, float tanSunRadius, float worldPerTexel);

// Trace texels per world unit at one unit from the camera: a width w at camera
// distance d covers w * this / d texels. projectionYScale is the projection
// matrix's [1][1], whose sign the Vulkan Y flip may have turned.
[[nodiscard]] float rayTracedShadowTexelsPerUnit(float projectionYScale, uint32_t fullHeight, uint32_t traceScale);

// --- Sampling the sun's disc -------------------------------------------------
//
// Mirrors of rtConcentricDisk, rtOrthonormalBasis and rtSunConeDirection in
// src/shaders/ray_traced_shadow_trace.glsl. The shader is what runs; these are
// here so the tests can pin what it computes.

// tan(half the sun's angular diameter), the scale of a point on the unit disc
// around the sun's direction. Zero for a sun with no extent: the hard shadow.
[[nodiscard]] float sunConeTangent(float angularDiameterDegrees);

// A point on the unit disc from a point in the unit square, area-preserving and
// continuous (Shirley and Chiu's concentric map).
[[nodiscard]] glm::vec2 concentricDiskSample(glm::vec2 u);

// Two unit vectors completing n to an orthonormal basis, branch-free (Duff et
// al., "Building an Orthonormal Basis, Revisited").
void orthonormalBasis(const glm::vec3& n, glm::vec3& tangent, glm::vec3& bitangent);

// A direction towards a point on the sun's disc; u = (0.5, 0.5) is the centre.
[[nodiscard]] glm::vec3 sunConeDirection(const glm::vec3& towardLight, float tanRadius, glm::vec2 u);

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

// The mask trace's push block. Mirrors TracePushConstants in
// src/shaders/ray_traced_shadow_trace.comp, whose reflected offsets
// tests/golden/shader_interface.txt pins.
struct RayTracedShadowTracePushConstants {
    // FrameConstants and ObjectFrameData for this frame slot, by address.
    uint64_t frameConstants = 0;
    uint64_t objectFrameData = 0;
    // Inverse of the jittered view-projection the prepass rasterised with.
    glm::mat4 inverseViewProjection{1.0f};
    // The render-resolution sub-rect the depth prepass drew, and 1 / it.
    glm::uvec2 fullExtent{0u};
    glm::vec2 inverseFullExtent{0.0f};
    // rayTracedShadowTraceExtent of the sub-rect at traceScale.
    glm::uvec2 traceExtent{0u};
    uint32_t traceScale = 1;
    // Zero, or the reference mode's rays per pixel.
    uint32_t referenceSamples = 0;
};
static_assert(offsetof(RayTracedShadowTracePushConstants, objectFrameData) == 8);
static_assert(offsetof(RayTracedShadowTracePushConstants, inverseViewProjection) == 16);
static_assert(offsetof(RayTracedShadowTracePushConstants, fullExtent) == 80);
static_assert(offsetof(RayTracedShadowTracePushConstants, inverseFullExtent) == 88);
static_assert(offsetof(RayTracedShadowTracePushConstants, traceExtent) == 96);
static_assert(offsetof(RayTracedShadowTracePushConstants, traceScale) == 104);
static_assert(offsetof(RayTracedShadowTracePushConstants, referenceSamples) == 108);
static_assert(sizeof(RayTracedShadowTracePushConstants) == 112);

// The mask resolve's push block. Mirrors ResolvePushConstants in
// src/shaders/ray_traced_shadow_resolve.comp.
struct RayTracedShadowResolvePushConstants {
    uint64_t frameConstants = 0;
    uint32_t traceScale = 1;
    // 1 when the resolve reads the filter image rather than the trace image.
    uint32_t filtered = 0;
    glm::uvec2 fullExtent{0u};
    glm::vec2 inverseFullExtent{0.0f};
    glm::mat4 inverseViewProjection{1.0f};
};
static_assert(offsetof(RayTracedShadowResolvePushConstants, traceScale) == 8);
static_assert(offsetof(RayTracedShadowResolvePushConstants, fullExtent) == 16);
static_assert(offsetof(RayTracedShadowResolvePushConstants, inverseFullExtent) == 24);
static_assert(offsetof(RayTracedShadowResolvePushConstants, inverseViewProjection) == 32);
static_assert(offsetof(RayTracedShadowResolvePushConstants, filtered) == 12);
static_assert(sizeof(RayTracedShadowResolvePushConstants) == 96);

// The filter passes' push block, shared by the radius pass and both blur axes.
// Mirrors FilterPushConstants in src/shaders/ray_traced_shadow_filter.glsl.
struct RayTracedShadowFilterPushConstants {
    // rayTracedShadowTraceExtent of the sub-rect at the trace scale.
    glm::uvec2 traceExtent{0u};
    // The blur's axis, (1, 0) or (0, 1); unused by the radius pass.
    glm::ivec2 direction{0};
    // rayTracedShadowTexelsPerUnit for this frame.
    float texelsPerUnit = 0.0f;
    // sunConeTangent of the sun's angular diameter.
    float tanSunRadius = 0.0f;
};
static_assert(offsetof(RayTracedShadowFilterPushConstants, direction) == 8);
static_assert(offsetof(RayTracedShadowFilterPushConstants, texelsPerUnit) == 16);
static_assert(offsetof(RayTracedShadowFilterPushConstants, tanSunRadius) == 20);
static_assert(sizeof(RayTracedShadowFilterPushConstants) == 24);

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
