#pragma once

// The GPU-free half of ray-traced sun shadows: which draw items cast, how an
// instance's transform is laid out, how far a shadow ray starts off its
// surface, and the binary contracts the fragment shader reads. Everything that
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

#include <array>
#include <cstddef>
#include <cstdint>

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

// Where a mesh vertex keeps what the alpha test needs, in floats. The fragment
// shader reads candidate triangles' UVs straight out of the vertex buffer by
// device address, so these restate renderer::Vertex for GLSL; the .cpp pins
// both against the struct with static_asserts, and
// tools/check_shader_constants.py pins the GLSL copies in
// src/shaders/ray_traced_shadows.glsl against these.
inline constexpr uint32_t kRtVertexStrideFloats = 15;
inline constexpr uint32_t kRtVertexUvOffsetFloats = 6;

// The skinned caster. Its BLAS is built from positions a compute pass skins out
// every frame that traces (skin_positions.comp), three floats per vertex, in
// workgroups of this many threads.
inline constexpr uint32_t kRtSkinnedPositionStrideFloats = 3;
inline constexpr uint32_t kRtPreSkinLocalSize = 64;

// Workgroups that cover vertexCount with kRtPreSkinLocalSize threads each; the
// shader returns early past the end.
[[nodiscard]] uint32_t rayTracingPreSkinGroupCount(uint32_t vertexCount);

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
// Mirrors RtGeometryRecord in src/shaders/simple_bindless.frag (two uvec2).
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

} // namespace ve::renderer
