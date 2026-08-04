#pragma once

// GPU-free core for irradiance-probe global illumination.
//
// Three pieces live here, all free of Vulkan state so they can be unit tested on
// the CPU the way ClusterGrid.h, CascadeMath.h, PunctualShadowAtlas.h and
// VolumetricFog.h are:
//   * the octahedral direction <-> square mapping each probe stores through,
//   * the probe grid's index/position round trip, and
//   * the trilinear blend weights a shading point uses over the eight probes
//     surrounding it.
//
// Why probes and not DDGI-style ray tracing: this engine's target device reports
// neither VK_KHR_ray_query nor VK_KHR_acceleration_structure, so probe radiance
// has to be gathered by rasterising the scene from each probe rather than by
// tracing rays. That is the classic irradiance-volume approach, and it reuses
// the cubemap capture and irradiance convolution the IBL path already has.
//
// The octahedral convention is deliberately identical to the one already in
// simple.frag / ssr_trace.frag / gtao.frag rather than a second one of its own.
// Two mappings that disagree would produce lighting that is wrong by a rotation
// or a fold -- which reads as a shading bug, far from the mapping that caused
// it.

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

// Probe grid dimensions. Deliberately coarse: irradiance is low frequency, and
// the cost of a probe is a scene capture, so resolution goes into update rate
// rather than into probe count.
inline constexpr uint32_t kProbeGridX = 8;
inline constexpr uint32_t kProbeGridY = 4;
inline constexpr uint32_t kProbeGridZ = 8;
inline constexpr uint32_t kProbeCount = kProbeGridX * kProbeGridY * kProbeGridZ;

// Per-probe octahedral tile resolution, before the one-texel border each side
// that makes hardware bilinear filtering work across the octahedral seam.
inline constexpr uint32_t kProbeIrradianceResolution = 8;
// Depth (visibility) is stored at higher resolution than irradiance: it is what
// stops light leaking through walls, and that needs sharper detail than the
// irradiance itself does.
inline constexpr uint32_t kProbeDepthResolution = 16;

// Where the grid sits in the world. Kept as data rather than constants so the
// grid can be fitted to a scene later without touching the mapping code.
struct ProbeGridBounds {
    // World position of probe (0, 0, 0).
    glm::vec3 origin{0.0f};
    // World-space distance between adjacent probes on each axis.
    glm::vec3 spacing{1.0f};
};

// Direction to a point on the [0,1]^2 octahedral square, and back.
//
// Mirrors octEncode/octDecode in the shaders exactly, including the sign
// handling on the lower hemisphere fold. A unit test pins the round trip over
// the whole sphere.
[[nodiscard]] glm::vec2 octahedralEncode(const glm::vec3& direction);
[[nodiscard]] glm::vec3 octahedralDecode(const glm::vec2& encoded);

// Linear probe index from grid coordinates, and its inverse. X varies fastest,
// matching how the probe atlas is laid out.
[[nodiscard]] uint32_t probeIndex(uint32_t x, uint32_t y, uint32_t z);
[[nodiscard]] glm::uvec3 probeCoord(uint32_t index);

// World position of a probe.
[[nodiscard]] glm::vec3 probeWorldPosition(uint32_t index, const ProbeGridBounds& bounds);

// The eight probes surrounding a shading point, as a base coordinate plus the
// fractional position within that cell.
struct ProbeBlend {
    // Coordinate of the lower corner probe; the other seven are +1 on each axis.
    glm::uvec3 baseCoord{0};
    // Trilinear weights along each axis, in [0, 1].
    glm::vec3 fraction{0.0f};
};

// Clamps to the grid rather than extrapolating, so a point outside the volume
// takes the nearest probes' irradiance instead of an invented value.
[[nodiscard]] ProbeBlend probeBlendAt(const glm::vec3& worldPosition, const ProbeGridBounds& bounds);

// Weight of one of the eight corners of a ProbeBlend cell. cornerIndex bit 0 is
// the X offset, bit 1 Y, bit 2 Z -- the same order the shader unpacks.
[[nodiscard]] float probeCornerWeight(const ProbeBlend& blend, uint32_t cornerIndex);

// Grid coordinate of one corner, clamped to the grid.
[[nodiscard]] glm::uvec3 probeCornerCoord(const ProbeBlend& blend, uint32_t cornerIndex);

} // namespace ve::renderer
