#pragma once

// GPU-free core for froxel volumetric fog.
//
// Two pieces live here, both free of Vulkan state so they can be unit tested on
// the CPU the way ClusterGrid.h, CascadeMath.h, and PunctualShadowAtlas.h are:
//   * the froxel depth distribution, and
//   * the front-to-back scattering integration each column performs.
//
// The fog volume is a 3D texture over the view frustum: XY are screen tiles and
// Z is an exponential depth slicing, the same shape the clustered lighting grid
// uses. It is deliberately *not* the same grid. The light grid is 16x9x24,
// which is far too coarse for fog, and fog wants its own much nearer far plane
// so the slices land where fog is actually visible rather than being spread out
// to the scene's far plane. What is shared is the addressing scheme, so a fog
// froxel can find the light cluster covering it and reuse its light list.

#include "renderer/RuntimeSettings.h"

#include <cstdint>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace ve::renderer {

// Fog volume dimensions. XY is roughly a twelfth of 1080p, which is the usual
// trade for this technique: fog is low frequency, so the resolution goes into Z
// where the integration happens.
inline constexpr uint32_t kFogGridX = 160;
inline constexpr uint32_t kFogGridY = 90;
inline constexpr uint32_t kFogGridZ = 64;
inline constexpr uint32_t kFogFroxelCount = kFogGridX * kFogGridY * kFogGridZ;

// Near plane of the fog volume. Independent of the camera's, and larger,
// because a camera near plane of 0.1 spends most of the exponential slice
// distribution in the first metre where almost nothing is visible.
inline constexpr float kFogNearPlane = 0.5f;

// How far fog is evaluated. Past this the integration is clamped to its last
// slice, so distant geometry keeps the fog colour it had at the volume's edge
// instead of abruptly clearing.
//
// Defined with FogSettings in RuntimeSettings.h, since it is that struct's
// default, and re-exported here so the fog math reads with the rest of its
// constants.
using ve::kDefaultFogMaxDistance;

// View-space depth at the near edge of a slice, and its inverse.
//
// These two must round-trip: the injection pass turns a slice index into a
// world position with the first, and the apply pass turns a fragment's view
// depth back into a slice with the second. A mismatch shows up as fog that
// slides relative to the geometry it is supposed to sit in front of, so a unit
// test pins the round trip.
[[nodiscard]] float fogSliceViewDepth(float slice, float maxDistance);
[[nodiscard]] float fogViewDepthToSlice(float viewDepth, float maxDistance);

// Thickness of one slice in view-space units. Non-uniform: slices grow
// exponentially with depth, and the integration needs the real thickness to
// convert an extinction coefficient into a per-slice transmittance.
[[nodiscard]] float fogSliceThickness(uint32_t slice, float maxDistance);

// One froxel's medium before integration, as the injection pass writes it.
struct FogFroxelSample {
    // Light scattered toward the eye from inside this froxel, per unit length.
    glm::vec3 inScattering{0.0f};
    // Extinction coefficient (absorption + out-scattering), per unit length.
    float extinction = 0.0f;
};

// The running result of the front-to-back march, as the integration pass stores
// it per froxel: total light gathered up to and including this slice, plus how
// much of the background still shows through.
struct FogIntegratedSample {
    glm::vec3 inScattering{0.0f};
    float transmittance = 1.0f;
};

// Integrates one slice into the running total.
//
// Uses the analytic integral of scattering over a homogeneous slab rather than
// a midpoint sample: over a thick slice, scaling by (1 - exp(-sigma*d)) / sigma
// accounts for light that is scattered in and then partly absorbed before
// leaving the slice, which a naive `scatter * thickness` overestimates badly at
// high density. That difference is what makes dense fog stay grey rather than
// blowing out to white.
[[nodiscard]] FogIntegratedSample integrateFogSlice(const FogIntegratedSample& accumulated,
                                                    const FogFroxelSample& slice,
                                                    float sliceThickness);

// Henyey-Greenstein phase function: how much light travelling along
// `lightDirection` scatters toward `viewDirection`. anisotropy in (-1, 1);
// positive values scatter forward, which is what produces the bright halo when
// looking toward a light through fog.
[[nodiscard]] float henyeyGreensteinPhase(const glm::vec3& viewDirection,
                                          const glm::vec3& lightDirection,
                                          float anisotropy);

// Largest value henyeyGreensteinPhase can return for a given anisotropy, over
// every possible scattering angle.
//
// This exists to make the fog's per-light culling *conservative* rather than
// approximate. A froxel can bound a light's possible contribution before doing
// any expensive work -- colour * intensity * attenuation * this -- and skip the
// light only when even that bound is negligible. Because the bound can never
// understate the real contribution, a light that would have been visible is
// never dropped; the cull only ever removes work whose result was going to be
// below the threshold anyway.
//
// A unit test pins the bounding property directly against the phase function
// over a sweep of angles and anisotropies.
[[nodiscard]] float maxHenyeyGreensteinPhase(float anisotropy);

// Light cluster covering a fog froxel's centre.
//
// The fog volume and the clustered lighting grid are different resolutions with
// different depth ranges, so a fog froxel has to be mapped into the light grid
// explicitly to reuse its light list. This is the one place the two systems have
// to agree: getting it wrong hands a froxel some other cluster's lights, which
// looks like fog lit slightly wrong rather than anything obviously broken.
//
// Defined as "the cluster the fragment shader would pick for a fragment at this
// froxel's screen position and view depth", and a unit test asserts exactly
// that against ClusterGrid.h's own clusterIndex rather than re-deriving it.
[[nodiscard]] uint32_t fogFroxelClusterIndex(uint32_t froxelX,
                                             uint32_t froxelY,
                                             uint32_t froxelZ,
                                             float fogMaxDistance,
                                             float clusterZNear,
                                             float clusterZFar);

// Density at a world height for exponential height fog. `falloff` of zero gives
// uniform density; larger values pull the fog down toward `baseHeight`.
[[nodiscard]] float fogHeightDensity(float worldHeight, float baseHeight, float falloff);

} // namespace ve::renderer
