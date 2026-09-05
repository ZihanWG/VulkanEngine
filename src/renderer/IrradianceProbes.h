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
// Why probes and not DDGI-style ray tracing: this was written against a device
// -- an Apple M3 through MoltenVK -- that reports neither VK_KHR_ray_query nor
// VK_KHR_acceleration_structure, so probe radiance had to be gathered by
// rasterising the scene from each probe rather than by tracing rays. That is the
// classic irradiance-volume approach, and it reuses the cubemap capture and
// irradiance convolution the IBL path already has.
//
// That premise is device-specific and no longer universal here: an RTX 3080 Ti
// Laptop reports rayQuery, accelerationStructure and VK_KHR_ray_tracing_pipeline,
// so DDGI is available on that machine. Treat the rasterised capture as the
// portability floor, not as the only option, and keep the capability query rather
// than the hardware assumption. (Ampere reports
// rayTracingInvocationReorderReorderingHint = NONE -- shader execution reordering
// is Ada, so a gather here has to tolerate divergence rather than assume it away.)
//
// What a switch would NOT touch is most of this header: the octahedral tile
// layout, the border texel, and the Chebyshev visibility weight describe how
// probe irradiance is stored and sampled, not how its radiance was gathered.
//
// The octahedral convention is deliberately identical to the one already in
// simple.frag / ssr_trace.frag / gtao.frag rather than a second one of its own.
// Two mappings that disagree would produce lighting that is wrong by a rotation
// or a fold -- which reads as a shading bug, far from the mapping that caused
// it.

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
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

// --- Atlas layout ---
//
// Every probe owns one square tile in each of two atlases, irradiance and depth.
// A tile is that probe's octahedral square plus a one-texel border on all four
// sides.
//
// The border is not padding. The octahedral square's edges are seams: texels on
// opposite sides of an edge are neighbours on the sphere but far apart in the
// image. Hardware bilinear filtering knows nothing about that, so a sample near
// an edge would blend against a texel pointing somewhere else entirely. The
// border holds the wrapped copy of the interior texels the seam actually joins
// to, which turns the whole tile into something a plain linear sampler can read.
//
// Tiles run X then Y across an atlas row and Z down the rows, so the tile
// coordinate follows from the linear probe index with the same X-fastest order
// probeIndex uses. One atlas row is therefore one horizontal slab of the grid,
// which is what makes the debug preview readable without counting texels.
inline constexpr uint32_t kProbeBorderTexels = 1;
inline constexpr uint32_t kProbeIrradianceTileSize = kProbeIrradianceResolution + 2 * kProbeBorderTexels;
inline constexpr uint32_t kProbeDepthTileSize = kProbeDepthResolution + 2 * kProbeBorderTexels;

inline constexpr uint32_t kProbeAtlasTilesX = kProbeGridX * kProbeGridY;
inline constexpr uint32_t kProbeAtlasTilesY = kProbeGridZ;
static_assert(kProbeAtlasTilesX * kProbeAtlasTilesY == kProbeCount,
              "The atlas must hold exactly one tile per probe, or probeTileCoord aliases.");

inline constexpr uint32_t kProbeIrradianceAtlasWidth = kProbeAtlasTilesX * kProbeIrradianceTileSize;
inline constexpr uint32_t kProbeIrradianceAtlasHeight = kProbeAtlasTilesY * kProbeIrradianceTileSize;
inline constexpr uint32_t kProbeDepthAtlasWidth = kProbeAtlasTilesX * kProbeDepthTileSize;
inline constexpr uint32_t kProbeDepthAtlasHeight = kProbeAtlasTilesY * kProbeDepthTileSize;

// --- Capture ---
//
// A probe gathers radiance by rasterising the scene from its own position into
// the six faces of a cube, which the convolution below turns into the probe's
// two octahedral tiles. Six rasterisations rather than one octahedral pass
// because the octahedral map is not a projective transform -- no vertex shader
// can produce it.
//
// The face convention, the projections, and the face directions are the ones
// PunctualShadowAtlas.h already defines for point-light shadow cubes. Reused
// rather than re-derived: a second cube convention in the same engine is the
// kind of thing that produces plausible-but-rotated lighting.
inline constexpr uint32_t kProbeCaptureFaceCount = 6;

// Resolution of one captured cube face.
//
// Small on purpose, and the convolution is what sets the bound rather than the
// rasterisation. Every output texel integrates over *every* captured texel, so
// this squared times six is the inner loop: 16 gives 1536 samples per probe,
// which is the same order as DDGI's 256 rays per probe and still cheap enough
// to run several probes a frame. Doubling it would quadruple the convolution.
inline constexpr uint32_t kProbeCaptureFaceResolution = 16;

// Near plane for a capture face.
//
// Deliberately not punctualShadowNearPlane(kProbeMaxDistance), which would put
// it past a metre: a probe sits *inside* the room it is measuring, so geometry
// a few centimetres away still has to rasterise. The depth precision that near
// plane exists to protect does not matter here -- the capture's depth buffer
// only resolves occlusion between surfaces, and distance is written explicitly
// by the fragment shader rather than read back out of depth.
inline constexpr float kProbeCaptureNearPlane = 0.05f;

// Cap on how many probes one frame may capture, which is what bounds the
// capture atlas. The per-frame count is a runtime setting under this.
inline constexpr uint32_t kMaxProbesPerFrame = 16;

// Sharpness of the lobe the depth convolution integrates over.
//
// Visibility wants a much tighter filter than irradiance: a probe's stored
// distance in a direction should describe what is actually in that direction,
// not an average over the hemisphere, or a wall stops occluding anything.
// DDGI uses 50 with a 256-ray budget and leans on temporal accumulation to make
// that many samples enough. Until that phase exists this is deliberately
// blunter, trading some sharpness for a result that does not swim frame to
// frame at 1536 samples.
inline constexpr float kProbeDepthLobeExponent = 20.0f;

// Floor under a probe's visibility weight. A hard zero puts a visible edge in
// the shading wherever the weight crosses it, which reads as a crease that does
// not correspond to any geometry.
inline constexpr float kProbeMinVisibility = 0.02f;

// Floor under the backface weight, for the same reason: a probe exactly edge-on
// to a surface should fade, not vanish.
inline constexpr float kProbeBackfaceFloor = 0.2f;

// Fraction of a probe's previous value kept when it is re-captured.
//
// Lower than a per-frame accumulator like DDGI's would use, because a probe here
// is re-captured once per round-robin cycle rather than every frame -- the same
// hysteresis costs sixty times as much latency. This buys two things: the step
// when a probe is re-captured after the lighting moved becomes a ramp, and the
// sub-texel jitter above averages into extra angular resolution.
inline constexpr float kProbeDefaultHysteresis = 0.7f;

// How much of a surface's indirect light comes from the probe grid rather than
// from the constant ambient term, when a probe captures that surface.
//
// This is what turns one bounce into many: the capture reads the irradiance the
// probes already hold, so the next capture sees light that has bounced once
// more. It is a feedback loop, and the weight is what bounds it -- see
// probeBounceAmplification.
//
// Below one on purpose. The physically exact answer is a full replacement, but
// the grid is coarse and every term feeding it is approximate, so an exact
// feedback gain compounds those approximations as fast as it compounds light.
inline constexpr float kProbeDefaultBounceWeight = 0.5f;

// Distances written into the depth atlas are clamped to this. Two reasons, and
// only the first is about storage: the atlas keeps distance and distance
// squared in a 16-bit float pair, and the squared channel is what runs out of
// range first (this bound squared has to stay comfortably inside the half
// float's ~65504 maximum). The second is that "nothing was hit in this
// direction" needs a finite answer the visibility test can still reason about,
// rather than an infinity that poisons the Chebyshev bound.
inline constexpr float kProbeMaxDistance = 64.0f;
static_assert(kProbeMaxDistance * kProbeMaxDistance < 65504.0f,
              "Squared probe distance must stay representable in a 16-bit float.");

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

// --- Atlas addressing ---
//
// coreResolution is the octahedral square's resolution *without* the border, so
// callers pass kProbeIrradianceResolution or kProbeDepthResolution and the same
// code serves both atlases. Tile size and atlas size follow from it.

// Which tile in the atlas a probe owns, in tiles.
[[nodiscard]] glm::uvec2 probeTileCoord(uint32_t index);

// Top-left texel of a probe's tile, border included.
[[nodiscard]] glm::uvec2 probeTileOrigin(uint32_t index, uint32_t coreResolution);

// Atlas dimensions in texels for a given core resolution.
[[nodiscard]] glm::uvec2 probeAtlasSize(uint32_t coreResolution);

// Atlas UV to sample a probe along a direction.
//
// The result lands inside the probe's own core square, at least half a texel
// away from the tile's outer edge, so a bilinear tap never reaches into the
// neighbouring probe's tile -- it reaches into this tile's border, which is
// exactly what the border was written for.
[[nodiscard]] glm::vec2 probeAtlasUv(uint32_t index, const glm::vec3& direction, uint32_t coreResolution);

// Direction stored at a tile-local texel. Coordinates are tile-local and include
// the border, so the core spans [1, coreResolution] on both axes.
//
// Defined for core texels. A border texel has no direction of its own -- it is a
// copy of one -- so resolve it through probeBorderSource first; passing one here
// clamps into the core rather than inventing an answer.
[[nodiscard]] glm::vec3 probeTexelDirection(uint32_t x, uint32_t y, uint32_t coreResolution);

// The core texel a border texel copies, in tile-local coordinates.
//
// The octahedral square folds onto itself at its edges: crossing an edge
// re-enters the square on the same edge, mirrored about that edge's midpoint.
// So a border row or column is the adjacent core row or column *reversed*, and
// a border corner is the diagonally opposite core corner.
//
// A coordinate that is already a core texel is returned unchanged, which makes
// the function total and lets the border pass run over a whole tile without
// branching on the caller's side.
[[nodiscard]] glm::ivec2 probeBorderSource(int32_t x, int32_t y, uint32_t coreResolution);

// --- Capture addressing ---

// View-projection a probe rasterises one cube face with. Thin wrapper over the
// point-shadow cube projection so the two cannot drift apart; only the near and
// far planes differ.
//
// jitterTexels shifts the whole face by a sub-texel amount, in texels. It exists
// because the capture is otherwise *deterministic*: the same fixed directions
// every time, so re-capturing an unchanged scene reproduces the previous result
// exactly and averaging successive captures would gain nothing at all. Moving
// the sample grid between updates is what turns accumulation from a no-op into
// a genuine increase in angular resolution.
[[nodiscard]] glm::mat4 probeCaptureFaceViewProjection(const glm::vec3& probePosition,
                                                       uint32_t face,
                                                       const glm::vec2& jitterTexels = glm::vec2{0.0f});

// World direction a capture texel looks along, under the same jitter the face
// was rasterised with. Texel centres, so the (x, y) range is [0, resolution).
//
// The jitter has to be applied with the opposite sign to the projection's: a
// projection that pushes the image right by j texels puts the direction that was
// at x - j under texel x. Getting that backwards doubles the offset instead of
// cancelling it, which shows up as a probe whose stored radiance is smeared
// rather than as anything obviously wrong -- so a round-trip test pins it.
[[nodiscard]] glm::vec3 probeCubeTexelDirection(uint32_t face,
                                                uint32_t x,
                                                uint32_t y,
                                                uint32_t resolution,
                                                const glm::vec2& jitterTexels = glm::vec2{0.0f});

// Sub-texel offset for one update, from a low-discrepancy sequence so successive
// captures spread over the texel instead of clustering the way random values do.
[[nodiscard]] glm::vec2 probeCaptureJitter(uint32_t updateIndex);

// Solid angle a capture texel subtends, in steradians.
//
// Not optional and not uniform: a cube face's texels shrink toward its corners
// by up to a factor of three, so summing radiance without this weight
// systematically over-counts the corners of every face. The result is a smooth
// bias in the gathered irradiance -- it looks like the GI being "a bit off"
// rather than like an integration bug, which is why a test pins the total over
// all six faces to 4*pi.
[[nodiscard]] float probeCubeTexelSolidAngle(uint32_t x, uint32_t y, uint32_t resolution);

// Where a (probe slot, face) pair lands in the capture atlas. `slot` indexes
// this frame's batch, not the probe grid: the atlas only ever holds the probes
// being captured right now.
[[nodiscard]] glm::uvec2 probeCaptureTileOrigin(uint32_t slot, uint32_t face);
[[nodiscard]] glm::uvec2 probeCaptureAtlasSize();

// The probes to capture this frame, appended to `outProbeIndices`, and the
// cursor the next frame should start from.
//
// Plain round robin. Its job is that the refresh interval of any one probe is
// bounded -- a scheme that picked probes by importance would let a probe nothing
// currently looks at go stale indefinitely, and staleness in GI reads as light
// that lags the scene rather than as a missing update.
[[nodiscard]] uint32_t probeUpdateBatch(uint32_t cursor,
                                        uint32_t probesPerFrame,
                                        std::vector<uint32_t>& outProbeIndices);

// --- Shading weights ---

// How much of a probe's irradiance survives the geometry between it and the
// shading point, from the two distance moments stored in the depth atlas.
//
// This is the whole reason the depth atlas exists. Trilinear blending alone
// happily takes light from a probe on the other side of a wall, and that failure
// looks like a room being softly lit from nowhere rather than like an addressing
// bug -- it is the single most characteristic artefact of probe GI.
//
// A Chebyshev bound on P(distance <= d) from the mean and mean square, cubed to
// sharpen the falloff, and floored so a probe never contributes exactly nothing
// (a hard zero puts a visible edge where the weight crosses it).
//
// The absolute value on the variance is load-bearing, not defensive: the moments
// are read through a bilinear filter, and interpolating between two texels can
// produce a mean square below the squared mean even though no single texel can.
// The variance then goes negative and the weight comes back negative, which
// subtracts light -- a black fringe that looks like an occlusion artefact.
[[nodiscard]] float probeChebyshevVisibility(float meanDistance,
                                             float meanSquaredDistance,
                                             float distanceToProbe);

// Smooth rejection of probes sitting behind the shading surface.
//
// A hard dot-product cutoff would drop probes discontinuously as a surface turns,
// which shows up as a hard line across otherwise smooth shading. The wrapped
// cosine falls off gradually instead, and the floor keeps a probe exactly edge-on
// contributing a little rather than nothing at all.
//
// The result is a relative weight, normalised against the other seven probes by
// the caller, so its absolute scale carries no meaning.
[[nodiscard]] float probeDirectionWeight(const glm::vec3& surfaceNormal, const glm::vec3& directionToProbe);

// World position to sample the probe grid from.
//
// Offset off the surface before looking up, or a surface samples probes whose
// stored visibility says the surface itself is in the way, and every flat plane
// self-occludes into darkness. Biased along both the normal and the view
// direction: normal alone still fails at grazing angles, where the offset barely
// clears the surface along the line the eye is actually looking down.
// Steady-state amplification of the multi-bounce feedback loop.
//
// A captured surface emits albedo * (direct + weight * previousIndirect), and
// that feeds the next capture, so the series is geometric in albedo * weight and
// settles at 1 / (1 - albedo * weight). Worth having as a function rather than a
// comment: it is the number that says whether a chosen weight is safe on a given
// scene, and the one that goes to infinity at a perfectly white surface with
// full feedback -- which is physically correct and useless as a renderer.
//
// Clamped rather than allowed to diverge, so a degenerate input produces a large
// number instead of an infinity that would reach the shading as NaN.
[[nodiscard]] float probeBounceAmplification(float albedo, float bounceWeight);

[[nodiscard]] glm::vec3 probeSamplePosition(const glm::vec3& worldPosition,
                                            const glm::vec3& surfaceNormal,
                                            const glm::vec3& viewDirection,
                                            float normalBias);

} // namespace ve::renderer
