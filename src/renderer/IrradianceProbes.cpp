#include "renderer/IrradianceProbes.h"

#include "renderer/PunctualShadowAtlas.h"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

namespace ve::renderer {

namespace {

[[nodiscard]] float signNotZero(float value)
{
    // Deliberately not std::copysign or glm::sign: both map 0 to 0 (or to -0),
    // and the octahedral fold needs +1 there. The shaders use `>= 0.0 ? 1 : -1`,
    // and this has to match them exactly.
    return value >= 0.0f ? 1.0f : -1.0f;
}

[[nodiscard]] uint32_t clampToGrid(int64_t value, uint32_t extent)
{
    if (value < 0) {
        return 0;
    }
    const auto maxIndex = static_cast<int64_t>(extent) - 1;
    return static_cast<uint32_t>(std::min<int64_t>(value, maxIndex));
}

} // namespace

glm::vec2 octahedralEncode(const glm::vec3& direction)
{
    const float length = std::abs(direction.x) + std::abs(direction.y) + std::abs(direction.z);
    if (length <= 0.0f) {
        // A zero direction has no octahedral point; centre is the least
        // surprising answer and keeps callers from propagating NaN.
        return glm::vec2{0.5f};
    }

    const glm::vec3 normalized = direction / length;
    glm::vec2 encoded{normalized.x, normalized.y};
    if (normalized.z < 0.0f) {
        encoded = glm::vec2{(1.0f - std::abs(normalized.y)) * signNotZero(normalized.x),
                            (1.0f - std::abs(normalized.x)) * signNotZero(normalized.y)};
    }

    return encoded * 0.5f + 0.5f;
}

glm::vec3 octahedralDecode(const glm::vec2& encoded)
{
    const glm::vec2 remapped = encoded * 2.0f - 1.0f;
    glm::vec3 direction{remapped.x, remapped.y, 1.0f - std::abs(remapped.x) - std::abs(remapped.y)};
    if (direction.z < 0.0f) {
        direction = glm::vec3{(1.0f - std::abs(direction.y)) * signNotZero(direction.x),
                              (1.0f - std::abs(direction.x)) * signNotZero(direction.y),
                              direction.z};
    }

    const float length = glm::length(direction);
    if (length <= 0.0f) {
        return glm::vec3{0.0f, 0.0f, 1.0f};
    }

    return direction / length;
}

uint32_t probeIndex(uint32_t x, uint32_t y, uint32_t z)
{
    const uint32_t clampedX = std::min(x, kProbeGridX - 1);
    const uint32_t clampedY = std::min(y, kProbeGridY - 1);
    const uint32_t clampedZ = std::min(z, kProbeGridZ - 1);
    return clampedX + clampedY * kProbeGridX + clampedZ * kProbeGridX * kProbeGridY;
}

glm::uvec3 probeCoord(uint32_t index)
{
    const uint32_t clamped = std::min(index, kProbeCount - 1);
    const uint32_t sliceSize = kProbeGridX * kProbeGridY;
    return glm::uvec3{clamped % kProbeGridX, (clamped % sliceSize) / kProbeGridX, clamped / sliceSize};
}

glm::vec3 probeWorldPosition(uint32_t index, const ProbeGridBounds& bounds)
{
    const glm::uvec3 coord = probeCoord(index);
    return bounds.origin + glm::vec3{static_cast<float>(coord.x) * bounds.spacing.x,
                                     static_cast<float>(coord.y) * bounds.spacing.y,
                                     static_cast<float>(coord.z) * bounds.spacing.z};
}

ProbeBlend probeBlendAt(const glm::vec3& worldPosition, const ProbeGridBounds& bounds)
{
    ProbeBlend blend{};

    const glm::vec3 spacing{std::max(bounds.spacing.x, 1.0e-4f),
                            std::max(bounds.spacing.y, 1.0e-4f),
                            std::max(bounds.spacing.z, 1.0e-4f)};
    const glm::vec3 gridSpace = (worldPosition - bounds.origin) / spacing;

    const glm::vec3 floored{std::floor(gridSpace.x), std::floor(gridSpace.y), std::floor(gridSpace.z)};

    blend.baseCoord = glm::uvec3{clampToGrid(static_cast<int64_t>(floored.x), kProbeGridX),
                                 clampToGrid(static_cast<int64_t>(floored.y), kProbeGridY),
                                 clampToGrid(static_cast<int64_t>(floored.z), kProbeGridZ)};

    // Outside the volume the fraction is clamped rather than extrapolated, so a
    // point beyond the grid takes the edge probes' irradiance instead of an
    // invented value that grows with distance.
    blend.fraction = glm::clamp(gridSpace - floored, glm::vec3{0.0f}, glm::vec3{1.0f});
    if (gridSpace.x < 0.0f) {
        blend.fraction.x = 0.0f;
    }
    if (gridSpace.y < 0.0f) {
        blend.fraction.y = 0.0f;
    }
    if (gridSpace.z < 0.0f) {
        blend.fraction.z = 0.0f;
    }

    return blend;
}

float probeCornerWeight(const ProbeBlend& blend, uint32_t cornerIndex)
{
    const uint32_t corner = cornerIndex & 7u;
    const float weightX = (corner & 1u) != 0u ? blend.fraction.x : 1.0f - blend.fraction.x;
    const float weightY = (corner & 2u) != 0u ? blend.fraction.y : 1.0f - blend.fraction.y;
    const float weightZ = (corner & 4u) != 0u ? blend.fraction.z : 1.0f - blend.fraction.z;
    return weightX * weightY * weightZ;
}

glm::uvec3 probeCornerCoord(const ProbeBlend& blend, uint32_t cornerIndex)
{
    const uint32_t corner = cornerIndex & 7u;
    return glm::uvec3{
        std::min(blend.baseCoord.x + ((corner & 1u) != 0u ? 1u : 0u), kProbeGridX - 1),
        std::min(blend.baseCoord.y + ((corner & 2u) != 0u ? 1u : 0u), kProbeGridY - 1),
        std::min(blend.baseCoord.z + ((corner & 4u) != 0u ? 1u : 0u), kProbeGridZ - 1)};
}

glm::uvec2 probeTileCoord(uint32_t index)
{
    const uint32_t clamped = std::min(index, kProbeCount - 1);
    return glm::uvec2{clamped % kProbeAtlasTilesX, clamped / kProbeAtlasTilesX};
}

glm::uvec2 probeTileOrigin(uint32_t index, uint32_t coreResolution)
{
    const uint32_t tileSize = coreResolution + 2 * kProbeBorderTexels;
    const glm::uvec2 tile = probeTileCoord(index);
    return tile * tileSize;
}

glm::uvec2 probeAtlasSize(uint32_t coreResolution)
{
    const uint32_t tileSize = coreResolution + 2 * kProbeBorderTexels;
    return glm::uvec2{kProbeAtlasTilesX * tileSize, kProbeAtlasTilesY * tileSize};
}

glm::vec2 probeAtlasUv(uint32_t index, const glm::vec3& direction, uint32_t coreResolution)
{
    const float core = static_cast<float>(std::max(coreResolution, 1u));
    const glm::vec2 octant = octahedralEncode(direction);
    const glm::uvec2 origin = probeTileOrigin(index, coreResolution);

    // octant spans the core square's full extent, so texel centres land at
    // (i + 0.5) / core and the two extremes sit exactly on the core's outer
    // boundary -- half a texel inside the tile, with the border texel on the
    // other side of the tap. That is the whole point: the widest bilinear
    // footprint reaches the border and stops there.
    const glm::vec2 texel = glm::vec2{origin} + glm::vec2{static_cast<float>(kProbeBorderTexels)} + octant * core;

    return texel / glm::vec2{probeAtlasSize(coreResolution)};
}

glm::vec3 probeTexelDirection(uint32_t x, uint32_t y, uint32_t coreResolution)
{
    const uint32_t core = std::max(coreResolution, 1u);
    const uint32_t coreX = std::clamp(x, kProbeBorderTexels, core);
    const uint32_t coreY = std::clamp(y, kProbeBorderTexels, core);

    const glm::vec2 octant{(static_cast<float>(coreX - kProbeBorderTexels) + 0.5f) / static_cast<float>(core),
                           (static_cast<float>(coreY - kProbeBorderTexels) + 0.5f) / static_cast<float>(core)};
    return octahedralDecode(octant);
}

glm::ivec2 probeBorderSource(int32_t x, int32_t y, uint32_t coreResolution)
{
    const auto core = static_cast<int32_t>(std::max(coreResolution, 1u));
    const auto border = static_cast<int32_t>(kProbeBorderTexels);
    const int32_t last = core + border - 1; // last core texel on either axis

    const bool leftBorder = x < border;
    const bool rightBorder = x > last;
    const bool topBorder = y < border;
    const bool bottomBorder = y > last;

    if (leftBorder || rightBorder) {
        if (topBorder || bottomBorder) {
            // A corner texel lies diagonally across the seam from the core
            // corner opposite it, not from the one it touches.
            return glm::ivec2{leftBorder ? last : border, topBorder ? last : border};
        }
        // An edge re-enters the square on its own side, mirrored about the
        // edge's midpoint, so the border column is the adjacent core column
        // read bottom-to-top.
        return glm::ivec2{leftBorder ? border : last, core + 2 * border - 1 - y};
    }

    if (topBorder || bottomBorder) {
        return glm::ivec2{core + 2 * border - 1 - x, topBorder ? border : last};
    }

    return glm::ivec2{x, y};
}

glm::mat4 probeCaptureFaceViewProjection(const glm::vec3& probePosition, uint32_t face, const glm::vec2& jitterTexels)
{
    const glm::mat4 viewProjection =
        computePointShadowFaceViewProjection(probePosition, face, kProbeMaxDistance, kProbeCaptureNearPlane);
    if (jitterTexels.x == 0.0f && jitterTexels.y == 0.0f) {
        return viewProjection;
    }

    // A clip-space translation applied *after* the view-projection, not a term
    // poked into it. Modifying column 2 of the combined matrix would scale the
    // offset by world z rather than view z, so the shift would vary across the
    // face instead of being the constant sub-texel slide this needs -- which is
    // exactly what it did until the round-trip test said otherwise.
    //
    // Left-multiplying by a translation whose x and y ride w survives the
    // perspective divide as a constant NDC offset. One texel spans 2/resolution
    // of NDC.
    const float scale = 2.0f / static_cast<float>(kProbeCaptureFaceResolution);
    glm::mat4 clipTranslation{1.0f};
    clipTranslation[3][0] = jitterTexels.x * scale;
    clipTranslation[3][1] = jitterTexels.y * scale;
    return clipTranslation * viewProjection;
}

glm::vec3 probeCubeTexelDirection(uint32_t face,
                                  uint32_t x,
                                  uint32_t y,
                                  uint32_t resolution,
                                  const glm::vec2& jitterTexels)
{
    const float size = static_cast<float>(std::max(resolution, 1u));
    // Texel centre in the face's [-1, 1] plane. The 90-degree projection puts
    // the face plane exactly one unit from the probe, so this doubles as the
    // unnormalised direction's tangential components.
    //
    // The jitter is subtracted, not added: the projection shifts the image, so
    // the direction landing under texel x is the one that was at x minus the
    // shift. Adding it would double the offset instead of cancelling it, which
    // reads as a smeared probe rather than as anything obviously wrong.
    const float u = (static_cast<float>(x) + 0.5f - jitterTexels.x) / size * 2.0f - 1.0f;
    const float v = (static_cast<float>(y) + 0.5f - jitterTexels.y) / size * 2.0f - 1.0f;

    // The basis has to be the one glm::lookAt actually builds inside
    // computePointShadowFaceViewProjection, or the convolution reads every texel
    // as looking somewhere it does not -- and that fails smoothly, as lighting
    // arriving from the wrong direction, rather than as anything obviously
    // broken. Derived here to keep the shader mirror simple, and pinned to the
    // matrix by a round-trip test rather than by this comment.
    //
    // The reference axis switch on near-vertical faces is shadowUpVector's, not
    // a reinvention: lookAt degenerates when the view direction is parallel to
    // the up vector, and both +Y and -Y fall back to the same +Z.
    const glm::vec3 forward = pointShadowFaceDirection(face);
    const glm::vec3 worldUp =
        std::abs(forward.y) > 0.999f ? glm::vec3{0.0f, 0.0f, 1.0f} : glm::vec3{0.0f, 1.0f, 0.0f};
    const glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    const glm::vec3 up = glm::cross(right, forward);

    return glm::normalize(forward + right * u + up * v);
}

float probeCubeTexelSolidAngle(uint32_t x, uint32_t y, uint32_t resolution)
{
    const float size = static_cast<float>(std::max(resolution, 1u));
    const float u = (static_cast<float>(x) + 0.5f) / size * 2.0f - 1.0f;
    const float v = (static_cast<float>(y) + 0.5f) / size * 2.0f - 1.0f;

    // Jacobian of the cube-face parameterisation: a texel of side 2/resolution
    // in the face plane projects onto (1 + u^2 + v^2)^(3/2) less sphere the
    // further it sits from the face centre.
    const float denominator = 1.0f + u * u + v * v;
    return (4.0f / (size * size)) / (denominator * std::sqrt(denominator));
}

glm::uvec2 probeCaptureTileOrigin(uint32_t slot, uint32_t face)
{
    // One row per probe slot, one column per face. Keeping a probe's six faces
    // on one row is what lets the convolution walk them with a single row
    // offset, and it makes the debug preview readable as one probe per line.
    const uint32_t clampedSlot = std::min(slot, kMaxProbesPerFrame - 1);
    const uint32_t clampedFace = std::min(face, kProbeCaptureFaceCount - 1);
    return glm::uvec2{clampedFace * kProbeCaptureFaceResolution, clampedSlot * kProbeCaptureFaceResolution};
}

glm::uvec2 probeCaptureAtlasSize()
{
    return glm::uvec2{kProbeCaptureFaceCount * kProbeCaptureFaceResolution,
                      kMaxProbesPerFrame * kProbeCaptureFaceResolution};
}

float probeChebyshevVisibility(float meanDistance, float meanSquaredDistance, float distanceToProbe)
{
    // Nothing between here and the probe: the stored mean already reaches at
    // least this far.
    if (!(distanceToProbe > meanDistance)) {
        // Written as a negated compare so a NaN distance decodes to fully
        // visible rather than falling through into the divide below.
        return 1.0f;
    }

    const float variance = std::abs(meanDistance * meanDistance - meanSquaredDistance);
    const float excess = distanceToProbe - meanDistance;
    const float bound = variance / (variance + excess * excess);

    // Cubed: the raw Chebyshev bound falls off far too gently to read as a wall,
    // leaving light bleeding a long way past the occluder.
    const float sharpened = bound * bound * bound;
    return std::clamp(sharpened, kProbeMinVisibility, 1.0f);
}

float probeDirectionWeight(const glm::vec3& surfaceNormal, const glm::vec3& directionToProbe)
{
    const float wrapped = (glm::dot(surfaceNormal, directionToProbe) + 1.0f) * 0.5f;
    return wrapped * wrapped + kProbeBackfaceFloor;
}

glm::vec3 probeSamplePosition(const glm::vec3& worldPosition,
                              const glm::vec3& surfaceNormal,
                              const glm::vec3& viewDirection,
                              float normalBias)
{
    // Weighted toward the view direction rather than split evenly: at grazing
    // angles the normal offset barely moves the sample along the line the eye
    // looks down, which is exactly where self-occlusion shows.
    const glm::vec3 offset = surfaceNormal * 0.2f + viewDirection * 0.8f;
    return worldPosition + offset * std::max(normalBias, 0.0f);
}

glm::vec2 probeCaptureJitter(uint32_t updateIndex)
{
    // Halton (2, 3), the sequence the TAA jitter already uses here. Successive
    // captures land spread across the texel rather than clustering, which random
    // offsets would do -- and clustering is what would leave the accumulated
    // result still aliased after many updates.
    const auto halton = [](uint32_t index, uint32_t base) {
        float result = 0.0f;
        float fraction = 1.0f;
        uint32_t current = index;
        while (current > 0) {
            fraction /= static_cast<float>(base);
            result += fraction * static_cast<float>(current % base);
            current /= base;
        }
        return result;
    };

    // Index 0 of a Halton sequence is 0, which would make the first update
    // unjittered and every cycle-length-th one identical to it; offsetting by
    // one keeps the sequence moving from the very first capture.
    const uint32_t index = updateIndex + 1;
    return glm::vec2{halton(index, 2) - 0.5f, halton(index, 3) - 0.5f};
}

uint32_t probeUpdateBatch(uint32_t cursor, uint32_t probesPerFrame, std::vector<uint32_t>& outProbeIndices)
{
    outProbeIndices.clear();

    // Clamped rather than asserted: the count is a live setting, and a frame
    // that asks for more probes than exist should capture each one once rather
    // than capture some of them twice into the same tile.
    const uint32_t count = std::min({probesPerFrame, kMaxProbesPerFrame, kProbeCount});
    if (count == 0) {
        return cursor % kProbeCount;
    }

    uint32_t next = cursor % kProbeCount;
    outProbeIndices.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        outProbeIndices.push_back(next);
        next = (next + 1) % kProbeCount;
    }

    return next;
}

} // namespace ve::renderer
