#include "renderer/IrradianceProbes.h"

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

} // namespace ve::renderer
