#include "renderer/IrradianceProbes.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

using Catch::Approx;
using ve::renderer::kMaxProbesPerFrame;
using ve::renderer::kProbeAtlasTilesX;
using ve::renderer::kProbeAtlasTilesY;
using ve::renderer::kProbeBorderTexels;
using ve::renderer::kProbeCaptureFaceCount;
using ve::renderer::kProbeCaptureFaceResolution;
using ve::renderer::kProbeCount;
using ve::renderer::kProbeDepthResolution;
using ve::renderer::kProbeGridX;
using ve::renderer::kProbeGridY;
using ve::renderer::kProbeGridZ;
using ve::renderer::kProbeIrradianceResolution;
using ve::renderer::octahedralDecode;
using ve::renderer::octahedralEncode;
using ve::renderer::ProbeBlend;
using ve::renderer::probeAtlasSize;
using ve::renderer::probeAtlasUv;
using ve::renderer::probeBlendAt;
using ve::renderer::probeBorderSource;
using ve::renderer::probeChebyshevVisibility;
using ve::renderer::probeDirectionWeight;
using ve::renderer::probeSamplePosition;
using ve::renderer::probeCaptureJitter;
using ve::renderer::probeBounceAmplification;
using ve::renderer::probeCaptureAtlasSize;
using ve::renderer::probeCaptureFaceViewProjection;
using ve::renderer::probeCaptureTileOrigin;
using ve::renderer::probeCubeTexelDirection;
using ve::renderer::probeCubeTexelSolidAngle;
using ve::renderer::probeUpdateBatch;
using ve::renderer::probeCoord;
using ve::renderer::probeCornerCoord;
using ve::renderer::probeCornerWeight;
using ve::renderer::ProbeGridBounds;
using ve::renderer::probeIndex;
using ve::renderer::probeTexelDirection;
using ve::renderer::probeTileCoord;
using ve::renderer::probeTileOrigin;
using ve::renderer::probeWorldPosition;

namespace {

// A spread of directions over the whole sphere, including the axis and fold
// boundaries where an octahedral mapping is most likely to be wrong.
std::vector<glm::vec3> sphereDirections()
{
    std::vector<glm::vec3> directions;
    for (int theta = 0; theta <= 36; ++theta) {
        for (int phi = 0; phi <= 18; ++phi) {
            const float a = static_cast<float>(theta) * 10.0f * 3.14159265f / 180.0f;
            const float b = static_cast<float>(phi) * 10.0f * 3.14159265f / 180.0f;
            directions.push_back(glm::normalize(
                glm::vec3{std::sin(b) * std::cos(a), std::cos(b), std::sin(b) * std::sin(a)}));
        }
    }
    // The six axes and the eight octant diagonals: the fold seams.
    for (const glm::vec3& axis : {glm::vec3{1, 0, 0},
                                  glm::vec3{-1, 0, 0},
                                  glm::vec3{0, 1, 0},
                                  glm::vec3{0, -1, 0},
                                  glm::vec3{0, 0, 1},
                                  glm::vec3{0, 0, -1}}) {
        directions.push_back(axis);
    }
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sy = -1; sy <= 1; sy += 2) {
            for (int sz = -1; sz <= 1; sz += 2) {
                directions.push_back(glm::normalize(glm::vec3{static_cast<float>(sx),
                                                              static_cast<float>(sy),
                                                              static_cast<float>(sz)}));
            }
        }
    }
    return directions;
}

// The exact mapping the shaders use (simple.frag octEncode / ssr_trace.frag
// octDecode), transcribed. The CPU side must agree with this or probe lighting
// comes out rotated or folded -- which reads as a shading bug, not a mapping
// one, so it is worth pinning rather than trusting.
glm::vec2 shaderOctEncode(glm::vec3 n)
{
    n /= (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    glm::vec2 e{n.x, n.y};
    if (n.z < 0.0f) {
        e = glm::vec2{(1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f),
                      (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f)};
    }
    return e * 0.5f + 0.5f;
}

glm::vec3 shaderOctDecode(glm::vec2 e)
{
    e = e * 2.0f - 1.0f;
    glm::vec3 n{e.x, e.y, 1.0f - std::abs(e.x) - std::abs(e.y)};
    if (n.z < 0.0f) {
        n = glm::vec3{(1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f),
                      (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f),
                      n.z};
    }
    return glm::normalize(n);
}

} // namespace

TEST_CASE("Octahedral mapping round-trips over the whole sphere", "[probes]")
{
    for (const glm::vec3& direction : sphereDirections()) {
        const glm::vec2 encoded = octahedralEncode(direction);

        // Stays inside the unit square, which is what makes it addressable as a
        // texture tile at all.
        CHECK(encoded.x >= -1.0e-5f);
        CHECK(encoded.x <= 1.0f + 1.0e-5f);
        CHECK(encoded.y >= -1.0e-5f);
        CHECK(encoded.y <= 1.0f + 1.0e-5f);

        const glm::vec3 decoded = octahedralDecode(encoded);
        CHECK(decoded.x == Approx(direction.x).margin(1.0e-4f));
        CHECK(decoded.y == Approx(direction.y).margin(1.0e-4f));
        CHECK(decoded.z == Approx(direction.z).margin(1.0e-4f));
        // Decoding always yields a unit vector, so callers can dot with it
        // directly.
        CHECK(glm::length(decoded) == Approx(1.0f).margin(1.0e-5f));
    }
}

TEST_CASE("The CPU octahedral mapping matches the shaders' convention", "[probes]")
{
    // The engine already had this mapping in three shaders before probes
    // existed. A second, subtly different one would light probes through a
    // rotated or folded basis -- wrong in a way that looks like a shading bug
    // rather than a mapping bug, so it is pinned rather than assumed.
    for (const glm::vec3& direction : sphereDirections()) {
        const glm::vec2 mine = octahedralEncode(direction);
        const glm::vec2 theirs = shaderOctEncode(direction);
        CHECK(mine.x == Approx(theirs.x).margin(1.0e-5f));
        CHECK(mine.y == Approx(theirs.y).margin(1.0e-5f));

        const glm::vec3 mineDecoded = octahedralDecode(mine);
        const glm::vec3 theirsDecoded = shaderOctDecode(theirs);
        CHECK(mineDecoded.x == Approx(theirsDecoded.x).margin(1.0e-4f));
        CHECK(mineDecoded.y == Approx(theirsDecoded.y).margin(1.0e-4f));
        CHECK(mineDecoded.z == Approx(theirsDecoded.z).margin(1.0e-4f));
    }
}

TEST_CASE("Octahedral mapping survives degenerate directions", "[probes]")
{
    // A zero direction has no octahedral point. It must not produce NaN, which
    // would propagate into probe irradiance and never wash out.
    const glm::vec2 zero = octahedralEncode(glm::vec3{0.0f});
    CHECK(std::isfinite(zero.x));
    CHECK(std::isfinite(zero.y));

    const glm::vec3 decodedCentre = octahedralDecode(glm::vec2{0.5f});
    CHECK(std::isfinite(decodedCentre.x));
    CHECK(glm::length(decodedCentre) == Approx(1.0f).margin(1.0e-5f));

    // Corners of the square are the -Z pole; all four must decode to a unit
    // vector rather than a zero-length one.
    for (const glm::vec2& corner :
         {glm::vec2{0, 0}, glm::vec2{1, 0}, glm::vec2{0, 1}, glm::vec2{1, 1}}) {
        const glm::vec3 decoded = octahedralDecode(corner);
        CHECK(std::isfinite(decoded.x));
        CHECK(glm::length(decoded) == Approx(1.0f).margin(1.0e-4f));
    }

    // Unnormalised input encodes the same as its normalised form: the mapping
    // is direction-only, so callers need not normalise first.
    const glm::vec3 direction{3.0f, -4.0f, 12.0f};
    const glm::vec2 raw = octahedralEncode(direction);
    const glm::vec2 unit = octahedralEncode(glm::normalize(direction));
    CHECK(raw.x == Approx(unit.x).margin(1.0e-5f));
    CHECK(raw.y == Approx(unit.y).margin(1.0e-5f));
}

TEST_CASE("Probe index and coordinate round-trip", "[probes]")
{
    std::set<uint32_t> seen;
    for (uint32_t z = 0; z < kProbeGridZ; ++z) {
        for (uint32_t y = 0; y < kProbeGridY; ++y) {
            for (uint32_t x = 0; x < kProbeGridX; ++x) {
                const uint32_t index = probeIndex(x, y, z);
                REQUIRE(index < kProbeCount);
                // Every coordinate maps to a distinct slot, so no two probes
                // share storage.
                CHECK(seen.insert(index).second);

                const glm::uvec3 coord = probeCoord(index);
                CHECK(coord.x == x);
                CHECK(coord.y == y);
                CHECK(coord.z == z);
            }
        }
    }
    CHECK(seen.size() == kProbeCount);

    // Out-of-range inputs clamp rather than aliasing onto a valid probe from a
    // wrapped index.
    CHECK(probeIndex(kProbeGridX + 5, 0, 0) == probeIndex(kProbeGridX - 1, 0, 0));
    CHECK(probeCoord(kProbeCount + 100).x < kProbeGridX);
}

TEST_CASE("Probe positions follow the grid bounds", "[probes]")
{
    ProbeGridBounds bounds{};
    bounds.origin = glm::vec3{-10.0f, 0.0f, -10.0f};
    bounds.spacing = glm::vec3{2.5f, 1.5f, 2.5f};

    CHECK(probeWorldPosition(probeIndex(0, 0, 0), bounds) == bounds.origin);

    const glm::vec3 stepX = probeWorldPosition(probeIndex(1, 0, 0), bounds) -
                            probeWorldPosition(probeIndex(0, 0, 0), bounds);
    CHECK(stepX.x == Approx(bounds.spacing.x));
    CHECK(stepX.y == Approx(0.0f));
    CHECK(stepX.z == Approx(0.0f));

    const glm::vec3 stepZ = probeWorldPosition(probeIndex(0, 0, 1), bounds) -
                            probeWorldPosition(probeIndex(0, 0, 0), bounds);
    CHECK(stepZ.z == Approx(bounds.spacing.z));
    CHECK(stepZ.x == Approx(0.0f));

    // A probe position feeds straight back into the blend as that probe's own
    // cell corner, with no fractional offset.
    const uint32_t sampleIndex = probeIndex(3, 2, 5);
    const ProbeBlend blend = probeBlendAt(probeWorldPosition(sampleIndex, bounds), bounds);
    CHECK(blend.baseCoord.x == 3u);
    CHECK(blend.baseCoord.y == 2u);
    CHECK(blend.baseCoord.z == 5u);
    CHECK(blend.fraction.x == Approx(0.0f).margin(1.0e-4f));
}

TEST_CASE("Trilinear probe weights partition unity", "[probes]")
{
    ProbeGridBounds bounds{};
    bounds.origin = glm::vec3{-8.0f, -1.0f, -8.0f};
    bounds.spacing = glm::vec3{2.0f};

    // The weights over the eight surrounding probes must sum to exactly one at
    // every point, or the GI term brightens and darkens with position -- a
    // gradient artefact that is easy to mistake for the lighting itself.
    for (float x = -12.0f; x <= 12.0f; x += 0.7f) {
        for (float y = -3.0f; y <= 8.0f; y += 0.6f) {
            for (float z = -12.0f; z <= 12.0f; z += 0.9f) {
                const ProbeBlend blend = probeBlendAt(glm::vec3{x, y, z}, bounds);

                float total = 0.0f;
                for (uint32_t corner = 0; corner < 8; ++corner) {
                    const float weight = probeCornerWeight(blend, corner);
                    CHECK(weight >= 0.0f);
                    CHECK(weight <= 1.0f);
                    total += weight;

                    const glm::uvec3 coord = probeCornerCoord(blend, corner);
                    CHECK(coord.x < kProbeGridX);
                    CHECK(coord.y < kProbeGridY);
                    CHECK(coord.z < kProbeGridZ);
                }
                CHECK(total == Approx(1.0f).margin(1.0e-5f));
            }
        }
    }
}

TEST_CASE("Points outside the grid clamp instead of extrapolating", "[probes]")
{
    ProbeGridBounds bounds{};
    bounds.origin = glm::vec3{0.0f};
    bounds.spacing = glm::vec3{1.0f};

    // Far outside in every direction: the blend must resolve to real probes with
    // no fractional overshoot, so a distant point takes the edge probes' value
    // rather than an extrapolated one that grows without bound.
    for (const glm::vec3& outside : {glm::vec3{-1000.0f, -1000.0f, -1000.0f},
                                     glm::vec3{1000.0f, 1000.0f, 1000.0f},
                                     glm::vec3{-50.0f, 2.0f, 500.0f}}) {
        const ProbeBlend blend = probeBlendAt(outside, bounds);
        CHECK(blend.baseCoord.x < kProbeGridX);
        CHECK(blend.baseCoord.y < kProbeGridY);
        CHECK(blend.baseCoord.z < kProbeGridZ);

        float total = 0.0f;
        for (uint32_t corner = 0; corner < 8; ++corner) {
            CHECK(blend.fraction.x >= 0.0f);
            CHECK(blend.fraction.x <= 1.0f);
            total += probeCornerWeight(blend, corner);
        }
        CHECK(total == Approx(1.0f).margin(1.0e-5f));
    }

    // Degenerate spacing must not divide by zero.
    ProbeGridBounds degenerate{};
    degenerate.spacing = glm::vec3{0.0f};
    const ProbeBlend blend = probeBlendAt(glm::vec3{1.0f, 2.0f, 3.0f}, degenerate);
    CHECK(std::isfinite(blend.fraction.x));
    CHECK(blend.baseCoord.x < kProbeGridX);
}

namespace {

// One probe's tile, holding a direction per texel. Directions are the sharpest
// test signal available: they are what the octahedral square actually
// parameterises, so a border texel copied from the wrong place points somewhere
// visibly else rather than merely being slightly off.
struct ProbeTile {
    uint32_t core = 0;
    uint32_t size = 0;
    std::vector<glm::vec3> texels;

    [[nodiscard]] glm::vec3 at(int32_t x, int32_t y) const
    {
        const int32_t clampedX = std::clamp(x, 0, static_cast<int32_t>(size) - 1);
        const int32_t clampedY = std::clamp(y, 0, static_cast<int32_t>(size) - 1);
        return texels[static_cast<size_t>(clampedY) * size + static_cast<size_t>(clampedX)];
    }
};

// The rule under test, plus the two ways of getting it wrong that a reader would
// actually reach for.
enum class BorderRule {
    // What probeBorderSource implements.
    Octahedral,
    // A border treated as padding. Costs half a texel of bias at every edge --
    // small, but it is the difference between filtering and not filtering there.
    ClampToEdge,
    // The seam wrapped the way an equirectangular map wraps: off the left edge,
    // on at the right. Plausible, and wrong by most of a hemisphere, because the
    // octahedral square folds back onto its *own* edge rather than the opposite
    // one.
    TorusWrap
};

ProbeTile makeProbeTile(uint32_t core, BorderRule rule)
{
    ProbeTile tile{};
    tile.core = core;
    tile.size = core + 2 * kProbeBorderTexels;
    tile.texels.resize(static_cast<size_t>(tile.size) * tile.size);

    const auto border = static_cast<int32_t>(kProbeBorderTexels);
    const auto last = static_cast<int32_t>(core + kProbeBorderTexels - 1);
    const auto size = static_cast<int32_t>(tile.size);

    for (int32_t y = 0; y < size; ++y) {
        for (int32_t x = 0; x < size; ++x) {
            glm::ivec2 source{x, y};
            switch (rule) {
            case BorderRule::Octahedral:
                source = probeBorderSource(x, y, core);
                break;
            case BorderRule::ClampToEdge:
                source = glm::ivec2{std::clamp(x, border, last), std::clamp(y, border, last)};
                break;
            case BorderRule::TorusWrap:
                source = glm::ivec2{x < border ? last : (x > last ? border : x),
                                    y < border ? last : (y > last ? border : y)};
                break;
            }
            tile.texels[static_cast<size_t>(y) * tile.size + static_cast<size_t>(x)] =
                probeTexelDirection(static_cast<uint32_t>(source.x), static_cast<uint32_t>(source.y), core);
        }
    }
    return tile;
}

// The filtering the GPU would do: texel centres at (i + 0.5), so a tap at pixel
// coordinate p blends the texels straddling p - 0.5.
glm::vec3 sampleBilinear(const ProbeTile& tile, const glm::vec2& pixel)
{
    const glm::vec2 shifted = pixel - 0.5f;
    const float baseX = std::floor(shifted.x);
    const float baseY = std::floor(shifted.y);
    const float fractionX = shifted.x - baseX;
    const float fractionY = shifted.y - baseY;

    const auto x0 = static_cast<int32_t>(baseX);
    const auto y0 = static_cast<int32_t>(baseY);

    const glm::vec3 top = glm::mix(tile.at(x0, y0), tile.at(x0 + 1, y0), fractionX);
    const glm::vec3 bottom = glm::mix(tile.at(x0, y0 + 1), tile.at(x0 + 1, y0 + 1), fractionX);
    return glm::mix(top, bottom, fractionY);
}

// Angle between the filtered direction and the one it should reconstruct.
float reconstructionError(const ProbeTile& tile, const glm::vec3& direction)
{
    // Probe 0's tile starts at the atlas origin, so the atlas UV scales straight
    // back into tile-local pixels.
    const glm::vec2 uv = probeAtlasUv(0, direction, tile.core);
    const glm::vec2 pixel = uv * glm::vec2{probeAtlasSize(tile.core)};

    const glm::vec3 filtered = sampleBilinear(tile, pixel);
    if (glm::length(filtered) <= 0.0f) {
        return 3.14159265f;
    }
    return std::acos(std::clamp(glm::dot(glm::normalize(filtered), direction), -1.0f, 1.0f));
}

// A direction is "on the seam" when its tap straddles the square's boundary --
// exactly the taps that reach a border texel.
bool onSeam(const glm::vec3& direction, uint32_t core)
{
    const glm::vec2 octant = octahedralEncode(direction) * static_cast<float>(core);
    return octant.x < 0.5f || octant.x > static_cast<float>(core) - 0.5f || octant.y < 0.5f ||
           octant.y > static_cast<float>(core) - 0.5f;
}

} // namespace

TEST_CASE("Probe tiles pack the atlas without overlapping", "[probes]")
{
    for (const uint32_t core : {kProbeIrradianceResolution, kProbeDepthResolution}) {
        const uint32_t tileSize = core + 2 * kProbeBorderTexels;
        const glm::uvec2 atlas = probeAtlasSize(core);
        CHECK(atlas.x == kProbeAtlasTilesX * tileSize);
        CHECK(atlas.y == kProbeAtlasTilesY * tileSize);

        std::set<uint64_t> occupiedTiles;
        for (uint32_t index = 0; index < kProbeCount; ++index) {
            const glm::uvec2 tile = probeTileCoord(index);
            CHECK(tile.x < kProbeAtlasTilesX);
            CHECK(tile.y < kProbeAtlasTilesY);
            // Two probes sharing a tile would have one silently overwrite the
            // other's irradiance every update.
            CHECK(occupiedTiles.insert(static_cast<uint64_t>(tile.y) * kProbeAtlasTilesX + tile.x).second);

            const glm::uvec2 origin = probeTileOrigin(index, core);
            CHECK(origin.x == tile.x * tileSize);
            CHECK(origin.y == tile.y * tileSize);
            CHECK(origin.x + tileSize <= atlas.x);
            CHECK(origin.y + tileSize <= atlas.y);
        }
        CHECK(occupiedTiles.size() == kProbeCount);

        // Out-of-range indices clamp into the atlas rather than addressing past
        // its end.
        const glm::uvec2 clamped = probeTileOrigin(kProbeCount + 17, core);
        CHECK(clamped.x + tileSize <= atlas.x);
        CHECK(clamped.y + tileSize <= atlas.y);
    }
}

TEST_CASE("Probe atlas UVs stay inside the probe's own tile", "[probes]")
{
    // A tap that leaves the tile reads another probe entirely, which shows up as
    // one probe's lighting bleeding into its neighbour -- a gradient artefact
    // that looks like the GI itself rather than an addressing bug.
    for (const uint32_t core : {kProbeIrradianceResolution, kProbeDepthResolution}) {
        const auto tileSize = static_cast<float>(core + 2 * kProbeBorderTexels);
        const glm::vec2 atlas{probeAtlasSize(core)};

        for (const uint32_t index : {0u, 1u, kProbeAtlasTilesX - 1u, kProbeCount / 2u, kProbeCount - 1u}) {
            const glm::vec2 origin{probeTileOrigin(index, core)};

            for (const glm::vec3& direction : sphereDirections()) {
                const glm::vec2 pixel = probeAtlasUv(index, direction, core) * atlas - origin;

                // Inside the core square...
                CHECK(pixel.x >= static_cast<float>(kProbeBorderTexels) - 1.0e-4f);
                CHECK(pixel.y >= static_cast<float>(kProbeBorderTexels) - 1.0e-4f);
                CHECK(pixel.x <= tileSize - static_cast<float>(kProbeBorderTexels) + 1.0e-4f);
                CHECK(pixel.y <= tileSize - static_cast<float>(kProbeBorderTexels) + 1.0e-4f);

                // ...and far enough in that the bilinear footprint, which spans
                // half a texel either side, still cannot escape the tile.
                CHECK(pixel.x - 0.5f >= -1.0e-4f);
                CHECK(pixel.y - 0.5f >= -1.0e-4f);
                CHECK(pixel.x + 0.5f <= tileSize + 1.0e-4f);
                CHECK(pixel.y + 0.5f <= tileSize + 1.0e-4f);
            }
        }
    }
}

TEST_CASE("Texel directions round-trip through the atlas UV", "[probes]")
{
    // probeTexelDirection is what the update pass writes with and probeAtlasUv is
    // what the lookup reads with. If the two disagree by so much as half a texel
    // the whole atlas is shifted, so they are pinned against each other rather
    // than each being checked alone.
    const uint32_t core = kProbeIrradianceResolution;
    const glm::vec2 atlas{probeAtlasSize(core)};

    for (const uint32_t index : {0u, 5u, kProbeCount - 1u}) {
        const glm::vec2 origin{probeTileOrigin(index, core)};
        for (uint32_t y = kProbeBorderTexels; y <= core; ++y) {
            for (uint32_t x = kProbeBorderTexels; x <= core; ++x) {
                const glm::vec3 direction = probeTexelDirection(x, y, core);
                const glm::vec2 pixel = probeAtlasUv(index, direction, core) * atlas - origin;
                CHECK(pixel.x == Approx(static_cast<float>(x) + 0.5f).margin(1.0e-3f));
                CHECK(pixel.y == Approx(static_cast<float>(y) + 0.5f).margin(1.0e-3f));
            }
        }
    }
}

TEST_CASE("The octahedral border mirrors the seam", "[probes]")
{
    for (const uint32_t core : {kProbeIrradianceResolution, kProbeDepthResolution}) {
        const auto last = static_cast<int32_t>(core + kProbeBorderTexels - 1);
        const auto border = static_cast<int32_t>(kProbeBorderTexels);
        const auto size = static_cast<int32_t>(core + 2 * kProbeBorderTexels);

        // Core texels are their own source, so the border pass can run over a
        // whole tile without the caller classifying texels first.
        for (int32_t y = border; y <= last; ++y) {
            for (int32_t x = border; x <= last; ++x) {
                const glm::ivec2 source = probeBorderSource(x, y, core);
                CHECK(source.x == x);
                CHECK(source.y == y);
            }
        }

        // Corners take the diagonally opposite core corner. Taking the one they
        // touch instead is the intuitive mistake and is wrong: a corner texel is
        // across two seams, not one.
        CHECK(probeBorderSource(0, 0, core) == glm::ivec2{last, last});
        CHECK(probeBorderSource(size - 1, 0, core) == glm::ivec2{border, last});
        CHECK(probeBorderSource(0, size - 1, core) == glm::ivec2{last, border});
        CHECK(probeBorderSource(size - 1, size - 1, core) == glm::ivec2{border, border});

        // Edges take the adjacent core row/column, reversed.
        for (int32_t i = border; i <= last; ++i) {
            const int32_t mirrored = size - 1 - i;
            CHECK(probeBorderSource(0, i, core) == glm::ivec2{border, mirrored});
            CHECK(probeBorderSource(size - 1, i, core) == glm::ivec2{last, mirrored});
            CHECK(probeBorderSource(i, 0, core) == glm::ivec2{mirrored, border});
            CHECK(probeBorderSource(i, size - 1, core) == glm::ivec2{mirrored, last});
        }

        // Every border texel resolves to a real core texel; one landing back in
        // the border would copy an uninitialised value.
        for (int32_t y = 0; y < size; ++y) {
            for (int32_t x = 0; x < size; ++x) {
                const glm::ivec2 source = probeBorderSource(x, y, core);
                CHECK(source.x >= border);
                CHECK(source.x <= last);
                CHECK(source.y >= border);
                CHECK(source.y <= last);
            }
        }
    }
}

TEST_CASE("The border keeps bilinear filtering continuous across the seam", "[probes]")
{
    // This is the invariant the border exists for, and the structural test above
    // cannot reach it: a mirrored-the-wrong-way rule still passes every "lands in
    // the core" check while filtering against a direction from the far side of
    // the sphere.
    //
    // So: fill a tile with the direction each texel stands for, filter it the way
    // the GPU would, and measure the angle between the reconstructed direction
    // and the real one. Then do the same with each wrong border for comparison,
    // because an error bound with nothing to compare against says only that the
    // number is small, not that the rule earned it.
    for (const uint32_t core : {kProbeIrradianceResolution, kProbeDepthResolution}) {
        const ProbeTile octahedralTile = makeProbeTile(core, BorderRule::Octahedral);
        const ProbeTile clampTile = makeProbeTile(core, BorderRule::ClampToEdge);
        const ProbeTile torusTile = makeProbeTile(core, BorderRule::TorusWrap);

        float interiorMax = 0.0f;
        float seamMax = 0.0f;
        float clampSeamMax = 0.0f;
        float torusSeamMax = 0.0f;

        for (const glm::vec3& direction : sphereDirections()) {
            const float error = reconstructionError(octahedralTile, direction);
            if (onSeam(direction, core)) {
                seamMax = std::max(seamMax, error);
                clampSeamMax = std::max(clampSeamMax, reconstructionError(clampTile, direction));
                torusSeamMax = std::max(torusSeamMax, reconstructionError(torusTile, direction));
            } else {
                interiorMax = std::max(interiorMax, error);
            }
        }

        INFO("core " << core << " interior " << interiorMax << " seam " << seamMax << " clamped " << clampSeamMax
                     << " torus " << torusSeamMax);

        // Filtering at the seam is no worse than filtering anywhere else -- the
        // tile behaves like an ordinary texture. Measured, the two maxima agree
        // to five significant figures, so the margin here is slack, not fit.
        CHECK(seamMax <= interiorMax * 1.1f);

        // Padding instead of wrapping loses the filtering at every edge and
        // biases the reconstruction by roughly half a texel. Modest -- which is
        // exactly why it survives casual inspection.
        CHECK(clampSeamMax > seamMax * 1.3f);

        // Wrapping to the opposite edge instead of folding back onto the same one
        // is off by a large angle. This is the failure the rule is really
        // guarding against, and the one that would read as noise in the GI rather
        // than as an addressing bug.
        CHECK(torusSeamMax > 0.5f);
    }
}

TEST_CASE("Capture texel directions agree with the projection that rasterises them", "[probes]")
{
    // The one convention in the capture path that cannot be checked by looking
    // at it. The convolution reconstructs each capture texel's world direction
    // from its position in the atlas; the rasteriser puts geometry there using
    // the face view-projection. If the two disagree the probe gathers real
    // radiance and files it under the wrong direction -- lighting that arrives
    // from the wrong side, which reads as a lighting bug, not a mapping one.
    //
    // So rather than trusting the basis derivation, project each texel's
    // direction back through the actual matrix and check it lands on that
    // texel's own pixel.
    const glm::vec3 probePosition{3.0f, 2.0f, -5.0f};
    const auto resolution = static_cast<float>(kProbeCaptureFaceResolution);

    for (uint32_t face = 0; face < kProbeCaptureFaceCount; ++face) {
        const glm::mat4 viewProjection = probeCaptureFaceViewProjection(probePosition, face);

        for (uint32_t y = 0; y < kProbeCaptureFaceResolution; ++y) {
            for (uint32_t x = 0; x < kProbeCaptureFaceResolution; ++x) {
                const glm::vec3 direction = probeCubeTexelDirection(face, x, y, kProbeCaptureFaceResolution);
                CHECK(glm::length(direction) == Approx(1.0f).margin(1.0e-5f));

                // A point along that direction, comfortably inside the frustum.
                const glm::vec4 clip = viewProjection * glm::vec4{probePosition + direction * 10.0f, 1.0f};
                REQUIRE(clip.w > 0.0f);
                const glm::vec2 ndc{clip.x / clip.w, clip.y / clip.w};

                // Framebuffer pixel, with y running down the image the way the
                // viewport is set up (positive height, no flip).
                const glm::vec2 pixel{(ndc.x * 0.5f + 0.5f) * resolution, (ndc.y * 0.5f + 0.5f) * resolution};
                CHECK(pixel.x == Approx(static_cast<float>(x) + 0.5f).margin(1.0e-3f));
                CHECK(pixel.y == Approx(static_cast<float>(y) + 0.5f).margin(1.0e-3f));
            }
        }
    }
}

TEST_CASE("Capture texel solid angles sum to the whole sphere", "[probes]")
{
    // Cube texels are not equal in solid angle: a face's corners subtend up to
    // three times less than its centre. Integrating radiance without that weight
    // over-counts every face's corners, and the result is a smooth bias -- the
    // GI comes out "a bit wrong" everywhere rather than visibly broken
    // somewhere, which is the hardest kind of error to notice.
    //
    // The six faces tile the sphere exactly, so the weights have exactly one
    // correct total: 4*pi.
    //
    // The sum is a midpoint rule over a smooth integrand, so it approaches that
    // rather than hitting it. Checking *convergence* is the sharper test anyway:
    // a plausible-but-wrong Jacobian can land within a few percent at one
    // resolution, but only the right one keeps closing on 4*pi as the grid
    // refines.
    constexpr double kSphere = 4.0 * 3.14159265358979;
    double previousError = 1.0e9;

    for (const uint32_t resolution : {4u, 8u, kProbeCaptureFaceResolution, 32u, 64u}) {
        double total = 0.0;
        float smallest = 1.0e9f;
        float largest = 0.0f;
        for (uint32_t y = 0; y < resolution; ++y) {
            for (uint32_t x = 0; x < resolution; ++x) {
                const float solidAngle = probeCubeTexelSolidAngle(x, y, resolution);
                CHECK(solidAngle > 0.0f);
                smallest = std::min(smallest, solidAngle);
                largest = std::max(largest, solidAngle);
                total += static_cast<double>(solidAngle);
            }
        }
        total *= static_cast<double>(kProbeCaptureFaceCount);

        const double error = std::abs(total - kSphere) / kSphere;
        INFO("resolution " << resolution << " total " << total << " relative error " << error);

        // Loose enough to hold at the coarsest grid tested, which is four times
        // coarser than anything the engine captures at.
        CHECK(total == Approx(kSphere).epsilon(0.02));
        // Each refinement halves the texel size, and a midpoint rule is second
        // order, so the error should fall by roughly four. Requiring only that
        // it halves leaves room for the corner texels without letting a
        // non-converging formula through.
        CHECK(error < previousError * 0.5);
        previousError = error;

        // And the weighting is doing real work: a uniform weight would be flat.
        CHECK(largest > smallest * 2.0f);
    }

    // At the resolution actually used, the quadrature error is far below
    // anything that could show up as a lighting difference.
    CHECK(previousError < 1.0e-3);
}

TEST_CASE("Cosine convolution reproduces a constant radiance field", "[probes]")
{
    // The convolution normalises by the accumulated weight, so a scene of
    // uniform radiance L must come back as exactly L in every direction. That
    // pins the normalisation and the solid-angle weighting together: get either
    // wrong and this comes out darkened, brightened, or direction-dependent --
    // all of which look like plausible lighting rather than a bug.
    //
    // This mirrors what probe_convolve.comp does, over the same helpers it
    // mirrors, so a divergence between them is what the numbers below would
    // catch on the CPU side.
    constexpr float kConstantRadiance = 0.7f;

    for (const uint32_t outputResolution : {ve::renderer::kProbeIrradianceResolution}) {
        float darkest = 1.0e9f;
        float brightest = 0.0f;

        for (uint32_t oy = 0; oy < outputResolution; ++oy) {
            for (uint32_t ox = 0; ox < outputResolution; ++ox) {
                const glm::vec3 outputDirection =
                    probeTexelDirection(ox + kProbeBorderTexels, oy + kProbeBorderTexels, outputResolution);

                float weightedSum = 0.0f;
                float weightTotal = 0.0f;
                for (uint32_t face = 0; face < kProbeCaptureFaceCount; ++face) {
                    for (uint32_t y = 0; y < kProbeCaptureFaceResolution; ++y) {
                        for (uint32_t x = 0; x < kProbeCaptureFaceResolution; ++x) {
                            const glm::vec3 sampleDirection =
                                probeCubeTexelDirection(face, x, y, kProbeCaptureFaceResolution);
                            const float cosine = glm::dot(outputDirection, sampleDirection);
                            if (cosine <= 0.0f) {
                                continue;
                            }
                            const float weight =
                                cosine * probeCubeTexelSolidAngle(x, y, kProbeCaptureFaceResolution);
                            weightedSum += kConstantRadiance * weight;
                            weightTotal += weight;
                        }
                    }
                }

                REQUIRE(weightTotal > 0.0f);
                const float result = weightedSum / weightTotal;
                darkest = std::min(darkest, result);
                brightest = std::max(brightest, result);
            }
        }

        INFO("darkest " << darkest << " brightest " << brightest);
        CHECK(darkest == Approx(kConstantRadiance).margin(1.0e-4f));
        CHECK(brightest == Approx(kConstantRadiance).margin(1.0e-4f));
    }
}

TEST_CASE("Every hemisphere gathers a meaningful share of the capture", "[probes]")
{
    // A weight total that collapses toward zero for some output directions would
    // make those texels dominated by one or two samples, which is noise rather
    // than irradiance. The cosine lobe over a full sphere of samples should
    // gather close to pi steradians-weighted no matter which way it points, so
    // the spread across output directions bounds how uneven the estimate can be.
    const uint32_t outputResolution = ve::renderer::kProbeIrradianceResolution;
    float smallestTotal = 1.0e9f;
    float largestTotal = 0.0f;

    for (uint32_t oy = 0; oy < outputResolution; ++oy) {
        for (uint32_t ox = 0; ox < outputResolution; ++ox) {
            const glm::vec3 outputDirection =
                probeTexelDirection(ox + kProbeBorderTexels, oy + kProbeBorderTexels, outputResolution);
            float weightTotal = 0.0f;
            for (uint32_t face = 0; face < kProbeCaptureFaceCount; ++face) {
                for (uint32_t y = 0; y < kProbeCaptureFaceResolution; ++y) {
                    for (uint32_t x = 0; x < kProbeCaptureFaceResolution; ++x) {
                        const float cosine = glm::dot(
                            outputDirection, probeCubeTexelDirection(face, x, y, kProbeCaptureFaceResolution));
                        if (cosine > 0.0f) {
                            weightTotal += cosine * probeCubeTexelSolidAngle(x, y, kProbeCaptureFaceResolution);
                        }
                    }
                }
            }
            smallestTotal = std::min(smallestTotal, weightTotal);
            largestTotal = std::max(largestTotal, weightTotal);
        }
    }

    // The analytic answer is pi for every direction; the discretisation moves it
    // by a fraction of a percent, not by a factor.
    INFO("smallest " << smallestTotal << " largest " << largestTotal);
    CHECK(smallestTotal == Approx(3.14159265f).epsilon(0.02));
    CHECK(largestTotal == Approx(3.14159265f).epsilon(0.02));
}

TEST_CASE("Capture tiles pack the capture atlas without overlapping", "[probes]")
{
    const glm::uvec2 atlas = probeCaptureAtlasSize();
    CHECK(atlas.x == kProbeCaptureFaceCount * kProbeCaptureFaceResolution);
    CHECK(atlas.y == kMaxProbesPerFrame * kProbeCaptureFaceResolution);

    std::set<uint64_t> occupied;
    for (uint32_t slot = 0; slot < kMaxProbesPerFrame; ++slot) {
        for (uint32_t face = 0; face < kProbeCaptureFaceCount; ++face) {
            const glm::uvec2 origin = probeCaptureTileOrigin(slot, face);
            CHECK(origin.x + kProbeCaptureFaceResolution <= atlas.x);
            CHECK(origin.y + kProbeCaptureFaceResolution <= atlas.y);
            CHECK(occupied.insert(static_cast<uint64_t>(origin.y) * atlas.x + origin.x).second);
        }
    }
    CHECK(occupied.size() == static_cast<size_t>(kMaxProbesPerFrame) * kProbeCaptureFaceCount);

    // Out-of-range slots and faces clamp into the atlas rather than addressing
    // past its end.
    const glm::uvec2 clamped = probeCaptureTileOrigin(kMaxProbesPerFrame + 4, kProbeCaptureFaceCount + 2);
    CHECK(clamped.x + kProbeCaptureFaceResolution <= atlas.x);
    CHECK(clamped.y + kProbeCaptureFaceResolution <= atlas.y);
}

TEST_CASE("Round-robin capture visits every probe once per cycle", "[probes]")
{
    // The property that matters is bounded staleness. A probe skipped
    // indefinitely holds lighting from an older version of the scene, and that
    // reads as light lagging behind the geometry rather than as a missing
    // update, so "every probe, within a known number of frames" is the
    // invariant rather than "eventually".
    for (const uint32_t perFrame : {1u, 3u, 8u, kMaxProbesPerFrame}) {
        std::vector<uint32_t> batch;
        std::vector<uint32_t> visitCount(kProbeCount, 0);

        uint32_t cursor = 0;
        const uint32_t frames = (kProbeCount + perFrame - 1) / perFrame;
        for (uint32_t frame = 0; frame < frames; ++frame) {
            cursor = probeUpdateBatch(cursor, perFrame, batch);
            CHECK(batch.size() == std::min(perFrame, kProbeCount));
            for (const uint32_t index : batch) {
                REQUIRE(index < kProbeCount);
                ++visitCount[index];
            }
        }

        for (uint32_t index = 0; index < kProbeCount; ++index) {
            INFO("perFrame " << perFrame << " probe " << index);
            CHECK(visitCount[index] >= 1);
        }
    }

    std::vector<uint32_t> batch;

    // Zero probes per frame is a legal setting (updates paused); it must not
    // advance the cursor, or resuming would skip a stripe of the grid.
    const uint32_t held = probeUpdateBatch(17, 0, batch);
    CHECK(batch.empty());
    CHECK(held == 17u);

    // Asking for more probes than exist captures each one once rather than
    // capturing some of them twice into the same capture tile.
    const uint32_t wrapped = probeUpdateBatch(0, kProbeCount + 50, batch);
    CHECK(batch.size() <= kProbeCount);
    CHECK(std::set<uint32_t>(batch.begin(), batch.end()).size() == batch.size());
    CHECK(wrapped < kProbeCount);
}

TEST_CASE("Chebyshev visibility rejects probes behind an occluder", "[probes]")
{
    // The single most characteristic artefact of probe GI is light arriving
    // through a wall, because trilinear blending on its own has no idea the wall
    // is there. This weight is the only thing that stops it, and it fails
    // quietly: too permissive and rooms glow from nowhere, too aggressive and
    // everything self-occludes into darkness.
    const auto moments = [](float distance) {
        // What a probe records when everything it sees in a direction sits at
        // exactly one distance: the two moments of a single sample.
        return std::pair<float, float>{distance, distance * distance};
    };

    // A probe whose stored distance reaches the shading point is fully visible.
    {
        const auto [mean, meanSquared] = moments(10.0f);
        CHECK(probeChebyshevVisibility(mean, meanSquared, 5.0f) == Approx(1.0f));
        CHECK(probeChebyshevVisibility(mean, meanSquared, 10.0f) == Approx(1.0f));
    }

    // Past it, the weight falls and keeps falling -- monotonically, or the
    // shading would brighten again further behind the occluder.
    {
        const auto [mean, meanSquared] = moments(10.0f);
        float previous = 1.0f;
        for (float distance = 10.0f; distance <= 30.0f; distance += 0.5f) {
            const float visibility = probeChebyshevVisibility(mean, meanSquared, distance);
            CHECK(visibility <= previous + 1.0e-6f);
            CHECK(visibility >= 0.0f);
            CHECK(visibility <= 1.0f);
            previous = visibility;
        }
        // Well past a hard occluder the probe is effectively rejected.
        CHECK(probeChebyshevVisibility(mean, meanSquared, 25.0f) < 0.05f);
    }

    // A blurry occluder -- high variance -- rejects far more gently than a sharp
    // one at the same mean. That is the whole point of storing the second
    // moment rather than just the distance.
    {
        const float sharp = probeChebyshevVisibility(10.0f, 100.0f, 14.0f);
        const float blurry = probeChebyshevVisibility(10.0f, 160.0f, 14.0f);
        INFO("sharp " << sharp << " blurry " << blurry);
        CHECK(blurry > sharp);
    }

    // Bilinear filtering of the two moments can produce a mean square below the
    // squared mean, which no single texel can. Without the absolute value the
    // variance goes negative and the weight comes back negative -- light gets
    // *subtracted*, showing up as a black fringe that reads as an occlusion
    // artefact rather than as arithmetic.
    for (const float meanSquared : {0.0f, 50.0f, 99.0f}) {
        const float visibility = probeChebyshevVisibility(10.0f, meanSquared, 20.0f);
        INFO("meanSquared " << meanSquared << " -> " << visibility);
        CHECK(std::isfinite(visibility));
        CHECK(visibility >= 0.0f);
        CHECK(visibility <= 1.0f);
    }

    // Degenerate inputs must not produce NaN: a NaN weight propagates into the
    // accumulated irradiance and never washes out.
    CHECK(std::isfinite(probeChebyshevVisibility(0.0f, 0.0f, 0.0f)));
    CHECK(std::isfinite(probeChebyshevVisibility(0.0f, 0.0f, 5.0f)));
    CHECK(std::isfinite(probeChebyshevVisibility(-1.0f, -1.0f, 5.0f)));
    // A NaN distance decodes to visible rather than sliding through the divide.
    CHECK(probeChebyshevVisibility(10.0f, 100.0f, std::numeric_limits<float>::quiet_NaN()) == Approx(1.0f));
}

TEST_CASE("Probe direction weight falls off smoothly behind the surface", "[probes]")
{
    const glm::vec3 normal{0.0f, 1.0f, 0.0f};

    // Directly above beats edge-on beats directly below, with no discontinuity
    // in between -- a hard cutoff would draw a line across smooth shading where
    // no geometry changes.
    const float above = probeDirectionWeight(normal, glm::vec3{0.0f, 1.0f, 0.0f});
    const float edge = probeDirectionWeight(normal, glm::vec3{1.0f, 0.0f, 0.0f});
    const float below = probeDirectionWeight(normal, glm::vec3{0.0f, -1.0f, 0.0f});
    CHECK(above > edge);
    CHECK(edge > below);

    // A probe fully behind the surface still contributes a little rather than
    // vanishing, which is what keeps the falloff from having an edge of its own.
    CHECK(below > 0.0f);

    float previous = -1.0f;
    for (int step = 0; step <= 64; ++step) {
        const float angle = static_cast<float>(step) / 64.0f * 3.14159265f;
        const glm::vec3 direction{std::sin(angle), std::cos(angle), 0.0f};
        const float weight = probeDirectionWeight(normal, direction);
        CHECK(std::isfinite(weight));
        if (previous >= 0.0f) {
            // Monotonic as the probe swings from overhead to underneath.
            CHECK(weight <= previous + 1.0e-6f);
        }
        previous = weight;
    }
}

TEST_CASE("The probe sample position leaves the surface it shades", "[probes]")
{
    const glm::vec3 position{2.0f, 1.0f, -3.0f};
    const glm::vec3 normal{0.0f, 1.0f, 0.0f};
    const glm::vec3 view{0.0f, 0.0f, 1.0f};

    // Zero bias is the identity, so the offset is opt-in rather than baked in.
    CHECK(probeSamplePosition(position, normal, view, 0.0f) == position);

    // A head-on surface moves off itself along the normal.
    const glm::vec3 biased = probeSamplePosition(position, normal, normal, 0.5f);
    CHECK(glm::dot(biased - position, normal) > 0.0f);

    // At a grazing angle the normal component alone barely clears the surface,
    // which is exactly where self-occlusion shows; the view term is what carries
    // the offset there.
    const glm::vec3 grazing = probeSamplePosition(position, normal, view, 0.5f);
    CHECK(glm::length(grazing - position) > 0.2f);

    // A negative bias must not pull the sample *into* the surface.
    CHECK(probeSamplePosition(position, normal, view, -1.0f) == position);
}

TEST_CASE("Jittered capture directions agree with the jittered projection", "[probes]")
{
    // The capture is deterministic -- the same fixed directions every update --
    // so accumulating successive captures of an unchanged scene would gain
    // exactly nothing. Sub-texel jitter is what makes accumulation worth doing,
    // and it only works if the projection and the direction reconstruction move
    // by the same amount in opposite directions. Get the sign wrong and the
    // offset doubles instead of cancelling, which reads as a smeared probe
    // rather than as anything obviously broken.
    const glm::vec3 probePosition{-1.0f, 4.0f, 2.0f};
    const auto resolution = static_cast<float>(kProbeCaptureFaceResolution);

    for (const uint32_t update : {0u, 1u, 5u, 37u}) {
        const glm::vec2 jitter = probeCaptureJitter(update);
        // A sub-texel offset: larger would sample a neighbouring texel's cone
        // and stop being an anti-aliasing device.
        CHECK(std::abs(jitter.x) <= 0.5f);
        CHECK(std::abs(jitter.y) <= 0.5f);

        for (uint32_t face = 0; face < kProbeCaptureFaceCount; ++face) {
            const glm::mat4 viewProjection = probeCaptureFaceViewProjection(probePosition, face, jitter);
            for (const uint32_t texel : {0u, 3u, kProbeCaptureFaceResolution - 1u}) {
                const glm::vec3 direction =
                    probeCubeTexelDirection(face, texel, texel, kProbeCaptureFaceResolution, jitter);
                const glm::vec4 clip = viewProjection * glm::vec4{probePosition + direction * 8.0f, 1.0f};
                REQUIRE(clip.w > 0.0f);
                const glm::vec2 pixel{((clip.x / clip.w) * 0.5f + 0.5f) * resolution,
                                      ((clip.y / clip.w) * 0.5f + 0.5f) * resolution};
                CHECK(pixel.x == Approx(static_cast<float>(texel) + 0.5f).margin(1.0e-3f));
                CHECK(pixel.y == Approx(static_cast<float>(texel) + 0.5f).margin(1.0e-3f));
            }
        }
    }

    // Successive updates must actually move, or accumulation averages identical
    // samples and buys nothing -- the exact failure this whole mechanism exists
    // to avoid.
    std::vector<glm::vec2> offsets;
    for (uint32_t update = 0; update < 16; ++update) {
        offsets.push_back(probeCaptureJitter(update));
    }
    for (size_t i = 0; i < offsets.size(); ++i) {
        for (size_t j = i + 1; j < offsets.size(); ++j) {
            CHECK(glm::length(offsets[i] - offsets[j]) > 1.0e-4f);
        }
    }

    // Zero jitter is exactly the unjittered projection, so the default path is
    // untouched by any of this.
    for (uint32_t face = 0; face < kProbeCaptureFaceCount; ++face) {
        const glm::mat4 plain = probeCaptureFaceViewProjection(probePosition, face);
        const glm::mat4 zero = probeCaptureFaceViewProjection(probePosition, face, glm::vec2{0.0f});
        CHECK(plain == zero);
        CHECK(probeCubeTexelDirection(face, 2, 5, kProbeCaptureFaceResolution) ==
              probeCubeTexelDirection(face, 2, 5, kProbeCaptureFaceResolution, glm::vec2{0.0f}));
    }
}

TEST_CASE("Bounce feedback amplification stays finite", "[probes]")
{
    // Feeding the probe atlas back into the capture is a feedback loop: each
    // round multiplies by albedo * weight. That is the intended behaviour --
    // it is what makes a second bounce exist -- but it is also the one place in
    // this subsystem where a bad parameter does not merely look wrong, it grows
    // without bound. The exponent is what says whether a chosen weight is safe.

    // No feedback is no amplification, which is the reference the multi-bounce
    // result is judged against.
    CHECK(probeBounceAmplification(0.8f, 0.0f) == Approx(1.0f));
    CHECK(probeBounceAmplification(0.0f, 1.0f) == Approx(1.0f));

    // The geometric series, spot-checked against the closed form.
    CHECK(probeBounceAmplification(0.5f, 0.5f) == Approx(1.0f / (1.0f - 0.25f)));
    CHECK(probeBounceAmplification(0.8f, 0.5f) == Approx(1.0f / (1.0f - 0.40f)));

    // Monotonic in both, and never below one -- indirect light cannot make a
    // surface darker than no indirect light at all.
    float previous = 0.0f;
    for (int step = 0; step <= 20; ++step) {
        const float weight = static_cast<float>(step) / 20.0f;
        const float gain = probeBounceAmplification(0.7f, weight);
        CHECK(gain >= 1.0f);
        CHECK(gain >= previous);
        CHECK(std::isfinite(gain));
        previous = gain;
    }

    // The degenerate corner: a perfectly white surface with full feedback
    // genuinely diverges. It must come back as a large finite number rather than
    // an infinity, which would reach the shading as a NaN and never wash out.
    const float degenerate = probeBounceAmplification(1.0f, 1.0f);
    CHECK(std::isfinite(degenerate));
    CHECK(degenerate > 10.0f);

    // Out-of-range inputs clamp rather than producing a negative denominator,
    // which would return a *negative* gain -- light that subtracts.
    CHECK(probeBounceAmplification(2.0f, 2.0f) == Approx(degenerate));
    CHECK(probeBounceAmplification(-1.0f, 0.5f) == Approx(1.0f));
    CHECK(probeBounceAmplification(0.5f, -1.0f) == Approx(1.0f));

    // The default weight keeps a bright surface well inside a sane range.
    const float defaultGain = probeBounceAmplification(0.8f, ve::renderer::kProbeDefaultBounceWeight);
    INFO("albedo 0.8 at the default weight amplifies by " << defaultGain);
    CHECK(defaultGain < 2.0f);
}
