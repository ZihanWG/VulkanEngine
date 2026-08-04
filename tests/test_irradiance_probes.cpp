#include "renderer/IrradianceProbes.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

using Catch::Approx;
using ve::renderer::kProbeAtlasTilesX;
using ve::renderer::kProbeAtlasTilesY;
using ve::renderer::kProbeBorderTexels;
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
