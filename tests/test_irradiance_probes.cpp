#include "renderer/IrradianceProbes.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>

#include <cmath>
#include <set>

using Catch::Approx;
using ve::renderer::kProbeCount;
using ve::renderer::kProbeGridX;
using ve::renderer::kProbeGridY;
using ve::renderer::kProbeGridZ;
using ve::renderer::octahedralDecode;
using ve::renderer::octahedralEncode;
using ve::renderer::ProbeBlend;
using ve::renderer::probeBlendAt;
using ve::renderer::probeCoord;
using ve::renderer::probeCornerCoord;
using ve::renderer::probeCornerWeight;
using ve::renderer::ProbeGridBounds;
using ve::renderer::probeIndex;
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
