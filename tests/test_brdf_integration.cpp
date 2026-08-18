#include "core/JobSystem.h"
#include "renderer/BrdfIntegration.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using ve::JobSystem;
using ve::renderer::integrateBrdf;
using ve::renderer::kBrdfLutChannels;
using ve::renderer::makeSplitSumBrdfLut;

TEST_CASE("The BRDF table is byte-identical serial or parallel", "[brdf]")
{
    // The safety argument for building it on the pool. Every texel writes only
    // its own two bytes, so unlike the LOD chains there is no ordering to
    // preserve -- but that has to be true in fact, not just in principle.
    JobSystem jobSystem;

    for (uint32_t size : {1U, 2U, 7U, 32U}) {
        const std::vector<uint8_t> serial = makeSplitSumBrdfLut(size, nullptr);
        const std::vector<uint8_t> parallel = makeSplitSumBrdfLut(size, &jobSystem);

        REQUIRE(serial.size() == static_cast<size_t>(size) * size * kBrdfLutChannels);
        CHECK(serial == parallel);
    }
}

TEST_CASE("The BRDF table is reproducible across runs", "[brdf]")
{
    // It depends on nothing but its arguments, which is what would let it be
    // precomputed and shipped. If this ever fails, that option is gone too.
    CHECK(makeSplitSumBrdfLut(16) == makeSplitSumBrdfLut(16));
}

TEST_CASE("Split-sum values sit where the model says they should", "[brdf]")
{
    // Smooth and head-on: almost all energy is in the scale term and the Fresnel
    // bias is near zero. This is the corner the split-sum approximation is
    // clearest about, so it is the one worth pinning.
    const auto smoothHeadOn = integrateBrdf(1.0f, 0.02f);
    CHECK(smoothHeadOn[0] > 0.9f);
    CHECK(smoothHeadOn[1] < 0.1f);

    // Both terms are energy fractions, so neither may leave [0, 1] anywhere in
    // the table -- a value outside it would brighten or darken IBL silently.
    for (uint32_t y = 0; y < 16; ++y) {
        const float roughness = (static_cast<float>(y) + 0.5f) / 16.0f;
        for (uint32_t x = 0; x < 16; ++x) {
            const float normalView = (static_cast<float>(x) + 0.5f) / 16.0f;
            const auto brdf = integrateBrdf(normalView, roughness);
            CHECK(brdf[0] >= 0.0f);
            CHECK(brdf[0] <= 1.0f);
            CHECK(brdf[1] >= 0.0f);
            CHECK(brdf[1] <= 1.0f);
        }
    }
}

TEST_CASE("A zero-sized BRDF table is rejected", "[brdf]")
{
    CHECK_THROWS_AS(makeSplitSumBrdfLut(0), std::runtime_error);

    JobSystem jobSystem;
    CHECK_THROWS_AS(makeSplitSumBrdfLut(0, &jobSystem), std::runtime_error);
}
