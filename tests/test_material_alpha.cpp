#include "renderer/Material.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

using ve::renderer::AlphaMode;
using ve::renderer::alphaModeFromString;
using ve::renderer::alphaModeToString;
using ve::renderer::kNoAlphaTestCutoff;
using ve::renderer::Material;

TEST_CASE("Alpha mode parses the glTF spellings", "[material][alpha]")
{
    CHECK(alphaModeFromString("OPAQUE") == AlphaMode::Opaque);
    CHECK(alphaModeFromString("MASK") == AlphaMode::Mask);
    CHECK(alphaModeFromString("BLEND") == AlphaMode::Blend);
}

TEST_CASE("Unknown or empty alpha modes fall back to opaque", "[material][alpha]")
{
    // Material::alphaMode is a free-form string coming from asset JSON and glTF,
    // so anything unrecognised has to degrade to the safe mode rather than land in
    // an unhandled bucket.
    CHECK(alphaModeFromString("") == AlphaMode::Opaque);
    CHECK(alphaModeFromString("mask") == AlphaMode::Opaque);
    CHECK(alphaModeFromString("TRANSPARENT") == AlphaMode::Opaque);
}

TEST_CASE("Alpha mode round-trips through its string form", "[material][alpha]")
{
    for (const AlphaMode mode : {AlphaMode::Opaque, AlphaMode::Mask, AlphaMode::Blend}) {
        CHECK(alphaModeFromString(alphaModeToString(mode)) == mode);
    }
}

TEST_CASE("Only MASK materials report an alpha-test cutoff", "[material][alpha]")
{
    // The shaders branch on the sign of materialParams.w, so OPAQUE and BLEND must
    // encode as a negative sentinel; a zero cutoff would still enable the test.
    Material material{};
    material.alphaCutoff = 0.25f;

    material.alphaMode = "OPAQUE";
    CHECK(material.alphaTestCutoff() == Catch::Approx(kNoAlphaTestCutoff));
    CHECK(material.alphaTestCutoff() < 0.0f);

    material.alphaMode = "BLEND";
    CHECK(material.alphaTestCutoff() == Catch::Approx(kNoAlphaTestCutoff));
    CHECK(material.alphaTestCutoff() < 0.0f);

    material.alphaMode = "MASK";
    CHECK(material.alphaTestCutoff() == Catch::Approx(0.25f));
}

TEST_CASE("A MASK material with a zero cutoff still enables the test", "[material][alpha]")
{
    // Boundary case for the sign-based encoding: cutoff 0 is a legal glTF value and
    // must stay distinguishable from "no test at all".
    Material material{};
    material.alphaMode = "MASK";
    material.alphaCutoff = 0.0f;

    CHECK(material.alphaTestCutoff() == Catch::Approx(0.0f));
    CHECK(material.alphaTestCutoff() >= 0.0f);
}

TEST_CASE("Default materials are opaque", "[material][alpha]")
{
    const Material material{};
    CHECK(material.alphaModeValue() == AlphaMode::Opaque);
    CHECK(material.alphaTestCutoff() < 0.0f);
}
