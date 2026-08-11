#include "renderer/RenderScale.h"

#include "renderer/RuntimeSettings.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using ve::renderer::clampRenderScale;
using ve::renderer::kMaxRenderScale;
using ve::renderer::kMinRenderScale;
using ve::renderer::RenderDimensions;
using ve::renderer::scaledRenderDimensions;

TEST_CASE("Render scale clamps into the supported range", "[render-scale]")
{
    CHECK(clampRenderScale(0.75f) == Catch::Approx(0.75f));
    CHECK(clampRenderScale(kMinRenderScale) == Catch::Approx(kMinRenderScale));
    CHECK(clampRenderScale(kMaxRenderScale) == Catch::Approx(kMaxRenderScale));

    // Below the floor the image is unusable; above 1.0 would be supersampling,
    // which the composite's bilinear stretch is the wrong filter for.
    CHECK(clampRenderScale(0.01f) == Catch::Approx(kMinRenderScale));
    CHECK(clampRenderScale(-3.0f) == Catch::Approx(kMinRenderScale));
    CHECK(clampRenderScale(2.0f) == Catch::Approx(kMaxRenderScale));
}

TEST_CASE("A NaN render scale falls back to native resolution", "[render-scale]")
{
    // std::clamp would propagate NaN (every comparison is false), and it would
    // then reach vkCreateImage as a garbage extent rather than failing here.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(clampRenderScale(nan) == Catch::Approx(kMaxRenderScale));

    const RenderDimensions dimensions = scaledRenderDimensions(1600, 900, nan);
    CHECK(dimensions.width == 1600u);
    CHECK(dimensions.height == 900u);
}

TEST_CASE("Scale 1.0 leaves the resolution untouched", "[render-scale]")
{
    const RenderDimensions dimensions = scaledRenderDimensions(1512, 982, 1.0f);
    CHECK(dimensions.width == 1512u);
    CHECK(dimensions.height == 982u);
}

TEST_CASE("Scaled dimensions round to nearest", "[render-scale]")
{
    // 1025 * 0.5 = 512.5 -- truncation would lose a pixel on every odd
    // dimension, which over a chain of mips compounds into a visible offset.
    const RenderDimensions half = scaledRenderDimensions(1025, 769, 0.5f);
    CHECK(half.width == 513u);
    CHECK(half.height == 385u);

    const RenderDimensions threeQuarters = scaledRenderDimensions(1920, 1080, 0.75f);
    CHECK(threeQuarters.width == 1440u);
    CHECK(threeQuarters.height == 810u);
}

TEST_CASE("Scaled dimensions never collapse to zero", "[render-scale]")
{
    // A one-pixel-tall window at the minimum scale still has to produce a legal
    // attachment size.
    const RenderDimensions tiny = scaledRenderDimensions(3, 1, kMinRenderScale);
    CHECK(tiny.width >= 1u);
    CHECK(tiny.height == 1u);
}

TEST_CASE("A zero-sized surface passes through unchanged", "[render-scale]")
{
    // A minimised window: the caller's own zero-extent guard should be the one
    // that fires, not a silently invented 1x1 target.
    const RenderDimensions minimised = scaledRenderDimensions(0, 0, 0.5f);
    CHECK(minimised.width == 0u);
    CHECK(minimised.height == 0u);
}

TEST_CASE("Render scale is clamped by the runtime settings clamp", "[render-scale][settings]")
{
    ve::RenderScaleSettings renderScale{};
    ve::DynamicResolutionSettings dynamicResolution{};
    ve::ToneMappingSettings toneMapping{};
    ve::BloomSettings bloom{};
    ve::TaaSettings taa{};
    ve::SsrSettings ssr{};
    ve::SsaoSettings ssao{};
    ve::FogSettings fog{};
    ve::CsmSettings csm{};
    ve::LodSettings lod{};
    ve::GiSettings gi{};
    ve::DebugUiSettings debugUi{};

    renderScale.scale = 4.0f;
    ve::clampRuntimeSettings(
        renderScale, dynamicResolution, toneMapping, bloom, taa, ssr, ssao, fog, csm, lod, gi, debugUi);
    CHECK(renderScale.scale == Catch::Approx(kMaxRenderScale));

    renderScale.scale = 0.0f;
    ve::clampRuntimeSettings(
        renderScale, dynamicResolution, toneMapping, bloom, taa, ssr, ssao, fog, csm, lod, gi, debugUi);
    CHECK(renderScale.scale == Catch::Approx(kMinRenderScale));
}

TEST_CASE("Render scale defaults to native", "[render-scale][settings]")
{
    // The default has to be 1.0: a persisted settings file written before this
    // setting existed leaves it at the default, and that must not change how the
    // engine looks for anyone who never touches the slider.
    const ve::RuntimeSettings defaults{};
    CHECK(defaults.renderScale.scale == Catch::Approx(1.0f));
}

TEST_CASE("Sharpness is clamped and defaults to a usable value", "[render-scale][settings]")
{
    ve::RenderScaleSettings renderScale{};
    ve::DynamicResolutionSettings dynamicResolution{};
    ve::ToneMappingSettings toneMapping{};
    ve::BloomSettings bloom{};
    ve::TaaSettings taa{};
    ve::SsrSettings ssr{};
    ve::SsaoSettings ssao{};
    ve::FogSettings fog{};
    ve::CsmSettings csm{};
    ve::LodSettings lod{};
    ve::GiSettings gi{};
    ve::DebugUiSettings debugUi{};

    // Non-zero by default: an A/B on the geometry stress scene, whose silhouette
    // density sits at the render-resolution scale, showed the filter working, so
    // the benefit is not hypothetical. It costs nothing at scale 1.0, where the
    // renderer skips it. (Not justified by "0.5 is too soft" -- what makes 0.5
    // bad without TAA is aliasing, which is TAA's job, not this filter's.)
    CHECK(renderScale.sharpness > 0.0f);

    renderScale.sharpness = 9.0f;
    ve::clampRuntimeSettings(
        renderScale, dynamicResolution, toneMapping, bloom, taa, ssr, ssao, fog, csm, lod, gi, debugUi);
    CHECK(renderScale.sharpness == Catch::Approx(1.0f));

    renderScale.sharpness = -1.0f;
    ve::clampRuntimeSettings(
        renderScale, dynamicResolution, toneMapping, bloom, taa, ssr, ssao, fog, csm, lod, gi, debugUi);
    CHECK(renderScale.sharpness == Catch::Approx(0.0f));
}
