#include "renderer/RuntimeSettings.h"

#include "renderer/CascadeMath.h"
#include "renderer/MeshLod.h"
#include "renderer/VolumetricFog.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using ve::BloomSettings;
using ve::clampRuntimeSettings;
using ve::CsmSettings;
using ve::DebugUiSettings;
using ve::ExposureMode;
using ve::FogSettings;
using ve::GiSettings;
using ve::LodSettings;
using ve::exposureModeValue;
using ve::SsaoSettings;
using ve::TaaSettings;
using ve::SsrSettings;
using ve::ToneMappingSettings;

namespace {

// Run the clamp over fresh structs the caller can pre-mutate.
struct Settings {
    ToneMappingSettings toneMapping;
    BloomSettings bloom;
    TaaSettings taa;
    SsrSettings ssr;
    SsaoSettings ssao;
    FogSettings fog;
    CsmSettings csm;
    LodSettings lod;
    GiSettings gi;
    DebugUiSettings debugUi;

    void clamp() { clampRuntimeSettings(toneMapping, bloom, taa, ssr, ssao, fog, csm, lod, gi, debugUi); }
};

} // namespace

TEST_CASE("exposureModeValue maps ints to valid modes", "[settings]")
{
    CHECK(exposureModeValue(0) == ExposureMode::Manual);
    CHECK(exposureModeValue(1) == ExposureMode::LogAverage);
    CHECK(exposureModeValue(2) == ExposureMode::Histogram);
    // Out-of-range values fall back to Histogram.
    CHECK(exposureModeValue(-5) == ExposureMode::Histogram);
    CHECK(exposureModeValue(99) == ExposureMode::Histogram);
}

TEST_CASE("Disabling auto exposure forces manual mode", "[settings]")
{
    Settings s;
    s.toneMapping.enableAutoExposure = false;
    s.toneMapping.exposureMode = 2;
    s.clamp();
    CHECK(s.toneMapping.exposureMode == static_cast<int>(ExposureMode::Manual));
}

TEST_CASE("Tone-mapping numeric ranges are clamped", "[settings]")
{
    Settings s;
    s.toneMapping.operatorType = 7;
    s.toneMapping.manualExposure = -3.0f;
    s.toneMapping.targetLuminance = -1.0f;
    s.toneMapping.minExposure = -2.0f;
    s.toneMapping.maxExposure = -5.0f; // below minExposure
    s.toneMapping.adaptationRate = -1.0f;
    s.clamp();

    CHECK(s.toneMapping.operatorType == 1);
    CHECK(s.toneMapping.manualExposure == Catch::Approx(0.0f));
    CHECK(s.toneMapping.targetLuminance >= ve::kMinAverageLuminance);
    CHECK(s.toneMapping.minExposure == Catch::Approx(0.0f));
    CHECK(s.toneMapping.maxExposure >= s.toneMapping.minExposure);
    CHECK(s.toneMapping.adaptationRate == Catch::Approx(0.0f));
}

TEST_CASE("Percentiles stay ordered with high above low", "[settings]")
{
    Settings s;
    s.toneMapping.lowPercentile = 0.9f;
    s.toneMapping.highPercentile = 0.2f; // inverted
    s.clamp();
    CHECK(s.toneMapping.highPercentile > s.toneMapping.lowPercentile);
    CHECK(s.toneMapping.lowPercentile >= 0.0f);
    CHECK(s.toneMapping.highPercentile <= 1.0f);
}

TEST_CASE("Bloom and TAA settings are clamped", "[settings]")
{
    Settings s;
    s.bloom.threshold = -1.0f;
    s.bloom.intensity = -1.0f;
    s.bloom.radius = 10.0f;
    s.taa.feedback = 5.0f;
    s.clamp();

    CHECK(s.bloom.threshold == Catch::Approx(0.0f));
    CHECK(s.bloom.intensity == Catch::Approx(0.0f));
    CHECK(s.bloom.radius == Catch::Approx(4.0f));
    CHECK(s.taa.feedback == Catch::Approx(0.98f));
}

TEST_CASE("Cascade settings and selected-cascade index are clamped", "[settings]")
{
    SECTION("cascade count clamps into [1, max] and bounds the selected index")
    {
        Settings s;
        s.csm.cascadeCount = 99;
        s.csm.lambda = 5.0f;
        s.debugUi.selectedCsmCascade = 100;
        s.clamp();

        CHECK(s.csm.cascadeCount == ve::renderer::kMaxShadowCascades);
        CHECK(s.csm.lambda == Catch::Approx(1.0f));
        CHECK(s.debugUi.selectedCsmCascade == s.csm.cascadeCount - 1U);
    }

    SECTION("zero cascade count is raised to one without underflowing the index")
    {
        Settings s;
        s.csm.cascadeCount = 0;
        s.debugUi.selectedCsmCascade = 3;
        s.clamp();

        CHECK(s.csm.cascadeCount == 1U);
        CHECK(s.debugUi.selectedCsmCascade == 0U);
    }

    SECTION("shadow distance is clamped within the near/far range")
    {
        Settings s;
        s.csm.nearPlane = 0.1f;
        s.csm.farPlane = 100.0f;
        s.csm.shadowDistance = 1000.0f;
        s.clamp();
        CHECK(s.csm.shadowDistance == Catch::Approx(100.0f));
    }
}

TEST_CASE("Debug-UI preview ranges are clamped", "[settings]")
{
    Settings s;
    s.debugUi.renderTargetPreviewExposure = 100.0f;
    s.debugUi.renderTargetPreviewScale = 100.0f;
    s.clamp();
    CHECK(s.debugUi.renderTargetPreviewExposure == Catch::Approx(8.0f));
    CHECK(s.debugUi.renderTargetPreviewScale == Catch::Approx(2.0f));
}

TEST_CASE("SSR settings clamp into their valid ranges", "[settings]")
{
    Settings settings;
    settings.ssr.maxSteps = 4096;
    settings.ssr.refinementSteps = -3;
    settings.ssr.maxDistance = 100000.0f;
    settings.ssr.thickness = 0.0f;
    settings.ssr.intensity = -1.0f;
    settings.ssr.maxRoughness = 2.0f;
    settings.ssr.screenEdgeFade = 0.9f;
    settings.clamp();

    CHECK(settings.ssr.maxSteps == 128);
    CHECK(settings.ssr.refinementSteps == 0);
    CHECK(settings.ssr.maxDistance == 200.0f);
    CHECK(settings.ssr.thickness == 0.01f);
    CHECK(settings.ssr.intensity == 0.0f);
    CHECK(settings.ssr.maxRoughness == 1.0f);
    CHECK(settings.ssr.screenEdgeFade == 0.49f);
}

TEST_CASE("LOD settings clamp into usable ranges", "[settings][lod]")
{
    Settings settings;
    settings.lod.referenceRadiusPixels = 0.0f;
    settings.lod.bias = 100.0f;
    settings.lod.shadowBias = -100.0f;
    settings.clamp();

    // A reference radius at or below a pixel would select the coarsest level for
    // everything on screen.
    CHECK(settings.lod.referenceRadiusPixels >= 8.0f);
    CHECK(settings.lod.bias <= 4.0f);
    CHECK(settings.lod.shadowBias >= -4.0f);
}

TEST_CASE("Forced LOD keeps its select-by-distance sentinel", "[settings][lod]")
{
    Settings settings;

    settings.lod.forcedLod = -1;
    settings.clamp();
    CHECK(settings.lod.forcedLod == -1);

    // Anything below the sentinel collapses onto it rather than becoming a
    // different negative value the shader would not recognise.
    settings.lod.forcedLod = -50;
    settings.clamp();
    CHECK(settings.lod.forcedLod == -1);

    // And a forced level can never exceed the longest chain the builder makes.
    settings.lod.forcedLod = 99;
    settings.clamp();
    CHECK(settings.lod.forcedLod == static_cast<int>(ve::renderer::kMaxMeshLods) - 1);
}

TEST_CASE("GTAO loop counts stay inside the shader's bounds", "[settings][ssao]")
{
    Settings settings;

    // Slices and steps bound nested loops in the GTAO shader, so an absurd
    // stored value is a performance cliff rather than a visual artefact.
    settings.ssao.sliceCount = 4096;
    settings.ssao.stepsPerSlice = 4096;
    settings.clamp();
    CHECK(settings.ssao.sliceCount == 8);
    CHECK(settings.ssao.stepsPerSlice == 16);

    settings.ssao.sliceCount = 0;
    settings.ssao.stepsPerSlice = 0;
    settings.clamp();
    CHECK(settings.ssao.sliceCount == 1);
    CHECK(settings.ssao.stepsPerSlice == 2);

    // A zero radius collapses the horizon search to a point, which reports no
    // occlusion at all -- indistinguishable from the feature being broken.
    settings.ssao.radius = 0.0f;
    settings.clamp();
    CHECK(settings.ssao.radius > 0.0f);
}

TEST_CASE("Fog max distance stays past the volume's near plane", "[settings][fog]")
{
    Settings settings;

    // maxDistance is a divisor in the froxel slice distribution and must stay
    // strictly greater than the fog volume's near plane.
    settings.fog.maxDistance = 0.0f;
    settings.clamp();
    CHECK(settings.fog.maxDistance > ve::renderer::kFogNearPlane);

    settings.fog.maxDistance = -1000.0f;
    settings.clamp();
    CHECK(settings.fog.maxDistance > ve::renderer::kFogNearPlane);
}

TEST_CASE("Fog anisotropy stays inside the phase function's domain", "[settings][fog]")
{
    Settings settings;

    // The Henyey-Greenstein denominator reaches zero at |g| == 1, so the clamp
    // has to stop short of it on both sides.
    settings.fog.anisotropy = 1.0f;
    settings.clamp();
    CHECK(settings.fog.anisotropy < 1.0f);

    settings.fog.anisotropy = -1.0f;
    settings.clamp();
    CHECK(settings.fog.anisotropy > -1.0f);
}

TEST_CASE("Fog temporal blend never keeps the whole history", "[settings][fog]")
{
    Settings settings;

    // At 1.0 the reprojected volume would be all history and never converge.
    settings.fog.temporalBlend = 1.0f;
    settings.clamp();
    CHECK(settings.fog.temporalBlend < 1.0f);
}

TEST_CASE("Fog scattering colour stays a valid albedo", "[settings][fog]")
{
    Settings settings;

    settings.fog.scatteringColor[0] = 5.0f;
    settings.fog.scatteringColor[1] = -5.0f;
    settings.fog.scatteringColor[2] = 0.5f;
    settings.clamp();

    CHECK(settings.fog.scatteringColor[0] == Catch::Approx(1.0f));
    CHECK(settings.fog.scatteringColor[1] == Catch::Approx(0.0f));
    CHECK(settings.fog.scatteringColor[2] == Catch::Approx(0.5f));
}
