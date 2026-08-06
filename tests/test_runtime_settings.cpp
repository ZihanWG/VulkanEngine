#include "renderer/RuntimeSettings.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>

using ve::RuntimeSettings;
using ve::RuntimeSettingsLoadStatus;

namespace {

// Unique temp path per test so parallel ctest runs do not collide.
std::filesystem::path makeTempSettingsPath()
{
    std::random_device device;
    const std::string name = "ve_runtime_settings_" + std::to_string(device()) + ".json";
    return std::filesystem::temp_directory_path() / name;
}

// Mutate every field that RuntimeSettings.cpp actually serializes to a non-default
// value. Fields that are intentionally not persisted (csm near/far plane, depth
// bias) are deliberately left untouched so the round-trip assertions stay honest.
RuntimeSettings makeNonDefaultSettings()
{
    RuntimeSettings settings;

    settings.toneMapping.manualExposure = 2.5f;
    settings.toneMapping.enableAutoExposure = true; // required for exposureMode to round-trip
    settings.toneMapping.exposureMode = 1;          // LogAverage
    settings.toneMapping.targetLuminance = 0.25f;
    settings.toneMapping.minExposure = 0.05f;
    settings.toneMapping.maxExposure = 12.0f;
    settings.toneMapping.adaptationRate = 2.0f;
    settings.toneMapping.histogramMinLogLuminance = -8.5f;
    settings.toneMapping.histogramMaxLogLuminance = 3.5f;
    settings.toneMapping.lowPercentile = 0.1f;
    settings.toneMapping.highPercentile = 0.9f;
    settings.toneMapping.operatorType = 1; // ACES

    settings.bloom.enabled = false;
    settings.bloom.useMipChain = false;
    settings.bloom.threshold = 1.5f;
    settings.bloom.intensity = 0.25f;
    settings.bloom.radius = 1.75f;

    settings.taa.enabled = true;
    settings.taa.jitterEnabled = false;
    settings.taa.neighborhoodClampEnabled = false;
    settings.taa.reprojectionEnabled = false;
    settings.taa.feedback = 0.75f;

    settings.ssr.enabled = false;
    settings.ssr.maxSteps = 64;
    settings.ssr.refinementSteps = 3;
    settings.ssr.maxDistance = 55.0f;
    settings.ssr.thickness = 0.5f;
    settings.ssr.intensity = 1.5f;
    settings.ssr.maxRoughness = 0.4f;
    settings.ssr.screenEdgeFade = 0.2f;

    settings.ssao.enabled = true;
    settings.ssao.radius = 1.25f;
    settings.ssao.intensity = 2.0f;
    settings.ssao.power = 3.0f;
    settings.ssao.sliceCount = 5;
    settings.ssao.stepsPerSlice = 9;
    settings.ssao.falloff = 0.8f;
    settings.ssao.thickness = 0.75f;

    settings.fog.enabled = true;
    settings.fog.density = 0.12f;
    settings.fog.maxDistance = 96.0f;
    settings.fog.anisotropy = -0.4f;
    settings.fog.heightFalloff = 0.2f;
    settings.fog.baseHeight = 3.5f;
    settings.fog.ambientScale = 1.4f;
    settings.fog.scatteringColor[0] = 0.8f;
    settings.fog.scatteringColor[1] = 0.6f;
    settings.fog.scatteringColor[2] = 0.4f;
    settings.fog.lightCullThreshold = 0.2f;
    settings.fog.temporalEnabled = false;
    settings.fog.temporalBlend = 0.5f;

    settings.punctualShadows.enabled = false;
    settings.punctualShadows.gpuCasterCulling = true;
    settings.punctualShadows.debugView = true;

    settings.csm.cascadeCount = 3;
    settings.csm.lambda = 0.7f;
    settings.csm.shadowDistance = 60.0f;
    settings.csm.enableTexelSnapping = false;
    settings.csm.enableCascadeDebugColors = true;

    settings.useGpuCulling = false;
    settings.useGpuShadowCulling = false;
    settings.enableGpuOcclusionCulling = false;
    settings.enableTwoPhaseOcclusion = false;
    settings.enableAsyncCompute = false;
    settings.enableBindlessMaterialTextures = false;

    settings.debugUi.showRenderGraphPanel = false;
    settings.debugUi.showSceneHierarchyPanel = false;
    settings.debugUi.showMaterialInspectorPanel = false;
    settings.debugUi.showTextureDebugPanel = false;
    settings.debugUi.showRenderTargetDebugPanel = false;
    settings.debugUi.showGpuTimingGraphs = false;
    settings.debugUi.showCullingStats = false;
    settings.debugUi.showExposureGraphs = false;
    settings.debugUi.selectedCsmCascade = 2;
    settings.debugUi.renderTargetPreviewExposure = 1.5f;
    settings.debugUi.renderTargetPreviewScale = 2.0f;

    return settings;
}

} // namespace

TEST_CASE("RuntimeSettings save -> load round-trips every persisted field", "[settings]")
{
    const std::filesystem::path path = makeTempSettingsPath();
    const RuntimeSettings original = makeNonDefaultSettings();

    REQUIRE(ve::saveRuntimeSettings(path, original));

    RuntimeSettings loaded;
    const ve::RuntimeSettingsLoadResult result = ve::loadRuntimeSettingsDetailed(path, loaded);
    CHECK(result.status == RuntimeSettingsLoadStatus::Loaded);

    CHECK(loaded.toneMapping.manualExposure == Catch::Approx(original.toneMapping.manualExposure));
    CHECK(loaded.toneMapping.enableAutoExposure == original.toneMapping.enableAutoExposure);
    CHECK(loaded.toneMapping.exposureMode == original.toneMapping.exposureMode);
    CHECK(loaded.toneMapping.targetLuminance == Catch::Approx(original.toneMapping.targetLuminance));
    CHECK(loaded.toneMapping.minExposure == Catch::Approx(original.toneMapping.minExposure));
    CHECK(loaded.toneMapping.maxExposure == Catch::Approx(original.toneMapping.maxExposure));
    CHECK(loaded.toneMapping.adaptationRate == Catch::Approx(original.toneMapping.adaptationRate));
    CHECK(loaded.toneMapping.histogramMinLogLuminance ==
          Catch::Approx(original.toneMapping.histogramMinLogLuminance));
    CHECK(loaded.toneMapping.histogramMaxLogLuminance ==
          Catch::Approx(original.toneMapping.histogramMaxLogLuminance));
    CHECK(loaded.toneMapping.lowPercentile == Catch::Approx(original.toneMapping.lowPercentile));
    CHECK(loaded.toneMapping.highPercentile == Catch::Approx(original.toneMapping.highPercentile));
    CHECK(loaded.toneMapping.operatorType == original.toneMapping.operatorType);

    CHECK(loaded.bloom.enabled == original.bloom.enabled);
    CHECK(loaded.bloom.useMipChain == original.bloom.useMipChain);
    CHECK(loaded.bloom.threshold == Catch::Approx(original.bloom.threshold));
    CHECK(loaded.bloom.intensity == Catch::Approx(original.bloom.intensity));
    CHECK(loaded.bloom.radius == Catch::Approx(original.bloom.radius));

    CHECK(loaded.taa.enabled == original.taa.enabled);
    CHECK(loaded.taa.jitterEnabled == original.taa.jitterEnabled);
    CHECK(loaded.taa.neighborhoodClampEnabled == original.taa.neighborhoodClampEnabled);
    CHECK(loaded.taa.reprojectionEnabled == original.taa.reprojectionEnabled);
    CHECK(loaded.taa.feedback == Catch::Approx(original.taa.feedback));

    CHECK(loaded.ssr.enabled == original.ssr.enabled);
    CHECK(loaded.ssr.maxSteps == original.ssr.maxSteps);
    CHECK(loaded.ssr.refinementSteps == original.ssr.refinementSteps);
    CHECK(loaded.ssr.maxDistance == Catch::Approx(original.ssr.maxDistance));
    CHECK(loaded.ssr.thickness == Catch::Approx(original.ssr.thickness));
    CHECK(loaded.ssr.intensity == Catch::Approx(original.ssr.intensity));
    CHECK(loaded.ssr.maxRoughness == Catch::Approx(original.ssr.maxRoughness));
    CHECK(loaded.ssr.screenEdgeFade == Catch::Approx(original.ssr.screenEdgeFade));

    CHECK(loaded.ssao.enabled == original.ssao.enabled);
    CHECK(loaded.ssao.radius == Catch::Approx(original.ssao.radius));
    CHECK(loaded.ssao.intensity == Catch::Approx(original.ssao.intensity));
    CHECK(loaded.ssao.power == Catch::Approx(original.ssao.power));
    CHECK(loaded.ssao.sliceCount == original.ssao.sliceCount);
    CHECK(loaded.ssao.stepsPerSlice == original.ssao.stepsPerSlice);
    CHECK(loaded.ssao.falloff == Catch::Approx(original.ssao.falloff));
    CHECK(loaded.ssao.thickness == Catch::Approx(original.ssao.thickness));

    CHECK(loaded.fog.enabled == original.fog.enabled);
    CHECK(loaded.fog.density == Catch::Approx(original.fog.density));
    CHECK(loaded.fog.maxDistance == Catch::Approx(original.fog.maxDistance));
    CHECK(loaded.fog.anisotropy == Catch::Approx(original.fog.anisotropy));
    CHECK(loaded.fog.heightFalloff == Catch::Approx(original.fog.heightFalloff));
    CHECK(loaded.fog.baseHeight == Catch::Approx(original.fog.baseHeight));
    CHECK(loaded.fog.ambientScale == Catch::Approx(original.fog.ambientScale));
    CHECK(loaded.fog.scatteringColor[0] == Catch::Approx(original.fog.scatteringColor[0]));
    CHECK(loaded.fog.scatteringColor[1] == Catch::Approx(original.fog.scatteringColor[1]));
    CHECK(loaded.fog.scatteringColor[2] == Catch::Approx(original.fog.scatteringColor[2]));
    CHECK(loaded.fog.lightCullThreshold == Catch::Approx(original.fog.lightCullThreshold));
    CHECK(loaded.fog.temporalEnabled == original.fog.temporalEnabled);
    CHECK(loaded.fog.temporalBlend == Catch::Approx(original.fog.temporalBlend));

    CHECK(loaded.punctualShadows.enabled == original.punctualShadows.enabled);
    CHECK(loaded.punctualShadows.gpuCasterCulling == original.punctualShadows.gpuCasterCulling);
    CHECK(loaded.punctualShadows.debugView == original.punctualShadows.debugView);

    CHECK(loaded.csm.cascadeCount == original.csm.cascadeCount);
    CHECK(loaded.csm.lambda == Catch::Approx(original.csm.lambda));
    CHECK(loaded.csm.shadowDistance == Catch::Approx(original.csm.shadowDistance));
    CHECK(loaded.csm.enableTexelSnapping == original.csm.enableTexelSnapping);
    CHECK(loaded.csm.enableCascadeDebugColors == original.csm.enableCascadeDebugColors);

    CHECK(loaded.useGpuCulling == original.useGpuCulling);
    CHECK(loaded.useGpuShadowCulling == original.useGpuShadowCulling);
    CHECK(loaded.enableGpuOcclusionCulling == original.enableGpuOcclusionCulling);
    CHECK(loaded.enableTwoPhaseOcclusion == original.enableTwoPhaseOcclusion);
    CHECK(loaded.enableAsyncCompute == original.enableAsyncCompute);
    CHECK(loaded.enableBindlessMaterialTextures == original.enableBindlessMaterialTextures);

    CHECK(loaded.debugUi.showRenderGraphPanel == original.debugUi.showRenderGraphPanel);
    CHECK(loaded.debugUi.showSceneHierarchyPanel == original.debugUi.showSceneHierarchyPanel);
    CHECK(loaded.debugUi.showMaterialInspectorPanel == original.debugUi.showMaterialInspectorPanel);
    CHECK(loaded.debugUi.showTextureDebugPanel == original.debugUi.showTextureDebugPanel);
    CHECK(loaded.debugUi.showRenderTargetDebugPanel == original.debugUi.showRenderTargetDebugPanel);
    CHECK(loaded.debugUi.showGpuTimingGraphs == original.debugUi.showGpuTimingGraphs);
    CHECK(loaded.debugUi.showCullingStats == original.debugUi.showCullingStats);
    CHECK(loaded.debugUi.showExposureGraphs == original.debugUi.showExposureGraphs);
    CHECK(loaded.debugUi.selectedCsmCascade == original.debugUi.selectedCsmCascade);
    CHECK(loaded.debugUi.renderTargetPreviewExposure ==
          Catch::Approx(original.debugUi.renderTargetPreviewExposure));
    CHECK(loaded.debugUi.renderTargetPreviewScale ==
          Catch::Approx(original.debugUi.renderTargetPreviewScale));

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

TEST_CASE("Loading a missing settings file reports Missing and yields defaults", "[settings]")
{
    const std::filesystem::path path = makeTempSettingsPath();
    std::error_code removeError;
    std::filesystem::remove(path, removeError); // ensure it does not exist

    RuntimeSettings loaded = makeNonDefaultSettings(); // start dirty to prove it gets reset
    const ve::RuntimeSettingsLoadResult result = ve::loadRuntimeSettingsDetailed(path, loaded);

    CHECK(result.status == RuntimeSettingsLoadStatus::Missing);
    CHECK(loaded.bloom.enabled == RuntimeSettings{}.bloom.enabled);
    CHECK(loaded.csm.cascadeCount == RuntimeSettings{}.csm.cascadeCount);
}

TEST_CASE("Loading malformed JSON reports Malformed and yields defaults", "[settings]")
{
    const std::filesystem::path path = makeTempSettingsPath();
    {
        std::ofstream output(path);
        output << "{ this is not valid json ]";
    }

    RuntimeSettings loaded;
    const ve::RuntimeSettingsLoadResult result = ve::loadRuntimeSettingsDetailed(path, loaded);

    CHECK(result.status == RuntimeSettingsLoadStatus::Malformed);
    CHECK(loaded.bloom.threshold == Catch::Approx(RuntimeSettings{}.bloom.threshold));

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

// The example file is the only documentation of the settings schema, and it had
// silently fallen behind twice -- it was missing the `gi` and `lod` sections
// entirely before GTAO, fog, and punctual shadows were added. Loading it over
// defaults and demanding nothing changes catches both a missing section and a
// value that drifted from its default, which reading the file cannot.
TEST_CASE("The example settings file documents the current defaults", "[settings]")
{
    const std::filesystem::path examplePath =
        std::filesystem::path(VULKAN_ENGINE_CONFIG_DIR) / "runtime_settings.example.json";
    REQUIRE(std::filesystem::exists(examplePath));

    RuntimeSettings loaded;
    const ve::RuntimeSettingsLoadResult result = ve::loadRuntimeSettingsDetailed(examplePath, loaded);
    REQUIRE(result.status == RuntimeSettingsLoadStatus::Loaded);

    const RuntimeSettings defaults;

    CHECK(loaded.ssao.enabled == defaults.ssao.enabled);
    CHECK(loaded.ssao.radius == Catch::Approx(defaults.ssao.radius));
    CHECK(loaded.ssao.intensity == Catch::Approx(defaults.ssao.intensity));
    CHECK(loaded.ssao.power == Catch::Approx(defaults.ssao.power));
    CHECK(loaded.ssao.sliceCount == defaults.ssao.sliceCount);
    CHECK(loaded.ssao.stepsPerSlice == defaults.ssao.stepsPerSlice);
    CHECK(loaded.ssao.falloff == Catch::Approx(defaults.ssao.falloff));
    CHECK(loaded.ssao.thickness == Catch::Approx(defaults.ssao.thickness));

    CHECK(loaded.fog.enabled == defaults.fog.enabled);
    CHECK(loaded.fog.density == Catch::Approx(defaults.fog.density));
    CHECK(loaded.fog.maxDistance == Catch::Approx(defaults.fog.maxDistance));
    CHECK(loaded.fog.anisotropy == Catch::Approx(defaults.fog.anisotropy));
    CHECK(loaded.fog.heightFalloff == Catch::Approx(defaults.fog.heightFalloff));
    CHECK(loaded.fog.baseHeight == Catch::Approx(defaults.fog.baseHeight));
    CHECK(loaded.fog.ambientScale == Catch::Approx(defaults.fog.ambientScale));
    CHECK(loaded.fog.scatteringColor[0] == Catch::Approx(defaults.fog.scatteringColor[0]));
    CHECK(loaded.fog.scatteringColor[1] == Catch::Approx(defaults.fog.scatteringColor[1]));
    CHECK(loaded.fog.scatteringColor[2] == Catch::Approx(defaults.fog.scatteringColor[2]));
    CHECK(loaded.fog.lightCullThreshold == Catch::Approx(defaults.fog.lightCullThreshold));
    CHECK(loaded.fog.temporalEnabled == defaults.fog.temporalEnabled);
    CHECK(loaded.fog.temporalBlend == Catch::Approx(defaults.fog.temporalBlend));

    CHECK(loaded.punctualShadows.enabled == defaults.punctualShadows.enabled);
    CHECK(loaded.punctualShadows.gpuCasterCulling == defaults.punctualShadows.gpuCasterCulling);
    CHECK(loaded.punctualShadows.debugView == defaults.punctualShadows.debugView);

    CHECK(loaded.lod.enabled == defaults.lod.enabled);
    CHECK(loaded.lod.referenceRadiusPixels == Catch::Approx(defaults.lod.referenceRadiusPixels));
    CHECK(loaded.lod.bias == Catch::Approx(defaults.lod.bias));
    CHECK(loaded.lod.shadowBias == Catch::Approx(defaults.lod.shadowBias));
    CHECK(loaded.lod.forcedLod == defaults.lod.forcedLod);
    CHECK(loaded.lod.debugHeatmap == defaults.lod.debugHeatmap);

    CHECK(loaded.gi.enabled == defaults.gi.enabled);
    CHECK(loaded.gi.probesPerFrame == defaults.gi.probesPerFrame);
    CHECK(loaded.gi.previewGain == Catch::Approx(defaults.gi.previewGain));
    CHECK(loaded.gi.intensity == Catch::Approx(defaults.gi.intensity));
    CHECK(loaded.gi.surfaceBias == Catch::Approx(defaults.gi.surfaceBias));
    CHECK(loaded.gi.hysteresis == Catch::Approx(defaults.gi.hysteresis));
    CHECK(loaded.gi.bounceWeight == Catch::Approx(defaults.gi.bounceWeight));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        CHECK(loaded.gi.gridOrigin[axis] == Catch::Approx(defaults.gi.gridOrigin[axis]));
        CHECK(loaded.gi.gridSpacing[axis] == Catch::Approx(defaults.gi.gridSpacing[axis]));
    }
}
