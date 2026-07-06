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
