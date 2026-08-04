#include "renderer/RuntimeSettings.h"

#include "renderer/MeshLod.h"

#include "core/Logger.h"
#include "renderer/CascadeMath.h"
#include "renderer/IrradianceProbes.h"

#include <json.hpp>

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ve {

ExposureMode exposureModeValue(int exposureMode)
{
    if (exposureMode == static_cast<int>(ExposureMode::Manual)) {
        return ExposureMode::Manual;
    }
    if (exposureMode == static_cast<int>(ExposureMode::LogAverage)) {
        return ExposureMode::LogAverage;
    }

    return ExposureMode::Histogram;
}

void clampRuntimeSettings(ToneMappingSettings& toneMapping,
                          BloomSettings& bloom,
                          TaaSettings& taa,
                          SsrSettings& ssr,
                          CsmSettings& csm,
                          LodSettings& lod,
                          GiSettings& gi,
                          DebugUiSettings& debugUi)
{
    toneMapping.operatorType = std::clamp(toneMapping.operatorType, 0, 1);
    if (!toneMapping.enableAutoExposure) {
        toneMapping.exposureMode = static_cast<int>(ExposureMode::Manual);
    } else {
        toneMapping.exposureMode = static_cast<int>(exposureModeValue(toneMapping.exposureMode));
    }
    toneMapping.manualExposure = std::max(toneMapping.manualExposure, 0.0f);
    toneMapping.targetLuminance = std::max(toneMapping.targetLuminance, kMinAverageLuminance);
    toneMapping.minExposure = std::max(toneMapping.minExposure, 0.0f);
    toneMapping.maxExposure = std::max(toneMapping.maxExposure, toneMapping.minExposure);
    toneMapping.adaptationRate = std::max(toneMapping.adaptationRate, 0.0f);

    toneMapping.lowPercentile = std::clamp(toneMapping.lowPercentile, 0.0f, 1.0f);
    toneMapping.highPercentile = std::clamp(toneMapping.highPercentile, 0.0f, 1.0f);
    if (toneMapping.highPercentile <= toneMapping.lowPercentile) {
        toneMapping.highPercentile = std::min(1.0f, toneMapping.lowPercentile + 0.01f);
        toneMapping.lowPercentile = std::min(toneMapping.lowPercentile, toneMapping.highPercentile - 0.01f);
    }

    bloom.threshold = std::max(bloom.threshold, 0.0f);
    bloom.intensity = std::max(bloom.intensity, 0.0f);
    bloom.radius = std::clamp(bloom.radius, 0.25f, 4.0f);

    taa.feedback = std::clamp(taa.feedback, 0.0f, 0.98f);

    ssr.maxSteps = std::clamp(ssr.maxSteps, 8, 128);
    ssr.refinementSteps = std::clamp(ssr.refinementSteps, 0, 8);
    ssr.maxDistance = std::clamp(ssr.maxDistance, 1.0f, 200.0f);
    ssr.thickness = std::clamp(ssr.thickness, 0.01f, 2.0f);
    ssr.intensity = std::clamp(ssr.intensity, 0.0f, 4.0f);
    ssr.maxRoughness = std::clamp(ssr.maxRoughness, 0.05f, 1.0f);
    ssr.screenEdgeFade = std::clamp(ssr.screenEdgeFade, 0.01f, 0.49f);

    // A reference radius at or below 1px would make every object select the
    // coarsest level; the bias range is what a user can meaningfully explore
    // before everything pins to one end. forcedLod is clamped against the chain
    // cap, and -1 stays the "select by distance" sentinel.
    lod.referenceRadiusPixels = std::clamp(lod.referenceRadiusPixels, 8.0f, 4096.0f);
    lod.bias = std::clamp(lod.bias, -4.0f, 4.0f);
    lod.shadowBias = std::clamp(lod.shadowBias, -4.0f, 4.0f);
    lod.forcedLod = std::clamp(lod.forcedLod, -1, static_cast<int>(renderer::kMaxMeshLods) - 1);

    // Probe spacing is a divisor in the grid-space lookup, so a zero or negative
    // value would fold the whole volume onto one probe. The upper bound keeps the
    // grid from spanning further than the depth atlas can describe: a probe
    // cannot report a distance past kProbeMaxDistance, so spacing beyond that
    // range makes visibility meaningless rather than merely coarse.
    for (float& spacing : gi.gridSpacing) {
        spacing = std::clamp(spacing, 0.05f, renderer::kProbeMaxDistance);
    }
    // Zero is a legal setting: it pauses capture without losing the cursor, so
    // resuming continues round-robin rather than restarting. The upper bound is
    // what the capture atlas is sized for.
    gi.probesPerFrame = std::clamp(gi.probesPerFrame, 0, static_cast<int>(renderer::kMaxProbesPerFrame));

    csm.cascadeCount = std::clamp(csm.cascadeCount, 1U, renderer::kMaxShadowCascades);
    csm.lambda = std::clamp(csm.lambda, 0.0f, 1.0f);
    csm.shadowDistance = std::clamp(csm.shadowDistance, csm.nearPlane + 0.001f, csm.farPlane);

    debugUi.renderTargetPreviewExposure = std::clamp(debugUi.renderTargetPreviewExposure, 0.05f, 8.0f);
    debugUi.renderTargetPreviewScale = std::clamp(debugUi.renderTargetPreviewScale, 0.25f, 2.0f);
    // activeCascadeCount() == clamp(cascadeCount, 1, max); after the clamp above
    // that is just csm.cascadeCount, which is >= 1 so the subtraction is safe.
    debugUi.selectedCsmCascade = std::min(debugUi.selectedCsmCascade, csm.cascadeCount - 1U);
}

namespace {

using Json = nlohmann::json;

std::string pathString(const std::filesystem::path& path)
{
    return path.string();
}

std::string lowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

const Json* objectMember(const Json& object, const char* name)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return nullptr;
    }
    if (!it->is_object()) {
        throw std::runtime_error(std::string("Expected object member '") + name + "'.");
    }
    return &(*it);
}

void readBool(const Json& object, const char* name, bool& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return;
    }
    if (!it->is_boolean()) {
        throw std::runtime_error(std::string("Expected boolean member '") + name + "'.");
    }
    value = it->get<bool>();
}

void readFloat(const Json& object, const char* name, float& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return;
    }
    if (!it->is_number()) {
        throw std::runtime_error(std::string("Expected numeric member '") + name + "'.");
    }
    value = it->get<float>();
}

void readInt(const Json& object, const char* name, int& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return;
    }
    if (!it->is_number_integer()) {
        throw std::runtime_error(std::string("Expected integer member '") + name + "'.");
    }
    value = it->get<int>();
}

void readUint32(const Json& object, const char* name, uint32_t& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return;
    }
    if (!it->is_number_integer()) {
        throw std::runtime_error(std::string("Expected unsigned integer member '") + name + "'.");
    }

    const int parsed = it->get<int>();
    if (parsed < 0) {
        throw std::runtime_error(std::string("Expected non-negative integer member '") + name + "'.");
    }
    value = static_cast<uint32_t>(parsed);
}

int exposureModeFromString(const std::string& value)
{
    const std::string normalized = lowerAscii(value);
    if (normalized == "manual") {
        return 0;
    }
    if (normalized == "logaverage" || normalized == "log-average" || normalized == "log_average") {
        return 1;
    }
    if (normalized == "histogram" || normalized == "histogrampercentile" ||
        normalized == "histogram-percentile" || normalized == "histogram_percentile") {
        return 2;
    }

    throw std::runtime_error("Unknown exposureMode string '" + value + "'.");
}

const char* exposureModeName(int exposureMode)
{
    if (exposureMode == 0) {
        return "Manual";
    }
    if (exposureMode == 1) {
        return "LogAverage";
    }
    return "Histogram";
}

void readExposureMode(const Json& object, ToneMappingSettings& settings)
{
    const auto it = object.find("exposureMode");
    if (it == object.end()) {
        return;
    }
    if (it->is_number_integer()) {
        settings.exposureMode = it->get<int>();
        return;
    }
    if (it->is_string()) {
        settings.exposureMode = exposureModeFromString(it->get<std::string>());
        return;
    }

    throw std::runtime_error("Expected integer or string member 'exposureMode'.");
}

int toneMapperFromString(const std::string& value)
{
    const std::string normalized = lowerAscii(value);
    if (normalized == "reinhard") {
        return 0;
    }
    if (normalized == "aces" || normalized == "acesfitted" || normalized == "aces-fitted" ||
        normalized == "aces_fitted") {
        return 1;
    }

    throw std::runtime_error("Unknown toneMapper string '" + value + "'.");
}

const char* toneMapperName(int operatorType)
{
    return operatorType == 1 ? "ACES" : "Reinhard";
}

void readToneMapper(const Json& object, ToneMappingSettings& settings)
{
    const auto toneMapperIt = object.find("toneMapper");
    if (toneMapperIt != object.end()) {
        if (toneMapperIt->is_number_integer()) {
            settings.operatorType = toneMapperIt->get<int>();
            return;
        }
        if (toneMapperIt->is_string()) {
            settings.operatorType = toneMapperFromString(toneMapperIt->get<std::string>());
            return;
        }
        throw std::runtime_error("Expected integer or string member 'toneMapper'.");
    }

    readInt(object, "operatorType", settings.operatorType);
}

void fromJson(const Json& json, RuntimeSettings& settings)
{
    if (!json.is_object()) {
        throw std::runtime_error("Runtime settings root must be a JSON object.");
    }

    if (const Json* toneMapping = objectMember(json, "toneMapping")) {
        readFloat(*toneMapping, "manualExposure", settings.toneMapping.manualExposure);
        readBool(*toneMapping, "enableAutoExposure", settings.toneMapping.enableAutoExposure);
        readExposureMode(*toneMapping, settings.toneMapping);
        readFloat(*toneMapping, "targetLuminance", settings.toneMapping.targetLuminance);
        readFloat(*toneMapping, "minExposure", settings.toneMapping.minExposure);
        readFloat(*toneMapping, "maxExposure", settings.toneMapping.maxExposure);
        readFloat(*toneMapping, "adaptationRate", settings.toneMapping.adaptationRate);
        readFloat(*toneMapping, "histogramMinLogLuminance", settings.toneMapping.histogramMinLogLuminance);
        readFloat(*toneMapping, "histogramMaxLogLuminance", settings.toneMapping.histogramMaxLogLuminance);
        readFloat(*toneMapping, "lowPercentile", settings.toneMapping.lowPercentile);
        readFloat(*toneMapping, "highPercentile", settings.toneMapping.highPercentile);
        readToneMapper(*toneMapping, settings.toneMapping);
    }

    if (const Json* bloom = objectMember(json, "bloom")) {
        readBool(*bloom, "enabled", settings.bloom.enabled);
        readBool(*bloom, "useMipChain", settings.bloom.useMipChain);
        readFloat(*bloom, "threshold", settings.bloom.threshold);
        readFloat(*bloom, "intensity", settings.bloom.intensity);
        readFloat(*bloom, "radius", settings.bloom.radius);
    }

    if (const Json* taa = objectMember(json, "taa")) {
        readBool(*taa, "enabled", settings.taa.enabled);
        readBool(*taa, "jitterEnabled", settings.taa.jitterEnabled);
        readBool(*taa, "neighborhoodClampEnabled", settings.taa.neighborhoodClampEnabled);
        readBool(*taa, "reprojectionEnabled", settings.taa.reprojectionEnabled);
        readFloat(*taa, "feedback", settings.taa.feedback);
    }

    if (const Json* lod = objectMember(json, "lod")) {
        readBool(*lod, "enabled", settings.lod.enabled);
        readFloat(*lod, "referenceRadiusPixels", settings.lod.referenceRadiusPixels);
        readFloat(*lod, "bias", settings.lod.bias);
        readFloat(*lod, "shadowBias", settings.lod.shadowBias);
        readInt(*lod, "forcedLod", settings.lod.forcedLod);
        readBool(*lod, "debugHeatmap", settings.lod.debugHeatmap);
    }

    if (const Json* ssr = objectMember(json, "ssr")) {
        readBool(*ssr, "enabled", settings.ssr.enabled);
        readInt(*ssr, "maxSteps", settings.ssr.maxSteps);
        readInt(*ssr, "refinementSteps", settings.ssr.refinementSteps);
        readFloat(*ssr, "maxDistance", settings.ssr.maxDistance);
        readFloat(*ssr, "thickness", settings.ssr.thickness);
        readFloat(*ssr, "intensity", settings.ssr.intensity);
        readFloat(*ssr, "maxRoughness", settings.ssr.maxRoughness);
        readFloat(*ssr, "screenEdgeFade", settings.ssr.screenEdgeFade);
    }

    if (const Json* csm = objectMember(json, "csm")) {
        readUint32(*csm, "cascadeCount", settings.csm.cascadeCount);
        readFloat(*csm, "lambda", settings.csm.lambda);
        readFloat(*csm, "shadowDistance", settings.csm.shadowDistance);
        readBool(*csm, "enableTexelSnapping", settings.csm.enableTexelSnapping);
        readBool(*csm, "enableCascadeDebugColors", settings.csm.enableCascadeDebugColors);
    }

    if (const Json* gi = objectMember(json, "gi")) {
        readBool(*gi, "enabled", settings.gi.enabled);
        readBool(*gi, "debugPattern", settings.gi.debugPattern);
        readInt(*gi, "probesPerFrame", settings.gi.probesPerFrame);
        readFloat(*gi, "gridOriginX", settings.gi.gridOrigin[0]);
        readFloat(*gi, "gridOriginY", settings.gi.gridOrigin[1]);
        readFloat(*gi, "gridOriginZ", settings.gi.gridOrigin[2]);
        readFloat(*gi, "gridSpacingX", settings.gi.gridSpacing[0]);
        readFloat(*gi, "gridSpacingY", settings.gi.gridSpacing[1]);
        readFloat(*gi, "gridSpacingZ", settings.gi.gridSpacing[2]);
    }

    if (const Json* renderer = objectMember(json, "renderer")) {
        readBool(*renderer, "useGpuCulling", settings.useGpuCulling);
        readBool(*renderer, "useGpuShadowCulling", settings.useGpuShadowCulling);
        readBool(*renderer, "enableGpuOcclusionCulling", settings.enableGpuOcclusionCulling);
        readBool(*renderer, "enableTwoPhaseOcclusion", settings.enableTwoPhaseOcclusion);
        readBool(*renderer, "enableAsyncCompute", settings.enableAsyncCompute);
        readBool(*renderer, "enableBindlessMaterialTextures", settings.enableBindlessMaterialTextures);
    }

    if (const Json* debugUi = objectMember(json, "debugUi")) {
        readBool(*debugUi, "advancedMode", settings.debugUi.advancedMode);
        readBool(*debugUi, "showRenderGraphPanel", settings.debugUi.showRenderGraphPanel);
        readBool(*debugUi, "showSceneHierarchyPanel", settings.debugUi.showSceneHierarchyPanel);
        readBool(*debugUi, "showMaterialInspectorPanel", settings.debugUi.showMaterialInspectorPanel);
        readBool(*debugUi, "showTextureDebugPanel", settings.debugUi.showTextureDebugPanel);
        readBool(*debugUi, "showRenderTargetDebugPanel", settings.debugUi.showRenderTargetDebugPanel);
        readBool(*debugUi, "showIrradianceProbePanel", settings.debugUi.showIrradianceProbePanel);
        readBool(*debugUi, "showGpuTimingGraphs", settings.debugUi.showGpuTimingGraphs);
        readBool(*debugUi, "showCullingStats", settings.debugUi.showCullingStats);
        readBool(*debugUi, "showExposureGraphs", settings.debugUi.showExposureGraphs);
        readUint32(*debugUi, "selectedCsmCascade", settings.debugUi.selectedCsmCascade);
        readFloat(*debugUi, "renderTargetPreviewExposure", settings.debugUi.renderTargetPreviewExposure);
        readFloat(*debugUi, "renderTargetPreviewScale", settings.debugUi.renderTargetPreviewScale);
    }
}

Json toJson(const RuntimeSettings& settings)
{
    return Json{
        {"schemaVersion", 1},
        {"toneMapping",
         Json{{"manualExposure", settings.toneMapping.manualExposure},
              {"enableAutoExposure", settings.toneMapping.enableAutoExposure},
              {"exposureMode",
               exposureModeName(settings.toneMapping.enableAutoExposure ? settings.toneMapping.exposureMode : 0)},
              {"targetLuminance", settings.toneMapping.targetLuminance},
              {"minExposure", settings.toneMapping.minExposure},
              {"maxExposure", settings.toneMapping.maxExposure},
              {"adaptationRate", settings.toneMapping.adaptationRate},
              {"histogramMinLogLuminance", settings.toneMapping.histogramMinLogLuminance},
              {"histogramMaxLogLuminance", settings.toneMapping.histogramMaxLogLuminance},
              {"lowPercentile", settings.toneMapping.lowPercentile},
              {"highPercentile", settings.toneMapping.highPercentile},
              {"toneMapper", toneMapperName(settings.toneMapping.operatorType)}}},
        {"bloom",
         Json{{"enabled", settings.bloom.enabled},
              {"useMipChain", settings.bloom.useMipChain},
              {"threshold", settings.bloom.threshold},
              {"intensity", settings.bloom.intensity},
              {"radius", settings.bloom.radius}}},
        {"taa",
         Json{{"enabled", settings.taa.enabled},
              {"jitterEnabled", settings.taa.jitterEnabled},
              {"neighborhoodClampEnabled", settings.taa.neighborhoodClampEnabled},
              {"reprojectionEnabled", settings.taa.reprojectionEnabled},
              {"feedback", settings.taa.feedback}}},
        {"lod",
         Json{{"enabled", settings.lod.enabled},
              {"referenceRadiusPixels", settings.lod.referenceRadiusPixels},
              {"bias", settings.lod.bias},
              {"shadowBias", settings.lod.shadowBias},
              {"forcedLod", settings.lod.forcedLod},
              {"debugHeatmap", settings.lod.debugHeatmap}}},
        {"ssr",
         Json{{"enabled", settings.ssr.enabled},
              {"maxSteps", settings.ssr.maxSteps},
              {"refinementSteps", settings.ssr.refinementSteps},
              {"maxDistance", settings.ssr.maxDistance},
              {"thickness", settings.ssr.thickness},
              {"intensity", settings.ssr.intensity},
              {"maxRoughness", settings.ssr.maxRoughness},
              {"screenEdgeFade", settings.ssr.screenEdgeFade}}},
        {"csm",
         Json{{"cascadeCount", settings.csm.cascadeCount},
              {"lambda", settings.csm.lambda},
              {"shadowDistance", settings.csm.shadowDistance},
              {"enableTexelSnapping", settings.csm.enableTexelSnapping},
              {"enableCascadeDebugColors", settings.csm.enableCascadeDebugColors}}},
        {"gi",
         Json{{"enabled", settings.gi.enabled},
              {"debugPattern", settings.gi.debugPattern},
              {"probesPerFrame", settings.gi.probesPerFrame},
              {"gridOriginX", settings.gi.gridOrigin[0]},
              {"gridOriginY", settings.gi.gridOrigin[1]},
              {"gridOriginZ", settings.gi.gridOrigin[2]},
              {"gridSpacingX", settings.gi.gridSpacing[0]},
              {"gridSpacingY", settings.gi.gridSpacing[1]},
              {"gridSpacingZ", settings.gi.gridSpacing[2]}}},
        {"renderer",
         Json{{"useGpuCulling", settings.useGpuCulling},
              {"useGpuShadowCulling", settings.useGpuShadowCulling},
              {"enableGpuOcclusionCulling", settings.enableGpuOcclusionCulling},
              {"enableTwoPhaseOcclusion", settings.enableTwoPhaseOcclusion},
              {"enableAsyncCompute", settings.enableAsyncCompute},
              {"enableBindlessMaterialTextures", settings.enableBindlessMaterialTextures}}},
        {"debugUi",
         Json{{"advancedMode", settings.debugUi.advancedMode},
              {"showRenderGraphPanel", settings.debugUi.showRenderGraphPanel},
              {"showSceneHierarchyPanel", settings.debugUi.showSceneHierarchyPanel},
              {"showMaterialInspectorPanel", settings.debugUi.showMaterialInspectorPanel},
              {"showTextureDebugPanel", settings.debugUi.showTextureDebugPanel},
              {"showRenderTargetDebugPanel", settings.debugUi.showRenderTargetDebugPanel},
              {"showIrradianceProbePanel", settings.debugUi.showIrradianceProbePanel},
              {"showGpuTimingGraphs", settings.debugUi.showGpuTimingGraphs},
              {"showCullingStats", settings.debugUi.showCullingStats},
              {"showExposureGraphs", settings.debugUi.showExposureGraphs},
              {"selectedCsmCascade", settings.debugUi.selectedCsmCascade},
              {"renderTargetPreviewExposure", settings.debugUi.renderTargetPreviewExposure},
              {"renderTargetPreviewScale", settings.debugUi.renderTargetPreviewScale}}}};
}

} // namespace

bool loadRuntimeSettings(const std::filesystem::path& path, RuntimeSettings& outSettings)
{
    const RuntimeSettingsLoadResult result = loadRuntimeSettingsDetailed(path, outSettings);
    return result.status == RuntimeSettingsLoadStatus::Loaded || result.status == RuntimeSettingsLoadStatus::Missing;
}

RuntimeSettingsLoadResult loadRuntimeSettingsDetailed(const std::filesystem::path& path, RuntimeSettings& outSettings)
{
    outSettings = RuntimeSettings{};

    std::error_code existsError;
    if (!std::filesystem::exists(path, existsError)) {
        if (existsError) {
            const std::string message =
                "Runtime settings file could not be checked at " + pathString(path) + ": " + existsError.message() +
                ". Using defaults.";
            Logger::warn(message);
            return {RuntimeSettingsLoadStatus::IoError, message};
        }

        const std::string message = "Runtime settings file missing at " + pathString(path) + "; using defaults.";
        Logger::info(message);
        return {RuntimeSettingsLoadStatus::Missing, message};
    }

    try {
        std::ifstream input(path);
        if (!input) {
            const std::string message = "Runtime settings file could not be opened at " + pathString(path) +
                                        "; using defaults.";
            Logger::warn(message);
            return {RuntimeSettingsLoadStatus::IoError, message};
        }

        const Json json = Json::parse(input);
        fromJson(json, outSettings);

        const std::string message = "Runtime settings loaded from " + pathString(path) + ".";
        Logger::info(message);
        return {RuntimeSettingsLoadStatus::Loaded, message};
    } catch (const std::exception& error) {
        outSettings = RuntimeSettings{};
        const std::string message = "Runtime settings file at " + pathString(path) +
                                    " is malformed; using defaults: " + error.what();
        Logger::warn(message);
        return {RuntimeSettingsLoadStatus::Malformed, message};
    }
}

bool saveRuntimeSettings(const std::filesystem::path& path, const RuntimeSettings& settings)
{
    return saveRuntimeSettingsDetailed(path, settings).saved;
}

RuntimeSettingsSaveResult saveRuntimeSettingsDetailed(const std::filesystem::path& path,
                                                      const RuntimeSettings& settings)
{
    try {
        const std::filesystem::path parentPath = path.parent_path();
        if (!parentPath.empty()) {
            std::error_code createError;
            std::filesystem::create_directories(parentPath, createError);
            if (createError) {
                const std::string message = "Could not create runtime settings directory " +
                                            pathString(parentPath) + ": " + createError.message();
                Logger::warn(message);
                return {false, message};
            }
        }

        std::ofstream output(path);
        if (!output) {
            const std::string message = "Could not open runtime settings file for writing at " + pathString(path) + ".";
            Logger::warn(message);
            return {false, message};
        }

        output << toJson(settings).dump(4) << '\n';
        if (!output) {
            const std::string message = "Failed while writing runtime settings file at " + pathString(path) + ".";
            Logger::warn(message);
            return {false, message};
        }

        const std::string message = "Runtime settings saved to " + pathString(path) + ".";
        Logger::info(message);
        return {true, message};
    } catch (const std::exception& error) {
        const std::string message = "Runtime settings save failed for " + pathString(path) + ": " + error.what();
        Logger::warn(message);
        return {false, message};
    }
}

} // namespace ve
