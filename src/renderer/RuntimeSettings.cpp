#include "renderer/RuntimeSettings.h"

#include "core/Logger.h"

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

    if (const Json* csm = objectMember(json, "csm")) {
        readUint32(*csm, "cascadeCount", settings.csm.cascadeCount);
        readFloat(*csm, "lambda", settings.csm.lambda);
        readFloat(*csm, "shadowDistance", settings.csm.shadowDistance);
        readBool(*csm, "enableTexelSnapping", settings.csm.enableTexelSnapping);
        readBool(*csm, "enableCascadeDebugColors", settings.csm.enableCascadeDebugColors);
    }

    if (const Json* renderer = objectMember(json, "renderer")) {
        readBool(*renderer, "useGpuCulling", settings.useGpuCulling);
        readBool(*renderer, "useGpuShadowCulling", settings.useGpuShadowCulling);
        readBool(*renderer, "enableGpuOcclusionCulling", settings.enableGpuOcclusionCulling);
        readBool(*renderer, "enableBindlessMaterialTextures", settings.enableBindlessMaterialTextures);
    }

    if (const Json* debugUi = objectMember(json, "debugUi")) {
        readBool(*debugUi, "showRenderGraphPanel", settings.debugUi.showRenderGraphPanel);
        readBool(*debugUi, "showSceneHierarchyPanel", settings.debugUi.showSceneHierarchyPanel);
        readBool(*debugUi, "showMaterialInspectorPanel", settings.debugUi.showMaterialInspectorPanel);
        readBool(*debugUi, "showTextureDebugPanel", settings.debugUi.showTextureDebugPanel);
        readBool(*debugUi, "showRenderTargetDebugPanel", settings.debugUi.showRenderTargetDebugPanel);
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
              {"exposureMode", exposureModeName(settings.toneMapping.enableAutoExposure
                                                    ? settings.toneMapping.exposureMode
                                                    : 0)},
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
        {"csm",
         Json{{"cascadeCount", settings.csm.cascadeCount},
              {"lambda", settings.csm.lambda},
              {"shadowDistance", settings.csm.shadowDistance},
              {"enableTexelSnapping", settings.csm.enableTexelSnapping},
              {"enableCascadeDebugColors", settings.csm.enableCascadeDebugColors}}},
        {"renderer",
         Json{{"useGpuCulling", settings.useGpuCulling},
              {"useGpuShadowCulling", settings.useGpuShadowCulling},
              {"enableGpuOcclusionCulling", settings.enableGpuOcclusionCulling},
              {"enableBindlessMaterialTextures", settings.enableBindlessMaterialTextures}}},
        {"debugUi",
         Json{{"showRenderGraphPanel", settings.debugUi.showRenderGraphPanel},
              {"showSceneHierarchyPanel", settings.debugUi.showSceneHierarchyPanel},
              {"showMaterialInspectorPanel", settings.debugUi.showMaterialInspectorPanel},
              {"showTextureDebugPanel", settings.debugUi.showTextureDebugPanel},
              {"showRenderTargetDebugPanel", settings.debugUi.showRenderTargetDebugPanel},
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
