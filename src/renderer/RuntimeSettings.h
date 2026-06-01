#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace ve {

struct CsmSettings {
    uint32_t cascadeCount = 4;
    float lambda = 0.5f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    float shadowDistance = 40.0f;
    bool enableTexelSnapping = true;
    bool enableCascadeDebugColors = false;
    float depthBiasConstant = 0.002f;
    float depthBiasSlope = 0.005f;
};

struct ToneMappingSettings {
    float manualExposure = 1.0f;
    bool enableAutoExposure = true;

    int exposureMode = 2;
    // 0 = manual
    // 1 = log-average luminance
    // 2 = histogram percentile

    float targetLuminance = 0.18f;
    float minExposure = 0.1f;
    float maxExposure = 8.0f;
    float adaptationRate = 1.5f;

    float histogramMinLogLuminance = -10.0f;
    float histogramMaxLogLuminance = 4.0f;
    float lowPercentile = 0.05f;
    float highPercentile = 0.95f;

    int operatorType = 0;
    // 0 = Reinhard
    // 1 = ACES fitted approximation
};

struct BloomSettings {
    bool enabled = true;
    bool useMipChain = true;
    float threshold = 1.0f;
    float intensity = 0.1f;
    float radius = 1.0f;
};

struct TaaSettings {
    bool enabled = false;
    bool jitterEnabled = true;
    bool neighborhoodClampEnabled = true;
    float feedback = 0.88f;
};

struct DebugUiSettings {
    bool showRenderGraphPanel = true;
    bool showSceneHierarchyPanel = true;
    bool showMaterialInspectorPanel = true;
    bool showTextureDebugPanel = true;
    bool showRenderTargetDebugPanel = true;
    bool showGpuTimingGraphs = true;
    bool showCullingStats = true;
    bool showExposureGraphs = true;
    uint32_t selectedCsmCascade = 0;
    float renderTargetPreviewExposure = 1.0f;
    float renderTargetPreviewScale = 1.0f;
};

struct RuntimeSettings {
    ToneMappingSettings toneMapping;
    BloomSettings bloom;
    TaaSettings taa;
    CsmSettings csm;
    DebugUiSettings debugUi;
    bool useGpuCulling = true;
    bool useGpuShadowCulling = true;
    bool enableGpuOcclusionCulling = false;
    bool enableBindlessMaterialTextures = true;
};

enum class RuntimeSettingsLoadStatus {
    Loaded,
    Missing,
    Malformed,
    IoError
};

struct RuntimeSettingsLoadResult {
    RuntimeSettingsLoadStatus status = RuntimeSettingsLoadStatus::Loaded;
    std::string message;
};

struct RuntimeSettingsSaveResult {
    bool saved = false;
    std::string message;
};

bool loadRuntimeSettings(const std::filesystem::path& path, RuntimeSettings& outSettings);
RuntimeSettingsLoadResult loadRuntimeSettingsDetailed(const std::filesystem::path& path,
                                                      RuntimeSettings& outSettings);

bool saveRuntimeSettings(const std::filesystem::path& path, const RuntimeSettings& settings);
RuntimeSettingsSaveResult saveRuntimeSettingsDetailed(const std::filesystem::path& path,
                                                      const RuntimeSettings& settings);

} // namespace ve
