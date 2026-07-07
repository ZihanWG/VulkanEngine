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

// Auto-exposure selection, mirroring ToneMappingSettings::exposureMode. Lives
// with the settings (not in a renderer-internal header) so the GPU-independent
// clamping logic and its unit tests can use it.
enum class ExposureMode : int {
    Manual = 0,
    LogAverage = 1,
    Histogram = 2,
};

// Floor for average-luminance targets to avoid divide-by-zero in exposure math.
inline constexpr float kMinAverageLuminance = 0.0001f;

// Map an arbitrary stored int to a valid ExposureMode (defaults to Histogram).
[[nodiscard]] ExposureMode exposureModeValue(int exposureMode);

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
    // Reprojects the history sample along the velocity buffer (camera + object
    // motion). Off falls back to same-UV history sampling for A/B comparison.
    bool reprojectionEnabled = true;
    float feedback = 0.88f;
};

// Screen-space ambient occlusion computed in the composite pass from the main
// depth buffer. Disabled by default; the toggle is only honoured when the depth
// image supports sampling (Renderer::ssaoAvailable_). Lives with the other
// post-process settings structs so PostProcessStack and Renderer share the type
// without a circular include.
struct SsaoSettings {
    bool enabled = false;
    float radius = 0.5f;
    float bias = 0.025f;
    float intensity = 1.0f;
    float power = 2.0f;
    int sampleCount = 16;
};

struct DebugUiSettings {
    // Master toggle: when false the debug window shows only the common post-process
    // knobs (tone mapping, bloom, SSAO, TAA, exposure); when true it reveals the
    // scene, GPU, diagnostics sections and the side panels.
    bool advancedMode = false;
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

// Screen-space reflections: a view-space linear march with binary refinement
// against the main depth buffer, additively blended into scene color before
// TAA. Requires a samplable main depth image (same gate as SSAO).
struct SsrSettings {
    bool enabled = true;
    int maxSteps = 48;
    int refinementSteps = 5;
    float maxDistance = 30.0f;
    float thickness = 0.35f;
    float intensity = 1.0f;
    // Surfaces rougher than this trace no reflections (mirror-to-glossy fade).
    float maxRoughness = 0.6f;
    float screenEdgeFade = 0.1f;
};

struct RuntimeSettings {
    ToneMappingSettings toneMapping;
    BloomSettings bloom;
    TaaSettings taa;
    SsrSettings ssr;
    CsmSettings csm;
    DebugUiSettings debugUi;
    bool useGpuCulling = true;
    bool useGpuShadowCulling = true;
    // Hi-Z occlusion defaults ON now that two-phase re-testing removes the
    // disocclusion false negatives that used to make it opt-in.
    bool enableGpuOcclusionCulling = true;
    // Phase-1 (previous-frame pyramid) cull + phase-2 candidate re-test against a
    // mid-frame rebuild. Off falls back to the conservative single-phase test
    // that only runs while the camera holds still.
    bool enableTwoPhaseOcclusion = true;
    // Runs ClusterBuild/LightCull on the async compute queue, overlapping the
    // shadow passes. Ignored (graphics-queue fallback) when the device exposes
    // no async-capable queue.
    bool enableAsyncCompute = true;
    bool enableBindlessMaterialTextures = true;
};

// Clamp the GPU-independent runtime settings into valid ranges in place
// (tone-mapping, bloom, TAA, cascade, and debug-UI preview fields). Renderer
// state that is not part of these structs (e.g. GPU occlusion tuning) is clamped
// separately by the caller. Pure and unit-tested.
void clampRuntimeSettings(ToneMappingSettings& toneMapping,
                          BloomSettings& bloom,
                          TaaSettings& taa,
                          SsrSettings& ssr,
                          CsmSettings& csm,
                          DebugUiSettings& debugUi);

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
