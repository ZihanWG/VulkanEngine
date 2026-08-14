#include "renderer/RuntimeSettings.h"

#include "renderer/MeshLod.h"

#include "core/Logger.h"
#include "renderer/CascadeMath.h"
#include "renderer/DynamicResolution.h"
#include "renderer/IrradianceProbes.h"
#include "renderer/RenderScale.h"
#include "renderer/VolumetricFog.h"

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

void clampRuntimeSettings(RenderScaleSettings& renderScale,
                          DynamicResolutionSettings& dynamicResolution,
                          ToneMappingSettings& toneMapping,
                          BloomSettings& bloom,
                          TaaSettings& taa,
                          SsrSettings& ssr,
                          SsaoSettings& ssao,
                          FogSettings& fog,
                          CsmSettings& csm,
                          LodSettings& lod,
                          GiSettings& gi,
                          DebugUiSettings& debugUi)
{
    renderScale.scale = renderer::clampRenderScale(renderScale.scale);
    // Past 1.0 the anti-ringing clamp is doing all the work and the result stops
    // changing, so the range ends where the effect does.
    renderScale.sharpness = std::clamp(renderScale.sharpness, 0.0f, 1.0f);

    // The bounds are clamped but deliberately not ordered: the controller treats
    // them as an unordered pair, so a hand-edited file with them swapped still
    // behaves, and "fixing" the order here would silently rewrite the user's file
    // on the next save.
    dynamicResolution.minScale = renderer::clampRenderScale(dynamicResolution.minScale);
    dynamicResolution.maxScale = renderer::clampRenderScale(dynamicResolution.maxScale);
    // Upper bound is 10 fps: past that the target stops being a frame budget.
    dynamicResolution.targetFrameMs =
        std::clamp(dynamicResolution.targetFrameMs, renderer::kDynamicResolutionMinTargetMs, 100.0f);

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
    // Below about a quarter of a standard deviation the box is narrower than the
    // neighbourhood's own noise and rejects everything; past three it contains
    // the whole neighbourhood and rejects nothing.
    taa.varianceGamma = std::clamp(taa.varianceGamma, 0.25f, 3.0f);

    ssr.maxSteps = std::clamp(ssr.maxSteps, 8, 128);
    ssr.refinementSteps = std::clamp(ssr.refinementSteps, 0, 8);
    ssr.maxDistance = std::clamp(ssr.maxDistance, 1.0f, 200.0f);
    ssr.thickness = std::clamp(ssr.thickness, 0.01f, 2.0f);
    ssr.intensity = std::clamp(ssr.intensity, 0.0f, 4.0f);
    ssr.maxRoughness = std::clamp(ssr.maxRoughness, 0.05f, 1.0f);
    ssr.screenEdgeFade = std::clamp(ssr.screenEdgeFade, 0.01f, 0.49f);

    // Ranges match the GTAO panel's sliders. Slice and step counts bound the
    // shader's nested loops, so a large stored value is a performance cliff
    // rather than a visual one; a zero radius would collapse the horizon search
    // to a single point and report no occlusion at all.
    ssao.radius = std::clamp(ssao.radius, 0.05f, 5.0f);
    ssao.intensity = std::clamp(ssao.intensity, 0.0f, 4.0f);
    ssao.power = std::clamp(ssao.power, 0.1f, 8.0f);
    ssao.sliceCount = std::clamp(ssao.sliceCount, 1, 8);
    ssao.stepsPerSlice = std::clamp(ssao.stepsPerSlice, 2, 16);
    ssao.falloff = std::clamp(ssao.falloff, 0.05f, 1.0f);
    ssao.thickness = std::clamp(ssao.thickness, 0.0f, 4.0f);

    // Density and max distance are the two that break things outright: fog is
    // off when either is zero (isVolumetricFogActive checks density, and the
    // shader treats a zero max distance as "no fog"), and maxDistance is a
    // divisor in the froxel slice distribution. Anisotropy must stay inside
    // (-1, 1) or the Henyey-Greenstein phase function's denominator reaches
    // zero. Temporal blend at 1.0 would keep the history forever and never
    // converge.
    fog.density = std::clamp(fog.density, 0.0f, 0.3f);
    fog.maxDistance = std::clamp(fog.maxDistance, renderer::kFogNearPlane + 0.001f, 400.0f);
    fog.anisotropy = std::clamp(fog.anisotropy, -0.9f, 0.9f);
    fog.heightFalloff = std::clamp(fog.heightFalloff, 0.0f, 0.5f);
    fog.baseHeight = std::clamp(fog.baseHeight, -100.0f, 100.0f);
    fog.ambientScale = std::clamp(fog.ambientScale, 0.0f, 3.0f);
    for (float& channel : fog.scatteringColor) {
        channel = std::clamp(channel, 0.0f, 1.0f);
    }
    fog.lightCullThreshold = std::clamp(fog.lightCullThreshold, 0.0f, 1.0f);
    fog.temporalBlend = std::clamp(fog.temporalBlend, 0.0f, 0.98f);

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
    // Upper bound matches drawRenderTargetPreview, which clamps its tint at 16.
    gi.previewGain = std::clamp(gi.previewGain, 1.0f, 16.0f);
    gi.intensity = std::clamp(gi.intensity, 0.0f, 4.0f);
    // The upper bound is roughly one probe spacing: a bias past that samples a
    // different cell than the surface is in, which is worse than self-occlusion.
    gi.surfaceBias = std::clamp(gi.surfaceBias, 0.0f, 4.0f);
    // Capped below 1: at exactly 1 a probe would keep its previous value
    // forever and never take the capture it just paid for.
    gi.hysteresis = std::clamp(gi.hysteresis, 0.0f, 0.99f);
    // Capped below 1: at exactly 1 with a white surface the bounce series does
    // not converge. probeBounceAmplification reports what a given value costs.
    gi.bounceWeight = std::clamp(gi.bounceWeight, 0.0f, 0.95f);

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

    if (const Json* renderScale = objectMember(json, "renderScale")) {
        readFloat(*renderScale, "scale", settings.renderScale.scale);
        readFloat(*renderScale, "sharpness", settings.renderScale.sharpness);
    }

    if (const Json* dynamicResolution = objectMember(json, "dynamicResolution")) {
        readBool(*dynamicResolution, "enabled", settings.dynamicResolution.enabled);
        readFloat(*dynamicResolution, "targetFrameMs", settings.dynamicResolution.targetFrameMs);
        readFloat(*dynamicResolution, "minScale", settings.dynamicResolution.minScale);
        readFloat(*dynamicResolution, "maxScale", settings.dynamicResolution.maxScale);
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
        readBool(*taa, "varianceClipping", settings.taa.varianceClipping);
        readFloat(*taa, "varianceGamma", settings.taa.varianceGamma);
        readBool(*taa, "rejectionFeedback", settings.taa.rejectionFeedback);
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

    if (const Json* ssao = objectMember(json, "ssao")) {
        readBool(*ssao, "enabled", settings.ssao.enabled);
        readFloat(*ssao, "radius", settings.ssao.radius);
        readFloat(*ssao, "intensity", settings.ssao.intensity);
        readFloat(*ssao, "power", settings.ssao.power);
        readInt(*ssao, "sliceCount", settings.ssao.sliceCount);
        readInt(*ssao, "stepsPerSlice", settings.ssao.stepsPerSlice);
        readFloat(*ssao, "falloff", settings.ssao.falloff);
        readFloat(*ssao, "thickness", settings.ssao.thickness);
        readBool(*ssao, "ambientOnly", settings.ssao.ambientOnly);
    }

    if (const Json* fog = objectMember(json, "fog")) {
        readBool(*fog, "enabled", settings.fog.enabled);
        readFloat(*fog, "density", settings.fog.density);
        readFloat(*fog, "maxDistance", settings.fog.maxDistance);
        readFloat(*fog, "anisotropy", settings.fog.anisotropy);
        readFloat(*fog, "heightFalloff", settings.fog.heightFalloff);
        readFloat(*fog, "baseHeight", settings.fog.baseHeight);
        readFloat(*fog, "ambientScale", settings.fog.ambientScale);
        readFloat(*fog, "scatteringColorR", settings.fog.scatteringColor[0]);
        readFloat(*fog, "scatteringColorG", settings.fog.scatteringColor[1]);
        readFloat(*fog, "scatteringColorB", settings.fog.scatteringColor[2]);
        readFloat(*fog, "lightCullThreshold", settings.fog.lightCullThreshold);
        readBool(*fog, "temporalEnabled", settings.fog.temporalEnabled);
        readFloat(*fog, "temporalBlend", settings.fog.temporalBlend);
    }

    if (const Json* punctual = objectMember(json, "punctualShadows")) {
        readBool(*punctual, "enabled", settings.punctualShadows.enabled);
        readBool(*punctual, "gpuCasterCulling", settings.punctualShadows.gpuCasterCulling);
        readBool(*punctual, "debugView", settings.punctualShadows.debugView);
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
        readFloat(*gi, "previewGain", settings.gi.previewGain);
        readFloat(*gi, "intensity", settings.gi.intensity);
        readFloat(*gi, "surfaceBias", settings.gi.surfaceBias);
        readFloat(*gi, "hysteresis", settings.gi.hysteresis);
        readFloat(*gi, "bounceWeight", settings.gi.bounceWeight);
        readBool(*gi, "debugIrradianceOnly", settings.gi.debugIrradianceOnly);
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
        readBool(*renderer, "enableLayeredCascades", settings.enableLayeredCascades);
        readBool(*renderer, "enableAdaptiveOcclusion", settings.enableAdaptiveOcclusion);
        readBool(*renderer, "useClusteredLighting", settings.useClusteredLighting);
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
        {"renderScale",
         Json{{"scale", settings.renderScale.scale}, {"sharpness", settings.renderScale.sharpness}}},
        {"dynamicResolution",
         Json{{"enabled", settings.dynamicResolution.enabled},
              {"targetFrameMs", settings.dynamicResolution.targetFrameMs},
              {"minScale", settings.dynamicResolution.minScale},
              {"maxScale", settings.dynamicResolution.maxScale}}},
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
              {"varianceClipping", settings.taa.varianceClipping},
              {"varianceGamma", settings.taa.varianceGamma},
              {"rejectionFeedback", settings.taa.rejectionFeedback},
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
        {"ssao",
         Json{{"enabled", settings.ssao.enabled},
              {"radius", settings.ssao.radius},
              {"intensity", settings.ssao.intensity},
              {"power", settings.ssao.power},
              {"sliceCount", settings.ssao.sliceCount},
              {"stepsPerSlice", settings.ssao.stepsPerSlice},
              {"falloff", settings.ssao.falloff},
              {"thickness", settings.ssao.thickness},
              {"ambientOnly", settings.ssao.ambientOnly}}},
        {"fog",
         Json{{"enabled", settings.fog.enabled},
              {"density", settings.fog.density},
              {"maxDistance", settings.fog.maxDistance},
              {"anisotropy", settings.fog.anisotropy},
              {"heightFalloff", settings.fog.heightFalloff},
              {"baseHeight", settings.fog.baseHeight},
              {"ambientScale", settings.fog.ambientScale},
              {"scatteringColorR", settings.fog.scatteringColor[0]},
              {"scatteringColorG", settings.fog.scatteringColor[1]},
              {"scatteringColorB", settings.fog.scatteringColor[2]},
              {"lightCullThreshold", settings.fog.lightCullThreshold},
              {"temporalEnabled", settings.fog.temporalEnabled},
              {"temporalBlend", settings.fog.temporalBlend}}},
        {"punctualShadows",
         Json{{"enabled", settings.punctualShadows.enabled},
              {"gpuCasterCulling", settings.punctualShadows.gpuCasterCulling},
              {"debugView", settings.punctualShadows.debugView}}},
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
              {"previewGain", settings.gi.previewGain},
              {"intensity", settings.gi.intensity},
              {"surfaceBias", settings.gi.surfaceBias},
              {"hysteresis", settings.gi.hysteresis},
              {"bounceWeight", settings.gi.bounceWeight},
              {"debugIrradianceOnly", settings.gi.debugIrradianceOnly},
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
              {"enableLayeredCascades", settings.enableLayeredCascades},
              {"enableAdaptiveOcclusion", settings.enableAdaptiveOcclusion},
              {"useClusteredLighting", settings.useClusteredLighting},
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
