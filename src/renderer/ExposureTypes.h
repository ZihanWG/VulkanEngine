#pragma once

// GPU-layout structs, dispatch constants, and small value helpers for the
// auto-exposure pipeline (luminance reduction, histogram, exposure reduce).
// Relocated out of RendererInternal.h's anonymous namespace into a focused,
// shareable header so the exposure subsystem code can reuse them without pulling
// in the full renderer. The structs mirror the corresponding GLSL std430 layouts.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <glm/vec4.hpp>

namespace ve {

inline constexpr uint32_t kLuminanceLocalSizeX = 16;
inline constexpr uint32_t kLuminanceLocalSizeY = 16;
inline constexpr uint32_t kHistogramBinCount = 256;
inline constexpr uint32_t kHistogramLocalSizeX = 16;
inline constexpr uint32_t kHistogramLocalSizeY = 16;
inline constexpr float kDefaultHistogramMinLogLuminance = -10.0f;
inline constexpr float kDefaultHistogramMaxLogLuminance = 4.0f;

struct LuminancePushConstants {
    glm::uvec4 params{0, 0, 0, 0};
};

static_assert(sizeof(LuminancePushConstants) == 16);

struct LuminancePartial {
    float sumLogLuminance = 0.0f;
    float sampleCount = 0.0f;
    float padding0 = 0.0f;
    float padding1 = 0.0f;
};

static_assert(sizeof(LuminancePartial) == 16);

struct HistogramPushConstants {
    glm::uvec4 params{0, 0, 0, 0};
    glm::vec4 logLuminanceRange{kDefaultHistogramMinLogLuminance, kDefaultHistogramMaxLogLuminance, 0.0001f, 0.0f};
};

static_assert(offsetof(HistogramPushConstants, params) == 0);
static_assert(offsetof(HistogramPushConstants, logLuminanceRange) == 16);
static_assert(sizeof(HistogramPushConstants) == 32);

struct ExposureState {
    float exposure = 1.0f;
    float averageLuminance = 0.18f;
    float histogramLuminance = 0.18f;
    uint32_t mode = 0;
};

static_assert(offsetof(ExposureState, exposure) == 0);
static_assert(offsetof(ExposureState, averageLuminance) == 4);
static_assert(offsetof(ExposureState, histogramLuminance) == 8);
static_assert(offsetof(ExposureState, mode) == 12);
static_assert(sizeof(ExposureState) == 16);

struct ExposureReducePushConstants {
    glm::uvec4 params{0, 0, 0, 0};
    glm::vec4 exposureParams{1.0f, 0.18f, 0.1f, 8.0f};
    glm::vec4 adaptationParams{0.0f, 1.5f, 0.05f, 0.95f};
    glm::vec4 histogramRange{kDefaultHistogramMinLogLuminance, kDefaultHistogramMaxLogLuminance, 0.0001f, 0.0f};
};

static_assert(offsetof(ExposureReducePushConstants, params) == 0);
static_assert(offsetof(ExposureReducePushConstants, exposureParams) == 16);
static_assert(offsetof(ExposureReducePushConstants, adaptationParams) == 32);
static_assert(offsetof(ExposureReducePushConstants, histogramRange) == 48);
static_assert(sizeof(ExposureReducePushConstants) == 64);

inline float toneMappingExposureValue(float exposure)
{
    return std::max(exposure, 0.0f);
}

inline std::pair<float, float> sanitizedHistogramLogRange(float minLogLuminance, float maxLogLuminance)
{
    if (!std::isfinite(minLogLuminance) || !std::isfinite(maxLogLuminance) ||
        maxLogLuminance <= minLogLuminance + 0.001f) {
        return {kDefaultHistogramMinLogLuminance, kDefaultHistogramMaxLogLuminance};
    }

    return {minLogLuminance, maxLogLuminance};
}

inline std::pair<float, float> sanitizedPercentileRange(float lowPercentile, float highPercentile)
{
    float low = std::clamp(lowPercentile, 0.0f, 1.0f);
    float high = std::clamp(highPercentile, 0.0f, 1.0f);
    if (high <= low) {
        low = 0.05f;
        high = 0.95f;
    }

    return {low, high};
}

} // namespace ve
