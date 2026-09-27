// TAA disocclusion test, shared by taa_resolve.frag. Mirrors
// src/renderer/TaaDisocclusion.h function for function, and the flag bits in
// src/renderer/RendererInternal.h; tools/check_shader_constants.py compares the
// constants. The reasoning lives with the C++ half, next to its unit tests.

#ifndef VE_TAA_DISOCCLUSION_GLSL
#define VE_TAA_DISOCCLUSION_GLSL

// Must match ve::renderer::kTaaHistoryDepthOffset / kTaaHistorySkyDepth.
const float kTaaHistoryDepthOffset = 1.0;
const float kTaaHistorySkyDepth = 60000.0;

// Must match ve::renderer::kTaaDisocclusion* (TaaResolvePushConstants flags).
const uint kTaaDisocclusionReject = 1u;
const uint kTaaDisocclusionDebug = 2u;

float taaPreviousViewDepth(vec4 numerator, vec4 denominator, vec2 ndc, float depth)
{
    const vec4 clip = vec4(ndc, depth, 1.0);
    return dot(numerator, clip) / dot(denominator, clip);
}

float taaCurrentViewDepth(vec4 denominator, vec2 ndc, float depth)
{
    return 1.0 / dot(denominator, vec4(ndc, depth, 1.0));
}

float taaEncodeHistoryDepth(float viewDepth)
{
    return kTaaHistoryDepthOffset + clamp(viewDepth, 0.0, kTaaHistorySkyDepth);
}

// Zero or less means the history texel carries no depth and cannot be judged.
float taaDecodeHistoryDepth(float alpha)
{
    return alpha - kTaaHistoryDepthOffset;
}

bool taaHistoryDisoccluded(vec4 historyDepths,
                           float expectedMin,
                           float expectedMax,
                           bool neighbourhoodHasSky,
                           float tolerance)
{
    if (expectedMin > expectedMax) {
        return false;
    }
    const float lower = expectedMin * (1.0 - tolerance);
    const float upper = neighbourhoodHasSky ? 3.0e38 : expectedMax * (1.0 + tolerance);

    bool anyKnown = false;
    for (int i = 0; i < 4; ++i) {
        const float depth = historyDepths[i];
        if (depth <= 0.0) {
            continue;
        }
        anyKnown = true;
        if (depth >= lower && depth <= upper) {
            return false;
        }
    }
    return anyKnown;
}

#endif
