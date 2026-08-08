#version 460

layout(set = 0, binding = 0) uniform sampler2D uSceneColor;
layout(set = 0, binding = 1) uniform sampler2D uLegacyBloomColor;
layout(set = 0, binding = 2) uniform sampler2D uMipBloomColor;
layout(std430, set = 0, binding = 3) readonly buffer ExposureState {
    float exposure;
    float averageLuminance;
    float histogramLuminance;
    uint mode;
} uExposure;
// Precomputed GTAO visibility term (1 = fully lit) from the dedicated GTAO pass.
layout(set = 0, binding = 4) uniform sampler2D uAmbientOcclusion;

layout(push_constant) uniform CompositePushConstants {
    float exposure;
    float bloomIntensity;
    uint toneMappingOperator;
    uint bloomEnabled;
    uint bloomMethod;
    uint useGpuExposure;
    // Non-zero bypasses everything below and outputs scene colour scaled by it.
    float debugRawGain;
    uint pad1;
    mat4 invProjection;
    vec4 ssaoParams0; // reserved
    vec4 ssaoParams1; // x = ambient-occlusion enabled
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 toneMapReinhard(vec3 color)
{
    return color / (color + vec3(1.0));
}

vec3 toneMapAces(vec3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

void main()
{
    vec3 sceneColor = texture(uSceneColor, vUV).rgb;

    // Raw passthrough for debug views. Everything below this exists to turn a
    // linear HDR scene into a display image, and every part of it works against
    // reading an absolute value: auto-exposure actively cancels the brightness
    // change a debug term is trying to show, and the tone curve compresses what
    // survives. A view of the value itself has to skip all of it.
    if (pc.debugRawGain > 0.0) {
        outColor = vec4(sceneColor * pc.debugRawGain, 1.0);
        return;
    }

    // Reference path only. Occlusion normally lands on the ambient term inside
    // the main pass, where it can be applied without darkening direct lighting;
    // this whole-scene multiply is kept so the two can be compared, and is
    // enabled only when SsaoSettings::ambientOnly is off.
    if (pc.ssaoParams1.x != 0.0) {
        sceneColor *= texture(uAmbientOcclusion, vUV).r;
    }

    vec3 legacyBloom = texture(uLegacyBloomColor, vUV).rgb;
    vec3 mipBloom = texture(uMipBloomColor, vUV).rgb;
    vec3 selectedBloom = pc.bloomMethod == 1u ? mipBloom : legacyBloom;
    vec3 bloomColor = selectedBloom * (pc.bloomEnabled != 0u ? pc.bloomIntensity : 0.0);
    float exposure = pc.useGpuExposure != 0u ? uExposure.exposure : pc.exposure;
    vec3 exposedColor = max((sceneColor + bloomColor) * max(exposure, 0.0), vec3(0.0));

    vec3 mappedColor = pc.toneMappingOperator == 1u ? toneMapAces(exposedColor) : toneMapReinhard(exposedColor);
    outColor = vec4(mappedColor, 1.0);
}
