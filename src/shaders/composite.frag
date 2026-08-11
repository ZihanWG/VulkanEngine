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
    // Contrast-adaptive sharpening strength, 0 = off. The renderer zeroes this
    // whenever the frame is not upscaled, so a native frame is untouched.
    float sharpness;
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

// Scene colour -> display colour. Factored out of main so the sharpen filter can
// run on tone-mapped values: sharpening linear HDR would boost highlights far
// more than shadows for the same local contrast, because the tone curve has not
// compressed them yet.
vec3 toneMapScene(vec3 scene, vec3 bloomColor, float exposure)
{
    vec3 exposed = max((scene + bloomColor) * max(exposure, 0.0), vec3(0.0));
    return pc.toneMappingOperator == 1u ? toneMapAces(exposed) : toneMapReinhard(exposed);
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

    vec3 mappedColor = toneMapScene(sceneColor, bloomColor, exposure);

    if (pc.sharpness > 0.0) {
        // Taps are one *source* texel apart, not one output pixel.
        //
        // This is the whole trick. Between two source texels a bilinear
        // magnification is a linear ramp, and the second difference of a linear
        // ramp is zero -- so output-pixel-spaced taps would find no contrast to
        // restore anywhere except exactly at source texel centres, and the
        // sharpening would come out modulated by the upscale grid. One source
        // texel is the spacing at which real detail actually exists.
        vec2 sourceTexel = 1.0 / vec2(textureSize(uSceneColor, 0));
        vec3 up = texture(uSceneColor, vUV + vec2(0.0, -sourceTexel.y)).rgb;
        vec3 down = texture(uSceneColor, vUV + vec2(0.0, sourceTexel.y)).rgb;
        vec3 left = texture(uSceneColor, vUV + vec2(-sourceTexel.x, 0.0)).rgb;
        vec3 right = texture(uSceneColor, vUV + vec2(sourceTexel.x, 0.0)).rgb;

        // Bloom is deliberately shared across the taps rather than re-fetched:
        // it is a heavily blurred mip chain, so it carries no detail worth
        // sharpening, and reusing it keeps this to four extra fetches.
        vec3 mappedUp = toneMapScene(up, bloomColor, exposure);
        vec3 mappedDown = toneMapScene(down, bloomColor, exposure);
        vec3 mappedLeft = toneMapScene(left, bloomColor, exposure);
        vec3 mappedRight = toneMapScene(right, bloomColor, exposure);

        vec3 neighborhoodAverage = (mappedUp + mappedDown + mappedLeft + mappedRight) * 0.25;
        vec3 sharpened = mappedColor + pc.sharpness * (mappedColor - neighborhoodAverage);

        // Anti-ringing, as a *bounded* overshoot rather than none at all.
        //
        // The obvious limiter -- clamp into the range already present in the
        // five taps -- silently removes almost the whole effect. At an edge the
        // centre tap is by definition at the extreme of its own neighbourhood,
        // so any push away from the local mean lands outside the range and gets
        // clipped straight back. What survives is a slightly steeper ramp with
        // no over- or undershoot, and the rim of over- and undershoot is exactly
        // what the eye reads as sharp. Measured on a 1D model of a bilinear-
        // magnified edge, that limiter left the ramp ends bit-identical.
        //
        // So the bound scales with local contrast instead: a low-contrast area
        // can only move a little (noise stays noise), while a real edge is
        // allowed a rim proportional to the step it already has. Ringing is
        // controlled by how much overshoot is permitted, not by forbidding it.
        const float overshootFraction = 0.25;
        vec3 lowest = min(mappedColor, min(min(mappedUp, mappedDown), min(mappedLeft, mappedRight)));
        vec3 highest = max(mappedColor, max(max(mappedUp, mappedDown), max(mappedLeft, mappedRight)));
        vec3 slack = (highest - lowest) * overshootFraction;

        // Display-referred output, so the final range is the one the swapchain
        // can hold; the undershoot above can go below zero on a dark edge.
        mappedColor = clamp(clamp(sharpened, lowest - slack, highest + slack), vec3(0.0), vec3(1.0));
    }

    outColor = vec4(mappedColor, 1.0);
}
