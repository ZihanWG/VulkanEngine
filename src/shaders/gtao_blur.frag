#version 460

// Depth-aware bilateral denoise for the raw GTAO visibility term. GTAO rotates
// its slice directions per pixel (interleaved gradient noise), so the raw output
// is spatially noisy; this pass averages a small neighborhood weighted by both a
// spatial Gaussian and a view-space depth-similarity Gaussian, so occlusion does
// not bleed across depth discontinuities.

layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(set = 0, binding = 1) uniform sampler2D uRawAo;

layout(set = 0, binding = 2, std430) readonly buffer GtaoParamsBuffer {
    mat4 view;
    mat4 projection;
    mat4 inverseProjection;
    vec4 params0; // x = radius, y = falloff, z = slice count, w = steps per slice
    vec4 params1; // x = intensity, y = power, z = thickness, w = frame index
} params;

layout(location = 0) in vec2 vUV;
layout(location = 0) out float outAO;

// View-space Z (negative, increasing magnitude with distance) from hardware depth.
float viewZFromDepth(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = params.inverseProjection * clip;
    return viewPos.z / viewPos.w;
}

void main()
{
    const float centerDepth = textureLod(uDepth, vUV, 0.0).r;
    if (centerDepth >= 0.99999) {
        outAO = 1.0; // sky: leave fully lit, and skip the (undefined) neighborhood
        return;
    }

    const float centerViewZ = viewZFromDepth(vUV, centerDepth);
    const vec2 texel = 1.0 / vec2(textureSize(uRawAo, 0));

    // Depth rejection scales with distance so the tolerance stays perceptually
    // even across the frame; params1.z (thickness) tunes how aggressively edges
    // are preserved.
    const float depthSigma = max(abs(centerViewZ) * 0.05 * max(params.params1.z, 0.05), 1e-3);
    const int radius = 2; // 5x5 neighborhood

    float aoSum = 0.0;
    float weightSum = 0.0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const vec2 offset = vec2(float(x), float(y)) * texel;
            const vec2 sampleUV = vUV + offset;
            const float sampleDepth = textureLod(uDepth, sampleUV, 0.0).r;
            if (sampleDepth >= 0.99999) {
                continue;
            }

            const float sampleViewZ = viewZFromDepth(sampleUV, sampleDepth);
            const float dz = sampleViewZ - centerViewZ;

            // Spatial Gaussian (sigma ~= radius) times depth-similarity Gaussian.
            const float spatial = exp(-float(x * x + y * y) / (2.0 * float(radius) * float(radius)));
            const float depthWeight = exp(-(dz * dz) / (2.0 * depthSigma * depthSigma));
            const float weight = spatial * depthWeight;

            aoSum += textureLod(uRawAo, sampleUV, 0.0).r * weight;
            weightSum += weight;
        }
    }

    outAO = weightSum > 0.0 ? aoSum / weightSum : textureLod(uRawAo, vUV, 0.0).r;
}
