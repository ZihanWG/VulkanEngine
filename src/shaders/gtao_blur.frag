#version 460

// Joint-bilateral upsample + denoise for half-resolution GTAO. The trace runs at
// half resolution (4x fewer horizon searches); this full-resolution pass gathers
// the half-res visibility neighborhood, weighting each tap by a spatial Gaussian
// and by view-space depth similarity measured against the full-resolution depth
// buffer. That simultaneously removes the trace's per-pixel slice noise and
// upsamples without bleeding occlusion across depth discontinuities.

layout(set = 0, binding = 0) uniform sampler2D uDepth;  // full resolution
layout(set = 0, binding = 1) uniform sampler2D uRawAo;  // half resolution (linear)

layout(set = 0, binding = 2, std430) readonly buffer GtaoParamsBuffer {
    mat4 view;
    mat4 projection;
    mat4 inverseProjection;
    vec4 params0; // x = radius, y = falloff, z = slice count, w = steps per slice
    vec4 params1; // x = intensity, y = power, z = thickness, w = frame index
} params;

layout(location = 0) in vec2 vUV;
layout(location = 0) out float outAO;

// View-space Z (negative, magnitude grows with distance) from hardware depth.
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
        outAO = 1.0; // sky stays fully lit
        return;
    }

    const float centerViewZ = viewZFromDepth(vUV, centerDepth);
    // Sample spacing = one half-resolution texel, so a 3x3 gather spans the
    // full-res footprint the upsample interpolates over.
    const vec2 halfTexel = 1.0 / vec2(textureSize(uRawAo, 0));

    // Depth rejection scales with distance; params1.z (thickness) tunes it.
    const float depthSigma = max(abs(centerViewZ) * 0.05 * max(params.params1.z, 0.05), 1e-3);

    float aoSum = 0.0;
    float weightSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            const vec2 sampleUV = vUV + vec2(float(x), float(y)) * halfTexel;
            const float sampleDepth = textureLod(uDepth, sampleUV, 0.0).r;
            if (sampleDepth >= 0.99999) {
                continue;
            }

            const float sampleViewZ = viewZFromDepth(sampleUV, sampleDepth);
            const float dz = sampleViewZ - centerViewZ;

            const float spatial = exp(-float(x * x + y * y) * 0.5);
            const float depthWeight = exp(-(dz * dz) / (2.0 * depthSigma * depthSigma));
            const float weight = spatial * depthWeight;

            aoSum += textureLod(uRawAo, sampleUV, 0.0).r * weight;
            weightSum += weight;
        }
    }

    outAO = weightSum > 0.0 ? aoSum / weightSum : textureLod(uRawAo, vUV, 0.0).r;
}
