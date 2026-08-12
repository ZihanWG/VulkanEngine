#version 460
#include "sub_rect.glsl"

// Ground-Truth Ambient Occlusion (Jimenez et al. 2016, "Practical Realtime
// Strategies for Accurate Indirect Occlusion"). For each pixel this marches a
// set of screen-space slices around the view direction, finds the maximum
// horizon angle on each side of every slice, and integrates the cosine-weighted
// visibility arc between the two horizons against the projected surface normal.
//
// Inputs: the main depth buffer (view-position reconstruction) and the thin
// G-buffer (octahedral world normal in .xy, matching ssr_trace.frag). Output is
// a single-channel visibility term in [0, 1] (1 = fully lit); the composite pass
// multiplies it into ambient/indirect scene color.

layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(set = 0, binding = 1) uniform sampler2D uNormalRoughness;

layout(set = 0, binding = 2, std430) readonly buffer GtaoParamsBuffer {
    mat4 view;
    mat4 projection;
    mat4 inverseProjection;
    // x = radius (view-space units), y = falloff range (0..1 of radius),
    // z = slice count, w = steps per slice.
    vec4 params0;
    // x = intensity, y = power, z = thickness (view-space units), w = frame index (jitter).
    vec4 params1;
    // xy = written/allocated. The thin G-buffer and depth share the scene
    // allocation. Only the fetches are scaled; UVs feeding the view-space
    // reconstruction stay in written-region space.
    vec4 subRect;
} params;

layout(location = 0) in vec2 vUV;
layout(location = 0) out float outAO;

const float PI = 3.14159265359;
const float HALF_PI = 1.57079632679;

vec3 octDecode(vec2 e)
{
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

vec3 viewPositionFromDepth(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = params.inverseProjection * clip;
    return viewPos.xyz / viewPos.w;
}

// Interleaved gradient noise: per-pixel slice-rotation + step-offset jitter that
// a spatial denoise (and later TAA) integrates out. The frame index rotates the
// pattern so temporal accumulation converges.
float interleavedGradientNoise(vec2 pixel, float frame)
{
    pixel += frame * 5.588238;
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

void main()
{
    float depth = textureLod(uDepth, veSubRectUv(vUV, params.subRect.xy, vec2(textureSize(uDepth, 0))), 0.0).r;
    if (depth >= 0.99999) {
        outAO = 1.0; // sky / background is never occluded
        return;
    }

    const vec3 P = viewPositionFromDepth(vUV, depth);
    const vec3 N = normalize(mat3(params.view) *
                             octDecode(texture(uNormalRoughness,
                                               veSubRectUv(vUV, params.subRect.xy, vec2(textureSize(uNormalRoughness, 0))))
                                           .xy));
    const vec3 V = normalize(-P); // view-space camera looks down -Z, so -P points at the eye

    const float radius = max(params.params0.x, 0.01);
    const float falloffRange = clamp(params.params0.y, 0.01, 1.0);
    const int sliceCount = max(int(params.params0.z), 1);
    const int stepsPerSlice = max(int(params.params0.w), 1);
    const float intensity = clamp(params.params1.x, 0.0, 4.0);
    const float power = max(params.params1.y, 0.0001);

    // Screen-space UV extent of a view-space `radius` sphere at this depth,
    // recovered from the projection focal lengths.
    const vec2 radiusUV = 0.5 * radius * vec2(params.projection[0][0], params.projection[1][1]) / max(-P.z, 1e-4);

    const float noise = interleavedGradientNoise(gl_FragCoord.xy, params.params1.w);
    const float falloffStart = 1.0 - falloffRange;

    float visibility = 0.0;

    for (int slice = 0; slice < sliceCount; ++slice) {
        const float phi = (float(slice) + noise) * PI / float(sliceCount);
        const vec2 omega = vec2(cos(phi), sin(phi)); // screen-space slice direction
        const vec2 dirUV = omega * radiusUV;

        // Build the slice-plane basis in view space and project the normal into
        // it (GTAO integrates the arc in the plane spanned by V and the slice).
        const vec3 directionVec = vec3(omega, 0.0);
        const vec3 orthoDirectionVec = normalize(directionVec - dot(directionVec, V) * V);
        const vec3 axisVec = normalize(cross(orthoDirectionVec, V)); // slice-plane normal
        const vec3 projectedNormal = N - axisVec * dot(N, axisVec);
        const float projectedNormalLength = length(projectedNormal);
        if (projectedNormalLength < 1e-4) {
            continue;
        }

        const float cosNormal = clamp(dot(projectedNormal, V) / projectedNormalLength, -1.0, 1.0);
        const float signNormal = sign(dot(orthoDirectionVec, projectedNormal));
        const float n = signNormal * acos(cosNormal); // projected-normal angle from V

        // Horizon search: the largest elevation cosine on each side of the slice.
        float cosHorizonPos = -1.0;
        float cosHorizonNeg = -1.0;
        for (int step = 0; step < stepsPerSlice; ++step) {
            const float t = (float(step) + noise) / float(stepsPerSlice);

            // Positive side (+omega).
            {
                const vec2 sampleUV = vUV + dirUV * t;
                if (all(greaterThanEqual(sampleUV, vec2(0.0))) && all(lessThanEqual(sampleUV, vec2(1.0)))) {
                    const float sampleDepth =
                        textureLod(uDepth, veSubRectUv(sampleUV, params.subRect.xy, vec2(textureSize(uDepth, 0))), 0.0)
                            .r;
                    if (sampleDepth < 0.99999) {
                        const vec3 delta = viewPositionFromDepth(sampleUV, sampleDepth) - P;
                        const float dist = length(delta);
                        float cosHorizon = dot(delta, V) / max(dist, 1e-5);
                        // Soften the radius boundary so distant geometry fades out.
                        const float falloff = clamp((1.0 - dist / radius) / max(falloffRange, 1e-4) + falloffStart, 0.0, 1.0);
                        cosHorizon = mix(-1.0, cosHorizon, falloff);
                        cosHorizonPos = max(cosHorizonPos, cosHorizon);
                    }
                }
            }

            // Negative side (-omega).
            {
                const vec2 sampleUV = vUV - dirUV * t;
                if (all(greaterThanEqual(sampleUV, vec2(0.0))) && all(lessThanEqual(sampleUV, vec2(1.0)))) {
                    const float sampleDepth =
                        textureLod(uDepth, veSubRectUv(sampleUV, params.subRect.xy, vec2(textureSize(uDepth, 0))), 0.0)
                            .r;
                    if (sampleDepth < 0.99999) {
                        const vec3 delta = viewPositionFromDepth(sampleUV, sampleDepth) - P;
                        const float dist = length(delta);
                        float cosHorizon = dot(delta, V) / max(dist, 1e-5);
                        const float falloff = clamp((1.0 - dist / radius) / max(falloffRange, 1e-4) + falloffStart, 0.0, 1.0);
                        cosHorizon = mix(-1.0, cosHorizon, falloff);
                        cosHorizonNeg = max(cosHorizonNeg, cosHorizon);
                    }
                }
            }
        }

        // Horizon angles relative to V, clamped into the hemisphere around the
        // projected normal, then the closed-form cosine-weighted arc integral.
        float h1 = acos(clamp(cosHorizonPos, -1.0, 1.0));
        float h2 = -acos(clamp(cosHorizonNeg, -1.0, 1.0));
        h1 = n + clamp(h1 - n, -HALF_PI, HALF_PI);
        h2 = n + clamp(h2 - n, -HALF_PI, HALF_PI);

        const float sinN = sin(n);
        const float arc1 = 0.25 * (-cos(2.0 * h1 - n) + cosNormal + 2.0 * h1 * sinN);
        const float arc2 = 0.25 * (-cos(2.0 * h2 - n) + cosNormal + 2.0 * h2 * sinN);
        visibility += projectedNormalLength * (arc1 + arc2);
    }

    visibility /= float(sliceCount);

    float occlusion = clamp((1.0 - visibility) * intensity, 0.0, 1.0);
    outAO = pow(clamp(1.0 - occlusion, 0.0, 1.0), power);
}
