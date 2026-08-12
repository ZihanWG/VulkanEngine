#version 460
#include "sub_rect.glsl"

// Screen-space reflections: view-space linear march against the main depth
// buffer with binary refinement, sampling the pre-reflection scene-color copy
// at the hit point. The output is the fresnel- and confidence-weighted
// reflection contribution; the pipeline blends it additively (ONE + ONE) into
// scene color, so this shader pre-multiplies every weight.

layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(set = 0, binding = 1) uniform sampler2D uNormalRoughness;
layout(set = 0, binding = 2) uniform sampler2D uSceneColorCopy;

layout(set = 0, binding = 3, std430) readonly buffer SsrParamsBuffer {
    mat4 view;
    mat4 projection;
    mat4 inverseProjection;
    // x = max steps, y = refinement steps, z = max distance (view units), w = thickness (view units)
    vec4 marchParams;
    // x = intensity, y = max roughness, z = screen-edge fade start, w = frame index (jitter)
    vec4 weightParams;
    // xy = written/allocated for the thin G-buffer, which is sub-rected. Depth
    // and the scene-colour copy are sized to what is written, so they are
    // sampled unscaled.
    vec4 subRect;
} params;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outReflection;

vec3 octDecode(vec2 e)
{
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

// Reconstructs the view-space position of a sample from its UV + depth.
vec3 viewPositionFromDepth(vec2 uv, float depth)
{
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view = params.inverseProjection * clip;
    return view.xyz / view.w;
}

// Interleaved gradient noise: per-pixel march-start jitter that TAA integrates.
float interleavedGradientNoise(vec2 pixel)
{
    pixel += params.weightParams.w * 5.588238;
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

float screenEdgeFade(vec2 uv)
{
    float fadeStart = clamp(params.weightParams.z, 0.0, 0.49);
    vec2 distanceToEdge = min(uv, 1.0 - uv);
    float edge = min(distanceToEdge.x, distanceToEdge.y);
    return clamp(edge / max(fadeStart, 0.0001), 0.0, 1.0);
}

void main()
{
    outReflection = vec4(0.0);

    float depth = texture(uDepth, vUV).r;
    if (depth >= 0.99999) {
        return; // sky
    }

    vec4 normalRoughness = texture(uNormalRoughness, veSubRectUv(vUV, params.subRect.xy, vec2(textureSize(uNormalRoughness, 0))));
    float roughness = normalRoughness.z;
    float metallic = normalRoughness.w;
    float maxRoughness = max(params.weightParams.y, 0.0001);
    if (roughness >= maxRoughness) {
        return;
    }

    vec3 viewPos = viewPositionFromDepth(vUV, depth);
    vec3 normalWS = octDecode(normalRoughness.xy);
    vec3 normalVS = normalize(mat3(params.view) * normalWS);

    vec3 viewDir = normalize(viewPos);
    vec3 rayDir = normalize(reflect(viewDir, normalVS));

    // Rays bending back toward the camera leave the depth buffer's domain fast;
    // fade them out instead of producing streaks.
    float towardCameraFade = clamp(-sign(rayDir.z) * 0.5 + 0.5 + rayDir.z, 0.0, 1.0);
    if (rayDir.z > 0.9) {
        return;
    }

    uint maxSteps = uint(max(params.marchParams.x, 1.0));
    float maxDistance = max(params.marchParams.z, 0.01);
    float thickness = max(params.marchParams.w, 0.001);
    float stepLength = maxDistance / float(maxSteps);

    float jitter = interleavedGradientNoise(gl_FragCoord.xy);
    float t = stepLength * (0.5 + jitter);

    vec2 hitUV = vec2(-1.0);
    bool hit = false;
    float previousT = 0.0;

    for (uint stepIndex = 0u; stepIndex < maxSteps; ++stepIndex) {
        vec3 samplePos = viewPos + rayDir * t;
        vec4 clip = params.projection * vec4(samplePos, 1.0);
        if (clip.w <= 0.0001) {
            break;
        }
        vec3 ndc = clip.xyz / clip.w;
        vec2 uv = ndc.xy * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
            break;
        }

        float sceneDepth = texture(uDepth, uv).r;
        vec3 scenePos = viewPositionFromDepth(uv, sceneDepth);

        // Depth increases away from the camera along -Z in view space.
        float rayDepth = -samplePos.z;
        float surfaceDepth = -scenePos.z;
        if (rayDepth > surfaceDepth && rayDepth - surfaceDepth < thickness) {
            // Binary refinement between the previous miss and this hit.
            float lo = previousT;
            float hi = t;
            uint refineSteps = uint(max(params.marchParams.y, 0.0));
            for (uint refineIndex = 0u; refineIndex < refineSteps; ++refineIndex) {
                float mid = 0.5 * (lo + hi);
                vec3 midPos = viewPos + rayDir * mid;
                vec4 midClip = params.projection * vec4(midPos, 1.0);
                vec3 midNdc = midClip.xyz / midClip.w;
                vec2 midUV = midNdc.xy * 0.5 + 0.5;
                float midSceneDepth = texture(uDepth, midUV).r;
                vec3 midScenePos = viewPositionFromDepth(midUV, midSceneDepth);
                if (-midPos.z > -midScenePos.z) {
                    hi = mid;
                    uv = midUV;
                } else {
                    lo = mid;
                }
            }
            hitUV = uv;
            hit = true;
            break;
        }

        previousT = t;
        t += stepLength;
    }

    if (!hit) {
        return;
    }

    vec3 reflectedColor = texture(uSceneColorCopy, hitUV).rgb;

    // Fresnel with a grayscale F0 approximation (metal tint is not stored in
    // the thin G-buffer; documented limitation).
    float f0 = mix(0.04, 1.0, metallic);
    float normalView = max(dot(normalVS, -viewDir), 0.0);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - normalView, 5.0);

    float roughnessFade = 1.0 - clamp(roughness / maxRoughness, 0.0, 1.0);
    float confidence = screenEdgeFade(hitUV) * roughnessFade * towardCameraFade;
    float intensity = max(params.weightParams.x, 0.0);

    outReflection = vec4(reflectedColor * fresnel * confidence * intensity, 0.0);
}
