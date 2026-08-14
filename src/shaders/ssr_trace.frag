#version 460
#include "sub_rect.glsl"

// Screen-space reflections: march against the main depth buffer with binary
// refinement, sampling the pre-reflection scene-color copy at the hit point.
//
// The pipeline blends additively (ONE + ONE) into scene colour, and scene colour
// already contains the main pass's specular IBL for this pixel. Adding a
// reflection on top of that double-counts the specular energy -- a mirror got its
// highlight roughly twice. What this shader outputs is therefore a *difference*:
//
//   want:  mix(iblSpec, ssr, conf)  =  iblSpec + (ssr - iblSpec) * conf
//   emit:  (ssrColour - prefilteredEnv) * specularWeight * conf
//
// Both terms carry the same specularWeight (F * brdf.x + brdf.y), so it factors
// out of the difference -- which is why this needs no albedo, and why an
// imprecise F0 cannot reintroduce the double-count. It only reweights the
// swap. Where confidence is 0 the output is 0 and the IBL stands untouched;
// where it is 1 the reflection fully replaces it.

layout(set = 0, binding = 0) uniform sampler2D uDepth;
layout(set = 0, binding = 1) uniform sampler2D uNormalRoughness;
layout(set = 0, binding = 2) uniform sampler2D uSceneColorCopy;
// The two the main pass used for specular IBL, so this pass can subtract exactly
// what it is replacing. Bound late (see ScreenSpaceReflections::updateIblDescriptors).
layout(set = 0, binding = 4) uniform samplerCube uPrefilteredEnvMap;
layout(set = 0, binding = 5) uniform sampler2D uBrdfLut;

layout(set = 0, binding = 3, std430) readonly buffer SsrParamsBuffer {
    mat4 view;
    mat4 projection;
    mat4 inverseProjection;
    // x = max steps, y = refinement steps, z = max distance (view units), w = thickness (view units)
    vec4 marchParams;
    // x = intensity, y = max roughness, z = screen-edge fade start, w = frame index (jitter)
    vec4 weightParams;
    // xy = written/allocated. The thin G-buffer, the scene-colour copy and depth
    // all share the scene allocation, so one scale covers every source. Only the
    // texture fetches are scaled -- UVs feeding the view-space reconstruction
    // must stay in written-region space. zw unused: whether the IBL bindings are
    // valid is decided on the CPU, which simply does not run this pass until they
    // are (see frameSsrActive_).
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

    float depth = texture(uDepth, veSubRectUv(vUV, params.subRect.xy, vec2(textureSize(uDepth, 0)))).r;
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

    // March uniformly in SCREEN space, not in view space.
    //
    // A fixed view-space step covers many pixels near the camera and a fraction
    // of one far away, so the old loop skipped geometry close up -- holes in
    // reflections -- while spending most of its steps re-reading the same distant
    // texel. Stepping uniformly along the ray's screen-space projection instead
    // gives every step the same pixel stride at any distance, at the same step
    // count.
    //
    // Depth cannot be interpolated linearly along that path: what varies linearly
    // across the screen is its reciprocal. So the march carries 1/depth and
    // inverts it per step, which is exact rather than an approximation, and does
    // not assume anything about the projection's w row.
    // Push the origin off the surface along its normal before projecting.
    //
    // The old view-space march got this for free: its first sample sat half a
    // fixed world-space step away, which was always well clear of the surface.
    // A screen-space step is a shrinking world-space distance as the camera
    // closes in, so without a bias the first sample lands on the originating
    // surface, passes the `rayDepth > surfaceDepth` test by a hair, and the
    // surface reflects itself. It shows up as a brightening rather than as
    // garbage, which is what makes it worth naming.
    //
    // Scaled by view depth so the bias stays roughly constant in pixels.
    const vec3 rayOrigin = viewPos + normalVS * max(-viewPos.z * 0.002, 0.005);
    vec3 rayEnd = rayOrigin + rayDir * maxDistance;
    // Keep the far end in front of the near plane, or the projection wraps and the
    // screen-space segment becomes meaningless.
    const float kMinViewDepth = 0.01;
    if (-rayEnd.z < kMinViewDepth) {
        const float denom = rayDir.z;
        if (abs(denom) > 1e-6) {
            rayEnd = rayOrigin + rayDir * max((-kMinViewDepth - rayOrigin.z) / denom, 0.0);
        }
    }

    const float startDepth = max(-rayOrigin.z, kMinViewDepth);
    const float endDepth = max(-rayEnd.z, kMinViewDepth);

    vec4 startClip = params.projection * vec4(rayOrigin, 1.0);
    vec4 endClip = params.projection * vec4(rayEnd, 1.0);
    if (startClip.w <= 0.0001 || endClip.w <= 0.0001) {
        return;
    }
    const vec2 startUV = (startClip.xy / startClip.w) * 0.5 + 0.5;
    const vec2 endUV = (endClip.xy / endClip.w) * 0.5 + 0.5;

    const float invStartDepth = 1.0 / startDepth;
    const float invEndDepth = 1.0 / endDepth;

    float jitter = interleavedGradientNoise(gl_FragCoord.xy);
    const float stepFraction = 1.0 / float(maxSteps);
    float u = stepFraction * (0.5 + jitter);

    vec2 hitUV = vec2(-1.0);
    bool hit = false;
    float previousU = 0.0;

    for (uint stepIndex = 0u; stepIndex < maxSteps; ++stepIndex) {
        vec2 uv = mix(startUV, endUV, u);
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
            break;
        }

        float sceneDepth = texture(uDepth, veSubRectUv(uv, params.subRect.xy, vec2(textureSize(uDepth, 0)))).r;
        vec3 scenePos = viewPositionFromDepth(uv, sceneDepth);

        // Depth increases away from the camera along -Z in view space.
        float rayDepth = 1.0 / mix(invStartDepth, invEndDepth, u);
        float surfaceDepth = -scenePos.z;
        if (rayDepth > surfaceDepth && rayDepth - surfaceDepth < thickness) {
            // Binary refinement between the previous miss and this hit, bisecting
            // the same screen-space parameter.
            float lo = previousU;
            float hi = u;
            uint refineSteps = uint(max(params.marchParams.y, 0.0));
            for (uint refineIndex = 0u; refineIndex < refineSteps; ++refineIndex) {
                float mid = 0.5 * (lo + hi);
                vec2 midUV = mix(startUV, endUV, mid);
                float midRayDepth = 1.0 / mix(invStartDepth, invEndDepth, mid);
                float midSceneDepth =
                    texture(uDepth, veSubRectUv(midUV, params.subRect.xy, vec2(textureSize(uDepth, 0)))).r;
                vec3 midScenePos = viewPositionFromDepth(midUV, midSceneDepth);
                if (midRayDepth > -midScenePos.z) {
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

        previousU = u;
        u += stepFraction;
    }

    if (!hit) {
        return;
    }

    vec3 reflectedColor =
        texture(uSceneColorCopy, veSubRectUv(hitUV, params.subRect.xy, vec2(textureSize(uSceneColorCopy, 0)))).rgb;

    // Fresnel with a grayscale F0 approximation (metal tint is not stored in
    // the thin G-buffer; documented limitation).
    float f0 = mix(0.04, 1.0, metallic);
    float normalView = max(dot(normalVS, -viewDir), 0.0);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - normalView, 5.0);

    float roughnessFade = 1.0 - clamp(roughness / maxRoughness, 0.0, 1.0);
    float confidence = screenEdgeFade(hitUV) * roughnessFade * towardCameraFade;

    // Intensity scales how far the reflection *replaces* the IBL, so the combined
    // weight has to saturate at 1. It used to scale a purely additive term, where
    // the setting's 0..4 range simply meant "brighter"; against a signed
    // correction the same 4.0 subtracts four times the specular IBL the main pass
    // actually wrote. Scene colour goes negative, and it is R16G16B16A16_SFLOAT,
    // so the negative survives into bloom and the exposure histogram rather than
    // clamping at the pixel. Above 1.0 this now means "reach full replacement at
    // lower confidence" instead of "over-subtract".
    float replacement = clamp(confidence * max(params.weightParams.x, 0.0), 0.0, 1.0);

    // The same split-sum weighting the main pass applied to its specular IBL.
    // Reconstructed here so the two terms cancel rather than accumulate.
    const vec2 brdf = texture(uBrdfLut, vec2(clamp(normalView, 0.0, 1.0), roughness)).rg;
    const vec3 specularWeight = vec3(fresnel * brdf.x + brdf.y);

    // What the main pass already put here, in world space: the cubemap is
    // world-oriented, and params.view rotates world into view, so its transpose
    // takes the view-space reflection ray back out.
    const vec3 rayDirWS = normalize(transpose(mat3(params.view)) * rayDir);
    const float maxPrefilterMip = max(float(textureQueryLevels(uPrefilteredEnvMap) - 1), 0.0);
    const vec3 prefilteredColor = textureLod(uPrefilteredEnvMap, rayDirWS, roughness * maxPrefilterMip).rgb;

    // Signed on purpose. Where the traced reflection is darker than the
    // environment the correction is negative, which is what makes this a
    // replacement rather than an addition.
    outReflection = vec4((reflectedColor - prefilteredColor) * specularWeight * replacement, 0.0);
}
