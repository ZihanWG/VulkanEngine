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
layout(set = 0, binding = 4) uniform sampler2D uDepth;

layout(push_constant) uniform CompositePushConstants {
    float exposure;
    float bloomIntensity;
    uint toneMappingOperator;
    uint bloomEnabled;
    uint bloomMethod;
    uint useGpuExposure;
    uint pad0;
    uint pad1;
    mat4 invProjection;
    vec4 ssaoParams0; // x=radius, y=bias, z=intensity, w=power
    vec4 ssaoParams1; // x=enabled, y=sampleCount
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

// Reconstruct a view-space position from a UV and 0..1 depth using the inverse
// projection matrix (robust to the exact depth convention).
vec3 reconstructViewPos(vec2 uv, float depth)
{
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = pc.invProjection * ndc;
    return viewPos.xyz / viewPos.w;
}

float ssaoHash(vec2 p)
{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Screen-space ambient occlusion sampled on a per-pixel rotated Vogel disk.
// Returns an occlusion multiplier in [0, 1] (1 = unoccluded).
float computeSsao(vec2 uv)
{
    float depth = textureLod(uDepth, uv, 0.0).r;
    if (depth >= 1.0) {
        return 1.0; // background / sky
    }

    vec3 P = reconstructViewPos(uv, depth);
    vec3 N = normalize(cross(dFdx(P), dFdy(P)));
    if (dot(N, P) > 0.0) {
        N = -N; // make the normal face the camera (view space looks down -Z)
    }

    float radius = pc.ssaoParams0.x;
    float bias = pc.ssaoParams0.y;
    int sampleCount = max(int(pc.ssaoParams1.y), 1);

    // Convert the view-space radius to a UV radius using the projection focal
    // lengths recovered from the inverse projection diagonal.
    vec2 focal = vec2(1.0 / pc.invProjection[0][0], 1.0 / pc.invProjection[1][1]) * 0.5;
    float invViewDepth = 1.0 / max(-P.z, 1e-4);
    vec2 radiusUV = radius * focal * invViewDepth;

    float rotation = ssaoHash(gl_FragCoord.xy) * 6.2831853;
    const float goldenAngle = 2.39996323;
    float occlusion = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        float t = (float(i) + 0.5) / float(sampleCount);
        float r = sqrt(t);
        float theta = float(i) * goldenAngle + rotation;
        vec2 dir = vec2(cos(theta), sin(theta));
        vec2 sampleUv = uv + dir * radiusUV * r;
        if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0) {
            continue;
        }

        float sampleDepth = textureLod(uDepth, sampleUv, 0.0).r;
        if (sampleDepth >= 1.0) {
            continue;
        }

        vec3 S = reconstructViewPos(sampleUv, sampleDepth);
        vec3 diff = S - P;
        float dist = length(diff);
        if (dist < 1e-5) {
            continue;
        }

        vec3 dirVS = diff / dist;
        float ndl = max(dot(N, dirVS) - bias, 0.0);
        float rangeCheck = smoothstep(0.0, 1.0, radius / dist);
        occlusion += ndl * rangeCheck;
    }

    occlusion /= float(sampleCount);
    float ao = 1.0 - occlusion * pc.ssaoParams0.z;               // intensity
    ao = pow(clamp(ao, 0.0, 1.0), max(pc.ssaoParams0.w, 0.0001)); // power
    return ao;
}

void main()
{
    vec3 sceneColor = texture(uSceneColor, vUV).rgb;
    if (pc.ssaoParams1.x != 0.0) {
        sceneColor *= computeSsao(vUV);
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
