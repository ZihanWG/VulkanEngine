#version 460

layout(set = 0, binding = 0) uniform samplerCube uEnvironmentMap;

layout(push_constant) uniform SkyboxPushConstants {
    mat4 inverseViewProjection;
    float exposure;
    uint toneMappingOperator;
} pc;

layout(location = 0) in vec3 vDirection;
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

vec3 applyToneMapping(vec3 hdrColor)
{
    vec3 exposedColor = max(hdrColor * max(pc.exposure, 0.0), vec3(0.0));
    if (pc.toneMappingOperator == 1u) {
        return toneMapAces(exposedColor);
    }

    return toneMapReinhard(exposedColor);
}

void main()
{
    vec3 environmentColor = texture(uEnvironmentMap, normalize(vDirection)).rgb;
    outColor = vec4(applyToneMapping(environmentColor), 1.0);
}
