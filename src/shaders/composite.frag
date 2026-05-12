#version 460

layout(set = 0, binding = 0) uniform sampler2D uSceneColor;
layout(set = 0, binding = 1) uniform sampler2D uBloomColor;

layout(push_constant) uniform CompositePushConstants {
    float exposure;
    float bloomIntensity;
    uint toneMappingOperator;
    uint bloomEnabled;
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
    vec3 bloomColor = texture(uBloomColor, vUV).rgb * (pc.bloomEnabled != 0u ? pc.bloomIntensity : 0.0);
    vec3 exposedColor = max((sceneColor + bloomColor) * max(pc.exposure, 0.0), vec3(0.0));

    vec3 mappedColor = pc.toneMappingOperator == 1u ? toneMapAces(exposedColor) : toneMapReinhard(exposedColor);
    outColor = vec4(mappedColor, 1.0);
}
