#version 460

layout(set = 0, binding = 0) uniform sampler2D uSceneColor;

layout(push_constant) uniform BloomExtractPushConstants {
    float threshold;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 hdrColor = texture(uSceneColor, vUV).rgb;
    float luminance = dot(hdrColor, vec3(0.2126, 0.7152, 0.0722));
    vec3 brightColor = luminance > pc.threshold ? hdrColor : vec3(0.0);
    outColor = vec4(brightColor, 1.0);
}
