#version 460
#include "sub_rect.glsl"

layout(set = 0, binding = 0) uniform sampler2D uSceneColor;

layout(push_constant) uniform BloomExtractPushConstants {
    float threshold;
    float padding;
    // Scene colour is sub-rected; this target is not.
    vec2 sourceUvScale;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 hdrColor = texture(uSceneColor, veSubRectUv(vUV, pc.sourceUvScale, vec2(textureSize(uSceneColor, 0)))).rgb;
    float luminance = dot(hdrColor, vec3(0.2126, 0.7152, 0.0722));
    vec3 brightColor = luminance > pc.threshold ? hdrColor : vec3(0.0);
    outColor = vec4(brightColor, 1.0);
}
