#version 460
#include "sub_rect.glsl"

layout(set = 0, binding = 0) uniform sampler2D uCurrentMip;
layout(set = 0, binding = 1) uniform sampler2D uLowerMip;

layout(push_constant) uniform BloomUpsamplePushConstants {
    vec2 texelSize;
    float radius;
    float padding;
    // Two different mips, so two different written fractions.
    vec2 currentUvScale;
    vec2 lowerUvScale;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// uv and texelRadius are both in written-region units; the helper scales and
// clamps each tap into the allocation.
vec3 tentSample(sampler2D sourceTexture, vec2 uv, vec2 texelRadius, vec2 uvScale)
{
    const vec2 sourceSize = vec2(textureSize(sourceTexture, 0));
#define SAMPLE(offsetUv) texture(sourceTexture, veSubRectClamp((offsetUv) * uvScale, uvScale, sourceSize))
    vec3 color = SAMPLE(uv).rgb * 4.0;
    color += SAMPLE(uv + vec2(-texelRadius.x, -texelRadius.y)).rgb;
    color += SAMPLE(uv + vec2( 0.0,          -texelRadius.y)).rgb * 2.0;
    color += SAMPLE(uv + vec2( texelRadius.x, -texelRadius.y)).rgb;
    color += SAMPLE(uv + vec2(-texelRadius.x,  0.0)).rgb * 2.0;
    color += SAMPLE(uv + vec2( texelRadius.x,  0.0)).rgb * 2.0;
    color += SAMPLE(uv + vec2(-texelRadius.x,  texelRadius.y)).rgb;
    color += SAMPLE(uv + vec2( 0.0,           texelRadius.y)).rgb * 2.0;
    color += SAMPLE(uv + vec2( texelRadius.x,  texelRadius.y)).rgb;
    return color * 0.0625;
#undef SAMPLE
}

void main()
{
    // texelSize is one allocated texel of the lower mip; the tent walks in
    // written-region units, so it is divided by that mip's uv scale.
    vec2 radius = pc.texelSize / max(pc.lowerUvScale, vec2(1e-6)) * max(pc.radius, 0.0);
    vec3 current = texture(uCurrentMip,
                           veSubRectUv(vUV, pc.currentUvScale, vec2(textureSize(uCurrentMip, 0))))
                       .rgb;
    vec3 lower = tentSample(uLowerMip, vUV, radius, pc.lowerUvScale);
    outColor = vec4(current + lower, 1.0);
}
