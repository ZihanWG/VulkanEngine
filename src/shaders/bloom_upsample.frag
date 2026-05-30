#version 460

layout(set = 0, binding = 0) uniform sampler2D uCurrentMip;
layout(set = 0, binding = 1) uniform sampler2D uLowerMip;

layout(push_constant) uniform BloomUpsamplePushConstants {
    vec2 texelSize;
    float radius;
    float padding;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 tentSample(sampler2D sourceTexture, vec2 uv, vec2 texelRadius)
{
    vec3 color = texture(sourceTexture, uv).rgb * 4.0;
    color += texture(sourceTexture, uv + vec2(-texelRadius.x, -texelRadius.y)).rgb;
    color += texture(sourceTexture, uv + vec2( 0.0,          -texelRadius.y)).rgb * 2.0;
    color += texture(sourceTexture, uv + vec2( texelRadius.x, -texelRadius.y)).rgb;
    color += texture(sourceTexture, uv + vec2(-texelRadius.x,  0.0)).rgb * 2.0;
    color += texture(sourceTexture, uv + vec2( texelRadius.x,  0.0)).rgb * 2.0;
    color += texture(sourceTexture, uv + vec2(-texelRadius.x,  texelRadius.y)).rgb;
    color += texture(sourceTexture, uv + vec2( 0.0,           texelRadius.y)).rgb * 2.0;
    color += texture(sourceTexture, uv + vec2( texelRadius.x,  texelRadius.y)).rgb;
    return color * 0.0625;
}

void main()
{
    vec2 radius = pc.texelSize * max(pc.radius, 0.0);
    vec3 current = texture(uCurrentMip, vUV).rgb;
    vec3 lower = tentSample(uLowerMip, vUV, radius);
    outColor = vec4(current + lower, 1.0);
}
