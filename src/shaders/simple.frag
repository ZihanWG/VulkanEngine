#version 460

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUv;

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 texel = texture(uTexture, vUv);
    outColor = vec4(texel.rgb * vColor, texel.a);
}
