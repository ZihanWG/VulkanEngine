#version 460

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 texColor = texture(uTexture, vUV);
    outColor = vec4(vColor, 1.0) * texColor;
}
