#version 460

layout(set = 0, binding = 0) uniform samplerCube uEnvironmentMap;

layout(location = 0) in vec3 vDirection;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 environmentColor = texture(uEnvironmentMap, normalize(vDirection)).rgb;
    outColor = vec4(environmentColor, 1.0);
}
