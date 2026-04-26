#version 460

layout(set = 0, binding = 0) uniform sampler2D uTexture;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec3 vLightDirection;
layout(location = 4) in vec3 vLightColor;
layout(location = 5) in vec3 vAmbientColor;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 texColor = texture(uTexture, vUV);
    vec3 normal = normalize(vNormal);
    vec3 lightToSurface = normalize(-vLightDirection);
    float diffuse = max(dot(normal, lightToSurface), 0.0);
    vec3 lighting = vAmbientColor + vLightColor * diffuse;
    vec3 baseColor = texColor.rgb * vColor;

    outColor = vec4(baseColor * lighting, texColor.a);
}
