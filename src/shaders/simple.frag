#version 460

layout(set = 0, binding = 0) uniform sampler2D uTexture;
layout(set = 0, binding = 1) uniform sampler2D uShadowMap;

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec3 vLightDirection;
layout(location = 4) in vec3 vLightColor;
layout(location = 5) in vec3 vAmbientColor;
layout(location = 6) in vec4 vLightSpacePosition;
layout(location = 0) out vec4 outColor;

float sampleShadowFactor()
{
    vec3 shadowCoord = vLightSpacePosition.xyz / vLightSpacePosition.w;
    vec2 shadowUV = shadowCoord.xy * 0.5 + 0.5;

    if (shadowCoord.z < 0.0 || shadowCoord.z > 1.0
        || shadowUV.x < 0.0 || shadowUV.x > 1.0
        || shadowUV.y < 0.0 || shadowUV.y > 1.0) {
        return 1.0;
    }

    float closestDepth = texture(uShadowMap, shadowUV).r;
    float currentDepth = shadowCoord.z;
    float bias = 0.003;
    return currentDepth - bias <= closestDepth ? 1.0 : 0.0;
}

void main()
{
    vec4 texColor = texture(uTexture, vUV);
    vec3 normal = normalize(vNormal);
    vec3 lightToSurface = normalize(-vLightDirection);
    float diffuse = max(dot(normal, lightToSurface), 0.0);
    float shadowFactor = sampleShadowFactor();
    vec3 lighting = vAmbientColor + vLightColor * diffuse * shadowFactor;
    vec3 baseColor = texColor.rgb * vColor;

    outColor = vec4(baseColor * lighting, texColor.a);
}
