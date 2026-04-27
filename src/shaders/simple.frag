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
layout(location = 7) flat in vec4 vShadowSettings;
layout(location = 0) out vec4 outColor;

float shadowDepthBias()
{
    float constantBias = max(vShadowSettings.x, 0.0);
    float slopeBias = max(vShadowSettings.y, 0.0);
    vec3 normal = normalize(vNormal);
    vec3 lightToSurface = normalize(-vLightDirection);
    float normalLight = max(dot(normal, lightToSurface), 0.0);

    // Shadow acne comes from comparing nearby finite-precision depths for the
    // same surface. This tiny shader bias complements raster depth bias; pushing
    // it too high causes peter panning where shadows detach from casters.
    return max(constantBias, slopeBias * (1.0 - normalLight));
}

float compareShadowDepth(vec2 shadowUV, float currentDepth, float bias)
{
    float closestDepth = texture(uShadowMap, shadowUV).r;
    return currentDepth - bias <= closestDepth ? 1.0 : 0.0;
}

float sampleShadowFactor()
{
    vec3 shadowCoord = vLightSpacePosition.xyz / vLightSpacePosition.w;
    vec2 shadowUV = shadowCoord.xy * 0.5 + 0.5;

    if (shadowCoord.z < 0.0 || shadowCoord.z > 1.0
        || shadowUV.x < 0.0 || shadowUV.x > 1.0
        || shadowUV.y < 0.0 || shadowUV.y > 1.0) {
        return 1.0;
    }

    float currentDepth = shadowCoord.z;
    float bias = shadowDepthBias();
    bool enablePcf = vShadowSettings.z > 0.5;
    int pcfRadius = clamp(int(vShadowSettings.w + 0.5), 0, 4);

    if (!enablePcf || pcfRadius == 0) {
        return compareShadowDepth(shadowUV, currentDepth, bias);
    }

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    float litSamples = 0.0;
    int sampleCount = 0;

    // PCF keeps manual depth comparisons but averages nearby texels. A 3x3
    // kernel softens jagged shadow-map edges without changing descriptor layout
    // or switching to sampler compare mode.
    for (int y = -pcfRadius; y <= pcfRadius; ++y) {
        for (int x = -pcfRadius; x <= pcfRadius; ++x) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            litSamples += compareShadowDepth(shadowUV + offset, currentDepth, bias);
            sampleCount += 1;
        }
    }

    return litSamples / float(sampleCount);
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
