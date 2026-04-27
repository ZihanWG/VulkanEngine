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
layout(location = 8) in vec3 vWorldPosition;
layout(location = 9) flat in vec3 vCameraPosition;
layout(location = 10) flat in vec4 vBaseColorFactor;
layout(location = 11) flat in vec4 vMaterialParams;
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
    vec4 materialColor = texColor * vBaseColorFactor;
    vec3 baseColor = materialColor.rgb * vColor;
    float alpha = materialColor.a;
    float metallic = clamp(vMaterialParams.x, 0.0, 1.0);
    float roughness = clamp(vMaterialParams.y, 0.04, 1.0);

    vec3 normal = normalize(vNormal);
    vec3 lightToSurface = normalize(-vLightDirection);
    vec3 viewDirection = normalize(vCameraPosition - vWorldPosition);
    vec3 halfVector = normalize(lightToSurface + viewDirection);

    float diffuse = max(dot(normal, lightToSurface), 0.0);
    float specularAngle = max(dot(normal, halfVector), 0.0);
    float shininess = mix(128.0, 4.0, roughness);
    float specularStrength = mix(1.0, 0.2, roughness);
    vec3 diffuseColor = baseColor * (1.0 - metallic);
    vec3 specularColor = mix(vec3(0.04), baseColor, metallic);
    vec3 specular = specularColor * pow(specularAngle, shininess) * specularStrength * diffuse;

    float shadowFactor = sampleShadowFactor();
    vec3 ambient = vAmbientColor * baseColor * (1.0 - metallic * 0.5);
    vec3 direct = vLightColor * (diffuseColor * diffuse + specular) * shadowFactor;

    outColor = vec4(ambient + direct, alpha);
}
