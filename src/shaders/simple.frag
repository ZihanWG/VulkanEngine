#version 460

layout(set = 0, binding = 0) uniform sampler2D uTexture;
layout(set = 0, binding = 1) uniform sampler2D uShadowMap;
layout(set = 0, binding = 2) uniform sampler2D uNormalMap;

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
layout(location = 12) in vec3 vTangent;
layout(location = 13) in vec3 vBitangent;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const float EPSILON = 0.0001;

float shadowDepthBias(vec3 normal)
{
    float constantBias = max(vShadowSettings.x, 0.0);
    float slopeBias = max(vShadowSettings.y, 0.0);
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

float sampleShadowFactor(vec3 normal)
{
    vec3 shadowCoord = vLightSpacePosition.xyz / vLightSpacePosition.w;
    vec2 shadowUV = shadowCoord.xy * 0.5 + 0.5;

    if (shadowCoord.z < 0.0 || shadowCoord.z > 1.0
        || shadowUV.x < 0.0 || shadowUV.x > 1.0
        || shadowUV.y < 0.0 || shadowUV.y > 1.0) {
        return 1.0;
    }

    float currentDepth = shadowCoord.z;
    float bias = shadowDepthBias(normal);
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

float distributionGGX(vec3 normal, vec3 halfVector, float roughness)
{
    float alpha = roughness * roughness;
    float alphaSquared = alpha * alpha;
    float normalHalf = max(dot(normal, halfVector), 0.0);
    float normalHalfSquared = normalHalf * normalHalf;
    float denominator = normalHalfSquared * (alphaSquared - 1.0) + 1.0;

    return alphaSquared / max(PI * denominator * denominator, EPSILON);
}

float geometrySchlickGGX(float normalDirection, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return normalDirection / max(normalDirection * (1.0 - k) + k, EPSILON);
}

float geometrySmith(vec3 normal, vec3 viewDirection, vec3 lightDirection, float roughness)
{
    float normalView = max(dot(normal, viewDirection), 0.0);
    float normalLight = max(dot(normal, lightDirection), 0.0);

    return geometrySchlickGGX(normalView, roughness) * geometrySchlickGGX(normalLight, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main()
{
    vec4 texColor = texture(uTexture, vUV);
    vec4 materialColor = texColor * vBaseColorFactor;
    vec3 baseColor = materialColor.rgb;
    float alpha = materialColor.a;
    float metallic = clamp(vMaterialParams.x, 0.0, 1.0);
    float roughness = clamp(vMaterialParams.y, 0.04, 1.0);

    vec3 normalTS = texture(uNormalMap, vUV).xyz * 2.0 - 1.0;
    mat3 tbn = mat3(normalize(vTangent), normalize(vBitangent), normalize(vNormal));
    vec3 normal = normalize(tbn * normalTS);
    vec3 lightDirection = normalize(-vLightDirection);
    vec3 viewDirection = normalize(vCameraPosition - vWorldPosition);
    vec3 halfVector = normalize(viewDirection + lightDirection);

    float normalLight = max(dot(normal, lightDirection), 0.0);
    float normalView = max(dot(normal, viewDirection), 0.0);
    float halfView = max(dot(halfVector, viewDirection), 0.0);

    vec3 f0 = mix(vec3(0.04), baseColor, metallic);
    vec3 fresnel = fresnelSchlick(halfView, f0);
    float distribution = distributionGGX(normal, halfVector, roughness);
    float geometry = geometrySmith(normal, viewDirection, lightDirection, roughness);

    vec3 diffuse = (1.0 - metallic) * baseColor / PI;
    vec3 specular = distribution * geometry * fresnel / max(4.0 * normalView * normalLight, EPSILON);

    float shadowFactor = sampleShadowFactor(normal);
    vec3 ambient = vAmbientColor * baseColor;
    vec3 direct = (diffuse + specular) * vLightColor * normalLight * shadowFactor;

    outColor = vec4(ambient + direct, alpha);
}
