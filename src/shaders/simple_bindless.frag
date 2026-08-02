#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require

// Punctual light record. Mirrors ve::renderer::GpuLight (64-byte std430 stride).
struct GpuLight {
    vec4 positionRange;   // xyz = world position, w = range
    vec4 colorIntensity;  // rgb = color, a = intensity
    vec4 directionType;   // xyz = spot direction, w = type (0 point, 1 spot)
    // x = cos(outer), y = 1/(cos(inner)-cos(outer)),
    // z = punctual shadow atlas slot as a float (< 0 means this light does not
    // cast this frame -- not a caster, culled, or the atlas ran out of tiles).
    vec4 spotScaleOffset;
};

layout(buffer_reference, std430) readonly buffer LightBuffer {
    GpuLight lights[];
};

// One punctual shadow atlas tile. Mirrors ve::renderer::GpuShadowSlot
// (96-byte std430 stride).
struct GpuShadowSlot {
    mat4 viewProjection;
    vec4 atlasUvOffsetScale; // xy = tile UV origin, zw = tile UV extent
    vec4 params;             // x = constant bias, y = normal bias, z = atlas texel in UV
};

layout(buffer_reference, std430) readonly buffer ShadowSlotBuffer {
    GpuShadowSlot slots[];
};

// Per-cluster light list produced by light_cull.comp. cells[i] = (offset, count)
// into the flat light index list. Grid dims must match the culling shaders.
struct ClusterCell {
    uint offset;
    uint count;
};

layout(buffer_reference, std430) readonly buffer ClusterGridBuffer {
    ClusterCell cells[];
};

layout(buffer_reference, std430) readonly buffer LightIndexBuffer {
    uint indices[];
};

const uint kClusterGridX = 16u;
const uint kClusterGridY = 9u;
const uint kClusterGridZ = 24u;

// Matches ve::PushConstants. The vertex stage reads the leading object-data
// address + cascade index; the fragment stage reads the punctual-light and
// cluster-grid fields, so it declares them at their explicit byte offsets.
layout(push_constant) uniform PushConstants {
    layout(offset = 20) uint lightCount;
    layout(offset = 24) LightBuffer lightBuffer;
    layout(offset = 32) ClusterGridBuffer clusterGrid;
    layout(offset = 40) LightIndexBuffer lightIndexList;
    layout(offset = 48) float clusterZNear;
    layout(offset = 52) float clusterZFar;
    layout(offset = 56) float screenWidth;
    layout(offset = 60) float screenHeight;
    layout(offset = 64) uint useClustered;
    layout(offset = 68) uint debugClusterHeatmap;
    layout(offset = 80) uint debugLodHeatmap;
    layout(offset = 88) ShadowSlotBuffer punctualShadowSlots;
    layout(offset = 96) uint debugPunctualShadows;
} pc;

layout(set = 0, binding = 1) uniform sampler2DArray uShadowMap;
layout(set = 0, binding = 4) uniform samplerCube uDiffuseIrradianceMap;
layout(set = 0, binding = 5) uniform samplerCube uPrefilteredEnvMap;
layout(set = 0, binding = 6) uniform sampler2D uBrdfLut;
layout(set = 0, binding = 7) uniform sampler2D uPunctualShadowAtlas;

layout(set = 1, binding = 0) uniform sampler2D uBaseColorTextures[];
layout(set = 1, binding = 1) uniform sampler2D uNormalTextures[];
layout(set = 1, binding = 2) uniform sampler2D uMetallicRoughnessTextures[];

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vLightDirection;
layout(location = 3) in vec3 vLightColor;
layout(location = 4) in vec3 vAmbientColor;
layout(location = 5) in vec4 vLightSpacePosition[4];
layout(location = 9) flat in vec4 vShadowSettings;
layout(location = 10) in vec3 vWorldPosition;
layout(location = 11) flat in vec3 vCameraPosition;
layout(location = 12) flat in vec4 vBaseColorFactor;
layout(location = 13) flat in vec4 vMaterialParams;
layout(location = 14) in vec3 vTangent;
layout(location = 15) in vec3 vBitangent;
layout(location = 16) flat in uvec4 vTextureIndices;
layout(location = 17) in float vViewDepth;
layout(location = 18) flat in vec4 vCascadeSplits;
layout(location = 19) flat in uint vCascadeCount;
layout(location = 20) flat in float vCascadeDebugEnabled;
layout(location = 21) flat in vec4 vEmissiveFactor;
layout(location = 22) in vec4 vCurrClipPos;
layout(location = 23) in vec4 vPrevClipPos;
// LOD level chosen by the cull pass, recovered from gl_InstanceIndex's high bits.
layout(location = 24) flat in uint vLodIndex;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;
layout(location = 2) out vec4 outNormalRoughness;

// Octahedral encode of a unit vector into [0,1]^2 for the thin G-buffer.
vec2 octEncode(vec3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return e * 0.5 + 0.5;
}

// UV-space motion vector from the unjittered current/previous clip positions.
// NDC-to-UV halves the delta; Vulkan's y-down NDC matches y-down UV, so no flip.
vec2 computeVelocity()
{
    if (vPrevClipPos.w <= 0.0 || vCurrClipPos.w <= 0.0) {
        return vec2(0.0);
    }
    vec2 currNdc = vCurrClipPos.xy / vCurrClipPos.w;
    vec2 prevNdc = vPrevClipPos.xy / vPrevClipPos.w;
    return (currNdc - prevNdc) * 0.5;
}

const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const float SCHLICK_FRESNEL_AVERAGE = 1.0 / 21.0;

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

int selectShadowCascade()
{
    int cascadeCount = clamp(int(vCascadeCount), 1, 4);
    if (vViewDepth <= 0.0 || vViewDepth > vCascadeSplits[cascadeCount - 1]) {
        return -1;
    }

    for (int cascade = 0; cascade < 4; ++cascade) {
        if (cascade >= cascadeCount) {
            break;
        }
        if (vViewDepth <= vCascadeSplits[cascade]) {
            return cascade;
        }
    }

    return cascadeCount - 1;
}

// Green -> yellow -> orange -> red as detail drops, so a glance shows both the
// spatial LOD distribution and any popping as the camera moves.
vec3 lodDebugColor(uint lodIndex)
{
    if (lodIndex == 0u) {
        return vec3(0.15, 0.85, 0.25);
    }
    if (lodIndex == 1u) {
        return vec3(0.95, 0.90, 0.20);
    }
    if (lodIndex == 2u) {
        return vec3(1.0, 0.50, 0.10);
    }
    return vec3(0.95, 0.15, 0.15);
}

vec3 cascadeDebugColor(int cascadeIndex)
{
    if (cascadeIndex == 0) {
        return vec3(1.0, 0.0, 0.0);
    }
    if (cascadeIndex == 1) {
        return vec3(0.0, 1.0, 0.0);
    }
    if (cascadeIndex == 2) {
        return vec3(0.0, 0.25, 1.0);
    }
    return vec3(1.0, 1.0, 0.0);
}

float compareShadowDepth(vec2 shadowUV, float currentDepth, float bias, int cascadeIndex)
{
    float closestDepth = texture(uShadowMap, vec3(shadowUV, float(cascadeIndex))).r;
    return currentDepth - bias <= closestDepth ? 1.0 : 0.0;
}

float sampleShadowFactor(vec3 normal, int cascadeIndex)
{
    if (cascadeIndex < 0) {
        return 1.0;
    }

    vec4 lightSpacePosition = vLightSpacePosition[cascadeIndex];
    vec3 shadowCoord = lightSpacePosition.xyz / lightSpacePosition.w;
    vec2 shadowUV = shadowCoord.xy * 0.5 + 0.5;

    if (shadowCoord.z < 0.0 || shadowCoord.z > 1.0 || shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
        shadowUV.y < 0.0 || shadowUV.y > 1.0) {
        return 1.0;
    }

    float currentDepth = shadowCoord.z;
    float bias = shadowDepthBias(normal);
    bool enablePcf = vShadowSettings.z > 0.5;
    int pcfRadius = clamp(int(vShadowSettings.w + 0.5), 0, 4);

    if (!enablePcf || pcfRadius == 0) {
        return compareShadowDepth(shadowUV, currentDepth, bias, cascadeIndex);
    }

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0).xy);
    float litSamples = 0.0;
    int sampleCount = 0;

    // PCF keeps manual depth comparisons but averages nearby texels. A 3x3
    // kernel softens jagged shadow-map edges without changing descriptor layout
    // or switching to sampler compare mode.
    for (int y = -pcfRadius; y <= pcfRadius; ++y) {
        for (int x = -pcfRadius; x <= pcfRadius; ++x) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            litSamples += compareShadowDepth(shadowUV + offset, currentDepth, bias, cascadeIndex);
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

vec3 approximateMultiScatterCompensation(
    vec3 f0,
    float roughness,
    float metallic,
    float multiScatterStrength,
    vec3 prefilteredColor,
    vec2 brdf)
{
    // Compact Kulla-Conty-inspired approximation. A production implementation
    // would use a dedicated energy-compensation LUT; this estimates missing
    // single-scatter energy from the existing split-sum BRDF lookup.
    vec3 averageFresnel = f0 + (1.0 - f0) * SCHLICK_FRESNEL_AVERAGE;
    vec3 singleScatterEnergy = clamp(averageFresnel * brdf.x + vec3(brdf.y), vec3(0.0), vec3(1.0));
    vec3 missingEnergy = max(vec3(1.0) - singleScatterEnergy, vec3(0.0));
    float roughEnergy = roughness * roughness;
    float specularWeight = mix(0.25, 1.0, metallic);

    return prefilteredColor
        * averageFresnel
        * missingEnergy
        * roughEnergy
        * specularWeight
        * clamp(multiScatterStrength, 0.0, 1.0);
}

// Visibility of one punctual light at the shaded point, read from the shadow
// atlas. Returns 1.0 (fully lit) whenever the light has no tile this frame, so
// the caller can multiply unconditionally.
//
// The negative-slot sentinel is what guards the buffer dereference: the CPU
// stamps it into every light whenever the atlas is unavailable or disabled, and
// the slot address is only non-zero when at least one light got a tile. So a
// light with a valid slot always implies a valid buffer.
float punctualShadowFactor(GpuLight light, vec3 worldPosition, vec3 normal)
{
    float encodedSlot = light.spotScaleOffset.z;
    if (encodedSlot < 0.0) {
        return 1.0;
    }

    GpuShadowSlot slot = pc.punctualShadowSlots.slots[uint(encodedSlot)];

    // Offsetting the sample along the surface normal fixes acne more cheaply
    // than a large depth bias would, and it does not detach contact shadows the
    // way peter-panning bias does.
    vec3 biasedPosition = worldPosition + normal * slot.params.y;

    vec4 lightSpace = slot.viewProjection * vec4(biasedPosition, 1.0);
    if (lightSpace.w <= 0.0) {
        return 1.0; // behind the light
    }

    vec3 projected = lightSpace.xyz / lightSpace.w;
    vec2 tileUv = projected.xy * 0.5 + 0.5;

    // Outside the light's own frustum there is no depth to compare against. For
    // a spot that is exactly outside the lit cone, where the falloff has already
    // reached zero, so returning "lit" here changes nothing visually.
    if (projected.z < 0.0 || projected.z > 1.0 || tileUv.x < 0.0 || tileUv.x > 1.0 || tileUv.y < 0.0 ||
        tileUv.y > 1.0) {
        return 1.0;
    }

    float currentDepth = projected.z - slot.params.x;
    vec2 atlasUv = slot.atlasUvOffsetScale.xy + tileUv * slot.atlasUvOffsetScale.zw;

    // 3x3 PCF in atlas space. Taps stay inside the tile except within one texel
    // of its border, and the border of a tile is the edge of the light's cone
    // where the falloff has already gone to zero.
    float texel = slot.params.z;
    float litSamples = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(float(x), float(y)) * texel;
            float closestDepth = texture(uPunctualShadowAtlas, atlasUv + offset).r;
            litSamples += currentDepth <= closestDepth ? 1.0 : 0.0;
        }
    }

    return litSamples / 9.0;
}

// Cook-Torrance contribution of one punctual (point/spot) light, with inverse-
// square falloff, a smooth range cutoff, and an optional spot cone. Reuses the
// same GGX/Smith/Fresnel terms as the directional term above so point, spot, and
// sun lighting stay energy-consistent.
vec3 evaluatePunctualLight(GpuLight light,
                           vec3 normal,
                           vec3 viewDirection,
                           vec3 worldPosition,
                           vec3 baseColor,
                           float metallic,
                           float roughness,
                           vec3 f0)
{
    vec3 toLight = light.positionRange.xyz - worldPosition;
    float distance = length(toLight);
    float range = max(light.positionRange.w, EPSILON);
    if (distance > range || distance < EPSILON) {
        return vec3(0.0);
    }

    vec3 lightDirection = toLight / distance;
    float distanceAttenuation = 1.0 / max(distance * distance, EPSILON);
    float rangeFade = clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0);
    float attenuation = distanceAttenuation * rangeFade * rangeFade;

    if (light.directionType.w > 0.5) {
        vec3 spotDirection = normalize(light.directionType.xyz);
        float cosAngle = dot(-lightDirection, spotDirection);
        float spotAttenuation =
            clamp((cosAngle - light.spotScaleOffset.x) * light.spotScaleOffset.y, 0.0, 1.0);
        attenuation *= spotAttenuation * spotAttenuation;
    }

    if (attenuation <= 0.0) {
        return vec3(0.0);
    }

    // Sampled after the cheap rejections above so fully attenuated fragments
    // never pay for the atlas fetch.
    attenuation *= punctualShadowFactor(light, worldPosition, normal);
    if (attenuation <= 0.0) {
        return vec3(0.0);
    }

    vec3 halfVector = normalize(viewDirection + lightDirection);
    float normalLight = max(dot(normal, lightDirection), 0.0);
    float normalView = max(dot(normal, viewDirection), 0.0);
    float halfView = max(dot(halfVector, viewDirection), 0.0);

    vec3 fresnel = fresnelSchlick(halfView, f0);
    float distribution = distributionGGX(normal, halfVector, roughness);
    float geometry = geometrySmith(normal, viewDirection, lightDirection, roughness);

    vec3 diffuse = (1.0 - metallic) * baseColor / PI;
    vec3 specular =
        distribution * geometry * fresnel / max(4.0 * normalView * normalLight, EPSILON);

    vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a;
    return (diffuse + specular) * radiance * normalLight * attenuation;
}

// Blue -> cyan -> green -> yellow -> red ramp for the cluster light-count overlay.
vec3 clusterHeatmapColor(uint count)
{
    float t = clamp(float(count) / 16.0, 0.0, 1.0);
    const vec3 c0 = vec3(0.0, 0.0, 0.3);
    const vec3 c1 = vec3(0.0, 0.6, 1.0);
    const vec3 c2 = vec3(0.0, 1.0, 0.2);
    const vec3 c3 = vec3(1.0, 0.9, 0.0);
    const vec3 c4 = vec3(1.0, 0.1, 0.0);
    if (t < 0.25) {
        return mix(c0, c1, t / 0.25);
    }
    if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) / 0.25);
    }
    if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) / 0.25);
    }
    return mix(c3, c4, (t - 0.75) / 0.25);
}

void main()
{
    vec4 texColor = texture(uBaseColorTextures[nonuniformEXT(vTextureIndices.x)], vUV);
    vec4 materialColor = texColor * vBaseColorFactor;
    vec3 baseColor = materialColor.rgb;
    float alpha = materialColor.a;

    // glTF MASK cutout. vMaterialParams.w carries the cutoff and is negative for
    // OPAQUE/BLEND, so this branch costs one compare for materials that never
    // clip. Discarding here — before any lighting, shadow, or IBL work — also
    // keeps the clipped fragments off the velocity and G-buffer attachments.
    if (vMaterialParams.w >= 0.0 && alpha < vMaterialParams.w) {
        discard;
    }

    vec4 mrSample = texture(uMetallicRoughnessTextures[nonuniformEXT(vTextureIndices.z)], vUV);
    float textureMetallic = mrSample.r;
    float textureRoughness = mrSample.g;
    float metallic = clamp(vMaterialParams.x * textureMetallic, 0.0, 1.0);
    float roughness = clamp(vMaterialParams.y * textureRoughness, 0.04, 1.0);
    float multiScatterStrength = vMaterialParams.z;

    vec3 normalTS = texture(uNormalTextures[nonuniformEXT(vTextureIndices.y)], vUV).xyz * 2.0 - 1.0;
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
    vec3 specular =
        distribution * geometry * fresnel / max(4.0 * normalView * normalLight, EPSILON);

    int cascadeIndex = selectShadowCascade();
    float shadowFactor = sampleShadowFactor(normal, cascadeIndex);
    vec3 irradiance = texture(uDiffuseIrradianceMap, normal).rgb;
    vec3 kD = (1.0 - metallic) * baseColor;
    vec3 diffuseIbl = irradiance * kD;

    vec3 reflectionDirection = normalize(reflect(-viewDirection, normal));
    float maxPrefilterMip = max(float(textureQueryLevels(uPrefilteredEnvMap) - 1), 0.0);
    vec3 prefilteredColor =
        textureLod(uPrefilteredEnvMap, reflectionDirection, roughness * maxPrefilterMip).rgb;
    vec2 brdf = texture(uBrdfLut, vec2(clamp(normalView, 0.0, 1.0), roughness)).rg;
    vec3 iblFresnel = fresnelSchlick(normalView, f0);
    vec3 specularIbl = prefilteredColor * (iblFresnel * brdf.x + brdf.y);
    specularIbl += approximateMultiScatterCompensation(
        f0,
        roughness,
        metallic,
        multiScatterStrength,
        prefilteredColor,
        brdf);

    vec3 ambient = diffuseIbl + specularIbl + vAmbientColor * baseColor * 0.05;
    vec3 direct = (diffuse + specular) * vLightColor * normalLight * shadowFactor;

    // Punctual (point/spot) lights. When clustered culling ran this frame the
    // fragment loops only the lights assigned to its froxel; otherwise it falls
    // back to evaluating every light (also the path for the brute-force compare).
    vec3 punctual = vec3(0.0);
    // Darkest visibility any punctual light reports at this fragment, for the
    // debug view below. Starts fully lit so a fragment no light reaches reads
    // as unshadowed rather than black.
    float punctualVisibility = 1.0;
    uint clusterLightCount = 0u;
    if (pc.useClustered != 0u) {
        uint tileX = min(uint(gl_FragCoord.x / (pc.screenWidth / float(kClusterGridX))), kClusterGridX - 1u);
        uint tileY = min(uint(gl_FragCoord.y / (pc.screenHeight / float(kClusterGridY))), kClusterGridY - 1u);
        float zNear = max(pc.clusterZNear, 1.0e-4);
        float zFar = max(pc.clusterZFar, zNear + 1.0e-4);
        float viewDepth = max(vViewDepth, zNear);
        uint slice = uint(clamp(log(viewDepth / zNear) / log(zFar / zNear) * float(kClusterGridZ),
                                0.0,
                                float(kClusterGridZ - 1u)));
        uint clusterIndex = tileX + tileY * kClusterGridX + slice * kClusterGridX * kClusterGridY;

        ClusterCell cell = pc.clusterGrid.cells[clusterIndex];
        clusterLightCount = cell.count;
        for (uint i = 0u; i < cell.count; ++i) {
            uint lightIndex = pc.lightIndexList.indices[cell.offset + i];
            punctualVisibility = min(punctualVisibility,
                                     punctualShadowFactor(pc.lightBuffer.lights[lightIndex], vWorldPosition, normal));
            punctual += evaluatePunctualLight(pc.lightBuffer.lights[lightIndex],
                                              normal,
                                              viewDirection,
                                              vWorldPosition,
                                              baseColor,
                                              metallic,
                                              roughness,
                                              f0);
        }
    } else {
        for (uint lightIndex = 0u; lightIndex < pc.lightCount; ++lightIndex) {
            punctualVisibility = min(punctualVisibility,
                                     punctualShadowFactor(pc.lightBuffer.lights[lightIndex], vWorldPosition, normal));
            punctual += evaluatePunctualLight(pc.lightBuffer.lights[lightIndex],
                                              normal,
                                              viewDirection,
                                              vWorldPosition,
                                              baseColor,
                                              metallic,
                                              roughness,
                                              f0);
        }
    }

    // Emissive: a constant glow added before tone mapping so it blooms. The
    // factor (vEmissiveFactor.rgb) is modulated by an emissive map sampled from
    // the sRGB base-color array only when one is bound (vEmissiveFactor.w).
    vec3 emissive = vEmissiveFactor.rgb;
    if (vEmissiveFactor.w > 0.5) {
        emissive *= texture(uBaseColorTextures[nonuniformEXT(vTextureIndices.w)], vUV).rgb;
    }

    vec3 finalColor = ambient + direct + punctual + emissive;

    // Debug overlay: the punctual shadow visibility term on its own. White is
    // fully lit, black fully shadowed, so a flat white frame means the atlas is
    // never being sampled while any structure means the lookup works and the
    // term is simply subtle in the shaded image.
    if (pc.debugPunctualShadows != 0u) {
        outColor = vec4(vec3(punctualVisibility), 1.0);
        outVelocity = computeVelocity();
        outNormalRoughness = vec4(octEncode(normal), roughness, metallic);
        return;
    }

    // Debug overlay: visualize how many lights touch each froxel. Saturates the
    // ramp at 16 lights, which is plenty to read cluster occupancy at a glance.
    if (pc.debugClusterHeatmap != 0u && pc.useClustered != 0u) {
        outColor = vec4(clusterHeatmapColor(clusterLightCount), 1.0);
        return;
    }

    // Keep the lighting term as luminance so silhouettes and shading still read
    // through the tint, rather than flattening the scene into blocks of colour.
    if (pc.debugLodHeatmap != 0u) {
        float luminance = dot(finalColor, vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(lodDebugColor(vLodIndex) * (0.35 + 0.65 * clamp(luminance, 0.0, 1.0)), alpha);
        outVelocity = computeVelocity();
        outNormalRoughness = vec4(octEncode(normal), roughness, metallic);
        return;
    }

    if (vCascadeDebugEnabled > 0.5 && cascadeIndex >= 0) {
        finalColor = mix(finalColor, cascadeDebugColor(cascadeIndex), 0.3);
    }

    outColor = vec4(finalColor, alpha);
    outVelocity = computeVelocity();
    outNormalRoughness = vec4(octEncode(normal), roughness, metallic);
}
