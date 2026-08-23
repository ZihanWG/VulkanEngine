#version 460
#include "sub_rect.glsl"

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require

// FrameConstants and its buffer reference, needed by the push block below and by
// the virtual shadow map lookup. The fragment stage reads the same block the
// vertex stage does; see the note on the frameConstants push field.
#include "object_frame_data.glsl"
#include "virtual_shadow_map.glsl"

// Punctual light record. Mirrors ve::renderer::GpuLight (64-byte std430 stride).
struct GpuLight {
    vec4 positionRange;   // xyz = world position, w = range
    vec4 colorIntensity;  // rgb = color, a = intensity
    vec4 directionType;   // xyz = spot direction, w = type (0 point, 1 spot)
    // x = cos(outer), y = 1/(cos(inner)-cos(outer)),
    // z = punctual shadow atlas slot as a float (< 0 means this light does not
    // cast this frame -- not a caster, culled, or the atlas ran out of tiles),
    // w = shadow normal-offset bias, carried here so the shadow lookup can bias
    // the position before it picks a cube face without reading the slot buffer.
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

#include "cluster_grid.glsl"

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
    layout(offset = 100) float fogMaxDistance;
    layout(offset = 104) float aoAmbientStrength;
    // Read directly rather than forwarded as varyings: the virtual shadow map
    // lookup needs a mat4 and three vec4s, which would be seven more flat
    // locations for data identical across every fragment in the frame. The main
    // pipeline already pushes the whole block to both stages, so the address is
    // here for the taking.
    layout(offset = 112) FrameConstantsBuffer frameConstants;
    // The AO target is allocated at the maximum render resolution; only its
    // sub-rect is written, so the reprojected lookup has to be scaled into it.
    layout(offset = 120) vec2 aoUvScale;
} pc;

layout(set = 0, binding = 1) uniform sampler2DArray uShadowMap;
layout(set = 0, binding = 4) uniform samplerCube uDiffuseIrradianceMap;
layout(set = 0, binding = 5) uniform samplerCube uPrefilteredEnvMap;
layout(set = 0, binding = 6) uniform sampler2D uBrdfLut;
layout(set = 0, binding = 7) uniform sampler2D uPunctualShadowAtlas;
// Integrated fog volume: rgb = light gathered in front of this froxel,
// a = how much of the background still shows through.
layout(set = 0, binding = 8) uniform sampler3D uFogVolume;

// Irradiance-probe GI. Two octahedral atlases, one tile per probe: incoming
// radiance, and the two distance moments that say what is between the probe and
// the surface asking.
layout(set = 0, binding = 9) uniform sampler2D uProbeIrradianceAtlas;
layout(set = 0, binding = 10) uniform sampler2D uProbeDepthAtlas;
layout(set = 0, binding = 11) uniform ProbeShadingParams {
    // xyz = grid origin, w = intensity (zero disables the lookup)
    vec4 gridOrigin;
    // xyz = spacing, w = surface bias
    vec4 gridSpacing;
    // x != 0 outputs probe irradiance alone instead of shaded colour
    vec4 debug;
} probeParams;

// Denoised ambient occlusion. The GTAO pass writes this later in the same
// frame, so what is read here is the previous frame's result -- see the
// reprojection below.
layout(set = 0, binding = 12) uniform sampler2D uAmbientOcclusion;

// The same cascade array as binding 1, through a depth-comparison sampler.
//
// A comparison sampler filters the *results* of the depth tests, not the depths:
// one fetch returns the bilinear blend of four pass/fail outcomes, which is a
// sub-texel gradient. The manual path could only average whole 0/1 outcomes,
// so its edge was a staircase no matter how many taps went into it. Binding 1
// stays a plain sampler2DArray because four other consumers want raw depth.
layout(set = 0, binding = 13) uniform sampler2DArrayShadow uShadowMapCompare;
// The virtual shadow map page pool, sampled through the page table rather than
// with a direct UV. Shares the cascades' immutable compare sampler.
layout(set = 0, binding = 14) uniform sampler2DShadow uVsmPagePool;

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
layout(location = 25) flat in vec4 vShadowQuality;
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

// How far into the far edge of its cascade this fragment sits, 0 in the body of
// the cascade and rising to 1 at the split. Drives the cross-fade below.
//
// Without a fade the split is a hard line where filter width and texel density
// change together, which reads as a visible seam sweeping across the ground as
// the camera moves -- the more so once the cascades are well stabilised, because
// then the seam is the only thing still moving.
float cascadeBlendWeight(int cascadeIndex)
{
    const float band = vShadowQuality.y;
    if (band <= 0.0 || cascadeIndex < 0) {
        return 0.0;
    }

    const float farSplit = vCascadeSplits[cascadeIndex];
    const float nearSplit = cascadeIndex == 0 ? 0.0 : vCascadeSplits[cascadeIndex - 1];
    const float range = max(farSplit - nearSplit, 1e-4);
    const float bandStart = farSplit - range * band;
    if (vViewDepth <= bandStart) {
        return 0.0;
    }

    return clamp((vViewDepth - bandStart) / max(farSplit - bandStart, 1e-4), 0.0, 1.0);
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
    // The reference value rides in .w; the hardware does the compare and the
    // bilinear weighting of its four results in one fetch.
    return texture(uShadowMapCompare, vec4(shadowUV, float(cascadeIndex), currentDepth - bias));
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

    // Each tap is now a hardware 2x2 comparison fetch rather than a single
    // point test, so a radius covers the same texels but returns a smooth
    // gradient across each one instead of a 0/1 verdict.
    for (int y = -pcfRadius; y <= pcfRadius; ++y) {
        for (int x = -pcfRadius; x <= pcfRadius; ++x) {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            litSamples += compareShadowDepth(shadowUV + offset, currentDepth, bias, cascadeIndex);
            sampleCount += 1;
        }
    }

    return litSamples / float(sampleCount);
}


// --- Irradiance-probe lookup -------------------------------------------------
//
// Mirrors ve::renderer::IrradianceProbes. Every constant and every formula here
// has a unit-tested twin on the CPU; the tests pin the two together because a
// divergence produces lighting that is plausible and wrong rather than obviously
// broken.

// Must match ve::renderer::kProbeGrid* / kProbe*Resolution / kProbeAtlasTiles*.
const uint kProbeGridX = 8u;
const uint kProbeGridY = 4u;
const uint kProbeGridZ = 8u;
const int kProbeBorderTexels = 1;
const int kProbeIrradianceResolution = 8;
const int kProbeDepthResolution = 16;
const int kProbeAtlasTilesX = 32;
const int kProbeAtlasTilesY = 8;
// Must match ve::renderer::kProbeMinVisibility / kProbeBackfaceFloor.
const float kProbeMinVisibility = 0.02;
const float kProbeBackfaceFloor = 0.2;

// The engine's octahedral encode, identical to octEncode above; named separately
// only to keep the probe block self-contained and easy to compare against the
// CPU mirror.
vec2 probeOctEncode(vec3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return e * 0.5 + 0.5;
}

// Mirrors ve::renderer::probeAtlasUv. The result sits inside the probe's own
// core square, at least half a texel from the tile edge, so a bilinear tap
// reaches this tile's border rather than the neighbouring probe.
vec2 probeAtlasUv(uint probeIndex, vec3 direction, int coreResolution)
{
    int tileSize = coreResolution + 2 * kProbeBorderTexels;
    ivec2 tile = ivec2(int(probeIndex) % kProbeAtlasTilesX, int(probeIndex) / kProbeAtlasTilesX);
    vec2 texel = vec2(tile * tileSize) + float(kProbeBorderTexels) +
                 probeOctEncode(direction) * float(coreResolution);
    return texel / vec2(kProbeAtlasTilesX * tileSize, kProbeAtlasTilesY * tileSize);
}

// Mirrors ve::renderer::probeChebyshevVisibility. This is what stops light
// arriving through a wall, which is the characteristic failure of probe GI and
// looks like a room being softly lit from nowhere.
float probeChebyshevVisibility(vec2 moments, float distanceToProbe)
{
    if (!(distanceToProbe > moments.x)) {
        return 1.0;
    }
    // abs() because bilinear filtering of the two moments can put the mean square
    // below the squared mean, which no single texel can; the variance would go
    // negative and subtract light.
    float variance = abs(moments.x * moments.x - moments.y);
    float excess = distanceToProbe - moments.x;
    float bound = variance / (variance + excess * excess);
    return clamp(bound * bound * bound, kProbeMinVisibility, 1.0);
}

// Eight surrounding probes, trilinear in the grid, each weighted by how much of
// it the surface can actually see.
vec3 sampleProbeIrradiance(vec3 worldPosition, vec3 normal, vec3 viewDirection)
{
    vec3 spacing = max(probeParams.gridSpacing.xyz, vec3(1.0e-4));

    // Offset off the surface before looking up, or a flat plane samples probes
    // whose stored visibility says the plane itself is in the way and shades
    // itself black. Weighted toward the view direction: a normal-only offset
    // barely clears the surface at grazing angles, which is where it matters.
    vec3 biased = worldPosition + (normal * 0.2 + viewDirection * 0.8) * max(probeParams.gridSpacing.w, 0.0);

    vec3 gridSpace = (biased - probeParams.gridOrigin.xyz) / spacing;
    vec3 floored = floor(gridSpace);
    // Clamped rather than extrapolated, so a point outside the volume takes the
    // edge probes instead of an invented value that grows with distance.
    ivec3 baseCoord = clamp(ivec3(floored), ivec3(0), ivec3(kProbeGridX - 1u, kProbeGridY - 1u, kProbeGridZ - 1u));
    vec3 fraction = clamp(gridSpace - floored, vec3(0.0), vec3(1.0));

    vec3 total = vec3(0.0);
    float totalWeight = 0.0;

    for (int corner = 0; corner < 8; ++corner) {
        ivec3 offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        ivec3 coord = min(baseCoord + offset,
                          ivec3(kProbeGridX - 1u, kProbeGridY - 1u, kProbeGridZ - 1u));
        uint probeIndex = uint(coord.x) + uint(coord.y) * kProbeGridX + uint(coord.z) * kProbeGridX * kProbeGridY;

        vec3 probePosition = probeParams.gridOrigin.xyz + vec3(coord) * spacing;
        vec3 toProbe = probePosition - biased;
        float distanceToProbe = length(toProbe);
        vec3 directionToProbe = distanceToProbe > 1.0e-6 ? toProbe / distanceToProbe : normal;

        // Trilinear weight for this corner.
        vec3 axisWeight = mix(1.0 - fraction, fraction, vec3(offset));
        float weight = axisWeight.x * axisWeight.y * axisWeight.z;

        // Smooth backface rejection: a hard cutoff would draw a line across
        // otherwise smooth shading where no geometry changes.
        float wrapped = (dot(normal, directionToProbe) + 1.0) * 0.5;
        weight *= wrapped * wrapped + kProbeBackfaceFloor;

        // Visibility from the probe's own stored depth.
        vec2 moments = texture(uProbeDepthAtlas, probeAtlasUv(probeIndex, -directionToProbe,
                                                              kProbeDepthResolution)).rg;
        weight *= probeChebyshevVisibility(moments, distanceToProbe);

        if (weight <= 0.0) {
            continue;
        }

        total += texture(uProbeIrradianceAtlas, probeAtlasUv(probeIndex, normal,
                                                             kProbeIrradianceResolution)).rgb * weight;
        totalWeight += weight;
    }

    // Normalised by the accumulated weight rather than by the trilinear weights
    // alone: visibility and backface terms remove probes unevenly, and without
    // renormalising, a point next to a wall would simply go dark instead of
    // taking its light from the probes it can still see.
    return totalWeight > 0.0 ? total / totalWeight : vec3(0.0);
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
// Cube face containing a direction: the axis with the largest magnitude,
// resolved by sign. Face order is +X, -X, +Y, -Y, +Z, -Z.
//
// This mirrors ve::renderer::pointShadowFaceIndex verbatim, including the >=
// comparisons that break ties toward the earlier axis. The CPU builds one
// projection per face in that same order, so the two must agree exactly -- a
// mismatch samples the wrong tile, which looks like plausible-but-wrong shadows
// rather than an obvious failure.
uint pointShadowFaceIndex(vec3 direction)
{
    vec3 magnitude = abs(direction);
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
        return direction.x >= 0.0 ? 0u : 1u;
    }
    if (magnitude.y >= magnitude.z) {
        return direction.y >= 0.0 ? 2u : 3u;
    }

    return direction.z >= 0.0 ? 4u : 5u;
}

float punctualShadowFactor(GpuLight light, vec3 worldPosition, vec3 normal)
{
    float encodedSlot = light.spotScaleOffset.z;
    if (encodedSlot < 0.0) {
        return 1.0;
    }

    uint baseSlotIndex = uint(encodedSlot);
    bool isPoint = light.directionType.w <= 0.5;

    // Sine of the angle between the surface and the light: 0 when the light hits
    // head-on, 1 at grazing incidence. One shadow texel spans the most depth at
    // grazing incidence, which is where acne appears -- on surfaces edge-on to a
    // downward spot it reads as combed streaks along the shadow boundaries.
    vec3 toLight = normalize(light.positionRange.xyz - worldPosition);
    float normalLight = clamp(dot(normal, toLight), 0.0, 1.0);
    float grazing = sqrt(max(1.0 - normalLight * normalLight, 0.0));

    // Bias before choosing a face, which is why the bias cannot come from the
    // slot: the face index is what selects the slot, so reading it there would
    // mean touching the 96-byte slot buffer twice to reach one float. The bias
    // is a single global value the CPU packs into every light instead, and
    // GpuLight is already loaded here.
    //
    // Both biases scale with the grazing angle, the same shape shadowDepthBias
    // uses on the CSM path. Offsetting along the normal fixes acne more cheaply
    // than depth bias alone; keeping the head-on term small stops contact
    // shadows from detaching into peter panning.
    float normalBias = light.spotScaleOffset.w;
    vec3 biasedPosition = worldPosition + normal * (normalBias * (0.2 + grazing));

    // Select the face from the *biased* position, the same one the projection
    // below uses. Selecting from the unbiased direction instead lets the normal
    // offset push a sample across a face boundary into a face whose frustum no
    // longer contains it; the bounds test then reports "outside the light" and
    // returns fully lit, drawing a bright seam along every cube face boundary.
    // Deriving both from one position is what makes that unrepresentable.
    uint slotIndex = baseSlotIndex;
    if (isPoint) {
        slotIndex += pointShadowFaceIndex(biasedPosition - light.positionRange.xyz);
    }

    GpuShadowSlot slot = pc.punctualShadowSlots.slots[slotIndex];

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

    float currentDepth = projected.z - slot.params.x * (1.0 + grazing * 7.0);
    vec2 atlasUv = slot.atlasUvOffsetScale.xy + tileUv * slot.atlasUvOffsetScale.zw;

    // 3x3 PCF in atlas space, with every tap clamped into this slot's own tile.
    //
    // The clamp is not cosmetic. Neighbouring atlas texels belong to a different
    // tile -- for a cube face that is the adjacent face, and with the quadtree
    // allocator it can be an unrelated light entirely -- so a tap that walks out
    // compares against unrelated depth. On a spot that only ever happened at the
    // cone edge where the falloff is already zero, which is why it went
    // unnoticed; on a cube face the tile border is the *middle* of the lit
    // scene, and the mismatch draws hard seams along the face boundaries.
    float texel = slot.params.z;
    vec2 tileMin = slot.atlasUvOffsetScale.xy;
    // Last addressable texel of the tile: the sampler is NEAREST, so staying
    // within the final texel is enough to keep the fetch inside.
    vec2 tileMax = tileMin + slot.atlasUvOffsetScale.zw - vec2(texel);

    float litSamples = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(float(x), float(y)) * texel;
            vec2 sampleUv = clamp(atlasUv + offset, tileMin, tileMax);
            float closestDepth = texture(uPunctualShadowAtlas, sampleUv).r;
            litSamples += currentDepth <= closestDepth ? 1.0 : 0.0;
        }
    }

    return litSamples / 9.0;
}

// Shadow term for the debug view, gated on the light actually reaching this
// fragment.
//
// evaluatePunctualLight applies the shadow only after its range and cone
// early-outs, so an out-of-range light never darkens the shaded image. The
// debug accumulation has to reproduce those gates or it reports occlusion from
// lights that contribute no light at all -- with several casters that min()s the
// whole frame to black and looks like a catastrophic shadow bug.
float punctualShadowDebugFactor(GpuLight light, vec3 worldPosition, vec3 normal)
{
    vec3 toLight = light.positionRange.xyz - worldPosition;
    float distance = length(toLight);
    float range = max(light.positionRange.w, EPSILON);
    if (distance > range || distance < EPSILON) {
        return 1.0;
    }

    if (light.directionType.w > 0.5) {
        vec3 spotDirection = normalize(light.directionType.xyz);
        float cosAngle = dot(-(toLight / distance), spotDirection);
        if (clamp((cosAngle - light.spotScaleOffset.x) * light.spotScaleOffset.y, 0.0, 1.0) <= 0.0) {
            return 1.0;
        }
    }

    return punctualShadowFactor(light, worldPosition, normal);
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

    // Declared outside the branch: the cascade debug overlay below reads it, and
    // -1 is exactly what it should see on the VSM path, which has no cascades.
    int cascadeIndex = -1;
    float shadowFactor;
    uint vsmSampledLevel = kVsmNoResidentLevel;
    if ((pc.frameConstants.values.vsmPageTable.w & kVsmFlagEnabled) != 0u) {
        // Virtual shadow maps replace the cascades outright rather than blending
        // with them: the two disagree about texel size everywhere, so mixing
        // them would show the disagreement as a seam instead of hiding it.
        shadowFactor = vsmShadowFactorLevel(pc.frameConstants.values,
                                            uVsmPagePool,
                                            vWorldPosition,
                                            normal,
                                            clamp(int(vShadowSettings.w + 0.5), 0, 4),
                                            vsmSampledLevel);
    } else {
        cascadeIndex = selectShadowCascade();
        shadowFactor = sampleShadowFactor(normal, cascadeIndex);
        // Cross-fade into the next cascade over the last slice of this one. Only
        // the fragments inside the band pay for the second lookup.
        const int cascadeCount = clamp(int(vCascadeCount), 1, 4);
        if (cascadeIndex >= 0 && cascadeIndex + 1 < cascadeCount) {
            const float blend = cascadeBlendWeight(cascadeIndex);
            if (blend > 0.0) {
                shadowFactor = mix(shadowFactor, sampleShadowFactor(normal, cascadeIndex + 1), blend);
            }
        }
    }
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

    // Probe GI replaces the constant environment irradiance rather than adding to
    // it: both answer "what diffuse light arrives here", and summing them would
    // double-count. Zero intensity keeps the IBL term, which is what "off" has
    // always meant.
    // Kept unscaled by intensity: the debug view below reports what the probes
    // actually hold, and folding a user-set multiplier into a diagnostic means
    // its display gain has to be retuned every time that multiplier moves.
    vec3 probeIrradiance = vec3(0.0);
    if (probeParams.gridOrigin.w > 0.0) {
        probeIrradiance = sampleProbeIrradiance(vWorldPosition, normal, viewDirection);
        diffuseIbl = probeIrradiance * probeParams.gridOrigin.w * kD;
    }

    // Ambient occlusion belongs on the ambient term alone. Multiplying the
    // composited scene colour by it, which is what the composite pass used to
    // do, darkens direct lighting too -- a crease in full sunlight should not go
    // dark because geometry occludes the *sky*.
    //
    // Within ambient it lands on the diffuse half only. AO answers "how much of
    // the hemisphere reaches this point", which is the diffuse question; the
    // specular equivalent is specular occlusion, a different quantity this
    // engine does not compute, and reusing the diffuse term for it was always an
    // approximation.
    //
    // Keeping it off the specular half is also what makes SSR conservative. The
    // trace subtracts the specular IBL it replaces (see ssr_trace.frag) and
    // cannot see this factor, so an AO-attenuated specularIbl would be
    // over-subtracted: the reflection would go negative in exactly the creases
    // AO darkens, and scene colour is R16G16B16A16_SFLOAT, so that negative
    // reaches bloom and the exposure histogram instead of clamping at the pixel.
    //
    // The AO image still holds the previous frame's result at this point, since
    // the GTAO pass runs after this one, so the sample is reprojected along the
    // motion vector this fragment already computes for TAA. A reprojection that
    // lands off screen has no history to read and falls back to unoccluded --
    // occlusion appearing a frame late is far less objectionable than sampling
    // an unrelated pixel.
    vec3 ambientDiffuse = diffuseIbl + vAmbientColor * baseColor * 0.05;
    if (pc.aoAmbientStrength > 0.0) {
        vec2 screenUv = gl_FragCoord.xy / vec2(pc.screenWidth, pc.screenHeight);
        vec2 aoUv = screenUv - computeVelocity();
        float occlusion = 1.0;
        if (aoUv == clamp(aoUv, vec2(0.0), vec2(1.0))) {
            occlusion =
                texture(uAmbientOcclusion, veSubRectUv(aoUv, pc.aoUvScale, vec2(textureSize(uAmbientOcclusion, 0)))).r;
        }
        ambientDiffuse *= mix(1.0, occlusion, clamp(pc.aoAmbientStrength, 0.0, 1.0));
    }

    vec3 ambient = ambientDiffuse + specularIbl;
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
            if (pc.debugPunctualShadows != 0u) {
                punctualVisibility =
                    min(punctualVisibility,
                        punctualShadowDebugFactor(pc.lightBuffer.lights[lightIndex], vWorldPosition, normal));
            }
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
            if (pc.debugPunctualShadows != 0u) {
                punctualVisibility =
                    min(punctualVisibility,
                        punctualShadowDebugFactor(pc.lightBuffer.lights[lightIndex], vWorldPosition, normal));
            }
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

    // Volumetric fog. Applied here rather than in composite because the view
    // depth needed to find the froxel is already a varying, so no depth
    // attachment has to be made samplable; it also lands in scene colour before
    // bloom and TAA, which is where fog belongs.
    //
    // A max distance of zero means fog is off, so no separate flag is needed.
    if (pc.fogMaxDistance > 0.0) {
        // Exponential slice distribution, inverting ve::renderer's
        // fogSliceViewDepth. Its round trip against the injection pass is
        // pinned by a unit test.
        const float kFogNearPlane = 0.5;
        float fogFar = max(pc.fogMaxDistance, kFogNearPlane + 1.0e-3);
        float fogDepth = clamp(vViewDepth, kFogNearPlane, fogFar);
        float fogW = log(fogDepth / kFogNearPlane) / log(fogFar / kFogNearPlane);

        vec2 fogUv = vec2(gl_FragCoord.x / pc.screenWidth, gl_FragCoord.y / pc.screenHeight);
        vec4 fog = texture(uFogVolume, vec3(fogUv, clamp(fogW, 0.0, 1.0)));

        finalColor = finalColor * fog.a + fog.rgb;
    }

    // Debug overlay: the punctual shadow visibility term on its own. White is
    // fully lit, black fully shadowed, so a flat white frame means the atlas is
    // never being sampled while any structure means the lookup works and the
    // term is simply subtle in the shaded image.
    // Probe irradiance on its own. An indirect term this subtle is easy to
    // mistake for no term at all, which is the same reason the punctual shadow
    // path has a visibility-only view: "I cannot see it" is not evidence either
    // way without one.
    if (probeParams.debug.x != 0.0) {
        outColor = vec4(probeIrradiance, 1.0);
        outVelocity = computeVelocity();
        outNormalRoughness = vec4(octEncode(normal), roughness, metallic);
        return;
    }

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

    // The VSM counterpart. Not a mix like the cascades' 0.3: the tint is applied
    // before tone mapping and exposure, so a blended colour cannot be read back
    // out of a capture -- which is the entire point of this view. Same shape as
    // the LOD heatmap instead: the hue is the answer, and scene luminance only
    // modulates its brightness so the geometry stays legible.
    if ((pc.frameConstants.values.vsmPageTable.w & kVsmFlagDebugLevels) != 0u) {
        float debugLuminance = dot(finalColor, vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(vsmLevelDebugColor(vsmSampledLevel) * (0.35 + 0.65 * clamp(debugLuminance, 0.0, 1.0)),
                        alpha);
        outVelocity = computeVelocity();
        outNormalRoughness = vec4(octEncode(normal), roughness, metallic);
        return;
    }

    outColor = vec4(finalColor, alpha);
    outVelocity = computeVelocity();
    outNormalRoughness = vec4(octEncode(normal), roughness, metallic);
}
