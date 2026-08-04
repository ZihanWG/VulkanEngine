#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require

// Fragment stage for irradiance-probe capture. Writes what a probe sees along
// one direction: outgoing radiance in rgb, distance to the surface in a.
//
// Deliberately a fraction of simple_bindless.frag. What a probe contributes is
// low-frequency diffuse light gathered over 8x8 directions, so the parts of the
// full shading model that describe view-dependent detail cannot survive the
// convolution and are not worth evaluating here:
//
//   kept     albedo, lambert from the directional light and from every punctual
//            light, cascaded and punctual shadows, ambient, emissive
//   dropped  specular of every kind, normal mapping, IBL
//
// Punctual lights are read as a flat array rather than through the per-froxel
// cluster lists. Those lists are built for the camera's froxel grid, so there is
// no froxel to look a probe up in -- a probe is not on screen. Looping every
// light instead is affordable precisely because the capture is tiny: six 16x16
// faces per probe against a full framebuffer.

layout(set = 0, binding = 1) uniform sampler2DArray uShadowMap;
layout(set = 0, binding = 7) uniform sampler2D uPunctualShadowAtlas;
layout(set = 1, binding = 0) uniform sampler2D uBaseColorTextures[];

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vWorldPosition;
layout(location = 3) flat in vec3 vLightDirection;
layout(location = 4) flat in vec3 vLightColor;
layout(location = 5) flat in vec3 vAmbientColor;
layout(location = 6) in vec4 vLightSpacePosition[4];
layout(location = 10) flat in vec4 vShadowSettings;
layout(location = 11) flat in vec4 vCascadeSplits;
layout(location = 12) flat in uint vCascadeCount;
layout(location = 13) flat in vec4 vBaseColorFactor;
layout(location = 14) flat in uvec4 vTextureIndices;
layout(location = 15) flat in vec4 vEmissiveFactor;
layout(location = 16) flat in vec4 vMaterialParams;
layout(location = 17) in float vCameraViewDepth;

// Mirrors GpuLight / GpuShadowSlot in simple_bindless.frag and their C++ twins.
struct GpuLight {
    vec4 positionRange;   // xyz = world position, w = range
    vec4 colorIntensity;  // rgb = colour, a = intensity
    vec4 directionType;   // xyz = spot direction, w = type (0 point, 1 spot)
    // x = cos(outer), y = 1/(cos(inner)-cos(outer)), z = shadow slot as a float
    vec4 spotScaleOffset;
};

layout(buffer_reference, std430) readonly buffer LightBuffer {
    GpuLight lights[];
};

struct GpuShadowSlot {
    mat4 viewProjection;
    vec4 atlasUvOffsetScale;
    vec4 params;
};

layout(buffer_reference, std430) readonly buffer ShadowSlotBuffer {
    GpuShadowSlot slots[];
};

layout(push_constant) uniform PushConstants {
    // Stands in for the vertex stage's buffer reference, which this stage does
    // not read. A reference and a uvec2 are both 8 bytes with 8-byte alignment,
    // so the members after it land at the same offsets in both stages -- which
    // is what has to hold, since one push fills the range for both.
    uvec2 objectFrameData;
    mat4 faceViewProjection;
    vec4 probePosition;
    LightBuffer lights;
    ShadowSlotBuffer punctualShadowSlots;
    uint lightCount;
} pc;

// Must match ve::renderer::kProbeMaxDistance.
const float kProbeMaxDistance = 64.0;

layout(location = 0) out vec4 outRadianceDistance;

// Transcribed from simple_bindless.frag rather than shared, matching how this
// project already keeps octDecode in several shaders: there is no GLSL include
// step in the build. The cascade convention has to stay identical to the main
// pass or probes gather shadows that disagree with what the camera sees.
float shadowDepthBias(vec3 normal)
{
    float constantBias = max(vShadowSettings.x, 0.0);
    float slopeBias = max(vShadowSettings.y, 0.0);
    vec3 lightToSurface = normalize(-vLightDirection);
    float normalLight = max(dot(normal, lightToSurface), 0.0);
    return max(constantBias, slopeBias * (1.0 - normalLight));
}

int selectShadowCascade()
{
    int cascadeCount = clamp(int(vCascadeCount), 1, 4);
    if (vCameraViewDepth <= 0.0 || vCameraViewDepth > vCascadeSplits[cascadeCount - 1]) {
        return -1;
    }

    for (int cascade = 0; cascade < 4; ++cascade) {
        if (cascade >= cascadeCount) {
            break;
        }
        if (vCameraViewDepth <= vCascadeSplits[cascade]) {
            return cascade;
        }
    }

    return cascadeCount - 1;
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

    // Single tap, no PCF. The capture is 16x16 per face and then convolved over
    // a cosine lobe, so softening the shadow edge here would be filtered away
    // several times over.
    float closestDepth = texture(uShadowMap, vec3(shadowUV, float(cascadeIndex))).r;
    return shadowCoord.z - shadowDepthBias(normal) <= closestDepth ? 1.0 : 0.0;
}


// Cube face containing a direction. Mirrors ve::renderer::pointShadowFaceIndex
// and its twin in simple_bindless.frag verbatim, including the >= ties: the CPU
// builds one projection per face in this order, and a mismatch samples the wrong
// tile, which looks like plausible-but-wrong shadows rather than a failure.
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

// Visibility of one punctual light, from the shadow atlas the main pass shares.
//
// A single tap, not the main pass's 3x3 PCF: the capture is 16x16 per face and
// then convolved over a cosine lobe, so a softened shadow edge would be filtered
// away several times over.
//
// The negative-slot sentinel guards the buffer dereference exactly as it does in
// the main pass -- a light with a valid slot implies a valid slot buffer.
float capturePunctualShadow(GpuLight light, vec3 worldPosition, vec3 normal)
{
    float encodedSlot = light.spotScaleOffset.z;
    if (!(encodedSlot >= 0.0)) {
        return 1.0;
    }

    uint baseSlotIndex = uint(encodedSlot);
    bool isPoint = light.directionType.w <= 0.5;

    vec3 toLight = normalize(light.positionRange.xyz - worldPosition);
    float normalLight = clamp(dot(normal, toLight), 0.0, 1.0);
    float grazing = sqrt(max(1.0 - normalLight * normalLight, 0.0));

    // Bias first, then pick the face from the biased position, so the face
    // selection and the projection cannot take different inputs. Selecting from
    // the unbiased direction lets the offset push a sample across a face
    // boundary into a frustum that no longer contains it, which reads as a
    // bright seam along every cube face boundary.
    float normalBias = pc.punctualShadowSlots.slots[baseSlotIndex].params.y;
    vec3 biasedPosition = worldPosition + normal * (normalBias * (0.2 + grazing));

    uint slotIndex = baseSlotIndex;
    if (isPoint) {
        slotIndex += pointShadowFaceIndex(biasedPosition - light.positionRange.xyz);
    }

    GpuShadowSlot slot = pc.punctualShadowSlots.slots[slotIndex];
    vec4 lightSpace = slot.viewProjection * vec4(biasedPosition, 1.0);
    if (lightSpace.w <= 0.0) {
        return 1.0;
    }

    vec3 projected = lightSpace.xyz / lightSpace.w;
    vec2 tileUv = projected.xy * 0.5 + 0.5;
    if (projected.z < 0.0 || projected.z > 1.0 || tileUv.x < 0.0 || tileUv.x > 1.0 ||
        tileUv.y < 0.0 || tileUv.y > 1.0) {
        return 1.0;
    }

    float currentDepth = projected.z - slot.params.x * (1.0 + grazing * 7.0);
    vec2 atlasUv = slot.atlasUvOffsetScale.xy + tileUv * slot.atlasUvOffsetScale.zw;
    float closestDepth = texture(uPunctualShadowAtlas, atlasUv).r;
    return currentDepth <= closestDepth ? 1.0 : 0.0;
}

// Diffuse-only contribution of one punctual light.
//
// No specular, for the same reason the directional term has none: it cannot
// survive a convolution over 8x8 directions. The attenuation and cone shaping
// are the main pass's, so a light reaches exactly as far here as it does on
// screen; the normalisation is the capture's own unnormalised lambert, which
// keeps the punctual and directional terms consistent with each other.
vec3 capturePunctualLight(GpuLight light, vec3 worldPosition, vec3 normal)
{
    vec3 toLight = light.positionRange.xyz - worldPosition;
    float distanceToLight = length(toLight);
    float range = max(light.positionRange.w, 1.0e-4);
    if (distanceToLight > range || distanceToLight < 1.0e-4) {
        return vec3(0.0);
    }

    vec3 lightDirection = toLight / distanceToLight;
    float distanceAttenuation = 1.0 / max(distanceToLight * distanceToLight, 1.0e-4);
    float rangeFade = clamp(1.0 - pow(distanceToLight / range, 4.0), 0.0, 1.0);
    float attenuation = distanceAttenuation * rangeFade * rangeFade;

    if (light.directionType.w > 0.5) {
        vec3 spotDirection = normalize(light.directionType.xyz);
        float cosAngle = dot(-lightDirection, spotDirection);
        float spotAttenuation =
            clamp((cosAngle - light.spotScaleOffset.x) * light.spotScaleOffset.y, 0.0, 1.0);
        attenuation *= spotAttenuation * spotAttenuation;
    }

    float normalLight = max(dot(normal, lightDirection), 0.0);
    attenuation *= normalLight;
    if (attenuation <= 0.0) {
        return vec3(0.0);
    }

    // Sampled only after the cheap rejections, so a fragment outside a light's
    // cone or range never pays for the atlas fetch. With every light looped for
    // every capture fragment, that gate is what keeps the cost bounded.
    attenuation *= capturePunctualShadow(light, worldPosition, normal);
    if (attenuation <= 0.0) {
        return vec3(0.0);
    }

    return light.colorIntensity.rgb * light.colorIntensity.a * attenuation;
}

void main()
{
    // Sampled unconditionally, like the main pass: materials with no base-color
    // texture are given a white fallback slot in the heap rather than a sentinel
    // index, so there is nothing to branch on.
    vec4 baseColor = vBaseColorFactor * texture(uBaseColorTextures[nonuniformEXT(vTextureIndices.x)], vUV);

    // glTF MASK: a negative cutoff means the material has no alpha test.
    float alphaCutoff = vMaterialParams.w;
    if (alphaCutoff >= 0.0 && baseColor.a < alphaCutoff) {
        discard;
    }

    // Two-sided: a probe inside a room sees the inward face of walls that were
    // authored facing out, and back-face culling is off for this pass, so a
    // normal pointing away from the probe is flipped rather than shaded black.
    vec3 normal = normalize(vNormal);
    vec3 toProbe = pc.probePosition.xyz - vWorldPosition;
    if (dot(normal, toProbe) < 0.0) {
        normal = -normal;
    }

    vec3 lightToSurface = normalize(-vLightDirection);
    float normalLight = max(dot(normal, lightToSurface), 0.0);

    int cascadeIndex = selectShadowCascade();
    float shadowFactor = sampleShadowFactor(normal, cascadeIndex);

    vec3 direct = vLightColor * normalLight * shadowFactor;

    // Every punctual light, not a culled subset. See the header: the cluster
    // lists cannot answer for a position that is not on screen.
    for (uint lightIndex = 0u; lightIndex < pc.lightCount; ++lightIndex) {
        direct += capturePunctualLight(pc.lights.lights[lightIndex], vWorldPosition, normal);
    }
    // Emissive rides the base-color heap in slot w, the same arrangement the
    // main pass uses.
    vec3 emissive = vEmissiveFactor.rgb * texture(uBaseColorTextures[nonuniformEXT(vTextureIndices.w)], vUV).rgb;
    vec3 radiance = baseColor.rgb * (direct + vAmbientColor) + emissive;

    // Clamped for the same reason the depth atlas is: the convolution sums this
    // in a 16-bit float, and one blown-out emissive texel would otherwise
    // dominate a whole probe.
    float distance = min(length(toProbe), kProbeMaxDistance);

    outRadianceDistance = vec4(radiance, distance);
}
