#ifndef VE_PROBE_CAPTURE_SHADING_GLSL
#define VE_PROBE_CAPTURE_SHADING_GLSL

// What a probe records for one surface it sees: the shading both probe gathers
// share. probe_capture.frag rasterises the scene from the probe and calls this
// per fragment; probe_trace.comp traces a ray per capture texel and calls it at
// the hit. One copy, so the two gathers differ only in which surface they find
// and in how the sun's visibility is answered -- the cascades for the raster
// capture, a shadow ray for the traced one. See docs/irradiance_probes.md.
//
// Deliberately a fraction of simple_bindless.frag. What a probe contributes is
// low-frequency diffuse light gathered over 8x8 directions, so the parts of the
// full shading model that describe view-dependent detail cannot survive the
// convolution and are not worth evaluating here:
//
//   kept     albedo, lambert from the directional light and from every punctual
//            light, punctual shadows, ambient, emissive, the grid's own
//            irradiance for multi-bounce
//   dropped  specular of every kind, normal mapping, IBL
//
// Punctual lights are read as a flat array rather than through the per-froxel
// cluster lists. Those lists are built for the camera's froxel grid, so there is
// no froxel to look a probe up in -- a probe is not on screen. Looping every
// light instead is affordable precisely because the capture is tiny.
//
// The includer declares, before including this file, with
// GL_EXT_buffer_reference enabled and irradiance_probes.glsl included:
//   uPunctualShadowAtlas    sampler2D, set 0 binding 7;
//   uProbeIrradianceAtlas   sampler2D, set 0 binding 9;
//   uProbeDepthAtlas        sampler2D, set 0 binding 10;
//   probeParams             the ProbeShadingParams block, set 0 binding 11.

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
// Explicit LOD: every fetch here sits behind a per-surface early-out or inside a
// loop that continues per surface, which is non-uniform control flow, where
// implicit-LOD derivatives are undefined -- and a compute shader has none at all.
//
// The negative-slot sentinel guards the buffer dereference exactly as it does in
// the main pass -- a light with a valid slot implies a valid slot buffer.
float capturePunctualShadow(ShadowSlotBuffer shadowSlots, GpuLight light, vec3 worldPosition, vec3 normal)
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
    float normalBias = shadowSlots.slots[baseSlotIndex].params.y;
    vec3 biasedPosition = worldPosition + normal * (normalBias * (0.2 + grazing));

    uint slotIndex = baseSlotIndex;
    if (isPoint) {
        slotIndex += pointShadowFaceIndex(biasedPosition - light.positionRange.xyz);
    }

    GpuShadowSlot slot = shadowSlots.slots[slotIndex];
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
    float closestDepth = textureLod(uPunctualShadowAtlas, atlasUv, 0.0).r;
    return currentDepth <= closestDepth ? 1.0 : 0.0;
}

// Diffuse-only contribution of one punctual light.
//
// No specular, for the same reason the directional term has none: it cannot
// survive a convolution over 8x8 directions. The attenuation and cone shaping
// are the main pass's, so a light reaches exactly as far here as it does on
// screen; the normalisation is the capture's own unnormalised lambert, which
// keeps the punctual and directional terms consistent with each other.
vec3 capturePunctualLight(ShadowSlotBuffer shadowSlots, GpuLight light, vec3 worldPosition, vec3 normal)
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

    // Sampled only after the cheap rejections, so a surface outside a light's
    // cone or range never pays for the atlas fetch. With every light looped for
    // every captured surface, that gate is what keeps the cost bounded.
    attenuation *= capturePunctualShadow(shadowSlots, light, worldPosition, normal);
    if (attenuation <= 0.0) {
        return vec3(0.0);
    }

    return light.colorIntensity.rgb * light.colorIntensity.a * attenuation;
}

// --- Probe lookup, for multi-bounce -----------------------------------------
//
// Transcribed from simple_bindless.frag; the constants and the weighting are the
// same ones, because a capture that gathered indirect light differently from the
// way the camera gathers it would make probes disagree with the shading they
// feed.

vec2 probeOctEncode(vec3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return e * 0.5 + 0.5;
}

vec2 probeAtlasUv(uint probeIndex, vec3 direction, int coreResolution)
{
    int tileSize = coreResolution + 2 * kProbeBorderTexels;
    ivec2 tile = ivec2(int(probeIndex) % kProbeAtlasTilesX, int(probeIndex) / kProbeAtlasTilesX);
    vec2 texel = vec2(tile * tileSize) + float(kProbeBorderTexels) +
                 probeOctEncode(direction) * float(coreResolution);
    return texel / vec2(kProbeAtlasTilesX * tileSize, kProbeAtlasTilesY * tileSize);
}

float probeChebyshevVisibility(vec2 moments, float distanceToProbe)
{
    if (!(distanceToProbe > moments.x)) {
        return 1.0;
    }
    float variance = abs(moments.x * moments.x - moments.y);
    float excess = distanceToProbe - moments.x;
    float bound = variance / (variance + excess * excess);
    return clamp(bound * bound * bound, kProbeMinVisibility, 1.0);
}

vec3 sampleProbeIrradiance(vec3 worldPosition, vec3 normal, vec3 viewDirection, out float coverage)
{
    vec3 spacing = max(probeParams.gridSpacing.xyz, vec3(1.0e-4));
    vec3 biased = worldPosition + (normal * 0.2 + viewDirection * 0.8) * max(probeParams.gridSpacing.w, 0.0);

    vec3 gridSpace = (biased - probeParams.gridOrigin.xyz) / spacing;
    vec3 floored = floor(gridSpace);
    ivec3 baseCoord = clamp(ivec3(floored), ivec3(0),
                            ivec3(kProbeGridX - 1u, kProbeGridY - 1u, kProbeGridZ - 1u));
    vec3 fraction = clamp(gridSpace - floored, vec3(0.0), vec3(1.0));

    vec3 total = vec3(0.0);
    float totalWeight = 0.0;

    for (int corner = 0; corner < 8; ++corner) {
        ivec3 offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        ivec3 coord = min(baseCoord + offset,
                          ivec3(kProbeGridX - 1u, kProbeGridY - 1u, kProbeGridZ - 1u));
        uint probeIndex = uint(coord.x) + uint(coord.y) * kProbeGridX +
                          uint(coord.z) * kProbeGridX * kProbeGridY;

        vec3 probePosition = probeParams.gridOrigin.xyz + vec3(coord) * spacing;
        vec3 toProbe = probePosition - biased;
        float distanceToProbe = length(toProbe);
        vec3 directionToProbe = distanceToProbe > 1.0e-6 ? toProbe / distanceToProbe : normal;

        vec3 axisWeight = mix(1.0 - fraction, fraction, vec3(offset));
        float weight = axisWeight.x * axisWeight.y * axisWeight.z;

        float wrapped = (dot(normal, directionToProbe) + 1.0) * 0.5;
        weight *= wrapped * wrapped + kProbeBackfaceFloor;

        vec2 moments = textureLod(uProbeDepthAtlas,
                                  probeAtlasUv(probeIndex, -directionToProbe, kProbeDepthResolution), 0.0).rg;
        weight *= probeChebyshevVisibility(moments, distanceToProbe);

        if (weight <= 0.0) {
            continue;
        }

        // Alpha is the probe's state; see simple_bindless.frag.
        vec4 probeTexel = textureLod(uProbeIrradianceAtlas,
                                     probeAtlasUv(probeIndex, normal, kProbeIrradianceResolution), 0.0);
        weight *= probeTexel.a;
        total += probeTexel.rgb * weight;
        totalWeight += weight;
    }

    coverage = totalWeight > 0.0 ? 1.0 : 0.0;
    return totalWeight > 0.0 ? total / totalWeight : vec3(0.0);
}

// --- The surface a probe sees ------------------------------------------------

// Everything a gather needs about the lights, gathered so the two callers pass
// one value rather than a dozen.
struct ProbeCaptureLighting {
    // FrameConstants::lightDirection: the direction the sun's light travels.
    vec3 sunDirection;
    vec3 sunColor;
    vec3 ambient;
    LightBuffer lights;
    uint lightCount;
    ShadowSlotBuffer shadowSlots;
    // How much of the surface's indirect light comes from the probe grid rather
    // than the constant ambient; see probe_capture.frag's push block.
    float bounceWeight;
};

// Outgoing radiance of a surface towards the probe. `normal` already faces the
// probe -- both gathers are two-sided -- and sunVisibility is however the caller
// answers whether the sun reaches worldPosition.
vec3 probeCaptureRadiance(ProbeCaptureLighting lighting,
                          vec3 albedo,
                          vec3 emissive,
                          vec3 worldPosition,
                          vec3 normal,
                          vec3 probePosition,
                          float sunVisibility)
{
    vec3 towardSun = normalize(-lighting.sunDirection);
    float normalLight = max(dot(normal, towardSun), 0.0);
    vec3 direct = lighting.sunColor * normalLight * sunVisibility;

    // Every punctual light, not a culled subset. See the header: the cluster
    // lists cannot answer for a position that is not on screen.
    for (uint lightIndex = 0u; lightIndex < lighting.lightCount; ++lightIndex) {
        direct += capturePunctualLight(lighting.shadowSlots, lighting.lights.lights[lightIndex], worldPosition, normal);
    }

    // Indirect light on this surface. Interpolated between the constant ambient
    // and what the grid already holds rather than summed, because both answer
    // the same question -- summing them would count the sky twice, and it would
    // also make the feedback unbounded instead of a convex combination.
    //
    // The weight is what bounds the loop: each round of feedback multiplies by
    // albedo * weight, so the series settles at 1 / (1 - albedo * weight). See
    // ve::renderer::probeBounceAmplification.
    vec3 indirect = lighting.ambient;
    if (lighting.bounceWeight > 0.0) {
        vec3 toProbeGrid = normalize(probePosition - worldPosition);
        float coverage;
        vec3 gridIrradiance = sampleProbeIrradiance(worldPosition, normal, toProbeGrid, coverage);
        // No usable probe around the surface: it keeps the plain ambient.
        indirect = mix(lighting.ambient, gridIrradiance, clamp(lighting.bounceWeight, 0.0, 1.0) * coverage);
    }

    return albedo * (direct + indirect) + emissive;
}

#endif // VE_PROBE_CAPTURE_SHADING_GLSL
