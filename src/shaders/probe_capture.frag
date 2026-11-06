#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require

// Fragment stage for irradiance-probe capture. Writes what a probe sees along
// one direction: outgoing radiance in rgb, distance to the surface in a.
//
// The shading itself is probe_capture_shading.glsl, shared with the ray-traced
// gather (probe_trace.comp). What stays here is what only the raster capture
// has: the interpolated surface, and the cascaded shadow map that answers the
// sun's visibility for it.

layout(set = 0, binding = 1) uniform sampler2DArray uShadowMap;
layout(set = 0, binding = 7) uniform sampler2D uPunctualShadowAtlas;
// The probe atlases this pass is itself feeding. Reading them here is what makes
// light bounce more than once: a surface a probe captures is lit partly by the
// indirect light the grid already holds, so the next capture sees light that has
// bounced one more time. Written by the update pass later in the same frame, so
// what this reads is the previous frame's contents.
layout(set = 0, binding = 9) uniform sampler2D uProbeIrradianceAtlas;
layout(set = 0, binding = 10) uniform sampler2D uProbeDepthAtlas;
#include "irradiance_probes.glsl"

layout(set = 0, binding = 11) uniform ProbeShadingParams {
    vec4 gridOrigin;   // xyz = origin, w = intensity
    vec4 gridSpacing;  // xyz = spacing, w = surface bias
    vec4 debug;
    // xyz = this probe's relocation offset from its grid point (zero unless
    // gi.probeRelocation moved it); written by the convolution.
    vec4 probeOffsets[kProbeCount];
} probeParams;
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

#include "probe_capture_shading.glsl"

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
    // How much of a captured surface's indirect light comes from the probe grid
    // rather than the constant ambient. Zero is single-bounce -- exactly the
    // behaviour before this existed, and the reference the result is judged
    // against. Held at zero by the CPU until the grid has been captured once,
    // because before that the probes hold their neutral seed and blending
    // toward it would make the whole scene darker, not brighter.
    float bounceWeight;
} pc;

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
    // Explicit LOD throughout this shader: every fetch here sits behind a
    // per-fragment early-out or inside a loop that continues per fragment, which
    // is non-uniform control flow, where implicit-LOD derivatives are undefined.
    // The main pass lost the device on an Intel UHD 770 over exactly this (see
    // punctualShadowFactor in simple_bindless.frag). Every image read here is
    // single-mip, so level 0 is what the implicit form resolved to anyway.
    float closestDepth = textureLod(uShadowMap, vec3(shadowUV, float(cascadeIndex)), 0.0).r;
    return shadowCoord.z - shadowDepthBias(normal) <= closestDepth ? 1.0 : 0.0;
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

    int cascadeIndex = selectShadowCascade();
    float shadowFactor = sampleShadowFactor(normal, cascadeIndex);

    ProbeCaptureLighting lighting;
    lighting.sunDirection = vLightDirection;
    lighting.sunColor = vLightColor;
    lighting.ambient = vAmbientColor;
    lighting.lights = pc.lights;
    lighting.lightCount = pc.lightCount;
    lighting.shadowSlots = pc.punctualShadowSlots;
    lighting.bounceWeight = pc.bounceWeight;

    // Emissive rides the base-color heap in slot w, the same arrangement the
    // main pass uses.
    vec3 emissive = vEmissiveFactor.rgb * texture(uBaseColorTextures[nonuniformEXT(vTextureIndices.w)], vUV).rgb;
    vec3 radiance =
        probeCaptureRadiance(lighting, baseColor.rgb, emissive, vWorldPosition, normal, pc.probePosition.xyz, shadowFactor);

    // Clamped for the same reason the depth atlas is: the convolution sums this
    // in a 16-bit float, and one blown-out emissive texel would otherwise
    // dominate a whole probe.
    float distance = min(length(toProbe), kProbeMaxDistance);

    outRadianceDistance = vec4(radiance, distance);
}
