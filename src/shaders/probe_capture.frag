#version 460

#extension GL_EXT_nonuniform_qualifier : require

// Fragment stage for irradiance-probe capture. Writes what a probe sees along
// one direction: outgoing radiance in rgb, distance to the surface in a.
//
// Deliberately a fraction of simple_bindless.frag. What a probe contributes is
// low-frequency diffuse light gathered over 8x8 directions, so the parts of the
// full shading model that describe view-dependent detail cannot survive the
// convolution and are not worth evaluating here:
//
//   kept     albedo, lambert from the directional light, cascaded shadows,
//            ambient, emissive
//   dropped  specular of every kind, normal mapping, punctual lights, IBL
//
// Dropping punctual lights is the one that is a real approximation rather than
// a free simplification -- a room lit only by spot lights gathers no indirect
// light at all here. It needs the cluster light lists, which are built for the
// camera's froxel grid and do not describe a probe's position, so it is a
// separate piece of work rather than a line to add.

layout(set = 0, binding = 1) uniform sampler2DArray uShadowMap;
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

layout(push_constant) uniform PushConstants {
    // Stands in for the vertex stage's buffer reference, which this stage does
    // not read. A reference and a uvec2 are both 8 bytes with 8-byte alignment,
    // so the members after it land at the same offsets in both stages -- which
    // is what has to hold, since one push fills the range for both.
    uvec2 objectFrameData;
    mat4 faceViewProjection;
    vec4 probePosition;
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
