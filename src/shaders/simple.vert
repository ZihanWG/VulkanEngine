#version 460

#extension GL_EXT_buffer_reference : require

#include "object_frame_data.glsl"
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
    // Explicit offset: this block declares only the leading fields, so the
    // address has to be placed where ve::PushConstants actually keeps it.
    layout(offset = 112) FrameConstantsBuffer frameConstants;
} pc;

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp). Draws that do not go
// through the cull pass leave the high bits zero, so the mask is always safe.
const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;
layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vLightDirection;
layout(location = 3) out vec3 vLightColor;
layout(location = 4) out vec3 vAmbientColor;
layout(location = 5) out vec4 vLightSpacePosition[4];
layout(location = 9) flat out vec4 vShadowSettings;
layout(location = 10) out vec3 vWorldPosition;
layout(location = 11) flat out vec3 vCameraPosition;
layout(location = 12) flat out vec4 vBaseColorFactor;
layout(location = 13) flat out vec4 vMaterialParams;
layout(location = 14) out vec3 vTangent;
layout(location = 15) out vec3 vBitangent;
layout(location = 16) flat out uvec4 vTextureIndices;
layout(location = 17) out float vViewDepth;
layout(location = 18) flat out vec4 vCascadeSplits;
layout(location = 19) flat out uint vCascadeCount;
layout(location = 20) flat out float vCascadeDebugEnabled;
layout(location = 21) flat out vec4 vEmissiveFactor;
layout(location = 22) out vec4 vCurrClipPos;
layout(location = 23) out vec4 vPrevClipPos;
layout(location = 24) flat out uint vLodIndex;
// x = normal-offset bias (already consumed above), y = cascade blend band. The
// fragment stage needs y to know how wide the cross-fade at each split is.
layout(location = 25) flat out vec4 vShadowQuality;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];
    vLodIndex = gl_InstanceIndex >> 16;

    vec4 worldPosition = objectData.model * vec4(inPosition, 1.0);
    gl_Position = pc.frameConstants.values.jitteredViewProjection * worldPosition;
    mat3 normalMatrix = transpose(inverse(mat3(objectData.model)));
    mat3 modelMatrix = mat3(objectData.model);
    vec3 normalWS = normalize(normalMatrix * inNormal);
    vec3 tangentWS = normalize(modelMatrix * inTangent.xyz);
    tangentWS = normalize(tangentWS - normalWS * dot(normalWS, tangentWS));
    vec3 bitangentWS = normalize(cross(normalWS, tangentWS) * inTangent.w);

    vUV = inUV;
    vNormal = normalWS;
    vTangent = tangentWS;
    vBitangent = bitangentWS;
    vLightDirection = pc.frameConstants.values.lightDirection.xyz;
    vLightColor = pc.frameConstants.values.lightColor.xyz;
    vAmbientColor = pc.frameConstants.values.ambientColor.xyz;
    // Normal-offset shadow bias, applied here rather than in the fragment shader.
    //
    // Doing it per-cascade at this point costs four vector ops per vertex and
    // nothing per pixel, and it gets the scaling right for free: the length of a
    // cascade matrix's first column is 2/orthoWidth, so its reciprocal is that
    // cascade's world half-extent. One unitless bias therefore produces a small
    // offset in a tight near cascade and a proportionally larger one in a far
    // cascade, which is what a single world-space number could never do.
    //
    // The interpolated vertex normal is the right normal to use: offsetting along
    // the normal-mapped one would push the lookup around by texture detail that
    // casts no shadow.
    const float normalBias = pc.frameConstants.values.shadowQuality.x;
    const vec3 lightToSurface = normalize(-pc.frameConstants.values.lightDirection.xyz);
    const float grazing = 1.0 - max(dot(normalWS, lightToSurface), 0.0);

    for (uint cascade = 0; cascade < 4; ++cascade) {
        vec4 shadowWorldPosition = worldPosition;
        if (normalBias > 0.0) {
            const float columnScale = length(pc.frameConstants.values.cascadeViewProjection[cascade][0].xyz);
            const float cascadeExtent = columnScale > 1e-6 ? 1.0 / columnScale : 0.0;
            // Keep a small head-on term so contact shadows do not detach, the
            // same shape the punctual path uses.
            shadowWorldPosition.xyz += normalWS * (normalBias * cascadeExtent * (0.2 + grazing));
        }
        vLightSpacePosition[cascade] =
            pc.frameConstants.values.cascadeViewProjection[cascade] * shadowWorldPosition;
    }
    vShadowSettings = pc.frameConstants.values.shadowSettings;
    vShadowQuality = pc.frameConstants.values.shadowQuality;
    vWorldPosition = worldPosition.xyz;
    vCameraPosition = pc.frameConstants.values.cameraPosition.xyz;
    vBaseColorFactor = objectData.baseColorFactor;
    vMaterialParams = objectData.materialParams;
    vTextureIndices = objectData.textureIndices;
    vViewDepth = dot(pc.frameConstants.values.cameraForward.xyz, worldPosition.xyz - pc.frameConstants.values.cameraPosition.xyz);
    vCascadeSplits = pc.frameConstants.values.cascadeSplits;
    vCascadeCount = uint(max(pc.frameConstants.values.cameraForward.w, 1.0) + 0.5);
    vCascadeDebugEnabled = pc.frameConstants.values.cameraPosition.w;
    vEmissiveFactor = objectData.emissiveFactor;
    vCurrClipPos = pc.frameConstants.values.viewProjection * worldPosition;
    vPrevClipPos = objectData.prevMvpNoJitter * vec4(inPosition, 1.0);
}
