#version 460

#extension GL_EXT_buffer_reference : require

// Skinned variant of simple.vert: applies a linear-blend skinning matrix to the
// vertex before the usual model/MVP transforms, then emits the identical varyings
// so it reuses the main fragment shader. The joint-matrix palette is delivered by
// buffer-device-address through the shared push constant (offset matches
// ve::PushConstants::jointMatricesAddress).

struct ObjectFrameData {
    mat4 mvp;
    mat4 model;
    mat4 lightMvp[4];
    vec4 lightDirection;
    vec4 lightColor;
    vec4 ambientColor;
    vec4 cascadeSplits;
    vec4 shadowSettings;
    vec4 baseColorFactor;
    vec4 materialParams;
    vec4 cameraPosition;
    vec4 cameraForward;
    uvec4 textureIndices;
    vec4 emissiveFactor;
    mat4 currMvpNoJitter;
    mat4 prevMvpNoJitter;
};

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    ObjectFrameData objects[];
};

layout(buffer_reference, std430) readonly buffer JointPalette {
    mat4 jointMatrices[];
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
    layout(offset = 72) JointPalette jointPalette;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;
layout(location = 5) in uvec4 inJointIndices;
layout(location = 6) in vec4 inJointWeights;

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

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex];

    // Linear-blend skinning: weighted sum of the bound joints' palette matrices.
    mat4 skinMatrix = inJointWeights.x * pc.jointPalette.jointMatrices[inJointIndices.x] +
                      inJointWeights.y * pc.jointPalette.jointMatrices[inJointIndices.y] +
                      inJointWeights.z * pc.jointPalette.jointMatrices[inJointIndices.z] +
                      inJointWeights.w * pc.jointPalette.jointMatrices[inJointIndices.w];

    vec4 skinnedPosition = skinMatrix * vec4(inPosition, 1.0);
    mat3 skinRotation = mat3(skinMatrix);
    vec3 skinnedNormal = skinRotation * inNormal;
    vec3 skinnedTangent = skinRotation * inTangent.xyz;

    vec4 worldPosition = objectData.model * skinnedPosition;
    gl_Position = objectData.mvp * skinnedPosition;
    mat3 normalMatrix = transpose(inverse(mat3(objectData.model)));
    mat3 modelMatrix = mat3(objectData.model);
    vec3 normalWS = normalize(normalMatrix * skinnedNormal);
    vec3 tangentWS = normalize(modelMatrix * skinnedTangent);
    tangentWS = normalize(tangentWS - normalWS * dot(normalWS, tangentWS));
    vec3 bitangentWS = normalize(cross(normalWS, tangentWS) * inTangent.w);

    vUV = inUV;
    vNormal = normalWS;
    vTangent = tangentWS;
    vBitangent = bitangentWS;
    vLightDirection = objectData.lightDirection.xyz;
    vLightColor = objectData.lightColor.xyz;
    vAmbientColor = objectData.ambientColor.xyz;
    for (uint cascade = 0; cascade < 4; ++cascade) {
        vLightSpacePosition[cascade] = objectData.lightMvp[cascade] * skinnedPosition;
    }
    vShadowSettings = objectData.shadowSettings;
    vWorldPosition = worldPosition.xyz;
    vCameraPosition = objectData.cameraPosition.xyz;
    vBaseColorFactor = objectData.baseColorFactor;
    vMaterialParams = objectData.materialParams;
    vTextureIndices = objectData.textureIndices;
    vViewDepth = dot(objectData.cameraForward.xyz, worldPosition.xyz - objectData.cameraPosition.xyz);
    vCascadeSplits = objectData.cascadeSplits;
    vCascadeCount = uint(max(objectData.cameraForward.w, 1.0) + 0.5);
    vCascadeDebugEnabled = objectData.cameraPosition.w;
    vEmissiveFactor = objectData.emissiveFactor;
    // Motion vectors reuse this frame's skinned position for both projections,
    // so they capture camera + rigid object motion but not joint-space motion
    // (that would need the previous frame's joint palette).
    vCurrClipPos = objectData.currMvpNoJitter * skinnedPosition;
    vPrevClipPos = objectData.prevMvpNoJitter * skinnedPosition;
}
