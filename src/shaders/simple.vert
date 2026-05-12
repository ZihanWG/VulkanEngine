#version 460

#extension GL_EXT_buffer_reference : require

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
};

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    ObjectFrameData objects[];
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
} pc;

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

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex];

    vec4 worldPosition = objectData.model * vec4(inPosition, 1.0);
    gl_Position = objectData.mvp * vec4(inPosition, 1.0);
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
    vLightDirection = objectData.lightDirection.xyz;
    vLightColor = objectData.lightColor.xyz;
    vAmbientColor = objectData.ambientColor.xyz;
    for (uint cascade = 0; cascade < 4; ++cascade) {
        vLightSpacePosition[cascade] = objectData.lightMvp[cascade] * vec4(inPosition, 1.0);
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
}
