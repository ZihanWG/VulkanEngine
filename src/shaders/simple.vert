#version 460

#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    mat4 mvp;
    mat4 model;
    mat4 lightMvp;
    vec4 lightDirection;
    vec4 lightColor;
    vec4 ambientColor;
    vec4 shadowSettings;
    vec4 baseColorFactor;
    vec4 materialParams;
    vec4 cameraPosition;
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;
layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec3 vLightDirection;
layout(location = 4) out vec3 vLightColor;
layout(location = 5) out vec3 vAmbientColor;
layout(location = 6) out vec4 vLightSpacePosition;
layout(location = 7) flat out vec4 vShadowSettings;
layout(location = 8) out vec3 vWorldPosition;
layout(location = 9) flat out vec3 vCameraPosition;
layout(location = 10) flat out vec4 vBaseColorFactor;
layout(location = 11) flat out vec4 vMaterialParams;
layout(location = 12) out vec3 vTangent;
layout(location = 13) out vec3 vBitangent;

void main()
{
    vec4 worldPosition = pc.objectFrameData.model * vec4(inPosition, 1.0);
    gl_Position = pc.objectFrameData.mvp * vec4(inPosition, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(pc.objectFrameData.model)));
    mat3 modelMatrix = mat3(pc.objectFrameData.model);
    vec3 normalWS = normalize(normalMatrix * inNormal);
    vec3 tangentWS = normalize(modelMatrix * inTangent.xyz);
    tangentWS = normalize(tangentWS - normalWS * dot(normalWS, tangentWS));
    vec3 bitangentWS = normalize(cross(normalWS, tangentWS) * inTangent.w);

    vColor = inColor;
    vUV = inUV;
    vNormal = normalWS;
    vTangent = tangentWS;
    vBitangent = bitangentWS;
    vLightDirection = pc.objectFrameData.lightDirection.xyz;
    vLightColor = pc.objectFrameData.lightColor.xyz;
    vAmbientColor = pc.objectFrameData.ambientColor.xyz;
    vLightSpacePosition = pc.objectFrameData.lightMvp * vec4(inPosition, 1.0);
    vShadowSettings = pc.objectFrameData.shadowSettings;
    vWorldPosition = worldPosition.xyz;
    vCameraPosition = pc.objectFrameData.cameraPosition.xyz;
    vBaseColorFactor = pc.objectFrameData.baseColorFactor;
    vMaterialParams = pc.objectFrameData.materialParams;
}
