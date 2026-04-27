#version 460

#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    mat4 mvp;
    mat4 model;
    mat4 lightMvp;
    vec4 lightDirection;
    vec4 lightColor;
    vec4 ambientColor;
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inNormal;
layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vNormal;
layout(location = 3) out vec3 vLightDirection;
layout(location = 4) out vec3 vLightColor;
layout(location = 5) out vec3 vAmbientColor;
layout(location = 6) out vec4 vLightSpacePosition;

void main()
{
    gl_Position = pc.objectFrameData.mvp * vec4(inPosition, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(pc.objectFrameData.model)));

    vColor = inColor;
    vUV = inUV;
    vNormal = normalize(normalMatrix * inNormal);
    vLightDirection = pc.objectFrameData.lightDirection.xyz;
    vLightColor = pc.objectFrameData.lightColor.xyz;
    vAmbientColor = pc.objectFrameData.ambientColor.xyz;
    vLightSpacePosition = pc.objectFrameData.lightMvp * vec4(inPosition, 1.0);
}
