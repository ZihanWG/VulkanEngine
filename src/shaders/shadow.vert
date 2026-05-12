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

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex];
    uint cascade = min(pc.cascadeIndex, 3u);
    gl_Position = objectData.lightMvp[cascade] * vec4(inPosition, 1.0);
}
