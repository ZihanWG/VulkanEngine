#version 460

#extension GL_EXT_buffer_reference : require

struct ObjectFrameData {
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
    uvec4 textureIndices;
};

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    ObjectFrameData objects[];
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
} pc;

layout(location = 0) in vec3 inPosition;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex];
    gl_Position = objectData.lightMvp * vec4(inPosition, 1.0);
}
