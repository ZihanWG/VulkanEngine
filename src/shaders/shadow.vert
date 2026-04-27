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
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
} pc;

layout(location = 0) in vec3 inPosition;

void main()
{
    gl_Position = pc.objectFrameData.lightMvp * vec4(inPosition, 1.0);
}
