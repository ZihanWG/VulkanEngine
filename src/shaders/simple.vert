#version 460

#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430) readonly buffer FrameDataBuffer {
    mat4 mvp;
};

layout(push_constant) uniform PushConstants {
    FrameDataBuffer frameData;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;

void main()
{
    gl_Position = pc.frameData.mvp * vec4(inPosition, 1.0);
    vColor = inColor;
}
