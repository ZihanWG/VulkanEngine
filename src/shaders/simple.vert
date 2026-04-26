#version 460

#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430) readonly buffer ObjectFrameDataBuffer {
    mat4 mvp;
};

layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;
layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUV;

void main()
{
    gl_Position = pc.objectFrameData.mvp * vec4(inPosition, 1.0);
    vColor = inColor;
    vUV = inUV;
}
