#version 460

layout(set = 0, binding = 0) uniform FrameData {
    mat4 mvp;
} uFrame;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 vColor;

void main()
{
    gl_Position = uFrame.mvp * vec4(inPosition, 1.0);
    vColor = inColor;
}
