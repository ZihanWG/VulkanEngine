#version 460

layout(push_constant) uniform SkyboxPushConstants {
    mat4 inverseViewProjection;
    mat4 prevViewProjection;
} pc;

layout(location = 0) out vec3 vDirection;
layout(location = 1) out vec2 vNdc;

void main()
{
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );

    vec2 ndc = positions[gl_VertexIndex];
    vec4 worldPosition = pc.inverseViewProjection * vec4(ndc, 1.0, 1.0);
    vDirection = worldPosition.xyz / worldPosition.w;
    vNdc = ndc;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
