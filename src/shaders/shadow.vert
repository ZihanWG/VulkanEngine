#version 460

#extension GL_EXT_buffer_reference : require

#include "object_frame_data.glsl"
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
} pc;

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp). Draws that do not go
// through the cull pass leave the high bits zero, so the mask is always safe.
const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];
    uint cascade = min(pc.cascadeIndex, 3u);
    gl_Position = objectData.lightMvp[cascade] * vec4(inPosition, 1.0);
}
