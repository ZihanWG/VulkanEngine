#version 460

// Layered variant: one multiview pass renders every cascade at once, and
// gl_ViewIndex says which. Otherwise identical to shadow.vert.
#extension GL_EXT_multiview : require
#define VE_SHADOW_MULTIVIEW 1

#extension GL_EXT_buffer_reference : require

#include "object_frame_data.glsl"
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
    // Explicit offset: this block declares only the leading fields, so the
    // address has to land where ve::PushConstants actually keeps it.
    layout(offset = 112) FrameConstantsBuffer frameConstants;
} pc;

#include "shadow_cascade.glsl"

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp). Draws that do not go
// through the cull pass leave the high bits zero, so the mask is always safe.
const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];
    uint cascade = veShadowCascadeIndex();
    // The model transform is applied here rather than folded into a per-object
    // matrix on the CPU. That trades one extra mat4 multiply per vertex for 256
    // bytes off every object record.
    gl_Position = pc.frameConstants.values.cascadeViewProjection[cascade] *
                  (objectData.model * vec4(inPosition, 1.0));
}
