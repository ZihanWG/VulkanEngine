#version 460

#extension GL_EXT_buffer_reference : require

// Depth-only vertex stage for rendering one rect with its own projection.
//
// Two passes use it, because they are the same operation: the punctual
// (spot/point) shadow atlas draws one tile per light slot, and the virtual
// shadow map page pass draws one page per clipmap slot. Both open a single
// rendering scope, set a viewport and scissor per rect, and push that rect's
// view-projection.
//
// The CSM path precomputes a per-object light MVP for each of its four cascades
// and indexes that array (shadow.vert). That does not scale here: the atlas
// renders up to 64 slots per frame, so storing a per-object matrix per slot
// would blow up ObjectFrameData. Instead the slot's view-projection arrives as
// a push constant -- one push per slot, not per slot per object -- and is
// combined with the object's model matrix on the fly.

#include "object_frame_data.glsl"
// The mat4 sits at offset 16, not 8: push-constant layout rules round the
// 8-byte buffer reference up to the matrix's 16-byte alignment. The C++
// PunctualShadowPushConstants mirror carries explicit padding to match.
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    mat4 lightViewProjection;
} pc;

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp). The direct-draw
// path this pass uses offsets the buffer address per draw instead and leaves
// the instance index at zero, so the mask is a no-op there but keeps the
// shader correct if it is ever fed indirect draws.
const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];
    gl_Position = pc.lightViewProjection * objectData.model * vec4(inPosition, 1.0);
}
