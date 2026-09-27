#version 460

#extension GL_EXT_buffer_reference : require

// Depth-only stage for the opaque prepass. Mirrors shadow.vert, with the camera's
// jittered view-projection in place of a cascade's.
//
// That matrix is load-bearing: the main pass tests LESS_OR_EQUAL against the
// depth this writes, so the two stages must transform a vertex identically. The
// expression below is written to match simple.vert's -- model first, then
// jitteredViewProjection -- rather than folding them into one matrix, because a
// different association order is a different rounding and would leave surfaces
// failing their own depth test.
//
// TAA jitter changes that matrix every frame, which is harmless here: both passes
// in one frame read the same FrameConstants.

#include "lod_transition.glsl"
#include "object_frame_data.glsl"
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
    // Explicit offset: this block declares only the leading fields, so the
    // address has to land where ve::PushConstants actually keeps it.
    layout(offset = 112) FrameConstantsBuffer frameConstants;
} pc;

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp). The prepass replays
// the main pass's compacted draw buffer, which is packed, so the mask is
// required rather than defensive.
const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;

void main()
{
    // Half of a LOD cross-fade: leave it out. Its fragments are dithered in the
    // main pass, and a full-coverage depth written here would let the main pass
    // keep neither half where they disagree -- the pixel would hold this
    // geometry's depth and nobody's colour. Every vertex lands at the same
    // point outside the clip volume, so the triangles are culled before
    // rasterization. The main pass writes these draws' depth itself.
    if (((gl_InstanceIndex >> 16) & kLodInstanceFading) != 0u) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];
    gl_Position = pc.frameConstants.values.jitteredViewProjection * (objectData.model * vec4(inPosition, 1.0));
}
