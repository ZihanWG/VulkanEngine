#version 460

#extension GL_EXT_buffer_reference : require

// Alpha-tested stage for the opaque prepass. Mirrors depth_prepass.vert but also
// forwards everything shadow_masked.frag's cutout test needs, which is the same
// set shadow_masked.vert forwards -- the two differ only in which matrix they
// project with.
//
// Without this variant a MASK material cannot be prepassed at all: a depth-only
// replay would write the depth of a cutout leaf's whole quad and the main pass
// would then reject everything behind it, which is the "leaves cast a rectangle"
// artifact moved out of the shadow map and into depth.
//
// The projection must match simple.vert's exactly, for the reason given in
// depth_prepass.vert: the main pass tests LESS_OR_EQUAL against what this writes.

#include "object_frame_data.glsl"
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
    // Explicit offset: this block declares only the leading fields, so the
    // address has to land where ve::PushConstants actually keeps it.
    layout(offset = 112) FrameConstantsBuffer frameConstants;
} pc;

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp).
const uint kObjectIndexMask = 0xFFFFu;

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vBaseColorTextureIndex;
layout(location = 2) flat out float vAlphaCutoff;
layout(location = 3) flat out float vBaseColorAlpha;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[gl_InstanceIndex & kObjectIndexMask];
    gl_Position = pc.frameConstants.values.jitteredViewProjection * (objectData.model * vec4(inPosition, 1.0));

    vUV = inUV;
    vBaseColorTextureIndex = objectData.textureIndices.x;
    vAlphaCutoff = objectData.materialParams.w;
    vBaseColorAlpha = objectData.baseColorFactor.a;
}
