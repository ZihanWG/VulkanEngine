#version 460

#extension GL_EXT_buffer_reference : require

// Alpha-tested caster vertex stage for the virtual shadow map page pass.
//
// It is shadow_punctual.vert plus everything the cutout test needs forwarded to
// the fragment stage, exactly as shadow_masked.vert is shadow.vert plus the same
// four varyings. It cannot BE shadow_masked.vert: that one projects with
// frameConstants.cascadeViewProjection[cascade], and a page's projection is a
// per-page push, not a cascade index.
//
// Opaque casters keep using the depth-only path with no fragment shader at all,
// so these varyings and the discard are only paid by MASK materials.

#include "object_frame_data.glsl"

// Same block the opaque page pass pushes (ve::PunctualShadowPushConstants). The
// mat4 sits at offset 16, not 8: push-constant layout rules round the 8-byte
// buffer reference up to the matrix's 16-byte alignment.
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    mat4 lightViewProjection;
} pc;

// gl_InstanceIndex carries the object-data slot in its low 16 bits. The page
// cull writes the slot unpacked, but the mask costs nothing and keeps this
// correct if a level ever gets packed into the high bits the way cull.comp does.
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
    gl_Position = pc.lightViewProjection * objectData.model * vec4(inPosition, 1.0);

    vUV = inUV;
    vBaseColorTextureIndex = objectData.textureIndices.x;
    vAlphaCutoff = objectData.materialParams.w;
    vBaseColorAlpha = objectData.baseColorFactor.a;
}
