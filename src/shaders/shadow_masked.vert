#version 460

#extension GL_EXT_buffer_reference : require

// Alpha-tested shadow-caster vertex stage. Mirrors shadow.vert but also forwards
// everything the cutout test needs to the fragment stage. Opaque casters keep
// using the cheaper depth-only shadow.vert (no fragment shader at all), so the
// extra varyings here are only paid by MASK materials.

#include "object_frame_data.glsl"
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    uint cascadeIndex;
} pc;

// gl_InstanceIndex packs the object-data slot in the low 16 bits and the
// cull-selected LOD level in the high bits (see cull.comp). The shadow cull
// dispatch packs too, so a cutout caster that selects any level above 0 would
// index far past the end of the object buffer without this mask.
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
    uint cascade = min(pc.cascadeIndex, 3u);
    gl_Position = objectData.lightMvp[cascade] * vec4(inPosition, 1.0);

    vUV = inUV;
    vBaseColorTextureIndex = objectData.textureIndices.x;
    vAlphaCutoff = objectData.materialParams.w;
    vBaseColorAlpha = objectData.baseColorFactor.a;
}
