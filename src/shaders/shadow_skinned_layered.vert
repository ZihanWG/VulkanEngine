#version 460

// Layered variant of shadow_skinned.vert: one multiview pass renders the
// skinned caster into every cascade at once, and gl_ViewIndex says which.
//
// It exists because the pushed-matrix convention shadow_skinned.vert shares with
// shadow_punctual.vert cannot answer gl_ViewIndex -- a push constant has one
// value for the whole draw, while multiview needs a different projection per
// layer. Without this shader the layered cascade path simply skipped the skinned
// caster, which silently dropped its shadow and, because the pose then also had
// to stay out of the cascade cache key, froze the whole shadow map (0/4 cascades
// redrawn, CSMShadowPass reading 0.000 ms). That failure presented as a large
// performance win, which is the worst way for it to present.
//
// So this file takes the cascades' own convention instead: index
// cascadeViewProjection[] out of the frame-constants buffer, exactly as
// shadow_layered.vert does for the non-skinned casters. Everything else --
// the skinning loop, the object slot, the vertex bindings -- is
// shadow_skinned.vert unchanged, and has to stay that way: a caster that
// deforms differently from the surface it casts for produces a shadow subtly
// detached from its own geometry.
#extension GL_EXT_multiview : require
#define VE_SHADOW_MULTIVIEW 1

#extension GL_EXT_buffer_reference : require

#include "object_frame_data.glsl"

layout(buffer_reference, std430) readonly buffer JointPalette {
    mat4 jointMatrices[];
};

// Three buffer references and nothing else. The non-layered variant's mat4 is
// gone -- that is the entire point of this shader -- so the layout is a plain
// run of 8-byte addresses with no alignment padding to mirror.
// ve::SkinnedLayeredShadowPushConstants pins it with a static_assert.
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    JointPalette jointPalette;
    FrameConstantsBuffer frameConstants;
} pc;

#include "shadow_cascade.glsl"

// As in shadow_skinned.vert: drawn directly rather than through the cull pass,
// so the object-data slot is pushed by offsetting the buffer address per draw
// and the instance index stays zero.
const uint kObjectSlot = 0u;

layout(location = 0) in vec3 inPosition;
// Locations 1-4 come from the geometry binding and are unused in a depth-only
// pass; binding 1 supplies 5 and 6.
layout(location = 5) in uvec4 inJointIndices;
layout(location = 6) in vec4 inJointWeights;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[kObjectSlot];

    vec4 skinned = vec4(0.0);
    for (int influence = 0; influence < 4; ++influence) {
        float weight = inJointWeights[influence];
        if (weight <= 0.0) {
            continue;
        }
        skinned += weight * (pc.jointPalette.jointMatrices[inJointIndices[influence]] * vec4(inPosition, 1.0));
    }

    uint cascade = veShadowCascadeIndex();
    gl_Position = pc.frameConstants.values.cascadeViewProjection[cascade] * (objectData.model * skinned);
}
