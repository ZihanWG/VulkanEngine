#version 460

#extension GL_EXT_buffer_reference : require

// Depth-only vertex stage for a SKINNED caster, rendering one rect with its own
// projection.
//
// It exists because the skinned mesh cannot ride the paths the other casters
// use. Those are indirect draws over one shared vertex buffer, batched by mesh
// and indexed through gl_InstanceIndex; the skinned mesh has a second vertex
// binding the others do not have (joint indices and weights) and a per-frame
// palette the others do not read, so it cannot join a batch. It is drawn
// directly instead, after the indirect draws, once per rect it overlaps.
//
// The projection arrives as a push constant exactly as it does in
// shadow_punctual.vert, and for the same reason: a cascade, an atlas tile and a
// clipmap page are all "one rect with one view-projection" as far as a caster is
// concerned. Sharing that shape is what lets one shader serve all three, instead
// of the cascades' own shadow.vert convention of indexing a per-object matrix
// array -- which would need a fourth pipeline and a fourth push layout.

#include "object_frame_data.glsl"

layout(buffer_reference, std430) readonly buffer JointPalette {
    mat4 jointMatrices[];
};

// The mat4 sits at offset 16, not 8: push-constant layout rounds the 8-byte
// buffer reference up to the matrix's 16-byte alignment. The palette reference
// then follows the matrix at 80. ve::SkinnedShadowPushConstants mirrors both
// with explicit padding, pinned by static_assert.
layout(push_constant) uniform PushConstants {
    ObjectFrameDataBuffer objectFrameData;
    mat4 lightViewProjection;
    JointPalette jointPalette;
} pc;

// The skinned mesh is drawn directly rather than through the cull pass, so it
// has no packed LOD level in the instance index and no batch to index. Its
// object-data slot is pushed by offsetting the buffer address per draw, the same
// way the punctual and page passes do it, which leaves the instance index zero.
const uint kObjectSlot = 0u;

layout(location = 0) in vec3 inPosition;
// Locations 1-4 (colour, uv, normal, tangent) are declared by the geometry
// binding but not consumed here: this is a depth-only pass. Binding 1 supplies
// 5 and 6, which are the whole reason this shader is separate.
layout(location = 5) in uvec4 inJointIndices;
layout(location = 6) in vec4 inJointWeights;

void main()
{
    ObjectFrameData objectData = pc.objectFrameData.objects[kObjectSlot];

    // Linear blend skinning, identical to simple_skinned.vert. It has to be:
    // a caster that deforms differently from the surface it is casting for
    // produces a shadow that is subtly detached from its own geometry, and
    // nothing about the resulting image says which of the two shaders is wrong.
    vec4 skinned = vec4(0.0);
    for (int influence = 0; influence < 4; ++influence) {
        float weight = inJointWeights[influence];
        if (weight <= 0.0) {
            continue;
        }
        skinned += weight * (pc.jointPalette.jointMatrices[inJointIndices[influence]] * vec4(inPosition, 1.0));
    }

    gl_Position = pc.lightViewProjection * objectData.model * skinned;
}
