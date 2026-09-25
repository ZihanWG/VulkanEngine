// Joint-matrix palette layout shared by every skinning vertex stage, mirrored
// from ve::renderer::kMaxSkinJoints and kSkinPreviousPaletteOffset in
// src/renderer/SkeletalAnimation.h. tools/check_shader_constants.py compares
// the two copies.
//
// Each per-frame palette buffer holds this frame's joint matrices at [0,
// kMaxSkinJoints) and the previous frame's at kSkinPreviousPaletteOffset. Only
// the main pass reads the second half, to give the velocity buffer joint-space
// motion; the shadow stages draw the current pose and never look past it.
//
// Requires GL_EXT_buffer_reference, which every includer already enables for
// object_frame_data.glsl.

#ifndef VE_JOINT_PALETTE_GLSL
#define VE_JOINT_PALETTE_GLSL

const uint kMaxSkinJoints = 64u;
const uint kSkinPreviousPaletteOffset = kMaxSkinJoints;

layout(buffer_reference, std430) readonly buffer JointPalette {
    mat4 jointMatrices[];
};

#endif // VE_JOINT_PALETTE_GLSL
