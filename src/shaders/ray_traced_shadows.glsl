// Layout constants shared by the ray-traced sun shadow shaders: the main
// fragment shader's alpha test, which reads mesh vertices by device address, and
// the pre-skin dispatch, which writes the positions the skinned BLAS is built
// from. Mirrors src/renderer/RayTracedShadows.h; tools/check_shader_constants.py
// compares the two copies, and the C++ side pins the vertex constants against
// renderer::Vertex with static_asserts. See docs/ray_traced_shadows.md.

#ifndef VE_RAY_TRACED_SHADOWS_GLSL
#define VE_RAY_TRACED_SHADOWS_GLSL

// renderer::Vertex as floats: its stride, and where its UV starts. Position is
// at float 0.
const uint kRtVertexStrideFloats = 15u;
const uint kRtVertexUvOffsetFloats = 6u;

// What the pre-skin dispatch writes per vertex: a bare position, the
// R32G32B32_SFLOAT stream the skinned BLAS is built and refitted from.
const uint kRtSkinnedPositionStrideFloats = 3u;

// Threads per pre-skin workgroup.
const uint kRtPreSkinLocalSize = 64u;

// Side of the ray-traced shadow mask's square workgroup.
const uint kRtMaskLocalSize = 8u;

// What a shadow-mask texel holds in place of a visibility when it was not
// traced: nothing covered the pixel, or its neighbours gave no surface to take
// a normal from. Any fragment reading it traces inline instead. Written by
// ray_traced_shadow_mask.comp, read by simple_bindless.frag; nothing on the CPU
// side reads the mask.
const float kRtMaskNotTraced = -1.0;

// Added to a shadow-mask visibility traced from a surface whose reconstructed
// normal faces away from the sun, so the reader can tell those texels apart.
// Such a ray starts on the sun's side of a surface that faces away from it,
// which is right for a normal-mapped wall past the sun's horizon and wrong at
// the terminator of a smooth-shaded surface, where the inline ray starts from
// the vertex normal instead. Only the fragment knows which of the two it is.
const float kRtMaskBackFacingBias = 2.0;

// Most rays a pixel traces in the reference mode (debugReferenceSamples), which
// averages that many stratified points on the sun's disc with no filter.
// Mirrors renderer::kRtReferenceMaxSamples, the setting's clamp.
const uint kRtReferenceMaxSamples = 64u;

// The resolve's depth test for a half-resolution sample: its camera distance may
// differ from the full-resolution pixel's by this fraction of that distance and
// still count as the same surface, with weight falling off as exp(-difference /
// tolerance). Where the samples around a pixel weigh less than kRtResolveMinWeight
// in total, none of them is about its surface and it traces inline instead.
const float kRtResolveDistanceTolerance = 0.05;
const float kRtResolveMinWeight = 0.05;

const float kRtHalfPi = 1.5707963;
const float kRtQuarterPi = 0.7853982;
// The golden angle, which spreads the reference mode's samples round the disc.
const float kRtGoldenAngle = 2.3999632;

#endif // VE_RAY_TRACED_SHADOWS_GLSL
