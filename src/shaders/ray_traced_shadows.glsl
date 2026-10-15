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

#endif // VE_RAY_TRACED_SHADOWS_GLSL
