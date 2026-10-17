// Shared by the soft shadow filter's two shaders, ray_traced_shadow_radius.comp
// and ray_traced_shadow_blur.comp: the bindings, the push block and what counts
// as a texel the filter may use. Both read one RGBA16F image and write the
// other -- the trace image and the filter image, in turn -- with the trace's
// texel layout: x = visibility (with the not-traced and back-facing encodings),
// w = camera distance, and y and z the trace's hit distance until the radius
// pass replaces them with the penumbra radius along screen y and x. See
// docs/ray_traced_shadows.md#the-filter.
//
// The includer includes ray_traced_shadows.glsl first.

#ifndef VE_RAY_TRACED_SHADOW_FILTER_GLSL
#define VE_RAY_TRACED_SHADOW_FILTER_GLSL

layout(local_size_x = kRtMaskLocalSize, local_size_y = kRtMaskLocalSize, local_size_z = 1) in;

// Read with texelFetch through the mask's nearest sampler.
layout(set = 0, binding = 0) uniform sampler2D uFilterSource;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D uFilterDestination;

// Mirrors ve::renderer::RayTracedShadowFilterPushConstants.
layout(push_constant) uniform FilterPushConstants {
    // The trace grid both images hold this frame: rayTracedShadowTraceExtent of
    // the render sub-rect at the trace scale.
    uvec2 traceExtent;
    // The blur's axis, (1, 0) or (0, 1). Unused by the radius pass.
    ivec2 direction;
    // Trace texels per world unit at one unit from the camera: the projection's
    // vertical scale times half the sub-rect's height, over the trace scale.
    float texelsPerUnit;
    // tan(half the sun's angular diameter).
    float tanSunRadius;
    // Radius pass only: 1 when the temporal pass ran, so the visibility to
    // filter is the accumulated mean in uHistory rather than this frame's ray.
    uint temporal;
    // Radius pass only: 1 to put each texel's history length in place of its
    // visibility, unblurred -- the history-length debug view.
    uint debugHistoryLength;
} pc;

// A texel the filter may read: traced this frame, from a surface facing the sun.
// A texel traced from a surface facing away answers a question the fragment may
// not be asking (see kRtMaskBackFacingBias), so it is neither blurred nor
// blurred into anything.
bool rtFilterUsable(vec4 texel)
{
    return texel.x >= 0.0 && texel.x < kRtMaskBackFacingBias;
}

bool rtFilterInside(ivec2 texel)
{
    return all(greaterThanEqual(texel, ivec2(0))) && all(lessThan(texel, ivec2(pc.traceExtent)));
}

#endif // VE_RAY_TRACED_SHADOW_FILTER_GLSL
