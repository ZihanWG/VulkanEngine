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
// ray_traced_shadow_resolve.comp, read by simple_bindless.frag; on the CPU side
// only the debug readback reads the mask, which writes it as NaN.
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

// The inline fallback's soft shadow: this many strata per side of the unit
// square, one ray each, over the sun's disc. Only where the mask does not answer
// for a fragment, and only when the sun has an extent.
const uint kRtInlineSoftGridSide = 2u;

// Temporal accumulation, ray_traced_shadow_temporal.comp. The ceiling on the
// frames a history texel may average, and the clamp on
// RayTracedShadowSettings::temporalMaxFrames. Mirrors
// renderer::kRtTemporalMaxFrames.
const uint kRtTemporalMaxFrames = 64u;
// How far the view depth a history texel recorded may differ from the depth the
// reprojected surface had last frame, as a fraction of it, and still be that
// surface.
const float kRtTemporalDepthTolerance = 0.02;
// The neighbourhood the history is clamped to: the 3x3 trace texels around
// it whose camera distance is within this fraction of its own.
const float kRtTemporalNeighbourTolerance = 0.05;
// A surface moving m trace texels a frame keeps at most maxFrames / (1 + m *
// kRtTemporalMotionScale) frames of history, and never fewer than
// kRtTemporalMinMovingFrames.
const float kRtTemporalMotionScale = 2.0;
const float kRtTemporalMinMovingFrames = 1.0;
// The filter's radius shrinks with the history length n as n^-exponent.
const float kRtTemporalRadiusExponent = 0.25;

// The soft shadow filter, ray_traced_shadow_radius.comp and
// ray_traced_shadow_blur.comp. Most trace texels the blur reaches either side of
// a texel, and the clamp on the radius. Mirrors renderer::kRtFilterMaxRadius.
const float kRtFilterMaxRadius = 16.0;
// The radius pass's blocker search: every kRtFilterSearchStride-th texel within
// kRtFilterSearchRadius of the centre, 5x5 taps over a 9x9 window, counting the
// ones whose camera distance is within kRtFilterSearchTolerance of the centre's,
// as a fraction of it.
const int kRtFilterSearchRadius = 4;
const int kRtFilterSearchStride = 2;
const float kRtFilterSearchTolerance = 0.1;
// The blur's Gaussian: sigma is this fraction of the radius, so the taps, which
// stop at the radius, reach two sigma.
const float kRtFilterSigmaScale = 0.5;
// The narrowest Gaussian the blur takes wherever the radius pass found an
// occluder at all.
const float kRtFilterMinSigma = 0.5;
// How far a blur tap's reciprocal camera distance may stray from the centre's
// plane, as a fraction of the centre's, with weight falling off as
// exp(-difference / tolerance).
const float kRtFilterDistanceTolerance = 0.02;

// How much surface one trace texel spans along a screen axis, in world units:
// the width a texel subtends at the surface's camera distance, combined with how
// far the surface recedes from one texel to the next along that axis -- which
// is what makes a floor seen at a glancing angle span far more than its width.
// Mirrors renderer::rayTracedShadowWorldPerTexel.
float rtWorldPerTexel(float cameraDistance, float texelsPerUnit, float distanceStep)
{
    float across = cameraDistance / max(texelsPerUnit, 1.0e-6);
    return sqrt(across * across + distanceStep * distanceStep);
}

// Half the narrowest width the penumbra of an occluder `occluderDistance` along
// the sun ray can have on the receiver, 2 * distance * tan(sun radius) across,
// in trace texels along an axis whose texels each span worldPerTexel, clamped
// to kRtFilterMaxRadius. The narrowest, because the blur's error is not
// symmetric: a kernel narrower than the penumbra leaves noise, one wider bends
// the penumbra's ends towards its middle. A penumbra running along the sun's
// direction on a slanted receiver stretches by 1 / cosine, but the side of a
// long shadow does not, and only the side is everywhere. Mirrors
// renderer::rayTracedShadowPenumbraRadius.
float rtPenumbraRadius(float occluderDistance, float tanSunRadius, float worldPerTexel)
{
    if (!(occluderDistance > 0.0) || !(worldPerTexel > 0.0)) {
        return 0.0;
    }
    return min(occluderDistance * tanSunRadius / worldPerTexel, kRtFilterMaxRadius);
}

const float kRtHalfPi = 1.5707963;
const float kRtQuarterPi = 0.7853982;
// The golden angle, which spreads the reference mode's samples round the disc.
const float kRtGoldenAngle = 2.3999632;

#endif // VE_RAY_TRACED_SHADOWS_GLSL
