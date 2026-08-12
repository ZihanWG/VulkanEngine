#ifndef VE_SUB_RECT_GLSL
#define VE_SUB_RECT_GLSL

// Sampling targets that are allocated at the maximum render resolution but only
// written in their top-left sub-rect.
//
// Render scale changes the sub-rect, not the allocation, so a scale change costs
// a viewport instead of reallocating every target. The price is that every
// consumer has to know how much of its source was actually written.
//
// `uv` is normalised over the *written* region -- which is what a fullscreen
// pass's interpolated UV already is, since its own viewport is the sub-rect.
// `uvScale` is written/allocated for that specific source: derived targets
// (halves, mips) round each side independently, so their ratio drifts from the
// scene's and each one carries its own.
//
// The clamp is the part that matters. Past the written region lies whatever the
// last larger frame left behind: stale image data, which produces a
// plausible-looking smear along the right and bottom edges rather than anything
// a validation layer would flag. Half a texel is the margin a bilinear tap needs
// to keep all four of its samples inside.

vec2 veSubRectClamp(vec2 scaledUv, vec2 uvScale, vec2 allocatedSize)
{
    const vec2 halfTexel = 0.5 / allocatedSize;
    return clamp(scaledUv, halfTexel, uvScale - halfTexel);
}

// The common case: scale a written-region UV into the allocation and clamp it.
vec2 veSubRectUv(vec2 uv, vec2 uvScale, vec2 allocatedSize)
{
    return veSubRectClamp(uv * uvScale, uvScale, allocatedSize);
}

// For callers that add an offset already expressed in allocated-texel units --
// one texel is one texel whether or not the frame wrote it, so offsets are
// applied after the scale, never scaled themselves.
vec2 veSubRectUvOffset(vec2 uv, vec2 offset, vec2 uvScale, vec2 allocatedSize)
{
    return veSubRectClamp(uv * uvScale + offset, uvScale, allocatedSize);
}

#endif // VE_SUB_RECT_GLSL
