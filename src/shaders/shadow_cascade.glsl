#ifndef VE_SHADOW_CASCADE_GLSL
#define VE_SHADOW_CASCADE_GLSL

// Which cascade the current invocation is rendering.
//
// Two ways in, one source. The layered variants render every cascade in a single
// multiview pass and read gl_ViewIndex; the fallback variants render one cascade
// per pass and read the index the renderer pushed. Defining VE_SHADOW_MULTIVIEW
// before including this block selects the first. The distinction cannot be a
// runtime branch: gl_ViewIndex needs the MultiView SPIR-V capability, and a
// module declaring it is not valid in a pipeline built without a view mask.
#ifdef VE_SHADOW_MULTIVIEW
uint veShadowCascadeIndex()
{
    return min(uint(gl_ViewIndex), 3u);
}
#else
uint veShadowCascadeIndex()
{
    return min(pc.cascadeIndex, 3u);
}
#endif

#endif // VE_SHADOW_CASCADE_GLSL
