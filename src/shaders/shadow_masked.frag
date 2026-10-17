#version 460

#extension GL_EXT_nonuniform_qualifier : require

// Cutout shadow casters. Without this the shadow map is depth-only and a MASK
// material throws a solid silhouette — the classic "leaves cast a rectangle"
// artifact. Only the bindless base-color array is needed, so it is bound as the
// single descriptor set of this pipeline (set 0) rather than at set 1 like the
// main pass.
layout(set = 0, binding = 0) uniform sampler2D uBaseColorTextures[];

layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vBaseColorTextureIndex;
layout(location = 2) flat in float vAlphaCutoff;
layout(location = 3) flat in float vBaseColorAlpha;

#ifdef VE_LOD_FADES_ONLY
// depth_prepass_fade.frag: the prepass's replay of LOD cross-fades. The same
// keep test simple_bindless.frag runs first, on the same per-pixel noise, so the
// depth written here belongs to whichever half that pixel shows. OPAQUE and
// BLEND carry a negative cutoff, so the test below never clips them.
#include "lod_transition.glsl"
layout(location = 4) flat in uint vLodIndex;
#endif

void main()
{
#ifdef VE_LOD_FADES_ONLY
    if (!lodDitherKeeps((vLodIndex & kLodInstanceOutgoing) != 0u,
                        lodDitherNoise(gl_FragCoord.xy),
                        (vLodIndex >> kLodInstanceFadeShift) & kLodFadeSteps)) {
        discard;
    }
#endif
    float alpha = texture(uBaseColorTextures[nonuniformEXT(vBaseColorTextureIndex)], vUV).a * vBaseColorAlpha;
    if (alpha < vAlphaCutoff) {
        discard;
    }
}
