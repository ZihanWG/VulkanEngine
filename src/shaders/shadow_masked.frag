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

void main()
{
    float alpha = texture(uBaseColorTextures[nonuniformEXT(vBaseColorTextureIndex)], vUV).a * vBaseColorAlpha;
    if (alpha < vAlphaCutoff) {
        discard;
    }
}
