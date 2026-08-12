#version 460
#include "sub_rect.glsl"

layout(set = 0, binding = 0) uniform sampler2D uSource;

layout(push_constant) uniform BloomDownsamplePushConstants {
    vec2 texelSize;
    float threshold;
    uint applyThreshold;
    // Only level 0 reads the sub-rected scene colour; deeper levels read a mip
    // that was written in full and leave this at 1.
    vec2 sourceUvScale;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 thresholdBright(vec3 color)
{
    if (pc.applyThreshold == 0u) {
        return color;
    }

    float luminance = dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    return luminance > pc.threshold ? color : vec3(0.0);
}

// uv arrives normalised over the written region, with the tap offset already
// added in source-texel units; the helper scales and clamps it into the
// allocation. At level 0 that matters, deeper in the chain it is the identity.
vec3 sampleBloom(vec2 uv)
{
    const vec2 scaled = veSubRectClamp(uv * pc.sourceUvScale, pc.sourceUvScale, vec2(textureSize(uSource, 0)));
    return thresholdBright(texture(uSource, scaled).rgb);
}

void main()
{
    vec2 t = pc.texelSize / max(pc.sourceUvScale, vec2(1e-6));

    vec3 color = sampleBloom(vUV) * 0.25;
    color += sampleBloom(vUV + vec2(-t.x, -t.y)) * 0.125;
    color += sampleBloom(vUV + vec2( t.x, -t.y)) * 0.125;
    color += sampleBloom(vUV + vec2(-t.x,  t.y)) * 0.125;
    color += sampleBloom(vUV + vec2( t.x,  t.y)) * 0.125;
    color += sampleBloom(vUV + vec2(-2.0 * t.x, 0.0)) * 0.0625;
    color += sampleBloom(vUV + vec2( 2.0 * t.x, 0.0)) * 0.0625;
    color += sampleBloom(vUV + vec2(0.0, -2.0 * t.y)) * 0.0625;
    color += sampleBloom(vUV + vec2(0.0,  2.0 * t.y)) * 0.0625;

    outColor = vec4(color, 1.0);
}
