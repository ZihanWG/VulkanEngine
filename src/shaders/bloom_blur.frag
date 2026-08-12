#version 460
#include "sub_rect.glsl"

layout(set = 0, binding = 0) uniform sampler2D uBloomInput;

layout(push_constant) uniform BloomBlurPushConstants {
    vec2 texelSize;
    uint horizontal;
    uint padding;
    vec2 sourceUvScale;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    // texelSize is one *allocated* texel; vUV is normalised over the written
    // region, so the step has to be divided by the uv scale to stay one physical
    // texel. Offsets are added before scaling here, as in bloom_downsample.
    const vec2 writtenTexel = pc.texelSize / max(pc.sourceUvScale, vec2(1e-6));
    vec2 direction = pc.horizontal != 0u ? vec2(writtenTexel.x, 0.0) : vec2(0.0, writtenTexel.y);
    const vec2 sourceSize = vec2(textureSize(uBloomInput, 0));

    vec3 color = texture(uBloomInput, veSubRectClamp((vUV) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.227027;
    color += texture(uBloomInput, veSubRectClamp((vUV + direction * 1.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.1945946;
    color += texture(uBloomInput, veSubRectClamp((vUV - direction * 1.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.1945946;
    color += texture(uBloomInput, veSubRectClamp((vUV + direction * 2.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.1216216;
    color += texture(uBloomInput, veSubRectClamp((vUV - direction * 2.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.1216216;
    color += texture(uBloomInput, veSubRectClamp((vUV + direction * 3.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.054054;
    color += texture(uBloomInput, veSubRectClamp((vUV - direction * 3.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.054054;
    color += texture(uBloomInput, veSubRectClamp((vUV + direction * 4.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.016216;
    color += texture(uBloomInput, veSubRectClamp((vUV - direction * 4.0) * pc.sourceUvScale, pc.sourceUvScale, sourceSize)).rgb * 0.016216;

    outColor = vec4(color, 1.0);
}
