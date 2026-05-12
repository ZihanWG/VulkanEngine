#version 460

layout(set = 0, binding = 0) uniform sampler2D uBloomInput;

layout(push_constant) uniform BloomBlurPushConstants {
    vec2 texelSize;
    uint horizontal;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec2 direction = pc.horizontal != 0u ? vec2(pc.texelSize.x, 0.0) : vec2(0.0, pc.texelSize.y);

    vec3 color = texture(uBloomInput, vUV).rgb * 0.227027;
    color += texture(uBloomInput, vUV + direction * 1.0).rgb * 0.1945946;
    color += texture(uBloomInput, vUV - direction * 1.0).rgb * 0.1945946;
    color += texture(uBloomInput, vUV + direction * 2.0).rgb * 0.1216216;
    color += texture(uBloomInput, vUV - direction * 2.0).rgb * 0.1216216;
    color += texture(uBloomInput, vUV + direction * 3.0).rgb * 0.054054;
    color += texture(uBloomInput, vUV - direction * 3.0).rgb * 0.054054;
    color += texture(uBloomInput, vUV + direction * 4.0).rgb * 0.016216;
    color += texture(uBloomInput, vUV - direction * 4.0).rgb * 0.016216;

    outColor = vec4(color, 1.0);
}
