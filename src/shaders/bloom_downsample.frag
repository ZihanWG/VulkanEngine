#version 460

layout(set = 0, binding = 0) uniform sampler2D uSource;

layout(push_constant) uniform BloomDownsamplePushConstants {
    vec2 texelSize;
    float threshold;
    uint applyThreshold;
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

vec3 sampleBloom(vec2 uv)
{
    return thresholdBright(texture(uSource, uv).rgb);
}

void main()
{
    vec2 t = pc.texelSize;

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
