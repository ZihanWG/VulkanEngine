#version 460

layout(set = 0, binding = 0) uniform sampler2D uCurrentColor;
layout(set = 0, binding = 1) uniform sampler2D uHistoryColor;

layout(push_constant) uniform TaaResolvePushConstants {
    vec2 texelSize;
    float feedback;
    uint historyValid;
    uint neighborhoodClampEnabled;
    uint padding0;
    uint padding1;
    uint padding2;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 currentColor = texture(uCurrentColor, vUV).rgb;
    vec3 resolvedColor = currentColor;

    if (pc.historyValid != 0u) {
        vec3 historyColor = texture(uHistoryColor, vUV).rgb;

        if (pc.neighborhoodClampEnabled != 0u) {
            vec3 minColor = currentColor;
            vec3 maxColor = currentColor;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    vec3 sampleColor = texture(uCurrentColor, vUV + vec2(x, y) * pc.texelSize).rgb;
                    minColor = min(minColor, sampleColor);
                    maxColor = max(maxColor, sampleColor);
                }
            }
            historyColor = clamp(historyColor, minColor, maxColor);
        }

        resolvedColor = mix(currentColor, historyColor, clamp(pc.feedback, 0.0, 0.98));
    }

    outColor = vec4(max(resolvedColor, vec3(0.0)), 1.0);
}
