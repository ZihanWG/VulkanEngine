#version 460

layout(set = 0, binding = 0) uniform sampler2D uCurrentColor;
layout(set = 0, binding = 1) uniform sampler2D uHistoryColor;
layout(set = 0, binding = 2) uniform sampler2D uVelocity;
layout(set = 0, binding = 3) uniform sampler2D uDepth;

layout(push_constant) uniform TaaResolvePushConstants {
    vec2 texelSize;
    float feedback;
    uint historyValid;
    uint neighborhoodClampEnabled;
    uint reprojectionEnabled;
    uint depthDilationEnabled;
    uint padding0;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// Velocity dilation: sample the motion vector of the closest (front-most)
// pixel in the 3x3 neighborhood so silhouette edges reproject with the
// foreground object's motion instead of the background's.
vec2 dilatedVelocityUV()
{
    if (pc.depthDilationEnabled == 0u) {
        return vUV;
    }

    float closestDepth = 1.0;
    vec2 closestUV = vUV;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 sampleUV = vUV + vec2(x, y) * pc.texelSize;
            float depth = texture(uDepth, sampleUV).r;
            if (depth < closestDepth) {
                closestDepth = depth;
                closestUV = sampleUV;
            }
        }
    }
    return closestUV;
}

void main()
{
    vec3 currentColor = texture(uCurrentColor, vUV).rgb;
    vec3 resolvedColor = currentColor;

    if (pc.historyValid != 0u) {
        vec2 historyUV = vUV;
        bool historyUsable = true;

        if (pc.reprojectionEnabled != 0u) {
            vec2 velocity = texture(uVelocity, dilatedVelocityUV()).rg;
            historyUV = vUV - velocity;
            historyUsable = all(greaterThanEqual(historyUV, vec2(0.0))) &&
                            all(lessThanEqual(historyUV, vec2(1.0)));
        }

        if (historyUsable) {
            vec3 historyColor = texture(uHistoryColor, historyUV).rgb;

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
    }

    outColor = vec4(max(resolvedColor, vec3(0.0)), 1.0);
}
