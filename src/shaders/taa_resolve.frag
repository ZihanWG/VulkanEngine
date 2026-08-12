#version 460
#include "sub_rect.glsl"

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
    // The resolve now runs at OUTPUT resolution: vUV spans the history, which is
    // written in full. Scene colour, velocity and depth are the low-resolution
    // sources and share one allocation, so one scale covers those three.
    vec2 sourceUvScale;
    vec2 padding1;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// One source texel, expressed in the output-normalised space vUV lives in.
// texelSize is one texel of the source *allocation*; the sources are written
// over sourceUvScale of it, so a step of one physical texel is that much larger
// once measured against the output. Getting this backwards shrinks every
// neighbourhood silently -- the same trap the bloom chain and gtao_blur have.
vec2 sourceTexelStep()
{
    return pc.texelSize / max(pc.sourceUvScale, vec2(1e-6));
}

// Velocity dilation: sample the motion vector of the closest (front-most)
// pixel in the 3x3 neighborhood so silhouette edges reproject with the
// foreground object's motion instead of the background's.
vec2 dilatedVelocityUV()
{
    if (pc.depthDilationEnabled == 0u) {
        return vUV;
    }

    const vec2 step = sourceTexelStep();
    float closestDepth = 1.0;
    vec2 closestUV = vUV;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            // The returned UV stays in output space, which is what the velocity
            // fetch and the history lookup both expect.
            vec2 sampleUV = clamp(vUV + vec2(x, y) * step, vec2(0.0), vec2(1.0));
            float depth =
                texture(uDepth, veSubRectUv(sampleUV, pc.sourceUvScale, vec2(textureSize(uDepth, 0)))).r;
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
    // vUV is normalised over the OUTPUT, which is what this pass's viewport now
    // covers and what the history is written over. The low-resolution sources
    // need it scaled by sourceUvScale.
    //
    // Phase 1 still reads the current frame with a plain bilinear tap, so this is
    // temporal accumulation *after* an upscale rather than an upscale driven by
    // the accumulation. Correct, and not yet the point: reconstructing detail
    // means weighting the jittered sample by where it actually landed inside this
    // output pixel, which is the next step.
    const vec2 currentAllocatedSize = vec2(textureSize(uCurrentColor, 0));
    vec3 currentColor = texture(uCurrentColor, veSubRectUv(vUV, pc.sourceUvScale, currentAllocatedSize)).rgb;
    vec3 resolvedColor = currentColor;

    if (pc.historyValid != 0u) {
        vec2 historyUV = vUV;
        bool historyUsable = true;

        if (pc.reprojectionEnabled != 0u) {
            vec2 velocity =
                texture(uVelocity,
                        veSubRectUv(dilatedVelocityUV(), pc.sourceUvScale, vec2(textureSize(uVelocity, 0))))
                    .rg;
            historyUV = vUV - velocity;
            historyUsable = all(greaterThanEqual(historyUV, vec2(0.0))) &&
                            all(lessThanEqual(historyUV, vec2(1.0)));
        }

        if (historyUsable) {
            // The history is written in full at output resolution, so it needs
            // no scaling and no sub-rect clamp -- it is the one source here that
            // is not low-resolution.
            vec3 historyColor = texture(uHistoryColor, historyUV).rgb;

            if (pc.neighborhoodClampEnabled != 0u) {
                vec3 minColor = currentColor;
                vec3 maxColor = currentColor;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        // One source texel apart in output space: the box has to
                        // bound what the *low-resolution* frame contains, or it
                        // would clamp the history against an upscaled blur of
                        // itself and stop rejecting anything.
                        vec3 sampleColor =
                            texture(uCurrentColor,
                                    veSubRectUv(clamp(vUV + vec2(x, y) * sourceTexelStep(), vec2(0.0), vec2(1.0)),
                                                pc.sourceUvScale,
                                                currentAllocatedSize))
                                .rgb;
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
