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
    // The current frame's jitter, in source pixels, range [-0.5, 0.5]. This is
    // what makes reconstruction possible rather than just accumulation: it says
    // where inside its texel each sample actually landed.
    vec2 jitterPixels;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// Gaussian falloff of the reconstruction kernel, in source pixels squared. At
// one texel away a sample keeps exp(-2.29) = 10% of the weight, at half a texel
// 57%. Tighter than this and an output pixel that no sample landed near goes
// noisy; looser and the reconstruction is a blur and there was no point undoing
// the jitter at all.
const float kReconstructionFalloff = 2.29;

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
    // vUV is normalised over the OUTPUT, which is what this pass's viewport
    // covers and what the history is written over.
    //
    // The current frame is reconstructed rather than sampled. Each source texel
    // holds the scene as seen through this frame's sub-pixel jitter, so the
    // sample it represents does not sit at its own centre: the projection was
    // shifted by +jitter, which means texel centre c holds what an unjittered
    // frame would have at c - jitter. Weighting the nine nearest samples by how
    // far their real positions land from this output pixel is what turns a
    // sequence of jittered low-resolution frames into a high-resolution one.
    // A plain bilinear tap throws that information away and can only ever
    // accumulate an upscale of itself.
    const vec2 currentAllocatedSize = vec2(textureSize(uCurrentColor, 0));
    const vec2 sourceSize = max(currentAllocatedSize * pc.sourceUvScale, vec2(1.0));
    const vec2 sourcePos = vUV * sourceSize;
    const vec2 baseTexel = floor(sourcePos - 0.5);

    vec3 weightedSum = vec3(0.0);
    float weightTotal = 0.0;
    vec3 minColor = vec3(1e30);
    vec3 maxColor = vec3(-1e30);

    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            // Clamped in texel space rather than UV space: the written region is
            // an integer number of texels and this keeps the gather inside it
            // without needing the sub-rect helper.
            const vec2 texel = clamp(baseTexel + vec2(float(x), float(y)), vec2(0.0), sourceSize - 1.0);
            const vec3 sampleColor = texture(uCurrentColor, (texel + 0.5) / currentAllocatedSize).rgb;

            // Where this sample really is, in source pixels, undoing the jitter.
            const vec2 offset = sourcePos - (texel + 0.5 - pc.jitterPixels);
            const float weight = exp(-kReconstructionFalloff * dot(offset, offset));

            weightedSum += sampleColor * weight;
            weightTotal += weight;

            // The rejection box comes from the same taps. It has to describe what
            // the low-resolution frame actually contains -- a box measured in
            // output texels would bound the history against an upscaled blur of
            // itself and stop rejecting anything.
            minColor = min(minColor, sampleColor);
            maxColor = max(maxColor, sampleColor);
        }
    }

    vec3 currentColor = weightTotal > 0.0 ? weightedSum / weightTotal
                                          : texture(uCurrentColor,
                                                    veSubRectUv(vUV, pc.sourceUvScale, currentAllocatedSize))
                                                .rgb;
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
                // Box already gathered above, from the same nine taps the
                // reconstruction used -- one gather, two jobs.
                historyColor = clamp(historyColor, minColor, maxColor);
            }

            resolvedColor = mix(currentColor, historyColor, clamp(pc.feedback, 0.0, 0.98));
        }
    }

    outColor = vec4(max(resolvedColor, vec3(0.0)), 1.0);
}
