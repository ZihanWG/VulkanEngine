// Cross-faded LOD transitions, shared by cull.comp (the state machine), the main
// fragment shader (the dither) and the depth prepass (which leaves fading draws
// out). Mirrors src/renderer/LodTransition.h function for function;
// tools/check_shader_constants.py compares the constants. The reasoning lives
// with the C++ half, next to its unit tests.

#ifndef VE_LOD_TRANSITION_GLSL
#define VE_LOD_TRANSITION_GLSL

// Must match ve::renderer::kLodCommandSlotsPerDrawItem.
const uint kLodCommandSlotsPerDrawItem = 2u;

// Must match ve::renderer::kLodInstance* -- the high half of firstInstance.
const uint kLodInstanceLevelMask = 15u;
const uint kLodInstanceFading = 16u;
const uint kLodInstanceOutgoing = 32u;
const uint kLodInstanceFadeShift = 8u;
const uint kLodFadeSteps = 255u;

// renderer::LodTransitionState, 16 bytes.
struct LodTransitionState {
    uint lastFrame;
    uint identity;
    uint levels;
    float progress;
};

uint lodTransitionIdentity(uint objectFrameDataIndex, uint firstIndex)
{
    return (objectFrameDataIndex * 0x9E3779B1u) ^ (firstIndex * 0x85EBCA77u) ^ 0xA5A5A5A5u;
}

uint packLodTransitionLevels(uint current, uint previous)
{
    return (current & kLodInstanceLevelMask) | ((previous & kLodInstanceLevelMask) << 4);
}

// renderer::advanceLodTransition. Returns the fade (0-kLodFadeSteps), writes the
// record to store back, and reports the levels to draw; fading is
// level != previousLevel.
uint advanceLodTransition(LodTransitionState stored,
                          uint identity,
                          uint selectedLevel,
                          uint frame,
                          float step,
                          out LodTransitionState next,
                          out uint level,
                          out uint previousLevel)
{
    level = selectedLevel;
    previousLevel = selectedLevel;

    bool continuing = step > 0.0 && stored.identity == identity && stored.lastFrame + 1u == frame;
    if (!continuing) {
        next = LodTransitionState(frame, identity, packLodTransitionLevels(selectedLevel, selectedLevel), 1.0);
        return kLodFadeSteps;
    }

    uint current = stored.levels & kLodInstanceLevelMask;
    uint previous = (stored.levels >> 4) & kLodInstanceLevelMask;
    float progress = clamp(stored.progress, 0.0, 1.0);

    if (selectedLevel == current) {
        progress = min(progress + step, 1.0);
    } else if (selectedLevel == previous) {
        previous = current;
        current = selectedLevel;
        progress = min(1.0 - progress + step, 1.0);
    } else {
        previous = progress >= 0.5 ? current : previous;
        current = selectedLevel;
        progress = min(step, 1.0);
    }
    if (progress >= 1.0) {
        previous = current;
    }

    next = LodTransitionState(frame, identity, packLodTransitionLevels(current, previous), progress);
    level = current;
    previousLevel = previous;
    if (previous == current) {
        return kLodFadeSteps;
    }
    return min(uint(progress * float(kLodFadeSteps) + 0.5), kLodFadeSteps);
}

// Interleaved gradient noise (Jimenez 2014): a per-pixel threshold in [0, 1)
// that depends on the pixel alone, so both halves of a fade read the same value
// and their keep tests below are exact complements.
float lodDitherNoise(vec2 pixel)
{
    return fract(52.9829189 * fract(dot(floor(pixel), vec2(0.06711056, 0.00583715))));
}

// renderer::lodDitherKeeps.
bool lodDitherKeeps(bool outgoing, float noise, uint fade)
{
    float threshold = float(fade) / float(kLodFadeSteps);
    return outgoing ? noise >= threshold : noise < threshold;
}

#endif
