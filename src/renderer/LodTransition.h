#pragma once

// Cross-faded LOD transitions, the GPU-free half.
//
// A discrete LOD switch is a pop: one frame shows level n, the next level n+1,
// and the silhouette jumps. The usual cheap cure, and the one here, is a short
// screen-space dither between the two: for a few frames both levels are drawn,
// each discarding a complementary half of its pixels by a per-pixel noise
// threshold, and the threshold sweeps from one level to the other. No blending
// and no sorting -- every surviving fragment is opaque and depth-tested as usual.
//
// The cull pass owns the decision, because only it knows which level it picked.
// It keeps one LodTransitionState per draw item in a persistent buffer, advances
// it with advanceLodTransition() whenever it emits that item, and when the item
// is mid-fade emits a second command for the outgoing level. lod_transition.glsl
// is the shader half and mirrors every function here; the constants are compared
// by tools/check_shader_constants.py.

#include <algorithm>
#include <cstdint>

namespace ve::renderer {

// Command slots each draw item owns in the main pass's indirect buffer when the
// GPU cull writes it: the level it selected, and the level it is fading out.
// The shadow dispatches never cross-fade and keep one slot.
inline constexpr uint32_t kLodCommandSlotsPerDrawItem = 2;

// The high half of firstInstance (gl_InstanceIndex >> 16), which already carried
// the selected level for the heatmap. Written as plain values rather than shifts
// for the constant checker.
//   bits 0-3   the level this command draws
//   bit 4      the command is half of a cross-fade, so the fragment dithers
//   bit 5      it is the outgoing half (keeps the pixels the incoming one drops)
//   bits 8-14  the fade: how much of the screen the incoming level covers, 0-127
// Bit 15 stays clear, which keeps firstInstance below 2^31: gl_InstanceIndex is
// a signed int, and a fade in the top bit would turn it negative.
inline constexpr uint32_t kLodInstanceLevelMask = 15;
inline constexpr uint32_t kLodInstanceFading = 16;
inline constexpr uint32_t kLodInstanceOutgoing = 32;
inline constexpr uint32_t kLodInstanceFadeShift = 8;
inline constexpr uint32_t kLodFadeSteps = 127;

// One draw item's transition, 16 bytes in a std430 array.
//
// `identity` ties the record to what the draw item is, because draw items are
// renumbered whenever the scene changes and a stale record must not fade a new
// object from somebody else's level. `lastFrame` ties it to a continuous run of
// frames, because an item that was culled for a while comes back at whatever
// level suits it now and fading in from a level last seen seconds ago would be a
// transition nobody watched start. Either mismatch snaps.
struct LodTransitionState {
    uint32_t lastFrame = 0;
    uint32_t identity = 0;
    // Current level in bits 0-3, the level being faded out in bits 4-7. Equal
    // when no fade is running.
    uint32_t levels = 0;
    // How far the fade into the current level has gone, 0-1.
    float progress = 1.0f;
};
static_assert(sizeof(LodTransitionState) == 16, "LodTransitionState is mirrored in GLSL as a 16-byte std430 struct.");

// What the cull pass does with a draw item this frame.
struct LodTransitionStep {
    LodTransitionState state;
    // The level to draw -- the incoming one while fading.
    uint32_t level = 0;
    // The outgoing level while fading; equal to `level` otherwise.
    uint32_t previousLevel = 0;
    bool fading = false;
    // The incoming level's share of the pixels, 0-kLodFadeSteps. Both commands
    // carry the same value, which is what makes their dithers complementary.
    uint32_t fade = kLodFadeSteps;
};

// A draw item's identity for the record: which object slot it draws and which
// authored range. Mixed rather than concatenated because both are wider than
// half a word on large scenes.
[[nodiscard]] inline uint32_t lodTransitionIdentity(uint32_t objectFrameDataIndex, uint32_t firstIndex)
{
    return (objectFrameDataIndex * 0x9E3779B1u) ^ (firstIndex * 0x85EBCA77u) ^ 0xA5A5A5A5u;
}

[[nodiscard]] inline uint32_t packLodTransitionLevels(uint32_t current, uint32_t previous)
{
    return (current & kLodInstanceLevelMask) | ((previous & kLodInstanceLevelMask) << 4);
}

// Advances `stored` by one frame towards `selectedLevel`.
//
// `step` is the progress one frame buys -- frame time over transition time --
// and 0 turns transitions off: every frame then snaps.
//
// Three ways a new selection can arrive:
// - the current level again: the fade, if any, continues;
// - the level being faded out (the camera turned back mid-fade): the fade runs
//   backwards from where it is, rather than restarting, so nothing pops;
// - anything else: a new fade starts, out of whichever level dominates the
//   screen right now.
[[nodiscard]] inline LodTransitionStep advanceLodTransition(
    const LodTransitionState& stored, uint32_t identity, uint32_t selectedLevel, uint32_t frame, float step)
{
    LodTransitionStep result;
    result.level = selectedLevel;
    result.previousLevel = selectedLevel;

    const bool continuing = step > 0.0f && stored.identity == identity && stored.lastFrame + 1u == frame;
    if (!continuing) {
        result.state = {frame, identity, packLodTransitionLevels(selectedLevel, selectedLevel), 1.0f};
        return result;
    }

    uint32_t current = stored.levels & kLodInstanceLevelMask;
    uint32_t previous = (stored.levels >> 4) & kLodInstanceLevelMask;
    float progress = std::clamp(stored.progress, 0.0f, 1.0f);

    if (selectedLevel == current) {
        progress = std::min(progress + step, 1.0f);
    } else if (selectedLevel == previous) {
        previous = current;
        current = selectedLevel;
        progress = std::min(1.0f - progress + step, 1.0f);
    } else {
        previous = progress >= 0.5f ? current : previous;
        current = selectedLevel;
        progress = std::min(step, 1.0f);
    }
    if (progress >= 1.0f) {
        previous = current;
    }

    result.state = {frame, identity, packLodTransitionLevels(current, previous), progress};
    result.level = current;
    result.previousLevel = previous;
    result.fading = previous != current;
    result.fade = result.fading ? std::min(static_cast<uint32_t>(progress * static_cast<float>(kLodFadeSteps) + 0.5f),
                                           kLodFadeSteps)
                                : kLodFadeSteps;
    return result;
}

// Whether a fragment of a cross-fading draw survives. `noise` is the per-pixel
// threshold in [0, 1), identical for both halves of the fade because it depends
// on the pixel alone; the incoming half keeps what falls below the fade and the
// outgoing half keeps the rest, so every pixel is drawn by exactly one of them.
[[nodiscard]] inline bool lodDitherKeeps(bool outgoing, float noise, uint32_t fade)
{
    const float threshold = static_cast<float>(fade) / static_cast<float>(kLodFadeSteps);
    return outgoing ? noise >= threshold : noise < threshold;
}

} // namespace ve::renderer
