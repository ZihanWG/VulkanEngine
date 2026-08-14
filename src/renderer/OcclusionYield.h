#pragma once

#include <cstdint>

namespace ve::renderer {

// Suspends Hi-Z occlusion culling when it stops paying for itself, and probes
// periodically to notice when it starts paying again. Controller only -- it
// decides a bool and nothing else, so it is free of Vulkan and unit-tested
// without a device (same reasoning as renderer/DynamicResolution.h).
//
// The problem it solves is measurable on the default scene: occlusion culling
// removed 0 draw items in every sampled frame while the Hi-Z pyramid it needs
// cost 1.36 ms, 8% of the frame. Occlusion culling earns its keep on a depth
// complex scene and is dead weight on an open one, and which of those is on
// screen is not known until it has been tried.
//
// Suspending is always safe. Skipping occlusion culling can only draw MORE than
// necessary -- never less -- so a wrong decision here costs frame time, never
// pixels. That is what lets the controller be aggressive.

// Consecutive tested frames of zero yield before suspending. One second at
// 60 Hz: long enough that a camera sweeping past a wall does not trip it,
// short enough that an open scene stops paying almost immediately.
inline constexpr uint32_t kOcclusionSuspendAfterZeroFrames = 60;

// Frames between probes while suspended. A probe costs two frames of pyramid
// build (one to fill it, one to test against it), so at 180 the amortised cost
// is about 1% of what running occlusion continuously would be, and the worst
// case for noticing that occlusion became useful again is three seconds.
inline constexpr uint32_t kOcclusionProbeIntervalFrames = 180;

static_assert(kOcclusionProbeIntervalFrames > kOcclusionSuspendAfterZeroFrames,
              "Probing more often than it takes to suspend would make the controller oscillate.");

class OcclusionYieldController {
public:
    enum class State : uint8_t {
        // Occlusion culling is paying; build the pyramid every frame.
        Active,
        // It stopped paying; stop building the pyramid.
        Suspended,
        // Building again to find out whether it pays now.
        Probing,
    };

    // Call once per frame, after the frame's culling stats are known.
    //
    // occlusionTestRan says whether occlusion culling actually ran this frame,
    // which is not the same as it having been asked for: the frame after a
    // suspension has no valid pyramid yet, and a zero from a frame that never
    // tested anything is not evidence about yield. culledDrawItems is only read
    // when occlusionTestRan is true.
    //
    // Returns whether the NEXT frame should build the pyramid. The occlusion
    // test itself needs no separate decision -- it already requires a valid
    // pyramid, so it follows from this one automatically, one frame behind.
    bool update(uint32_t culledDrawItems, bool occlusionTestRan);

    // Forget the accumulated evidence: the scene or the camera changed enough
    // that what occlusion culling used to yield says nothing about now.
    void reset();

    [[nodiscard]] bool shouldBuildPyramid() const { return state_ != State::Suspended; }
    [[nodiscard]] State state() const { return state_; }
    [[nodiscard]] uint32_t zeroYieldFrames() const { return zeroYieldFrames_; }
    [[nodiscard]] uint32_t framesSinceProbe() const { return framesSinceProbe_; }

private:
    State state_ = State::Active;
    uint32_t zeroYieldFrames_ = 0;
    uint32_t framesSinceProbe_ = 0;
};

} // namespace ve::renderer
