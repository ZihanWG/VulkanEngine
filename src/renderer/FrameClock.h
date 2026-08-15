#pragma once

// The single source of time for the frame path.
//
// Real mode tracks a wall clock, which is what a desktop run wants. Fixed mode
// advances a virtual clock by a constant step per frame no matter how long the
// frame actually took, which is what makes a rendered frame reproducible: every
// time-driven input -- light animation, skeletal poses, animated object
// transforms, exposure adaptation -- becomes a pure function of the frame
// number instead of of how fast the machine happened to be.
//
// The caller reads the wall clock and passes the value in; this class only
// decides what to do with it. That seam is what keeps the stepping rules
// unit-testable without a GPU, a window, or a real clock.

#include <cstdint>

namespace ve::renderer {

class FrameClock final {
public:
    // 60 Hz. Arbitrary but fixed: what matters for reproducibility is that the
    // step never varies, not its particular value.
    static constexpr double kDefaultFixedStepSeconds = 1.0 / 60.0;

    // Switches to fixed stepping without disturbing elapsed time, so a mode
    // change mid-run does not make animation jump. Non-positive steps are
    // rejected and leave the clock unchanged.
    void useFixedStep(double stepSeconds = kDefaultFixedStepSeconds);

    // Advances exactly one frame. Call once per rendered frame, before anything
    // reads elapsedSeconds() or deltaSeconds().
    //
    // In fixed mode wallClockSeconds is ignored entirely. In real mode the first
    // call establishes the origin and reports a zero delta rather than a spike,
    // and a non-monotonic input is clamped to a zero delta rather than allowed
    // to run time backwards.
    void advance(double wallClockSeconds);

    [[nodiscard]] double elapsedSeconds() const { return elapsedSeconds_; }
    [[nodiscard]] double deltaSeconds() const { return deltaSeconds_; }
    [[nodiscard]] bool fixedStep() const { return fixedStep_; }
    [[nodiscard]] double fixedStepSeconds() const { return fixedStepSeconds_; }
    [[nodiscard]] uint64_t frameCount() const { return frameCount_; }

private:
    double elapsedSeconds_ = 0.0;
    double deltaSeconds_ = 0.0;
    double fixedStepSeconds_ = kDefaultFixedStepSeconds;
    double lastWallClockSeconds_ = 0.0;
    uint64_t frameCount_ = 0;
    bool fixedStep_ = false;
    bool started_ = false;
};

} // namespace ve::renderer
