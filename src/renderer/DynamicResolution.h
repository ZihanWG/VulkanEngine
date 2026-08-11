#pragma once

#include "renderer/RenderScale.h"

#include <array>
#include <cstdint>

namespace ve {
struct DynamicResolutionSettings;
} // namespace ve

namespace ve::renderer {

// Drives the render scale from measured GPU frame time: over budget lowers it,
// spare headroom raises it back. This is the controller only -- it decides a
// number and nothing else, so it is free of Vulkan and unit-tested without a
// device (see renderer/RenderScale.h for the same reasoning).
//
// Why GPU frame time and not the CPU frame delta: the CPU delta includes the
// present wait, so under vsync it reads as the refresh interval no matter how
// much headroom the GPU has, and it also absorbs the CPU hitch that applying a
// scale change causes -- which would feed the controller its own cost and make
// it chase itself downward. GPU frame total is the quantity render scale
// actually moves. The flip side is that a CPU-bound frame is invisible here,
// which is correct: lowering the resolution would not fix it.

// Tuning that is not worth persisting or exposing. Named constants rather than
// literals because each one encodes a decision.
//
// The controller is deliberately asymmetric. Dropping frames is happening now
// and is worth reacting to; unused headroom costs only sharpness, and raising
// eagerly is what makes a resolution controller visibly pump.
//
// Samples are reduced with a **median**, not a moving average. A single hitch --
// a shader compile, a texture upload, another process taking the GPU -- must not
// drop the resolution, and an EMA cannot give that guarantee at any usable
// smoothing factor: with a weight slow enough to absorb a 4x spike it is too
// slow to respond to real load. A median over an odd window is immune to
// outliers up to half the window by construction. It is also the same discipline
// every measurement in this project uses, for the same reason.
inline constexpr uint32_t kDynamicResolutionSampleWindow = 9;
// Fraction below target that must be free before the scale goes back up. Also
// the deadband: between target*(1 - this) and target nothing happens, which is
// what stops a scale from ping-ponging across a boundary forever.
inline constexpr float kDynamicResolutionRaiseHeadroom = 0.15f;
// Measurements ignored after a change. Must exceed kDynamicResolutionSampleWindow
// so that by the time the controller acts again, every sample in the window was
// taken at the current resolution -- otherwise it would decide using frame times
// from the resolution it just left.
inline constexpr uint32_t kDynamicResolutionSettleFrames = 12;
static_assert(kDynamicResolutionSettleFrames > kDynamicResolutionSampleWindow,
              "The settle window must outlast the sample window, or a decision can be made from "
              "frame times measured at the previous resolution.");
inline constexpr float kDynamicResolutionMaxStepDown = 0.15f;
inline constexpr float kDynamicResolutionMaxStepUp = 0.05f;
// Scales are quantised to this grid. Applying a change rebuilds every
// screen-sized target, so a continuous scale would mean a rebuild on almost
// every frame; a grid plus the deadband bounds how often that happens.
inline constexpr float kDynamicResolutionScaleStep = 0.05f;
// Below this a "target" is not a frame budget, and dividing by it would produce
// an absurd scale request.
inline constexpr float kDynamicResolutionMinTargetMs = 1.0f;

class DynamicResolutionController final {
public:
    // Feed a *newly measured* GPU frame time, or a non-positive value when no
    // measurement landed this frame. Repeating the previous frame's value would
    // fill the median window with duplicates of one sample and defeat it, so the
    // caller is responsible for only passing fresh readings. Returns the scale to
    // use -- equal to currentScale when nothing should change, so the caller's
    // "did it move" comparison is the only trigger it needs.
    [[nodiscard]] float update(float gpuFrameMs, float currentScale, const DynamicResolutionSettings& settings);

    void reset();

    // Median of the window, or 0 before the first measurement.
    [[nodiscard]] float medianGpuFrameMs() const;
    [[nodiscard]] bool hasMeasurement() const { return sampleCount_ > 0; }
    [[nodiscard]] uint32_t changeCount() const { return changeCount_; }
    // Measurements since the last scale change; while below
    // kDynamicResolutionSettleFrames the controller is deliberately not acting.
    // Surfaced so the debug panel can say "settling" rather than looking stuck.
    [[nodiscard]] uint32_t measurementsSinceChange() const { return measurementsSinceChange_; }

private:
    std::array<float, kDynamicResolutionSampleWindow> samples_{};
    uint32_t sampleCursor_ = 0;
    // Saturates at the window size; also how many entries of samples_ are valid.
    uint32_t sampleCount_ = 0;
    uint32_t measurementsSinceChange_ = 0;
    uint32_t changeCount_ = 0;
};

} // namespace ve::renderer
