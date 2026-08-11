#include "renderer/DynamicResolution.h"

#include "renderer/RuntimeSettings.h"

#include <algorithm>
#include <cmath>

namespace ve::renderer {

namespace {

// Snapping toward the direction of travel, not to nearest. Rounding to nearest
// can land back on the scale we started from, and then the controller makes no
// progress at all while remaining convinced it should.
float snapDown(float scale)
{
    return std::floor(scale / kDynamicResolutionScaleStep) * kDynamicResolutionScaleStep;
}

float snapUp(float scale)
{
    return std::ceil(scale / kDynamicResolutionScaleStep) * kDynamicResolutionScaleStep;
}

} // namespace

void DynamicResolutionController::reset()
{
    samples_ = {};
    sampleCursor_ = 0;
    sampleCount_ = 0;
    measurementsSinceChange_ = 0;
    changeCount_ = 0;
}

float DynamicResolutionController::medianGpuFrameMs() const
{
    if (sampleCount_ == 0) {
        return 0.0f;
    }

    // Copy and sort rather than keeping the window ordered: nine floats, once per
    // frame, against a decision that rebuilds every render target.
    std::array<float, kDynamicResolutionSampleWindow> sorted = samples_;
    const auto end = sorted.begin() + sampleCount_;
    std::sort(sorted.begin(), end);
    return sorted[sampleCount_ / 2];
}

float DynamicResolutionController::update(float gpuFrameMs,
                                          float currentScale,
                                          const DynamicResolutionSettings& settings)
{
    if (!settings.enabled) {
        // Leave the scale where it is. Snapping back to 1.0 on disable would
        // undo a measurement mid-comparison, and the slider is right there.
        reset();
        return currentScale;
    }

    // The bounds may be stored the wrong way round (a settings file is editable
    // by hand); take them as an unordered pair rather than trusting the names.
    const float boundA = clampRenderScale(settings.minScale);
    const float boundB = clampRenderScale(settings.maxScale);
    const float minScale = std::min(boundA, boundB);
    const float maxScale = std::max(boundA, boundB);

    if (gpuFrameMs <= 0.0f) {
        // No fresh measurement: not a data point, and not progress toward
        // leaving the settle window either.
        return currentScale;
    }

    samples_[sampleCursor_] = gpuFrameMs;
    sampleCursor_ = (sampleCursor_ + 1) % kDynamicResolutionSampleWindow;
    sampleCount_ = std::min(sampleCount_ + 1, kDynamicResolutionSampleWindow);

    if (measurementsSinceChange_ < kDynamicResolutionSettleFrames) {
        ++measurementsSinceChange_;
        return currentScale;
    }

    const float median = medianGpuFrameMs();
    const float target = std::max(settings.targetFrameMs, kDynamicResolutionMinTargetMs);

    // GPU cost is per pixel and pixel count goes as the square of the scale, so
    // the scale that would land on target is currentScale * sqrt(target/measured).
    // That over-corrects by whatever share of the frame is resolution-independent
    // (shadows, culling, the cluster passes), which is exactly why the result is
    // then capped to a step rather than applied outright.
    const float requested = currentScale * std::sqrt(target / median);

    float desired = currentScale;
    if (median > target) {
        desired = std::clamp(snapDown(std::max(requested, currentScale - kDynamicResolutionMaxStepDown)),
                             minScale,
                             maxScale);
    } else if (median < target * (1.0f - kDynamicResolutionRaiseHeadroom)) {
        desired = std::clamp(snapUp(std::min(requested, currentScale + kDynamicResolutionMaxStepUp)),
                             minScale,
                             maxScale);

        // Veto a raise that would immediately have to be undone.
        //
        // The deadband is a fraction of *time*, but the step is a fraction of
        // *scale*, and time goes as scale squared -- so one step changes the
        // frame time by roughly 2 * step / scale. At scale 1.0 a 0.05 step is
        // 10%, inside the 15% deadband, and the controller settles. At 0.25 the
        // same step is 40%, far outside it, and the controller cannot help but
        // oscillate: every step up lands over budget, every step down lands
        // under it. Observed in practice as 640x360 <-> 768x432 sixty times in
        // twenty seconds, with a target that simply has no grid point.
        //
        // So instead of asking "is there headroom", ask "would the scale I am
        // about to move to still fit". Being a little under budget is the right
        // answer when the grid cannot express the right one; thrashing is not.
        const float scaleRatio = (desired * desired) / (currentScale * currentScale);
        if (median * scaleRatio > target) {
            return currentScale;
        }
    } else {
        // Inside the deadband: on budget, nothing to do.
        return currentScale;
    }
    // Half a step of tolerance so a scale already sitting on the grid, or one
    // pinned at a bound while still over budget, does not report a change every
    // frame. The counters below are what the debug panel trusts.
    if (std::abs(desired - currentScale) < kDynamicResolutionScaleStep * 0.5f) {
        return currentScale;
    }

    measurementsSinceChange_ = 0;
    ++changeCount_;
    return desired;
}

} // namespace ve::renderer
