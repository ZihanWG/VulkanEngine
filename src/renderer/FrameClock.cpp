#include "renderer/FrameClock.h"

namespace ve::renderer {

void FrameClock::useFixedStep(double stepSeconds)
{
    if (!(stepSeconds > 0.0)) {
        // Also rejects NaN, which would otherwise poison elapsed time forever.
        return;
    }

    fixedStepSeconds_ = stepSeconds;
    fixedStep_ = true;
}

void FrameClock::advance(double wallClockSeconds)
{
    ++frameCount_;

    if (fixedStep_) {
        deltaSeconds_ = fixedStepSeconds_;
        elapsedSeconds_ += fixedStepSeconds_;
        // Kept current so a later switch back to real time does not see a stale
        // origin and report one enormous delta.
        lastWallClockSeconds_ = wallClockSeconds;
        started_ = true;
        return;
    }

    if (!started_) {
        // First real-time frame: establish the origin. Reporting the time since
        // some unrelated epoch here would hand every consumer a huge first delta.
        deltaSeconds_ = 0.0;
        lastWallClockSeconds_ = wallClockSeconds;
        started_ = true;
        return;
    }

    const double delta = wallClockSeconds - lastWallClockSeconds_;
    deltaSeconds_ = delta > 0.0 ? delta : 0.0;
    elapsedSeconds_ += deltaSeconds_;
    lastWallClockSeconds_ = wallClockSeconds;
}

} // namespace ve::renderer
