#include "renderer/OcclusionYield.h"

namespace ve::renderer {

bool OcclusionYieldController::update(uint32_t culledDrawItems, bool occlusionTestRan)
{
    switch (state_) {
    case State::Active:
        if (!occlusionTestRan) {
            // Asked for but not delivered -- usually the frame right after the
            // pyramid was invalidated. Says nothing either way, so hold.
            break;
        }
        if (culledDrawItems > 0) {
            zeroYieldFrames_ = 0;
            break;
        }
        if (++zeroYieldFrames_ >= kOcclusionSuspendAfterZeroFrames) {
            state_ = State::Suspended;
            framesSinceProbe_ = 0;
        }
        break;

    case State::Suspended:
        if (++framesSinceProbe_ >= kOcclusionProbeIntervalFrames) {
            state_ = State::Probing;
        }
        break;

    case State::Probing:
        // Stay here until a test actually runs: the first probe frame only
        // fills the pyramid, and the verdict comes from the frame after it.
        if (!occlusionTestRan) {
            break;
        }
        if (culledDrawItems > 0) {
            state_ = State::Active;
            zeroYieldFrames_ = 0;
        } else {
            state_ = State::Suspended;
            framesSinceProbe_ = 0;
        }
        break;
    }

    return shouldBuildPyramid();
}

void OcclusionYieldController::reset()
{
    state_ = State::Active;
    zeroYieldFrames_ = 0;
    framesSinceProbe_ = 0;
}

} // namespace ve::renderer
