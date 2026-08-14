#include "renderer/OcclusionYield.h"

#include <catch2/catch_test_macros.hpp>

using ve::renderer::kOcclusionProbeIntervalFrames;
using ve::renderer::kOcclusionSuspendAfterZeroFrames;
using ve::renderer::OcclusionYieldController;
using State = ve::renderer::OcclusionYieldController::State;

namespace {

// Run `frames` frames in which occlusion tested and culled `culled` items.
void runTested(OcclusionYieldController& controller, uint32_t frames, uint32_t culled)
{
    for (uint32_t frame = 0; frame < frames; ++frame) {
        controller.update(culled, /*occlusionTestRan=*/true);
    }
}

// Run `frames` frames in which occlusion did not test anything.
void runUntested(OcclusionYieldController& controller, uint32_t frames)
{
    for (uint32_t frame = 0; frame < frames; ++frame) {
        controller.update(0, /*occlusionTestRan=*/false);
    }
}

// Drive a controller from Active into Suspended.
OcclusionYieldController suspendedController()
{
    OcclusionYieldController controller;
    runTested(controller, kOcclusionSuspendAfterZeroFrames, 0);
    return controller;
}

} // namespace

TEST_CASE("Occlusion culling starts active and builds the pyramid")
{
    const OcclusionYieldController controller;
    CHECK(controller.state() == State::Active);
    CHECK(controller.shouldBuildPyramid());
}

TEST_CASE("Occlusion culling that keeps paying is never suspended")
{
    OcclusionYieldController controller;
    runTested(controller, kOcclusionSuspendAfterZeroFrames * 4, 670);
    CHECK(controller.state() == State::Active);
    CHECK(controller.shouldBuildPyramid());
}

TEST_CASE("A run of zero-yield frames suspends the pyramid build")
{
    OcclusionYieldController controller;
    runTested(controller, kOcclusionSuspendAfterZeroFrames - 1, 0);
    CHECK(controller.state() == State::Active);
    CHECK(controller.shouldBuildPyramid());

    CHECK_FALSE(controller.update(0, /*occlusionTestRan=*/true));
    CHECK(controller.state() == State::Suspended);
    CHECK_FALSE(controller.shouldBuildPyramid());
}

TEST_CASE("A single paying frame resets the zero-yield run")
{
    OcclusionYieldController controller;
    runTested(controller, kOcclusionSuspendAfterZeroFrames - 1, 0);
    runTested(controller, 1, 5);
    CHECK(controller.zeroYieldFrames() == 0);

    // The run has to start over, so one more zero frame must not suspend.
    runTested(controller, 1, 0);
    CHECK(controller.state() == State::Active);
}

TEST_CASE("Frames where occlusion never tested are not evidence of zero yield")
{
    OcclusionYieldController controller;
    runUntested(controller, kOcclusionSuspendAfterZeroFrames * 2);
    CHECK(controller.state() == State::Active);
    CHECK(controller.zeroYieldFrames() == 0);
}

TEST_CASE("A suspended controller probes after the probe interval")
{
    OcclusionYieldController controller = suspendedController();

    runUntested(controller, kOcclusionProbeIntervalFrames - 1);
    CHECK(controller.state() == State::Suspended);
    CHECK_FALSE(controller.shouldBuildPyramid());

    CHECK(controller.update(0, /*occlusionTestRan=*/false));
    CHECK(controller.state() == State::Probing);
    CHECK(controller.shouldBuildPyramid());
}

TEST_CASE("Probing waits for a test rather than judging the fill frame")
{
    OcclusionYieldController controller = suspendedController();
    runUntested(controller, kOcclusionProbeIntervalFrames);
    REQUIRE(controller.state() == State::Probing);

    // The first probe frame only fills the pyramid; nothing has been tested
    // against it yet, and treating that as zero yield would re-suspend
    // immediately and make the probe useless.
    runUntested(controller, 1);
    CHECK(controller.state() == State::Probing);
    CHECK(controller.shouldBuildPyramid());
}

TEST_CASE("A probe that finds work reactivates occlusion culling")
{
    OcclusionYieldController controller = suspendedController();
    runUntested(controller, kOcclusionProbeIntervalFrames);
    REQUIRE(controller.state() == State::Probing);

    controller.update(670, /*occlusionTestRan=*/true);
    CHECK(controller.state() == State::Active);
    CHECK(controller.shouldBuildPyramid());
    CHECK(controller.zeroYieldFrames() == 0);
}

TEST_CASE("A probe that finds nothing re-suspends and restarts the interval")
{
    OcclusionYieldController controller = suspendedController();
    runUntested(controller, kOcclusionProbeIntervalFrames);
    REQUIRE(controller.state() == State::Probing);

    controller.update(0, /*occlusionTestRan=*/true);
    CHECK(controller.state() == State::Suspended);
    CHECK(controller.framesSinceProbe() == 0);

    // And it must wait the whole interval again rather than probing every frame.
    runUntested(controller, kOcclusionProbeIntervalFrames - 1);
    CHECK(controller.state() == State::Suspended);
}

TEST_CASE("A steady open scene probes at a bounded rate, not continuously")
{
    OcclusionYieldController controller;
    uint32_t buildFrames = 0;
    constexpr uint32_t kFrames = 6000;
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        const bool build = controller.shouldBuildPyramid();
        buildFrames += build ? 1u : 0u;
        // Only a frame that built the pyramid can test against it next frame.
        controller.update(0, /*occlusionTestRan=*/build);
    }

    // Without the controller this would be every frame. The bound is the initial
    // run plus two frames per probe interval, with slack for the phase.
    const uint32_t probes = kFrames / kOcclusionProbeIntervalFrames;
    CHECK(buildFrames <= kOcclusionSuspendAfterZeroFrames + (probes + 1) * 3);
    CHECK(buildFrames < kFrames / 10);
}

TEST_CASE("reset returns a suspended controller to active")
{
    OcclusionYieldController controller = suspendedController();
    REQUIRE_FALSE(controller.shouldBuildPyramid());

    controller.reset();
    CHECK(controller.state() == State::Active);
    CHECK(controller.shouldBuildPyramid());
    CHECK(controller.framesSinceProbe() == 0);
}
