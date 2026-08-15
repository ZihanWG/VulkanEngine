#include "renderer/FrameClock.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

using Catch::Approx;
using ve::renderer::FrameClock;

TEST_CASE("A real-time clock reports no delta on its very first frame")
{
    FrameClock clock;
    clock.advance(1234.5);

    // Not the time since some unrelated epoch: a first-frame spike would kick
    // every time-driven system before the scene has even settled.
    REQUIRE(clock.deltaSeconds() == Approx(0.0));
    REQUIRE(clock.elapsedSeconds() == Approx(0.0));
    REQUIRE(clock.frameCount() == 1);
}

TEST_CASE("A real-time clock tracks the wall clock it is given")
{
    FrameClock clock;
    clock.advance(100.0);
    clock.advance(100.25);
    REQUIRE(clock.deltaSeconds() == Approx(0.25));
    REQUIRE(clock.elapsedSeconds() == Approx(0.25));

    clock.advance(100.75);
    REQUIRE(clock.deltaSeconds() == Approx(0.5));
    REQUIRE(clock.elapsedSeconds() == Approx(0.75));
}

TEST_CASE("A real-time clock refuses to run backwards")
{
    FrameClock clock;
    clock.advance(50.0);
    clock.advance(51.0);
    REQUIRE(clock.elapsedSeconds() == Approx(1.0));

    // A non-monotonic reading must not rewind animation or hand a consumer a
    // negative delta to integrate.
    clock.advance(40.0);
    REQUIRE(clock.deltaSeconds() == Approx(0.0));
    REQUIRE(clock.elapsedSeconds() == Approx(1.0));
}

TEST_CASE("A fixed-step clock ignores the wall clock entirely")
{
    FrameClock clock;
    clock.useFixedStep(1.0 / 60.0);

    REQUIRE(clock.fixedStep());

    // Wildly varying wall-clock readings, exactly what a software rasterizer
    // under CI load produces. None of them may reach elapsed time.
    clock.advance(0.0);
    clock.advance(9999.0);
    clock.advance(3.0);

    REQUIRE(clock.deltaSeconds() == Approx(1.0 / 60.0));
    REQUIRE(clock.elapsedSeconds() == Approx(3.0 / 60.0));
    REQUIRE(clock.frameCount() == 3);
}

TEST_CASE("A fixed-step clock is a pure function of the frame number")
{
    FrameClock fast;
    FrameClock slow;
    fast.useFixedStep();
    slow.useFixedStep();

    for (int frame = 0; frame < 120; ++frame) {
        fast.advance(static_cast<double>(frame) * 0.001);
        slow.advance(static_cast<double>(frame) * 7.5);
    }

    // The whole point: two machines of wildly different speed must present the
    // same elapsed time to the frame path after the same number of frames.
    REQUIRE(fast.elapsedSeconds() == Approx(slow.elapsedSeconds()));
    REQUIRE(fast.elapsedSeconds() == Approx(2.0));
}

TEST_CASE("Switching to fixed stepping preserves elapsed time")
{
    FrameClock clock;
    clock.advance(10.0);
    clock.advance(11.0);
    REQUIRE(clock.elapsedSeconds() == Approx(1.0));

    clock.useFixedStep(0.5);
    clock.advance(500.0);

    // Continuity matters: a mode switch must not teleport animated transforms.
    REQUIRE(clock.elapsedSeconds() == Approx(1.5));
    REQUIRE(clock.deltaSeconds() == Approx(0.5));
}

TEST_CASE("A non-positive or NaN fixed step is rejected")
{
    FrameClock clock;

    clock.useFixedStep(0.0);
    REQUIRE_FALSE(clock.fixedStep());

    clock.useFixedStep(-1.0);
    REQUIRE_FALSE(clock.fixedStep());

    clock.useFixedStep(std::numeric_limits<double>::quiet_NaN());
    REQUIRE_FALSE(clock.fixedStep());

    // A rejected step must leave a usable clock behind, not a poisoned one.
    clock.advance(1.0);
    clock.advance(2.0);
    REQUIRE(clock.elapsedSeconds() == Approx(1.0));
}

TEST_CASE("The default fixed step is 60 Hz")
{
    FrameClock clock;
    clock.useFixedStep();

    clock.advance(0.0);
    REQUIRE(clock.deltaSeconds() == Approx(FrameClock::kDefaultFixedStepSeconds));
    REQUIRE(clock.fixedStepSeconds() == Approx(1.0 / 60.0));
}
