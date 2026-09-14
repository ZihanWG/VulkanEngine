// The overdraw ratio, which is the number the depth-prepass question was blocked
// on. The query itself needs a device; this covers the arithmetic that turns its
// two counters into the figure that gets quoted, because that figure is easy to
// get subtly wrong and impossible to sanity-check by eye once it is in a log.

#include "renderer/OverdrawQuery.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using ve::renderer::OverdrawQuery;

TEST_CASE("Invocations per pixel is invocations over rendered pixels", "[overdraw]")
{
    OverdrawQuery::FrameResult result{};
    result.valid = true;
    result.fragmentInvocations = 2014135;
    result.renderedPixels = 1280 * 720;

    // The measured Sponza figure, kept as the fixture so a change to the
    // arithmetic shows up against a number that was actually observed.
    CHECK(result.invocationsPerPixel() == Approx(2.1855).epsilon(0.001));
}

TEST_CASE("A frame with no rendered pixels reports zero rather than dividing by it", "[overdraw]")
{
    // Reachable: a minimised window drives the render extent to zero, and the
    // readout is printed on a timer that does not know that.
    OverdrawQuery::FrameResult result{};
    result.valid = true;
    result.fragmentInvocations = 1234;
    result.renderedPixels = 0;

    CHECK(result.invocationsPerPixel() == Approx(0.0));
}

TEST_CASE("The ratio follows the resolution it was recorded at", "[overdraw]")
{
    // renderedPixels travels with the result rather than being read from the
    // renderer when the log line is written: render scale and dynamic resolution
    // both move the extent, and a ratio computed against a later frame's
    // resolution would be wrong in a way nothing downstream could detect.
    OverdrawQuery::FrameResult full{};
    full.valid = true;
    full.fragmentInvocations = 2000000;
    full.renderedPixels = 1280 * 720;

    OverdrawQuery::FrameResult half{};
    half.valid = true;
    half.fragmentInvocations = 500000;
    half.renderedPixels = 640 * 360;

    // Quarter the pixels and quarter the invocations is the same overdraw.
    CHECK(full.invocationsPerPixel() == Approx(half.invocationsPerPixel()).epsilon(0.001));
}
