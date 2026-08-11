#include "renderer/DynamicResolution.h"

#include "renderer/RenderScale.h"
#include "renderer/RuntimeSettings.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

using ve::DynamicResolutionSettings;
using ve::renderer::DynamicResolutionController;
using ve::renderer::kDynamicResolutionScaleStep;
using ve::renderer::kDynamicResolutionSampleWindow;
using ve::renderer::kDynamicResolutionSettleFrames;
using ve::renderer::kMaxRenderScale;

namespace {

DynamicResolutionSettings enabledSettings()
{
    DynamicResolutionSettings settings{};
    settings.enabled = true;
    settings.targetFrameMs = 16.0f;
    settings.minScale = 0.5f;
    settings.maxScale = 1.0f;
    return settings;
}

// Feeds the controller a steady frame time for `frames` frames and returns the
// scale it settles on. Mirrors how Renderer drives it: one update per frame,
// with the scale it returns fed back in.
float run(DynamicResolutionController& controller,
          const DynamicResolutionSettings& settings,
          float gpuFrameMs,
          float startScale,
          int frames)
{
    float scale = startScale;
    for (int frame = 0; frame < frames; ++frame) {
        scale = controller.update(gpuFrameMs, scale, settings);
    }
    return scale;
}

} // namespace

TEST_CASE("Disabled controller never moves the scale", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    DynamicResolutionSettings settings = enabledSettings();
    settings.enabled = false;

    // Wildly over budget, and it still must not act.
    CHECK(run(controller, settings, 90.0f, 0.8f, 200) == Catch::Approx(0.8f));
    CHECK(controller.changeCount() == 0u);
}

TEST_CASE("Sustained overshoot lowers the scale toward the target", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // A constant 32 ms against a 16 ms budget: half the frame budget, so the
    // sqrt relation asks for ~0.707 of the current scale and the floor is the
    // configured min.
    const float scale = run(controller, settings, 32.0f, 1.0f, 400);
    CHECK(scale == Catch::Approx(settings.minScale));
    CHECK(controller.changeCount() > 0u);
}

TEST_CASE("A single spike does not move the scale", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // Settle on budget first.
    float scale = run(controller, settings, 15.0f, 1.0f, 100);
    const unsigned int changesBefore = controller.changeCount();

    // One 60 ms hitch. A median cannot be moved by a single outlier at all,
    // which is the property an EMA cannot give: to absorb a 4x spike it would
    // have to be too slow to respond to real load.
    scale = controller.update(60.0f, scale, settings);
    CHECK(scale == Catch::Approx(1.0f));
    CHECK(controller.changeCount() == changesBefore);

    // Four more hitches in a nine-sample window still leave the median at 15.
    for (int frame = 0; frame < 3; ++frame) {
        scale = controller.update(60.0f, scale, settings);
    }
    CHECK(scale == Catch::Approx(1.0f));
    CHECK(controller.changeCount() == changesBefore);
}

TEST_CASE("Headroom raises the scale back to the maximum", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // 4 ms against a 16 ms budget, starting from the floor.
    const float scale = run(controller, settings, 4.0f, 0.5f, 600);
    CHECK(scale == Catch::Approx(settings.maxScale));
}

TEST_CASE("The scale never leaves the configured bounds", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    DynamicResolutionSettings settings = enabledSettings();
    settings.minScale = 0.6f;
    settings.maxScale = 0.9f;

    CHECK(run(controller, settings, 100.0f, 0.9f, 400) >= 0.6f - 0.001f);

    controller.reset();
    CHECK(run(controller, settings, 1.0f, 0.6f, 400) <= 0.9f + 0.001f);
}

TEST_CASE("Swapped bounds are treated as an unordered pair", "[dynamic-resolution]")
{
    // A hand-edited settings file can have these the wrong way round. The
    // controller must still produce a scale inside the interval rather than
    // clamping to an empty range.
    DynamicResolutionController controller;
    DynamicResolutionSettings settings = enabledSettings();
    settings.minScale = 0.9f;
    settings.maxScale = 0.6f;

    const float scale = run(controller, settings, 100.0f, 0.9f, 400);
    CHECK(scale >= 0.6f - 0.001f);
    CHECK(scale <= 0.9f + 0.001f);
}

TEST_CASE("On-budget frame time is left alone", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // Inside the deadband: below target, but not by the margin required to raise.
    // 16 * (1 - 0.15) = 13.6, so 14.5 is on budget and nothing should happen.
    CHECK(run(controller, settings, 14.5f, 0.75f, 400) == Catch::Approx(0.75f));
    CHECK(controller.changeCount() == 0u);
}

TEST_CASE("A converged controller stops changing the scale", "[dynamic-resolution]")
{
    // The failure this guards against is a controller that oscillates across a
    // step boundary forever. Every change rebuilds every screen-sized target, so
    // an endless one-step wobble would be a permanent stutter.
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // Frame time proportional to pixel count, so the loop is closed rather than
    // fed a constant: at scale 1.0 the frame is 24 ms, which needs ~0.8.
    float scale = 1.0f;
    for (int frame = 0; frame < 500; ++frame) {
        const float frameMs = 24.0f * (scale * scale);
        scale = controller.update(frameMs, scale, settings);
    }
    const unsigned int changesAfterConverging = controller.changeCount();

    for (int frame = 0; frame < 500; ++frame) {
        const float frameMs = 24.0f * (scale * scale);
        scale = controller.update(frameMs, scale, settings);
    }
    CHECK(controller.changeCount() == changesAfterConverging);
    // And it landed on a sensible scale rather than a bound.
    CHECK(scale > settings.minScale);
    CHECK(scale < settings.maxScale);
}

TEST_CASE("No measurement means no decision", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // Non-positive samples are how Renderer says "no timestamp readback landed".
    CHECK(run(controller, settings, 0.0f, 0.8f, 100) == Catch::Approx(0.8f));
    CHECK_FALSE(controller.hasMeasurement());
    CHECK(controller.changeCount() == 0u);
}

TEST_CASE("A change is followed by a settling window", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    // Warm up past the initial settle, then force one change.
    float scale = run(controller, settings, 40.0f, 1.0f, kDynamicResolutionSettleFrames + 1);
    REQUIRE(controller.changeCount() == 1u);
    REQUIRE(scale < 1.0f);

    // The next several frames must be ignored: the GPU timestamps still describe
    // the previous resolution, so acting on them would double-count.
    const float afterChange = scale;
    for (uint32_t frame = 0; frame < kDynamicResolutionSettleFrames; ++frame) {
        scale = controller.update(40.0f, scale, settings);
        CHECK(scale == Catch::Approx(afterChange));
    }
    CHECK(controller.changeCount() == 1u);

    // And then it is allowed to act again.
    scale = controller.update(40.0f, scale, settings);
    CHECK(scale < afterChange);
    CHECK(controller.changeCount() == 2u);
}

TEST_CASE("Changes land on the quantisation grid", "[dynamic-resolution]")
{
    DynamicResolutionController controller;
    const DynamicResolutionSettings settings = enabledSettings();

    float scale = run(controller, settings, 20.0f, 1.0f, 200);
    const float steps = scale / kDynamicResolutionScaleStep;
    CHECK(steps == Catch::Approx(std::round(steps)).margin(0.01));
}

TEST_CASE("Dynamic resolution is off by default", "[dynamic-resolution][settings]")
{
    // A portfolio engine is usually being measured, and a resolution that moves
    // on its own invalidates every A/B comparison in the profiler panel.
    const ve::RuntimeSettings defaults{};
    CHECK_FALSE(defaults.dynamicResolution.enabled);
    CHECK(defaults.dynamicResolution.maxScale == Catch::Approx(kMaxRenderScale));
}

TEST_CASE("A target between two grid points does not oscillate", "[dynamic-resolution]")
{
    // The limit cycle this guards against was live in shipped code and the
    // "converged controller stops" test above missed it, because that test
    // happened to pick a target the grid could express.
    //
    // The deadband is a fraction of *time* while the step is a fraction of
    // *scale*, and time goes as scale squared, so one step moves the frame time
    // by roughly 2 * step / scale: 10% at scale 1.0, but 40% at 0.25. Below
    // about half scale no grid point need land inside the deadband at all, and
    // then every step up is over budget and every step down is under it.
    //
    // 100 ms at scale 1.0 puts 0.25 at 6.25 ms and 0.30 at 9.0 ms against an
    // 8 ms target -- no grid point fits, which is exactly the case observed in
    // practice as 640x360 <-> 768x432, sixty times in twenty seconds.
    DynamicResolutionController controller;
    DynamicResolutionSettings settings{};
    settings.enabled = true;
    settings.targetFrameMs = 8.0f;
    settings.minScale = 0.25f;
    settings.maxScale = 1.0f;

    float scale = 1.0f;
    for (int frame = 0; frame < 2000; ++frame) {
        scale = controller.update(100.0f * scale * scale, scale, settings);
    }

    const unsigned int changesAfterSettling = controller.changeCount();
    float lowest = scale;
    float highest = scale;
    for (int frame = 0; frame < 2000; ++frame) {
        scale = controller.update(100.0f * scale * scale, scale, settings);
        lowest = std::min(lowest, scale);
        highest = std::max(highest, scale);
    }

    CHECK(controller.changeCount() == changesAfterSettling);
    CHECK(lowest == Catch::Approx(highest));
    // Settling under budget is the correct answer when the grid cannot express
    // the right one. Settling *over* it would not be.
    CHECK(100.0f * scale * scale <= 8.0f);
}

TEST_CASE("A raise that would land over budget is refused", "[dynamic-resolution]")
{
    // The rule in isolation: from 0.25, one step up to 0.30 multiplies the frame
    // time by (0.30/0.25)^2 = 1.44. At 6.5 ms that predicts 9.4 ms against an
    // 8 ms target, so the raise must not happen even though 6.5 is comfortably
    // inside the "there is headroom" test that used to authorise it.
    DynamicResolutionController controller;
    DynamicResolutionSettings settings{};
    settings.enabled = true;
    settings.targetFrameMs = 8.0f;
    settings.minScale = 0.25f;
    settings.maxScale = 1.0f;

    CHECK(run(controller, settings, 6.5f, 0.25f, 400) == Catch::Approx(0.25f));
    CHECK(controller.changeCount() == 0u);

    // And the rule is not simply "never raise": with real headroom it still does.
    controller.reset();
    CHECK(run(controller, settings, 3.0f, 0.25f, 400) > 0.25f);
}
