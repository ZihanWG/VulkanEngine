#include "renderer/LodTransition.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using ve::renderer::advanceLodTransition;
using ve::renderer::kLodFadeSteps;
using ve::renderer::lodDitherKeeps;
using ve::renderer::lodTransitionIdentity;
using ve::renderer::LodTransitionState;
using ve::renderer::LodTransitionStep;

namespace {

constexpr uint32_t kIdentity = 0x1234u;
// A quarter of the fade per frame: four frames from one level to the next.
constexpr float kStep = 0.25f;

// Runs `frames` consecutive frames selecting `level`, starting at `frame`.
LodTransitionStep settle(LodTransitionState& state, uint32_t& frame, uint32_t level, int frames)
{
    LodTransitionStep step{};
    for (int i = 0; i < frames; ++i) {
        step = advanceLodTransition(state, kIdentity, level, frame++, kStep);
        state = step.state;
    }
    return step;
}

} // namespace

TEST_CASE("A draw item seen for the first time snaps to its level", "[lod][transition]")
{
    // A zeroed record, which is what the buffer holds before any frame wrote it.
    const LodTransitionStep step = advanceLodTransition(LodTransitionState{}, kIdentity, 2, 10, kStep);
    CHECK(step.level == 2);
    CHECK_FALSE(step.fading);
    CHECK(step.state.lastFrame == 10);
    CHECK(step.state.identity == kIdentity);
    CHECK(step.state.progress == 1.0f);
}

TEST_CASE("A level change fades across the configured number of frames", "[lod][transition]")
{
    LodTransitionState state{};
    uint32_t frame = 5;
    settle(state, frame, 1, 3);

    LodTransitionStep step = advanceLodTransition(state, kIdentity, 2, frame++, kStep);
    REQUIRE(step.fading);
    CHECK(step.level == 2);
    CHECK(step.previousLevel == 1);
    CHECK(step.fade == 32); // a quarter of 127, rounded

    // Three more frames at a quarter each: the fourth completes it.
    for (int i = 0; i < 2; ++i) {
        state = step.state;
        step = advanceLodTransition(state, kIdentity, 2, frame++, kStep);
        CHECK(step.fading);
    }
    state = step.state;
    step = advanceLodTransition(state, kIdentity, 2, frame++, kStep);
    CHECK_FALSE(step.fading);
    CHECK(step.level == 2);
    CHECK(step.previousLevel == 2);
}

TEST_CASE("Turning back mid-fade runs the fade backwards instead of popping", "[lod][transition]")
{
    LodTransitionState state{};
    uint32_t frame = 5;
    settle(state, frame, 1, 2);

    // Two frames into a fade from 1 to 2: level 2 covers half the pixels.
    settle(state, frame, 2, 2);
    CHECK(state.progress == 0.5f);

    // Back to 1. The fade reverses from where it stood, so level 1 goes on to
    // cover 0.5 + one step, not a fresh single step.
    const LodTransitionStep step = advanceLodTransition(state, kIdentity, 1, frame++, kStep);
    REQUIRE(step.fading);
    CHECK(step.level == 1);
    CHECK(step.previousLevel == 2);
    CHECK(step.state.progress == 0.75f);
}

TEST_CASE("A third level mid-fade fades out of whichever level dominates", "[lod][transition]")
{
    LodTransitionState state{};
    uint32_t frame = 5;
    settle(state, frame, 0, 2);

    // One step into 0 -> 1: level 0 still has three quarters of the pixels.
    settle(state, frame, 1, 1);
    LodTransitionStep step = advanceLodTransition(state, kIdentity, 2, frame++, kStep);
    CHECK(step.level == 2);
    CHECK(step.previousLevel == 0);

    // Three steps into 1 -> 2 before 3 arrives: level 2 dominates by then.
    state = step.state;
    settle(state, frame, 2, 2);
    step = advanceLodTransition(state, kIdentity, 3, frame++, kStep);
    CHECK(step.level == 3);
    CHECK(step.previousLevel == 2);
}

TEST_CASE("A record for a different draw item or a broken run of frames snaps", "[lod][transition]")
{
    LodTransitionState state{};
    uint32_t frame = 5;
    settle(state, frame, 1, 2);

    // The scene changed and this slot now holds another object.
    CHECK_FALSE(advanceLodTransition(state, kIdentity + 1, 2, frame, kStep).fading);
    // The item was culled for a frame and comes back at a new level.
    CHECK_FALSE(advanceLodTransition(state, kIdentity, 2, frame + 1, kStep).fading);
    // The same frame again is not a continuation either.
    CHECK_FALSE(advanceLodTransition(state, kIdentity, 2, frame - 1, kStep).fading);
    // The continuation itself does fade -- the three above are not vacuous.
    CHECK(advanceLodTransition(state, kIdentity, 2, frame, kStep).fading);
}

TEST_CASE("A zero step switches transitions off", "[lod][transition]")
{
    LodTransitionState state{};
    uint32_t frame = 5;
    settle(state, frame, 1, 2);
    const LodTransitionStep step = advanceLodTransition(state, kIdentity, 2, frame, 0.0f);
    CHECK_FALSE(step.fading);
    CHECK(step.level == 2);
}

TEST_CASE("The two halves of a fade dither into exactly one survivor per pixel", "[lod][transition]")
{
    for (uint32_t fade = 0; fade <= kLodFadeSteps; ++fade) {
        for (int sample = 0; sample < 1024; ++sample) {
            const float noise = static_cast<float>(sample) / 1024.0f;
            const bool incoming = lodDitherKeeps(false, noise, fade);
            const bool outgoing = lodDitherKeeps(true, noise, fade);
            REQUIRE(incoming != outgoing);
        }
    }
    // The endpoints: nothing of the incoming level at 0, all of it at the top.
    CHECK_FALSE(lodDitherKeeps(false, 0.0f, 0));
    CHECK(lodDitherKeeps(false, 0.999f, kLodFadeSteps));
}

TEST_CASE("Draw item identity separates object slots and authored ranges", "[lod][transition]")
{
    CHECK(lodTransitionIdentity(3, 100) != lodTransitionIdentity(4, 100));
    CHECK(lodTransitionIdentity(3, 100) != lodTransitionIdentity(3, 103));
    // A zeroed record must not look like any live draw item's slot 0, range 0.
    CHECK(lodTransitionIdentity(0, 0) != 0u);
}
