#include "core/CommandLine.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using ve::LaunchOptions;
using ve::parseLaunchOptions;

namespace {

// parseLaunchOptions takes the C main() signature, so tests build one. argv[0]
// is the program name and is always skipped, exactly as it is in a real run.
//
// This deliberately calls the core/CommandLine entry point rather than anything
// on Application: reaching into Application.cpp would pull Window, and with it
// an SDL3 import that breaks this executable's startup on Windows.
bool parse(const std::vector<std::string>& arguments, LaunchOptions& config)
{
    std::vector<std::string> owned;
    owned.reserve(arguments.size() + 1);
    owned.emplace_back("VulkanEngine");
    owned.insert(owned.end(), arguments.begin(), arguments.end());

    std::vector<char*> argv;
    argv.reserve(owned.size());
    for (std::string& argument : owned) {
        argv.push_back(argument.data());
    }

    return parseLaunchOptions(static_cast<int>(argv.size()), argv.data(), config);
}

} // namespace

TEST_CASE("No arguments leaves every default in place")
{
    LaunchOptions config{};
    REQUIRE(parse({}, config));

    REQUIRE_FALSE(config.assetLoadStats);
    REQUIRE_FALSE(config.failOnValidationError);
    REQUIRE_FALSE(config.deterministic);
    REQUIRE(config.exitAfterFrames == 0);
    REQUIRE(config.captureFrame == 0);
    REQUIRE(config.captureOutput.empty());
}

TEST_CASE("The boolean flags are recognized")
{
    LaunchOptions config{};
    REQUIRE(parse({"--asset-load-stats", "--fail-on-validation-error", "--deterministic"}, config));

    REQUIRE(config.assetLoadStats);
    REQUIRE(config.failOnValidationError);
    REQUIRE(config.deterministic);
}

TEST_CASE("An unrecognized argument is rejected rather than ignored")
{
    LaunchOptions config{};

    // Silently ignoring a typo would mean a CI run measuring something other
    // than what the workflow asked for.
    REQUIRE_FALSE(parse({"--determinstic"}, config));
}

TEST_CASE("Frame counts must be positive integers")
{
    LaunchOptions valid{};
    REQUIRE(parse({"--exit-after-frames", "10"}, valid));
    REQUIRE(valid.exitAfterFrames == 10);

    for (const std::string& bad : {std::string("0"), std::string("-1"), std::string("abc"), std::string("10x")}) {
        LaunchOptions rejected{};
        // Trailing garbage ("10x") must not be accepted as a valid prefix.
        REQUIRE_FALSE(parse({"--exit-after-frames", bad}, rejected));
    }

    LaunchOptions missingValue{};
    REQUIRE_FALSE(parse({"--exit-after-frames"}, missingValue));
}

TEST_CASE("Capture requires both a frame and an output path")
{
    // A fresh Config per parse, because parseArguments fills one rather than
    // resetting it -- which is exactly how main() calls it. Sharing one here
    // would let an earlier case's --capture-output satisfy a later case's
    // pairing check and hide the failure being tested.
    LaunchOptions bothConfig{};
    REQUIRE(parse({"--capture-frame", "30", "--capture-output", "out.png"}, bothConfig));
    REQUIRE(bothConfig.captureFrame == 30);
    REQUIRE(bothConfig.captureOutput == "out.png");

    LaunchOptions frameOnly{};
    REQUIRE_FALSE(parse({"--capture-frame", "30"}, frameOnly));

    LaunchOptions outputOnly{};
    REQUIRE_FALSE(parse({"--capture-output", "out.png"}, outputOnly));
}

TEST_CASE("A capture frame beyond the frame budget is rejected")
{
    LaunchOptions config{};

    // Otherwise the loop has to choose between dropping the capture and running
    // far past the budget waiting for a frame that never arrives.
    REQUIRE_FALSE(parse({"--capture-frame", "100", "--capture-output", "out.png", "--exit-after-frames", "10"}, config));

    // Equal is fine: the readback grace window covers the lag.
    LaunchOptions equalConfig{};
    REQUIRE(parse({"--capture-frame", "10", "--capture-output", "out.png", "--exit-after-frames", "10"}, equalConfig));
    REQUIRE(equalConfig.captureFrame == 10);

    // As is a capture comfortably inside the budget.
    LaunchOptions insideConfig{};
    REQUIRE(parse({"--capture-frame", "5", "--capture-output", "out.png", "--exit-after-frames", "60"}, insideConfig));
    REQUIRE(insideConfig.captureFrame == 5);
}

TEST_CASE("A capture frame without a frame budget is unconstrained")
{
    LaunchOptions config{};

    // With no --exit-after-frames the loop runs until the capture lands, so any
    // frame number is answerable.
    REQUIRE(parse({"--capture-frame", "100000", "--capture-output", "out.png"}, config));
    REQUIRE(config.captureFrame == 100000);
}

// --- Scene preset selection -----------------------------------------------
//
// The presets were previously reachable only from ImGui buttons, so the one
// scene that loads the CPU frame path could not be measured without a human
// driving the UI. An unknown name has to fail rather than fall back: silently
// running the default scene would report timings for the wrong workload, which
// is worse than not running at all.

TEST_CASE("Scene preset defaults to the built-in scene", "[command-line][scene]")
{
    LaunchOptions config{};
    REQUIRE(parse({}, config));
    REQUIRE(config.scene == ve::ScenePreset::Default);
}

TEST_CASE("Every scene preset name parses", "[command-line][scene]")
{
    struct Case {
        const char* name;
        ve::ScenePreset preset;
    };

    const Case cases[] = {
        {"default", ve::ScenePreset::Default},
        {"stress", ve::ScenePreset::Stress},
        {"fragment-stress", ve::ScenePreset::FragmentStress},
        {"occlusion", ve::ScenePreset::Occlusion},
        {"cornell", ve::ScenePreset::CornellBox},
    };

    for (const Case& testCase : cases) {
        LaunchOptions config{};
        REQUIRE(parse({"--scene", testCase.name}, config));
        REQUIRE(config.scene == testCase.preset);

        // Round trip: the name a preset reports must parse back to that preset,
        // or an error message would name a different scene than the one running.
        REQUIRE(ve::scenePresetName(testCase.preset) == testCase.name);
    }
}

TEST_CASE("An unknown scene name is rejected", "[command-line][scene]")
{
    LaunchOptions config{};
    REQUIRE_FALSE(parse({"--scene", "stres"}, config));

    // And it must not have half-applied anything on the way out.
    REQUIRE(config.scene == ve::ScenePreset::Default);
}

TEST_CASE("A scene flag with no value is rejected", "[command-line][scene]")
{
    LaunchOptions config{};
    REQUIRE_FALSE(parse({"--scene"}, config));
}

TEST_CASE("Scene selection composes with the measurement flags", "[command-line][scene]")
{
    // The combination a scripted measurement actually uses.
    LaunchOptions config{};
    REQUIRE(parse({"--scene", "stress", "--deterministic", "--exit-after-frames", "120"}, config));
    REQUIRE(config.scene == ve::ScenePreset::Stress);
    REQUIRE(config.deterministic);
    REQUIRE(config.exitAfterFrames == 120);
}
