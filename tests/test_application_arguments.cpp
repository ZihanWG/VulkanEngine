#include "core/Application.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using ve::Application;

namespace {

// parseArguments takes the C main() signature, so tests build one. argv[0] is
// the program name and is always skipped, exactly as it is in a real run.
bool parse(const std::vector<std::string>& arguments, Application::Config& config)
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

    return Application::parseArguments(static_cast<int>(argv.size()), argv.data(), config);
}

} // namespace

TEST_CASE("No arguments leaves every default in place")
{
    Application::Config config{};
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
    Application::Config config{};
    REQUIRE(parse({"--asset-load-stats", "--fail-on-validation-error", "--deterministic"}, config));

    REQUIRE(config.assetLoadStats);
    REQUIRE(config.failOnValidationError);
    REQUIRE(config.deterministic);
}

TEST_CASE("An unrecognized argument is rejected rather than ignored")
{
    Application::Config config{};

    // Silently ignoring a typo would mean a CI run measuring something other
    // than what the workflow asked for.
    REQUIRE_FALSE(parse({"--determinstic"}, config));
}

TEST_CASE("Frame counts must be positive integers")
{
    Application::Config valid{};
    REQUIRE(parse({"--exit-after-frames", "10"}, valid));
    REQUIRE(valid.exitAfterFrames == 10);

    for (const std::string& bad : {std::string("0"), std::string("-1"), std::string("abc"), std::string("10x")}) {
        Application::Config rejected{};
        // Trailing garbage ("10x") must not be accepted as a valid prefix.
        REQUIRE_FALSE(parse({"--exit-after-frames", bad}, rejected));
    }

    Application::Config missingValue{};
    REQUIRE_FALSE(parse({"--exit-after-frames"}, missingValue));
}

TEST_CASE("Capture requires both a frame and an output path")
{
    // A fresh Config per parse, because parseArguments fills one rather than
    // resetting it -- which is exactly how main() calls it. Sharing one here
    // would let an earlier case's --capture-output satisfy a later case's
    // pairing check and hide the failure being tested.
    Application::Config bothConfig{};
    REQUIRE(parse({"--capture-frame", "30", "--capture-output", "out.png"}, bothConfig));
    REQUIRE(bothConfig.captureFrame == 30);
    REQUIRE(bothConfig.captureOutput == "out.png");

    Application::Config frameOnly{};
    REQUIRE_FALSE(parse({"--capture-frame", "30"}, frameOnly));

    Application::Config outputOnly{};
    REQUIRE_FALSE(parse({"--capture-output", "out.png"}, outputOnly));
}

TEST_CASE("A capture frame beyond the frame budget is rejected")
{
    Application::Config config{};

    // Otherwise the loop has to choose between dropping the capture and running
    // far past the budget waiting for a frame that never arrives.
    REQUIRE_FALSE(parse({"--capture-frame", "100", "--capture-output", "out.png", "--exit-after-frames", "10"}, config));

    // Equal is fine: the readback grace window covers the lag.
    Application::Config equalConfig{};
    REQUIRE(parse({"--capture-frame", "10", "--capture-output", "out.png", "--exit-after-frames", "10"}, equalConfig));
    REQUIRE(equalConfig.captureFrame == 10);

    // As is a capture comfortably inside the budget.
    Application::Config insideConfig{};
    REQUIRE(parse({"--capture-frame", "5", "--capture-output", "out.png", "--exit-after-frames", "60"}, insideConfig));
    REQUIRE(insideConfig.captureFrame == 5);
}

TEST_CASE("A capture frame without a frame budget is unconstrained")
{
    Application::Config config{};

    // With no --exit-after-frames the loop runs until the capture lands, so any
    // frame number is answerable.
    REQUIRE(parse({"--capture-frame", "100000", "--capture-output", "out.png"}, config));
    REQUIRE(config.captureFrame == 100000);
}
