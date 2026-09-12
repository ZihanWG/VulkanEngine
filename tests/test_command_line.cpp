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

TEST_CASE("Window size parses WIDTHxHEIGHT and rejects everything else")
{
    LaunchOptions config{};
    REQUIRE(parse({"--window-size", "3840x2160"}, config));
    CHECK(config.width == 3840);
    CHECK(config.height == 2160);

    // Both halves get the same trailing-garbage rejection --exit-after-frames
    // gets, because a size that silently parses as a prefix would put the whole
    // measurement at a resolution nobody asked for -- and the frame time is the
    // thing being measured.
    for (const std::string& bad : {std::string("1280"),
                                   std::string("1280x"),
                                   std::string("x720"),
                                   std::string("1280x720junk"),
                                   std::string("1280xabc"),
                                   std::string("0x720"),
                                   std::string("1280x0"),
                                   std::string("-1280x720")}) {
        LaunchOptions rejected{};
        REQUIRE_FALSE(parse({"--window-size", bad}, rejected));
    }

    LaunchOptions missingValue{};
    REQUIRE_FALSE(parse({"--window-size"}, missingValue));
}

TEST_CASE("Window size defaults to 1280x720 when not given")
{
    // The default is what every existing measurement and the golden image were
    // taken at, so it is pinned rather than left to drift with the struct.
    const LaunchOptions defaults;
    CHECK(defaults.width == 1280);
    CHECK(defaults.height == 720);
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
    REQUIRE_FALSE(
        parse({"--capture-frame", "100", "--capture-output", "out.png", "--exit-after-frames", "10"}, config));

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
        {"sunlit", ve::ScenePreset::SunlitYard},
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

// The VSM stage flag exists for the same reason --scene does: the three toggles
// live in a git-ignored settings file and behind ImGui checkboxes, and the first
// of them is startup-only, so an A/B could not be scripted without either
// overwriting the user's settings or driving the UI by hand.

TEST_CASE("VSM stages are unset unless asked for", "[command-line][vsm]")
{
    LaunchOptions config{};
    REQUIRE(parse({}, config));

    // Unset, not "off": absence must leave the persisted settings alone, while
    // --vsm off deliberately overrides them.
    REQUIRE_FALSE(config.vsm.has_value());
}

TEST_CASE("Every VSM stage name parses", "[command-line][vsm]")
{
    struct TestCase {
        const char* name;
        ve::VsmMode mode;
    };

    const TestCase cases[] = {
        {"off", ve::VsmMode::Off},
        {"mark", ve::VsmMode::Mark},
        {"render", ve::VsmMode::Render},
        {"shadows", ve::VsmMode::Shadows},
    };

    for (const TestCase& testCase : cases) {
        LaunchOptions config{};
        REQUIRE(parse({"--vsm", testCase.name}, config));
        REQUIRE(config.vsm.has_value());
        REQUIRE(*config.vsm == testCase.mode);

        // Round trip, so a log line or an error message cannot name a different
        // stage than the one the run is in.
        REQUIRE(ve::vsmModeName(testCase.mode) == testCase.name);
    }
}

TEST_CASE("An unknown VSM stage name is rejected", "[command-line][vsm]")
{
    LaunchOptions config{};
    REQUIRE_FALSE(parse({"--vsm", "shadow"}, config));

    // Rejected outright rather than falling back to a stage nobody asked for: a
    // silent fallback would report VSM numbers for a cascade run.
    REQUIRE_FALSE(config.vsm.has_value());
}

TEST_CASE("A VSM flag with no value is rejected", "[command-line][vsm]")
{
    LaunchOptions config{};
    REQUIRE_FALSE(parse({"--vsm"}, config));
}

TEST_CASE("The VSM A/B command line parses as a whole", "[command-line][vsm]")
{
    // Exactly what the A/B script runs, both halves of it.
    LaunchOptions config{};
    REQUIRE(parse({"--deterministic",
                   "--vsm",
                   "shadows",
                   "--exit-after-frames",
                   "90",
                   "--capture-frame",
                   "60",
                   "--capture-output",
                   "/tmp/vsm.png"},
                  config));
    REQUIRE(config.deterministic);
    REQUIRE(config.vsm.has_value());
    REQUIRE(*config.vsm == ve::VsmMode::Shadows);
    REQUIRE(config.captureFrame == 60);
    REQUIRE(config.captureOutput == "/tmp/vsm.png");
}

// --capture-include-ui exists because the debug panel is the one thing a
// scripted run cannot otherwise see: it is drawn after the point the regression
// capture is taken, so anything it reports and the log does not has no evidence
// path at all.

TEST_CASE("The capture excludes the overlay unless asked", "[command-line][capture]")
{
    LaunchOptions config{};
    REQUIRE(parse({"--capture-frame", "60", "--capture-output", "/tmp/f.png"}, config));
    REQUIRE_FALSE(config.captureIncludeUi);
}

TEST_CASE("The overlay can be asked for", "[command-line][capture]")
{
    LaunchOptions config{};
    REQUIRE(parse({"--capture-frame", "60", "--capture-output", "/tmp/f.png", "--capture-include-ui"}, config));
    REQUIRE(config.captureIncludeUi);
    REQUIRE(config.captureFrame == 60);
}

TEST_CASE("Asking for the overlay without a capture is rejected", "[command-line][capture]")
{
    LaunchOptions config{};
    // Ignoring it would hand back a frame that looks right and answers nothing.
    REQUIRE_FALSE(parse({"--capture-include-ui"}, config));
}

// ---------------------------------------------------------------------------
// --settings: which runtime settings file a run reads.
//
// A note on the names below: a TEST_CASE name must not *begin* with "--".
// catch_discover_tests registers each case with CTest as its own invocation,
// `VulkanEngineTests "<name>"`, and Catch2 parses a leading "--" as one of its
// own options -- so such a case fails with "Unrecognised token" under ctest
// while passing when the binary is run directly. Run ctest, not the binary, or
// this class of mistake is invisible.
//
// The point of the flag is that a configuration sweep can select a
// configuration. What makes it safe is that a path it cannot use is a refusal
// rather than a silent fall back to the defaults -- a sweep that ran the
// default configuration on every leg reports green while sweeping nothing.
// ---------------------------------------------------------------------------

TEST_CASE("No --settings leaves the per-user path in charge", "[command-line][settings]")
{
    LaunchOptions config{};
    REQUIRE(parse({}, config));
    CHECK_FALSE(config.settingsPath.has_value());
}

TEST_CASE("The --settings flag takes the path of a file that exists", "[command-line][settings]")
{
    // The repository's own example file, so the case needs no fixture and no
    // temporary directory.
    const std::string path = std::string(VULKAN_ENGINE_CONFIG_DIR) + "/runtime_settings.example.json";

    LaunchOptions config{};
    REQUIRE(parse({"--settings", path}, config));
    REQUIRE(config.settingsPath.has_value());
    CHECK(config.settingsPath->filename() == "runtime_settings.example.json");
}

TEST_CASE("The --settings flag rejects a path that names no file", "[command-line][settings]")
{
    // Rejected at parse time rather than at load time, so the run fails before
    // a window opens and the message carries the path that was wrong.
    LaunchOptions config{};
    CHECK_FALSE(parse({"--settings", "no/such/settings.json"}, config));
    CHECK_FALSE(config.settingsPath.has_value());
}

TEST_CASE("The --settings flag rejects a directory", "[command-line][settings]")
{
    // is_regular_file, not exists: a directory is readable and is not settings.
    LaunchOptions config{};
    CHECK_FALSE(parse({"--settings", std::string(VULKAN_ENGINE_CONFIG_DIR)}, config));
}

TEST_CASE("A --settings flag with no value is rejected", "[command-line][settings]")
{
    LaunchOptions config{};
    CHECK_FALSE(parse({"--settings"}, config));
}

TEST_CASE("A sweep leg's command line parses as a whole", "[command-line][settings]")
{
    // The shape the CI matrix actually runs: one configuration file alongside
    // the determinism, frame budget and validation gating the default leg uses.
    const std::string path = std::string(VULKAN_ENGINE_CONFIG_DIR) + "/runtime_settings.example.json";

    LaunchOptions config{};
    REQUIRE(parse({"--settings",
                   path,
                   "--scene",
                   "sunlit",
                   "--deterministic",
                   "--exit-after-frames",
                   "10",
                   "--fail-on-validation-error"},
                  config));

    REQUIRE(config.settingsPath.has_value());
    CHECK(config.scene == ve::ScenePreset::SunlitYard);
    CHECK(config.deterministic);
    CHECK(config.exitAfterFrames == 10);
    CHECK(config.failOnValidationError);
}
