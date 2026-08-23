#include "core/CommandLine.h"

#include "core/Logger.h"

#include <charconv>
#include <string>
#include <string_view>
#include <system_error>

namespace ve {

namespace {

// One table, read by both directions, so a name can never parse to one preset
// and print back as another.
struct ScenePresetName {
    std::string_view name;
    ScenePreset preset;
};

constexpr ScenePresetName kScenePresetNames[] = {
    {"default", ScenePreset::Default},
    {"stress", ScenePreset::Stress},
    {"fragment-stress", ScenePreset::FragmentStress},
    {"occlusion", ScenePreset::Occlusion},
    {"cornell", ScenePreset::CornellBox},
};

// Same one-table rule as the scene presets, for the same reason.
struct VsmModeName {
    std::string_view name;
    VsmMode mode;
};

constexpr VsmModeName kVsmModeNames[] = {
    {"off", VsmMode::Off},
    {"mark", VsmMode::Mark},
    {"render", VsmMode::Render},
    {"shadows", VsmMode::Shadows},
};

// Shared by both --scene and --vsm so an unknown value reports what it could
// have been instead of only what it was.
template <typename Table>
std::string knownNames(const Table& table)
{
    std::string known;
    for (const auto& entry : table) {
        if (!known.empty()) {
            known += ", ";
        }
        known += entry.name;
    }
    return known;
}

} // namespace

bool parseVsmMode(std::string_view name, VsmMode& mode)
{
    for (const VsmModeName& entry : kVsmModeNames) {
        if (entry.name == name) {
            mode = entry.mode;
            return true;
        }
    }
    return false;
}

std::string_view vsmModeName(VsmMode mode)
{
    for (const VsmModeName& entry : kVsmModeNames) {
        if (entry.mode == mode) {
            return entry.name;
        }
    }
    return "off";
}

bool parseScenePreset(std::string_view name, ScenePreset& preset)
{
    for (const ScenePresetName& entry : kScenePresetNames) {
        if (entry.name == name) {
            preset = entry.preset;
            return true;
        }
    }
    return false;
}

std::string_view scenePresetName(ScenePreset preset)
{
    for (const ScenePresetName& entry : kScenePresetNames) {
        if (entry.preset == preset) {
            return entry.name;
        }
    }
    return "default";
}

bool parseLaunchOptions(int argc, char** argv, LaunchOptions& options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);

        if (argument == "--asset-load-stats") {
            options.assetLoadStats = true;
            continue;
        }

        if (argument == "--fail-on-validation-error") {
            options.failOnValidationError = true;
            continue;
        }

        if (argument == "--probe-aliasing") {
            options.probeAliasing = true;
            continue;
        }

        if (argument == "--scene") {
            if (index + 1 >= argc) {
                Logger::error("--scene requires a preset name.");
                return false;
            }
            const std::string_view value(argv[++index]);
            if (!parseScenePreset(value, options.scene)) {
                Logger::error("--scene expects one of: " + knownNames(kScenePresetNames) +
                              "; got: " + std::string(value));
                return false;
            }
            continue;
        }

        if (argument == "--vsm") {
            if (index + 1 >= argc) {
                Logger::error("--vsm requires a stage name.");
                return false;
            }
            const std::string_view value(argv[++index]);
            VsmMode mode = VsmMode::Off;
            if (!parseVsmMode(value, mode)) {
                Logger::error("--vsm expects one of: " + knownNames(kVsmModeNames) + "; got: " + std::string(value));
                return false;
            }
            options.vsm = mode;
            continue;
        }

        if (argument == "--deterministic") {
            options.deterministic = true;
            continue;
        }

        if (argument == "--capture-frame") {
            if (index + 1 >= argc) {
                Logger::error("--capture-frame requires a frame number.");
                return false;
            }
            const std::string_view value(argv[++index]);
            uint64_t frame = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), frame);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || frame == 0) {
                Logger::error("--capture-frame expects a positive integer, got: " + std::string(value));
                return false;
            }
            options.captureFrame = frame;
            continue;
        }

        if (argument == "--capture-include-ui") {
            options.captureIncludeUi = true;
            continue;
        }

        if (argument == "--capture-output") {
            if (index + 1 >= argc) {
                Logger::error("--capture-output requires a file path.");
                return false;
            }
            options.captureOutput = argv[++index];
            continue;
        }

        if (argument == "--exit-after-frames") {
            if (index + 1 >= argc) {
                Logger::error("--exit-after-frames requires a frame count.");
                return false;
            }
            const std::string_view value(argv[++index]);
            uint32_t frames = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), frames);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || frames == 0) {
                Logger::error("--exit-after-frames expects a positive integer, got: " + std::string(value));
                return false;
            }
            options.exitAfterFrames = frames;
            continue;
        }

        Logger::error("Unrecognized argument: " + std::string(argument));
        return false;
    }

    if ((options.captureFrame != 0) != !options.captureOutput.empty()) {
        Logger::error("--capture-frame and --capture-output must be given together.");
        return false;
    }

    // Rejected rather than ignored: a run asking for the overlay in an image it
    // never captures has been misconfigured, and silently dropping the flag
    // would hand back a frame that looks right and answers nothing.
    if (options.captureIncludeUi && options.captureFrame == 0) {
        Logger::error("--capture-include-ui requires --capture-frame.");
        return false;
    }

    // Without this the loop would have to choose between honouring the budget
    // (and dropping the capture) or honouring the capture (and running a million
    // frames). A request that cannot be satisfied is rejected instead.
    if (options.captureFrame != 0 && options.exitAfterFrames != 0 &&
        options.captureFrame > options.exitAfterFrames) {
        Logger::error("--capture-frame (" + std::to_string(options.captureFrame) +
                      ") must not exceed --exit-after-frames (" + std::to_string(options.exitAfterFrames) + ").");
        return false;
    }

    return true;
}

} // namespace ve
