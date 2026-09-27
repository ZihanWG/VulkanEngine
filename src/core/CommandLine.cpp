#include "core/CommandLine.h"

#include "core/Logger.h"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

namespace ve {

namespace {

constexpr ScenePresetName kScenePresetNames[] = {
    {"default", ScenePreset::Default},
    {"stress", ScenePreset::Stress},
    {"fragment-stress", ScenePreset::FragmentStress},
    {"occlusion", ScenePreset::Occlusion},
    {"cornell", ScenePreset::CornellBox},
    {"sunlit", ScenePreset::SunlitYard},
    {"gpu-stress", ScenePreset::GpuStress},
    {"sponza", ScenePreset::Sponza},
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
template <typename Table> std::string knownNames(const Table& table)
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

// std::from_chars has no floating-point overload in the libc++ that Xcode 16
// ships -- the call is deleted, so it fails to compile rather than at run time
// -- so this is strtof with the same whole-string check the integer options get
// from from_chars. strtof would also skip leading whitespace, which from_chars
// never accepted, so that is refused explicitly. The engine never calls
// setlocale, so the decimal point is the "C" locale's '.'.
bool parseFloat(std::string_view text, float& value)
{
    if (text.empty() || std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        return false;
    }
    const std::string owned(text);
    char* end = nullptr;
    const float parsed = std::strtof(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size()) {
        return false;
    }
    value = parsed;
    return true;
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

std::span<const ScenePresetName> scenePresets()
{
    return kScenePresetNames;
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

        if (argument == "--sync-validation") {
            options.syncValidation = true;
            continue;
        }

        if (argument == "--sync-validation-selftest") {
            // Both, always: the self-test only means something with the check it
            // is testing turned on. See LaunchOptions::syncValidationSelfTest.
            options.syncValidationSelfTest = true;
            options.syncValidation = true;
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

        if (argument == "--meshlet-analysis") {
            options.meshletAnalysis = true;
            continue;
        }

        if (argument == "--portability-fallbacks") {
            options.portabilityFallbacks = true;
            continue;
        }

        if (argument == "--overdraw") {
            options.overdraw = true;
            continue;
        }

        if (argument == "--camera-orbit") {
            if (index + 1 >= argc) {
                Logger::error("--camera-orbit requires an angle in radians per frame.");
                return false;
            }
            const std::string_view value(argv[++index]);
            float radians = 0.0f;
            // Bounded, not just finite: past half a radian a frame the camera
            // spins faster than any history or transition could follow, which
            // is a typo rather than a test.
            if (!parseFloat(value, radians) || !std::isfinite(radians) || radians == 0.0f || std::abs(radians) > 0.5f) {
                Logger::error("--camera-orbit expects a non-zero angle within [-0.5, 0.5] radians, got: " +
                              std::string(value));
                return false;
            }
            options.cameraOrbitRadiansPerFrame = radians;
            continue;
        }

        if (argument == "--vsm-dump-pool") {
            if (index + 1 >= argc) {
                Logger::error("--vsm-dump-pool requires a file path.");
                return false;
            }
            options.vsmDumpPool = argv[++index];
            continue;
        }

        if (argument == "--settings") {
            if (index + 1 >= argc) {
                Logger::error("--settings requires a file path.");
                return false;
            }
            // Existence is checked here rather than at load time so a typo fails
            // before the window opens, with the path in the message. The loader
            // treats a missing file as "use defaults", which is right for the
            // persisted path and wrong for one somebody asked for by name.
            std::filesystem::path path(argv[++index]);
            std::error_code error;
            if (!std::filesystem::is_regular_file(path, error)) {
                Logger::error("--settings names no readable file: " + path.string());
                return false;
            }
            options.settingsPath = std::move(path);
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

        if (argument == "--window-size") {
            if (index + 1 >= argc) {
                Logger::error("--window-size requires WIDTHxHEIGHT, for example 2560x1440.");
                return false;
            }
            const std::string_view value(argv[++index]);
            const size_t separator = value.find('x');
            if (separator == std::string_view::npos) {
                Logger::error("--window-size expects WIDTHxHEIGHT, got: " + std::string(value));
                return false;
            }

            // Both halves parsed the same way --exit-after-frames parses its
            // count: reject trailing characters rather than stopping at the first
            // one that does not fit, so "1280xabc" and "1280x720junk" fail loudly.
            const auto parseExtent = [](std::string_view text, uint32_t& out) {
                const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
                return result.ec == std::errc{} && result.ptr == text.data() + text.size() && out != 0;
            };

            uint32_t parsedWidth = 0;
            uint32_t parsedHeight = 0;
            if (!parseExtent(value.substr(0, separator), parsedWidth) ||
                !parseExtent(value.substr(separator + 1), parsedHeight)) {
                Logger::error("--window-size expects two positive integers, got: " + std::string(value));
                return false;
            }
            options.width = static_cast<int>(parsedWidth);
            options.height = static_cast<int>(parsedHeight);
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
    if (!options.vsmDumpPool.empty() && options.captureFrame == 0) {
        Logger::error("--vsm-dump-pool requires --capture-frame, so the pool and the frame it shaded match.");
        return false;
    }

    if (options.captureIncludeUi && options.captureFrame == 0) {
        Logger::error("--capture-include-ui requires --capture-frame.");
        return false;
    }

    // Without this the loop would have to choose between honouring the budget
    // (and dropping the capture) or honouring the capture (and running a million
    // frames). A request that cannot be satisfied is rejected instead.
    if (options.captureFrame != 0 && options.exitAfterFrames != 0 && options.captureFrame > options.exitAfterFrames) {
        Logger::error("--capture-frame (" + std::to_string(options.captureFrame) +
                      ") must not exceed --exit-after-frames (" + std::to_string(options.exitAfterFrames) + ").");
        return false;
    }

    return true;
}

} // namespace ve
