#include "core/CommandLine.h"

#include "core/Logger.h"

#include <charconv>
#include <string>
#include <string_view>
#include <system_error>

namespace ve {

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
