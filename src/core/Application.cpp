#include "core/Application.h"

#include "core/Logger.h"
#include "core/Window.h"
#include "renderer/AssetLoadStats.h"
#include "renderer/Renderer.h"
#include "rhi/ValidationTally.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <string_view>
#include <utility>

namespace ve {

bool Application::parseArguments(int argc, char** argv, Config& config)
{
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);

        if (argument == "--asset-load-stats") {
            config.assetLoadStats = true;
            continue;
        }

        if (argument == "--fail-on-validation-error") {
            config.failOnValidationError = true;
            continue;
        }

        if (argument == "--deterministic") {
            config.deterministic = true;
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
            config.captureFrame = frame;
            continue;
        }

        if (argument == "--capture-output") {
            if (index + 1 >= argc) {
                Logger::error("--capture-output requires a file path.");
                return false;
            }
            config.captureOutput = argv[++index];
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
            config.exitAfterFrames = frames;
            continue;
        }

        Logger::error("Unrecognized argument: " + std::string(argument));
        return false;
    }

    if ((config.captureFrame != 0) != !config.captureOutput.empty()) {
        Logger::error("--capture-frame and --capture-output must be given together.");
        return false;
    }

    // Without this the loop would have to choose between honouring the budget
    // (and dropping the capture) or honouring the capture (and running a million
    // frames). A request that cannot be satisfied is rejected instead.
    if (config.captureFrame != 0 && config.exitAfterFrames != 0 &&
        config.captureFrame > config.exitAfterFrames) {
        Logger::error("--capture-frame (" + std::to_string(config.captureFrame) +
                      ") must not exceed --exit-after-frames (" + std::to_string(config.exitAfterFrames) + ").");
        return false;
    }

    return true;
}

Application::Application()
    : Application(Config{})
{
}

Application::Application(Config config)
    : config_(std::move(config))
{
}

Application::~Application()
{
    shutdown();
}

int Application::run()
{
    try {
        initialize();
        mainLoop();
        shutdown();
        return reportValidationTally();
    } catch (const std::exception& exception) {
        Logger::error(exception.what());
        shutdown();
        return -1;
    }
}

void Application::initialize()
{
    // Enabled before the renderer exists: texture records are produced during
    // renderer construction, so flipping this any later would miss all of them.
    renderer::AssetLoadStatsRecorder::setEnabled(config_.assetLoadStats);

    window_ = std::make_unique<Window>(config_.title, config_.width, config_.height);

    const auto rendererInitStart = std::chrono::steady_clock::now();
    renderer_ = std::make_unique<Renderer>(*window_);
    rendererInitMs_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rendererInitStart).count();

    // Before the first drawFrame, which is where the clock first advances.
    if (config_.deterministic) {
        renderer_->useDeterministicFrameClock();
    }
    if (config_.captureFrame != 0) {
        renderer_->requestFrameCaptureAt(config_.captureFrame, config_.captureOutput);
    }

    window_->setEventCallback([this](const SDL_Event& event) {
        if (renderer_) {
            renderer_->handleEvent(event);
        }
    });
}

void Application::mainLoop()
{
    uint32_t framesDrawn = 0;
    double firstFrameMs = 0.0;

    while (!window_->shouldClose()) {
        window_->pollEvents();
        if (window_->shouldClose()) {
            break;
        }

        if (framesDrawn == 0) {
            const auto firstFrameStart = std::chrono::steady_clock::now();
            renderer_->drawFrame();
            firstFrameMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - firstFrameStart).count();
        } else {
            renderer_->drawFrame();
        }
        ++framesDrawn;

        const bool captureRequested = renderer_->frameCaptureRequested();
        const bool captureOutstanding = captureRequested && !renderer_->frameCaptureComplete();

        // The capture is done and nothing else was asked for.
        if (captureRequested && !captureOutstanding && config_.exitAfterFrames == 0) {
            break;
        }

        // A pending capture outranks the frame budget, because exiting at exactly
        // the capture frame would drop the readback -- it lands a few frames
        // later. parseArguments guarantees captureFrame <= exitAfterFrames, so
        // this can only ever extend the run by the grace window below.
        if (config_.exitAfterFrames != 0 && framesDrawn >= config_.exitAfterFrames && !captureOutstanding) {
            break;
        }

        // Bounds that extension. Without it, a capture that can never be
        // recorded (an unsupported swapchain format, say) would spin.
        if (captureOutstanding && framesDrawn >= config_.captureFrame + kCaptureReadbackGraceFrames) {
            Logger::error("Frame capture never completed; giving up after " + std::to_string(framesDrawn) +
                          " frames.");
            break;
        }
    }

    renderer_->waitIdle();

    captureCompleted_ = renderer_->frameCaptureComplete();

    if (config_.assetLoadStats) {
        renderer_->finalizeAssetLoadStats(rendererInitMs_, firstFrameMs);
        Logger::info(renderer::formatReport(renderer_->assetLoadStats()));
    }
}

void Application::shutdown()
{
    renderer_.reset();
    window_.reset();
}

int Application::reportValidationTally() const
{
    // Checked before validation: a run that never produced the image it was
    // asked for has not passed, whatever validation thought of it.
    if (config_.captureFrame != 0 && !captureCompleted_) {
        Logger::error("Failing because the requested frame capture was never written to " + config_.captureOutput +
                      ".");
        return kCaptureFailureExitCode;
    }

    if (!config_.failOnValidationError) {
        return 0;
    }

    const uint64_t errors = rhi::ValidationTally::errorCount();
    const uint64_t warnings = rhi::ValidationTally::warningCount();

    Logger::info("Validation tally: " + std::to_string(errors) + " error(s), " + std::to_string(warnings) +
                 " warning(s).");

    // Warnings are reported but do not fail. They move with layer and loader
    // versions, and a CI that goes red because the validation layer was updated
    // teaches people to ignore it.
    if (errors > 0) {
        Logger::error("Failing because the validation layer reported " + std::to_string(errors) + " error(s).");
        return kValidationFailureExitCode;
    }

    return 0;
}

} // namespace ve