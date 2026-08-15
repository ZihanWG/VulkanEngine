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
    if (config_.probeAliasing) {
        renderer_->logImageMemoryAliasingProbe();
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

        // After the first frame, not at init: the graph builds its transient
        // resources during beginFrame, so before then there is nothing to report.
        if (config_.probeAliasing && framesDrawn == 1) {
            renderer_->logTransientPoolReport();
        }

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