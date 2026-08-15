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

        if (config_.exitAfterFrames != 0 && framesDrawn >= config_.exitAfterFrames) {
            break;
        }
    }

    renderer_->waitIdle();

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