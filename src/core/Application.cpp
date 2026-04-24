#include "core/Application.h"

#include "core/Logger.h"
#include "core/Window.h"
#include "renderer/Renderer.h"

#include <exception>
#include <utility>

namespace ve {

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
        return 0;
    } catch (const std::exception& exception) {
        Logger::error(exception.what());
        shutdown();
        return -1;
    }
}

void Application::initialize()
{
    window_ = std::make_unique<Window>(config_.title, config_.width, config_.height);
    renderer_ = std::make_unique<Renderer>(*window_);
}

void Application::mainLoop()
{
    while (!window_->shouldClose()) {
        window_->pollEvents();
        if (window_->shouldClose()) {
            break;
        }
        renderer_->drawFrame();
    }

    renderer_->waitIdle();
}

void Application::shutdown()
{
    renderer_.reset();
    window_.reset();
}

} // namespace ve
