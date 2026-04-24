#pragma once

#include <memory>
#include <string>

namespace ve {

class Renderer;
class Window;

class Application final {
public:
    struct Config {
        std::string title = "VulkanEngine";
        int width = 1280;
        int height = 720;
    };

    explicit Application(Config config = {});
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    int run();

private:
    void initialize();
    void mainLoop();
    void shutdown();

    Config config_;
    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;
};

} // namespace ve
