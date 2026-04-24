#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <string>
#include <vector>

struct SDL_Window;

namespace ve {

struct WindowExtent {
    uint32_t width = 0;
    uint32_t height = 0;
};

class Window final {
public:
    Window(std::string title, int width, int height);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;
    Window& operator=(Window&&) = delete;

    void pollEvents();

    [[nodiscard]] bool shouldClose() const { return shouldClose_; }
    [[nodiscard]] bool wasResized() const { return resized_; }
    void clearResizedFlag() { resized_ = false; }

    [[nodiscard]] WindowExtent framebufferExtent() const;
    [[nodiscard]] bool isMinimized() const;

    [[nodiscard]] std::vector<const char*> requiredVulkanInstanceExtensions() const;
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;

    [[nodiscard]] SDL_Window* nativeHandle() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    std::string title_;
    bool shouldClose_ = false;
    bool resized_ = false;
};

} // namespace ve
