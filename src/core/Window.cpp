#include "core/Window.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <stdexcept>
#include <string_view>
#include <utility>

namespace ve {

namespace {

std::runtime_error sdlError(std::string_view action)
{
    std::string message(action);
    message += ": ";
    message += SDL_GetError();
    return std::runtime_error(message);
}

} // namespace

Window::Window(std::string title, int width, int height)
    : title_(std::move(title))
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw sdlError("SDL_Init failed");
    }

    const SDL_WindowFlags flags = static_cast<SDL_WindowFlags>(
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    window_ = SDL_CreateWindow(title_.c_str(), width, height, flags);
    if (!window_) {
        SDL_Quit();
        throw sdlError("SDL_CreateWindow failed");
    }
}

Window::~Window()
{
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

void Window::pollEvents()
{
    SDL_Event event{};
    const SDL_WindowID windowId = SDL_GetWindowID(window_);

    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            shouldClose_ = true;
            continue;
        }

        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == windowId) {
            shouldClose_ = true;
            continue;
        }

        if ((event.type == SDL_EVENT_WINDOW_RESIZED || event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
            && event.window.windowID == windowId) {
            resized_ = true;
        }
    }
}

WindowExtent Window::framebufferExtent() const
{
    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSizeInPixels(window_, &width, &height)) {
        return {};
    }

    return {
        static_cast<uint32_t>(width > 0 ? width : 0),
        static_cast<uint32_t>(height > 0 ? height : 0)
    };
}

bool Window::isMinimized() const
{
    const WindowExtent extent = framebufferExtent();
    return extent.width == 0 || extent.height == 0;
}

std::vector<const char*> Window::requiredVulkanInstanceExtensions() const
{
    Uint32 extensionCount = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
    if (!extensions || extensionCount == 0) {
        throw sdlError("SDL_Vulkan_GetInstanceExtensions failed");
    }

    return {extensions, extensions + extensionCount};
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface)) {
        throw sdlError("SDL_Vulkan_CreateSurface failed");
    }

    return surface;
}

} // namespace ve
