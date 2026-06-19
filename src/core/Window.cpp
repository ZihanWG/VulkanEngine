#include "core/Window.h"

#include "core/Logger.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ve {

namespace {

std::runtime_error sdlError(std::string_view action)
{
    std::string message(action);
    message += ": ";
    message += SDL_GetError();
    return std::runtime_error(message);
}

std::string sdlVersionString(int version)
{
    return std::to_string(SDL_VERSIONNUM_MAJOR(version)) + "." +
           std::to_string(SDL_VERSIONNUM_MINOR(version)) + "." +
           std::to_string(SDL_VERSIONNUM_MICRO(version));
}

std::string pointerString(const void* pointer)
{
    std::ostringstream stream;
    stream << pointer;
    return stream.str();
}

bool tryLoadSdlVulkanLibrary(const char* path, std::string& lastError)
{
    const std::string description = path != nullptr ? path : "SDL default Vulkan loader";
    Logger::info("Loading SDL Vulkan library: " + description);

    if (SDL_Vulkan_LoadLibrary(path)) {
        Logger::info("SDL_Vulkan_LoadLibrary succeeded: " + description);
        return true;
    }

    lastError = SDL_GetError();
    Logger::warn("SDL_Vulkan_LoadLibrary failed for " + description + ": " + lastError);
    return false;
}

void loadSdlVulkanLibrary()
{
    std::string lastError;

#if defined(__APPLE__)
    const char* vulkanSdk = std::getenv("VULKAN_SDK");
    if (vulkanSdk != nullptr && vulkanSdk[0] != '\0') {
        const std::string sdkPath(vulkanSdk);
        const std::vector<std::string> loaderCandidates = {
            sdkPath + "/lib/libvulkan.1.dylib",
            sdkPath + "/lib/libvulkan.dylib"
        };

        for (const std::string& loaderPath : loaderCandidates) {
            if (tryLoadSdlVulkanLibrary(loaderPath.c_str(), lastError)) {
                return;
            }
        }

        throw std::runtime_error("SDL_Vulkan_LoadLibrary failed for Vulkan SDK loader candidates under " +
                                 sdkPath + ". Last SDL error: " + lastError);
    }

    Logger::warn("VULKAN_SDK is not set; falling back to SDL's default Vulkan loader search path.");
#endif

    if (!tryLoadSdlVulkanLibrary(nullptr, lastError)) {
        throw std::runtime_error("SDL_Vulkan_LoadLibrary failed: " + lastError);
    }
}

} // namespace

Window::Window(std::string title, int width, int height)
    : title_(std::move(title))
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw sdlError("SDL_Init failed");
    }

    Logger::info("SDL runtime version: " + sdlVersionString(SDL_GetVersion()));
    const char* videoDriver = SDL_GetCurrentVideoDriver();
    Logger::info(std::string("SDL current video driver: ") + (videoDriver != nullptr ? videoDriver : "<none>"));

    try {
        loadSdlVulkanLibrary();
        vulkanLibraryLoaded_ = true;
    } catch (...) {
        SDL_Quit();
        throw;
    }

    if (vulkanGetInstanceProcAddr() == nullptr) {
        SDL_Vulkan_UnloadLibrary();
        vulkanLibraryLoaded_ = false;
        SDL_Quit();
        throw sdlError("SDL_Vulkan_GetVkGetInstanceProcAddr failed");
    }

    const SDL_WindowFlags flags = static_cast<SDL_WindowFlags>(
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    window_ = SDL_CreateWindow(title_.c_str(), width, height, flags);
    if (!window_) {
        SDL_Vulkan_UnloadLibrary();
        vulkanLibraryLoaded_ = false;
        SDL_Quit();
        throw sdlError("SDL_CreateWindow failed");
    }
    Logger::info("SDL window created: " + pointerString(window_));
}

Window::~Window()
{
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    if (vulkanLibraryLoaded_) {
        SDL_Vulkan_UnloadLibrary();
        vulkanLibraryLoaded_ = false;
    }
    SDL_Quit();
}

void Window::pollEvents()
{
    SDL_Event event{};
    const SDL_WindowID windowId = SDL_GetWindowID(window_);

    while (SDL_PollEvent(&event)) {
        if (eventCallback_) {
            eventCallback_(event);
        }

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
    Logger::info("Creating SDL Vulkan surface for window=" + pointerString(window_) +
                 " instance=" + pointerString(static_cast<const void*>(instance)));

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window_, instance, nullptr, &surface)) {
        Logger::error(std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError());
        throw sdlError("SDL_Vulkan_CreateSurface failed");
    }

    return surface;
}

PFN_vkGetInstanceProcAddr Window::vulkanGetInstanceProcAddr() const
{
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
}

} // namespace ve
