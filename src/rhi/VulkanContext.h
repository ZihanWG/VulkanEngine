#pragma once

#include "rhi/VulkanDevice.h"
#include "rhi/VulkanMemory.h"

#include <vector>

namespace ve {
class Window;
}

namespace ve::rhi {

class VulkanContext final {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    void initialize(const Window& window);
    void cleanup();
    void waitIdle() const;

    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkSurfaceKHR surface() const { return surface_; }
    [[nodiscard]] VkDevice vkDevice() const { return device_.device(); }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return device_.physicalDevice(); }
    [[nodiscard]] VkQueue graphicsQueue() const { return device_.graphicsQueue(); }
    [[nodiscard]] VkQueue presentQueue() const { return device_.presentQueue(); }
    [[nodiscard]] VmaAllocator allocator() const { return allocator_; }
    [[nodiscard]] const VulkanDevice& device() const { return device_; }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const { return device_.queueFamilies(); }

private:
    // The instance is the process-level Vulkan entry point. It must outlive the surface and device.
    void createInstance(const Window& window);
    void setupDebugMessenger();
    void createAllocator();

    [[nodiscard]] bool validationLayersAvailable() const;
    [[nodiscard]] std::vector<const char*> requiredInstanceExtensions(const Window& window) const;

    VkInstance instance_ = VK_NULL_HANDLE;

    // Debug utils routes validation layer messages into the engine logger in Debug builds.
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

    // The presentation surface is created by SDL3 and consumed by device selection and the swapchain.
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    // Logical device ownership is grouped here so every device child can be destroyed before cleanup.
    VulkanDevice device_;

    // VMA centralizes Vulkan memory allocation for images and buffers.
    VmaAllocator allocator_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
