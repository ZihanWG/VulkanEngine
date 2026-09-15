#pragma once

#include "rhi/VulkanDevice.h"
#include "rhi/VulkanMemory.h"

#include <filesystem>
#include <vector>

namespace ve {
class Window;
}

namespace ve::rhi {

// Instance-creation policy the caller decides, rather than the context.
struct VulkanContextOptions {
    // Turns on the validation layer's synchronization validation, which checks
    // that the ordering between accesses to a resource is actually established
    // by a barrier or a semaphore rather than merely happening to work.
    //
    // OFF by default, and not because it is optional in principle. It is the only
    // independent oracle for the render graph's inferred barriers -- the unit
    // tests cover the pure derivation functions, and core validation checks that
    // each call is well formed, but neither can see a missing dependency. It
    // costs real CPU time inside the layer, so it belongs on a scripted run
    // rather than on every launch.
    //
    // Requested on a build with the validation layer compiled out, or on a
    // loader that has no VK_EXT_layer_settings to configure it through, this is
    // a hard failure. Continuing would render a normal frame and report success
    // for a check that never ran, which is the one outcome worse than not
    // checking.
    bool synchronizationValidation = false;
};

class VulkanContext final {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    // shaderDirectory is forwarded to VulkanDevice, which hashes the compiled
    // SPIR-V in it to key the persisted pipeline cache.
    void initialize(const Window& window, std::filesystem::path shaderDirectory, VulkanContextOptions options = {});
    void cleanup();
    void waitIdle() const;

    [[nodiscard]] VkInstance instance() const
    {
        return instance_;
    }
    [[nodiscard]] VkSurfaceKHR surface() const
    {
        return surface_;
    }
    [[nodiscard]] VkDevice vkDevice() const
    {
        return device_.device();
    }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const
    {
        return device_.physicalDevice();
    }
    [[nodiscard]] VkPipelineCache pipelineCache() const
    {
        return device_.pipelineCache();
    }
    [[nodiscard]] VkQueue graphicsQueue() const
    {
        return device_.graphicsQueue();
    }
    [[nodiscard]] VkQueue presentQueue() const
    {
        return device_.presentQueue();
    }
    [[nodiscard]] VkQueue asyncComputeQueue() const
    {
        return device_.asyncComputeQueue();
    }
    [[nodiscard]] uint32_t asyncComputeQueueFamily() const
    {
        return device_.asyncComputeQueueFamily();
    }
    [[nodiscard]] bool asyncComputeAvailable() const
    {
        return device_.asyncComputeAvailable();
    }
    [[nodiscard]] VkQueue transferQueue() const
    {
        return device_.transferQueue();
    }
    [[nodiscard]] uint32_t transferQueueFamily() const
    {
        return device_.transferQueueFamily();
    }
    [[nodiscard]] bool transferQueueAvailable() const
    {
        return device_.transferQueueAvailable();
    }
    [[nodiscard]] VmaAllocator allocator() const
    {
        return allocator_;
    }
    [[nodiscard]] const VulkanDevice& device() const
    {
        return device_;
    }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const
    {
        return device_.queueFamilies();
    }

private:
    // The instance is the process-level Vulkan entry point. It must outlive the surface and device.
    void createInstance(const Window& window);
    void setupDebugMessenger();
    void createAllocator();

    [[nodiscard]] bool validationLayersAvailable() const;
    [[nodiscard]] std::vector<const char*> requiredInstanceExtensions(const Window& window) const;

    VulkanContextOptions options_{};

    VkInstance instance_ = VK_NULL_HANDLE;

    // Debug utils routes validation layer messages into the engine logger in Debug builds.
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

    // The presentation surface is created by SDL3 and consumed by device selection and the swapchain.
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    // Logical device ownership is grouped here so every device child can be destroyed before cleanup.
    VulkanDevice device_;

    // VMA centralizes Vulkan memory allocation for images and buffers.
    VmaVulkanFunctions vmaVulkanFunctions_{};
    VmaAllocator allocator_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
