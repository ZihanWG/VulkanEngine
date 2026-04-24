#pragma once

#include "rhi/VulkanCommon.h"

#include <optional>
#include <vector>

namespace ve::rhi {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool isComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanDevice final {
public:
    VulkanDevice() = default;
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

    void initialize(VkInstance instance, VkSurfaceKHR surface);
    void cleanup();

    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue graphicsQueue() const { return graphicsQueue_; }
    [[nodiscard]] VkQueue presentQueue() const { return presentQueue_; }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const { return queueFamilies_; }
    [[nodiscard]] bool descriptorIndexingEnabled() const { return descriptorIndexingEnabled_; }
    [[nodiscard]] bool bufferDeviceAddressEnabled() const { return bufferDeviceAddressEnabled_; }

    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport() const;

private:
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice candidate) const;
    int scoreDevice(VkPhysicalDevice candidate) const;
    void createLogicalDevice();

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(VkPhysicalDevice candidate) const;
    [[nodiscard]] bool checkDeviceExtensionSupport(VkPhysicalDevice candidate) const;
    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice candidate) const;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    QueueFamilyIndices queueFamilies_{};
    bool descriptorIndexingEnabled_ = false;
    bool bufferDeviceAddressEnabled_ = false;
};

} // namespace ve::rhi
