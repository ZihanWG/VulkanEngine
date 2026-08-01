#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <filesystem>
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

    // shaderDirectory is the directory holding the compiled SPIR-V modules this
    // run will load. It is hashed into the persisted pipeline cache so the blob
    // is discarded whenever the shaders change; pass an empty path to fall back
    // to device-identity-only validation.
    void initialize(VkInstance instance, VkSurfaceKHR surface, std::filesystem::path shaderDirectory);
    void cleanup();

    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkPipelineCache pipelineCache() const { return pipelineCache_; }
    [[nodiscard]] VkQueue graphicsQueue() const { return graphicsQueue_; }
    [[nodiscard]] VkQueue presentQueue() const { return presentQueue_; }
    // Async compute queue for overlapping compute with rasterization. Prefers a
    // dedicated compute-only family; falls back to a second queue in the
    // graphics family; unavailable when the device exposes neither.
    [[nodiscard]] VkQueue asyncComputeQueue() const { return asyncComputeQueue_; }
    [[nodiscard]] uint32_t asyncComputeQueueFamily() const { return asyncComputeQueueFamily_; }
    [[nodiscard]] bool asyncComputeAvailable() const { return asyncComputeAvailable_; }
    [[nodiscard]] bool asyncComputeDedicatedFamily() const { return asyncComputeDedicatedFamily_; }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const { return queueFamilies_; }
    [[nodiscard]] bool descriptorIndexingEnabled() const { return descriptorIndexingEnabled_; }
    [[nodiscard]] bool descriptorUpdateAfterBindEnabled() const { return descriptorUpdateAfterBindEnabled_; }
    [[nodiscard]] bool bufferDeviceAddressEnabled() const { return bufferDeviceAddressEnabled_; }
    [[nodiscard]] bool multiDrawIndirectEnabled() const { return multiDrawIndirectEnabled_; }
    [[nodiscard]] bool drawIndirectFirstInstanceEnabled() const { return drawIndirectFirstInstanceEnabled_; }
    [[nodiscard]] bool drawIndexedIndirectCountAvailable() const { return drawIndexedIndirectCountAvailable_; }
    [[nodiscard]] uint32_t maxDrawIndirectCount() const { return maxDrawIndirectCount_; }

    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport() const;

private:
    void pickPhysicalDevice();
    bool isDeviceSuitable(VkPhysicalDevice candidate) const;
    int scoreDevice(VkPhysicalDevice candidate) const;
    void createLogicalDevice();
    void createPipelineCache();
    void destroyPipelineCache();

    [[nodiscard]] QueueFamilyIndices findQueueFamilies(VkPhysicalDevice candidate) const;
    [[nodiscard]] bool checkDeviceExtensionSupport(VkPhysicalDevice candidate) const;
    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice candidate) const;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::filesystem::path shaderDirectory_;
    // Digest of shaderDirectory_ taken when the cache was created; reused when
    // saving so the blob is stamped with the shaders its pipelines came from.
    uint64_t shaderHash_ = 0;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue asyncComputeQueue_ = VK_NULL_HANDLE;
    uint32_t asyncComputeQueueFamily_ = UINT32_MAX;
    bool asyncComputeAvailable_ = false;
    bool asyncComputeDedicatedFamily_ = false;
    QueueFamilyIndices queueFamilies_{};
    bool descriptorIndexingEnabled_ = false;
    bool descriptorUpdateAfterBindEnabled_ = false;
    bool bufferDeviceAddressEnabled_ = false;
    bool multiDrawIndirectEnabled_ = false;
    bool drawIndirectFirstInstanceEnabled_ = false;
    bool drawIndexedIndirectCountAvailable_ = false;
    uint32_t maxDrawIndirectCount_ = 0;
};

} // namespace ve::rhi
