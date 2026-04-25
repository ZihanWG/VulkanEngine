#pragma once

#include "rhi/VulkanCommon.h"

#include <span>

namespace ve::rhi {

class VulkanDescriptorSetLayout final {
public:
    VulkanDescriptorSetLayout() = default;
    ~VulkanDescriptorSetLayout();

    VulkanDescriptorSetLayout(const VulkanDescriptorSetLayout&) = delete;
    VulkanDescriptorSetLayout& operator=(const VulkanDescriptorSetLayout&) = delete;
    VulkanDescriptorSetLayout(VulkanDescriptorSetLayout&& other) noexcept;
    VulkanDescriptorSetLayout& operator=(VulkanDescriptorSetLayout&& other) noexcept;

    void create(VkDevice device, std::span<const VkDescriptorSetLayoutBinding> bindings);
    void reset();

    [[nodiscard]] VkDescriptorSetLayout handle() const { return layout_; }

private:
    void moveFrom(VulkanDescriptorSetLayout& other) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
};

class VulkanDescriptorPool final {
public:
    VulkanDescriptorPool() = default;
    ~VulkanDescriptorPool();

    VulkanDescriptorPool(const VulkanDescriptorPool&) = delete;
    VulkanDescriptorPool& operator=(const VulkanDescriptorPool&) = delete;
    VulkanDescriptorPool(VulkanDescriptorPool&& other) noexcept;
    VulkanDescriptorPool& operator=(VulkanDescriptorPool&& other) noexcept;

    void create(VkDevice device, std::span<const VkDescriptorPoolSize> poolSizes, uint32_t maxSets);
    void reset();

    [[nodiscard]] VkDescriptorPool handle() const { return pool_; }

private:
    void moveFrom(VulkanDescriptorPool& other) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
