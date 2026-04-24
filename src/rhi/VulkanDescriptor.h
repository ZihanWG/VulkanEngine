#pragma once

#include "rhi/VulkanCommon.h"

namespace ve::rhi {

class VulkanDescriptorSetLayout final {
public:
    VulkanDescriptorSetLayout() = default;
    ~VulkanDescriptorSetLayout();

    VulkanDescriptorSetLayout(const VulkanDescriptorSetLayout&) = delete;
    VulkanDescriptorSetLayout& operator=(const VulkanDescriptorSetLayout&) = delete;
    VulkanDescriptorSetLayout(VulkanDescriptorSetLayout&& other) noexcept;
    VulkanDescriptorSetLayout& operator=(VulkanDescriptorSetLayout&& other) noexcept;

    void reset();

    [[nodiscard]] VkDescriptorSetLayout handle() const { return layout_; }

private:
    void moveFrom(VulkanDescriptorSetLayout& other) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
