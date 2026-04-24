#include "rhi/VulkanDescriptor.h"

#include <utility>

namespace ve::rhi {

VulkanDescriptorSetLayout::~VulkanDescriptorSetLayout()
{
    reset();
}

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(VulkanDescriptorSetLayout&& other) noexcept
{
    moveFrom(other);
}

VulkanDescriptorSetLayout& VulkanDescriptorSetLayout::operator=(VulkanDescriptorSetLayout&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanDescriptorSetLayout::reset()
{
    if (layout_) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
}

void VulkanDescriptorSetLayout::moveFrom(VulkanDescriptorSetLayout& other) noexcept
{
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
}

} // namespace ve::rhi
