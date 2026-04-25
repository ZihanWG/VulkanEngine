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

void VulkanDescriptorSetLayout::create(VkDevice device, std::span<const VkDescriptorSetLayoutBinding> bindings)
{
    reset();

    device_ = device;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &layout_));
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

VulkanDescriptorPool::~VulkanDescriptorPool()
{
    reset();
}

VulkanDescriptorPool::VulkanDescriptorPool(VulkanDescriptorPool&& other) noexcept
{
    moveFrom(other);
}

VulkanDescriptorPool& VulkanDescriptorPool::operator=(VulkanDescriptorPool&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanDescriptorPool::create(VkDevice device, std::span<const VkDescriptorPoolSize> poolSizes, uint32_t maxSets)
{
    reset();

    device_ = device;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pool_));
}

void VulkanDescriptorPool::reset()
{
    if (pool_) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
}

void VulkanDescriptorPool::moveFrom(VulkanDescriptorPool& other) noexcept
{
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
}

} // namespace ve::rhi
