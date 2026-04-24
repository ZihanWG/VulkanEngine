#include "rhi/VulkanBuffer.h"

#include "rhi/VulkanContext.h"

#include <utility>

namespace ve::rhi {

VulkanBuffer::~VulkanBuffer()
{
    reset();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
{
    moveFrom(other);
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanBuffer::create(VulkanContext& context, const VulkanBufferCreateInfo& createInfo)
{
    reset();

    context_ = &context;
    size_ = createInfo.size;

    VkBufferUsageFlags usage = createInfo.usage;
    if (createInfo.requestDeviceAddress) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = createInfo.size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = createInfo.memoryUsage;
    allocationInfo.flags = createInfo.allocationFlags;

    VK_CHECK(vmaCreateBuffer(context.allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, nullptr));

    if (createInfo.requestDeviceAddress) {
        VkBufferDeviceAddressInfo addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addressInfo.buffer = buffer_;
        deviceAddress_ = vkGetBufferDeviceAddress(context.vkDevice(), &addressInfo);
    }
}

void VulkanBuffer::reset()
{
    if (!context_) {
        return;
    }

    if (mapped_) {
        unmap();
    }

    if (buffer_) {
        vmaDestroyBuffer(context_->allocator(), buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }

    context_ = nullptr;
    size_ = 0;
    deviceAddress_ = 0;
}

void* VulkanBuffer::map()
{
    void* data = nullptr;
    VK_CHECK(vmaMapMemory(context_->allocator(), allocation_, &data));
    mapped_ = true;
    return data;
}

void VulkanBuffer::unmap()
{
    if (mapped_) {
        vmaUnmapMemory(context_->allocator(), allocation_);
        mapped_ = false;
    }
}

void VulkanBuffer::moveFrom(VulkanBuffer& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    deviceAddress_ = std::exchange(other.deviceAddress_, 0);
    mapped_ = std::exchange(other.mapped_, false);
}

} // namespace ve::rhi
