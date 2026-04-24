#pragma once

#include "rhi/VulkanMemory.h"

#include <cstddef>

namespace ve::rhi {

class VulkanContext;

struct VulkanBufferCreateInfo {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags allocationFlags = 0;
    bool requestDeviceAddress = false;
};

class VulkanBuffer final {
public:
    VulkanBuffer() = default;
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

    void create(VulkanContext& context, const VulkanBufferCreateInfo& createInfo);
    void reset();

    [[nodiscard]] VkBuffer buffer() const { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] VkDeviceAddress deviceAddress() const { return deviceAddress_; }

    void* map();
    void unmap();

private:
    void moveFrom(VulkanBuffer& other) noexcept;

    VulkanContext* context_ = nullptr;

    // VkBuffer is the GPU-visible handle; VMA owns the backing memory allocation.
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;

    // Non-zero only for buffers created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
    VkDeviceAddress deviceAddress_ = 0;
    bool mapped_ = false;
};

} // namespace ve::rhi
