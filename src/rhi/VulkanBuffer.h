#pragma once

#include "rhi/VulkanMemory.h"

#include <cstddef>
#include <span>

namespace ve::rhi {

class VulkanCommandContext;
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
    void createBuffer(VulkanContext& context, const VulkanBufferCreateInfo& createInfo);
    void createDeviceLocal(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const std::byte> data,
        VkBufferUsageFlags usage,
        bool requestDeviceAddress = false);
    void reset();
    void destroy() { reset(); }

    [[nodiscard]] VkBuffer buffer() const { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] VkDeviceAddress deviceAddress() const { return deviceAddress_; }
    [[nodiscard]] bool valid() const { return buffer_ != VK_NULL_HANDLE; }

    void* map();
    void unmap();
    void upload(std::span<const std::byte> data, VkDeviceSize offset = 0);

    static void copyBuffer(
        const VulkanContext& context,
        const VulkanCommandContext& commandContext,
        VkBuffer source,
        VkBuffer destination,
        VkDeviceSize size,
        VkPipelineStageFlags2 destinationStage = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
        VkAccessFlags2 destinationAccess = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT);

private:
    void moveFrom(VulkanBuffer& other) noexcept;

    VulkanContext* context_ = nullptr;

    // VkBuffer is the GPU-visible handle; VMA owns the backing memory allocation.
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;

    // Non-zero only for buffers created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
    VkDeviceAddress deviceAddress_ = 0;
    void* mappedData_ = nullptr;
    bool mapped_ = false;
};

} // namespace ve::rhi
