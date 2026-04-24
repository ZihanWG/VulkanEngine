#include "rhi/VulkanBuffer.h"

#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <cstring>
#include <stdexcept>
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
    createBuffer(context, createInfo);
}

void VulkanBuffer::createBuffer(VulkanContext& context, const VulkanBufferCreateInfo& createInfo)
{
    reset();

    if (createInfo.size == 0) {
        throw std::runtime_error("Cannot create a zero-sized VulkanBuffer.");
    }

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

void VulkanBuffer::createDeviceLocal(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    std::span<const std::byte> data,
    VkBufferUsageFlags usage,
    bool requestDeviceAddress)
{
    if (data.empty()) {
        throw std::runtime_error("Cannot create a GPU buffer from empty data.");
    }

    VulkanBuffer stagingBuffer;
    VulkanBufferCreateInfo stagingInfo{};
    stagingInfo.size = static_cast<VkDeviceSize>(data.size_bytes());
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    stagingInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    stagingBuffer.createBuffer(context, stagingInfo);
    stagingBuffer.upload(data);

    VulkanBufferCreateInfo deviceInfo{};
    deviceInfo.size = static_cast<VkDeviceSize>(data.size_bytes());
    deviceInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    deviceInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    deviceInfo.requestDeviceAddress = requestDeviceAddress;
    createBuffer(context, deviceInfo);

    VkPipelineStageFlags2 destinationStage = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    VkAccessFlags2 destinationAccess = VK_ACCESS_2_NONE;
    if ((usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0) {
        destinationAccess |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    }
    if ((usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0) {
        destinationAccess |= VK_ACCESS_2_INDEX_READ_BIT;
    }
    if (destinationAccess == 0) {
        destinationStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        destinationAccess = VK_ACCESS_2_MEMORY_READ_BIT;
    }

    copyBuffer(
        context,
        commandContext,
        stagingBuffer.buffer(),
        buffer_,
        size_,
        destinationStage,
        destinationAccess);
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
    if (!context_ || !allocation_) {
        throw std::runtime_error("Cannot map an uninitialized VulkanBuffer.");
    }

    if (mapped_) {
        return mappedData_;
    }

    VK_CHECK(vmaMapMemory(context_->allocator(), allocation_, &mappedData_));
    mapped_ = true;
    return mappedData_;
}

void VulkanBuffer::unmap()
{
    if (mapped_) {
        vmaUnmapMemory(context_->allocator(), allocation_);
        mappedData_ = nullptr;
        mapped_ = false;
    }
}

void VulkanBuffer::upload(std::span<const std::byte> data, VkDeviceSize offset)
{
    if (data.empty()) {
        return;
    }

    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(data.size_bytes());
    if (offset > size_ || byteSize > size_ - offset) {
        throw std::runtime_error("VulkanBuffer upload would exceed buffer size.");
    }

    auto* destination = static_cast<std::byte*>(map()) + static_cast<size_t>(offset);
    std::memcpy(destination, data.data(), data.size_bytes());
    vmaFlushAllocation(context_->allocator(), allocation_, offset, byteSize);
    unmap();
}

void VulkanBuffer::copyBuffer(
    const VulkanContext& context,
    const VulkanCommandContext& commandContext,
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize size,
    VkPipelineStageFlags2 destinationStage,
    VkAccessFlags2 destinationAccess)
{
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext.commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(context.vkDevice(), &allocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, source, destination, 1, &copyRegion);

    // The upload copy writes the GPU-local buffer. This barrier makes those
    // transfer writes visible to the first vertex/index fetch that consumes it.
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = destinationStage;
    barrier.dstAccessMask = destinationAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = destination;
    barrier.offset = 0;
    barrier.size = size;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;

    VK_CHECK(vkQueueSubmit2(context.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(context.graphicsQueue()));
    vkFreeCommandBuffers(context.vkDevice(), commandContext.commandPool(), 1, &commandBuffer);
}

void VulkanBuffer::moveFrom(VulkanBuffer& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    size_ = std::exchange(other.size_, 0);
    deviceAddress_ = std::exchange(other.deviceAddress_, 0);
    mappedData_ = std::exchange(other.mappedData_, nullptr);
    mapped_ = std::exchange(other.mapped_, false);
}

} // namespace ve::rhi
