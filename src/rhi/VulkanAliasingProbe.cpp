#include "rhi/VulkanAliasingProbe.h"

#include "core/Logger.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <algorithm>
#include <array>
#include <span>
#include <string>

namespace ve::rhi {

namespace {

// Two different formats and extents on purpose. Aliasing resources that happen to
// be identical proves much less than aliasing ones that are not, and the real
// transient pool mixes RGBA16F scene targets with R8 occlusion targets.
VkImageCreateInfo probeImageCreateInfo(VkFormat format, uint32_t width, uint32_t height)
{
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return info;
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}


// Clears one alias, reads the other back, and reports whether the second saw the
// first's bytes. See AliasingProbeResult::aliasedWriteObserved for why a false
// result is inconclusive rather than a failure.
bool observeAliasedWrite(VulkanContext& context,
                         const VulkanCommandContext& commandContext,
                         VmaAllocation allocation,
                         const VkImageCreateInfo& baseInfo,
                         std::string& detail)
{
    // Identical descriptions at the same offset: the strongest form of the
    // question, and the only one where a driver could reasonably preserve the
    // bytes across the alias.
    VkImageCreateInfo info = baseInfo;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    const VkDevice device = context.vkDevice();
    VkImage writer = VK_NULL_HANDLE;
    VkImage reader = VK_NULL_HANDLE;
    if (vmaCreateAliasingImage2(context.allocator(), allocation, 0, &info, &writer) != VK_SUCCESS ||
        vmaCreateAliasingImage2(context.allocator(), allocation, 0, &info, &reader) != VK_SUCCESS) {
        detail += "; readback skipped (transfer-usage aliases refused)";
        if (writer != VK_NULL_HANDLE) {
            vkDestroyImage(device, writer, nullptr);
        }
        return false;
    }

    VulkanBuffer readback;
    VulkanBufferCreateInfo bufferInfo{};
    bufferInfo.size = 4u * sizeof(uint16_t) * 4u; // a few RGBA16F texels is plenty
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    readback.createBuffer(context, bufferInfo);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext.commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    const auto transition = [&](VkImage image,
                                VkImageLayout oldLayout,
                                VkImageLayout newLayout,
                                VkPipelineStageFlags2 srcStage,
                                VkAccessFlags2 srcAccess,
                                VkPipelineStageFlags2 dstStage,
                                VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    };

    transition(writer,
               VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_PIPELINE_STAGE_2_NONE,
               VK_ACCESS_2_NONE,
               VK_PIPELINE_STAGE_2_CLEAR_BIT,
               VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkClearColorValue clearColor{};
    clearColor.float32[0] = 0.5f;
    clearColor.float32[1] = 0.25f;
    clearColor.float32[2] = 0.75f;
    clearColor.float32[3] = 1.0f;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(commandBuffer, writer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);

    // The alias handoff: the write through `writer` must be visible before
    // `reader` touches the same bytes. This is the barrier shape the real
    // allocator will emit at every handoff.
    transition(reader,
               VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_PIPELINE_STAGE_2_CLEAR_BIT,
               VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COPY_BIT,
               VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {2, 2, 1};
    vkCmdCopyImageToBuffer(
        commandBuffer, reader, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer(), 1, &region);

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

    std::array<uint16_t, 4> texel{};
    readback.download(std::as_writable_bytes(std::span<uint16_t>(texel.data(), texel.size())));

    vkFreeCommandBuffers(device, commandContext.commandPool(), 1, &commandBuffer);
    vkDestroyImage(device, reader, nullptr);
    vkDestroyImage(device, writer, nullptr);

    // Half-float 0.5 is 0x3800; anything non-zero means the reader saw the
    // writer's clear rather than untouched memory.
    const bool observed = texel[0] != 0 || texel[1] != 0 || texel[2] != 0;
    detail += observed ? "; a write through one alias was read back through the other"
                       : "; readback did not observe the write (inconclusive, not a failure)";
    return observed;
}

} // namespace

AliasingProbeResult probeImageMemoryAliasing(VulkanContext& context, const VulkanCommandContext& commandContext)
{
    AliasingProbeResult result{};

    const VkDevice device = context.vkDevice();
    const VmaAllocator allocator = context.allocator();
    if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE) {
        result.detail = "no device or allocator";
        return result;
    }

    const VkImageCreateInfo firstInfo = probeImageCreateInfo(VK_FORMAT_R16G16B16A16_SFLOAT, 512, 512);
    const VkImageCreateInfo secondInfo = probeImageCreateInfo(VK_FORMAT_R8_UNORM, 256, 256);

    // Memory requirements come from throwaway images: there is no way to ask for
    // them without an image, and creating one costs nothing without a binding.
    VkImage firstProbe = VK_NULL_HANDLE;
    VkImage secondProbe = VK_NULL_HANDLE;
    if (vkCreateImage(device, &firstInfo, nullptr, &firstProbe) != VK_SUCCESS ||
        vkCreateImage(device, &secondInfo, nullptr, &secondProbe) != VK_SUCCESS) {
        result.detail = "vkCreateImage failed for the requirement query";
        if (firstProbe != VK_NULL_HANDLE) {
            vkDestroyImage(device, firstProbe, nullptr);
        }
        return result;
    }

    VkMemoryRequirements firstRequirements{};
    VkMemoryRequirements secondRequirements{};
    vkGetImageMemoryRequirements(device, firstProbe, &firstRequirements);
    vkGetImageMemoryRequirements(device, secondProbe, &secondRequirements);
    vkDestroyImage(device, firstProbe, nullptr);
    vkDestroyImage(device, secondProbe, nullptr);

    result.firstImageBytes = firstRequirements.size;
    result.secondImageBytes = secondRequirements.size;
    result.commonMemoryTypeBits = firstRequirements.memoryTypeBits & secondRequirements.memoryTypeBits;

    if (result.commonMemoryTypeBits == 0) {
        // Not a bug: it is precisely the case the real pool must fall back from.
        result.detail = "no memory type accepts both images";
        return result;
    }

    const VkDeviceSize alignment = std::max(firstRequirements.alignment, secondRequirements.alignment);
    const VkDeviceSize secondOffset = alignUp(firstRequirements.size, secondRequirements.alignment);
    const VkDeviceSize poolSize = secondOffset + secondRequirements.size;

    VkMemoryRequirements poolRequirements{};
    poolRequirements.size = poolSize;
    poolRequirements.alignment = alignment;
    poolRequirements.memoryTypeBits = result.commonMemoryTypeBits;

    VmaAllocationCreateInfo allocationInfo{};
    // Not VMA_MEMORY_USAGE_AUTO: it asserts when the allocation is not made
    // through vmaCreateBuffer/vmaCreateImage, because it infers the memory type
    // from the resource being created (vk_mem_alloc.h:4053). A raw pool
    // allocation has no such resource, so the requirement is stated directly.
    allocationInfo.usage = VMA_MEMORY_USAGE_UNKNOWN;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    // Without CAN_ALIAS, VMA is entitled to assume one resource per allocation.
    allocationInfo.flags = VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;

    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationDetail{};
    if (vmaAllocateMemory(allocator, &poolRequirements, &allocationInfo, &allocation, &allocationDetail) !=
        VK_SUCCESS) {
        result.detail = "vmaAllocateMemory rejected the shared block";
        return result;
    }

    result.sharedAllocationBytes = allocationDetail.size;
    result.secondImageOffset = secondOffset;

    VkImage firstAliased = VK_NULL_HANDLE;
    VkImage secondAliased = VK_NULL_HANDLE;
    const VkResult firstBind = vmaCreateAliasingImage2(allocator, allocation, 0, &firstInfo, &firstAliased);
    const VkResult secondBind =
        vmaCreateAliasingImage2(allocator, allocation, secondOffset, &secondInfo, &secondAliased);

    result.supported = firstBind == VK_SUCCESS && secondBind == VK_SUCCESS;
    if (!result.supported) {
        result.detail = "vmaCreateAliasingImage2 failed (first=" + std::to_string(static_cast<int>(firstBind)) +
                        ", second=" + std::to_string(static_cast<int>(secondBind)) + ")";
    } else {
        result.detail = "two images bound into one allocation";
        result.aliasedWriteObserved =
            observeAliasedWrite(context, commandContext, allocation, firstInfo, result.detail);
    }

    if (secondAliased != VK_NULL_HANDLE) {
        vkDestroyImage(device, secondAliased, nullptr);
    }
    if (firstAliased != VK_NULL_HANDLE) {
        vkDestroyImage(device, firstAliased, nullptr);
    }
    vmaFreeMemory(allocator, allocation);

    return result;
}

} // namespace ve::rhi
