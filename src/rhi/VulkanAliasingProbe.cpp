#include "rhi/VulkanAliasingProbe.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"

#include <algorithm>
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

} // namespace

AliasingProbeResult probeImageMemoryAliasing(VulkanContext& context)
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
