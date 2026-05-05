#include "rhi/VulkanImage.h"

#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <string>
#include <utility>

namespace ve::rhi {

VulkanImage::~VulkanImage()
{
    reset();
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
{
    moveFrom(other);
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanImage::create(VulkanContext& context, const VulkanImageCreateInfo& createInfo)
{
    reset();

    context_ = &context;
    format_ = createInfo.format;
    extent_ = {createInfo.width, createInfo.height, 1};

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = createInfo.format;
    imageInfo.extent = extent_;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = createInfo.samples;
    imageInfo.tiling = createInfo.tiling;
    imageInfo.usage = createInfo.usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = createInfo.format;
    viewInfo.subresourceRange.aspectMask = createInfo.aspectMask;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(context.vkDevice(), &viewInfo, nullptr, &imageView_));

    if (!createInfo.debugName.empty()) {
        debug::setObjectName(context.vkDevice(), image_, VK_OBJECT_TYPE_IMAGE, createInfo.debugName);
        debug::setObjectName(context.vkDevice(), imageView_, VK_OBJECT_TYPE_IMAGE_VIEW, createInfo.debugName + "View");
    }
}

void VulkanImage::reset()
{
    if (!context_) {
        return;
    }

    if (imageView_) {
        vkDestroyImageView(context_->vkDevice(), imageView_, nullptr);
        imageView_ = VK_NULL_HANDLE;
    }

    if (image_) {
        vmaDestroyImage(context_->allocator(), image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }

    context_ = nullptr;
    format_ = VK_FORMAT_UNDEFINED;
    extent_ = {};
}

void VulkanImage::moveFrom(VulkanImage& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
    extent_ = std::exchange(other.extent_, VkExtent3D{});
}

} // namespace ve::rhi
