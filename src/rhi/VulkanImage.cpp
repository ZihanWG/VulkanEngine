#include "rhi/VulkanImage.h"

#include "rhi/VulkanTransientMemoryPool.h"

#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <stdexcept>
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

namespace {

// One description -> one VkImageCreateInfo, shared by the private-allocation
// path, the pool-bound path, and the requirement query. If these ever described
// the same image differently, the plan would reserve the wrong number of bytes.
VkImageCreateInfo imageCreateInfoFrom(const VulkanImageCreateInfo& createInfo)
{
    const uint32_t depth = std::max(createInfo.depth, 1u);
    const bool volumeImage = depth > 1;
    if (volumeImage && createInfo.arrayLayers != 1) {
        throw std::runtime_error("A 3D image cannot have array layers; set arrayLayers to 1.");
    }

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = volumeImage ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageInfo.format = createInfo.format;
    imageInfo.extent = {createInfo.width, createInfo.height, depth};
    imageInfo.mipLevels = std::max(createInfo.mipLevels, 1u);
    imageInfo.arrayLayers = createInfo.arrayLayers;
    imageInfo.samples = createInfo.samples;
    imageInfo.tiling = createInfo.tiling;
    imageInfo.usage = createInfo.usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return imageInfo;
}

} // namespace

bool VulkanImage::queryMemoryRequirements(VulkanContext& context,
                                          const VulkanImageCreateInfo& createInfo,
                                          VkMemoryRequirements& requirements)
{
    const VkImageCreateInfo imageInfo = imageCreateInfoFrom(createInfo);

    // A throwaway unbound image: there is no way to ask the driver what an image
    // needs without one, and an unbound image costs no memory.
    VkImage probe = VK_NULL_HANDLE;
    if (vkCreateImage(context.vkDevice(), &imageInfo, nullptr, &probe) != VK_SUCCESS) {
        return false;
    }
    vkGetImageMemoryRequirements(context.vkDevice(), probe, &requirements);
    vkDestroyImage(context.vkDevice(), probe, nullptr);
    return true;
}

void VulkanImage::createImageViewForCreateInfo(const VulkanImageCreateInfo& createInfo)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = createInfo.viewType;
    viewInfo.format = createInfo.format;
    viewInfo.subresourceRange.aspectMask = createInfo.aspectMask;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels_;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = createInfo.arrayLayers;

    VK_CHECK(vkCreateImageView(context_->vkDevice(), &viewInfo, nullptr, &imageView_));

    if (!createInfo.debugName.empty()) {
        debug::setObjectName(context_->vkDevice(), image_, VK_OBJECT_TYPE_IMAGE, createInfo.debugName);
        debug::setObjectName(
            context_->vkDevice(), imageView_, VK_OBJECT_TYPE_IMAGE_VIEW, createInfo.debugName + "View");
    }
}

bool VulkanImage::createAliased(VulkanContext& context,
                                const VulkanImageCreateInfo& createInfo,
                                VulkanTransientMemoryPool& pool,
                                VkDeviceSize offset)
{
    reset();

    if (!pool.available()) {
        return false;
    }

    context_ = &context;
    format_ = createInfo.format;
    extent_ = {createInfo.width, createInfo.height, std::max(createInfo.depth, 1u)};
    mipLevels_ = std::max(createInfo.mipLevels, 1u);

    const VkImageCreateInfo imageInfo = imageCreateInfoFrom(createInfo);
    if (vmaCreateAliasingImage2(context.allocator(), pool.allocation(), offset, &imageInfo, &image_) != VK_SUCCESS) {
        // Leave the object empty so the caller can fall back to create().
        image_ = VK_NULL_HANDLE;
        context_ = nullptr;
        format_ = VK_FORMAT_UNDEFINED;
        extent_ = {};
        mipLevels_ = 0;
        return false;
    }

    // allocation_ stays null on purpose: the pool owns this memory, and reset()
    // keys off exactly that to avoid freeing what it did not allocate.
    aliased_ = true;
    createImageViewForCreateInfo(createInfo);
    return true;
}

void VulkanImage::create(VulkanContext& context, const VulkanImageCreateInfo& createInfo)
{
    reset();

    context_ = &context;
    format_ = createInfo.format;
    const uint32_t depth = std::max(createInfo.depth, 1u);
    extent_ = {createInfo.width, createInfo.height, depth};
    mipLevels_ = std::max(createInfo.mipLevels, 1u);

    const VkImageCreateInfo imageInfo = imageCreateInfoFrom(createInfo);

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    aliased_ = false;
    createImageViewForCreateInfo(createInfo);
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
        if (aliased_) {
            // Pool-bound: destroy the image, leave the memory to its owner.
            // vmaDestroyImage here would free an allocation other images are
            // still bound into.
            vkDestroyImage(context_->vkDevice(), image_, nullptr);
        } else {
            vmaDestroyImage(context_->allocator(), image_, allocation_);
        }
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }

    aliased_ = false;
    context_ = nullptr;
    format_ = VK_FORMAT_UNDEFINED;
    extent_ = {};
    mipLevels_ = 0;
}

void VulkanImage::moveFrom(VulkanImage& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    // Must move with the handles: bloom images live in a std::vector that moves
    // its elements on resize, and a moved image that forgot it was pool-bound
    // would try to free memory it does not own.
    aliased_ = std::exchange(other.aliased_, false);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
    extent_ = std::exchange(other.extent_, VkExtent3D{});
    mipLevels_ = std::exchange(other.mipLevels_, 0);
}

} // namespace ve::rhi
