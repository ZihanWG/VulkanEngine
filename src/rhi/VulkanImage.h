#pragma once

#include "rhi/VulkanMemory.h"

#include <string>

namespace ve::rhi {

class VulkanContext;

struct VulkanImageCreateInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    // Depth > 1 makes this a VK_IMAGE_TYPE_3D image. Vulkan forbids array layers
    // on 3D images, so arrayLayers must stay 1 when depth > 1.
    uint32_t depth = 1;
    uint32_t arrayLayers = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    uint32_t mipLevels = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    std::string debugName;
};

class VulkanImage final {
public:
    VulkanImage() = default;
    ~VulkanImage();

    VulkanImage(const VulkanImage&) = delete;
    VulkanImage& operator=(const VulkanImage&) = delete;
    VulkanImage(VulkanImage&& other) noexcept;
    VulkanImage& operator=(VulkanImage&& other) noexcept;

    void create(VulkanContext& context, const VulkanImageCreateInfo& createInfo);
    void reset();

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkExtent3D extent() const { return extent_; }
    [[nodiscard]] uint32_t mipLevels() const { return mipLevels_; }

private:
    void moveFrom(VulkanImage& other) noexcept;

    VulkanContext* context_ = nullptr;

    // VkImage and its VMA allocation are destroyed together; the view is destroyed first.
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent3D extent_{};
    uint32_t mipLevels_ = 0;
};

} // namespace ve::rhi
