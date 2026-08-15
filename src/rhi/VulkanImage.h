#pragma once

#include "rhi/VulkanMemory.h"

#include <cstdint>
#include <string>

namespace ve::rhi {

class VulkanContext;
class VulkanTransientMemoryPool;

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

    // Creates the image bound into `pool` at `offset` instead of giving it a
    // private allocation. The pool owns the memory: reset() destroys this image
    // and its view but must never free what it did not allocate.
    //
    // The caller is responsible for the offset being one no lifetime-overlapping
    // image also occupies (renderer::planTransientMemory guarantees that), and
    // for destroying every aliased image before the pool it came from.
    //
    // Returns false when the pool is unavailable or the driver rejects the
    // binding, leaving the image empty so the caller can fall back to create().
    [[nodiscard]] bool createAliased(VulkanContext& context,
                                     const VulkanImageCreateInfo& createInfo,
                                     VulkanTransientMemoryPool& pool,
                                     VkDeviceSize offset);

    // Memory the driver wants for this description, without allocating any. Used
    // to build the plan before a single image exists.
    [[nodiscard]] static bool queryMemoryRequirements(VulkanContext& context,
                                                      const VulkanImageCreateInfo& createInfo,
                                                      VkMemoryRequirements& requirements);

    void reset();

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkExtent3D extent() const { return extent_; }
    [[nodiscard]] uint32_t mipLevels() const { return mipLevels_; }

private:
    void createImageViewForCreateInfo(const VulkanImageCreateInfo& createInfo);
    void moveFrom(VulkanImage& other) noexcept;

    VulkanContext* context_ = nullptr;

    // VkImage and its VMA allocation are destroyed together; the view is destroyed first.
    VkImage image_ = VK_NULL_HANDLE;
    // Null when the image is bound into a transient pool: the memory belongs to
    // the pool and this object must not free it.
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    bool aliased_ = false;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent3D extent_{};
    uint32_t mipLevels_ = 0;
};

} // namespace ve::rhi
