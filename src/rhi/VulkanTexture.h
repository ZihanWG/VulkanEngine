#pragma once

#include "rhi/VulkanMemory.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ve::rhi {

class VulkanCommandContext;
class VulkanContext;

class VulkanTexture final {
public:
    VulkanTexture() = default;
    ~VulkanTexture();

    VulkanTexture(const VulkanTexture&) = delete;
    VulkanTexture& operator=(const VulkanTexture&) = delete;
    VulkanTexture(VulkanTexture&& other) noexcept;
    VulkanTexture& operator=(VulkanTexture&& other) noexcept;

    void createFromRgba8(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t width,
        uint32_t height,
        std::span<const std::byte> pixels);
    void reset();

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkDescriptorImageInfo descriptorInfo() const;

private:
    void uploadPixels(
        const VulkanContext& context,
        const VulkanCommandContext& commandContext,
        VkBuffer stagingBuffer);
    void createSampler();
    void moveFrom(VulkanTexture& other) noexcept;

    VulkanContext* context_ = nullptr;

    // The image, allocation, view, and sampler together represent one sampled 2D texture.
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};

} // namespace ve::rhi
