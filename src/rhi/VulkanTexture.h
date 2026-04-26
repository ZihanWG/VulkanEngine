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

    void createCheckerboard(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t width = 256,
        uint32_t height = 256);
    void createFromRgba8(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t width,
        uint32_t height,
        std::span<const uint8_t> pixels,
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);
    void reset();
    void destroy() { reset(); }

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] bool valid() const { return image_ != VK_NULL_HANDLE; }

private:
    void uploadPixels(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const std::byte> pixels);
    void createImageView();
    void createSampler();
    void moveFrom(VulkanTexture& other) noexcept;

    VulkanContext* context_ = nullptr;

    // The image and its VMA allocation own GPU-local texture storage.
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;

    // The view and sampler are what the combined image sampler descriptor exposes.
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};

} // namespace ve::rhi
