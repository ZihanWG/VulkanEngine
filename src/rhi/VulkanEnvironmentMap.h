#pragma once

#include "rhi/VulkanMemory.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ve::rhi {

class VulkanCommandContext;
class VulkanContext;

class VulkanEnvironmentMap final {
public:
    VulkanEnvironmentMap() = default;
    ~VulkanEnvironmentMap();

    VulkanEnvironmentMap(const VulkanEnvironmentMap&) = delete;
    VulkanEnvironmentMap& operator=(const VulkanEnvironmentMap&) = delete;
    VulkanEnvironmentMap(VulkanEnvironmentMap&& other) noexcept;
    VulkanEnvironmentMap& operator=(VulkanEnvironmentMap&& other) noexcept;

    void createProcedural(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t faceSize = 32);
    void createProceduralDiffuseIrradiance(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t faceSize = 32);
    void createFromRgba8Faces(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t faceSize,
        std::span<const uint8_t> pixels,
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);
    void reset();
    void destroy() { reset(); }

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] uint32_t faceSize() const { return faceSize_; }
    [[nodiscard]] uint32_t mipLevels() const { return mipLevels_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkImageLayout layout() const { return layout_; }
    [[nodiscard]] bool valid() const { return image_ != VK_NULL_HANDLE; }

private:
    void uploadFaces(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const std::byte> pixels);
    void createImageView();
    void createSampler();
    void moveFrom(VulkanEnvironmentMap& other) noexcept;

    VulkanContext* context_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    uint32_t faceSize_ = 0;
    uint32_t mipLevels_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace ve::rhi
