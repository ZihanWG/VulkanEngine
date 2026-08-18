#pragma once

#include "rhi/VulkanMemory.h"

#include <cstdint>
#include <vector>

namespace ve {
class JobSystem;
}

namespace ve::rhi {

class VulkanCommandContext;
class VulkanContext;

class VulkanBrdfLut final {
public:
    VulkanBrdfLut() = default;
    ~VulkanBrdfLut();

    VulkanBrdfLut(const VulkanBrdfLut&) = delete;
    VulkanBrdfLut& operator=(const VulkanBrdfLut&) = delete;
    VulkanBrdfLut(VulkanBrdfLut&& other) noexcept;
    VulkanBrdfLut& operator=(VulkanBrdfLut&& other) noexcept;

    // The split-sum table is 8.4 million integration steps at the default size
    // and was ~97 ms on one thread -- 59% of scene creation, paid by every scene.
    // `jobSystem` spreads its rows; nullptr keeps it serial. The bytes are the
    // same either way, since every texel is an independent pure function of its
    // coordinates (renderer/BrdfIntegration.h).
    //
    // Must not be called from a JobSystem worker.
    void create(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t size = 256,
        JobSystem* jobSystem = nullptr);
    void reset();
    void destroy() { reset(); }

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkImageLayout layout() const { return layout_; }
    [[nodiscard]] bool valid() const { return image_ != VK_NULL_HANDLE; }

private:
    void uploadPixels(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        const std::vector<uint8_t>& pixels);
    void createImageView();
    void createSampler();
    void moveFrom(VulkanBrdfLut& other) noexcept;

    VulkanContext* context_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkFormat format_ = VK_FORMAT_R8G8_UNORM;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace ve::rhi
