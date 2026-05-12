#pragma once

#include "rhi/VulkanImage.h"

#include <cstdint>
#include <vector>

namespace ve::rhi {

class VulkanContext;

class VulkanShadowMap final {
public:
    VulkanShadowMap() = default;
    ~VulkanShadowMap();

    VulkanShadowMap(const VulkanShadowMap&) = delete;
    VulkanShadowMap& operator=(const VulkanShadowMap&) = delete;
    VulkanShadowMap(VulkanShadowMap&& other) noexcept;
    VulkanShadowMap& operator=(VulkanShadowMap&& other) noexcept;

    void create(VulkanContext& context, uint32_t width = 2048, uint32_t height = 2048, uint32_t layerCount = 1);
    void reset();

    [[nodiscard]] VkImage image() const { return image_.image(); }
    [[nodiscard]] VkImageView imageView() const { return image_.imageView(); }
    [[nodiscard]] VkImageView layerImageView(uint32_t layer) const;
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] VkFormat format() const { return image_.format(); }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] uint32_t layerCount() const { return layerCount_; }
    [[nodiscard]] VkImageLayout layout() const { return layout_; }
    [[nodiscard]] bool valid() const { return image_.image() != VK_NULL_HANDLE; }

    void setLayout(VkImageLayout layout) { layout_ = layout; }

private:
    void createLayerImageViews();
    void createSampler();
    void moveFrom(VulkanShadowMap& other) noexcept;

    VulkanContext* context_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VulkanImage image_;
    std::vector<VkImageView> layerImageViews_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    uint32_t layerCount_ = 1;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace ve::rhi
