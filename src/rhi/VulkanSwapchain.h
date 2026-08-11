#pragma once

#include "core/Window.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanImage.h"

#include <vector>

namespace ve::rhi {

class VulkanContext;

class VulkanSwapchain final {
public:
    VulkanSwapchain() = default;
    ~VulkanSwapchain();

    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;
    VulkanSwapchain(VulkanSwapchain&&) = delete;
    VulkanSwapchain& operator=(VulkanSwapchain&&) = delete;

    void initialize(VulkanContext& context, WindowExtent desiredExtent);
    void recreate(VulkanContext& context, WindowExtent desiredExtent);
    void cleanup();

    [[nodiscard]] VkSwapchainKHR handle() const { return swapchain_; }
    [[nodiscard]] VkFormat colorFormat() const { return colorFormat_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] bool supportsTransferSrc() const { return (imageUsage_ & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0; }
    [[nodiscard]] uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }
    [[nodiscard]] VkImage image(uint32_t index) const { return images_.at(index); }
    [[nodiscard]] VkImageView imageView(uint32_t index) const { return imageViews_.at(index); }
    [[nodiscard]] VkImageLayout imageLayout(uint32_t index) const { return imageLayouts_.at(index); }
    void setImageLayout(uint32_t index, VkImageLayout layout) { imageLayouts_.at(index) = layout; }

    // The main depth image is sized independently of the presentation images so
    // the scene can be rendered at a reduced internal resolution (see
    // renderer/RenderScale.h). (Re)creating the swapchain resets it to match
    // extent(); Renderer then calls resizeDepthImage with the render extent once
    // it knows the new swapchain size. The device must be idle -- this destroys
    // an image in-flight frames may still reference.
    void resizeDepthImage(VkExtent2D extent);

    [[nodiscard]] VkFormat depthFormat() const { return depthFormat_; }
    [[nodiscard]] VkExtent2D depthExtent() const { return depthExtent_; }
    [[nodiscard]] bool depthSupportsSampling() const { return depthSupportsSampling_; }
    [[nodiscard]] VkImage depthImage() const { return depthImage_.image(); }
    [[nodiscard]] VkImageView depthImageView() const { return depthImage_.imageView(); }
    [[nodiscard]] VkImageLayout depthImageLayout() const { return depthImageLayout_; }
    void setDepthImageLayout(VkImageLayout layout) { depthImageLayout_ = layout; }

private:
    void create(WindowExtent desiredExtent);
    void createImageViews();
    void createDepthImage();

    [[nodiscard]] VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
    [[nodiscard]] VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) const;
    [[nodiscard]] VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, WindowExtent desiredExtent) const;
    [[nodiscard]] VkFormat findDepthFormat() const;

    VulkanContext* context_ = nullptr;

    // The swapchain owns the presentation images supplied by the window surface.
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    bool depthSupportsSampling_ = false;
    VkExtent2D extent_{};
    VkImageUsageFlags imageUsage_ = 0;
    std::vector<VkImage> images_;

    // Image views are the attachment handles used by Dynamic Rendering.
    std::vector<VkImageView> imageViews_;

    // The renderer tracks current image layouts so barriers use the actual previous layout.
    std::vector<VkImageLayout> imageLayouts_;

    // Created now so triangle/depth work can use the same swapchain lifetime.
    // depthExtent_ tracks its size separately from extent_ because render scale
    // lets the two differ.
    VulkanImage depthImage_;
    VkExtent2D depthExtent_{};
    VkImageLayout depthImageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace ve::rhi
