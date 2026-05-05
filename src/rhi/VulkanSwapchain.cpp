#include "rhi/VulkanSwapchain.h"

#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ve::rhi {

VulkanSwapchain::~VulkanSwapchain()
{
    cleanup();
}

void VulkanSwapchain::initialize(VulkanContext& context, WindowExtent desiredExtent)
{
    context_ = &context;
    create(desiredExtent);
}

void VulkanSwapchain::recreate(VulkanContext& context, WindowExtent desiredExtent)
{
    context_ = &context;
    cleanup();
    create(desiredExtent);
}

void VulkanSwapchain::cleanup()
{
    if (!context_) {
        return;
    }

    depthImage_.reset();

    for (VkImageView imageView : imageViews_) {
        vkDestroyImageView(context_->vkDevice(), imageView, nullptr);
    }
    imageViews_.clear();
    images_.clear();
    imageLayouts_.clear();

    if (swapchain_) {
        vkDestroySwapchainKHR(context_->vkDevice(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    colorFormat_ = VK_FORMAT_UNDEFINED;
    depthFormat_ = VK_FORMAT_UNDEFINED;
    depthImageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    extent_ = {};
}

void VulkanSwapchain::create(WindowExtent desiredExtent)
{
    const SwapchainSupportDetails support = context_->device().querySwapchainSupport();
    const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(support.formats);
    const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
    const VkExtent2D selectedExtent = chooseExtent(support.capabilities, desiredExtent);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, support.capabilities.maxImageCount);
    }

    const QueueFamilyIndices& indices = context_->queueFamilies();
    std::array<uint32_t, 2> queueFamilyIndices = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    const bool usesSeparateQueues = indices.graphicsFamily.value() != indices.presentFamily.value();

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = context_->surface();
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = selectedExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    createInfo.imageSharingMode = usesSeparateQueues ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = usesSeparateQueues ? static_cast<uint32_t>(queueFamilyIndices.size()) : 0;
    createInfo.pQueueFamilyIndices = usesSeparateQueues ? queueFamilyIndices.data() : nullptr;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(context_->vkDevice(), &createInfo, nullptr, &swapchain_));

    VK_CHECK(vkGetSwapchainImagesKHR(context_->vkDevice(), swapchain_, &imageCount, nullptr));
    images_.resize(imageCount);
    VK_CHECK(vkGetSwapchainImagesKHR(context_->vkDevice(), swapchain_, &imageCount, images_.data()));

    colorFormat_ = surfaceFormat.format;
    extent_ = selectedExtent;
    imageLayouts_.assign(images_.size(), VK_IMAGE_LAYOUT_UNDEFINED);

    for (size_t index = 0; index < images_.size(); ++index) {
        debug::setObjectName(
            context_->vkDevice(), images_[index], VK_OBJECT_TYPE_IMAGE, "SwapchainImage" + std::to_string(index));
    }

    createImageViews();
    createDepthImage();
}

void VulkanSwapchain::createImageViews()
{
    imageViews_.resize(images_.size());

    for (size_t index = 0; index < images_.size(); ++index) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = images_[index];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = colorFormat_;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(context_->vkDevice(), &viewInfo, nullptr, &imageViews_[index]));
        debug::setObjectName(context_->vkDevice(),
                             imageViews_[index],
                             VK_OBJECT_TYPE_IMAGE_VIEW,
                             "SwapchainImageView" + std::to_string(index));
    }
}

void VulkanSwapchain::createDepthImage()
{
    depthFormat_ = findDepthFormat();

    VulkanImageCreateInfo depthInfo{};
    depthInfo.width = extent_.width;
    depthInfo.height = extent_.height;
    depthInfo.format = depthFormat_;
    depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthInfo.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthInfo.debugName = "MainDepth";

    depthImage_.create(*context_, depthInfo);
    depthImageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

VkSurfaceFormatKHR VulkanSwapchain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const
{
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats.front();
}

VkPresentModeKHR VulkanSwapchain::choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) const
{
    for (VkPresentModeKHR mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapchain::chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, WindowExtent desiredExtent) const
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    VkExtent2D actualExtent{desiredExtent.width, desiredExtent.height};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
}

VkFormat VulkanSwapchain::findDepthFormat() const
{
    const std::array<VkFormat, 3> candidates = {
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};

    for (VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(context_->physicalDevice(), format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }

    throw std::runtime_error("No supported depth format was found.");
}

} // namespace ve::rhi
