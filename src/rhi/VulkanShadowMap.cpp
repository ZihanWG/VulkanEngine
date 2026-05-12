#include "rhi/VulkanShadowMap.h"

#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace ve::rhi {

namespace {

VkFormat chooseShadowMapFormat(VkPhysicalDevice physicalDevice)
{
    constexpr std::array<VkFormat, 2> candidates = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM};

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    for (VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }

    throw std::runtime_error("No supported sampled depth format was found for the shadow map.");
}

} // namespace

VulkanShadowMap::~VulkanShadowMap()
{
    reset();
}

VulkanShadowMap::VulkanShadowMap(VulkanShadowMap&& other) noexcept
{
    moveFrom(other);
}

VulkanShadowMap& VulkanShadowMap::operator=(VulkanShadowMap&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanShadowMap::create(VulkanContext& context, uint32_t width, uint32_t height, uint32_t layerCount)
{
    reset();

    if (width == 0 || height == 0 || layerCount == 0) {
        throw std::runtime_error("Cannot create a zero-sized shadow map or shadow-map array.");
    }

    context_ = &context;
    device_ = context.vkDevice();
    extent_ = {width, height};
    layerCount_ = layerCount;

    VulkanImageCreateInfo imageInfo{};
    imageInfo.width = width;
    imageInfo.height = height;
    imageInfo.arrayLayers = layerCount;
    imageInfo.format = chooseShadowMapFormat(context.physicalDevice());
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    imageInfo.viewType = layerCount > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    imageInfo.debugName = layerCount > 1 ? "CascadedShadowMapArray" : "DirectionalShadowMap";
    image_.create(context, imageInfo);

    createLayerImageViews();
    createSampler();
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanShadowMap::reset()
{
    for (VkImageView layerView : layerImageViews_) {
        if (layerView != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, layerView, nullptr);
        }
    }
    layerImageViews_.clear();

    if (sampler_) {
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    image_.reset();
    context_ = nullptr;
    device_ = VK_NULL_HANDLE;
    extent_ = {};
    layerCount_ = 1;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

VkImageView VulkanShadowMap::layerImageView(uint32_t layer) const
{
    if (layer >= layerImageViews_.size()) {
        throw std::out_of_range("Shadow cascade layer image view index is out of range.");
    }

    return layerImageViews_[layer];
}

void VulkanShadowMap::createLayerImageViews()
{
    layerImageViews_.resize(layerCount_, VK_NULL_HANDLE);

    for (uint32_t layer = 0; layer < layerCount_; ++layer) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image_.image();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = image_.format();
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = layer;
        viewInfo.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &layerImageViews_[layer]));
        debug::setObjectName(device_,
                             layerImageViews_[layer],
                             VK_OBJECT_TYPE_IMAGE_VIEW,
                             "CascadedShadowMapLayer" + std::to_string(layer) + "View");
    }
}

void VulkanShadowMap::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(device_, &samplerInfo, nullptr, &sampler_));
    debug::setObjectName(device_, sampler_, VK_OBJECT_TYPE_SAMPLER, "CascadedShadowMapSampler");
}

void VulkanShadowMap::moveFrom(VulkanShadowMap& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    image_ = std::move(other.image_);
    layerImageViews_ = std::move(other.layerImageViews_);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    extent_ = std::exchange(other.extent_, VkExtent2D{});
    layerCount_ = std::exchange(other.layerCount_, 1);
    layout_ = std::exchange(other.layout_, VK_IMAGE_LAYOUT_UNDEFINED);
}

} // namespace ve::rhi
