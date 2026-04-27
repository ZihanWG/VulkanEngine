#include "rhi/VulkanShadowMap.h"

#include "rhi/VulkanContext.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace ve::rhi {

namespace {

VkFormat chooseShadowMapFormat(VkPhysicalDevice physicalDevice)
{
    constexpr std::array<VkFormat, 2> candidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D16_UNORM
    };

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

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

void VulkanShadowMap::create(VulkanContext& context, uint32_t width, uint32_t height)
{
    reset();

    if (width == 0 || height == 0) {
        throw std::runtime_error("Cannot create a zero-sized shadow map.");
    }

    context_ = &context;
    device_ = context.vkDevice();
    extent_ = {width, height};

    VulkanImageCreateInfo imageInfo{};
    imageInfo.width = width;
    imageInfo.height = height;
    imageInfo.format = chooseShadowMapFormat(context.physicalDevice());
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    imageInfo.debugName = "DirectionalShadowMap";
    image_.create(context, imageInfo);

    createSampler();
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanShadowMap::reset()
{
    if (sampler_) {
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    image_.reset();
    context_ = nullptr;
    device_ = VK_NULL_HANDLE;
    extent_ = {};
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
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
}

void VulkanShadowMap::moveFrom(VulkanShadowMap& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    image_ = std::move(other.image_);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    extent_ = std::exchange(other.extent_, VkExtent2D{});
    layout_ = std::exchange(other.layout_, VK_IMAGE_LAYOUT_UNDEFINED);
}

} // namespace ve::rhi
