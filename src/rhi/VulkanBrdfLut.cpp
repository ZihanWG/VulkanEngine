#include "rhi/VulkanBrdfLut.h"

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

constexpr uint32_t kLutChannels = 2;
constexpr uint32_t kIntegrationSampleCount = 128;
constexpr float kPi = 3.14159265359f;
constexpr float kEpsilon = 0.0001f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 operator+(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vec3 operator-(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 operator*(Vec3 value, float scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

float dot(Vec3 lhs, Vec3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 cross(Vec3 lhs, Vec3 rhs)
{
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

Vec3 normalize(Vec3 value)
{
    const float length = std::sqrt(std::max(dot(value, value), 0.0f));
    if (length <= 0.0f) {
        return {0.0f, 0.0f, 1.0f};
    }

    const float inverseLength = 1.0f / length;
    return value * inverseLength;
}

uint8_t floatToByte(float value)
{
    return static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
}

float radicalInverseVdc(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

Vec2 hammersley(uint32_t index, uint32_t sampleCount)
{
    return {
        static_cast<float>(index) / static_cast<float>(sampleCount),
        radicalInverseVdc(index)
    };
}

Vec3 importanceSampleGgx(Vec2 xi, Vec3 normal, float roughness)
{
    const float alpha = roughness * roughness;
    const float phi = 2.0f * kPi * xi.x;
    const float cosTheta = std::sqrt(
        (1.0f - xi.y) / std::max(1.0f + (alpha * alpha - 1.0f) * xi.y, kEpsilon));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));

    const Vec3 halfwayTangent{
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta
    };

    const Vec3 up = std::abs(normal.z) < 0.999f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(up, normal));
    const Vec3 bitangent = cross(normal, tangent);

    return normalize(
        tangent * halfwayTangent.x
        + bitangent * halfwayTangent.y
        + normal * halfwayTangent.z);
}

float geometrySchlickGgx(float normalDirection, float roughness)
{
    const float alpha = roughness * roughness;
    const float k = alpha / 2.0f;
    return normalDirection / std::max(normalDirection * (1.0f - k) + k, kEpsilon);
}

float geometrySmith(float normalView, float normalLight, float roughness)
{
    return geometrySchlickGgx(normalView, roughness) * geometrySchlickGgx(normalLight, roughness);
}

std::array<float, 2> integrateBrdf(float normalView, float roughness)
{
    const Vec3 normal{0.0f, 0.0f, 1.0f};
    const Vec3 viewDirection{
        std::sqrt(std::max(1.0f - normalView * normalView, 0.0f)),
        0.0f,
        normalView
    };

    float scale = 0.0f;
    float bias = 0.0f;

    for (uint32_t i = 0; i < kIntegrationSampleCount; ++i) {
        const Vec2 xi = hammersley(i, kIntegrationSampleCount);
        const Vec3 halfway = importanceSampleGgx(xi, normal, roughness);
        const Vec3 lightDirection = normalize(halfway * (2.0f * dot(viewDirection, halfway)) - viewDirection);

        const float normalLight = std::max(lightDirection.z, 0.0f);
        const float normalHalf = std::max(halfway.z, 0.0f);
        const float viewHalf = std::max(dot(viewDirection, halfway), 0.0f);

        if (normalLight > 0.0f) {
            const float geometry = geometrySmith(normalView, normalLight, roughness);
            const float geometryVisibility =
                geometry * viewHalf / std::max(normalHalf * normalView, kEpsilon);
            const float fresnel = std::pow(1.0f - viewHalf, 5.0f);

            scale += (1.0f - fresnel) * geometryVisibility;
            bias += fresnel * geometryVisibility;
        }
    }

    const float inverseSampleCount = 1.0f / static_cast<float>(kIntegrationSampleCount);
    return {scale * inverseSampleCount, bias * inverseSampleCount};
}

std::vector<uint8_t> makeSplitSumBrdfLut(uint32_t size)
{
    if (size == 0) {
        throw std::runtime_error("Cannot create a zero-sized BRDF LUT.");
    }

    std::vector<uint8_t> pixels(static_cast<size_t>(size) * size * kLutChannels);

    for (uint32_t y = 0; y < size; ++y) {
        const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
        for (uint32_t x = 0; x < size; ++x) {
            const float normalView = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
            const std::array<float, 2> brdf = integrateBrdf(normalView, roughness);
            const size_t offset = (static_cast<size_t>(y) * size + x) * kLutChannels;

            pixels[offset + 0] = floatToByte(brdf[0]);
            pixels[offset + 1] = floatToByte(brdf[1]);
        }
    }

    return pixels;
}

void validateBrdfLutFormatSupport(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

    if ((properties.optimalTilingFeatures & requiredFeatures) != requiredFeatures) {
        throw std::runtime_error("BRDF LUT format does not support sampled linear-filtered uploads.");
    }
}

VkImageMemoryBarrier2 brdfLutBarrier(
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

void recordImageBarrier(VkCommandBuffer commandBuffer, const VkImageMemoryBarrier2& barrier)
{
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

} // namespace

VulkanBrdfLut::~VulkanBrdfLut()
{
    reset();
}

VulkanBrdfLut::VulkanBrdfLut(VulkanBrdfLut&& other) noexcept
{
    moveFrom(other);
}

VulkanBrdfLut& VulkanBrdfLut::operator=(VulkanBrdfLut&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanBrdfLut::create(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t size)
{
    reset();

    if (size == 0) {
        throw std::runtime_error("Cannot create a zero-sized VulkanBrdfLut.");
    }

    context_ = &context;
    width_ = size;
    height_ = size;
    format_ = VK_FORMAT_R8G8_UNORM;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    validateBrdfLutFormatSupport(context.physicalDevice(), format_);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    const std::vector<uint8_t> pixels = makeSplitSumBrdfLut(size);
    uploadPixels(context, commandContext, pixels);
    createImageView();
    createSampler();
}

void VulkanBrdfLut::reset()
{
    if (!context_) {
        return;
    }

    if (sampler_) {
        vkDestroySampler(context_->vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    if (imageView_) {
        vkDestroyImageView(context_->vkDevice(), imageView_, nullptr);
        imageView_ = VK_NULL_HANDLE;
    }

    if (image_) {
        vmaDestroyImage(context_->allocator(), image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }

    context_ = nullptr;
    width_ = 0;
    height_ = 0;
    format_ = VK_FORMAT_R8G8_UNORM;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanBrdfLut::uploadPixels(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    const std::vector<uint8_t>& pixels)
{
    VulkanBuffer stagingBuffer;
    VulkanBufferCreateInfo stagingInfo{};
    stagingInfo.size = static_cast<VkDeviceSize>(pixels.size());
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    stagingInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    stagingBuffer.createBuffer(context, stagingInfo);
    stagingBuffer.upload(std::as_bytes(std::span<const uint8_t>(pixels.data(), pixels.size())));

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext.commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(context.vkDevice(), &allocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    const VkImageMemoryBarrier2 toTransfer = brdfLutBarrier(
        image_,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);
    recordImageBarrier(commandBuffer, toTransfer);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width_, height_, 1};

    vkCmdCopyBufferToImage(
        commandBuffer,
        stagingBuffer.buffer(),
        image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion);

    const VkImageMemoryBarrier2 toShaderRead = brdfLutBarrier(
        image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    recordImageBarrier(commandBuffer, toShaderRead);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;

    VK_CHECK(vkQueueSubmit2(context.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(context.graphicsQueue()));
    vkFreeCommandBuffers(context.vkDevice(), commandContext.commandPool(), 1, &commandBuffer);

    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void VulkanBrdfLut::createImageView()
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format_;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(context_->vkDevice(), &viewInfo, nullptr, &imageView_));
}

void VulkanBrdfLut::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_->vkDevice(), &samplerInfo, nullptr, &sampler_));
}

void VulkanBrdfLut::moveFrom(VulkanBrdfLut& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    width_ = std::exchange(other.width_, 0);
    height_ = std::exchange(other.height_, 0);
    format_ = std::exchange(other.format_, VK_FORMAT_R8G8_UNORM);
    layout_ = std::exchange(other.layout_, VK_IMAGE_LAYOUT_UNDEFINED);
}

} // namespace ve::rhi
