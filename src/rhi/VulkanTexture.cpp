#include "rhi/VulkanTexture.h"

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

constexpr uint32_t kRgbaChannels = 4;

struct StbiImageDeleter {
    void operator()(stbi_uc* pixels) const
    {
        stbi_image_free(pixels);
    }
};

using StbiPixels = std::unique_ptr<stbi_uc, StbiImageDeleter>;

std::vector<uint8_t> makeCheckerboardPixels(uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("Cannot create a zero-sized checkerboard texture.");
    }

    constexpr uint32_t tileSize = 32;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * kRgbaChannels);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const bool lightTile = (((x / tileSize) + (y / tileSize)) % 2) == 0;
            const uint8_t value = lightTile ? 235 : 35;
            const size_t offset = (static_cast<size_t>(y) * width + x) * kRgbaChannels;

            pixels[offset + 0] = value;
            pixels[offset + 1] = value;
            pixels[offset + 2] = lightTile ? 235 : 45;
            pixels[offset + 3] = 255;
        }
    }

    return pixels;
}

uint32_t calculateMipLevels(uint32_t width, uint32_t height)
{
    const uint32_t maxDimension = std::max(width, height);
    return static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(maxDimension)))) + 1;
}

void validateTextureFormatSupport(VkPhysicalDevice physicalDevice, VkFormat format, VkImageUsageFlags usage)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    VkFormatFeatureFlags requiredFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0) {
        requiredFeatures |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    }
    if ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0) {
        requiredFeatures |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    }

    if ((properties.optimalTilingFeatures & requiredFeatures) != requiredFeatures) {
        throw std::runtime_error("Texture format does not support the requested sampled image usage.");
    }
}

bool supportsLinearBlit(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT
        | VK_FORMAT_FEATURE_BLIT_DST_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures;
}

VkImageMemoryBarrier2 textureBarrier(
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask,
    uint32_t baseMipLevel,
    uint32_t levelCount)
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
    barrier.subresourceRange.baseMipLevel = baseMipLevel;
    barrier.subresourceRange.levelCount = levelCount;
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

VulkanTexture::~VulkanTexture()
{
    reset();
}

VulkanTexture::VulkanTexture(VulkanTexture&& other) noexcept
{
    moveFrom(other);
}

VulkanTexture& VulkanTexture::operator=(VulkanTexture&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanTexture::createCheckerboard(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t width,
    uint32_t height)
{
    const std::vector<uint8_t> pixels = makeCheckerboardPixels(width, height);
    createFromRgba8(
        context,
        commandContext,
        width,
        height,
        std::span<const uint8_t>(pixels.data(), pixels.size()),
        VK_FORMAT_R8G8B8A8_UNORM,
        true);
}

void VulkanTexture::createFromFile(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    const std::filesystem::path& path,
    bool generateMipmaps)
{
    int loadedWidth = 0;
    int loadedHeight = 0;
    const std::string filename = path.string();
    StbiPixels loadedPixels(stbi_load(filename.c_str(), &loadedWidth, &loadedHeight, nullptr, STBI_rgb_alpha));

    if (!loadedPixels) {
        const char* failureReason = stbi_failure_reason();
        throw std::runtime_error(
            "Failed to load texture file '" + filename + "': "
            + (failureReason ? failureReason : "unknown stb_image error"));
    }
    if (loadedWidth <= 0 || loadedHeight <= 0) {
        throw std::runtime_error("Texture file has invalid dimensions: " + filename);
    }

    const uint32_t width = static_cast<uint32_t>(loadedWidth);
    const uint32_t height = static_cast<uint32_t>(loadedHeight);
    const size_t byteCount = static_cast<size_t>(width) * height * kRgbaChannels;
    createFromRgba8(
        context,
        commandContext,
        width,
        height,
        std::span<const uint8_t>(loadedPixels.get(), byteCount),
        VK_FORMAT_R8G8B8A8_UNORM,
        generateMipmaps);
}

void VulkanTexture::createFromRgba8(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t width,
    uint32_t height,
    std::span<const uint8_t> pixels,
    VkFormat format,
    bool generateMipmaps)
{
    reset();

    if (width == 0 || height == 0) {
        throw std::runtime_error("Cannot create a zero-sized VulkanTexture.");
    }

    const size_t expectedByteCount = static_cast<size_t>(width) * height * kRgbaChannels;
    if (pixels.size_bytes() != expectedByteCount) {
        throw std::runtime_error("RGBA8 texture pixel data has the wrong byte count.");
    }

    context_ = &context;
    width_ = width;
    height_ = height;
    format_ = format;
    mipLevels_ = generateMipmaps ? calculateMipLevels(width_, height_) : 1;
    if (mipLevels_ > 1 && !supportsLinearBlit(context.physicalDevice(), format_)) {
        // Some otherwise valid sampled formats cannot be linearly blitted. For this
        // milestone, fall back to one mip level instead of adding CPU mip generation
        // or a compute path.
        mipLevels_ = 1;
    }

    VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (mipLevels_ > 1) {
        imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    validateTextureFormatSupport(context.physicalDevice(), format_, imageUsage);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = mipLevels_;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = imageUsage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    uploadPixels(context, commandContext, std::as_bytes(pixels));
    createImageView();
    createSampler();
}

void VulkanTexture::reset()
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
    mipLevels_ = 0;
    format_ = VK_FORMAT_UNDEFINED;
}

void VulkanTexture::uploadPixels(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    std::span<const std::byte> pixels)
{
    VulkanBuffer stagingBuffer;
    VulkanBufferCreateInfo stagingInfo{};
    stagingInfo.size = static_cast<VkDeviceSize>(pixels.size_bytes());
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    stagingInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    stagingBuffer.createBuffer(context, stagingInfo);
    stagingBuffer.upload(pixels);

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

    // The upload starts from UNDEFINED because the texture image has no useful contents yet.
    // All mip levels become transfer destinations before level 0 is copied and later blits run.
    const VkImageMemoryBarrier2 toTransfer = textureBarrier(
        image_,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        0,
        mipLevels_);
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

    if (mipLevels_ > 1) {
        generateMipmaps(commandBuffer);
    } else {
        // Transfer writes must be visible to fragment shader texture sampling.
        // The descriptor will always refer to the final shader-read-only layout.
        const VkImageMemoryBarrier2 toShaderRead = textureBarrier(
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            0,
            1);
        recordImageBarrier(commandBuffer, toShaderRead);
    }

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
}

void VulkanTexture::generateMipmaps(VkCommandBuffer commandBuffer)
{
    int32_t mipWidth = static_cast<int32_t>(width_);
    int32_t mipHeight = static_cast<int32_t>(height_);

    for (uint32_t mipLevel = 1; mipLevel < mipLevels_; ++mipLevel) {
        const VkImageMemoryBarrier2 previousToTransferSource = textureBarrier(
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            mipLevel - 1,
            1);
        recordImageBarrier(commandBuffer, previousToTransferSource);

        const int32_t nextMipWidth = std::max(1, mipWidth / 2);
        const int32_t nextMipHeight = std::max(1, mipHeight / 2);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = mipLevel - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nextMipWidth, nextMipHeight, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = mipLevel;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(
            commandBuffer,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_LINEAR);

        const VkImageMemoryBarrier2 previousToShaderRead = textureBarrier(
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            mipLevel - 1,
            1);
        recordImageBarrier(commandBuffer, previousToShaderRead);

        mipWidth = nextMipWidth;
        mipHeight = nextMipHeight;
    }

    const VkImageMemoryBarrier2 finalMipToShaderRead = textureBarrier(
        image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        mipLevels_ - 1,
        1);
    recordImageBarrier(commandBuffer, finalMipToShaderRead);
}

void VulkanTexture::createImageView()
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format_;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels_;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(context_->vkDevice(), &viewInfo, nullptr, &imageView_));
}

void VulkanTexture::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    // Anisotropy requires enabling the samplerAnisotropy device feature. This milestone
    // keeps sampler creation minimal and leaves anisotropy disabled.
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;

    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels_);
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_->vkDevice(), &samplerInfo, nullptr, &sampler_));
}

void VulkanTexture::moveFrom(VulkanTexture& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    width_ = std::exchange(other.width_, 0);
    height_ = std::exchange(other.height_, 0);
    mipLevels_ = std::exchange(other.mipLevels_, 0);
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
}

} // namespace ve::rhi
