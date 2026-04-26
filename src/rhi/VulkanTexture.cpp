#include "rhi/VulkanTexture.h"

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

constexpr uint32_t kRgbaChannels = 4;

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

void validateTextureFormatSupport(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((properties.optimalTilingFeatures & requiredFeatures) != requiredFeatures) {
        throw std::runtime_error("Texture format does not support transfer destination and sampled image usage.");
    }
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
    createFromRgba8(context, commandContext, width, height, std::span<const uint8_t>(pixels.data(), pixels.size()));
}

void VulkanTexture::createFromRgba8(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t width,
    uint32_t height,
    std::span<const uint8_t> pixels,
    VkFormat format)
{
    reset();

    if (width == 0 || height == 0) {
        throw std::runtime_error("Cannot create a zero-sized VulkanTexture.");
    }

    const size_t expectedByteCount = static_cast<size_t>(width) * height * kRgbaChannels;
    if (pixels.size_bytes() != expectedByteCount) {
        throw std::runtime_error("RGBA8 texture pixel data has the wrong byte count.");
    }

    validateTextureFormatSupport(context.physicalDevice(), format);

    context_ = &context;
    width_ = width;
    height_ = height;
    format_ = format;

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
    // This barrier makes the image writable by the transfer copy.
    VkImageMemoryBarrier2 toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    toTransfer.srcAccessMask = VK_ACCESS_2_NONE;
    toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = image_;
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.baseMipLevel = 0;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount = 1;

    VkDependencyInfo toTransferDependency{};
    toTransferDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    toTransferDependency.imageMemoryBarrierCount = 1;
    toTransferDependency.pImageMemoryBarriers = &toTransfer;
    vkCmdPipelineBarrier2(commandBuffer, &toTransferDependency);

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

    // Transfer writes must be visible to fragment shader texture sampling.
    // The descriptor will always refer to the final shader-read-only layout.
    VkImageMemoryBarrier2 toShaderRead{};
    toShaderRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toShaderRead.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toShaderRead.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toShaderRead.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    toShaderRead.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShaderRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShaderRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toShaderRead.image = image_;
    toShaderRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toShaderRead.subresourceRange.baseMipLevel = 0;
    toShaderRead.subresourceRange.levelCount = 1;
    toShaderRead.subresourceRange.baseArrayLayer = 0;
    toShaderRead.subresourceRange.layerCount = 1;

    VkDependencyInfo toShaderReadDependency{};
    toShaderReadDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    toShaderReadDependency.imageMemoryBarrierCount = 1;
    toShaderReadDependency.pImageMemoryBarriers = &toShaderRead;
    vkCmdPipelineBarrier2(commandBuffer, &toShaderReadDependency);

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

void VulkanTexture::createImageView()
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
    samplerInfo.maxLod = 0.0f;
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
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
}

} // namespace ve::rhi
