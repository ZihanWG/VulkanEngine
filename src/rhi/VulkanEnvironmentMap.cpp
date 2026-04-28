#include "rhi/VulkanEnvironmentMap.h"

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

constexpr uint32_t kCubeFaceCount = 6;
constexpr uint32_t kRgbaChannels = 4;

struct FaceGradient {
    std::array<uint8_t, 3> horizon;
    std::array<uint8_t, 3> zenith;
};

uint8_t lerpByte(uint8_t a, uint8_t b, float t)
{
    const float value = static_cast<float>(a) * (1.0f - t) + static_cast<float>(b) * t;
    return static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
}

std::vector<uint8_t> makeProceduralEnvironmentFaces(uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural environment map.");
    }

    // Layer order follows Vulkan cube-map convention: +X, -X, +Y, -Y, +Z, -Z.
    constexpr std::array<FaceGradient, kCubeFaceCount> gradients = {{
        {{{82, 110, 142}}, {{150, 191, 226}}},
        {{{72, 105, 126}}, {{128, 168, 203}}},
        {{{106, 153, 211}}, {{190, 223, 247}}},
        {{{58, 53, 48}}, {{137, 119, 91}}},
        {{{76, 108, 150}}, {{154, 198, 235}}},
        {{{48, 61, 89}}, {{111, 138, 181}}},
    }};

    std::vector<uint8_t> pixels(
        static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        const FaceGradient& gradient = gradients.at(face);
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = faceSize == 1
                ? 1.0f
                : static_cast<float>(y) / static_cast<float>(faceSize - 1);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = faceSize == 1
                    ? 0.5f
                    : static_cast<float>(x) / static_cast<float>(faceSize - 1);
                const float sideShade = 0.9f + 0.1f * (1.0f - std::abs(u * 2.0f - 1.0f));
                const size_t offset = ((static_cast<size_t>(face) * faceSize * faceSize)
                    + (static_cast<size_t>(y) * faceSize + x)) * kRgbaChannels;

                pixels[offset + 0] = lerpByte(
                    gradient.horizon[0],
                    static_cast<uint8_t>(static_cast<float>(gradient.zenith[0]) * sideShade),
                    v);
                pixels[offset + 1] = lerpByte(
                    gradient.horizon[1],
                    static_cast<uint8_t>(static_cast<float>(gradient.zenith[1]) * sideShade),
                    v);
                pixels[offset + 2] = lerpByte(
                    gradient.horizon[2],
                    static_cast<uint8_t>(static_cast<float>(gradient.zenith[2]) * sideShade),
                    v);
                pixels[offset + 3] = 255;
            }
        }
    }

    return pixels;
}

void validateEnvironmentFormatSupport(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    constexpr VkFormatFeatureFlags requiredFeatures =
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT
        | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

    if ((properties.optimalTilingFeatures & requiredFeatures) != requiredFeatures) {
        throw std::runtime_error("Environment map format does not support sampled cube image uploads.");
    }
}

VkImageMemoryBarrier2 environmentBarrier(
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
    barrier.subresourceRange.layerCount = kCubeFaceCount;
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

VulkanEnvironmentMap::~VulkanEnvironmentMap()
{
    reset();
}

VulkanEnvironmentMap::VulkanEnvironmentMap(VulkanEnvironmentMap&& other) noexcept
{
    moveFrom(other);
}

VulkanEnvironmentMap& VulkanEnvironmentMap::operator=(VulkanEnvironmentMap&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanEnvironmentMap::createProcedural(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t faceSize)
{
    const std::vector<uint8_t> pixels = makeProceduralEnvironmentFaces(faceSize);
    createFromRgba8Faces(
        context,
        commandContext,
        faceSize,
        std::span<const uint8_t>(pixels.data(), pixels.size()),
        VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanEnvironmentMap::createFromRgba8Faces(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t faceSize,
    std::span<const uint8_t> pixels,
    VkFormat format)
{
    reset();

    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized environment map.");
    }

    const size_t expectedByteCount =
        static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels;
    if (pixels.size_bytes() != expectedByteCount) {
        throw std::runtime_error("Environment cubemap RGBA8 face data has the wrong byte count.");
    }

    context_ = &context;
    faceSize_ = faceSize;
    mipLevels_ = 1;
    format_ = format;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    validateEnvironmentFormatSupport(context.physicalDevice(), format_);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format_;
    imageInfo.extent = {faceSize_, faceSize_, 1};
    imageInfo.mipLevels = mipLevels_;
    imageInfo.arrayLayers = kCubeFaceCount;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateImage(context.allocator(), &imageInfo, &allocationInfo, &image_, &allocation_, nullptr));

    uploadFaces(context, commandContext, std::as_bytes(pixels));
    createImageView();
    createSampler();
}

void VulkanEnvironmentMap::reset()
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
    faceSize_ = 0;
    mipLevels_ = 0;
    format_ = VK_FORMAT_UNDEFINED;
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanEnvironmentMap::uploadFaces(
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

    const VkImageMemoryBarrier2 toTransfer = environmentBarrier(
        image_,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);
    recordImageBarrier(commandBuffer, toTransfer);

    std::array<VkBufferImageCopy, kCubeFaceCount> copyRegions{};
    const VkDeviceSize faceByteSize =
        static_cast<VkDeviceSize>(faceSize_) * faceSize_ * kRgbaChannels;
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        VkBufferImageCopy& copyRegion = copyRegions.at(face);
        copyRegion.bufferOffset = faceByteSize * face;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = face;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = {0, 0, 0};
        copyRegion.imageExtent = {faceSize_, faceSize_, 1};
    }

    vkCmdCopyBufferToImage(
        commandBuffer,
        stagingBuffer.buffer(),
        image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<uint32_t>(copyRegions.size()),
        copyRegions.data());

    const VkImageMemoryBarrier2 toShaderRead = environmentBarrier(
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

void VulkanEnvironmentMap::createImageView()
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = format_;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels_;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = kCubeFaceCount;

    VK_CHECK(vkCreateImageView(context_->vkDevice(), &viewInfo, nullptr, &imageView_));
}

void VulkanEnvironmentMap::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
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

void VulkanEnvironmentMap::moveFrom(VulkanEnvironmentMap& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    image_ = std::exchange(other.image_, VK_NULL_HANDLE);
    allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
    imageView_ = std::exchange(other.imageView_, VK_NULL_HANDLE);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    faceSize_ = std::exchange(other.faceSize_, 0);
    mipLevels_ = std::exchange(other.mipLevels_, 0);
    format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
    layout_ = std::exchange(other.layout_, VK_IMAGE_LAYOUT_UNDEFINED);
}

} // namespace ve::rhi
