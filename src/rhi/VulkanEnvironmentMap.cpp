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
constexpr float kByteScale = 1.0f / 255.0f;
constexpr float kIrradianceScale = 0.55f;

struct FaceGradient {
    std::array<uint8_t, 3> horizon;
    std::array<uint8_t, 3> zenith;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

constexpr std::array<FaceGradient, kCubeFaceCount> kProceduralFaceGradients = {{
    {{{82, 110, 142}}, {{150, 191, 226}}},
    {{{72, 105, 126}}, {{128, 168, 203}}},
    {{{106, 153, 211}}, {{190, 223, 247}}},
    {{{58, 53, 48}}, {{137, 119, 91}}},
    {{{76, 108, 150}}, {{154, 198, 235}}},
    {{{48, 61, 89}}, {{111, 138, 181}}},
}};

constexpr std::array<Vec3, kCubeFaceCount> kCubeFaceDirections = {{
    {1.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, -1.0f},
}};

float lerpFloat(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

uint32_t calculateMipLevels(uint32_t faceSize)
{
    return static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(faceSize)))) + 1;
}

uint32_t mipFaceSize(uint32_t faceSize, uint32_t mipLevel)
{
    return std::max(1u, faceSize >> mipLevel);
}

size_t mipFaceByteCount(uint32_t faceSize, uint32_t mipLevel)
{
    const uint32_t size = mipFaceSize(faceSize, mipLevel);
    return static_cast<size_t>(size) * size * kRgbaChannels;
}

size_t mipChainByteCount(uint32_t faceSize, uint32_t mipLevels)
{
    size_t byteCount = 0;
    for (uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel) {
        byteCount += mipFaceByteCount(faceSize, mipLevel) * kCubeFaceCount;
    }
    return byteCount;
}

uint8_t floatToByte(float value)
{
    return static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
}

Vec3 operator+(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vec3 operator*(Vec3 value, float scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

float dot(Vec3 lhs, Vec3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 normalize(Vec3 value)
{
    const float length = std::sqrt(dot(value, value));
    if (length <= 0.0f) {
        return {0.0f, 1.0f, 0.0f};
    }

    const float inverseLength = 1.0f / length;
    return value * inverseLength;
}

Vec3 cubemapTexelDirection(uint32_t face, float u, float v)
{
    const float sc = u * 2.0f - 1.0f;
    const float tc = v * 2.0f - 1.0f;

    switch (face) {
    case 0:
        return normalize({1.0f, -tc, -sc});
    case 1:
        return normalize({-1.0f, -tc, sc});
    case 2:
        return normalize({sc, 1.0f, tc});
    case 3:
        return normalize({sc, -1.0f, -tc});
    case 4:
        return normalize({sc, -tc, 1.0f});
    case 5:
        return normalize({-sc, -tc, -1.0f});
    default:
        return {0.0f, 1.0f, 0.0f};
    }
}

Vec3 proceduralFaceColor(uint32_t face, float u, float v)
{
    const FaceGradient& gradient = kProceduralFaceGradients.at(face);
    const float sideShade = 0.9f + 0.1f * (1.0f - std::abs(u * 2.0f - 1.0f));

    return {
        lerpFloat(
            static_cast<float>(gradient.horizon[0]),
            static_cast<float>(gradient.zenith[0]) * sideShade,
            v) * kByteScale,
        lerpFloat(
            static_cast<float>(gradient.horizon[1]),
            static_cast<float>(gradient.zenith[1]) * sideShade,
            v) * kByteScale,
        lerpFloat(
            static_cast<float>(gradient.horizon[2]),
            static_cast<float>(gradient.zenith[2]) * sideShade,
            v) * kByteScale,
    };
}

Vec3 averageProceduralFaceColor(uint32_t face)
{
    constexpr uint32_t kSampleCount = 4;
    Vec3 color{};

    for (uint32_t y = 0; y < kSampleCount; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kSampleCount);
        for (uint32_t x = 0; x < kSampleCount; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kSampleCount);
            color = color + proceduralFaceColor(face, u, v);
        }
    }

    return color * (1.0f / static_cast<float>(kSampleCount * kSampleCount));
}

void writeRgba(std::vector<uint8_t>& pixels, size_t offset, Vec3 color)
{
    pixels[offset + 0] = floatToByte(color.x);
    pixels[offset + 1] = floatToByte(color.y);
    pixels[offset + 2] = floatToByte(color.z);
    pixels[offset + 3] = 255;
}

std::vector<uint8_t> makeProceduralEnvironmentFaces(uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural environment map.");
    }

    std::vector<uint8_t> pixels(
        static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    // Layer order follows Vulkan cube-map convention: +X, -X, +Y, -Y, +Z, -Z.
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = faceSize == 1
                ? 1.0f
                : static_cast<float>(y) / static_cast<float>(faceSize - 1);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = faceSize == 1
                    ? 0.5f
                    : static_cast<float>(x) / static_cast<float>(faceSize - 1);
                const size_t offset = ((static_cast<size_t>(face) * faceSize * faceSize)
                    + (static_cast<size_t>(y) * faceSize + x)) * kRgbaChannels;

                writeRgba(pixels, offset, proceduralFaceColor(face, u, v));
            }
        }
    }

    return pixels;
}

std::vector<uint8_t> makeProceduralDiffuseIrradianceFaces(uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural diffuse irradiance map.");
    }

    std::array<Vec3, kCubeFaceCount> faceColors{};
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        faceColors.at(face) = averageProceduralFaceColor(face);
    }

    std::vector<uint8_t> pixels(
        static_cast<size_t>(faceSize) * faceSize * kCubeFaceCount * kRgbaChannels);

    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        for (uint32_t y = 0; y < faceSize; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(faceSize);
            for (uint32_t x = 0; x < faceSize; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(faceSize);
                const Vec3 normal = cubemapTexelDirection(face, u, v);
                Vec3 irradiance{};
                float totalWeight = 0.0f;

                for (uint32_t sourceFace = 0; sourceFace < kCubeFaceCount; ++sourceFace) {
                    const float weight = std::max(dot(normal, kCubeFaceDirections.at(sourceFace)), 0.0f);
                    irradiance = irradiance + faceColors.at(sourceFace) * weight;
                    totalWeight += weight;
                }

                if (totalWeight > 0.0f) {
                    irradiance = irradiance * (kIrradianceScale / totalWeight);
                }

                const size_t offset = ((static_cast<size_t>(face) * faceSize * faceSize)
                    + (static_cast<size_t>(y) * faceSize + x)) * kRgbaChannels;
                writeRgba(pixels, offset, irradiance);
            }
        }
    }

    return pixels;
}

std::vector<uint8_t> makeProceduralPrefilteredSpecularFaces(uint32_t faceSize)
{
    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized procedural prefiltered environment map.");
    }

    const uint32_t mipLevels = calculateMipLevels(faceSize);

    std::array<Vec3, kCubeFaceCount> faceColors{};
    Vec3 globalColor{};
    for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
        faceColors.at(face) = averageProceduralFaceColor(face);
        globalColor = globalColor + faceColors.at(face);
    }
    globalColor = globalColor * (1.0f / static_cast<float>(kCubeFaceCount));

    std::vector<uint8_t> pixels(mipChainByteCount(faceSize, mipLevels));
    size_t mipOffset = 0;

    // This is intentionally a cheap CPU approximation: higher roughness mips
    // blend each texel toward a per-face average and then toward the global sky
    // color instead of doing importance-sampled convolution.
    for (uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel) {
        const uint32_t size = mipFaceSize(faceSize, mipLevel);
        const float roughness = mipLevels == 1
            ? 0.0f
            : static_cast<float>(mipLevel) / static_cast<float>(mipLevels - 1);
        const float blurBlend = std::clamp(std::pow(roughness, 0.75f), 0.0f, 1.0f);
        const float globalBlend = roughness * roughness * 0.6f;
        const float roughnessEnergy = lerpFloat(1.08f, 0.78f, roughness);

        for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
            const Vec3 lowFrequency =
                faceColors.at(face) * (1.0f - globalBlend) + globalColor * globalBlend;

            for (uint32_t y = 0; y < size; ++y) {
                const float v = size == 1
                    ? 0.5f
                    : static_cast<float>(y) / static_cast<float>(size - 1);
                for (uint32_t x = 0; x < size; ++x) {
                    const float u = size == 1
                        ? 0.5f
                        : static_cast<float>(x) / static_cast<float>(size - 1);
                    const Vec3 baseColor = proceduralFaceColor(face, u, v);
                    const Vec3 color =
                        (baseColor * (1.0f - blurBlend) + lowFrequency * blurBlend) * roughnessEnergy;
                    const size_t offset = mipOffset
                        + ((static_cast<size_t>(face) * size * size)
                            + (static_cast<size_t>(y) * size + x)) * kRgbaChannels;
                    writeRgba(pixels, offset, color);
                }
            }
        }

        mipOffset += mipFaceByteCount(faceSize, mipLevel) * kCubeFaceCount;
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
    VkAccessFlags2 dstAccessMask,
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
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = levelCount;
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

void VulkanEnvironmentMap::createProceduralDiffuseIrradiance(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t faceSize)
{
    const std::vector<uint8_t> pixels = makeProceduralDiffuseIrradianceFaces(faceSize);
    createFromRgba8Faces(
        context,
        commandContext,
        faceSize,
        std::span<const uint8_t>(pixels.data(), pixels.size()),
        VK_FORMAT_R8G8B8A8_UNORM);
}

void VulkanEnvironmentMap::createProceduralPrefilteredSpecular(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t faceSize)
{
    const std::vector<uint8_t> pixels = makeProceduralPrefilteredSpecularFaces(faceSize);
    createFromRgba8MipFaces(
        context,
        commandContext,
        faceSize,
        calculateMipLevels(faceSize),
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
    createFromRgba8MipFaces(
        context,
        commandContext,
        faceSize,
        1,
        pixels,
        format);
}

void VulkanEnvironmentMap::createFromRgba8MipFaces(
    VulkanContext& context,
    const VulkanCommandContext& commandContext,
    uint32_t faceSize,
    uint32_t mipLevels,
    std::span<const uint8_t> pixels,
    VkFormat format)
{
    reset();

    if (faceSize == 0) {
        throw std::runtime_error("Cannot create a zero-sized environment map.");
    }
    if (mipLevels == 0) {
        throw std::runtime_error("Cannot create an environment map with zero mip levels.");
    }

    const size_t expectedByteCount = mipChainByteCount(faceSize, mipLevels);
    if (pixels.size_bytes() != expectedByteCount) {
        throw std::runtime_error("Environment cubemap RGBA8 mip data has the wrong byte count.");
    }

    context_ = &context;
    faceSize_ = faceSize;
    mipLevels_ = mipLevels;
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

    uploadMipFaces(context, commandContext, std::as_bytes(pixels));
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

void VulkanEnvironmentMap::uploadMipFaces(
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
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        mipLevels_);
    recordImageBarrier(commandBuffer, toTransfer);

    std::vector<VkBufferImageCopy> copyRegions;
    copyRegions.reserve(static_cast<size_t>(mipLevels_) * kCubeFaceCount);

    VkDeviceSize bufferOffset = 0;
    for (uint32_t mipLevel = 0; mipLevel < mipLevels_; ++mipLevel) {
        const uint32_t size = mipFaceSize(faceSize_, mipLevel);
        const VkDeviceSize faceByteSize = static_cast<VkDeviceSize>(mipFaceByteCount(faceSize_, mipLevel));

        for (uint32_t face = 0; face < kCubeFaceCount; ++face) {
            VkBufferImageCopy copyRegion{};
            copyRegion.bufferOffset = bufferOffset + faceByteSize * face;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = mipLevel;
            copyRegion.imageSubresource.baseArrayLayer = face;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent = {size, size, 1};
            copyRegions.push_back(copyRegion);
        }

        bufferOffset += faceByteSize * kCubeFaceCount;
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
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        mipLevels_);
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
    samplerInfo.maxLod = mipLevels_ > 0 ? static_cast<float>(mipLevels_ - 1) : 0.0f;
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
