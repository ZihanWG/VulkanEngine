#include "rhi/VulkanTexture.h"

#include "assets/Ktx2.h"
#include "assets/TextureCook.h"
#include "renderer/AssetLoadStats.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanUploadBatch.h"

#define STB_IMAGE_IMPLEMENTATION
// Make stb_image's failure-reason string thread-local so concurrent decode jobs
// (see VulkanTexture::decodeImageFile / decodeImageBytes) do not race on it.
#define STBI_THREAD_LOCAL thread_local
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

// The texture cook is GPU-free, so it spells its formats as the raw numbers KTX2
// stores rather than including a Vulkan header. This is where those numbers are
// held to the real enumerators: a mismatch would produce a KTX2 file that
// declares one format and is decoded as another, which no validation layer can
// see. Same reasoning as the layout static_asserts on the GPU structs.
static_assert(assets::kVkFormatBc5UnormBlock == VK_FORMAT_BC5_UNORM_BLOCK);
static_assert(assets::kVkFormatBc7UnormBlock == VK_FORMAT_BC7_UNORM_BLOCK);
static_assert(assets::kVkFormatBc7SrgbBlock == VK_FORMAT_BC7_SRGB_BLOCK);
static_assert(assets::kVkFormatR8G8B8A8Unorm == VK_FORMAT_R8G8B8A8_UNORM);
static_assert(assets::kVkFormatR8G8B8A8Srgb == VK_FORMAT_R8G8B8A8_SRGB);

constexpr uint32_t kRgbaChannels = 4;

// Covers the formats this class actually creates today. Anything else falls back
// to the numeric value rather than pretending to know the name -- the asset-load
// baseline is evidence, so an unknown format must read as unknown.
std::string textureFormatName(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "VK_FORMAT_R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "VK_FORMAT_R8G8B8A8_SRGB";
    case VK_FORMAT_R8G8_UNORM:
        return "VK_FORMAT_R8G8_UNORM";
    case VK_FORMAT_R8_UNORM:
        return "VK_FORMAT_R8_UNORM";
    // The cooked formats. Named so --asset-load-stats reads as evidence of the
    // texture cook rather than as an unexplained VkFormat(146).
    case VK_FORMAT_BC5_UNORM_BLOCK:
        return "VK_FORMAT_BC5_UNORM_BLOCK";
    case VK_FORMAT_BC7_UNORM_BLOCK:
        return "VK_FORMAT_BC7_UNORM_BLOCK";
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return "VK_FORMAT_BC7_SRGB_BLOCK";
    case VK_FORMAT_UNDEFINED:
        return "VK_FORMAT_UNDEFINED";
    default:
        return "VkFormat(" + std::to_string(static_cast<int>(format)) + ")";
    }
}

// True for the BC/ETC/ASTC ranges. Always false today -- that is exactly the
// baseline fact this instrumentation exists to record.
bool isBlockCompressedFormat(VkFormat format)
{
    const int value = static_cast<int>(format);
    const bool bc = value >= static_cast<int>(VK_FORMAT_BC1_RGB_UNORM_BLOCK) &&
                    value <= static_cast<int>(VK_FORMAT_BC7_SRGB_BLOCK);
    const bool etcOrEac = value >= static_cast<int>(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK) &&
                          value <= static_cast<int>(VK_FORMAT_EAC_R11G11_SNORM_BLOCK);
    const bool astc = value >= static_cast<int>(VK_FORMAT_ASTC_4x4_UNORM_BLOCK) &&
                      value <= static_cast<int>(VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
    return bc || etcOrEac || astc;
}

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

    constexpr VkFormatFeatureFlags requiredFeatures = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures;
}

TextureColorSpace colorSpaceForFormat(VkFormat format)
{
    return format == VK_FORMAT_R8G8B8A8_SRGB ? TextureColorSpace::SRGB : TextureColorSpace::Linear;
}

VkImageMemoryBarrier2 textureBarrier(VkImage image,
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

// Shared by both upload paths so the one-time command buffer, the submit, and the
// wait exist in one place rather than two. The vkQueueWaitIdle here is the
// load-time serialization the async transfer-queue work exists to remove; it is
// startup only and never on the frame path.
VkCommandBuffer beginOneTimeUpload(VulkanContext& context, const VulkanCommandContext& commandContext)
{
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

    return commandBuffer;
}

void endOneTimeUpload(VulkanContext& context, const VulkanCommandContext& commandContext, VkCommandBuffer commandBuffer)
{
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

std::vector<uint8_t> readFileBytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open cooked texture: " + path.string());
    }

    const std::streamsize size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("Cooked texture is empty: " + path.string());
    }

    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) {
        throw std::runtime_error("Failed to read cooked texture: " + path.string());
    }

    return bytes;
}

} // namespace

VkFormat rgba8FormatForColorSpace(TextureColorSpace colorSpace)
{
    switch (colorSpace) {
    case TextureColorSpace::Linear:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case TextureColorSpace::SRGB:
        return VK_FORMAT_R8G8B8A8_SRGB;
    }

    throw std::runtime_error("Unsupported texture color space.");
}

assets::BlockCompressionCaps queryBlockCompressionCaps(VkPhysicalDevice physicalDevice)
{
    // Cached because the caps cannot change under a running device and every
    // texture load consults them. Keyed by handle so a second device (there is
    // only ever one today) cannot silently inherit the first one's answer.
    static std::mutex cacheMutex;
    static std::unordered_map<VkPhysicalDevice, assets::BlockCompressionCaps> cache;

    const std::lock_guard<std::mutex> lock(cacheMutex);
    if (const auto found = cache.find(physicalDevice); found != cache.end()) {
        return found->second;
    }

    // Sampling and linear filtering and transfer-dst. The textureCompressionBC
    // feature bit is not enough on its own: it does not promise a filterable
    // sampled image, and an unfilterable texture is no use to a mip chain.
    const auto samplable = [physicalDevice](VkFormat format) {
        constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                                                  | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
                                                  | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        return (properties.optimalTilingFeatures & required) == required;
    };

    assets::BlockCompressionCaps caps{};
    // Both BC7 spellings must work: the cook emits sRGB for colour and UNORM for
    // data maps, and reporting BC7 support while one of them is missing would
    // send half the textures down a path the device cannot sample.
    caps.bc7 = samplable(VK_FORMAT_BC7_SRGB_BLOCK) && samplable(VK_FORMAT_BC7_UNORM_BLOCK);
    caps.bc5 = samplable(VK_FORMAT_BC5_UNORM_BLOCK);

    cache.emplace(physicalDevice, caps);
    return caps;
}

bool cookedTextureAvailable(VkPhysicalDevice physicalDevice,
                            const std::filesystem::path& source,
                            assets::TextureUsage usage)
{
    if (source.empty()) {
        return false;
    }

    const assets::BlockCompressionCaps caps = queryBlockCompressionCaps(physicalDevice);
    if (!assets::cookedFormatUsable(caps, assets::cookedFormatForUsage(usage), usage)) {
        return false;
    }

    const std::filesystem::path sidecar = assets::ktx2SidecarPath(source, usage);

    std::error_code error;
    if (!std::filesystem::exists(sidecar, error) || error) {
        return false;
    }

    const auto sidecarTime = std::filesystem::last_write_time(sidecar, error);
    if (error) {
        return false;
    }

    const auto sourceTime = std::filesystem::last_write_time(source, error);
    if (error) {
        // No source to compare against (an embedded or generated image). The
        // cooked file is all there is, so trust it.
        return true;
    }

    // A cook older than its source is stale. Rendering an edited PNG as its
    // previous cooked version is the kind of wrong that survives a whole session
    // unnoticed, so it loses to the source rather than winning by being faster.
    return sidecarTime >= sourceTime;
}

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

void VulkanTexture::createCheckerboard(VulkanContext& context,
                                       const VulkanCommandContext& commandContext,
                                       uint32_t width,
                                       uint32_t height,
                                       TextureColorSpace colorSpace)
{
    const std::vector<uint8_t> pixels = makeCheckerboardPixels(width, height);
    createFromRgba8(context,
                    commandContext,
                    width,
                    height,
                    std::span<const uint8_t>(pixels.data(), pixels.size()),
                    rgba8FormatForColorSpace(colorSpace),
                    true);
    debugMetadata_ = TextureDebugMetadata{
        "Procedural checkerboard",
        {},
        colorSpace,
        TextureDebugSource::ProceduralFallback,
        true,
    };
}

void VulkanTexture::createFromFile(VulkanContext& context,
                                   const VulkanCommandContext& commandContext,
                                   const std::filesystem::path& path,
                                   TextureColorSpace colorSpace,
                                   bool generateMipmaps)
{
    createFromFile(context, commandContext, path, rgba8FormatForColorSpace(colorSpace), generateMipmaps);
    debugMetadata_.colorSpace = colorSpace;
}

DecodedImage VulkanTexture::decodeImageFile(const std::filesystem::path& path)
{
    int loadedWidth = 0;
    int loadedHeight = 0;
    const std::string filename = path.string();
    StbiPixels loadedPixels(stbi_load(filename.c_str(), &loadedWidth, &loadedHeight, nullptr, STBI_rgb_alpha));

    if (!loadedPixels) {
        const char* failureReason = stbi_failure_reason();
        throw std::runtime_error("Failed to load texture file '" + filename +
                                 "': " + (failureReason ? failureReason : "unknown stb_image error"));
    }
    if (loadedWidth <= 0 || loadedHeight <= 0) {
        throw std::runtime_error("Texture file has invalid dimensions: " + filename);
    }

    const uint32_t width = static_cast<uint32_t>(loadedWidth);
    const uint32_t height = static_cast<uint32_t>(loadedHeight);
    const size_t byteCount = static_cast<size_t>(width) * height * kRgbaChannels;
    return DecodedImage{std::vector<uint8_t>(loadedPixels.get(), loadedPixels.get() + byteCount), width, height};
}

DecodedImage VulkanTexture::decodeImageBytes(std::span<const uint8_t> encodedBytes)
{
    if (encodedBytes.empty()) {
        throw std::runtime_error("Encoded texture data is empty.");
    }
    if (encodedBytes.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::runtime_error("Encoded texture data is too large for stb_image.");
    }

    int loadedWidth = 0;
    int loadedHeight = 0;
    StbiPixels loadedPixels(stbi_load_from_memory(encodedBytes.data(),
                                                  static_cast<int>(encodedBytes.size()),
                                                  &loadedWidth,
                                                  &loadedHeight,
                                                  nullptr,
                                                  STBI_rgb_alpha));

    if (!loadedPixels) {
        const char* failureReason = stbi_failure_reason();
        throw std::runtime_error(std::string("Failed to decode embedded texture: ") +
                                 (failureReason ? failureReason : "unknown stb_image error"));
    }
    if (loadedWidth <= 0 || loadedHeight <= 0) {
        throw std::runtime_error("Embedded texture has invalid dimensions.");
    }

    const uint32_t width = static_cast<uint32_t>(loadedWidth);
    const uint32_t height = static_cast<uint32_t>(loadedHeight);
    const size_t byteCount = static_cast<size_t>(width) * height * kRgbaChannels;
    return DecodedImage{std::vector<uint8_t>(loadedPixels.get(), loadedPixels.get() + byteCount), width, height};
}

void VulkanTexture::createFromFile(VulkanContext& context,
                                   const VulkanCommandContext& commandContext,
                                   const std::filesystem::path& path,
                                   VkFormat format,
                                   bool generateMipmaps)
{
    const DecodedImage decoded = decodeImageFile(path);
    createFromRgba8(context, commandContext, decoded.width, decoded.height, decoded.pixels, format, generateMipmaps);
    debugMetadata_ = TextureDebugMetadata{
        path.filename().string(),
        path.string(),
        colorSpaceForFormat(format),
        TextureDebugSource::LoadedFromDisk,
        false,
    };
}

void VulkanTexture::createFromEncodedBytes(VulkanContext& context,
                                           const VulkanCommandContext& commandContext,
                                           std::span<const uint8_t> encodedBytes,
                                           TextureColorSpace colorSpace,
                                           bool generateMipmaps)
{
    createFromEncodedBytes(
        context, commandContext, encodedBytes, rgba8FormatForColorSpace(colorSpace), generateMipmaps);
    debugMetadata_.colorSpace = colorSpace;
}

void VulkanTexture::createFromEncodedBytes(VulkanContext& context,
                                           const VulkanCommandContext& commandContext,
                                           std::span<const uint8_t> encodedBytes,
                                           VkFormat format,
                                           bool generateMipmaps)
{
    const DecodedImage decoded = decodeImageBytes(encodedBytes);
    createFromRgba8(context, commandContext, decoded.width, decoded.height, decoded.pixels, format, generateMipmaps);
    debugMetadata_ = TextureDebugMetadata{
        "Embedded texture",
        {},
        colorSpaceForFormat(format),
        TextureDebugSource::GltfEmbeddedData,
        false,
    };
}

void VulkanTexture::createFromRgba8(VulkanContext& context,
                                    const VulkanCommandContext& commandContext,
                                    uint32_t width,
                                    uint32_t height,
                                    std::span<const uint8_t> pixels,
                                    VkFormat format,
                                    bool generateMipmaps,
                                    VulkanUploadBatch* batch)
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

    uploadPixels(context, commandContext, std::as_bytes(pixels), batch);
    createImageView();
    createSampler();
    debugMetadata_ = TextureDebugMetadata{
        "RGBA8 texture",
        {},
        colorSpaceForFormat(format_),
        TextureDebugSource::ProceduralFallback,
        false,
    };
}

void VulkanTexture::createFromKtx2(VulkanContext& context,
                                   const VulkanCommandContext& commandContext,
                                   const std::filesystem::path& path,
                                   assets::TextureUsage usage,
                                   VulkanUploadBatch* batch,
                                   std::span<const uint8_t> preloadedBytes)
{
    reset();

    // Reading off the device thread is worth ~a third of upload time on a real
    // scene, so the caller may have done it already.
    const std::vector<uint8_t> ownedBytes = preloadedBytes.empty() ? readFileBytes(path) : std::vector<uint8_t>{};
    const std::span<const uint8_t> fileBytes = preloadedBytes.empty()
                                                   ? std::span<const uint8_t>(ownedBytes)
                                                   : preloadedBytes;
    const assets::Ktx2Info info = assets::parseKtx2(fileBytes);

    // Re-checked here and not only at the call site: this is the last point
    // before the format reaches a descriptor, and a file cooked for a different
    // slot would otherwise be sampled with the wrong colour space -- a wrong
    // image that still renders, which is the worst kind.
    const assets::BlockCompressionCaps caps = queryBlockCompressionCaps(context.physicalDevice());
    if (!assets::cookedFormatUsable(caps, info.vkFormat, usage)) {
        throw std::runtime_error("Cooked texture '" + path.string() + "' holds vkFormat "
                                 + std::to_string(info.vkFormat) + ", which this device cannot sample for the "
                                 + std::string(assets::textureUsageName(usage)) + " slot.");
    }

    const std::vector<assets::Ktx2CopyRegion> regions = assets::ktx2CopyPlan(info);

    context_ = &context;
    width_ = info.pixelWidth;
    height_ = info.pixelHeight;
    format_ = static_cast<VkFormat>(info.vkFormat);
    // The baked chain is the chain. Nothing is generated, so this is also what
    // createSampler() derives maxLod from -- a file cooked with --no-mips gets a
    // sampler that stops at level 0 rather than sampling levels that do not exist.
    mipLevels_ = static_cast<uint32_t>(regions.size());

    // No TRANSFER_SRC: nothing blits out of this image. Dropping that usage is
    // what later makes a transfer-only queue legal for texture upload.
    const VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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

    uploadCookedLevels(context, commandContext, std::as_bytes(fileBytes), regions, batch);
    createImageView();
    createSampler();

    debugMetadata_ = TextureDebugMetadata{
        path.filename().string(),
        path.string(),
        assets::textureUsageIsSrgb(usage) ? TextureColorSpace::SRGB : TextureColorSpace::Linear,
        TextureDebugSource::LoadedFromDisk,
        false,
    };
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
    debugMetadata_ = {};
    loadStatsId_ = 0;
}

VkDeviceSize VulkanTexture::deviceSizeBytes() const
{
    if (!context_ || allocation_ == VK_NULL_HANDLE) {
        return 0;
    }

    VmaAllocationInfo allocationInfo{};
    vmaGetAllocationInfo(context_->allocator(), allocation_, &allocationInfo);
    return allocationInfo.size;
}

void VulkanTexture::setDebugMetadata(TextureDebugMetadata metadata)
{
    debugMetadata_ = std::move(metadata);
    renderer::AssetLoadStatsRecorder::amendTextureName(
        loadStatsId_, debugMetadata_.debugName, debugMetadata_.sourcePath);
}

void VulkanTexture::uploadPixels(VulkanContext& context,
                                 const VulkanCommandContext& commandContext,
                                 std::span<const std::byte> pixels,
                                 VulkanUploadBatch* batch)
{
    const auto stagingSize = static_cast<VkDeviceSize>(pixels.size_bytes());
    const bool batched = batch != nullptr && batch->recording();

    if (batched) {
        batch->flushIfOverBudget(stagingSize);
    }

    VulkanBuffer stagingBuffer;
    VulkanBufferCreateInfo stagingInfo{};
    stagingInfo.size = stagingSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    stagingInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    stagingBuffer.createBuffer(context, stagingInfo);
    stagingBuffer.upload(pixels);

    VkCommandBuffer commandBuffer = batched ? batch->commandBuffer() : beginOneTimeUpload(context, commandContext);

    // The upload starts from UNDEFINED because the texture image has no useful contents yet.
    // All mip levels become transfer destinations before level 0 is copied and later blits run.
    const VkImageMemoryBarrier2 toTransfer = textureBarrier(image_,
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
        commandBuffer, stagingBuffer.buffer(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    if (mipLevels_ > 1) {
        generateMipmaps(commandBuffer);
    } else {
        // Transfer writes must be visible to fragment shader texture sampling.
        // The descriptor will always refer to the final shader-read-only layout.
        const VkImageMemoryBarrier2 toShaderRead = textureBarrier(image_,
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

    if (batched) {
        batch->retainStaging(std::move(stagingBuffer), stagingSize);
    } else {
        endOneTimeUpload(context, commandContext, commandBuffer);
    }

    recordLoadStats();
}

void VulkanTexture::uploadCookedLevels(VulkanContext& context,
                                       const VulkanCommandContext& commandContext,
                                       std::span<const std::byte> fileBytes,
                                       const std::vector<assets::Ktx2CopyRegion>& regions,
                                       VulkanUploadBatch* batch)
{
    const auto stagingSize = static_cast<VkDeviceSize>(fileBytes.size_bytes());
    const bool batched = batch != nullptr && batch->recording();

    // Asked before the allocation exists, so the peak never counts both the
    // batch being flushed and the upload that forced the flush.
    if (batched) {
        batch->flushIfOverBudget(stagingSize);
    }

    // The whole file goes into staging, header and all. The few hundred bytes of
    // metadata are wasted, but it buys copying straight from the file's own
    // level offsets, which are already texel-block aligned.
    VulkanBuffer stagingBuffer;
    VulkanBufferCreateInfo stagingInfo{};
    stagingInfo.size = stagingSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    stagingInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    stagingBuffer.createBuffer(context, stagingInfo);
    stagingBuffer.upload(fileBytes);

    // A batch owns the command buffer and submits many textures at once; without
    // one this texture keeps its own submit-and-wait, which is what every caller
    // outside the glTF import loop still does.
    VkCommandBuffer commandBuffer = batched ? batch->commandBuffer() : beginOneTimeUpload(context, commandContext);

    const VkImageMemoryBarrier2 toTransfer = textureBarrier(image_,
                                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                            VK_PIPELINE_STAGE_2_NONE,
                                                            VK_ACCESS_2_NONE,
                                                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                            0,
                                                            mipLevels_);
    recordImageBarrier(commandBuffer, toTransfer);

    std::vector<VkBufferImageCopy> copyRegions;
    copyRegions.reserve(regions.size());
    for (const assets::Ktx2CopyRegion& region : regions) {
        VkBufferImageCopy copy{};
        copy.bufferOffset = static_cast<VkDeviceSize>(region.bufferOffset);
        // Zero means tightly packed, which cooked block rows are.
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = region.mipLevel;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;
        copy.imageOffset = {0, 0, 0};
        // The level's own extent, not its extent rounded up to a whole block. A
        // 2x2 or 1x1 tail level is legal precisely because it equals the mip
        // level's extent; rounding it to 4x4 would read past the level.
        copy.imageExtent = {region.width, region.height, 1};
        copyRegions.push_back(copy);
    }

    vkCmdCopyBufferToImage(commandBuffer,
                           stagingBuffer.buffer(),
                           image_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(copyRegions.size()),
                           copyRegions.data());

    // Every level was written by the same copy, so one barrier covers the whole
    // chain. The blit path needs a barrier per level only because each level is
    // read back as a blit source; nothing here reads the image at all.
    const VkImageMemoryBarrier2 toShaderRead = textureBarrier(image_,
                                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                              VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                              VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                                              0,
                                                              mipLevels_);
    recordImageBarrier(commandBuffer, toShaderRead);

    if (batched) {
        // The copy above reads this staging buffer on the GPU, so it has to
        // outlive the submit rather than this scope.
        batch->retainStaging(std::move(stagingBuffer), stagingSize);
    } else {
        endOneTimeUpload(context, commandContext, commandBuffer);
    }

    recordLoadStats();
}

void VulkanTexture::recordLoadStats()
{
    if (!renderer::AssetLoadStatsRecorder::enabled()) {
        return;
    }

    loadStatsId_ = renderer::AssetLoadStatsRecorder::nextTextureId();

    renderer::TextureLoadRecord record{};
    record.textureId = loadStatsId_;
    // The owner names the texture after construction; amendTextureName fills this
    // in then. Until it does, the placeholder keeps the report readable.
    record.debugName = debugMetadata_.debugName.empty() ? "unnamed#" + std::to_string(loadStatsId_)
                                                        : debugMetadata_.debugName;
    record.sourcePath = debugMetadata_.sourcePath;
    record.formatName = textureFormatName(format_);
    record.width = width_;
    record.height = height_;
    record.mipLevels = mipLevels_;
    record.deviceBytes = static_cast<uint64_t>(deviceSizeBytes());
    record.blockCompressed = isBlockCompressedFormat(format_);

    renderer::AssetLoadStatsRecorder::recordTexture(std::move(record));
}

void VulkanTexture::generateMipmaps(VkCommandBuffer commandBuffer)
{
    int32_t mipWidth = static_cast<int32_t>(width_);
    int32_t mipHeight = static_cast<int32_t>(height_);

    for (uint32_t mipLevel = 1; mipLevel < mipLevels_; ++mipLevel) {
        const VkImageMemoryBarrier2 previousToTransferSource = textureBarrier(image_,
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

        vkCmdBlitImage(commandBuffer,
                       image_,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image_,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &blit,
                       VK_FILTER_LINEAR);

        const VkImageMemoryBarrier2 previousToShaderRead = textureBarrier(image_,
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

    const VkImageMemoryBarrier2 finalMipToShaderRead = textureBarrier(image_,
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
    debugMetadata_ = std::exchange(other.debugMetadata_, {});
    loadStatsId_ = std::exchange(other.loadStatsId_, 0);
}

} // namespace ve::rhi
