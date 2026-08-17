#pragma once

#include "assets/Ktx2.h"
#include "assets/TextureCook.h"
#include "rhi/VulkanMemory.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ve::rhi {

class VulkanCommandContext;
class VulkanContext;
class VulkanUploadBatch;

// CPU-side decoded image (tightly packed RGBA8). Decoding does not touch Vulkan,
// so it can run on worker threads; the resulting pixels are uploaded later on the
// thread that owns the device via VulkanTexture::createFromRgba8.
struct DecodedImage {
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;

    [[nodiscard]] bool valid() const { return width > 0 && height > 0 && !pixels.empty(); }
};

enum class TextureColorSpace {
    Linear,
    SRGB
};

enum class TextureDebugSource {
    Unknown,
    LoadedFromDisk,
    GltfExternalFile,
    GltfEmbeddedData,
    ProceduralFallback
};

struct TextureDebugMetadata {
    std::string debugName;
    std::string sourcePath;
    TextureColorSpace colorSpace = TextureColorSpace::Linear;
    TextureDebugSource source = TextureDebugSource::Unknown;
    bool fallback = false;
};

[[nodiscard]] VkFormat rgba8FormatForColorSpace(TextureColorSpace colorSpace);

// Which block-compressed formats this device can actually sample with linear
// filtering. Queried from optimalTilingFeatures rather than inferred from the
// textureCompressionBC feature bit: that bit does not promise a filterable
// sampled image, and a texture that cannot be filtered is not usable here.
//
// Result is cached per physical device -- the caps cannot change under a running
// device, and the cook consults them once per texture load.
[[nodiscard]] assets::BlockCompressionCaps queryBlockCompressionCaps(VkPhysicalDevice physicalDevice);

// True when `source` has a cooked sidecar for `usage` that this device can use:
// the file exists, is not older than its source, and holds the format this slot
// expects. Everything else -- a missing cook, a stale cook, an unsupported
// format -- is a normal miss, not an error.
[[nodiscard]] bool cookedTextureAvailable(VkPhysicalDevice physicalDevice,
                                          const std::filesystem::path& source,
                                          assets::TextureUsage usage);

class VulkanTexture final {
public:
    VulkanTexture() = default;
    ~VulkanTexture();

    // CPU-only image decode (stb_image). Thread-safe across independent calls, so
    // these may be dispatched to a JobSystem. They throw std::runtime_error on
    // failure. Upload the result with createFromRgba8 on the device thread.
    [[nodiscard]] static DecodedImage decodeImageFile(const std::filesystem::path& path);
    [[nodiscard]] static DecodedImage decodeImageBytes(std::span<const uint8_t> encodedBytes);

    VulkanTexture(const VulkanTexture&) = delete;
    VulkanTexture& operator=(const VulkanTexture&) = delete;
    VulkanTexture(VulkanTexture&& other) noexcept;
    VulkanTexture& operator=(VulkanTexture&& other) noexcept;

    void createCheckerboard(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t width = 256,
        uint32_t height = 256,
        TextureColorSpace colorSpace = TextureColorSpace::SRGB);
    void createFromFile(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        const std::filesystem::path& path,
        TextureColorSpace colorSpace = TextureColorSpace::Linear,
        bool generateMipmaps = true);
    void createFromFile(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        const std::filesystem::path& path,
        VkFormat format,
        bool generateMipmaps = true);
    void createFromEncodedBytes(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const uint8_t> encodedBytes,
        TextureColorSpace colorSpace = TextureColorSpace::Linear,
        bool generateMipmaps = true);
    void createFromEncodedBytes(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const uint8_t> encodedBytes,
        VkFormat format,
        bool generateMipmaps = true);
    void createFromRgba8(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        uint32_t width,
        uint32_t height,
        std::span<const uint8_t> pixels,
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
        bool generateMipmaps = true,
        VulkanUploadBatch* batch = nullptr);

    // Uploads a cooked block-compressed KTX2 (docs/asset_system.md). The file's
    // own format and level count are used as-is, and the mip chain is copied
    // rather than generated -- there is no vkCmdBlitImage on this path, which is
    // what later makes a transfer-only queue legal. `usage` is the material slot
    // the texture is being loaded for; a file cooked for a different slot is
    // rejected rather than sampled with the wrong colour space.
    //
    // Throws std::runtime_error on a missing, malformed, or unusable file, so the
    // caller can fall back to the uncompressed source image.
    // `preloadedBytes`, when non-empty, is the file's contents already read --
    // the reads are worth doing off the device thread, and `path` is then used
    // only for diagnostics.
    void createFromKtx2(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        const std::filesystem::path& path,
        assets::TextureUsage usage,
        VulkanUploadBatch* batch = nullptr,
        std::span<const uint8_t> preloadedBytes = {});

    void reset();
    void destroy() { reset(); }

    [[nodiscard]] VkImage image() const { return image_; }
    [[nodiscard]] VkImageView imageView() const { return imageView_; }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] uint32_t width() const { return width_; }
    [[nodiscard]] uint32_t height() const { return height_; }
    [[nodiscard]] uint32_t mipLevels() const { return mipLevels_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] const TextureDebugMetadata& debugMetadata() const { return debugMetadata_; }
    [[nodiscard]] bool valid() const { return image_ != VK_NULL_HANDLE; }

    // Size of the VMA allocation backing the whole mip chain, or 0 when there is
    // no allocation. Queried rather than computed so block-compressed formats
    // and driver padding are reflected honestly.
    [[nodiscard]] VkDeviceSize deviceSizeBytes() const;

    // Also amends this texture's asset-load record, which is created during
    // upload -- before the owner has a name for it.
    void setDebugMetadata(TextureDebugMetadata metadata);

private:
    void uploadPixels(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const std::byte> pixels,
        VulkanUploadBatch* batch);
    // Copies one region per mip level out of a staging buffer holding the whole
    // cooked file, then transitions every level to SHADER_READ_ONLY in one
    // barrier. Two barriers total, against 2 + 2*(N-1) for the blit path.
    void uploadCookedLevels(
        VulkanContext& context,
        const VulkanCommandContext& commandContext,
        std::span<const std::byte> fileBytes,
        const std::vector<assets::Ktx2CopyRegion>& regions,
        VulkanUploadBatch* batch);
    void generateMipmaps(VkCommandBuffer commandBuffer);
    // Load-time instrumentation only; a no-op unless recording is enabled.
    void recordLoadStats();
    void createImageView();
    void createSampler();
    void moveFrom(VulkanTexture& other) noexcept;

    VulkanContext* context_ = nullptr;

    // The image and its VMA allocation own GPU-local texture storage.
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;

    // The view and sampler are what the combined image sampler descriptor exposes.
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t mipLevels_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    TextureDebugMetadata debugMetadata_{};

    // Load-time instrumentation only (see renderer/AssetLoadStats.h). Zero means
    // this texture was never recorded.
    uint64_t loadStatsId_ = 0;
};

} // namespace ve::rhi
