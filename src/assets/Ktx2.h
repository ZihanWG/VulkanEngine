#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

// KTX 2.0 container reader/writer for the offline texture cook
// (docs/asset_system.md). Hand-written rather than libktx: the container is a
// header, a level index, and a data-format descriptor, and libktx would drag in
// basisu and astcenc to reach BC7 through a UASTC transcode we do not want.
// `core/PngWriter.h` is the precedent for hand-rolling a container here.
//
// GPU-free by construction so it can live in VulkanEngineCore and be unit tested
// without a device. Formats are carried as raw VkFormat *numbers* -- KTX2 stores
// exactly that -- so this file never includes a Vulkan header. The Vulkan-facing
// side static_asserts these constants against the real enumerators.
namespace ve::assets {

// The subset of VkFormat the cook emits. BC7 covers colour (sRGB) and the
// data maps (UNORM); BC5 is the two-channel normal-map format.
inline constexpr uint32_t kVkFormatBc5UnormBlock = 141;
inline constexpr uint32_t kVkFormatBc7UnormBlock = 145;
inline constexpr uint32_t kVkFormatBc7SrgbBlock = 146;

struct Ktx2FormatInfo {
    uint32_t blockWidth = 0;
    uint32_t blockHeight = 0;
    uint32_t blockSizeBytes = 0;
};

// Nullopt-equivalent is a zeroed struct: an unsupported format has no block size.
[[nodiscard]] Ktx2FormatInfo ktx2FormatInfo(uint32_t vkFormat);
[[nodiscard]] bool ktx2FormatSupported(uint32_t vkFormat);

// Tightly packed block bytes for one mip level, as the encoder produced them.
[[nodiscard]] size_t ktx2LevelSizeBytes(uint32_t vkFormat, uint32_t baseWidth, uint32_t baseHeight, uint32_t level);

// Number of mip levels in a full chain down to 1x1. Independent of the block
// size: BC images keep going below one block, the last levels are just padded.
[[nodiscard]] uint32_t ktx2FullMipChainLength(uint32_t baseWidth, uint32_t baseHeight);

// A complete image ready to serialize. `levels[0]` is the base (largest) level;
// the writer is what reverses them into the file's smallest-first data order.
struct Ktx2Image {
    uint32_t vkFormat = 0;
    uint32_t pixelWidth = 0;
    uint32_t pixelHeight = 0;
    std::vector<std::vector<uint8_t>> levels;
};

// Where one mip level's bytes sit inside a serialized KTX2 blob. Index 0 is the
// base level, matching Ktx2Image, so the upload path can walk levels in Vulkan's
// mip order while the bytes stay in the file's order.
struct Ktx2LevelView {
    uint64_t byteOffset = 0;
    uint64_t byteLength = 0;
};

struct Ktx2Info {
    uint32_t vkFormat = 0;
    uint32_t pixelWidth = 0;
    uint32_t pixelHeight = 0;
    std::vector<Ktx2LevelView> levels;
};

// Both throw std::runtime_error rather than returning a status: a malformed KTX2
// is a cook bug or a corrupt file, never something a caller recovers from.
[[nodiscard]] std::vector<uint8_t> writeKtx2(const Ktx2Image& image);
void writeKtx2File(const std::filesystem::path& path, const Ktx2Image& image);

// Views point into `bytes`; the caller keeps that buffer alive.
[[nodiscard]] Ktx2Info parseKtx2(std::span<const uint8_t> bytes);

} // namespace ve::assets
