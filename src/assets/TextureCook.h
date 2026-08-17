#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

// Policy and CPU-side work for the offline texture cook (docs/asset_system.md):
// which block format a texture should be encoded to, how its mip chain is built,
// and how a texel block is extracted from an image whose extent is not a multiple
// of four.
//
// GPU-free on purpose. The device capability query lives at the Vulkan edge and
// arrives here as a plain struct of booleans, so the policy itself is a pure
// function the headless tests can pin down. Formats are raw VkFormat numbers for
// the same reason KTX2 stores them that way; see assets/Ktx2.h.
namespace ve::assets {

inline constexpr uint32_t kVkFormatR8G8B8A8Unorm = 37;
inline constexpr uint32_t kVkFormatR8G8B8A8Srgb = 43;

// What a texture is for, which is what decides its colour space and channel
// count. This is the material slot, not the file: the same PNG cooked as a base
// colour and as a roughness map gets different formats.
enum class TextureUsage {
    BaseColor,
    Emissive,
    NormalMap,
    MetallicRoughness,
    Occlusion
};

// Filled from vkGetPhysicalDeviceFormatProperties at the Vulkan edge: the caller
// must check for sampling *and* linear filtering, not just the
// textureCompressionBC feature bit, because the feature bit alone does not
// promise a filterable sampled image.
struct BlockCompressionCaps {
    bool bc7 = false;
    bool bc5 = false;
};

[[nodiscard]] bool textureUsageIsSrgb(TextureUsage usage);

// The stable spelling of a usage, shared by `vecook --usage`, the cook driver
// script, and the sidecar file name. One spelling so a renamed slot cannot make
// the tool and the runtime disagree about which file belongs to which slot.
[[nodiscard]] std::string_view textureUsageName(TextureUsage usage);
[[nodiscard]] bool parseTextureUsage(std::string_view name, TextureUsage& usage);

// Where the cook writes, and where the runtime looks: `foo.png` used as a base
// colour becomes `foo.base-color.ktx2` beside the source.
//
// The slot is in the name rather than only inside the file because one image can
// legitimately be used in two slots and must then be cooked twice, to different
// formats and different mip filtering. A single `foo.ktx2` would silently serve
// whichever slot cooked it last.
[[nodiscard]] std::filesystem::path ktx2SidecarPath(const std::filesystem::path& source, TextureUsage usage);

// The format the cook writes, ignoring device support -- the cook runs offline and
// has no device. A file cooked for a device that cannot sample it is still a
// correct file; the runtime is what falls back.
[[nodiscard]] uint32_t cookedFormatForUsage(TextureUsage usage);

// What the runtime should sample. Returns the cooked block format when this
// device can sample it, and the uncompressed RGBA8 format otherwise, in which
// case the caller loads the original image instead of the cooked file.
[[nodiscard]] uint32_t chooseTextureFormat(BlockCompressionCaps caps, TextureUsage usage);

// Whether a cooked file is usable as-is: its stored format must be the one this
// device would have chosen. A cooked BC7 file on a device without BC7 is not an
// error, just unusable.
[[nodiscard]] bool cookedFormatUsable(BlockCompressionCaps caps, uint32_t cookedVkFormat, TextureUsage usage);

// Box-filtered mip chain down to 1x1, level 0 first, tightly packed RGBA8.
// sRGB sources are averaged in linear light: filtering sRGB code values directly
// darkens every mip, and the error compounds down the chain. Alpha is always
// linear. Odd extents halve with floor and clamp at 1, matching the extents the
// KTX2 level index and Vulkan both assume.
[[nodiscard]] std::vector<std::vector<uint8_t>> generateMipChainRgba8(std::span<const uint8_t> basePixels,
                                                                      uint32_t width,
                                                                      uint32_t height,
                                                                      bool srgb);

// Copies one 4x4 RGBA8 texel block out of an image into `block` (64 bytes, row
// major). Extents are not required to be multiples of four: a trailing partial
// block repeats its last real row and column rather than encoding whatever was
// past the edge of the buffer. Clamping rather than zero-filling keeps the
// padding from bleeding a dark edge into the block's endpoints.
void extractRgba8Block(std::span<const uint8_t> pixels,
                       uint32_t width,
                       uint32_t height,
                       uint32_t blockX,
                       uint32_t blockY,
                       std::span<uint8_t, 64> block);

// How many 4x4 blocks cover an extent, rounding a partial block up.
[[nodiscard]] uint32_t rgba8BlockColumnCount(uint32_t width);
[[nodiscard]] uint32_t rgba8BlockRowCount(uint32_t height);

// Receives one padded 4x4 RGBA8 block and writes its compressed bytes. Called
// from several threads at once, so an encoder with global setup must be
// initialized before the first call.
using BlockEncodeFn = std::function<void(std::span<const uint8_t, 64> rgbaBlock, std::span<uint8_t> blockOut)>;

// Encodes block rows [beginBlockRow, endBlockRow) of one mip level into
// `levelOut`, which must be sized for the whole level. Blocks are walked in the
// row-major order every BC format and the KTX2 level layout assume; getting that
// order wrong scrambles a texture without tripping any validation.
//
// Taking a row range rather than the whole level is what lets the caller spread
// one level across a thread pool without the encoder knowing about threads:
// distinct row ranges write to disjoint bytes of `levelOut`.
void encodeRgba8BlockRows(std::span<const uint8_t> pixels,
                          uint32_t width,
                          uint32_t height,
                          uint32_t blockSizeBytes,
                          uint32_t beginBlockRow,
                          uint32_t endBlockRow,
                          const BlockEncodeFn& encodeBlock,
                          std::span<uint8_t> levelOut);

} // namespace ve::assets
