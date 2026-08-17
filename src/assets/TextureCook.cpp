#include "assets/TextureCook.h"

#include "assets/Ktx2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ve::assets {

namespace {

constexpr uint32_t kRgbaChannels = 4;
constexpr uint32_t kBlockExtent = 4;

// sRGB transfer function, both directions, on 0-255 code values. The forward
// table is built once because mip generation calls it per channel per texel.
[[nodiscard]] float srgbToLinear(float value)
{
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

[[nodiscard]] float linearToSrgb(float value)
{
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

[[nodiscard]] const std::array<float, 256>& srgbToLinearTable()
{
    static const std::array<float, 256> table = [] {
        std::array<float, 256> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = srgbToLinear(static_cast<float>(i) / 255.0f);
        }
        return values;
    }();

    return table;
}

[[nodiscard]] uint8_t encodeChannel(float linearAverage, bool srgb)
{
    const float encoded = srgb ? linearToSrgb(linearAverage) : linearAverage;
    const float scaled = std::round(std::clamp(encoded, 0.0f, 1.0f) * 255.0f);
    return static_cast<uint8_t>(scaled);
}

[[nodiscard]] uint32_t halveExtent(uint32_t extent)
{
    return std::max(1U, extent / 2U);
}

} // namespace

bool textureUsageIsSrgb(TextureUsage usage)
{
    switch (usage) {
    case TextureUsage::BaseColor:
    case TextureUsage::Emissive:
        return true;
    case TextureUsage::NormalMap:
    case TextureUsage::MetallicRoughness:
    case TextureUsage::Occlusion:
        return false;
    }

    return false;
}

uint32_t cookedFormatForUsage(TextureUsage usage)
{
    switch (usage) {
    case TextureUsage::BaseColor:
    case TextureUsage::Emissive:
        return kVkFormatBc7SrgbBlock;
    // BC5 is two independent 8-bit channels at the same 16 bytes per block BC7
    // costs, and a tangent-space normal only needs XY -- Z is reconstructed in the
    // shader. Spending BC7 on a discarded blue channel would buy nothing.
    case TextureUsage::NormalMap:
        return kVkFormatBc5UnormBlock;
    case TextureUsage::MetallicRoughness:
    case TextureUsage::Occlusion:
        return kVkFormatBc7UnormBlock;
    }

    return kVkFormatBc7UnormBlock;
}

uint32_t chooseTextureFormat(BlockCompressionCaps caps, TextureUsage usage)
{
    const uint32_t cooked = cookedFormatForUsage(usage);
    const bool supported = cooked == kVkFormatBc5UnormBlock ? caps.bc5 : caps.bc7;
    if (supported) {
        return cooked;
    }

    return textureUsageIsSrgb(usage) ? kVkFormatR8G8B8A8Srgb : kVkFormatR8G8B8A8Unorm;
}

bool cookedFormatUsable(BlockCompressionCaps caps, uint32_t cookedVkFormat, TextureUsage usage)
{
    return cookedVkFormat == cookedFormatForUsage(usage) && chooseTextureFormat(caps, usage) == cookedVkFormat;
}

std::vector<std::vector<uint8_t>> generateMipChainRgba8(std::span<const uint8_t> basePixels,
                                                        uint32_t width,
                                                        uint32_t height,
                                                        bool srgb)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("Texture cook: cannot build a mip chain for a zero-sized image.");
    }

    const size_t baseBytes = static_cast<size_t>(width) * height * kRgbaChannels;
    if (basePixels.size() < baseBytes) {
        throw std::runtime_error("Texture cook: base image is smaller than its declared extent.");
    }

    const std::array<float, 256>& toLinear = srgbToLinearTable();

    std::vector<std::vector<uint8_t>> levels;
    levels.reserve(ktx2FullMipChainLength(width, height));
    levels.emplace_back(basePixels.begin(), basePixels.begin() + static_cast<std::ptrdiff_t>(baseBytes));

    uint32_t sourceWidth = width;
    uint32_t sourceHeight = height;
    while (sourceWidth > 1U || sourceHeight > 1U) {
        const uint32_t targetWidth = halveExtent(sourceWidth);
        const uint32_t targetHeight = halveExtent(sourceHeight);
        const std::vector<uint8_t>& source = levels.back();

        std::vector<uint8_t> target(static_cast<size_t>(targetWidth) * targetHeight * kRgbaChannels);
        for (uint32_t y = 0; y < targetHeight; ++y) {
            // A dimension that has already collapsed to 1 contributes one row or
            // column twice rather than reading past the edge.
            const uint32_t y0 = std::min(y * 2U, sourceHeight - 1U);
            const uint32_t y1 = std::min(y * 2U + 1U, sourceHeight - 1U);
            for (uint32_t x = 0; x < targetWidth; ++x) {
                const uint32_t x0 = std::min(x * 2U, sourceWidth - 1U);
                const uint32_t x1 = std::min(x * 2U + 1U, sourceWidth - 1U);

                const std::array<size_t, 4> taps = {
                    (static_cast<size_t>(y0) * sourceWidth + x0) * kRgbaChannels,
                    (static_cast<size_t>(y0) * sourceWidth + x1) * kRgbaChannels,
                    (static_cast<size_t>(y1) * sourceWidth + x0) * kRgbaChannels,
                    (static_cast<size_t>(y1) * sourceWidth + x1) * kRgbaChannels,
                };

                uint8_t* destination = target.data() + (static_cast<size_t>(y) * targetWidth + x) * kRgbaChannels;
                for (uint32_t channel = 0; channel < kRgbaChannels; ++channel) {
                    // Alpha is linear in an sRGB image, so only RGB decode.
                    const bool encoded = srgb && channel < 3U;
                    float sum = 0.0f;
                    for (size_t tap : taps) {
                        const uint8_t value = source[tap + channel];
                        sum += encoded ? toLinear[value] : static_cast<float>(value) / 255.0f;
                    }

                    destination[channel] = encodeChannel(sum * 0.25f, encoded);
                }
            }
        }

        levels.push_back(std::move(target));
        sourceWidth = targetWidth;
        sourceHeight = targetHeight;
    }

    return levels;
}

void extractRgba8Block(std::span<const uint8_t> pixels,
                       uint32_t width,
                       uint32_t height,
                       uint32_t blockX,
                       uint32_t blockY,
                       std::span<uint8_t, 64> block)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("Texture cook: cannot extract a block from a zero-sized image.");
    }
    if (pixels.size() < static_cast<size_t>(width) * height * kRgbaChannels) {
        throw std::runtime_error("Texture cook: image is smaller than its declared extent.");
    }

    for (uint32_t row = 0; row < kBlockExtent; ++row) {
        const uint32_t y = std::min(blockY * kBlockExtent + row, height - 1U);
        for (uint32_t column = 0; column < kBlockExtent; ++column) {
            const uint32_t x = std::min(blockX * kBlockExtent + column, width - 1U);

            const size_t source = (static_cast<size_t>(y) * width + x) * kRgbaChannels;
            const size_t destination = (static_cast<size_t>(row) * kBlockExtent + column) * kRgbaChannels;
            for (uint32_t channel = 0; channel < kRgbaChannels; ++channel) {
                block[destination + channel] = pixels[source + channel];
            }
        }
    }
}

uint32_t rgba8BlockColumnCount(uint32_t width)
{
    return (width + kBlockExtent - 1U) / kBlockExtent;
}

uint32_t rgba8BlockRowCount(uint32_t height)
{
    return (height + kBlockExtent - 1U) / kBlockExtent;
}

void encodeRgba8BlockRows(std::span<const uint8_t> pixels,
                          uint32_t width,
                          uint32_t height,
                          uint32_t blockSizeBytes,
                          uint32_t beginBlockRow,
                          uint32_t endBlockRow,
                          const BlockEncodeFn& encodeBlock,
                          std::span<uint8_t> levelOut)
{
    if (blockSizeBytes == 0) {
        throw std::runtime_error("Texture cook: cannot encode blocks of zero bytes.");
    }

    const uint32_t blockColumns = rgba8BlockColumnCount(width);
    const uint32_t blockRows = rgba8BlockRowCount(height);
    if (endBlockRow > blockRows || beginBlockRow > endBlockRow) {
        throw std::runtime_error("Texture cook: block row range is outside the level.");
    }
    if (levelOut.size() != static_cast<size_t>(blockColumns) * blockRows * blockSizeBytes) {
        throw std::runtime_error("Texture cook: output buffer does not match the level's block count.");
    }

    std::array<uint8_t, 64> block{};
    for (uint32_t blockY = beginBlockRow; blockY < endBlockRow; ++blockY) {
        for (uint32_t blockX = 0; blockX < blockColumns; ++blockX) {
            extractRgba8Block(pixels, width, height, blockX, blockY, block);

            const size_t offset = (static_cast<size_t>(blockY) * blockColumns + blockX) * blockSizeBytes;
            encodeBlock(block, levelOut.subspan(offset, blockSizeBytes));
        }
    }
}

} // namespace ve::assets
