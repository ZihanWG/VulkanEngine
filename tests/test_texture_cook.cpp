#include "assets/Ktx2.h"
#include "assets/TextureCook.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

using ve::assets::BlockCompressionCaps;
using ve::assets::TextureUsage;
using ve::assets::chooseTextureFormat;
using ve::assets::cookedFormatForUsage;
using ve::assets::cookedFormatUsable;
using ve::assets::extractRgba8Block;
using ve::assets::generateMipChainRgba8;
using ve::assets::kVkFormatBc5UnormBlock;
using ve::assets::kVkFormatBc7SrgbBlock;
using ve::assets::kVkFormatBc7UnormBlock;
using ve::assets::kVkFormatR8G8B8A8Srgb;
using ve::assets::kVkFormatR8G8B8A8Unorm;
using ve::assets::textureUsageIsSrgb;

namespace {

constexpr BlockCompressionCaps kFullCaps{true, true};
constexpr BlockCompressionCaps kNoCaps{false, false};

std::vector<uint8_t> solidImage(uint32_t width, uint32_t height, std::array<uint8_t, 4> rgba)
{
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4U);
    for (size_t texel = 0; texel < pixels.size() / 4U; ++texel) {
        for (size_t channel = 0; channel < 4U; ++channel) {
            pixels[texel * 4U + channel] = rgba[channel];
        }
    }

    return pixels;
}

} // namespace

TEST_CASE("Cooked format follows the material slot, not the file", "[texture-cook]")
{
    CHECK(cookedFormatForUsage(TextureUsage::BaseColor) == kVkFormatBc7SrgbBlock);
    CHECK(cookedFormatForUsage(TextureUsage::Emissive) == kVkFormatBc7SrgbBlock);
    // A normal map is two channels, so BC5 spends the same 16 bytes per block on
    // XY rather than on a blue channel the shader reconstructs.
    CHECK(cookedFormatForUsage(TextureUsage::NormalMap) == kVkFormatBc5UnormBlock);
    CHECK(cookedFormatForUsage(TextureUsage::MetallicRoughness) == kVkFormatBc7UnormBlock);
    CHECK(cookedFormatForUsage(TextureUsage::Occlusion) == kVkFormatBc7UnormBlock);

    CHECK(textureUsageIsSrgb(TextureUsage::BaseColor));
    CHECK(textureUsageIsSrgb(TextureUsage::Emissive));
    CHECK_FALSE(textureUsageIsSrgb(TextureUsage::NormalMap));
    CHECK_FALSE(textureUsageIsSrgb(TextureUsage::MetallicRoughness));
    CHECK_FALSE(textureUsageIsSrgb(TextureUsage::Occlusion));

    // Every cooked format must be one the container can actually describe.
    for (TextureUsage usage : {TextureUsage::BaseColor, TextureUsage::Emissive, TextureUsage::NormalMap,
                               TextureUsage::MetallicRoughness, TextureUsage::Occlusion}) {
        CHECK(ve::assets::ktx2FormatSupported(cookedFormatForUsage(usage)));
    }
}

TEST_CASE("A device without block compression falls back to RGBA8", "[texture-cook]")
{
    CHECK(chooseTextureFormat(kFullCaps, TextureUsage::BaseColor) == kVkFormatBc7SrgbBlock);
    CHECK(chooseTextureFormat(kFullCaps, TextureUsage::NormalMap) == kVkFormatBc5UnormBlock);

    // The fallback keeps the colour space: dropping compression must not also
    // drop the sRGB decode, which would wash out every base colour in the scene.
    CHECK(chooseTextureFormat(kNoCaps, TextureUsage::BaseColor) == kVkFormatR8G8B8A8Srgb);
    CHECK(chooseTextureFormat(kNoCaps, TextureUsage::Emissive) == kVkFormatR8G8B8A8Srgb);
    CHECK(chooseTextureFormat(kNoCaps, TextureUsage::NormalMap) == kVkFormatR8G8B8A8Unorm);
    CHECK(chooseTextureFormat(kNoCaps, TextureUsage::MetallicRoughness) == kVkFormatR8G8B8A8Unorm);

    // The two capabilities are reported separately, so they must be consulted
    // separately: BC7 without BC5 still compresses colour.
    const BlockCompressionCaps colorOnly{true, false};
    CHECK(chooseTextureFormat(colorOnly, TextureUsage::BaseColor) == kVkFormatBc7SrgbBlock);
    CHECK(chooseTextureFormat(colorOnly, TextureUsage::NormalMap) == kVkFormatR8G8B8A8Unorm);
}

TEST_CASE("A cooked file is only used when it matches this device and slot", "[texture-cook]")
{
    CHECK(cookedFormatUsable(kFullCaps, kVkFormatBc7SrgbBlock, TextureUsage::BaseColor));
    CHECK(cookedFormatUsable(kFullCaps, kVkFormatBc5UnormBlock, TextureUsage::NormalMap));

    // A file cooked for a slot it is not being used in: an sRGB decode applied to
    // roughness is the silent-corruption case, so it is rejected rather than
    // sampled.
    CHECK_FALSE(cookedFormatUsable(kFullCaps, kVkFormatBc7SrgbBlock, TextureUsage::MetallicRoughness));
    CHECK_FALSE(cookedFormatUsable(kFullCaps, kVkFormatBc7UnormBlock, TextureUsage::BaseColor));

    // A perfectly valid BC7 file on a device that cannot sample BC7.
    CHECK_FALSE(cookedFormatUsable(kNoCaps, kVkFormatBc7SrgbBlock, TextureUsage::BaseColor));
    CHECK_FALSE(cookedFormatUsable(BlockCompressionCaps{true, false}, kVkFormatBc5UnormBlock,
                                   TextureUsage::NormalMap));
}

TEST_CASE("The mip chain runs down to 1x1 with the extents KTX2 expects", "[texture-cook]")
{
    const std::vector<uint8_t> base = solidImage(8, 4, {10, 20, 30, 40});
    const std::vector<std::vector<uint8_t>> levels = generateMipChainRgba8(base, 8, 4, false);

    REQUIRE(levels.size() == ve::assets::ktx2FullMipChainLength(8, 4));
    // 8x4 -> 4x2 -> 2x1 -> 1x1: the short edge clamps while the long one halves.
    const std::array<size_t, 4> expectedTexels = {8U * 4U, 4U * 2U, 2U * 1U, 1U * 1U};
    for (size_t level = 0; level < levels.size(); ++level) {
        CHECK(levels[level].size() == expectedTexels[level] * 4U);
    }

    CHECK(levels[0] == base);
}

TEST_CASE("A flat image survives the whole mip chain unchanged", "[texture-cook]")
{
    // The strongest available check on the filter without a reference encoder: a
    // constant image must average to itself at every level, in both colour
    // spaces, or the transfer functions are not inverses.
    for (bool srgb : {false, true}) {
        const std::array<uint8_t, 4> color = {188, 77, 5, 200};
        const std::vector<uint8_t> base = solidImage(16, 16, color);
        const std::vector<std::vector<uint8_t>> levels = generateMipChainRgba8(base, 16, 16, srgb);

        REQUIRE(levels.size() == 5);
        for (const std::vector<uint8_t>& level : levels) {
            for (size_t texel = 0; texel < level.size() / 4U; ++texel) {
                for (size_t channel = 0; channel < 4U; ++channel) {
                    CHECK(static_cast<int>(level[texel * 4U + channel]) == static_cast<int>(color[channel]));
                }
            }
        }
    }
}

TEST_CASE("sRGB mips average in linear light", "[texture-cook]")
{
    // One black texel and three white ones. Averaging the sRGB code values gives
    // 191; decoding, averaging in linear light and re-encoding gives 225. The
    // second is the correct one, and that gap is the whole reason the flag exists
    // -- filtering code values directly darkens every level of the chain.
    std::vector<uint8_t> base(2U * 2U * 4U, 255);
    for (size_t channel = 0; channel < 3U; ++channel) {
        base[channel] = 0;
    }

    const std::vector<std::vector<uint8_t>> linearChain = generateMipChainRgba8(base, 2, 2, false);
    REQUIRE(linearChain.size() == 2);
    CHECK(static_cast<int>(linearChain[1][0]) == 191);

    const std::vector<std::vector<uint8_t>> srgbChain = generateMipChainRgba8(base, 2, 2, true);
    REQUIRE(srgbChain.size() == 2);
    CHECK(static_cast<int>(srgbChain[1][0]) == 225);

    // Alpha is linear even in an sRGB image, so it must not move.
    CHECK(static_cast<int>(srgbChain[1][3]) == 255);
}

TEST_CASE("Mip generation rejects inputs it cannot filter", "[texture-cook]")
{
    const std::vector<uint8_t> base = solidImage(4, 4, {1, 2, 3, 4});
    CHECK_THROWS_AS(generateMipChainRgba8(base, 0, 4, false), std::runtime_error);
    // A buffer shorter than its declared extent would otherwise be read past.
    CHECK_THROWS_AS(generateMipChainRgba8(base, 8, 8, false), std::runtime_error);
}

TEST_CASE("A partial texel block clamps instead of reading past the edge", "[texture-cook]")
{
    // 5x5 is the awkward case: BC needs whole 4x4 blocks, so the cook covers a
    // 5-texel extent with two blocks and the second is 1 texel wide.
    constexpr uint32_t kWidth = 5;
    constexpr uint32_t kHeight = 5;
    std::vector<uint8_t> pixels(static_cast<size_t>(kWidth) * kHeight * 4U);
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const size_t texel = (static_cast<size_t>(y) * kWidth + x) * 4U;
            pixels[texel + 0] = static_cast<uint8_t>(x);
            pixels[texel + 1] = static_cast<uint8_t>(y);
            pixels[texel + 2] = 0;
            pixels[texel + 3] = 255;
        }
    }

    std::array<uint8_t, 64> block{};
    extractRgba8Block(pixels, kWidth, kHeight, 0, 0, block);
    for (uint32_t row = 0; row < 4U; ++row) {
        for (uint32_t column = 0; column < 4U; ++column) {
            const size_t texel = (static_cast<size_t>(row) * 4U + column) * 4U;
            CHECK(static_cast<int>(block[texel + 0]) == static_cast<int>(column));
            CHECK(static_cast<int>(block[texel + 1]) == static_cast<int>(row));
        }
    }

    // The trailing block starts at x=4 and only column 0 exists; the other three
    // repeat it rather than carrying whatever was in memory.
    extractRgba8Block(pixels, kWidth, kHeight, 1, 0, block);
    for (uint32_t row = 0; row < 4U; ++row) {
        for (uint32_t column = 0; column < 4U; ++column) {
            const size_t texel = (static_cast<size_t>(row) * 4U + column) * 4U;
            CHECK(static_cast<int>(block[texel + 0]) == 4);
            CHECK(static_cast<int>(block[texel + 1]) == static_cast<int>(row));
            CHECK(static_cast<int>(block[texel + 3]) == 255);
        }
    }

    // The bottom-right corner block has exactly one real texel.
    extractRgba8Block(pixels, kWidth, kHeight, 1, 1, block);
    for (size_t texel = 0; texel < 16U; ++texel) {
        CHECK(static_cast<int>(block[texel * 4U + 0]) == 4);
        CHECK(static_cast<int>(block[texel * 4U + 1]) == 4);
    }
}

TEST_CASE("A level is encoded in row-major block order", "[texture-cook]")
{
    // Block order is invisible to every validation layer: getting it wrong just
    // scrambles the texture. A fake encoder that stamps each block with its
    // top-left texel is enough to pin the order and the output offsets down.
    constexpr uint32_t kWidth = 9; // 3 block columns, the last one partial
    constexpr uint32_t kHeight = 5; // 2 block rows, the last one partial
    constexpr uint32_t kBlockSize = 16;

    std::vector<uint8_t> pixels(static_cast<size_t>(kWidth) * kHeight * 4U);
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const size_t texel = (static_cast<size_t>(y) * kWidth + x) * 4U;
            pixels[texel + 0] = static_cast<uint8_t>(x);
            pixels[texel + 1] = static_cast<uint8_t>(y);
        }
    }

    const auto stampTopLeft = [](std::span<const uint8_t, 64> block, std::span<uint8_t> out) {
        out[0] = block[0];
        out[1] = block[1];
        for (size_t byte = 2; byte < out.size(); ++byte) {
            out[byte] = 0xEE;
        }
    };

    const uint32_t blockColumns = ve::assets::rgba8BlockColumnCount(kWidth);
    const uint32_t blockRows = ve::assets::rgba8BlockRowCount(kHeight);
    REQUIRE(blockColumns == 3);
    REQUIRE(blockRows == 2);

    std::vector<uint8_t> level(static_cast<size_t>(blockColumns) * blockRows * kBlockSize);
    ve::assets::encodeRgba8BlockRows(pixels, kWidth, kHeight, kBlockSize, 0, blockRows, stampTopLeft, level);

    for (uint32_t blockY = 0; blockY < blockRows; ++blockY) {
        for (uint32_t blockX = 0; blockX < blockColumns; ++blockX) {
            const size_t offset = (static_cast<size_t>(blockY) * blockColumns + blockX) * kBlockSize;
            CHECK(static_cast<int>(level[offset + 0]) == static_cast<int>(blockX * 4U));
            CHECK(static_cast<int>(level[offset + 1]) == static_cast<int>(blockY * 4U));
        }
    }

    // Encoding row ranges separately -- which is how the cook spreads a level
    // across the thread pool -- must land the same bytes as one pass.
    std::vector<uint8_t> split(level.size());
    ve::assets::encodeRgba8BlockRows(pixels, kWidth, kHeight, kBlockSize, 0, 1, stampTopLeft, split);
    ve::assets::encodeRgba8BlockRows(pixels, kWidth, kHeight, kBlockSize, 1, 2, stampTopLeft, split);
    CHECK(split == level);

    // An empty range is legal and touches nothing: the pool hands out empty
    // chunks when a tail level has fewer block rows than workers.
    std::vector<uint8_t> untouched(level.size(), 0x11);
    ve::assets::encodeRgba8BlockRows(pixels, kWidth, kHeight, kBlockSize, 1, 1, stampTopLeft, untouched);
    CHECK(untouched == std::vector<uint8_t>(level.size(), 0x11));

    // A level buffer that does not match the block count would silently write
    // past a level boundary once the caller concatenates them.
    std::vector<uint8_t> wrongSize(level.size() - 1);
    CHECK_THROWS_AS(
        ve::assets::encodeRgba8BlockRows(pixels, kWidth, kHeight, kBlockSize, 0, blockRows, stampTopLeft, wrongSize),
        std::runtime_error);
    CHECK_THROWS_AS(
        ve::assets::encodeRgba8BlockRows(pixels, kWidth, kHeight, kBlockSize, 0, blockRows + 1, stampTopLeft, level),
        std::runtime_error);
}

TEST_CASE("Encoded level sizes agree with the KTX2 level index", "[texture-cook]")
{
    // The cook builds a mip chain, encodes each level, and hands the results to
    // writeKtx2, which rejects any level whose size disagrees with its extent.
    // This is the seam between the two, checked without running an encoder.
    constexpr uint32_t kWidth = 100;
    constexpr uint32_t kHeight = 60;

    const std::vector<uint8_t> base = solidImage(kWidth, kHeight, {1, 2, 3, 4});
    const std::vector<std::vector<uint8_t>> mipChain = generateMipChainRgba8(base, kWidth, kHeight, true);

    for (size_t level = 0; level < mipChain.size(); ++level) {
        const uint32_t levelWidth = std::max(1U, kWidth >> level);
        const uint32_t levelHeight = std::max(1U, kHeight >> level);
        const size_t blockBytes = static_cast<size_t>(ve::assets::rgba8BlockColumnCount(levelWidth))
                                  * ve::assets::rgba8BlockRowCount(levelHeight) * 16U;

        CHECK(blockBytes
              == ve::assets::ktx2LevelSizeBytes(kVkFormatBc7SrgbBlock, kWidth, kHeight, static_cast<uint32_t>(level)));
    }
}

TEST_CASE("Block extraction covers a 1x1 image and rejects a short buffer", "[texture-cook]")
{
    // The tail of every mip chain is 1x1, and it still has to produce a block.
    const std::vector<uint8_t> tiny = solidImage(1, 1, {9, 8, 7, 6});
    std::array<uint8_t, 64> block{};
    extractRgba8Block(tiny, 1, 1, 0, 0, block);
    for (size_t texel = 0; texel < 16U; ++texel) {
        CHECK(static_cast<int>(block[texel * 4U + 0]) == 9);
        CHECK(static_cast<int>(block[texel * 4U + 3]) == 6);
    }

    CHECK_THROWS_AS(extractRgba8Block(tiny, 8, 8, 0, 0, block), std::runtime_error);
    CHECK_THROWS_AS(extractRgba8Block(tiny, 0, 1, 0, 0, block), std::runtime_error);
}
