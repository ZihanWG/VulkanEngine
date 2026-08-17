#include "assets/Ktx2.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

using ve::assets::Ktx2Image;
using ve::assets::Ktx2Info;
using ve::assets::kVkFormatBc5UnormBlock;
using ve::assets::kVkFormatBc7SrgbBlock;
using ve::assets::kVkFormatBc7UnormBlock;
using ve::assets::ktx2FullMipChainLength;
using ve::assets::ktx2LevelSizeBytes;
using ve::assets::parseKtx2;
using ve::assets::writeKtx2;

namespace {

// Header field offsets from the KTX 2.0 specification. The tests read the file
// back byte by byte on purpose: a round trip through this project's own parser
// would happily agree with a writer that got the layout wrong.
constexpr size_t kOffsetVkFormat = 12;
constexpr size_t kOffsetPixelWidth = 20;
constexpr size_t kOffsetPixelHeight = 24;
constexpr size_t kOffsetPixelDepth = 28;
constexpr size_t kOffsetLayerCount = 32;
constexpr size_t kOffsetFaceCount = 36;
constexpr size_t kOffsetLevelCount = 40;
constexpr size_t kOffsetSupercompression = 44;
constexpr size_t kOffsetDfdByteOffset = 48;
constexpr size_t kOffsetDfdByteLength = 52;
constexpr size_t kOffsetKvdByteOffset = 56;
constexpr size_t kOffsetLevelIndex = 80;

uint32_t readU32(std::span<const uint8_t> bytes, size_t offset)
{
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8U)
           | (static_cast<uint32_t>(bytes[offset + 2]) << 16U) | (static_cast<uint32_t>(bytes[offset + 3]) << 24U);
}

uint64_t readU64(std::span<const uint8_t> bytes, size_t offset)
{
    return static_cast<uint64_t>(readU32(bytes, offset)) | (static_cast<uint64_t>(readU32(bytes, offset + 4)) << 32U);
}

// A full mip chain of distinct, size-correct fake block bytes. The cook's encoder
// is irrelevant to the container, so the tests never run it.
Ktx2Image makeImage(uint32_t vkFormat, uint32_t width, uint32_t height, uint32_t levelCount)
{
    Ktx2Image image{};
    image.vkFormat = vkFormat;
    image.pixelWidth = width;
    image.pixelHeight = height;

    for (uint32_t level = 0; level < levelCount; ++level) {
        std::vector<uint8_t> bytes(ktx2LevelSizeBytes(vkFormat, width, height, level));
        std::iota(bytes.begin(), bytes.end(), static_cast<uint8_t>(level * 7U + 1U));
        image.levels.push_back(std::move(bytes));
    }

    return image;
}

} // namespace

TEST_CASE("KTX2 mip chain length counts down to 1x1", "[ktx2]")
{
    CHECK(ktx2FullMipChainLength(1, 1) == 1);
    CHECK(ktx2FullMipChainLength(2, 2) == 2);
    CHECK(ktx2FullMipChainLength(1024, 1024) == 11);
    // The longer edge drives the chain; the short one clamps at 1.
    CHECK(ktx2FullMipChainLength(1024, 16) == 11);
    CHECK(ktx2FullMipChainLength(1023, 1023) == 10);
}

TEST_CASE("KTX2 level sizes round up to whole blocks", "[ktx2]")
{
    // 1024x1024 BC7 is 256x256 blocks of 16 bytes.
    CHECK(ktx2LevelSizeBytes(kVkFormatBc7SrgbBlock, 1024, 1024, 0) == 256U * 256U * 16U);
    CHECK(ktx2LevelSizeBytes(kVkFormatBc7SrgbBlock, 1024, 1024, 1) == 128U * 128U * 16U);

    // Non-multiple-of-4 extents cost a whole padded block, and the tail levels
    // below 4x4 still cost one. This is the policy the cook has to live with.
    CHECK(ktx2LevelSizeBytes(kVkFormatBc7SrgbBlock, 5, 5, 0) == 2U * 2U * 16U);
    CHECK(ktx2LevelSizeBytes(kVkFormatBc7SrgbBlock, 4, 4, 0) == 16U);
    CHECK(ktx2LevelSizeBytes(kVkFormatBc7SrgbBlock, 1024, 1024, 10) == 16U);
    CHECK(ktx2LevelSizeBytes(kVkFormatBc5UnormBlock, 8, 8, 0) == 2U * 2U * 16U);

    // An unsupported format has no block layout at all rather than a guess.
    CHECK(ktx2LevelSizeBytes(37 /* VK_FORMAT_R8G8B8A8_UNORM */, 64, 64, 0) == 0);
}

TEST_CASE("KTX2 header describes a single-face 2D image", "[ktx2]")
{
    const Ktx2Image image = makeImage(kVkFormatBc7SrgbBlock, 64, 32, 7);
    const std::vector<uint8_t> bytes = writeKtx2(image);

    const std::array<uint8_t, 12> expectedIdentifier = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                                        0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    CHECK(std::equal(expectedIdentifier.begin(), expectedIdentifier.end(), bytes.begin()));

    CHECK(readU32(bytes, kOffsetVkFormat) == kVkFormatBc7SrgbBlock);
    CHECK(readU32(bytes, kOffsetPixelWidth) == 64);
    CHECK(readU32(bytes, kOffsetPixelHeight) == 32);
    CHECK(readU32(bytes, kOffsetPixelDepth) == 0);
    CHECK(readU32(bytes, kOffsetLayerCount) == 0);
    CHECK(readU32(bytes, kOffsetFaceCount) == 1);
    CHECK(readU32(bytes, kOffsetLevelCount) == 7);
    CHECK(readU32(bytes, kOffsetSupercompression) == 0);

    // The DFD and the key/value block sit after the level index, in that order,
    // and both start 4-aligned as the specification requires.
    const uint32_t dfdOffset = readU32(bytes, kOffsetDfdByteOffset);
    const uint32_t kvdOffset = readU32(bytes, kOffsetKvdByteOffset);
    CHECK(dfdOffset == kOffsetLevelIndex + 24U * 7U);
    CHECK(dfdOffset % 4U == 0);
    CHECK(kvdOffset % 4U == 0);
    CHECK(kvdOffset == dfdOffset + readU32(bytes, kOffsetDfdByteLength));
}

TEST_CASE("KTX2 stores mip data smallest level first", "[ktx2]")
{
    const Ktx2Image image = makeImage(kVkFormatBc7UnormBlock, 256, 256, 9);
    const std::vector<uint8_t> bytes = writeKtx2(image);

    uint64_t previousOffset = 0;
    for (uint32_t level = 0; level < 9; ++level) {
        const size_t entry = kOffsetLevelIndex + 24U * level;
        const uint64_t byteOffset = readU64(bytes, entry);
        const uint64_t byteLength = readU64(bytes, entry + 8);

        CHECK(byteLength == ktx2LevelSizeBytes(kVkFormatBc7UnormBlock, 256, 256, level));
        // Level 0 is the largest and therefore last in the file, so offsets
        // decrease as the level index walks from base to smallest.
        if (level > 0) {
            CHECK(byteOffset < previousOffset);
        }
        // Every level starts on a texel-block boundary.
        CHECK(byteOffset % 16U == 0);
        previousOffset = byteOffset;
    }

    // The smallest level is the first data in the file, right after the metadata.
    const uint64_t smallestOffset = readU64(bytes, kOffsetLevelIndex + 24U * 8U);
    CHECK(smallestOffset >= readU32(bytes, kOffsetKvdByteOffset));

    // The base level ends exactly at the end of the file: no trailing padding.
    const uint64_t baseOffset = readU64(bytes, kOffsetLevelIndex);
    const uint64_t baseLength = readU64(bytes, kOffsetLevelIndex + 8);
    CHECK(baseOffset + baseLength == bytes.size());
}

TEST_CASE("KTX2 round-trips every level's bytes", "[ktx2]")
{
    const Ktx2Image image = makeImage(kVkFormatBc7SrgbBlock, 100, 60, 7);
    const std::vector<uint8_t> bytes = writeKtx2(image);
    const Ktx2Info parsed = parseKtx2(bytes);

    REQUIRE(parsed.vkFormat == kVkFormatBc7SrgbBlock);
    REQUIRE(parsed.pixelWidth == 100);
    REQUIRE(parsed.pixelHeight == 60);
    REQUIRE(parsed.levels.size() == image.levels.size());

    for (size_t level = 0; level < parsed.levels.size(); ++level) {
        const std::vector<uint8_t>& source = image.levels[level];
        REQUIRE(parsed.levels[level].byteLength == source.size());

        const uint8_t* stored = bytes.data() + parsed.levels[level].byteOffset;
        CHECK(std::equal(source.begin(), source.end(), stored));
    }
}

TEST_CASE("KTX2 round-trips a single-level image", "[ktx2]")
{
    // The cook can be asked for base-only output, and a level count of 1 is the
    // case where the smallest-first data order is invisible.
    const Ktx2Image image = makeImage(kVkFormatBc5UnormBlock, 32, 32, 1);
    const std::vector<uint8_t> bytes = writeKtx2(image);
    const Ktx2Info parsed = parseKtx2(bytes);

    REQUIRE(parsed.levels.size() == 1);
    CHECK(parsed.levels[0].byteLength == 8U * 8U * 16U);
    CHECK(std::equal(image.levels[0].begin(), image.levels[0].end(), bytes.data() + parsed.levels[0].byteOffset));
}

TEST_CASE("KTX2 data format descriptor matches the format", "[ktx2]")
{
    struct Expectation {
        uint32_t vkFormat;
        uint32_t colorModel;
        uint32_t transferFunction;
        uint32_t sampleCount;
    };

    // khr_df.h: BC7 is model 134, BC5 is 132; transfer 2 is sRGB, 1 is linear.
    const std::array<Expectation, 3> expectations = {
        Expectation{kVkFormatBc7SrgbBlock, 134, 2, 1},
        Expectation{kVkFormatBc7UnormBlock, 134, 1, 1},
        Expectation{kVkFormatBc5UnormBlock, 132, 1, 2},
    };

    for (const Expectation& expectation : expectations) {
        const Ktx2Image image = makeImage(expectation.vkFormat, 16, 16, 5);
        const std::vector<uint8_t> bytes = writeKtx2(image);

        const size_t dfd = readU32(bytes, kOffsetDfdByteOffset);
        const uint32_t descriptorBlockSize = 24U + 16U * expectation.sampleCount;

        CHECK(readU32(bytes, kOffsetDfdByteLength) == 4U + descriptorBlockSize);
        CHECK(readU32(bytes, dfd) == 4U + descriptorBlockSize);
        CHECK(readU32(bytes, dfd + 4) == 0); // KHRONOS vendor, basic descriptor
        CHECK(readU32(bytes, dfd + 8) == (2U | (descriptorBlockSize << 16U)));

        const uint32_t word2 = readU32(bytes, dfd + 12);
        CHECK((word2 & 0xffU) == expectation.colorModel);
        CHECK(((word2 >> 8U) & 0xffU) == 1U); // BT.709 primaries
        CHECK(((word2 >> 16U) & 0xffU) == expectation.transferFunction);
        CHECK(((word2 >> 24U) & 0xffU) == 0U); // straight alpha

        // Texel block dimensions are stored as size-1, so a 4x4 block reads 3,3.
        const uint32_t word3 = readU32(bytes, dfd + 16);
        CHECK((word3 & 0xffU) == 3U);
        CHECK(((word3 >> 8U) & 0xffU) == 3U);
        CHECK(((word3 >> 16U) & 0xffU) == 0U);
        CHECK(((word3 >> 24U) & 0xffU) == 0U);

        // bytesPlane0 is the block size; the remaining planes are unused.
        CHECK(readU32(bytes, dfd + 20) == 16U);
        CHECK(readU32(bytes, dfd + 24) == 0U);

        // Samples cover the whole 128-bit block, contiguously and in order.
        uint32_t expectedBitOffset = 0;
        for (uint32_t sample = 0; sample < expectation.sampleCount; ++sample) {
            const size_t sampleOffset = dfd + 28 + 16U * sample;
            const uint32_t word0 = readU32(bytes, sampleOffset);
            const uint32_t bitLength = (word0 >> 16U) & 0xffU;

            CHECK((word0 & 0xffffU) == expectedBitOffset);
            CHECK(bitLength == (128U / expectation.sampleCount) - 1U);
            CHECK(((word0 >> 24U) & 0xffU) == sample); // channel type, no qualifiers
            CHECK(readU32(bytes, sampleOffset + 8) == 0U);          // sampleLower
            CHECK(readU32(bytes, sampleOffset + 12) == 0xffffffffU); // sampleUpper

            expectedBitOffset += bitLength + 1U;
        }
        CHECK(expectedBitOffset == 128U);
    }
}

TEST_CASE("KTX2 writer rejects images it cannot describe", "[ktx2]")
{
    // An unsupported format would otherwise produce a file with no valid DFD.
    Ktx2Image wrongFormat = makeImage(kVkFormatBc7SrgbBlock, 16, 16, 1);
    wrongFormat.vkFormat = 37; // VK_FORMAT_R8G8B8A8_UNORM
    CHECK_THROWS_AS(writeKtx2(wrongFormat), std::runtime_error);

    Ktx2Image noLevels = makeImage(kVkFormatBc7SrgbBlock, 16, 16, 1);
    noLevels.levels.clear();
    CHECK_THROWS_AS(writeKtx2(noLevels), std::runtime_error);

    Ktx2Image zeroExtent = makeImage(kVkFormatBc7SrgbBlock, 16, 16, 1);
    zeroExtent.pixelHeight = 0;
    CHECK_THROWS_AS(writeKtx2(zeroExtent), std::runtime_error);

    // 16x16 has five levels; a sixth has no extent to belong to.
    Ktx2Image tooManyLevels = makeImage(kVkFormatBc7SrgbBlock, 16, 16, 5);
    tooManyLevels.levels.push_back(std::vector<uint8_t>(16));
    CHECK_THROWS_AS(writeKtx2(tooManyLevels), std::runtime_error);

    // A short level is the truncated-encode case: caught here, not in a driver.
    Ktx2Image shortLevel = makeImage(kVkFormatBc7SrgbBlock, 16, 16, 2);
    shortLevel.levels[1].pop_back();
    CHECK_THROWS_AS(writeKtx2(shortLevel), std::runtime_error);
}

TEST_CASE("KTX2 parser rejects malformed files", "[ktx2]")
{
    const Ktx2Image image = makeImage(kVkFormatBc7SrgbBlock, 32, 32, 6);
    const std::vector<uint8_t> good = writeKtx2(image);
    CHECK_NOTHROW(parseKtx2(good));

    CHECK_THROWS_AS(parseKtx2(std::span<const uint8_t>{}), std::runtime_error);

    std::vector<uint8_t> badIdentifier = good;
    badIdentifier[1] = 'X';
    CHECK_THROWS_AS(parseKtx2(badIdentifier), std::runtime_error);

    std::vector<uint8_t> truncated(good.begin(), good.begin() + 60);
    CHECK_THROWS_AS(parseKtx2(truncated), std::runtime_error);

    // A level index that disagrees with the extent is the silent-corruption case
    // the round trip alone would not notice.
    std::vector<uint8_t> wrongLevelSize = good;
    wrongLevelSize[kOffsetLevelIndex + 8] += 1;
    CHECK_THROWS_AS(parseKtx2(wrongLevelSize), std::runtime_error);

    std::vector<uint8_t> cubeMap = good;
    cubeMap[kOffsetFaceCount] = 6;
    CHECK_THROWS_AS(parseKtx2(cubeMap), std::runtime_error);

    std::vector<uint8_t> supercompressed = good;
    supercompressed[kOffsetSupercompression] = 1;
    CHECK_THROWS_AS(parseKtx2(supercompressed), std::runtime_error);

    // Truncating the payload leaves the index pointing past the end of the file.
    std::vector<uint8_t> missingData(good.begin(), good.end() - 16);
    CHECK_THROWS_AS(parseKtx2(missingData), std::runtime_error);
}
