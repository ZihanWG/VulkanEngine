#include "assets/Ktx2.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ve::assets {

namespace {

constexpr std::array<uint8_t, 12> kIdentifier = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                                 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

// Fixed section sizes from the KTX 2.0 specification. The level index follows the
// header immediately, so the first variable-length section (the DFD) starts at
// kHeaderAndIndexBytes + 24 * levelCount.
constexpr size_t kHeaderAndIndexBytes = 80;
constexpr size_t kLevelIndexEntryBytes = 24;

// Data format descriptor constants (Khronos Data Format 1.3, khr_df.h). Spelled
// out here rather than pulled in as a dependency: six values, and the header is
// not otherwise vendored.
constexpr uint32_t kDfdVersionNumber = 2;
constexpr uint32_t kDfdModelBc5 = 132;
constexpr uint32_t kDfdModelBc7 = 134;
constexpr uint32_t kDfdPrimariesBt709 = 1;
constexpr uint32_t kDfdTransferLinear = 1;
constexpr uint32_t kDfdTransferSrgb = 2;
constexpr uint32_t kDfdAlphaStraight = 0;
constexpr uint32_t kDfdChannelBc7Color = 0;
constexpr uint32_t kDfdChannelBc5Red = 0;
constexpr uint32_t kDfdChannelBc5Green = 1;

constexpr std::string_view kWriterIdentifier = "VulkanEngine vecook v1";

void appendU32(std::vector<uint8_t>& bytes, uint32_t value)
{
    bytes.push_back(static_cast<uint8_t>(value & 0xffU));
    bytes.push_back(static_cast<uint8_t>((value >> 8U) & 0xffU));
    bytes.push_back(static_cast<uint8_t>((value >> 16U) & 0xffU));
    bytes.push_back(static_cast<uint8_t>((value >> 24U) & 0xffU));
}

void appendU64(std::vector<uint8_t>& bytes, uint64_t value)
{
    appendU32(bytes, static_cast<uint32_t>(value & 0xffffffffULL));
    appendU32(bytes, static_cast<uint32_t>((value >> 32U) & 0xffffffffULL));
}

void writeU32At(std::vector<uint8_t>& bytes, size_t offset, uint32_t value)
{
    bytes[offset + 0] = static_cast<uint8_t>(value & 0xffU);
    bytes[offset + 1] = static_cast<uint8_t>((value >> 8U) & 0xffU);
    bytes[offset + 2] = static_cast<uint8_t>((value >> 16U) & 0xffU);
    bytes[offset + 3] = static_cast<uint8_t>((value >> 24U) & 0xffU);
}

void writeU64At(std::vector<uint8_t>& bytes, size_t offset, uint64_t value)
{
    writeU32At(bytes, offset, static_cast<uint32_t>(value & 0xffffffffULL));
    writeU32At(bytes, offset + 4, static_cast<uint32_t>((value >> 32U) & 0xffffffffULL));
}

[[nodiscard]] uint32_t readU32At(std::span<const uint8_t> bytes, size_t offset)
{
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8U)
           | (static_cast<uint32_t>(bytes[offset + 2]) << 16U) | (static_cast<uint32_t>(bytes[offset + 3]) << 24U);
}

[[nodiscard]] uint64_t readU64At(std::span<const uint8_t> bytes, size_t offset)
{
    return static_cast<uint64_t>(readU32At(bytes, offset))
           | (static_cast<uint64_t>(readU32At(bytes, offset + 4)) << 32U);
}

[[nodiscard]] uint32_t levelExtent(uint32_t base, uint32_t level)
{
    return std::max(1U, base >> level);
}

// KTX 2.0 requires every mip level to start at a multiple of the least common
// multiple of the texel block size and 4. Every block size the cook emits is a
// multiple of 4, so the LCM is just the block size.
[[nodiscard]] size_t mipAlignment(const Ktx2FormatInfo& info)
{
    return info.blockSizeBytes;
}

[[nodiscard]] size_t alignUp(size_t value, size_t alignment)
{
    return ((value + alignment - 1U) / alignment) * alignment;
}

// One key/value entry: NUL-terminated key, value bytes, then padding to 4.
void appendKeyValue(std::vector<uint8_t>& bytes, std::string_view key, std::string_view value)
{
    const uint32_t entryBytes = static_cast<uint32_t>(key.size() + 1U + value.size() + 1U);
    appendU32(bytes, entryBytes);
    bytes.insert(bytes.end(), key.begin(), key.end());
    bytes.push_back(0);
    bytes.insert(bytes.end(), value.begin(), value.end());
    bytes.push_back(0);
    bytes.resize(alignUp(bytes.size(), 4U), 0);
}

struct DfdSample {
    uint32_t bitOffset = 0;
    uint32_t bitLength = 0;
    uint32_t channelType = 0;
};

// Basic data format descriptor for one block-compressed format. KTX 2.0 makes
// this mandatory and tools read the block dimensions and transfer function from
// it rather than from vkFormat, so a wrong DFD is exactly the silent-failure mode
// the round-trip tests exist to catch.
[[nodiscard]] std::vector<uint8_t> makeDataFormatDescriptor(uint32_t vkFormat, const Ktx2FormatInfo& info)
{
    uint32_t colorModel = 0;
    uint32_t transferFunction = kDfdTransferLinear;
    std::vector<DfdSample> samples;

    switch (vkFormat) {
    case kVkFormatBc7SrgbBlock:
        transferFunction = kDfdTransferSrgb;
        [[fallthrough]];
    case kVkFormatBc7UnormBlock:
        colorModel = kDfdModelBc7;
        samples.push_back(DfdSample{0, 127, kDfdChannelBc7Color});
        break;
    case kVkFormatBc5UnormBlock:
        colorModel = kDfdModelBc5;
        samples.push_back(DfdSample{0, 63, kDfdChannelBc5Red});
        samples.push_back(DfdSample{64, 63, kDfdChannelBc5Green});
        break;
    default:
        throw std::runtime_error("KTX2: no data format descriptor for vkFormat "
                                 + std::to_string(vkFormat));
    }

    const uint32_t descriptorBlockSize = 24U + 16U * static_cast<uint32_t>(samples.size());

    std::vector<uint8_t> dfd;
    dfd.reserve(4U + descriptorBlockSize);
    appendU32(dfd, 4U + descriptorBlockSize);
    appendU32(dfd, 0); // vendorId KHRONOS, descriptorType BASICFORMAT
    appendU32(dfd, kDfdVersionNumber | (descriptorBlockSize << 16U));
    appendU32(dfd, colorModel | (kDfdPrimariesBt709 << 8U) | (transferFunction << 16U) | (kDfdAlphaStraight << 24U));

    // Texel block dimensions are stored as size-1, and unused dimensions as 0.
    appendU32(dfd, (info.blockWidth - 1U) | ((info.blockHeight - 1U) << 8U));
    appendU32(dfd, info.blockSizeBytes); // bytesPlane0, planes 1-3 unused
    appendU32(dfd, 0);                   // bytesPlane4-7 unused

    for (const DfdSample& sample : samples) {
        appendU32(dfd, sample.bitOffset | (sample.bitLength << 16U) | (sample.channelType << 24U));
        appendU32(dfd, 0);          // samplePosition0-3, all zero for a whole block
        appendU32(dfd, 0);          // sampleLower
        appendU32(dfd, 0xffffffffU); // sampleUpper: unsigned normalized
    }

    return dfd;
}

} // namespace

Ktx2FormatInfo ktx2FormatInfo(uint32_t vkFormat)
{
    switch (vkFormat) {
    case kVkFormatBc5UnormBlock:
    case kVkFormatBc7UnormBlock:
    case kVkFormatBc7SrgbBlock:
        return Ktx2FormatInfo{4, 4, 16};
    default:
        return Ktx2FormatInfo{};
    }
}

bool ktx2FormatSupported(uint32_t vkFormat)
{
    return ktx2FormatInfo(vkFormat).blockSizeBytes != 0;
}

size_t ktx2LevelSizeBytes(uint32_t vkFormat, uint32_t baseWidth, uint32_t baseHeight, uint32_t level)
{
    const Ktx2FormatInfo info = ktx2FormatInfo(vkFormat);
    if (info.blockSizeBytes == 0) {
        return 0;
    }

    const uint32_t width = levelExtent(baseWidth, level);
    const uint32_t height = levelExtent(baseHeight, level);
    const size_t blocksX = (width + info.blockWidth - 1U) / info.blockWidth;
    const size_t blocksY = (height + info.blockHeight - 1U) / info.blockHeight;
    return blocksX * blocksY * info.blockSizeBytes;
}

uint32_t ktx2FullMipChainLength(uint32_t baseWidth, uint32_t baseHeight)
{
    uint32_t largest = std::max(baseWidth, baseHeight);
    uint32_t levels = 1;
    while (largest > 1U) {
        largest >>= 1U;
        ++levels;
    }

    return levels;
}

std::vector<uint8_t> writeKtx2(const Ktx2Image& image)
{
    const Ktx2FormatInfo info = ktx2FormatInfo(image.vkFormat);
    if (info.blockSizeBytes == 0) {
        throw std::runtime_error("KTX2: unsupported vkFormat " + std::to_string(image.vkFormat));
    }
    if (image.pixelWidth == 0 || image.pixelHeight == 0) {
        throw std::runtime_error("KTX2: cannot write a zero-sized image.");
    }
    if (image.levels.empty()) {
        throw std::runtime_error("KTX2: an image needs at least a base level.");
    }

    const uint32_t levelCount = static_cast<uint32_t>(image.levels.size());
    if (levelCount > ktx2FullMipChainLength(image.pixelWidth, image.pixelHeight)) {
        throw std::runtime_error("KTX2: more mip levels than the extent allows.");
    }

    // Validating every level against the block layout here is what keeps a
    // truncated encode from becoming a file that only fails inside a driver.
    for (uint32_t level = 0; level < levelCount; ++level) {
        const size_t expected = ktx2LevelSizeBytes(image.vkFormat, image.pixelWidth, image.pixelHeight, level);
        if (image.levels[level].size() != expected) {
            throw std::runtime_error("KTX2: mip level " + std::to_string(level) + " is "
                                     + std::to_string(image.levels[level].size()) + " bytes, expected "
                                     + std::to_string(expected) + ".");
        }
    }

    const std::vector<uint8_t> dfd = makeDataFormatDescriptor(image.vkFormat, info);

    std::vector<uint8_t> keyValueData;
    appendKeyValue(keyValueData, "KTXorientation", "rd");
    appendKeyValue(keyValueData, "KTXwriter", kWriterIdentifier);

    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), kIdentifier.begin(), kIdentifier.end());
    appendU32(bytes, image.vkFormat);
    appendU32(bytes, 1); // typeSize is 1 for every block-compressed format
    appendU32(bytes, image.pixelWidth);
    appendU32(bytes, image.pixelHeight);
    appendU32(bytes, 0);          // pixelDepth: 2D
    appendU32(bytes, 0);          // layerCount: not an array
    appendU32(bytes, 1);          // faceCount
    appendU32(bytes, levelCount);
    appendU32(bytes, 0); // supercompressionScheme: none

    const size_t indexOffset = bytes.size();
    appendU32(bytes, 0); // dfdByteOffset, patched below
    appendU32(bytes, static_cast<uint32_t>(dfd.size()));
    appendU32(bytes, 0); // kvdByteOffset, patched below
    appendU32(bytes, static_cast<uint32_t>(keyValueData.size()));
    appendU64(bytes, 0); // sgdByteOffset: no supercompression global data
    appendU64(bytes, 0); // sgdByteLength

    const size_t levelIndexOffset = bytes.size();
    bytes.resize(levelIndexOffset + kLevelIndexEntryBytes * levelCount, 0);

    const size_t dfdOffset = bytes.size();
    bytes.insert(bytes.end(), dfd.begin(), dfd.end());
    const size_t keyValueOffset = bytes.size();
    bytes.insert(bytes.end(), keyValueData.begin(), keyValueData.end());

    writeU32At(bytes, indexOffset, static_cast<uint32_t>(dfdOffset));
    writeU32At(bytes, indexOffset + 8, static_cast<uint32_t>(keyValueOffset));

    // Mip data runs smallest level first, which is the reverse of the level index
    // and of everything Vulkan calls a mip level. Only this loop is reversed.
    const size_t alignment = mipAlignment(info);
    for (uint32_t reversed = 0; reversed < levelCount; ++reversed) {
        const uint32_t level = levelCount - 1U - reversed;
        bytes.resize(alignUp(bytes.size(), alignment), 0);

        const size_t levelOffset = bytes.size();
        const std::vector<uint8_t>& levelBytes = image.levels[level];
        bytes.insert(bytes.end(), levelBytes.begin(), levelBytes.end());

        const size_t entry = levelIndexOffset + kLevelIndexEntryBytes * level;
        writeU64At(bytes, entry, levelOffset);
        writeU64At(bytes, entry + 8, levelBytes.size());
        writeU64At(bytes, entry + 16, levelBytes.size()); // uncompressed == stored
    }

    return bytes;
}

void writeKtx2File(const std::filesystem::path& path, const Ktx2Image& image)
{
    const std::vector<uint8_t> bytes = writeKtx2(image);

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open KTX2 output file: " + path.string());
    }

    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("Failed to write KTX2 output file: " + path.string());
    }
}

Ktx2Info parseKtx2(std::span<const uint8_t> bytes)
{
    if (bytes.size() < kHeaderAndIndexBytes) {
        throw std::runtime_error("KTX2: file is shorter than the header.");
    }
    if (std::memcmp(bytes.data(), kIdentifier.data(), kIdentifier.size()) != 0) {
        throw std::runtime_error("KTX2: bad file identifier.");
    }

    Ktx2Info parsed{};
    parsed.vkFormat = readU32At(bytes, 12);
    parsed.pixelWidth = readU32At(bytes, 20);
    parsed.pixelHeight = readU32At(bytes, 24);

    const uint32_t pixelDepth = readU32At(bytes, 28);
    const uint32_t layerCount = readU32At(bytes, 32);
    const uint32_t faceCount = readU32At(bytes, 36);
    const uint32_t levelCount = readU32At(bytes, 40);
    const uint32_t supercompression = readU32At(bytes, 44);

    // The cook writes plain 2D single-face images and the upload path assumes it,
    // so anything else is rejected here rather than misread as a 2D image.
    if (pixelDepth != 0 || layerCount != 0 || faceCount != 1) {
        throw std::runtime_error("KTX2: only single-face 2D images are supported.");
    }
    if (supercompression != 0) {
        throw std::runtime_error("KTX2: supercompressed files are not supported.");
    }
    if (!ktx2FormatSupported(parsed.vkFormat)) {
        throw std::runtime_error("KTX2: unsupported vkFormat " + std::to_string(parsed.vkFormat));
    }
    if (levelCount == 0 || levelCount > ktx2FullMipChainLength(parsed.pixelWidth, parsed.pixelHeight)) {
        throw std::runtime_error("KTX2: implausible level count " + std::to_string(levelCount));
    }
    if (bytes.size() < kHeaderAndIndexBytes + kLevelIndexEntryBytes * levelCount) {
        throw std::runtime_error("KTX2: file is shorter than its level index.");
    }

    parsed.levels.resize(levelCount);
    for (uint32_t level = 0; level < levelCount; ++level) {
        const size_t entry = kHeaderAndIndexBytes + kLevelIndexEntryBytes * level;
        const uint64_t byteOffset = readU64At(bytes, entry);
        const uint64_t byteLength = readU64At(bytes, entry + 8);
        const uint64_t uncompressedLength = readU64At(bytes, entry + 16);

        if (byteLength != uncompressedLength) {
            throw std::runtime_error("KTX2: level " + std::to_string(level) + " claims compression.");
        }
        if (byteLength != ktx2LevelSizeBytes(parsed.vkFormat, parsed.pixelWidth, parsed.pixelHeight, level)) {
            throw std::runtime_error("KTX2: level " + std::to_string(level) + " has the wrong size for its extent.");
        }
        if (byteOffset > bytes.size() || byteLength > bytes.size() - byteOffset) {
            throw std::runtime_error("KTX2: level " + std::to_string(level) + " runs past the end of the file.");
        }

        parsed.levels[level] = Ktx2LevelView{byteOffset, byteLength};
    }

    return parsed;
}

} // namespace ve::assets
