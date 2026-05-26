#include "core/PngWriter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve {

namespace {

constexpr std::array<uint8_t, 8> kPngSignature = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
constexpr uint32_t kAdlerMod = 65521;
constexpr size_t kDeflateStoredBlockMaxSize = 65535;

void appendU32Be(std::vector<uint8_t>& bytes, uint32_t value)
{
    bytes.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
    bytes.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    bytes.push_back(static_cast<uint8_t>(value & 0xff));
}

uint32_t crc32Update(uint32_t crc, std::span<const uint8_t> data)
{
    for (uint8_t value : data) {
        crc ^= value;
        for (uint32_t bit = 0; bit < 8; ++bit) {
            crc = (crc & 1U) != 0U ? (crc >> 1U) ^ 0xedb88320U : crc >> 1U;
        }
    }

    return crc;
}

uint32_t pngChunkCrc(std::span<const uint8_t> type, std::span<const uint8_t> data)
{
    uint32_t crc = 0xffffffffU;
    crc = crc32Update(crc, type);
    crc = crc32Update(crc, data);
    return crc ^ 0xffffffffU;
}

uint32_t adler32(std::span<const uint8_t> data)
{
    uint32_t a = 1;
    uint32_t b = 0;
    for (uint8_t value : data) {
        a = (a + value) % kAdlerMod;
        b = (b + a) % kAdlerMod;
    }

    return (b << 16U) | a;
}

void appendChunk(std::vector<uint8_t>& png, const std::array<uint8_t, 4>& type, std::span<const uint8_t> data)
{
    appendU32Be(png, static_cast<uint32_t>(data.size()));
    png.insert(png.end(), type.begin(), type.end());
    png.insert(png.end(), data.begin(), data.end());
    appendU32Be(png, pngChunkCrc(type, data));
}

std::vector<uint8_t> makeScanlines(uint32_t width,
                                   uint32_t height,
                                   std::span<const uint8_t> rgbaPixels,
                                   uint32_t rowStrideBytes)
{
    const size_t tightRowBytes = static_cast<size_t>(width) * 4U;
    const size_t requiredBytes = height == 0 ? 0 : (static_cast<size_t>(height - 1U) * rowStrideBytes) + tightRowBytes;
    if (rgbaPixels.size() < requiredBytes) {
        throw std::runtime_error("PNG input pixel span is smaller than the requested image extent.");
    }

    std::vector<uint8_t> scanlines;
    scanlines.resize((tightRowBytes + 1U) * static_cast<size_t>(height));
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* source = rgbaPixels.data() + static_cast<size_t>(y) * rowStrideBytes;
        uint8_t* destination = scanlines.data() + static_cast<size_t>(y) * (tightRowBytes + 1U);
        destination[0] = 0;
        std::copy(source, source + tightRowBytes, destination + 1);
    }

    return scanlines;
}

std::vector<uint8_t> makeZlibStoredStream(std::span<const uint8_t> data)
{
    std::vector<uint8_t> stream;
    stream.reserve(data.size() + (data.size() / kDeflateStoredBlockMaxSize + 1U) * 5U + 6U);
    stream.push_back(0x78);
    stream.push_back(0x01);

    size_t offset = 0;
    while (offset < data.size() || (data.empty() && offset == 0)) {
        const size_t remaining = data.size() - offset;
        const size_t blockSize = std::min(remaining, kDeflateStoredBlockMaxSize);
        const bool finalBlock = offset + blockSize >= data.size();
        const uint16_t len = static_cast<uint16_t>(blockSize);
        const uint16_t nlen = static_cast<uint16_t>(~len);

        stream.push_back(finalBlock ? 0x01 : 0x00);
        stream.push_back(static_cast<uint8_t>(len & 0xff));
        stream.push_back(static_cast<uint8_t>((len >> 8U) & 0xff));
        stream.push_back(static_cast<uint8_t>(nlen & 0xff));
        stream.push_back(static_cast<uint8_t>((nlen >> 8U) & 0xff));
        stream.insert(stream.end(), data.begin() + static_cast<std::ptrdiff_t>(offset),
                      data.begin() + static_cast<std::ptrdiff_t>(offset + blockSize));

        offset += blockSize;
        if (data.empty()) {
            break;
        }
    }

    appendU32Be(stream, adler32(data));
    return stream;
}

} // namespace

void writePngRgba8(const std::filesystem::path& path,
                   uint32_t width,
                   uint32_t height,
                   std::span<const uint8_t> rgbaPixels,
                   uint32_t rowStrideBytes)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("Cannot write a zero-sized PNG.");
    }
    if (rowStrideBytes < width * 4U) {
        throw std::runtime_error("PNG row stride is smaller than the image width.");
    }

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    const std::vector<uint8_t> scanlines = makeScanlines(width, height, rgbaPixels, rowStrideBytes);
    const std::vector<uint8_t> idat = makeZlibStoredStream(scanlines);

    std::vector<uint8_t> png;
    png.reserve(kPngSignature.size() + 64U + idat.size());
    png.insert(png.end(), kPngSignature.begin(), kPngSignature.end());

    std::vector<uint8_t> ihdr;
    ihdr.reserve(13);
    appendU32Be(ihdr, width);
    appendU32Be(ihdr, height);
    ihdr.push_back(8);
    ihdr.push_back(6);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);

    appendChunk(png, std::array<uint8_t, 4>{'I', 'H', 'D', 'R'}, ihdr);
    appendChunk(png, std::array<uint8_t, 4>{'I', 'D', 'A', 'T'}, idat);
    appendChunk(png, std::array<uint8_t, 4>{'I', 'E', 'N', 'D'}, std::span<const uint8_t>{});

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open PNG output file: " + path.string());
    }

    output.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!output) {
        throw std::runtime_error("Failed to write PNG output file: " + path.string());
    }
}

} // namespace ve
