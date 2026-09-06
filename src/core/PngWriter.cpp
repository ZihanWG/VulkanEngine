#include "core/PngWriter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve {

namespace {

constexpr std::array<uint8_t, 8> kPngSignature = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
constexpr uint32_t kAdlerMod = 65521;
constexpr size_t kBytesPerPixel = 4;

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

// ---- scanline filtering ----------------------------------------------------

// PNG filters each row against its left and upper neighbours before the row
// reaches the compressor (PNG spec section 9). Rendered frames are full of
// smooth gradients, so the residuals cluster near zero and compress far better
// than the raw bytes: on a 1280x720 capture, filtering roughly halves the
// deflated size on its own.

uint8_t paethPredictor(uint8_t left, uint8_t above, uint8_t upperLeft)
{
    const int estimate = static_cast<int>(left) + static_cast<int>(above) - static_cast<int>(upperLeft);
    const int distanceLeft = std::abs(estimate - static_cast<int>(left));
    const int distanceAbove = std::abs(estimate - static_cast<int>(above));
    const int distanceUpperLeft = std::abs(estimate - static_cast<int>(upperLeft));

    if (distanceLeft <= distanceAbove && distanceLeft <= distanceUpperLeft) {
        return left;
    }

    return distanceAbove <= distanceUpperLeft ? above : upperLeft;
}

uint8_t filteredByte(uint8_t filter, uint8_t raw, uint8_t left, uint8_t above, uint8_t upperLeft)
{
    switch (filter) {
    case 1:
        return static_cast<uint8_t>(raw - left);
    case 2:
        return static_cast<uint8_t>(raw - above);
    case 3:
        return static_cast<uint8_t>(raw - static_cast<uint8_t>((static_cast<int>(left) + static_cast<int>(above)) / 2));
    case 4:
        return static_cast<uint8_t>(raw - paethPredictor(left, above, upperLeft));
    default:
        return raw;
    }
}

// Filtered rows, each prefixed with the filter byte the row was encoded with.
//
// The filter is chosen per row by the minimum-sum-of-absolute-differences
// heuristic the PNG spec suggests: treat each filtered byte as signed, sum the
// magnitudes, and keep the filter with the smallest total. It is a proxy for
// "how compressible is this row", and a cheap one -- five passes over a row of
// residuals, no trial compression.
std::vector<uint8_t> makeFilteredScanlines(uint32_t width,
                                           uint32_t height,
                                           std::span<const uint8_t> rgbaPixels,
                                           uint32_t rowStrideBytes)
{
    const size_t tightRowBytes = static_cast<size_t>(width) * kBytesPerPixel;
    const size_t requiredBytes = height == 0 ? 0 : (static_cast<size_t>(height - 1U) * rowStrideBytes) + tightRowBytes;
    if (rgbaPixels.size() < requiredBytes) {
        throw std::runtime_error("PNG input pixel span is smaller than the requested image extent.");
    }

    std::vector<uint8_t> filtered;
    filtered.reserve((tightRowBytes + 1U) * static_cast<size_t>(height));

    std::vector<uint8_t> previousRow(tightRowBytes, 0);
    std::vector<uint8_t> candidate(tightRowBytes, 0);
    std::vector<uint8_t> best(tightRowBytes, 0);

    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = rgbaPixels.data() + static_cast<size_t>(y) * rowStrideBytes;

        uint8_t bestFilter = 0;
        size_t bestScore = 0;
        for (uint8_t filter = 0; filter < 5; ++filter) {
            size_t score = 0;
            for (size_t i = 0; i < tightRowBytes; ++i) {
                const uint8_t left = i >= kBytesPerPixel ? row[i - kBytesPerPixel] : 0;
                const uint8_t above = previousRow[i];
                const uint8_t upperLeft = i >= kBytesPerPixel ? previousRow[i - kBytesPerPixel] : 0;
                const uint8_t value = filteredByte(filter, row[i], left, above, upperLeft);
                candidate[i] = value;
                // Signed magnitude: bytes at or above 128 stand for -128..-1.
                score += value < 128U ? value : 256U - value;
            }

            if (filter == 0 || score < bestScore) {
                bestScore = score;
                bestFilter = filter;
                best.swap(candidate);
            }
        }

        filtered.push_back(bestFilter);
        filtered.insert(filtered.end(), best.begin(), best.end());
        previousRow.assign(row, row + tightRowBytes);
    }

    return filtered;
}

// ---- deflate ---------------------------------------------------------------

// Fixed-Huffman deflate (RFC 1951 section 3.2.6) over an LZ77 pass.
//
// The engine links no zlib and this writer is deliberately dependency-free, so
// this is the smallest encoder that actually compresses: a hash-chained match
// search feeding deflate's static literal/length tree. What it replaced emitted
// stored blocks -- valid, and no compression at all, which cost about 4 bytes
// per pixel in every screenshot and every tracked golden image.
//
// Dynamic Huffman would save perhaps another 15% for considerably more code:
// the code-length alphabet, its own run-length encoding, and a tree builder.
// Fixed codes give up that last margin and need none of it.

constexpr size_t kWindowSize = 32768;
constexpr size_t kMinMatch = 3;
constexpr size_t kMaxMatch = 258;
constexpr size_t kHashSize = 1U << 15U;
constexpr size_t kMaxChainLength = 128;

// RFC 1951 section 3.2.5. Length code 257 + i covers kLengthBase[i] upward,
// with kLengthExtra[i] literal bits selecting the value inside that run.
constexpr std::array<uint16_t, 29> kLengthBase = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<uint8_t, 29> kLengthExtra = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<uint16_t, 30> kDistanceBase = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::array<uint8_t, 30> kDistanceExtra = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

class BitWriter
{
public:
    explicit BitWriter(std::vector<uint8_t>& output) : output_(output) {}

    // Deflate fills each byte starting from its least significant bit.
    void putBits(uint32_t value, uint32_t count)
    {
        if (count == 0) {
            return;
        }

        bits_ |= static_cast<uint64_t>(value & ((1U << count) - 1U)) << bitCount_;
        bitCount_ += count;
        while (bitCount_ >= 8) {
            output_.push_back(static_cast<uint8_t>(bits_ & 0xffU));
            bits_ >>= 8U;
            bitCount_ -= 8;
        }
    }

    // Huffman codes are the one thing in the stream that travels most
    // significant bit first, so they cannot go through putBits directly.
    void putCode(uint32_t code, uint32_t count)
    {
        for (uint32_t bit = 0; bit < count; ++bit) {
            putBits((code >> (count - 1U - bit)) & 1U, 1);
        }
    }

    void flushToByteBoundary()
    {
        if (bitCount_ > 0) {
            output_.push_back(static_cast<uint8_t>(bits_ & 0xffU));
            bits_ = 0;
            bitCount_ = 0;
        }
    }

private:
    std::vector<uint8_t>& output_;
    uint64_t bits_ = 0;
    uint32_t bitCount_ = 0;
};

// The static literal/length tree, given as code lengths in RFC 1951 section
// 3.2.6: 0-143 are eight bits from 0x30, 144-255 nine bits from 0x190,
// 256-279 seven bits from 0, and 280-287 eight bits from 0xc0.
void putLiteralOrLengthCode(BitWriter& writer, uint32_t symbol)
{
    if (symbol < 144U) {
        writer.putCode(0x30U + symbol, 8);
    } else if (symbol < 256U) {
        writer.putCode(0x190U + (symbol - 144U), 9);
    } else if (symbol < 280U) {
        writer.putCode(symbol - 256U, 7);
    } else {
        writer.putCode(0xc0U + (symbol - 280U), 8);
    }
}

std::vector<uint8_t> makeZlibDeflateStream(std::span<const uint8_t> data)
{
    std::vector<uint8_t> stream;
    stream.reserve(data.size() / 2U + 64U);
    stream.push_back(0x78);   // zlib header: deflate, 32 KiB window
    stream.push_back(0x01);   // check bits, and "fastest" as the level hint

    {
        BitWriter writer(stream);
        writer.putBits(1, 1);   // one block, and it is the last
        writer.putBits(1, 2);   // compressed with fixed Huffman codes

        // head maps a three-byte hash to the most recent position carrying it;
        // prev chains each position back to the one before it. Together they
        // walk candidate matches newest-first, so the first hit found is also
        // the nearest, which is the cheapest distance to encode.
        std::vector<int32_t> head(kHashSize, -1);
        std::vector<int32_t> prev(data.size(), -1);

        const auto hashAt = [&data](size_t index) {
            return ((static_cast<uint32_t>(data[index]) << 10U) ^
                    (static_cast<uint32_t>(data[index + 1U]) << 5U) ^
                    static_cast<uint32_t>(data[index + 2U])) &
                   static_cast<uint32_t>(kHashSize - 1U);
        };

        const auto insert = [&](size_t index) {
            if (index + kMinMatch <= data.size()) {
                const uint32_t hash = hashAt(index);
                prev[index] = head[hash];
                head[hash] = static_cast<int32_t>(index);
            }
        };

        size_t position = 0;
        while (position < data.size()) {
            size_t bestLength = 0;
            size_t bestDistance = 0;

            if (position + kMinMatch <= data.size()) {
                const size_t maxLength = std::min(kMaxMatch, data.size() - position);
                int32_t candidate = head[hashAt(position)];
                size_t chain = 0;

                while (candidate >= 0 && chain < kMaxChainLength) {
                    const size_t candidateIndex = static_cast<size_t>(candidate);
                    const size_t distance = position - candidateIndex;
                    if (distance > kWindowSize) {
                        break;
                    }

                    size_t length = 0;
                    while (length < maxLength && data[candidateIndex + length] == data[position + length]) {
                        ++length;
                    }

                    if (length > bestLength) {
                        bestLength = length;
                        bestDistance = distance;
                        if (bestLength >= maxLength) {
                            break;
                        }
                    }

                    candidate = prev[candidateIndex];
                    ++chain;
                }
            }

            if (bestLength >= kMinMatch) {
                size_t lengthCode = 0;
                while (lengthCode + 1U < kLengthBase.size() && kLengthBase[lengthCode + 1U] <= bestLength) {
                    ++lengthCode;
                }
                putLiteralOrLengthCode(writer, static_cast<uint32_t>(257U + lengthCode));
                writer.putBits(static_cast<uint32_t>(bestLength - kLengthBase[lengthCode]),
                               kLengthExtra[lengthCode]);

                size_t distanceCode = 0;
                while (distanceCode + 1U < kDistanceBase.size() &&
                       kDistanceBase[distanceCode + 1U] <= bestDistance) {
                    ++distanceCode;
                }
                // Distances use a fixed five-bit code, not the literal tree.
                writer.putCode(static_cast<uint32_t>(distanceCode), 5);
                writer.putBits(static_cast<uint32_t>(bestDistance - kDistanceBase[distanceCode]),
                               kDistanceExtra[distanceCode]);

                for (size_t offset = 0; offset < bestLength; ++offset) {
                    insert(position + offset);
                }
                position += bestLength;
            } else {
                putLiteralOrLengthCode(writer, data[position]);
                insert(position);
                ++position;
            }
        }

        putLiteralOrLengthCode(writer, 256);   // end of block
        writer.flushToByteBoundary();
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

    const std::vector<uint8_t> scanlines = makeFilteredScanlines(width, height, rgbaPixels, rowStrideBytes);
    const std::vector<uint8_t> idat = makeZlibDeflateStream(scanlines);

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
