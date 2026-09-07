// The PNG writer hand-rolls its own deflate, so these decode what it produces
// with a real decoder (stb_image, already linked here) rather than inspecting
// the bytes it emitted. A compressor that agrees only with itself is the whole
// failure this guards: the previous encoder emitted valid stored blocks and no
// compression at all, and nothing noticed until a tracked golden turned up ten
// times larger than it needed to be.

#include "core/PngWriter.h"

#include <catch2/catch_test_macros.hpp>

#include "stb_image.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

using ve::writePngRgba8;

namespace {

// A scratch path that cleans itself up, so a failing assertion cannot leave the
// temp directory filling with images across runs.
class TemporaryPng
{
public:
    explicit TemporaryPng(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / ("ve_png_test_" + name + ".png"))
    {
        std::filesystem::remove(path_);
    }

    ~TemporaryPng() { std::filesystem::remove(path_); }

    TemporaryPng(const TemporaryPng&) = delete;
    TemporaryPng& operator=(const TemporaryPng&) = delete;

    const std::filesystem::path& path() const { return path_; }

    std::vector<uint8_t> bytes() const
    {
        std::ifstream input(path_, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                                    std::istreambuf_iterator<char>());
    }

private:
    std::filesystem::path path_;
};

// Decoded RGBA8, or an empty vector when stb_image rejected the file.
std::vector<uint8_t> decodeRgba8(const std::vector<uint8_t>& png, int& width, int& height)
{
    int channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(png.data(), static_cast<int>(png.size()),
                                            &width, &height, &channels, 4);
    if (pixels == nullptr) {
        return {};
    }

    std::vector<uint8_t> decoded(pixels, pixels + static_cast<size_t>(width) * height * 4);
    stbi_image_free(pixels);
    return decoded;
}

// Gradients plus a deterministic speckle: smooth enough that the row filters
// have something to predict, noisy enough that it cannot compress to nothing.
std::vector<uint8_t> testImage(uint32_t width, uint32_t height, uint32_t seed = 7)
{
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    std::mt19937 rng(seed);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t base = (static_cast<size_t>(y) * width + x) * 4;
            pixels[base + 0] = static_cast<uint8_t>((x * 255U) / (width > 1 ? width - 1 : 1));
            pixels[base + 1] = static_cast<uint8_t>((y * 255U) / (height > 1 ? height - 1 : 1));
            pixels[base + 2] = static_cast<uint8_t>(rng() & 0x1fU);
            pixels[base + 3] = 255;
        }
    }
    return pixels;
}

} // namespace

TEST_CASE("PNG round-trips through a real decoder")
{
    constexpr uint32_t kWidth = 97;    // deliberately not a power of two
    constexpr uint32_t kHeight = 53;

    const std::vector<uint8_t> source = testImage(kWidth, kHeight);
    const TemporaryPng file("roundtrip");
    writePngRgba8(file.path(), kWidth, kHeight, source, kWidth * 4U);

    int width = 0;
    int height = 0;
    const std::vector<uint8_t> decoded = decodeRgba8(file.bytes(), width, height);

    REQUIRE(width == static_cast<int>(kWidth));
    REQUIRE(height == static_cast<int>(kHeight));
    REQUIRE(decoded.size() == source.size());
    CHECK(decoded == source);
}

TEST_CASE("PNG round-trips every byte value")
{
    // 256x256 covers all 65536 (left, above) neighbour pairs the row filters can
    // face, which is what catches a wrong Paeth tie-break or an unsigned
    // subtraction that should have wrapped.
    constexpr uint32_t kSide = 256;

    std::vector<uint8_t> source(static_cast<size_t>(kSide) * kSide * 4);
    for (uint32_t y = 0; y < kSide; ++y) {
        for (uint32_t x = 0; x < kSide; ++x) {
            const size_t base = (static_cast<size_t>(y) * kSide + x) * 4;
            source[base + 0] = static_cast<uint8_t>(x);
            source[base + 1] = static_cast<uint8_t>(y);
            source[base + 2] = static_cast<uint8_t>(x ^ y);
            source[base + 3] = static_cast<uint8_t>(255U - ((x + y) & 0xffU));
        }
    }

    const TemporaryPng file("allbytes");
    writePngRgba8(file.path(), kSide, kSide, source, kSide * 4U);

    int width = 0;
    int height = 0;
    const std::vector<uint8_t> decoded = decodeRgba8(file.bytes(), width, height);

    REQUIRE(width == static_cast<int>(kSide));
    REQUIRE(height == static_cast<int>(kSide));
    CHECK(decoded == source);
}

TEST_CASE("PNG honours a padded row stride")
{
    constexpr uint32_t kWidth = 40;
    constexpr uint32_t kHeight = 12;
    constexpr uint32_t kStride = kWidth * 4U + 37U;   // readback rows are padded

    const std::vector<uint8_t> tight = testImage(kWidth, kHeight, 11);

    std::vector<uint8_t> padded(static_cast<size_t>(kStride) * kHeight, 0xab);
    for (uint32_t y = 0; y < kHeight; ++y) {
        std::copy_n(tight.begin() + static_cast<ptrdiff_t>(y) * kWidth * 4,
                    kWidth * 4,
                    padded.begin() + static_cast<ptrdiff_t>(y) * kStride);
    }

    const TemporaryPng file("stride");
    writePngRgba8(file.path(), kWidth, kHeight, padded, kStride);

    int width = 0;
    int height = 0;
    const std::vector<uint8_t> decoded = decodeRgba8(file.bytes(), width, height);

    REQUIRE(decoded.size() == tight.size());
    CHECK(decoded == tight);   // the padding must not reach the file
}

TEST_CASE("PNG actually compresses")
{
    // A flat image is the clearest case: stored blocks would spend a byte per
    // channel regardless, so anything near the raw size means the compressor
    // is not compressing.
    constexpr uint32_t kSide = 256;
    constexpr size_t kRawBytes = static_cast<size_t>(kSide) * kSide * 4;

    const std::vector<uint8_t> flat(kRawBytes, 0x80);
    const TemporaryPng file("flat");
    writePngRgba8(file.path(), kSide, kSide, flat, kSide * 4U);

    const size_t written = file.bytes().size();
    CHECK(written < kRawBytes / 100);

    int width = 0;
    int height = 0;
    CHECK(decodeRgba8(file.bytes(), width, height) == flat);
}

TEST_CASE("PNG writer rejects invalid extents")
{
    const std::vector<uint8_t> pixels(64 * 4, 0);
    const TemporaryPng file("invalid");

    CHECK_THROWS(writePngRgba8(file.path(), 0, 4, pixels, 16));
    CHECK_THROWS(writePngRgba8(file.path(), 4, 0, pixels, 16));
    CHECK_THROWS(writePngRgba8(file.path(), 4, 4, pixels, 8));           // stride < width
    CHECK_THROWS(writePngRgba8(file.path(), 64, 64, pixels, 64 * 4U));   // span too small
}
