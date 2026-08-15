#include "renderer/ImageCompare.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using ve::renderer::compareRgba8;
using ve::renderer::ImageCompareOptions;
using ve::renderer::ImageCompareResult;
using ve::renderer::makeDifferenceImage;

namespace {

constexpr uint32_t kWidth = 4;
constexpr uint32_t kHeight = 4;
constexpr size_t kPixelCount = kWidth * kHeight;

std::vector<uint8_t> solidImage(uint8_t value = 128)
{
    std::vector<uint8_t> pixels(kPixelCount * 4, value);
    for (size_t pixel = 0; pixel < kPixelCount; ++pixel) {
        pixels[pixel * 4 + 3] = 255;
    }
    return pixels;
}

void setPixel(std::vector<uint8_t>& pixels, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
    const size_t base = (static_cast<size_t>(y) * kWidth + x) * 4;
    pixels[base + 0] = r;
    pixels[base + 1] = g;
    pixels[base + 2] = b;
    pixels[base + 3] = a;
}

} // namespace

TEST_CASE("Identical images match with zero tolerance")
{
    const std::vector<uint8_t> image = solidImage();

    const ImageCompareResult result = compareRgba8(image, image, kWidth, kHeight);

    REQUIRE(result.match);
    REQUIRE_FALSE(result.sizeMismatch);
    REQUIRE(result.differingPixels == 0);
    REQUIRE(result.maxChannelDelta == 0);
    REQUIRE(result.totalPixels == kPixelCount);
}

TEST_CASE("A size mismatch is reported rather than partially compared")
{
    const std::vector<uint8_t> image = solidImage();
    const std::vector<uint8_t> shorter(image.size() - 4, 0);

    const ImageCompareResult result = compareRgba8(shorter, image, kWidth, kHeight);

    REQUIRE(result.sizeMismatch);
    REQUIRE_FALSE(result.match);
}

TEST_CASE("A single changed pixel fails at zero tolerance")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 2, 1, 200, 128, 128);

    const ImageCompareResult result = compareRgba8(actual, expected, kWidth, kHeight);

    REQUIRE_FALSE(result.match);
    REQUIRE(result.differingPixels == 1);
    REQUIRE(result.maxChannelDelta == 72);
    REQUIRE(result.firstDifferingX == 2);
    REQUIRE(result.firstDifferingY == 1);
}

TEST_CASE("Channel tolerance absorbs a small delta")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 0, 0, 130, 128, 128);

    ImageCompareOptions options{};
    options.channelTolerance = 2;

    const ImageCompareResult result = compareRgba8(actual, expected, kWidth, kHeight, options);

    REQUIRE(result.match);
    REQUIRE(result.differingPixels == 0);
    // Still reported, so a drift that stays under tolerance is observable rather
    // than invisible.
    REQUIRE(result.maxChannelDelta == 2);
}

TEST_CASE("A delta one past the tolerance still fails")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 0, 0, 131, 128, 128);

    ImageCompareOptions options{};
    options.channelTolerance = 2;

    REQUIRE_FALSE(compareRgba8(actual, expected, kWidth, kHeight, options).match);
}

TEST_CASE("The differing-pixel fraction gates on the whole image, not one pixel")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 0, 0, 255, 255, 255);
    setPixel(actual, 1, 0, 255, 255, 255);

    ImageCompareOptions options{};

    // 2 of 16 pixels = 0.125. A budget just under that must fail...
    options.maxDifferingPixelFraction = 0.124;
    REQUIRE_FALSE(compareRgba8(actual, expected, kWidth, kHeight, options).match);

    // ...and a budget exactly at it must pass, so the boundary is not ambiguous.
    options.maxDifferingPixelFraction = 0.125;
    const ImageCompareResult allowed = compareRgba8(actual, expected, kWidth, kHeight, options);
    REQUIRE(allowed.match);
    REQUIRE(allowed.differingPixels == 2);
}

TEST_CASE("Alpha is ignored by default and honoured on request")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 3, 3, 128, 128, 128, 0);

    REQUIRE(compareRgba8(actual, expected, kWidth, kHeight).match);

    ImageCompareOptions options{};
    options.compareAlpha = true;
    REQUIRE_FALSE(compareRgba8(actual, expected, kWidth, kHeight, options).match);
}

TEST_CASE("The difference image marks differing pixels red and dims the rest")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 1, 1, 255, 128, 128);

    const std::vector<uint8_t> difference = makeDifferenceImage(actual, expected, kWidth, kHeight);

    REQUIRE(difference.size() == actual.size());

    const size_t changed = (1 * kWidth + 1) * 4;
    REQUIRE(difference[changed + 0] == 255);
    REQUIRE(difference[changed + 1] == 0);
    REQUIRE(difference[changed + 2] == 0);

    // An unchanged pixel keeps grey context rather than going black, so the
    // reader can see where in the frame the difference sits.
    const size_t unchanged = 0;
    REQUIRE(difference[unchanged + 0] == difference[unchanged + 1]);
    REQUIRE(difference[unchanged + 1] == difference[unchanged + 2]);
    REQUIRE(difference[unchanged + 0] > 0);
    REQUIRE(difference[unchanged + 0] < 128);
}

TEST_CASE("The difference image is empty on a size mismatch")
{
    const std::vector<uint8_t> image = solidImage();
    const std::vector<uint8_t> shorter(image.size() - 4, 0);

    REQUIRE(makeDifferenceImage(shorter, image, kWidth, kHeight).empty());
}

TEST_CASE("The summary names the outcome and the worst delta")
{
    const std::vector<uint8_t> expected = solidImage();
    std::vector<uint8_t> actual = expected;
    setPixel(actual, 0, 0, 138, 128, 128);

    const std::string summary = compareRgba8(actual, expected, kWidth, kHeight).summary();

    REQUIRE(summary.find("MISMATCH") != std::string::npos);
    REQUIRE(summary.find("max channel delta 10") != std::string::npos);
    REQUIRE(summary.find("(0, 0)") != std::string::npos);
}
