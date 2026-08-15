#include "renderer/ImageCompare.h"

#include <algorithm>
#include <array>
#include <cstdio>

namespace ve::renderer {

namespace {

constexpr size_t kChannels = 4;

[[nodiscard]] bool spansMatchDimensions(std::span<const uint8_t> pixels, uint32_t width, uint32_t height)
{
    const uint64_t expectedBytes = static_cast<uint64_t>(width) * height * kChannels;
    return expectedBytes > 0 && pixels.size() == expectedBytes;
}

[[nodiscard]] uint32_t pixelChannelDelta(std::span<const uint8_t> actual,
                                         std::span<const uint8_t> expected,
                                         size_t pixelBase,
                                         bool compareAlpha)
{
    const size_t channelCount = compareAlpha ? kChannels : kChannels - 1;
    uint32_t worst = 0;
    for (size_t channel = 0; channel < channelCount; ++channel) {
        const int32_t lhs = actual[pixelBase + channel];
        const int32_t rhs = expected[pixelBase + channel];
        worst = std::max(worst, static_cast<uint32_t>(std::abs(lhs - rhs)));
    }
    return worst;
}

} // namespace

std::string ImageCompareResult::summary() const
{
    if (sizeMismatch) {
        return "size mismatch";
    }

    std::array<char, 192> buffer{};
    const int written = std::snprintf(buffer.data(),
                                      buffer.size(),
                                      "%s: %llu/%llu pixels differ (%.6f%%), max channel delta %u",
                                      match ? "match" : "MISMATCH",
                                      static_cast<unsigned long long>(differingPixels),
                                      static_cast<unsigned long long>(totalPixels),
                                      differingPixelFraction * 100.0,
                                      maxChannelDelta);
    if (written <= 0) {
        return match ? "match" : "MISMATCH";
    }

    std::string text(buffer.data(), static_cast<size_t>(written));
    if (differingPixels > 0) {
        std::array<char, 64> position{};
        const int positionWritten = std::snprintf(
            position.data(), position.size(), ", first at (%u, %u)", firstDifferingX, firstDifferingY);
        if (positionWritten > 0) {
            text.append(position.data(), static_cast<size_t>(positionWritten));
        }
    }
    return text;
}

ImageCompareResult compareRgba8(std::span<const uint8_t> actual,
                                std::span<const uint8_t> expected,
                                uint32_t width,
                                uint32_t height,
                                const ImageCompareOptions& options)
{
    ImageCompareResult result{};

    if (!spansMatchDimensions(actual, width, height) || !spansMatchDimensions(expected, width, height)) {
        result.sizeMismatch = true;
        result.match = false;
        return result;
    }

    result.totalPixels = static_cast<uint64_t>(width) * height;

    bool foundFirst = false;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t pixelBase = (static_cast<size_t>(y) * width + x) * kChannels;
            const uint32_t delta = pixelChannelDelta(actual, expected, pixelBase, options.compareAlpha);

            result.maxChannelDelta = std::max(result.maxChannelDelta, delta);
            if (delta > options.channelTolerance) {
                ++result.differingPixels;
                if (!foundFirst) {
                    result.firstDifferingX = x;
                    result.firstDifferingY = y;
                    foundFirst = true;
                }
            }
        }
    }

    result.differingPixelFraction =
        result.totalPixels == 0 ? 0.0
                                : static_cast<double>(result.differingPixels) / static_cast<double>(result.totalPixels);
    result.match = result.differingPixelFraction <= options.maxDifferingPixelFraction;
    return result;
}

std::vector<uint8_t> makeDifferenceImage(std::span<const uint8_t> actual,
                                         std::span<const uint8_t> expected,
                                         uint32_t width,
                                         uint32_t height,
                                         const ImageCompareOptions& options)
{
    if (!spansMatchDimensions(actual, width, height) || !spansMatchDimensions(expected, width, height)) {
        return {};
    }

    std::vector<uint8_t> difference(actual.size());
    for (uint64_t pixel = 0; pixel < static_cast<uint64_t>(width) * height; ++pixel) {
        const size_t pixelBase = static_cast<size_t>(pixel) * kChannels;
        const uint32_t delta = pixelChannelDelta(actual, expected, pixelBase, options.compareAlpha);

        if (delta > options.channelTolerance) {
            // Amplified so a delta of 1 is still visible in an uploaded artifact
            // rather than an invisible near-black pixel.
            const uint32_t intensity = std::min<uint32_t>(255, 96 + delta * 8);
            difference[pixelBase + 0] = static_cast<uint8_t>(intensity);
            difference[pixelBase + 1] = 0;
            difference[pixelBase + 2] = 0;
        } else {
            // Matching content stays as dim greyscale context so the reader can
            // see *where* in the image the differences are.
            const uint32_t luminance = (static_cast<uint32_t>(actual[pixelBase + 0]) +
                                        actual[pixelBase + 1] + actual[pixelBase + 2]) /
                                       3U;
            const auto dimmed = static_cast<uint8_t>(luminance / 4U);
            difference[pixelBase + 0] = dimmed;
            difference[pixelBase + 1] = dimmed;
            difference[pixelBase + 2] = dimmed;
        }
        difference[pixelBase + 3] = 255;
    }

    return difference;
}

} // namespace ve::renderer
