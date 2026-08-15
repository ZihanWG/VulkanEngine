#pragma once

// RGBA8 image comparison for golden-image regression.
//
// GPU-free and free of any image format: callers decode PNG (or anything else)
// and hand over tightly packed RGBA8. That keeps the comparison policy -- which
// is the part that decides whether CI goes red -- unit testable without a
// decoder, a GPU, or a committed fixture.
//
// Tolerance exists because a golden is compared against a *rebuilt* renderer on
// a *repackaged* driver. Byte equality is the right default to start from, but
// only measurement can say whether it holds; see docs/headless_ci.md.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ve::renderer {

struct ImageCompareOptions {
    // Maximum absolute per-channel difference a pixel may have and still count
    // as matching.
    uint32_t channelTolerance = 0;

    // Fraction of pixels (0..1) allowed to exceed channelTolerance before the
    // comparison fails. Lets a handful of rounding-edge pixels pass without
    // accepting a wholesale image change.
    double maxDifferingPixelFraction = 0.0;

    // Screenshots are opaque; alpha differences are usually swapchain noise
    // rather than a rendering change.
    bool compareAlpha = false;
};

struct ImageCompareResult {
    bool match = false;
    bool sizeMismatch = false;

    uint32_t maxChannelDelta = 0;
    uint64_t differingPixels = 0;
    uint64_t totalPixels = 0;
    double differingPixelFraction = 0.0;

    // Raster position of the first pixel that exceeded the tolerance. Only
    // meaningful when differingPixels > 0.
    uint32_t firstDifferingX = 0;
    uint32_t firstDifferingY = 0;

    [[nodiscard]] std::string summary() const;
};

// Both spans must hold width*height*4 tightly packed bytes. A size mismatch
// (including a span whose length disagrees with width/height) returns
// sizeMismatch, never a partial comparison.
[[nodiscard]] ImageCompareResult compareRgba8(std::span<const uint8_t> actual,
                                              std::span<const uint8_t> expected,
                                              uint32_t width,
                                              uint32_t height,
                                              const ImageCompareOptions& options = {});

// A human-readable diff: matching pixels darkened toward grey, differing pixels
// pushed to red and scaled by how far off they are, so a small delta is still
// visible in an uploaded artifact. Returns empty on a size mismatch.
[[nodiscard]] std::vector<uint8_t> makeDifferenceImage(std::span<const uint8_t> actual,
                                                       std::span<const uint8_t> expected,
                                                       uint32_t width,
                                                       uint32_t height,
                                                       const ImageCompareOptions& options = {});

} // namespace ve::renderer
