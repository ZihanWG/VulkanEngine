// Golden-image comparison tool.
//
//   compare_images <actual.png> <expected.png> [options]
//     --channel-tolerance N     max per-channel delta a pixel may have (default 0)
//     --max-differing-fraction F  fraction of pixels allowed to exceed it (default 0)
//     --compare-alpha           include the alpha channel (off by default)
//     --diff-output PATH        write an amplified difference PNG on mismatch
//
// Exit codes: 0 match, 1 mismatch, 2 usage/IO error. The comparison policy
// itself lives in renderer/ImageCompare (GPU-free and unit tested); this file is
// only decoding, argument parsing, and reporting.
//
// It owns its own stb_image implementation because the engine's lives in
// rhi/VulkanTexture.cpp, and this tool deliberately does not link Vulkan.

#include "core/PngWriter.h"
#include "renderer/ImageCompare.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct DecodedPng {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
};

struct StbFree {
    void operator()(stbi_uc* pixels) const { stbi_image_free(pixels); }
};

bool decodePng(const std::filesystem::path& path, DecodedPng& out)
{
    int channelsInFile = 0;
    // Forced to RGBA8 so the comparison never has to care what the encoder chose.
    const std::unique_ptr<stbi_uc, StbFree> pixels(
        stbi_load(path.string().c_str(), &out.width, &out.height, &channelsInFile, 4));
    if (!pixels || out.width <= 0 || out.height <= 0) {
        std::fprintf(stderr, "compare_images: failed to read %s: %s\n", path.string().c_str(), stbi_failure_reason());
        return false;
    }

    const size_t byteCount = static_cast<size_t>(out.width) * static_cast<size_t>(out.height) * 4U;
    out.pixels.assign(pixels.get(), pixels.get() + byteCount);
    return true;
}

bool parseDouble(std::string_view text, double& value)
{
    // from_chars for double is not universally available in libstdc++/libc++ of
    // the vintages this project builds on, so strtod with explicit end checking.
    const std::string owned(text);
    char* end = nullptr;
    const double parsed = std::strtod(owned.c_str(), &end);
    if (end == owned.c_str() || (end != nullptr && *end != '\0')) {
        return false;
    }
    value = parsed;
    return true;
}

[[noreturn]] void usage()
{
    std::fprintf(stderr,
                 "usage: compare_images <actual.png> <expected.png>\n"
                 "         [--channel-tolerance N] [--max-differing-fraction F]\n"
                 "         [--compare-alpha] [--diff-output PATH]\n");
    std::exit(2);
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> positional;
    ve::renderer::ImageCompareOptions options{};
    std::filesystem::path diffOutput;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);

        if (argument == "--compare-alpha") {
            options.compareAlpha = true;
        } else if (argument == "--channel-tolerance") {
            if (index + 1 >= argc) {
                usage();
            }
            const std::string_view value(argv[++index]);
            uint32_t tolerance = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), tolerance);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                usage();
            }
            options.channelTolerance = tolerance;
        } else if (argument == "--max-differing-fraction") {
            if (index + 1 >= argc) {
                usage();
            }
            double fraction = 0.0;
            if (!parseDouble(argv[++index], fraction) || fraction < 0.0 || fraction > 1.0) {
                usage();
            }
            options.maxDifferingPixelFraction = fraction;
        } else if (argument == "--diff-output") {
            if (index + 1 >= argc) {
                usage();
            }
            diffOutput = argv[++index];
        } else if (argument.rfind("--", 0) == 0) {
            std::fprintf(stderr, "compare_images: unknown option %s\n", std::string(argument).c_str());
            usage();
        } else {
            positional.emplace_back(argument);
        }
    }

    if (positional.size() != 2) {
        usage();
    }

    DecodedPng actual;
    DecodedPng expected;
    if (!decodePng(positional[0], actual) || !decodePng(positional[1], expected)) {
        return 2;
    }

    if (actual.width != expected.width || actual.height != expected.height) {
        std::fprintf(stderr,
                     "compare_images: MISMATCH size: actual %dx%d, expected %dx%d\n",
                     actual.width,
                     actual.height,
                     expected.width,
                     expected.height);
        return 1;
    }

    const auto width = static_cast<uint32_t>(actual.width);
    const auto height = static_cast<uint32_t>(actual.height);

    const ve::renderer::ImageCompareResult result =
        ve::renderer::compareRgba8(actual.pixels, expected.pixels, width, height, options);

    std::printf("compare_images: %s\n", result.summary().c_str());

    if (!result.match && !diffOutput.empty()) {
        const std::vector<uint8_t> difference =
            ve::renderer::makeDifferenceImage(actual.pixels, expected.pixels, width, height, options);
        if (!difference.empty()) {
            try {
                if (diffOutput.has_parent_path()) {
                    std::filesystem::create_directories(diffOutput.parent_path());
                }
                ve::writePngRgba8(diffOutput, width, height, difference, width * 4U);
                std::printf("compare_images: wrote difference image to %s\n", diffOutput.string().c_str());
            } catch (const std::exception& error) {
                std::fprintf(stderr, "compare_images: failed to write difference image: %s\n", error.what());
            }
        }
    }

    return result.match ? 0 : 1;
}
