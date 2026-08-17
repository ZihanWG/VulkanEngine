// Offline texture cook: one image in, one block-compressed KTX2 with baked mips
// out. See docs/asset_system.md.
//
//   vecook <input.png|jpg> -o <output.ktx2> --usage <slot> [options]
//     --usage SLOT   base-color | emissive | normal | metallic-roughness | occlusion
//     --no-mips      write only the base level
//     --force        re-encode even when the output is newer than the input
//     --threads N    worker threads (0 = hardware default)
//     --verify       read the written file back, decode every level, report PSNR
//
// Exit codes: 0 cooked or already up to date, 1 cook failure, 2 usage/IO error.
//
// The policy this tool applies -- which format a slot cooks to, how mips are
// filtered, how a partial 4x4 block is padded, what block order a level uses --
// all lives in assets/TextureCook and assets/Ktx2, where it is unit tested
// without a device. This file is argument parsing, decode, threading, and
// reporting. It owns its own stb_image implementation because the engine's lives
// in rhi/VulkanTexture.cpp and this tool deliberately does not link Vulkan.

#include "assets/Ktx2.h"
#include "assets/TextureCook.h"
#include "core/JobSystem.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <bc7decomp.h>
#include <bc7enc.h>
#include <rgbcx.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ve::assets::TextureUsage;

// Structural-corruption floor for --verify, not a quality bar. BC7 on real
// content sits in the 30-45 dB range; a wrong block or level order collapses to
// single digits.
constexpr double kMinimumVerifyPsnrDb = 20.0;

struct StbFree {
    void operator()(stbi_uc* pixels) const { stbi_image_free(pixels); }
};

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    TextureUsage usage = TextureUsage::BaseColor;
    bool usageGiven = false;
    bool generateMips = true;
    bool force = false;
    bool verify = false;
    size_t threadCount = 0;
};

[[noreturn]] void usage()
{
    std::fprintf(stderr,
                 "usage: vecook <input.png> -o <output.ktx2> --usage <slot>\n"
                 "         slot: base-color | emissive | normal | metallic-roughness | occlusion\n"
                 "         [--no-mips] [--force] [--threads N] [--verify]\n");
    std::exit(2);
}

Options parseArguments(int argc, char** argv)
{
    Options options{};
    std::vector<std::string> positional;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> std::string_view {
            if (index + 1 >= argc) {
                usage();
            }
            return std::string_view(argv[++index]);
        };

        if (argument == "-o" || argument == "--output") {
            options.output = std::filesystem::path(next());
        } else if (argument == "--usage") {
            // The spelling lives in assets/TextureCook so this tool, the cook
            // driver, and the runtime's sidecar lookup cannot drift apart.
            if (!ve::assets::parseTextureUsage(next(), options.usage)) {
                usage();
            }
            options.usageGiven = true;
        } else if (argument == "--no-mips") {
            options.generateMips = false;
        } else if (argument == "--force") {
            options.force = true;
        } else if (argument == "--verify") {
            options.verify = true;
        } else if (argument == "--threads") {
            const std::string_view value = next();
            unsigned int threads = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), threads);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                usage();
            }
            options.threadCount = threads;
        } else if (argument.rfind("--", 0) == 0 || argument == "-h") {
            std::fprintf(stderr, "vecook: unknown option %s\n", std::string(argument).c_str());
            usage();
        } else {
            positional.emplace_back(argument);
        }
    }

    // The slot is required rather than guessed from the file name: cooking a
    // roughness map as sRGB is silent corruption, not a build error.
    if (positional.size() != 1 || options.output.empty() || !options.usageGiven) {
        usage();
    }

    options.input = std::filesystem::path(positional.front());
    return options;
}

// Mtime-based skip. A full Sponza cook is ~40 seconds of encoding single
// threaded, so re-encoding on every build is not acceptable; comparing write
// times is enough for a tool that is run by hand and by the build.
bool outputIsUpToDate(const Options& options)
{
    std::error_code error;
    if (!std::filesystem::exists(options.output, error) || error) {
        return false;
    }

    const auto outputTime = std::filesystem::last_write_time(options.output, error);
    if (error) {
        return false;
    }

    const auto inputTime = std::filesystem::last_write_time(options.input, error);
    if (error) {
        return false;
    }

    return outputTime >= inputTime;
}

struct DecodedImage {
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
};

bool decodeImage(const std::filesystem::path& path, DecodedImage& out)
{
    int width = 0;
    int height = 0;
    int channelsInFile = 0;
    // Forced to RGBA8: the cook's block extraction and mip filter both work on
    // four channels regardless of what the source file stored.
    const std::unique_ptr<stbi_uc, StbFree> pixels(
        stbi_load(path.string().c_str(), &width, &height, &channelsInFile, 4));
    if (!pixels || width <= 0 || height <= 0) {
        std::fprintf(stderr, "vecook: failed to read %s: %s\n", path.string().c_str(), stbi_failure_reason());
        return false;
    }

    out.width = static_cast<uint32_t>(width);
    out.height = static_cast<uint32_t>(height);
    const size_t byteCount = static_cast<size_t>(out.width) * out.height * 4U;
    out.pixels.assign(pixels.get(), pixels.get() + byteCount);
    return true;
}

// Both encoders keep global tables that must exist before any worker calls them.
ve::assets::BlockEncodeFn makeBlockEncoder(uint32_t vkFormat, bool perceptual)
{
    if (vkFormat == ve::assets::kVkFormatBc5UnormBlock) {
        rgbcx::init();
        // Channels 0 and 1 of the RGBA block: a tangent-space normal's X and Y.
        return [](std::span<const uint8_t, 64> rgbaBlock, std::span<uint8_t> blockOut) {
            rgbcx::encode_bc5(blockOut.data(), rgbaBlock.data(), 0, 1, 4);
        };
    }

    bc7enc_compress_block_init();

    bc7enc_compress_block_params params{};
    bc7enc_compress_block_params_init(&params);
    // Perceptual (YCbCr) error is right for colour and wrong for data: weighting
    // a roughness map as if its channels were red, green and blue would spend
    // precision on a channel ordering that means nothing.
    if (perceptual) {
        bc7enc_compress_block_params_init_perceptual_weights(&params);
    } else {
        bc7enc_compress_block_params_init_linear_weights(&params);
    }

    return [params](std::span<const uint8_t, 64> rgbaBlock, std::span<uint8_t> blockOut) {
        bc7enc_compress_block(blockOut.data(), rgbaBlock.data(), &params);
    };
}

// --verify exists because there is no KTX2 validator on this machine, and a
// malformed KTX2 fails silently rather than loudly. Reading the file back off
// disk and decoding every level checks, end to end and without a GPU, the three
// things the unit tests cannot: that the bytes actually reached the file, that
// the level index points at the level it claims, and that blocks were written in
// the order a decoder expects. A scrambled block or level order collapses PSNR;
// a correct lossy encode does not.
struct VerifyResult {
    double worstPsnrDb = 0.0;
    uint32_t worstLevel = 0;
    uint32_t levelsChecked = 0;
};

std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to reopen " + path.string() + " for verification");
    }

    const std::streamsize size = input.tellg();
    input.seekg(0);

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) {
        throw std::runtime_error("failed to read back " + path.string());
    }

    return bytes;
}

// Decodes one level's blocks back into a tightly packed RGBA8 image. Blocks
// covering the padded edge are clipped here, which is the mirror of the clamping
// extractRgba8Block does on the way in.
std::vector<uint8_t> decodeLevel(uint32_t vkFormat,
                                 std::span<const uint8_t> levelBytes,
                                 uint32_t width,
                                 uint32_t height,
                                 uint32_t blockSizeBytes)
{
    const uint32_t blockColumns = ve::assets::rgba8BlockColumnCount(width);
    const uint32_t blockRows = ve::assets::rgba8BlockRowCount(height);

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4U);
    std::array<uint8_t, 64> block{};

    for (uint32_t blockY = 0; blockY < blockRows; ++blockY) {
        for (uint32_t blockX = 0; blockX < blockColumns; ++blockX) {
            const size_t offset = (static_cast<size_t>(blockY) * blockColumns + blockX) * blockSizeBytes;
            const uint8_t* source = levelBytes.data() + offset;

            if (vkFormat == ve::assets::kVkFormatBc5UnormBlock) {
                block.fill(255);
                rgbcx::unpack_bc5(source, block.data(), 0, 1, 4);
            } else {
                bc7decomp::unpack_bc7(source, reinterpret_cast<bc7decomp::color_rgba*>(block.data()));
            }

            for (uint32_t row = 0; row < 4U; ++row) {
                const uint32_t y = blockY * 4U + row;
                if (y >= height) {
                    break;
                }
                for (uint32_t column = 0; column < 4U; ++column) {
                    const uint32_t x = blockX * 4U + column;
                    if (x >= width) {
                        break;
                    }

                    const size_t destination = (static_cast<size_t>(y) * width + x) * 4U;
                    const size_t texel = (static_cast<size_t>(row) * 4U + column) * 4U;
                    for (uint32_t channel = 0; channel < 4U; ++channel) {
                        pixels[destination + channel] = block[texel + channel];
                    }
                }
            }
        }
    }

    return pixels;
}

VerifyResult verifyCookedFile(const std::filesystem::path& path,
                              uint32_t vkFormat,
                              uint32_t blockSizeBytes,
                              const std::vector<std::vector<uint8_t>>& sourceChain)
{
    const std::vector<uint8_t> fileBytes = readFile(path);
    const ve::assets::Ktx2Info parsed = ve::assets::parseKtx2(fileBytes);

    if (parsed.vkFormat != vkFormat || parsed.levels.size() != sourceChain.size()) {
        throw std::runtime_error("verification: the file read back does not describe what was cooked");
    }

    // A normal map carries only two channels; scoring the other two would compare
    // against whatever the decoder left there.
    const uint32_t channelCount = vkFormat == ve::assets::kVkFormatBc5UnormBlock ? 2U : 4U;

    VerifyResult result{};
    result.worstPsnrDb = std::numeric_limits<double>::infinity();
    result.levelsChecked = static_cast<uint32_t>(parsed.levels.size());

    for (uint32_t level = 0; level < parsed.levels.size(); ++level) {
        const uint32_t levelWidth = std::max(1U, parsed.pixelWidth >> level);
        const uint32_t levelHeight = std::max(1U, parsed.pixelHeight >> level);

        const std::span<const uint8_t> levelBytes(fileBytes.data() + parsed.levels[level].byteOffset,
                                                  static_cast<size_t>(parsed.levels[level].byteLength));
        const std::vector<uint8_t> decoded = decodeLevel(vkFormat, levelBytes, levelWidth, levelHeight, blockSizeBytes);
        const std::vector<uint8_t>& expected = sourceChain[level];

        double squaredError = 0.0;
        size_t samples = 0;
        for (size_t texel = 0; texel < decoded.size() / 4U; ++texel) {
            for (uint32_t channel = 0; channel < channelCount; ++channel) {
                const double delta = static_cast<double>(decoded[texel * 4U + channel])
                                     - static_cast<double>(expected[texel * 4U + channel]);
                squaredError += delta * delta;
                ++samples;
            }
        }

        // A level that came back bit-exact (a flat block, say) has no error to
        // score; leave the worst-case untouched rather than reporting infinity.
        if (samples == 0 || squaredError == 0.0) {
            continue;
        }

        const double meanSquaredError = squaredError / static_cast<double>(samples);
        const double psnr = 10.0 * std::log10((255.0 * 255.0) / meanSquaredError);
        if (psnr < result.worstPsnrDb) {
            result.worstPsnrDb = psnr;
            result.worstLevel = level;
        }
    }

    return result;
}

std::string formatMiB(size_t bytes)
{
    char text[64];
    std::snprintf(text, sizeof(text), "%.2f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return text;
}

} // namespace

int main(int argc, char** argv)
{
    const Options options = parseArguments(argc, argv);

    std::error_code existsError;
    if (!std::filesystem::exists(options.input, existsError) || existsError) {
        std::fprintf(stderr, "vecook: input does not exist: %s\n", options.input.string().c_str());
        return 2;
    }

    if (!options.force && outputIsUpToDate(options)) {
        std::printf("vecook: %s is up to date\n", options.output.string().c_str());
        return 0;
    }

    DecodedImage source;
    if (!decodeImage(options.input, source)) {
        return 2;
    }

    const uint32_t vkFormat = ve::assets::cookedFormatForUsage(options.usage);
    const ve::assets::Ktx2FormatInfo formatInfo = ve::assets::ktx2FormatInfo(vkFormat);
    const bool srgb = ve::assets::textureUsageIsSrgb(options.usage);

    const auto started = std::chrono::steady_clock::now();

    try {
        std::vector<std::vector<uint8_t>> mipChain =
            ve::assets::generateMipChainRgba8(source.pixels, source.width, source.height, srgb);
        if (!options.generateMips) {
            mipChain.resize(1);
        }

        const ve::assets::BlockEncodeFn encodeBlock = makeBlockEncoder(vkFormat, srgb);
        ve::JobSystem jobs(options.threadCount);

        ve::assets::Ktx2Image cooked{};
        cooked.vkFormat = vkFormat;
        cooked.pixelWidth = source.width;
        cooked.pixelHeight = source.height;

        for (size_t level = 0; level < mipChain.size(); ++level) {
            const uint32_t levelWidth = std::max(1U, source.width >> level);
            const uint32_t levelHeight = std::max(1U, source.height >> level);
            const uint32_t blockRows = ve::assets::rgba8BlockRowCount(levelHeight);

            std::vector<uint8_t> encoded(
                ve::assets::ktx2LevelSizeBytes(vkFormat, source.width, source.height, static_cast<uint32_t>(level)));
            const std::span<const uint8_t> levelPixels(mipChain[level]);
            const std::span<uint8_t> levelOut(encoded);

            // Disjoint block-row ranges write to disjoint bytes of `encoded`, so
            // the chunks need no synchronization. One block row is a small job,
            // hence the minimum chunk: the tail levels are a handful of blocks
            // and are cheaper to encode inline than to hand to the pool.
            jobs.parallelFor(blockRows, 8, [&](size_t begin, size_t end) {
                ve::assets::encodeRgba8BlockRows(levelPixels,
                                                 levelWidth,
                                                 levelHeight,
                                                 formatInfo.blockSizeBytes,
                                                 static_cast<uint32_t>(begin),
                                                 static_cast<uint32_t>(end),
                                                 encodeBlock,
                                                 levelOut);
            });

            cooked.levels.push_back(std::move(encoded));
        }

        ve::assets::writeKtx2File(options.output, cooked);

        const auto elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();

        size_t cookedBytes = 0;
        for (const std::vector<uint8_t>& level : cooked.levels) {
            cookedBytes += level.size();
        }
        // The source comparison is against the decoded RGBA8 the runtime would
        // otherwise upload, not against the PNG on disk: the PNG is already
        // entropy coded, and it is the upload the cook exists to shrink.
        const size_t sourceBytes = static_cast<size_t>(source.width) * source.height * 4U;

        std::printf("vecook: %s -> %s\n", options.input.string().c_str(), options.output.string().c_str());
        std::printf("vecook: %ux%u %s, %zu level%s, %lld ms\n",
                    source.width,
                    source.height,
                    std::string(ve::assets::textureUsageName(options.usage)).c_str(),
                    cooked.levels.size(),
                    cooked.levels.size() == 1 ? "" : "s",
                    static_cast<long long>(elapsedMs));
        std::printf("vecook: %s uploaded as RGBA8 base level -> %s cooked with mips (%.2fx)\n",
                    formatMiB(sourceBytes).c_str(),
                    formatMiB(cookedBytes).c_str(),
                    static_cast<double>(sourceBytes) / static_cast<double>(cookedBytes));

        if (options.verify) {
            const VerifyResult verified =
                verifyCookedFile(options.output, vkFormat, formatInfo.blockSizeBytes, mipChain);

            if (std::isinf(verified.worstPsnrDb)) {
                std::printf("vecook: verified %u level%s, every one decoded bit-exact\n",
                            verified.levelsChecked,
                            verified.levelsChecked == 1 ? "" : "s");
            } else {
                std::printf("vecook: verified %u level%s, worst %.1f dB PSNR at level %u\n",
                            verified.levelsChecked,
                            verified.levelsChecked == 1 ? "" : "s",
                            verified.worstPsnrDb,
                            verified.worstLevel);
            }

            // A correct lossy encode lands well above this; a scrambled block or
            // level order lands far below it. The gate is deliberately loose --
            // it is checking for structural corruption, not judging quality.
            if (verified.worstPsnrDb < kMinimumVerifyPsnrDb) {
                std::fprintf(stderr,
                             "vecook: verification FAILED, %.1f dB is too low to be quantization error\n",
                             verified.worstPsnrDb);
                return 1;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vecook: %s\n", error.what());
        return 1;
    }

    return 0;
}
