#include "renderer/AssetLoadStats.h"

#include <catch2/catch_test_macros.hpp>

#include <thread>
#include <vector>

using ve::renderer::AssetLoadStats;
using ve::renderer::AssetLoadStatsRecorder;
using ve::renderer::formatMebibytes;
using ve::renderer::formatReport;
using ve::renderer::sortedByCost;
using ve::renderer::summarizeTextures;
using ve::renderer::TextureLoadRecord;
using ve::renderer::TextureTotals;

namespace {

constexpr uint64_t kMebibyte = 1024ull * 1024ull;

TextureLoadRecord makeRecord(std::string name, uint64_t bytes, bool compressed = false)
{
    TextureLoadRecord record{};
    record.debugName = std::move(name);
    record.formatName = compressed ? "VK_FORMAT_BC7_SRGB_BLOCK" : "VK_FORMAT_R8G8B8A8_SRGB";
    record.width = 1024;
    record.height = 1024;
    record.mipLevels = 11;
    record.deviceBytes = bytes;
    record.blockCompressed = compressed;
    return record;
}

// Restores the recorder to its default (disabled, empty) state so test order
// cannot leak through the process-wide sink.
struct RecorderGuard {
    RecorderGuard()
    {
        AssetLoadStatsRecorder::reset();
        AssetLoadStatsRecorder::setEnabled(false);
    }
    ~RecorderGuard()
    {
        AssetLoadStatsRecorder::reset();
        AssetLoadStatsRecorder::setEnabled(false);
    }
};

} // namespace

TEST_CASE("Texture totals sum device bytes and count compressed textures")
{
    const std::vector<TextureLoadRecord> textures = {
        makeRecord("base", 4 * kMebibyte),
        makeRecord("normal", 2 * kMebibyte, true),
        makeRecord("roughness", 1 * kMebibyte, true),
    };

    const TextureTotals totals = summarizeTextures(textures);

    REQUIRE(totals.count == 3);
    REQUIRE(totals.deviceBytes == 7 * kMebibyte);
    REQUIRE(totals.blockCompressedCount == 2);
    REQUIRE(totals.blockCompressedBytes == 3 * kMebibyte);
    REQUIRE(totals.largestBytes == 4 * kMebibyte);
    REQUIRE(totals.largestName == "base");
}

TEST_CASE("Empty texture set summarizes to zero without a largest entry")
{
    const TextureTotals totals = summarizeTextures({});

    REQUIRE(totals.count == 0);
    REQUIRE(totals.deviceBytes == 0);
    REQUIRE(totals.largestBytes == 0);
    REQUIRE(totals.largestName.empty());
}

TEST_CASE("The largest texture is chosen by name when sizes tie")
{
    const std::vector<TextureLoadRecord> ascending = {
        makeRecord("zulu", 8 * kMebibyte),
        makeRecord("alpha", 8 * kMebibyte),
    };
    const std::vector<TextureLoadRecord> descending = {
        makeRecord("alpha", 8 * kMebibyte),
        makeRecord("zulu", 8 * kMebibyte),
    };

    // Record order must not change the reported baseline.
    REQUIRE(summarizeTextures(ascending).largestName == "alpha");
    REQUIRE(summarizeTextures(descending).largestName == "alpha");
}

TEST_CASE("Cost sorting is descending by bytes then ascending by name")
{
    const std::vector<TextureLoadRecord> textures = {
        makeRecord("small", 1 * kMebibyte),
        makeRecord("zulu", 4 * kMebibyte),
        makeRecord("alpha", 4 * kMebibyte),
    };

    const std::vector<TextureLoadRecord> sorted = sortedByCost(textures);

    REQUIRE(sorted.size() == 3);
    REQUIRE(sorted[0].debugName == "alpha");
    REQUIRE(sorted[1].debugName == "zulu");
    REQUIRE(sorted[2].debugName == "small");
}

TEST_CASE("Mebibyte formatting keeps two decimals")
{
    REQUIRE(formatMebibytes(0) == "0.00");
    REQUIRE(formatMebibytes(kMebibyte) == "1.00");
    REQUIRE(formatMebibytes(kMebibyte / 2) == "0.50");
    REQUIRE(formatMebibytes(3 * kMebibyte + kMebibyte / 4) == "3.25");
}

TEST_CASE("The report is deterministic for a given stats value")
{
    AssetLoadStats stats{};
    stats.textures = {
        makeRecord("beta", 2 * kMebibyte),
        makeRecord("alpha", 8 * kMebibyte, true),
    };
    stats.timings.rendererInitMs = 812.5;
    stats.timings.textureUploadMs = 419.25;
    stats.memory.valid = true;
    stats.memory.deviceLocalUsedBytes = 64 * kMebibyte;
    stats.memory.allocationCount = 37;

    const std::string first = formatReport(stats);
    const std::string second = formatReport(stats);

    REQUIRE(first == second);

    // Record order is an implementation detail of when uploads happened; the
    // committed baseline must not change because of it.
    AssetLoadStats reordered = stats;
    std::swap(reordered.textures[0], reordered.textures[1]);
    REQUIRE(formatReport(reordered) == first);

    REQUIRE(first.find("812.50 ms") != std::string::npos);
    REQUIRE(first.find("419.25 ms") != std::string::npos);
    REQUIRE(first.find("alpha") != std::string::npos);
}

TEST_CASE("The report reports unavailable device memory rather than zeroes")
{
    AssetLoadStats stats{};
    stats.memory.valid = false;

    REQUIRE(formatReport(stats).find("unavailable") != std::string::npos);
}

TEST_CASE("The report row limit truncates and says how many were dropped")
{
    AssetLoadStats stats{};
    for (int index = 0; index < 5; ++index) {
        stats.textures.push_back(makeRecord("texture" + std::to_string(index), static_cast<uint64_t>(index) * kMebibyte));
    }

    const std::string report = formatReport(stats, /*largestTextureRows=*/2);

    REQUIRE(report.find("... 3 more") != std::string::npos);
}

TEST_CASE("The recorder drops records while disabled")
{
    const RecorderGuard guard;

    AssetLoadStatsRecorder::recordTexture(makeRecord("ignored", kMebibyte));

    REQUIRE(AssetLoadStatsRecorder::snapshotTextures().empty());
}

TEST_CASE("The recorder collects records from concurrent workers")
{
    const RecorderGuard guard;
    AssetLoadStatsRecorder::setEnabled(true);

    constexpr int kThreads = 4;
    constexpr int kPerThread = 64;

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int threadIndex = 0; threadIndex < kThreads; ++threadIndex) {
        workers.emplace_back([threadIndex]() {
            for (int index = 0; index < kPerThread; ++index) {
                AssetLoadStatsRecorder::recordTexture(
                    makeRecord("t" + std::to_string(threadIndex) + "_" + std::to_string(index), kMebibyte));
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    const std::vector<TextureLoadRecord> recorded = AssetLoadStatsRecorder::snapshotTextures();
    REQUIRE(recorded.size() == static_cast<size_t>(kThreads * kPerThread));
    REQUIRE(summarizeTextures(recorded).deviceBytes == static_cast<uint64_t>(kThreads * kPerThread) * kMebibyte);
}
