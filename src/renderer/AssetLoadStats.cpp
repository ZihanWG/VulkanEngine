#include "renderer/AssetLoadStats.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <mutex>
#include <string_view>

namespace ve::renderer {

namespace {

// The sink is function-local so it is initialized on first use and destroyed
// after every translation unit that could still record into it.
struct RecorderState {
    std::mutex mutex;
    std::vector<TextureLoadRecord> textures;
    uint64_t nextTextureId = 1;
    bool enabled = false;
};

RecorderState& recorderState()
{
    static RecorderState state;
    return state;
}

[[nodiscard]] std::string formatFixed(double value, int decimals)
{
    std::array<char, 64> buffer{};
    const int written = std::snprintf(buffer.data(), buffer.size(), "%.*f", decimals, value);
    if (written <= 0) {
        return "0";
    }
    return std::string(buffer.data(), static_cast<size_t>(written));
}

[[nodiscard]] std::string padRight(std::string text, size_t width)
{
    if (text.size() < width) {
        text.append(width - text.size(), ' ');
    }
    return text;
}

[[nodiscard]] std::string padLeft(std::string text, size_t width)
{
    if (text.size() < width) {
        text.insert(text.begin(), static_cast<std::ptrdiff_t>(width - text.size()), ' ');
    }
    return text;
}

void appendTimingLine(std::string& out, std::string_view label, double milliseconds)
{
    out += "  ";
    out += padRight(std::string(label), 26);
    out += padLeft(formatFixed(milliseconds, 2), 10);
    out += " ms\n";
}

} // namespace

TextureTotals summarizeTextures(std::span<const TextureLoadRecord> textures)
{
    TextureTotals totals{};
    totals.count = textures.size();

    for (const TextureLoadRecord& record : textures) {
        totals.deviceBytes += record.deviceBytes;
        if (record.blockCompressed) {
            ++totals.blockCompressedCount;
            totals.blockCompressedBytes += record.deviceBytes;
        }

        const bool larger = record.deviceBytes > totals.largestBytes;
        const bool tiedAndEarlierName = record.deviceBytes == totals.largestBytes &&
                                        (totals.largestName.empty() || record.debugName < totals.largestName);
        if (larger || tiedAndEarlierName) {
            totals.largestBytes = record.deviceBytes;
            totals.largestName = record.debugName;
        }
    }

    return totals;
}

std::vector<TextureLoadRecord> sortedByCost(std::span<const TextureLoadRecord> textures)
{
    std::vector<TextureLoadRecord> sorted(textures.begin(), textures.end());
    std::sort(sorted.begin(), sorted.end(), [](const TextureLoadRecord& lhs, const TextureLoadRecord& rhs) {
        if (lhs.deviceBytes != rhs.deviceBytes) {
            return lhs.deviceBytes > rhs.deviceBytes;
        }
        return lhs.debugName < rhs.debugName;
    });
    return sorted;
}

std::string formatMebibytes(uint64_t bytes)
{
    constexpr double kBytesPerMebibyte = 1024.0 * 1024.0;
    return formatFixed(static_cast<double>(bytes) / kBytesPerMebibyte, 2);
}

std::string formatReport(const AssetLoadStats& stats, size_t largestTextureRows)
{
    const TextureTotals totals = summarizeTextures(stats.textures);

    std::string out;
    out.reserve(2048);
    out += "\n=== Asset load baseline ===\n";

    out += "Timings\n";
    appendTimingLine(out, "renderer init (total)", stats.timings.rendererInitMs);
    appendTimingLine(out, "  scene create", stats.timings.sceneCreateMs);
    appendTimingLine(out, "    glTF import", stats.timings.gltfImportMs);
    appendTimingLine(out, "    texture decode wait", stats.timings.textureDecodeWaitMs);
    appendTimingLine(out, "    texture upload", stats.timings.textureUploadMs);
    appendTimingLine(out, "first frame", stats.timings.firstFrameMs);

    out += "Textures\n";
    out += "  count                     " + padLeft(std::to_string(totals.count), 10) + "\n";
    out += "  device bytes              " + padLeft(formatMebibytes(totals.deviceBytes), 10) + " MiB\n";
    out += "  block compressed          " + padLeft(std::to_string(totals.blockCompressedCount), 10) + " of " +
           std::to_string(totals.count) + "\n";
    if (!totals.largestName.empty()) {
        out += "  largest                   " + padLeft(formatMebibytes(totals.largestBytes), 10) + " MiB  " +
               totals.largestName + "\n";
    }

    out += "Device memory (VMA)\n";
    if (stats.memory.valid) {
        out += "  device-local used         " + padLeft(formatMebibytes(stats.memory.deviceLocalUsedBytes), 10) +
               " MiB\n";
        out += "  device-local allocated    " + padLeft(formatMebibytes(stats.memory.deviceLocalAllocatedBytes), 10) +
               " MiB\n";
        out += "  device-local budget       " + padLeft(formatMebibytes(stats.memory.deviceLocalBudgetBytes), 10) +
               " MiB\n";
        out += "  allocations               " + padLeft(std::to_string(stats.memory.allocationCount), 10) + "\n";
    } else {
        out += "  unavailable\n";
    }

    if (largestTextureRows > 0 && !stats.textures.empty()) {
        const std::vector<TextureLoadRecord> sorted = sortedByCost(stats.textures);
        const size_t rowCount = std::min(largestTextureRows, sorted.size());
        out += "Largest textures\n";
        for (size_t index = 0; index < rowCount; ++index) {
            const TextureLoadRecord& record = sorted[index];
            out += "  " + padLeft(formatMebibytes(record.deviceBytes), 9) + " MiB  " +
                   padRight(std::to_string(record.width) + "x" + std::to_string(record.height), 12) +
                   padRight("mips " + std::to_string(record.mipLevels), 9) + padRight(record.formatName, 34) +
                   record.debugName + "\n";
        }
        if (sorted.size() > rowCount) {
            out += "  ... " + std::to_string(sorted.size() - rowCount) + " more\n";
        }
    }

    out += "=== end asset load baseline ===\n";
    return out;
}

void AssetLoadStatsRecorder::setEnabled(bool enabled)
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.enabled = enabled;
}

bool AssetLoadStatsRecorder::enabled()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.enabled;
}

uint64_t AssetLoadStatsRecorder::nextTextureId()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.nextTextureId++;
}

void AssetLoadStatsRecorder::recordTexture(TextureLoadRecord record)
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.enabled) {
        return;
    }
    state.textures.push_back(std::move(record));
}

void AssetLoadStatsRecorder::amendTextureName(uint64_t textureId, std::string debugName, std::string sourcePath)
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.enabled || textureId == 0) {
        return;
    }

    const auto match = std::find_if(state.textures.begin(), state.textures.end(), [textureId](const auto& record) {
        return record.textureId == textureId;
    });
    if (match == state.textures.end()) {
        return;
    }

    if (!debugName.empty()) {
        match->debugName = std::move(debugName);
    }
    if (!sourcePath.empty()) {
        match->sourcePath = std::move(sourcePath);
    }
}

std::vector<TextureLoadRecord> AssetLoadStatsRecorder::snapshotTextures()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.textures;
}

void AssetLoadStatsRecorder::reset()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.textures.clear();
}

} // namespace ve::renderer
