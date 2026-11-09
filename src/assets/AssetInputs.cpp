#include "assets/AssetInputs.h"

#include "core/JobSystem.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <string_view>
#include <system_error>

namespace ve::assets {

namespace {

// FNV-1a, as in rhi::hashShaderDirectory. Not cryptographic: it only has to
// change reliably when an input file changes.
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

// Folded in where a file's size and contents would go when it cannot be read,
// so "missing" is distinct from every real file, including an empty one.
constexpr uint64_t kUnreadableMarker = UINT64_MAX;

// The unit of parallel work. A file is hashed as FNV-1a over the FNV-1a digests
// of its consecutive blocks, so one large file spreads across the pool instead
// of pinning a single thread. Hashed whole, Bistro's 227 MB vertex blob and
// 172 MB buffer landed in one parallelFor chunk and held its 1.7 GB digest at
// ~0.49 s of a 1.85 s startup; in blocks it is ~0.24 s (2.3 s on one thread).
// Eight interleaved FNV lanes, twice as fast per byte, only took that to
// ~0.20 s: the rest is reading the files back out of the OS cache, so plain
// FNV-1a stays.
//
// Part of the digest's definition: changing it, or anything else about how the
// value is formed, changes every digest. Two binaries that disagree on it
// report different inputs for identical files, so keep it fixed.
constexpr uint64_t kBlockBytes = uint64_t{4} << 20;

// Read buffer per participating thread. Bounded so that hashing a scene of
// large files in parallel never holds more than this per thread.
constexpr std::size_t kReadChunkBytes = std::size_t{1} << 20;

void hashBytes(uint64_t& hash, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
}

void hashValue(uint64_t& hash, uint64_t value)
{
    hashBytes(hash, &value, sizeof(value));
}

// Length first, so two adjacent keys can never be re-split into a different
// pair that hashes the same.
void hashText(uint64_t& hash, std::string_view text)
{
    hashValue(hash, static_cast<uint64_t>(text.size()));
    hashBytes(hash, text.data(), text.size());
}

// UTF-8 with forward slashes on every platform. path::generic_string() would
// go through the Windows ANSI code page and can throw on a name it cannot
// represent; a key must never take startup down.
std::string genericUtf8(const std::filesystem::path& path)
{
    const std::u8string text = path.generic_u8string();
    return std::string(text.begin(), text.end());
}

struct FileBlock {
    std::size_t fileIndex = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t hash = kFnvOffsetBasis;
    bool complete = false;
};

// Exactly block.length bytes from block.offset, or not complete: a file that
// shrank or failed part way through must not yield a plausible partial hash.
void hashBlock(const std::filesystem::path& path, FileBlock& block, std::vector<char>& buffer)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return;
    }
    file.seekg(static_cast<std::streamoff>(block.offset));
    uint64_t remaining = block.length;
    while (remaining > 0 && file) {
        const auto want = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
        file.read(buffer.data(), want);
        const std::streamsize got = file.gcount();
        if (got <= 0) {
            break;
        }
        hashBytes(block.hash, buffer.data(), static_cast<std::size_t>(got));
        remaining -= static_cast<uint64_t>(got);
    }
    block.complete = remaining == 0;
}

struct RecorderState {
    std::mutex mutex;
    std::set<std::filesystem::path> paths;
};

// Function-local so it is constructed on first use and outlives every
// translation unit that could still record into it.
RecorderState& recorderState()
{
    static RecorderState state;
    return state;
}

} // namespace

std::string assetInputKey(const std::filesystem::path& path, std::span<const AssetInputRoot> roots)
{
    const std::filesystem::path normalized = path.lexically_normal();

    const AssetInputRoot* bestRoot = nullptr;
    std::filesystem::path bestRelative;
    std::ptrdiff_t bestDepth = -1;
    for (const AssetInputRoot& root : roots) {
        if (root.directory.empty()) {
            continue;
        }
        const std::filesystem::path directory = root.directory.lexically_normal();
        const std::filesystem::path relative = normalized.lexically_relative(directory);
        // Empty: no common root (another drive, or relative against absolute).
        // "..": outside. ".": the directory itself, which is not a file in it.
        if (relative.empty() || relative == "." || *relative.begin() == "..") {
            continue;
        }
        const std::ptrdiff_t depth = std::distance(directory.begin(), directory.end());
        if (depth > bestDepth) {
            bestRoot = &root;
            bestRelative = relative;
            bestDepth = depth;
        }
    }

    if (bestRoot == nullptr) {
        return genericUtf8(normalized);
    }
    return bestRoot->name + "/" + genericUtf8(bestRelative);
}

std::vector<AssetInputFile> assetInputFiles(std::span<const std::filesystem::path> paths,
                                            std::span<const AssetInputRoot> roots)
{
    std::vector<AssetInputFile> files;
    files.reserve(paths.size());
    for (const std::filesystem::path& path : paths) {
        files.push_back(AssetInputFile{assetInputKey(path, roots), path});
    }
    return files;
}

AssetInputDigest digestAssetInputs(std::span<const AssetInputFile> files, JobSystem* jobSystem)
{
    std::vector<AssetInputFile> sorted(files.begin(), files.end());
    // The path breaks ties only when two different files share a key, which
    // takes two roots with the same name; ordinary inputs sort by key alone.
    const auto less = [](const AssetInputFile& a, const AssetInputFile& b) {
        return a.key != b.key ? a.key < b.key : a.path < b.path;
    };
    std::sort(sorted.begin(), sorted.end(), less);
    sorted.erase(std::unique(sorted.begin(),
                             sorted.end(),
                             [](const AssetInputFile& a, const AssetInputFile& b) {
                                 return a.key == b.key && a.path == b.path;
                             }),
                 sorted.end());

    // Sizes are taken once, up front, and are what the digest covers; a file
    // that cannot even be stat'd is unreadable without being opened.
    std::vector<uint64_t> sizes(sorted.size(), 0);
    std::vector<bool> readable(sorted.size(), false);
    std::vector<FileBlock> blocks;
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(sorted[index].path, error) || error) {
            continue;
        }
        const uintmax_t size = std::filesystem::file_size(sorted[index].path, error);
        if (error) {
            continue;
        }
        sizes[index] = static_cast<uint64_t>(size);
        readable[index] = true;
        for (uint64_t offset = 0; offset < sizes[index]; offset += kBlockBytes) {
            FileBlock block{};
            block.fileIndex = index;
            block.offset = offset;
            block.length = std::min(kBlockBytes, sizes[index] - offset);
            blocks.push_back(block);
        }
    }

    const auto hashRange = [&sorted, &blocks](std::size_t begin, std::size_t end) {
        std::vector<char> buffer(kReadChunkBytes);
        for (std::size_t index = begin; index < end; ++index) {
            hashBlock(sorted[blocks[index].fileIndex].path, blocks[index], buffer);
        }
    };
    if (jobSystem != nullptr && blocks.size() > 1) {
        jobSystem->parallelFor(blocks.size(), 1, hashRange);
    } else {
        hashRange(0, blocks.size());
    }

    // Blocks were laid out file by file, in offset order, so one pass folds
    // each file's block digests in the order they occur in the file.
    std::vector<uint64_t> contentHashes(sorted.size(), kFnvOffsetBasis);
    for (const FileBlock& block : blocks) {
        if (!block.complete) {
            readable[block.fileIndex] = false;
        }
        hashValue(contentHashes[block.fileIndex], block.hash);
    }

    AssetInputDigest result{};
    result.fileCount = sorted.size();
    uint64_t hash = kFnvOffsetBasis;
    hashValue(hash, static_cast<uint64_t>(sorted.size()));
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        hashText(hash, sorted[index].key);
        if (!readable[index]) {
            ++result.unreadableCount;
            hashValue(hash, kUnreadableMarker);
            hashValue(hash, kUnreadableMarker);
            continue;
        }
        result.totalBytes += sizes[index];
        hashValue(hash, sizes[index]);
        hashValue(hash, contentHashes[index]);
    }
    result.digest = hash;
    return result;
}

std::string formatAssetInputDigest(const AssetInputDigest& digest, std::string_view label)
{
    std::array<char, 32> hex{};
    std::snprintf(hex.data(), hex.size(), "0x%016llx", static_cast<unsigned long long>(digest.digest));

    std::string line = std::string(label) + ": " + std::to_string(digest.fileCount) +
                       (digest.fileCount == 1 ? " file" : " files") + ", digest " + hex.data();
    if (digest.unreadableCount != 0) {
        line += " (" + std::to_string(digest.unreadableCount) + " no longer readable)";
    }
    return line;
}

void AssetInputRecorder::record(const std::filesystem::path& path)
{
    if (path.empty()) {
        return;
    }
    std::error_code error;
    std::filesystem::path absolute = std::filesystem::absolute(path, error);
    if (error) {
        absolute = path;
    }
    absolute = absolute.lexically_normal();

    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.paths.insert(std::move(absolute));
}

void AssetInputRecorder::recordGltf(const std::filesystem::path& gltfPath, std::span<const std::string> bufferUris)
{
    record(gltfPath);
    for (const std::string& uri : bufferUris) {
        if (uri.empty() || uri.rfind("data:", 0) == 0) {
            continue;
        }
        // Resolved the way MeshCache's staleness check resolves it.
        const std::filesystem::path resolved = gltfPath.parent_path() / std::filesystem::path(uri);
        std::error_code error;
        if (std::filesystem::is_regular_file(resolved, error)) {
            record(resolved);
        }
    }
}

std::vector<std::filesystem::path> AssetInputRecorder::snapshot()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return std::vector<std::filesystem::path>(state.paths.begin(), state.paths.end());
}

std::size_t AssetInputRecorder::count()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    return state.paths.size();
}

void AssetInputRecorder::reset()
{
    RecorderState& state = recorderState();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.paths.clear();
}

} // namespace ve::assets
