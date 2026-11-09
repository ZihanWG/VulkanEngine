#pragma once

// Which asset files a run read, folded into one digest.
//
// The engine reads assets/ (and the imported sample scenes) from wherever the
// build was configured to look, at run time. An already-built binary therefore
// renders whatever those directories hold *now*: a branch switch or an
// uncommitted edit in a shared checkout changes the output of a binary nobody
// rebuilt, and before this line nothing in the log said so. Two capture logs
// whose "Asset inputs:" lines differ ran on different inputs, whatever else
// they agree on.
//
// The loaders record each file as they read it -- including whether a cooked
// sidecar or its source was the one used -- and the digest is taken over that
// list once startup has finished. Only files the run actually read count: an
// unrelated file that changed in assets/ does not move the digest.
//
// The lines are compared across binaries -- an A/B is two builds -- so how the
// digest is formed is part of its meaning: change it and every build before the
// change reports different inputs for the same files.
//
// GPU-free on purpose: the loaders that record into it sit on both sides of the
// Core/Runtime split, and the digest is unit-tested headlessly.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ve {
class JobSystem;
} // namespace ve

namespace ve::assets {

// A directory whose files are named relative to it in the digest, so the same
// files reached through two different checkouts (or worktrees) hash the same.
struct AssetInputRoot {
    std::string name;
    std::filesystem::path directory;
};

// One file as the digest sees it: `key` is what gets hashed as its name,
// `path` is where its bytes are read from.
struct AssetInputFile {
    std::string key;
    std::filesystem::path path;
};

struct AssetInputDigest {
    std::size_t fileCount = 0;
    // Files on the list that could not be read back when the digest was taken
    // (deleted or locked since they were loaded). Each still contributes a
    // marker, so a vanished file never looks like an unchanged one.
    std::size_t unreadableCount = 0;
    uint64_t totalBytes = 0;
    uint64_t digest = 0;
};

// "<root name>/<path relative to that root>" with forward slashes, for the most
// specific root that contains `path`; the normalized path itself when no root
// does. Lexical only -- touches no file.
[[nodiscard]] std::string assetInputKey(const std::filesystem::path& path, std::span<const AssetInputRoot> roots);

// Pairs every path with its key, in the order given.
[[nodiscard]] std::vector<AssetInputFile> assetInputFiles(std::span<const std::filesystem::path> paths,
                                                          std::span<const AssetInputRoot> roots);

// FNV-1a, the same family rhi::hashShaderDirectory uses, over each file's key,
// byte size and contents. Sorted by key first, so the order the files were
// loaded in -- which the JobSystem makes nondeterministic -- does not matter,
// and an exact duplicate entry counts once. A file's contents are hashed in
// fixed-size blocks, and with a JobSystem the blocks are hashed in parallel, so
// one large file does not serialize the digest (call it from a thread that is
// not a pool worker). The value does not depend on whether a JobSystem was
// used. Never throws for an unreadable file.
[[nodiscard]] AssetInputDigest digestAssetInputs(std::span<const AssetInputFile> files, JobSystem* jobSystem = nullptr);

// "Asset inputs: 12 files, digest 0x0123456789abcdef", plus an unreadable count
// when there is one. Fixed width hex, so two logs line up under a diff. No
// timing and no byte total on purpose: the line has to be identical across two
// runs on identical inputs, and the digest already covers the sizes.
[[nodiscard]] std::string formatAssetInputDigest(const AssetInputDigest& digest,
                                                 std::string_view label = "Asset inputs");

// Process-wide list of the asset files this run has read.
//
// A sink rather than plumbing for the same reason as AssetLoadStatsRecorder: the
// reads happen in a dozen loaders on both sides of the library split, several of
// them on JobSystem workers. Always on -- one mutex and a set insert per file
// loaded is nothing next to the load itself.
class AssetInputRecorder final {
public:
    // Thread-safe. Stores the absolute, lexically normal form, so one file
    // reached by two spellings is recorded once. Call it after the read has
    // succeeded: a file that was looked for and not found is not an input.
    static void record(const std::filesystem::path& path);

    // The glTF itself plus every external buffer it references (data: URIs and
    // GLB-internal buffers are already inside the file). Buffers that do not
    // resolve to an existing file are skipped rather than guessed at.
    static void recordGltf(const std::filesystem::path& gltfPath, std::span<const std::string> bufferUris);

    // Sorted and unique.
    [[nodiscard]] static std::vector<std::filesystem::path> snapshot();
    [[nodiscard]] static std::size_t count();
    static void reset();
};

} // namespace ve::assets
