#pragma once

// Cooked mesh geometry: the output of loadGltfGeometry, serialized so a later run
// can skip parse, vertex assembly and LOD simplification entirely. On Sponza that
// is ~333 ms of a ~349 ms import.
//
// GPU-free, and modelled on rhi/VulkanPipelineCache.h -- including the part that
// matters most here. A cooked KTX2 states its own format, so a stale one is
// visible; a cooked mesh is just bytes. If `Vertex` gains a field or a LOD
// threshold changes, an old file still parses cleanly and produces **wrong
// geometry from a valid-looking header**. The header exists to make that
// impossible, and `meshCacheStatus()` returns a reason rather than a bool so the
// rejection shows up in a log line instead of as a silent slow path.

#include "renderer/GltfGeometry.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace ve::renderer {

// 'VEMC' little-endian.
inline constexpr uint32_t kMeshCacheMagic = 0x434D4556u;
inline constexpr uint32_t kMeshCacheVersion = 1u;

// What the running engine expects a cooked file to have been built from. Every
// field is something that silently changes the correct bytes.
struct MeshCacheExpectation {
    uint32_t vertexStride = static_cast<uint32_t>(sizeof(Vertex));
    uint32_t primitiveStride = static_cast<uint32_t>(sizeof(MeshPrimitive));
    uint32_t lodStride = static_cast<uint32_t>(sizeof(MeshLod));
    uint64_t lodSettingsHash = 0;
    uint64_t sourceSizeBytes = 0;
    int64_t sourceWriteTime = 0;
    // Size and write time of every external buffer the glTF references. An ASCII
    // glTF holds no vertex data itself, so without this a replaced .bin leaves
    // the cook looking fresh.
    uint64_t bufferHash = 0;
};

struct MeshCacheHeader {
    uint32_t magic = kMeshCacheMagic;
    uint32_t version = kMeshCacheVersion;
    uint32_t vertexStride = 0;
    uint32_t primitiveStride = 0;
    uint32_t lodStride = 0;
    uint32_t meshCount = 0;
    uint64_t lodSettingsHash = 0;
    uint64_t sourceSizeBytes = 0;
    int64_t sourceWriteTime = 0;
    uint64_t bufferHash = 0;
};

// Why a cooked file was or was not used. Everything but Usable means "load the
// glTF instead"; the distinction only drives logging, but it is the difference
// between "your cook is stale, re-run it" and an unexplained slow startup.
enum class MeshCacheStatus {
    Usable,
    Missing,              // No file, or it was empty/unreadable.
    Malformed,            // Not one of ours, or truncated before the header ends.
    VersionMismatch,      // Written by a different format version.
    VertexLayoutMismatch, // sizeof(Vertex)/MeshPrimitive/MeshLod changed.
    LodSettingsMismatch,  // Built with different LOD thresholds.
    SourceChanged,        // The glTF has been edited since the cook.
};

[[nodiscard]] std::string_view meshCacheStatusName(MeshCacheStatus status);

// Where the cook writes and the runtime looks: `Sponza.gltf` -> `Sponza.vemesh`,
// beside the source. Unlike a cooked texture there is no material slot, so one
// glTF has exactly one cooked file.
[[nodiscard]] std::filesystem::path meshCacheSidecarPath(const std::filesystem::path& gltfPath);

// Builds the expectation for `gltfPath` from the running engine's own types and
// the file on disk. Both the cook and the runtime go through this so they cannot
// disagree about what a match means. Returns false when the source cannot be
// stat'd.
[[nodiscard]] bool makeMeshCacheExpectation(const std::filesystem::path& gltfPath,
                                            MeshCacheExpectation& expectation);

// Fingerprint of the settings that decide what the LOD chains contain. Changing
// any of them changes the correct output, so it has to invalidate the cook.
[[nodiscard]] uint64_t hashLodBuildSettings(const LodBuildSettings& settings);

// Pure validation. Never throws; a blob it rejects is never parsed.
[[nodiscard]] MeshCacheStatus meshCacheStatus(std::span<const std::byte> blob,
                                              const MeshCacheExpectation& expected);

[[nodiscard]] std::vector<std::byte> writeMeshCache(std::span<const CpuMeshData> meshes,
                                                    const MeshCacheExpectation& expectation);

// Only call after meshCacheStatus() returned Usable. Throws std::runtime_error if
// the payload past the header is inconsistent, which the status check cannot see.
[[nodiscard]] std::vector<CpuMeshData> readMeshCache(std::span<const std::byte> blob);

} // namespace ve::renderer
