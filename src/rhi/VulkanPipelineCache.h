#pragma once

#include "rhi/VulkanCommon.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

// Free-function helpers for VkPipelineCache disk persistence and header
// validation. They are intentionally decoupled from VulkanDevice so the
// validation logic (the only part that carries real correctness risk) can be
// unit-tested without a live VkDevice or GPU.
//
// On-disk layout is an engine-owned envelope followed verbatim by the driver
// blob returned by vkGetPipelineCacheData:
//
//     [ PipelineCacheEnvelope (24 bytes) ][ VkPipelineCacheHeaderVersionOne ... ]
//
// The envelope exists because the driver header only identifies the GPU and
// driver, not the shaders the cached pipelines were compiled from. A cache
// written before a shader interface change is therefore "valid" by the driver's
// own rules, and MoltenVK reacts to the disagreement by failing pipeline
// creation outright (VK_ERROR_INITIALIZATION_FAILED, "Fragment input(s)
// `user(locnN)` mismatching vertex shader output type(s)") rather than
// discarding the stale entries. Folding a digest of the compiled SPIR-V into the
// envelope makes any shader edit invalidate the blob, which turns that cryptic
// startup crash back into a one-line "shaders changed" log message.
namespace ve::rhi {

// Resolve the writable on-disk cache path. Prefers a persistent per-user cache
// directory (LOCALAPPDATA on Windows, XDG_CACHE_HOME/HOME/.cache elsewhere) and
// falls back to temp_directory_path()/"VulkanEngine". The file name is
// "pipeline_cache.bin". Never throws.
[[nodiscard]] std::filesystem::path pipelineCacheFilePath();

// Read the raw cache blob. Returns an empty vector on a missing/unreadable file.
// Never throws.
[[nodiscard]] std::vector<std::byte> readPipelineCacheBlob(const std::filesystem::path& path);

// Pure validation of a blob against the running physical device. Requires the
// leading VkPipelineCacheHeaderVersionOne to match: headerSize, headerVersion,
// vendorID, deviceID, and the pipelineCacheUUID. A blob shorter than the header,
// or any field mismatch, returns false so the driver is never fed a foreign or
// truncated cache. Never throws.
[[nodiscard]] bool pipelineCacheHeaderMatches(std::span<const std::byte> blob,
                                              const VkPhysicalDeviceProperties& props);

// Atomically persist the blob: write to a ".tmp" sibling then rename over the
// destination, creating parent directories as needed. Returns false on any
// failure. Never throws.
bool writePipelineCacheBlob(const std::filesystem::path& path, std::span<const std::byte> data);

// Digest of every compiled SPIR-V module (*.spv, non-recursive) in shaderDir.
// Covers each module's file name and byte contents, and is independent of
// directory iteration order. Returns 0 when the directory is missing, cannot be
// enumerated, or holds no modules -- an "unknown" sentinel that a real digest
// never collides with, so an unreadable shader directory degrades to the old
// device-only validation instead of permanently rejecting the cache. Never
// throws.
[[nodiscard]] uint64_t hashShaderDirectory(const std::filesystem::path& shaderDir);

// 'VEPC' little-endian. Distinct from any driver header, whose first four bytes
// are the header size (32).
inline constexpr uint32_t kPipelineCacheMagic = 0x43504556u;
inline constexpr uint32_t kPipelineCacheEnvelopeVersion = 1u;

// Engine-owned prefix binding the driver blob to the shaders it was built from.
// Written and read as raw bytes; the file never leaves the machine that wrote
// it, so native endianness and alignment are the wire format (as they already
// are for the driver header that follows).
struct PipelineCacheEnvelope {
    uint32_t magic = kPipelineCacheMagic;
    uint32_t envelopeVersion = kPipelineCacheEnvelopeVersion;
    uint64_t shaderHash = 0;
    uint64_t driverDataSize = 0;
};

// Why a stored cache was or was not handed to the driver. Everything but Usable
// means "start from an empty cache"; the distinction only drives logging.
enum class PipelineCacheStatus {
    Usable,
    Missing,        // No file on disk, or the file was empty/unreadable.
    Malformed,      // Not an engine envelope, wrong envelope version, or truncated.
    ShaderMismatch, // Written from a different set of compiled SPIR-V modules.
    DeviceMismatch, // Written by a different GPU, driver, or driver version.
};

struct PipelineCacheContents {
    PipelineCacheStatus status = PipelineCacheStatus::Missing;
    // The driver blob to feed vkCreatePipelineCache, valid only when status is
    // Usable. Borrows from the blob passed to decodePipelineCacheBlob().
    std::span<const std::byte> driverData{};
};

// Wrap a driver blob in the engine envelope for writing. Never throws.
[[nodiscard]] std::vector<std::byte> encodePipelineCacheBlob(std::span<const std::byte> driverData,
                                                             uint64_t shaderHash);

// Pure validation of a stored blob: envelope integrity, then shader digest, then
// device identity. Returns the borrowed driver payload only when every check
// passes. Never throws.
[[nodiscard]] PipelineCacheContents decodePipelineCacheBlob(std::span<const std::byte> blob,
                                                            const VkPhysicalDeviceProperties& props,
                                                            uint64_t shaderHash);

} // namespace ve::rhi
