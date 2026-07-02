#pragma once

#include "rhi/VulkanCommon.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

// Free-function helpers for VkPipelineCache disk persistence and header
// validation. They are intentionally decoupled from VulkanDevice so the
// validation logic (the only part that carries real correctness risk) can be
// unit-tested without a live VkDevice or GPU.
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

} // namespace ve::rhi
