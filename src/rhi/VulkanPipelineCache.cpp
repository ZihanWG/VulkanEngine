#include "rhi/VulkanPipelineCache.h"

#include "core/Logger.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>

namespace ve::rhi {

namespace {

// The on-disk pipeline-cache header is a fixed 32-byte layout defined by the
// Vulkan spec (4 x uint32 + 16-byte UUID). Guard against any ABI padding
// surprise so the validation below matches the wire format exactly.
static_assert(sizeof(VkPipelineCacheHeaderVersionOne) == 32,
              "VkPipelineCacheHeaderVersionOne must be exactly 32 bytes to match the on-disk header.");

std::filesystem::path cacheBaseDirectory()
{
    // Prefer a persistent per-user cache location; only fall back to the temp
    // directory (which may be wiped) when none is available.
#if defined(_WIN32)
    if (const char* localAppData = std::getenv("LOCALAPPDATA"); localAppData != nullptr && localAppData[0] != '\0') {
        return std::filesystem::path(localAppData) / "VulkanEngine";
    }
#else
    if (const char* xdgCache = std::getenv("XDG_CACHE_HOME"); xdgCache != nullptr && xdgCache[0] != '\0') {
        return std::filesystem::path(xdgCache) / "VulkanEngine";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "VulkanEngine";
    }
#endif
    std::error_code ec;
    std::filesystem::path temp = std::filesystem::temp_directory_path(ec);
    if (ec) {
        temp = std::filesystem::path(".");
    }
    return temp / "VulkanEngine";
}

} // namespace

std::filesystem::path pipelineCacheFilePath()
{
    return cacheBaseDirectory() / "pipeline_cache.bin";
}

std::vector<std::byte> readPipelineCacheBlob(const std::filesystem::path& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return {};
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }

    const std::streamoff size = file.tellg();
    if (size <= 0) {
        return {};
    }

    std::vector<std::byte> blob(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    if (!file) {
        return {};
    }
    return blob;
}

bool pipelineCacheHeaderMatches(std::span<const std::byte> blob, const VkPhysicalDeviceProperties& props)
{
    if (blob.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
        return false;
    }

    // Copy the prefix into a properly aligned struct rather than reinterpreting
    // the raw byte pointer (avoids alignment / strict-aliasing UB).
    VkPipelineCacheHeaderVersionOne header{};
    std::memcpy(&header, blob.data(), sizeof(header));

    return header.headerSize == sizeof(VkPipelineCacheHeaderVersionOne)
        && header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE
        && header.vendorID == props.vendorID
        && header.deviceID == props.deviceID
        && std::memcmp(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

bool writePipelineCacheBlob(const std::filesystem::path& path, std::span<const std::byte> data)
{
    if (data.empty()) {
        return false;
    }

    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            return false;
        }
    }

    const std::filesystem::path tempPath = std::filesystem::path(path).concat(".tmp");
    {
        std::ofstream file(tempPath, std::ios::binary | std::ios::trunc);
        if (!file) {
            return false;
        }
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!file) {
            return false;
        }
    }

    std::filesystem::rename(tempPath, path, ec);
    if (ec) {
        // Fall back to a non-atomic copy if rename across the temp path fails,
        // then remove the temp file. Best-effort: never throw out of teardown.
        std::filesystem::remove(tempPath, ec);
        return false;
    }
    return true;
}

} // namespace ve::rhi
