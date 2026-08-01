#include "rhi/VulkanPipelineCache.h"

#include "core/Logger.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace ve::rhi {

namespace {

// The on-disk pipeline-cache header is a fixed 32-byte layout defined by the
// Vulkan spec (4 x uint32 + 16-byte UUID). Guard against any ABI padding
// surprise so the validation below matches the wire format exactly.
static_assert(sizeof(VkPipelineCacheHeaderVersionOne) == 32,
              "VkPipelineCacheHeaderVersionOne must be exactly 32 bytes to match the on-disk header.");

// Likewise for the engine envelope: it is memcpy'd to and from disk, so any
// padding the compiler inserted would silently become part of the format.
static_assert(sizeof(PipelineCacheEnvelope) == 24,
              "PipelineCacheEnvelope must be exactly 24 bytes to match the on-disk envelope.");

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

// FNV-1a. Not cryptographic -- it only has to change reliably when the SPIR-V
// changes, and it is cheap enough to run over every module at startup.
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void hashBytes(uint64_t& hash, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
}

void hashText(uint64_t& hash, std::string_view text)
{
    hashBytes(hash, text.data(), text.size());
}

void hashValue(uint64_t& hash, uint64_t value)
{
    hashBytes(hash, &value, sizeof(value));
}

// Enumerate the compiled modules in sorted order so the digest does not depend
// on filesystem iteration order. Returns false if the directory could not be
// fully enumerated -- a partial listing must not produce a plausible-looking
// digest.
bool collectShaderModules(const std::filesystem::path& shaderDir, std::vector<std::filesystem::path>& modules)
{
    std::error_code ec;
    std::filesystem::directory_iterator entry(shaderDir, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec) {
        return false;
    }

    const std::filesystem::directory_iterator end;
    for (; entry != end; entry.increment(ec)) {
        const std::filesystem::path& path = entry->path();
        if (path.extension() != ".spv") {
            continue;
        }
        std::error_code statusError;
        if (!std::filesystem::is_regular_file(path, statusError) || statusError) {
            continue;
        }
        modules.push_back(path);
    }
    // A failed increment leaves the iterator equal to end, so the loop exits
    // normally and this is the only place the truncation is visible.
    if (ec) {
        return false;
    }

    std::sort(modules.begin(), modules.end());
    return true;
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

uint64_t hashShaderDirectory(const std::filesystem::path& shaderDir)
{
    std::error_code ec;
    if (shaderDir.empty() || !std::filesystem::is_directory(shaderDir, ec) || ec) {
        return 0;
    }

    std::vector<std::filesystem::path> modules;
    if (!collectShaderModules(shaderDir, modules) || modules.empty()) {
        return 0;
    }

    uint64_t hash = kFnvOffsetBasis;
    hashValue(hash, static_cast<uint64_t>(modules.size()));

    for (const std::filesystem::path& modulePath : modules) {
        // Only the file name, never the absolute path: an out-of-tree build
        // directory must not change the digest of identical shaders.
        const std::string name = modulePath.filename().string();
        hashText(hash, name);

        std::ifstream file(modulePath, std::ios::binary | std::ios::ate);
        if (!file) {
            // Fold in a marker so an unreadable module still yields a digest
            // distinct from the one where that module reads cleanly.
            hashValue(hash, UINT64_MAX);
            continue;
        }

        const std::streamoff size = file.tellg();
        if (size < 0) {
            hashValue(hash, UINT64_MAX);
            continue;
        }

        hashValue(hash, static_cast<uint64_t>(size));
        std::vector<char> contents(static_cast<std::size_t>(size));
        file.seekg(0, std::ios::beg);
        file.read(contents.data(), static_cast<std::streamsize>(contents.size()));
        if (!file) {
            hashValue(hash, UINT64_MAX);
            continue;
        }
        hashBytes(hash, contents.data(), contents.size());
    }

    // 0 is reserved for "could not hash the shader directory", so a real digest
    // must never land on it.
    return hash == 0 ? 1 : hash;
}

std::vector<std::byte> encodePipelineCacheBlob(std::span<const std::byte> driverData, uint64_t shaderHash)
{
    PipelineCacheEnvelope envelope{};
    envelope.magic = kPipelineCacheMagic;
    envelope.envelopeVersion = kPipelineCacheEnvelopeVersion;
    envelope.shaderHash = shaderHash;
    envelope.driverDataSize = static_cast<uint64_t>(driverData.size());

    std::vector<std::byte> blob(sizeof(envelope) + driverData.size());
    std::memcpy(blob.data(), &envelope, sizeof(envelope));
    if (!driverData.empty()) {
        std::memcpy(blob.data() + sizeof(envelope), driverData.data(), driverData.size());
    }
    return blob;
}

PipelineCacheContents decodePipelineCacheBlob(std::span<const std::byte> blob,
                                              const VkPhysicalDeviceProperties& props,
                                              uint64_t shaderHash)
{
    if (blob.empty()) {
        return {PipelineCacheStatus::Missing, {}};
    }
    if (blob.size() < sizeof(PipelineCacheEnvelope)) {
        return {PipelineCacheStatus::Malformed, {}};
    }

    // Copy the prefix into a properly aligned struct rather than reinterpreting
    // the raw byte pointer (avoids alignment / strict-aliasing UB).
    PipelineCacheEnvelope envelope{};
    std::memcpy(&envelope, blob.data(), sizeof(envelope));

    // A cache file written before the envelope existed lands here (its first
    // four bytes are the driver header size) and is discarded, which is the
    // desired one-time migration.
    if (envelope.magic != kPipelineCacheMagic || envelope.envelopeVersion != kPipelineCacheEnvelopeVersion) {
        return {PipelineCacheStatus::Malformed, {}};
    }
    if (envelope.driverDataSize != static_cast<uint64_t>(blob.size() - sizeof(PipelineCacheEnvelope))) {
        return {PipelineCacheStatus::Malformed, {}};
    }

    // Shader digest before device identity: after a shader edit both would
    // usually pass except this one, and it is the more useful thing to report.
    if (envelope.shaderHash != shaderHash) {
        return {PipelineCacheStatus::ShaderMismatch, {}};
    }

    const std::span<const std::byte> driverData = blob.subspan(sizeof(PipelineCacheEnvelope));
    if (!pipelineCacheHeaderMatches(driverData, props)) {
        return {PipelineCacheStatus::DeviceMismatch, {}};
    }

    return {PipelineCacheStatus::Usable, driverData};
}

} // namespace ve::rhi
