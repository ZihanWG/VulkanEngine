#pragma once

// Load-time instrumentation for the asset pipeline baseline.
//
// This is measurement scaffolding, not part of the resource ownership model.
// Nothing here owns, retains, or dereferences a Vulkan handle: Vulkan-facing
// code pushes plain value records in, and every aggregation/formatting function
// below is pure so it can be unit tested headlessly.
//
// Scope: startup asset loading only. The steady-state frame path is untouched
// and must stay that way -- per-frame GPU timing already belongs to GpuProfiler.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ve::renderer {

// One texture as it exists on the device after upload, including its mip chain.
//
// textureId exists because creation happens before naming in every current call
// path: the owner sets debug metadata only after the texture is constructed, so
// the record is amended by id rather than guessed at by position.
struct TextureLoadRecord {
    uint64_t textureId = 0;
    std::string debugName;
    std::string formatName;
    std::string sourcePath;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
    uint64_t deviceBytes = 0;
    bool blockCompressed = false;
};

// Wall-clock segments of startup. decodeWait and upload are measured separately
// on purpose: decode is already dispatched to the JobSystem, so the split is the
// evidence for where the serial cost actually sits.
struct AssetLoadTimings {
    double rendererInitMs = 0.0;
    double sceneCreateMs = 0.0;
    double gltfImportMs = 0.0;
    double textureDecodeWaitMs = 0.0;
    double textureUploadMs = 0.0;
    double firstFrameMs = 0.0;
};

struct DeviceMemoryUsage {
    uint64_t deviceLocalUsedBytes = 0;
    uint64_t deviceLocalBudgetBytes = 0;
    uint64_t deviceLocalAllocatedBytes = 0;
    uint32_t allocationCount = 0;
    bool valid = false;
};

struct AssetLoadStats {
    std::vector<TextureLoadRecord> textures;
    AssetLoadTimings timings;
    DeviceMemoryUsage memory;
};

struct TextureTotals {
    size_t count = 0;
    size_t blockCompressedCount = 0;
    uint64_t deviceBytes = 0;
    uint64_t blockCompressedBytes = 0;
    uint64_t largestBytes = 0;
    std::string largestName;
};

// Pure. Ties on byte size break on debugName so the result is order-independent
// with respect to the order records were recorded in.
[[nodiscard]] TextureTotals summarizeTextures(std::span<const TextureLoadRecord> textures);

// Pure. Returns records sorted by deviceBytes descending, then debugName
// ascending -- a stable, diffable order for committing a baseline.
[[nodiscard]] std::vector<TextureLoadRecord> sortedByCost(std::span<const TextureLoadRecord> textures);

// Pure. Binary MiB with two decimals; used by the report and by tests.
[[nodiscard]] std::string formatMebibytes(uint64_t bytes);

// Pure and deterministic for a given AssetLoadStats value: no clock reads, no
// locale-dependent formatting, no iteration over unordered containers.
[[nodiscard]] std::string formatReport(const AssetLoadStats& stats, size_t largestTextureRows = 10);

// Process-wide sink for texture records produced during startup.
//
// A sink rather than plumbing because every VulkanTexture creation path funnels
// through one upload function; threading the stats object through ~20 call sites
// would touch far more code than the measurement is worth. Disabled by default:
// with recording off, the only observable difference is the absent report.
class AssetLoadStatsRecorder final {
public:
    static void setEnabled(bool enabled);
    [[nodiscard]] static bool enabled();

    // Process-unique and monotonic. Handed out even while recording is disabled
    // so an id never means two different textures across an enable/disable edge.
    [[nodiscard]] static uint64_t nextTextureId();

    // Thread-safe. Safe to call from JobSystem workers.
    static void recordTexture(TextureLoadRecord record);

    // No-op when textureId was never recorded (recording off, or the texture was
    // created before it was enabled).
    static void amendTextureName(uint64_t textureId, std::string debugName, std::string sourcePath);

    [[nodiscard]] static std::vector<TextureLoadRecord> snapshotTextures();
    static void reset();
};

} // namespace ve::renderer
