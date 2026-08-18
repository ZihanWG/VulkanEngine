#include "renderer/MeshCache.h"

#include <cstring>
#include <system_error>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ve::renderer {

namespace {

// FNV-1a. The fingerprint only has to change when the settings change, and this
// file never leaves the machine that wrote it, so a cryptographic digest would
// buy nothing.
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t hashBytes(const void* data, size_t size, uint64_t seed = kFnvOffsetBasis)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    uint64_t hash = seed;
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }

    return hash;
}

template <typename T>
void append(std::vector<std::byte>& out, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

void appendArray(std::vector<std::byte>& out, const void* data, size_t byteCount)
{
    const auto* bytes = static_cast<const std::byte*>(data);
    out.insert(out.end(), bytes, bytes + byteCount);
}

// Reads forward through the blob, refusing to run past the end. Every read is
// checked because a truncated cook must fail here rather than hand out a vector
// sized from a garbage count.
class BlobReader final {
public:
    explicit BlobReader(std::span<const std::byte> blob) : blob_(blob) {}

    template <typename T>
    T read()
    {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        readInto(&value, sizeof(T));
        return value;
    }

    void readInto(void* destination, size_t byteCount)
    {
        if (byteCount > blob_.size() - offset_) {
            throw std::runtime_error("Cooked mesh file is truncated.");
        }

        std::memcpy(destination, blob_.data() + offset_, byteCount);
        offset_ += byteCount;
    }

    // Guards a count read from the file against the bytes actually remaining, so
    // a corrupt length cannot make the reader allocate gigabytes before failing.
    [[nodiscard]] uint64_t readCount(size_t elementSize)
    {
        const auto count = read<uint64_t>();
        if (elementSize != 0 && count > (blob_.size() - offset_) / elementSize) {
            throw std::runtime_error("Cooked mesh file declares more elements than it contains.");
        }

        return count;
    }

private:
    std::span<const std::byte> blob_;
    size_t offset_ = 0;
};

} // namespace

std::string_view meshCacheStatusName(MeshCacheStatus status)
{
    switch (status) {
    case MeshCacheStatus::Usable:
        return "usable";
    case MeshCacheStatus::Missing:
        return "missing";
    case MeshCacheStatus::Malformed:
        return "malformed";
    case MeshCacheStatus::VersionMismatch:
        return "built by a different cache format version";
    case MeshCacheStatus::VertexLayoutMismatch:
        return "built with a different vertex layout";
    case MeshCacheStatus::LodSettingsMismatch:
        return "built with different LOD settings";
    case MeshCacheStatus::SourceChanged:
        return "the source glTF changed since the cook";
    }

    return "unknown";
}

std::filesystem::path meshCacheSidecarPath(const std::filesystem::path& gltfPath)
{
    std::filesystem::path sidecar = gltfPath;
    sidecar.replace_extension(".vemesh");
    return sidecar;
}

bool makeMeshCacheExpectation(const std::filesystem::path& gltfPath, MeshCacheExpectation& expectation)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(gltfPath, error);
    if (error) {
        return false;
    }

    const auto writeTime = std::filesystem::last_write_time(gltfPath, error);
    if (error) {
        return false;
    }

    expectation = MeshCacheExpectation{};
    expectation.lodSettingsHash = hashLodBuildSettings(LodBuildSettings{});
    expectation.sourceSizeBytes = static_cast<uint64_t>(size);
    expectation.sourceWriteTime = writeTime.time_since_epoch().count();
    return true;
}

uint64_t hashLodBuildSettings(const LodBuildSettings& settings)
{
    // Hashed field by field rather than as a struct: padding bytes are
    // indeterminate, and a fingerprint that varies run to run would reject every
    // cook it was supposed to accept.
    uint64_t hash = hashBytes(&settings.maxLods, sizeof(settings.maxLods));
    hash = hashBytes(&settings.minIndexCount, sizeof(settings.minIndexCount), hash);
    hash = hashBytes(&settings.targetError, sizeof(settings.targetError), hash);
    hash = hashBytes(&settings.indexRatio, sizeof(settings.indexRatio), hash);
    hash = hashBytes(&settings.minReduction, sizeof(settings.minReduction), hash);
    return hash;
}

MeshCacheStatus meshCacheStatus(std::span<const std::byte> blob, const MeshCacheExpectation& expected)
{
    if (blob.empty()) {
        return MeshCacheStatus::Missing;
    }
    if (blob.size() < sizeof(MeshCacheHeader)) {
        return MeshCacheStatus::Malformed;
    }

    MeshCacheHeader header{};
    std::memcpy(&header, blob.data(), sizeof(header));

    if (header.magic != kMeshCacheMagic) {
        return MeshCacheStatus::Malformed;
    }
    // Version is checked before anything else in the header is trusted: a future
    // format may not even lay these fields out the same way.
    if (header.version != kMeshCacheVersion) {
        return MeshCacheStatus::VersionMismatch;
    }
    if (header.vertexStride != expected.vertexStride || header.primitiveStride != expected.primitiveStride
        || header.lodStride != expected.lodStride) {
        return MeshCacheStatus::VertexLayoutMismatch;
    }
    if (header.lodSettingsHash != expected.lodSettingsHash) {
        return MeshCacheStatus::LodSettingsMismatch;
    }
    if (header.sourceSizeBytes != expected.sourceSizeBytes || header.sourceWriteTime != expected.sourceWriteTime) {
        return MeshCacheStatus::SourceChanged;
    }

    return MeshCacheStatus::Usable;
}

std::vector<std::byte> writeMeshCache(std::span<const CpuMeshData> meshes, const MeshCacheExpectation& expectation)
{
    MeshCacheHeader header{};
    header.vertexStride = expectation.vertexStride;
    header.primitiveStride = expectation.primitiveStride;
    header.lodStride = expectation.lodStride;
    header.meshCount = static_cast<uint32_t>(meshes.size());
    header.lodSettingsHash = expectation.lodSettingsHash;
    header.sourceSizeBytes = expectation.sourceSizeBytes;
    header.sourceWriteTime = expectation.sourceWriteTime;

    std::vector<std::byte> blob;
    append(blob, header);

    for (const CpuMeshData& mesh : meshes) {
        append(blob, static_cast<uint64_t>(mesh.debugName.size()));
        appendArray(blob, mesh.debugName.data(), mesh.debugName.size());

        append(blob, static_cast<uint64_t>(mesh.vertices.size()));
        appendArray(blob, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));

        append(blob, static_cast<uint64_t>(mesh.indices.size()));
        appendArray(blob, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));

        append(blob, mesh.indexCount);

        append(blob, static_cast<uint64_t>(mesh.primitives.size()));
        appendArray(blob, mesh.primitives.data(), mesh.primitives.size() * sizeof(MeshPrimitive));

        append(blob, static_cast<uint64_t>(mesh.lods.size()));
        appendArray(blob, mesh.lods.data(), mesh.lods.size() * sizeof(MeshLod));

        append(blob, mesh.localBounds);
    }

    return blob;
}

std::vector<CpuMeshData> readMeshCache(std::span<const std::byte> blob)
{
    if (blob.size() < sizeof(MeshCacheHeader)) {
        throw std::runtime_error("Cooked mesh file is shorter than its header.");
    }

    MeshCacheHeader header{};
    BlobReader reader(blob);
    reader.readInto(&header, sizeof(header));

    // meshCount comes from the file, so it is guarded the same way the array
    // counts are: a corrupt value must not size a vector before the first read
    // fails. Every mesh costs at least its six length fields plus its bounds.
    constexpr size_t kMinBytesPerMesh = 5 * sizeof(uint64_t) + sizeof(uint32_t) + sizeof(Aabb);
    if (static_cast<size_t>(header.meshCount) > (blob.size() - sizeof(MeshCacheHeader)) / kMinBytesPerMesh) {
        throw std::runtime_error("Cooked mesh file declares more meshes than it contains.");
    }

    std::vector<CpuMeshData> meshes(header.meshCount);
    for (CpuMeshData& mesh : meshes) {
        mesh.debugName.resize(static_cast<size_t>(reader.readCount(1)));
        reader.readInto(mesh.debugName.data(), mesh.debugName.size());

        mesh.vertices.resize(static_cast<size_t>(reader.readCount(sizeof(Vertex))));
        reader.readInto(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));

        mesh.indices.resize(static_cast<size_t>(reader.readCount(sizeof(uint32_t))));
        reader.readInto(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));

        mesh.indexCount = reader.read<uint32_t>();

        mesh.primitives.resize(static_cast<size_t>(reader.readCount(sizeof(MeshPrimitive))));
        reader.readInto(mesh.primitives.data(), mesh.primitives.size() * sizeof(MeshPrimitive));

        mesh.lods.resize(static_cast<size_t>(reader.readCount(sizeof(MeshLod))));
        reader.readInto(mesh.lods.data(), mesh.lods.size() * sizeof(MeshLod));

        mesh.localBounds = reader.read<Aabb>();

        // The LOD table indexes the index buffer, so a file that passed every
        // header check can still be internally inconsistent.
        if (mesh.indexCount > mesh.indices.size()) {
            throw std::runtime_error("Cooked mesh declares more authored indices than it stores.");
        }
        for (const MeshLod& lod : mesh.lods) {
            if (static_cast<size_t>(lod.firstIndex) + lod.indexCount > mesh.indices.size()) {
                throw std::runtime_error("Cooked mesh has a LOD level pointing past its index buffer.");
            }
        }
    }

    return meshes;
}

} // namespace ve::renderer
