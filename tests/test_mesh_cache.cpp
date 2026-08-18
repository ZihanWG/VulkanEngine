#include "renderer/MeshCache.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using ve::renderer::Aabb;
using ve::renderer::CpuMeshData;
using ve::renderer::hashLodBuildSettings;
using ve::renderer::LodBuildSettings;
using ve::renderer::MeshCacheExpectation;
using ve::renderer::MeshCacheStatus;
using ve::renderer::meshCacheStatus;
using ve::renderer::MeshLod;
using ve::renderer::MeshPrimitive;
using ve::renderer::readMeshCache;
using ve::renderer::Vertex;
using ve::renderer::writeMeshCache;

namespace {

MeshCacheExpectation makeExpectation()
{
    MeshCacheExpectation expectation{};
    expectation.lodSettingsHash = hashLodBuildSettings(LodBuildSettings{});
    expectation.sourceSizeBytes = 123456;
    expectation.sourceWriteTime = 1'700'000'000;
    return expectation;
}

CpuMeshData makeMesh(std::string name, uint32_t vertexCount, uint32_t indexCount)
{
    CpuMeshData mesh;
    mesh.debugName = std::move(name);
    mesh.vertices.resize(vertexCount);
    for (uint32_t i = 0; i < vertexCount; ++i) {
        mesh.vertices[i].position = {static_cast<float>(i), 1.0f, 2.0f};
        mesh.vertices[i].uv = {0.25f, 0.75f};
    }
    mesh.indices.resize(indexCount);
    for (uint32_t i = 0; i < indexCount; ++i) {
        mesh.indices[i] = i % vertexCount;
    }
    mesh.indexCount = indexCount;
    mesh.primitives.push_back(MeshPrimitive{0, indexCount, 3, 0, 1});
    mesh.lods.push_back(MeshLod{0, indexCount});
    mesh.localBounds.expand({-1.0f, -2.0f, -3.0f});
    mesh.localBounds.expand({4.0f, 5.0f, 6.0f});
    return mesh;
}

} // namespace

TEST_CASE("A cooked mesh round-trips exactly", "[mesh-cache]")
{
    const std::vector<CpuMeshData> written = {makeMesh("Mesh A", 8, 24), makeMesh("Mesh B", 4, 6)};
    const MeshCacheExpectation expectation = makeExpectation();
    const std::vector<std::byte> blob = writeMeshCache(written, expectation);

    REQUIRE(meshCacheStatus(blob, expectation) == MeshCacheStatus::Usable);
    const std::vector<CpuMeshData> read = readMeshCache(blob);

    REQUIRE(read.size() == written.size());
    for (size_t mesh = 0; mesh < read.size(); ++mesh) {
        CHECK(read[mesh].debugName == written[mesh].debugName);
        CHECK(read[mesh].indexCount == written[mesh].indexCount);
        CHECK(read[mesh].indices == written[mesh].indices);
        REQUIRE(read[mesh].vertices.size() == written[mesh].vertices.size());
        for (size_t v = 0; v < read[mesh].vertices.size(); ++v) {
            CHECK(read[mesh].vertices[v].position == written[mesh].vertices[v].position);
            CHECK(read[mesh].vertices[v].uv == written[mesh].vertices[v].uv);
        }
        REQUIRE(read[mesh].primitives.size() == 1);
        CHECK(read[mesh].primitives[0].materialIndex == 3);
        REQUIRE(read[mesh].lods.size() == 1);
        CHECK(read[mesh].lods[0].indexCount == written[mesh].indexCount);
        CHECK(read[mesh].localBounds.min == written[mesh].localBounds.min);
        CHECK(read[mesh].localBounds.max == written[mesh].localBounds.max);
    }
}

TEST_CASE("A changed vertex layout is rejected", "[mesh-cache]")
{
    // The case the whole header exists for: an old file still parses cleanly and
    // would hand back wrong geometry, so it has to be refused on inspection.
    const MeshCacheExpectation written = makeExpectation();
    const std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, written);

    MeshCacheExpectation grownVertex = written;
    grownVertex.vertexStride += 4;
    CHECK(meshCacheStatus(blob, grownVertex) == MeshCacheStatus::VertexLayoutMismatch);

    MeshCacheExpectation grownPrimitive = written;
    grownPrimitive.primitiveStride += 4;
    CHECK(meshCacheStatus(blob, grownPrimitive) == MeshCacheStatus::VertexLayoutMismatch);

    MeshCacheExpectation grownLod = written;
    grownLod.lodStride += 4;
    CHECK(meshCacheStatus(blob, grownLod) == MeshCacheStatus::VertexLayoutMismatch);
}

TEST_CASE("Changed LOD settings are rejected", "[mesh-cache]")
{
    // The other silent-corruption case: the geometry is laid out identically but
    // the chains inside it are wrong for the settings now in force.
    const MeshCacheExpectation written = makeExpectation();
    const std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, written);

    LodBuildSettings changed{};
    changed.targetError *= 2.0f;
    MeshCacheExpectation expectation = written;
    expectation.lodSettingsHash = hashLodBuildSettings(changed);
    CHECK(meshCacheStatus(blob, expectation) == MeshCacheStatus::LodSettingsMismatch);

    // Every field must participate, or a cook survives a change it should not.
    const uint64_t base = hashLodBuildSettings(LodBuildSettings{});
    LodBuildSettings maxLods{};
    maxLods.maxLods += 1;
    LodBuildSettings minIndex{};
    minIndex.minIndexCount += 1;
    LodBuildSettings ratio{};
    ratio.indexRatio += 0.1;
    LodBuildSettings reduction{};
    reduction.minReduction += 0.1;
    CHECK(hashLodBuildSettings(maxLods) != base);
    CHECK(hashLodBuildSettings(minIndex) != base);
    CHECK(hashLodBuildSettings(ratio) != base);
    CHECK(hashLodBuildSettings(reduction) != base);

    // And it must be stable, or it would reject every cook it should accept.
    CHECK(hashLodBuildSettings(LodBuildSettings{}) == base);
}

TEST_CASE("An edited source glTF is rejected", "[mesh-cache]")
{
    const MeshCacheExpectation written = makeExpectation();
    const std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, written);

    MeshCacheExpectation resized = written;
    resized.sourceSizeBytes += 1;
    CHECK(meshCacheStatus(blob, resized) == MeshCacheStatus::SourceChanged);

    MeshCacheExpectation touched = written;
    touched.sourceWriteTime += 1;
    CHECK(meshCacheStatus(blob, touched) == MeshCacheStatus::SourceChanged);
}

TEST_CASE("Foreign, truncated and empty blobs are refused before parsing", "[mesh-cache]")
{
    const MeshCacheExpectation expectation = makeExpectation();
    const std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, expectation);

    CHECK(meshCacheStatus(std::span<const std::byte>{}, expectation) == MeshCacheStatus::Missing);

    std::vector<std::byte> shortBlob(blob.begin(), blob.begin() + 8);
    CHECK(meshCacheStatus(shortBlob, expectation) == MeshCacheStatus::Malformed);

    std::vector<std::byte> foreign = blob;
    foreign[0] = std::byte{0x00};
    CHECK(meshCacheStatus(foreign, expectation) == MeshCacheStatus::Malformed);

    // Version is checked before the layout fields, because a future format may
    // not lay them out the same way at all.
    std::vector<std::byte> futureVersion = blob;
    futureVersion[4] = std::byte{0xFF};
    CHECK(meshCacheStatus(futureVersion, expectation) == MeshCacheStatus::VersionMismatch);
}

TEST_CASE("Ranges that would reach the GPU are validated", "[mesh-cache]")
{
    // Primitive ranges and LOD ranges become indexed indirect draws, and index
    // values become vertex fetches. A cook that passes every header check can
    // still be internally inconsistent, and a fallback cannot undo an
    // out-of-bounds fetch that already happened.
    const MeshCacheExpectation expectation = makeExpectation();

    CpuMeshData pastIndices = makeMesh("M", 4, 6);
    pastIndices.primitives[0].indexCount = 99;
    CHECK_THROWS_AS(readMeshCache(writeMeshCache(std::vector<CpuMeshData>{pastIndices}, expectation)),
                    std::runtime_error);

    CpuMeshData pastLods = makeMesh("M", 4, 6);
    pastLods.primitives[0].lodBase = 5;
    CHECK_THROWS_AS(readMeshCache(writeMeshCache(std::vector<CpuMeshData>{pastLods}, expectation)),
                    std::runtime_error);

    CpuMeshData badIndex = makeMesh("M", 4, 6);
    badIndex.indices[3] = 99;
    CHECK_THROWS_AS(readMeshCache(writeMeshCache(std::vector<CpuMeshData>{badIndex}, expectation)),
                    std::runtime_error);

    // The valid mesh those were derived from still round-trips, so the checks
    // are not rejecting everything.
    CHECK_NOTHROW(readMeshCache(writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, expectation)));
}

TEST_CASE("A replaced external buffer invalidates the cook", "[mesh-cache]")
{
    // An ASCII glTF holds no vertex data of its own. Fingerprinting only the
    // .gltf would call the cook fresh after its .bin had been replaced, and the
    // runtime would upload the old geometry while parsing the new scene.
    MeshCacheExpectation cooked = makeExpectation();
    cooked.bufferHash = 0x1111'2222'3333'4444ULL;
    const std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, cooked);
    REQUIRE(meshCacheStatus(blob, cooked) == MeshCacheStatus::Usable);

    MeshCacheExpectation binChanged = cooked;
    binChanged.bufferHash = 0x5555'6666'7777'8888ULL;
    CHECK(meshCacheStatus(blob, binChanged) == MeshCacheStatus::SourceChanged);
}

TEST_CASE("An implausible mesh count is refused before allocating", "[mesh-cache]")
{
    // meshCount is read from the file and sizes a vector. A corrupt value has to
    // be rejected against the bytes actually present, or a truncated cook asks
    // for gigabytes before the first read fails.
    const MeshCacheExpectation expectation = makeExpectation();
    std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 4, 6)}, expectation);

    const uint32_t huge = 0xFFFFFF00u;
    std::memcpy(blob.data() + 5 * sizeof(uint32_t), &huge, sizeof(huge));
    REQUIRE(meshCacheStatus(blob, expectation) == MeshCacheStatus::Usable);
    CHECK_THROWS_AS(readMeshCache(blob), std::runtime_error);
}

TEST_CASE("A payload truncated past the header throws rather than over-reading", "[mesh-cache]")
{
    // meshCacheStatus only inspects the header, so a body cut short has to fail
    // in the reader -- and it must not size a vector from a count it cannot honour.
    const MeshCacheExpectation expectation = makeExpectation();
    const std::vector<std::byte> blob = writeMeshCache(std::vector<CpuMeshData>{makeMesh("M", 64, 192)}, expectation);

    std::vector<std::byte> truncated(blob.begin(), blob.end() - 64);
    REQUIRE(meshCacheStatus(truncated, expectation) == MeshCacheStatus::Usable);
    CHECK_THROWS_AS(readMeshCache(truncated), std::runtime_error);
}
