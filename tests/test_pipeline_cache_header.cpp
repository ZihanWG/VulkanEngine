// Headless coverage for the pipeline-cache disk helpers. Only the GPU-free
// logic is exercised here: header validation against a synthetic
// VkPhysicalDeviceProperties, envelope encode/decode (including the shader
// digest that keys the cache to the SPIR-V it was built from), shader-directory
// hashing, and a write -> read byte round-trip. No VkDevice or GPU is required
// (and none is created), matching the rest of the test suite.
#include "rhi/VulkanPipelineCache.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr std::array<uint8_t, VK_UUID_SIZE> kUuid = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10};

VkPhysicalDeviceProperties makeProps()
{
    VkPhysicalDeviceProperties props{};
    props.vendorID = 0x10DE; // NVIDIA, arbitrary but concrete.
    props.deviceID = 0x2216;
    std::memcpy(props.pipelineCacheUUID, kUuid.data(), VK_UUID_SIZE);
    return props;
}

// Serialize a valid 32-byte header (+ optional payload) exactly as a driver
// would write it to disk.
std::vector<std::byte> makeBlob(const VkPipelineCacheHeaderVersionOne& header,
                                std::span<const std::byte> payload = {})
{
    std::vector<std::byte> blob(sizeof(header) + payload.size());
    std::memcpy(blob.data(), &header, sizeof(header));
    if (!payload.empty()) {
        std::memcpy(blob.data() + sizeof(header), payload.data(), payload.size());
    }
    return blob;
}

VkPipelineCacheHeaderVersionOne makeHeader(const VkPhysicalDeviceProperties& props)
{
    VkPipelineCacheHeaderVersionOne header{};
    header.headerSize = sizeof(VkPipelineCacheHeaderVersionOne);
    header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    header.vendorID = props.vendorID;
    header.deviceID = props.deviceID;
    std::memcpy(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE);
    return header;
}

// A driver blob as vkGetPipelineCacheData would return it: valid header plus
// opaque payload.
std::vector<std::byte> makeDriverData(const VkPhysicalDeviceProperties& props)
{
    const std::array<std::byte, 5> payload = {
        std::byte{0xa0}, std::byte{0xa1}, std::byte{0xa2}, std::byte{0xa3}, std::byte{0xa4}};
    return makeBlob(makeHeader(props), payload);
}

// Each shader-hash case gets its own directory so the cases stay independent
// and can run in any order.
std::filesystem::path makeShaderDir(std::string_view caseName)
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "VulkanEngineTests" / "shader_hash" / caseName;
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    std::filesystem::create_directories(directory, ec);
    return directory;
}

void writeFile(const std::filesystem::path& path, std::string_view contents)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

} // namespace

TEST_CASE("Pipeline cache header matches for the originating device", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::vector<std::byte> blob = makeBlob(makeHeader(props));
    REQUIRE(ve::rhi::pipelineCacheHeaderMatches(blob, props));
}

TEST_CASE("Pipeline cache header carries a trailing payload without affecting validation", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::array<std::byte, 4> payload = {
        std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}};
    const std::vector<std::byte> blob = makeBlob(makeHeader(props), payload);
    REQUIRE(ve::rhi::pipelineCacheHeaderMatches(blob, props));
}

TEST_CASE("Pipeline cache header is rejected on any device/driver mismatch", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();

    SECTION("wrong vendor")
    {
        VkPipelineCacheHeaderVersionOne header = makeHeader(props);
        header.vendorID = 0x1002; // AMD
        REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(makeBlob(header), props));
    }

    SECTION("wrong device")
    {
        VkPipelineCacheHeaderVersionOne header = makeHeader(props);
        header.deviceID += 1;
        REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(makeBlob(header), props));
    }

    SECTION("wrong UUID")
    {
        VkPipelineCacheHeaderVersionOne header = makeHeader(props);
        header.pipelineCacheUUID[0] ^= 0xFF;
        REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(makeBlob(header), props));
    }

    SECTION("wrong header version")
    {
        VkPipelineCacheHeaderVersionOne header = makeHeader(props);
        header.headerVersion = static_cast<VkPipelineCacheHeaderVersion>(0);
        REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(makeBlob(header), props));
    }

    SECTION("wrong header size")
    {
        VkPipelineCacheHeaderVersionOne header = makeHeader(props);
        header.headerSize = 24;
        REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(makeBlob(header), props));
    }
}

TEST_CASE("Pipeline cache header rejects a blob shorter than the header", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::array<std::byte, 4> tooShort = {
        std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}};
    REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(tooShort, props));

    const std::span<const std::byte> empty{};
    REQUIRE_FALSE(ve::rhi::pipelineCacheHeaderMatches(empty, props));
}

TEST_CASE("Pipeline cache blob survives a write/read round-trip", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::array<std::byte, 6> payload = {
        std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
        std::byte{0x44}, std::byte{0x55}, std::byte{0x66}};
    const std::vector<std::byte> original = makeBlob(makeHeader(props), payload);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "VulkanEngineTests" / "pipeline_cache_roundtrip.bin";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    REQUIRE(ve::rhi::writePipelineCacheBlob(path, original));

    const std::vector<std::byte> readBack = ve::rhi::readPipelineCacheBlob(path);
    REQUIRE(readBack.size() == original.size());
    REQUIRE(std::memcmp(readBack.data(), original.data(), original.size()) == 0);
    REQUIRE(ve::rhi::pipelineCacheHeaderMatches(readBack, props));

    std::filesystem::remove(path, ec);
}

TEST_CASE("Reading a missing pipeline cache returns an empty blob", "[pipeline_cache]")
{
    const std::filesystem::path missing =
        std::filesystem::temp_directory_path() / "VulkanEngineTests" / "does_not_exist_pipeline_cache.bin";
    std::error_code ec;
    std::filesystem::remove(missing, ec);
    REQUIRE(ve::rhi::readPipelineCacheBlob(missing).empty());
}

TEST_CASE("Pipeline cache envelope round-trips the driver blob unchanged", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::vector<std::byte> driverData = makeDriverData(props);
    constexpr uint64_t kShaderHash = 0x1234'5678'9abc'def0ull;

    const std::vector<std::byte> blob = ve::rhi::encodePipelineCacheBlob(driverData, kShaderHash);
    REQUIRE(blob.size() == sizeof(ve::rhi::PipelineCacheEnvelope) + driverData.size());

    const ve::rhi::PipelineCacheContents contents = ve::rhi::decodePipelineCacheBlob(blob, props, kShaderHash);
    REQUIRE(contents.status == ve::rhi::PipelineCacheStatus::Usable);
    REQUIRE(contents.driverData.size() == driverData.size());
    REQUIRE(std::memcmp(contents.driverData.data(), driverData.data(), driverData.size()) == 0);
}

// The regression this whole envelope exists for: editing a shader changes the
// digest, so the blob compiled from the old SPIR-V must never reach the driver.
TEST_CASE("Pipeline cache is rejected when the shader hash differs", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::vector<std::byte> driverData = makeDriverData(props);
    constexpr uint64_t kOldShaderHash = 0x1111'2222'3333'4444ull;
    constexpr uint64_t kNewShaderHash = 0x1111'2222'3333'4445ull;

    const std::vector<std::byte> blob = ve::rhi::encodePipelineCacheBlob(driverData, kOldShaderHash);

    const ve::rhi::PipelineCacheContents contents = ve::rhi::decodePipelineCacheBlob(blob, props, kNewShaderHash);
    REQUIRE(contents.status == ve::rhi::PipelineCacheStatus::ShaderMismatch);
    // Nothing to hand vkCreatePipelineCache: a rejected blob must not leak a
    // usable payload to the caller.
    REQUIRE(contents.driverData.empty());

    // The same bytes are still accepted by the unchanged shader set, proving the
    // rejection came from the digest rather than a corrupted envelope.
    REQUIRE(ve::rhi::decodePipelineCacheBlob(blob, props, kOldShaderHash).status ==
            ve::rhi::PipelineCacheStatus::Usable);
}

TEST_CASE("Pipeline cache envelope rejects device mismatch, damage, and absence", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    constexpr uint64_t kShaderHash = 0x0f0f'0f0f'0f0f'0f0full;
    const std::vector<std::byte> driverData = makeDriverData(props);

    SECTION("device mismatch behind a matching shader hash")
    {
        VkPhysicalDeviceProperties otherGpu = props;
        otherGpu.deviceID += 1;
        const std::vector<std::byte> blob = ve::rhi::encodePipelineCacheBlob(driverData, kShaderHash);
        REQUIRE(ve::rhi::decodePipelineCacheBlob(blob, otherGpu, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::DeviceMismatch);
    }

    SECTION("empty blob reads as missing rather than malformed")
    {
        const std::span<const std::byte> empty{};
        REQUIRE(ve::rhi::decodePipelineCacheBlob(empty, props, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::Missing);
    }

    SECTION("legacy file written before the envelope existed")
    {
        // A bare driver blob: its first four bytes are the header size (32),
        // which must not be mistaken for the engine magic.
        REQUIRE(ve::rhi::decodePipelineCacheBlob(driverData, props, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::Malformed);
    }

    SECTION("wrong magic")
    {
        std::vector<std::byte> blob = ve::rhi::encodePipelineCacheBlob(driverData, kShaderHash);
        blob[0] = static_cast<std::byte>(std::to_integer<uint8_t>(blob[0]) ^ 0xFF);
        REQUIRE(ve::rhi::decodePipelineCacheBlob(blob, props, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::Malformed);
    }

    SECTION("unknown envelope version")
    {
        ve::rhi::PipelineCacheEnvelope envelope{};
        envelope.magic = ve::rhi::kPipelineCacheMagic;
        envelope.envelopeVersion = ve::rhi::kPipelineCacheEnvelopeVersion + 1;
        envelope.shaderHash = kShaderHash;
        envelope.driverDataSize = driverData.size();

        std::vector<std::byte> blob(sizeof(envelope) + driverData.size());
        std::memcpy(blob.data(), &envelope, sizeof(envelope));
        std::memcpy(blob.data() + sizeof(envelope), driverData.data(), driverData.size());

        REQUIRE(ve::rhi::decodePipelineCacheBlob(blob, props, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::Malformed);
    }

    SECTION("truncated payload contradicts the recorded size")
    {
        std::vector<std::byte> blob = ve::rhi::encodePipelineCacheBlob(driverData, kShaderHash);
        blob.pop_back();
        REQUIRE(ve::rhi::decodePipelineCacheBlob(blob, props, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::Malformed);
    }

    SECTION("blob shorter than the envelope")
    {
        const std::array<std::byte, 8> stub = {
            std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
            std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}};
        REQUIRE(ve::rhi::decodePipelineCacheBlob(stub, props, kShaderHash).status ==
                ve::rhi::PipelineCacheStatus::Malformed);
    }
}

TEST_CASE("Shader directory hash tracks the compiled SPIR-V", "[pipeline_cache]")
{
    SECTION("stable across calls and sensitive to module contents")
    {
        const std::filesystem::path directory = makeShaderDir("contents");
        writeFile(directory / "simple.vert.spv", "vertex-spirv");
        writeFile(directory / "simple.frag.spv", "fragment-spirv");

        const uint64_t original = ve::rhi::hashShaderDirectory(directory);
        REQUIRE(original != 0);
        REQUIRE(ve::rhi::hashShaderDirectory(directory) == original);

        // Same byte count, different bytes: a digest over sizes alone would miss
        // this, and so would the driver's own header check.
        writeFile(directory / "simple.vert.spv", "vertex-spirvX");
        const uint64_t edited = ve::rhi::hashShaderDirectory(directory);
        REQUIRE(edited != original);

        writeFile(directory / "simple.vert.spv", "vertex-spirv");
        REQUIRE(ve::rhi::hashShaderDirectory(directory) == original);
    }

    SECTION("sensitive to added and removed modules")
    {
        const std::filesystem::path directory = makeShaderDir("membership");
        writeFile(directory / "a.spv", "aaa");
        const uint64_t single = ve::rhi::hashShaderDirectory(directory);

        writeFile(directory / "b.spv", "bbb");
        const uint64_t pair = ve::rhi::hashShaderDirectory(directory);
        REQUIRE(pair != single);

        std::error_code ec;
        std::filesystem::remove(directory / "b.spv", ec);
        REQUIRE(ve::rhi::hashShaderDirectory(directory) == single);
    }

    SECTION("module names matter, non-SPIR-V neighbours do not")
    {
        const std::filesystem::path named = makeShaderDir("named");
        writeFile(named / "a.spv", "shared-bytes");
        const uint64_t before = ve::rhi::hashShaderDirectory(named);

        // Renaming a module changes the interface the cache was built against.
        std::error_code ec;
        std::filesystem::rename(named / "a.spv", named / "b.spv", ec);
        REQUIRE_FALSE(ec);
        REQUIRE(ve::rhi::hashShaderDirectory(named) != before);

        // GLSL sources and stray files sitting next to the modules are not what
        // the driver consumes, so they must not churn the digest.
        std::filesystem::rename(named / "b.spv", named / "a.spv", ec);
        writeFile(named / "a.vert", "glsl source");
        writeFile(named / "notes.txt", "scratch");
        REQUIRE(ve::rhi::hashShaderDirectory(named) == before);
    }

    SECTION("unhashable directories report the zero sentinel")
    {
        const std::filesystem::path empty = makeShaderDir("empty");
        REQUIRE(ve::rhi::hashShaderDirectory(empty) == 0);

        std::error_code ec;
        std::filesystem::remove_all(empty, ec);
        REQUIRE(ve::rhi::hashShaderDirectory(empty) == 0);

        REQUIRE(ve::rhi::hashShaderDirectory(std::filesystem::path{}) == 0);
    }
}

TEST_CASE("A wrapped pipeline cache survives a write/read round-trip", "[pipeline_cache]")
{
    const VkPhysicalDeviceProperties props = makeProps();
    const std::filesystem::path directory = makeShaderDir("roundtrip");
    writeFile(directory / "simple.frag.spv", "fragment-spirv");

    const uint64_t shaderHash = ve::rhi::hashShaderDirectory(directory);
    const std::vector<std::byte> driverData = makeDriverData(props);
    const std::vector<std::byte> blob = ve::rhi::encodePipelineCacheBlob(driverData, shaderHash);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "VulkanEngineTests" / "pipeline_cache_envelope.bin";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    REQUIRE(ve::rhi::writePipelineCacheBlob(path, blob));

    const std::vector<std::byte> readBack = ve::rhi::readPipelineCacheBlob(path);
    REQUIRE(ve::rhi::decodePipelineCacheBlob(readBack, props, shaderHash).status ==
            ve::rhi::PipelineCacheStatus::Usable);

    // Recompile one module and the same on-disk file stops being usable, which
    // is exactly the startup-crash mitigation.
    writeFile(directory / "simple.frag.spv", "fragment-spirv-v2");
    REQUIRE(ve::rhi::decodePipelineCacheBlob(readBack, props, ve::rhi::hashShaderDirectory(directory)).status ==
            ve::rhi::PipelineCacheStatus::ShaderMismatch);

    std::filesystem::remove(path, ec);
    std::filesystem::remove_all(directory, ec);
}
