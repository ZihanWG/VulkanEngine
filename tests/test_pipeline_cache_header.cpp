// Headless coverage for the pipeline-cache disk helpers. Only the GPU-free
// logic is exercised here: header validation against a synthetic
// VkPhysicalDeviceProperties and a write -> read byte round-trip. No VkDevice or
// GPU is required (and none is created), matching the rest of the test suite.
#include "rhi/VulkanPipelineCache.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <span>
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
