#include "rhi/AsyncComputeQueueSelection.h"
#include "rhi/PortabilityFallbacks.h"
#include "rhi/TransferQueueSelection.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using ve::rhi::AsyncComputeQueueChoice;
using ve::rhi::kNoQueueFamily;
using ve::rhi::moltenVkDefaultQueueFamilies;
using ve::rhi::QueueFamilyCapabilities;
using ve::rhi::selectAsyncComputeQueue;
using ve::rhi::selectTransferQueueFamily;

namespace {

// The RTX 3080 Ti Laptop's families: universal with 16 queues, a TRANSFER-only
// family with 2, a COMPUTE|TRANSFER family with 8, and a video family that has
// none of the three bits.
const std::vector<QueueFamilyCapabilities> kRtxFamilies = {
    {true, true, true, 16},
    {false, false, true, 2},
    {false, true, true, 8},
    {false, false, false, 1},
};

// MoltenVK without MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES: four identical
// universal families, one queue each.
const std::vector<QueueFamilyCapabilities> kMoltenVkDefault(4, QueueFamilyCapabilities{true, true, true, 1});

AsyncComputeQueueChoice selectCompute(const std::vector<QueueFamilyCapabilities>& families, uint32_t graphicsFamily)
{
    return selectAsyncComputeQueue(std::span<const QueueFamilyCapabilities>(families), graphicsFamily);
}

} // namespace

TEST_CASE("A compute-only family is the async compute queue", "[async-compute-queue]")
{
    const AsyncComputeQueueChoice choice = selectCompute(kRtxFamilies, 0);
    CHECK(choice.available());
    CHECK(choice.family == 2);
    CHECK(choice.queueIndex == 0);
    CHECK(choice.dedicatedFamily);
}

TEST_CASE("A second graphics queue is the async compute fallback", "[async-compute-queue]")
{
    // lavapipe's shape: one universal family with more than one queue.
    const std::vector<QueueFamilyCapabilities> families = {{true, true, true, 2}};
    const AsyncComputeQueueChoice choice = selectCompute(families, 0);
    CHECK(choice.available());
    CHECK(choice.family == 0);
    CHECK(choice.queueIndex == 1);
    CHECK_FALSE(choice.dedicatedFamily);
}

TEST_CASE("One queue per universal family leaves compute inline", "[async-compute-queue]")
{
    // The Mac by default. Several families do not add up to a second queue:
    // only the graphics family's own count can supply one.
    CHECK_FALSE(selectCompute(kMoltenVkDefault, 0).available());
    CHECK_FALSE(selectCompute({}, 0).available());
}

TEST_CASE("A compute-only family with no queues is not usable", "[async-compute-queue]")
{
    const std::vector<QueueFamilyCapabilities> families = {{true, true, true, 1}, {false, true, true, 0}};
    CHECK_FALSE(selectCompute(families, 0).available());
}

TEST_CASE("Portability fallbacks present every family as MoltenVK does", "[portability-fallbacks]")
{
    const std::vector<QueueFamilyCapabilities> rewritten = moltenVkDefaultQueueFamilies(kRtxFamilies);

    // Indices are kept: the graphics and present families were chosen from the
    // real list and must still name the same families.
    REQUIRE(rewritten.size() == kRtxFamilies.size());
    for (const QueueFamilyCapabilities& family : rewritten) {
        CHECK(family.graphics);
        CHECK(family.compute);
        CHECK(family.transfer);
        CHECK(family.queueCount == 1);
    }
}

TEST_CASE("Portability fallbacks leave a capable GPU with no extra queue", "[portability-fallbacks]")
{
    // The point of the flag, end to end: the same selections that find both
    // queues on the RTX machine find neither on its rewritten families.
    REQUIRE(selectCompute(kRtxFamilies, 0).available());
    REQUIRE(selectTransferQueueFamily(kRtxFamilies) == 1);

    const std::vector<QueueFamilyCapabilities> rewritten = moltenVkDefaultQueueFamilies(kRtxFamilies);
    CHECK_FALSE(selectCompute(rewritten, 0).available());
    CHECK(selectTransferQueueFamily(rewritten) == kNoQueueFamily);
}
