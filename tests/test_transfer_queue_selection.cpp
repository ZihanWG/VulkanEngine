#include "rhi/TransferQueueSelection.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

using ve::rhi::kNoQueueFamily;
using ve::rhi::QueueFamilyCapabilities;
using ve::rhi::selectTransferQueueFamily;

namespace {

constexpr QueueFamilyCapabilities kUniversal{true, true, true, 1};
constexpr QueueFamilyCapabilities kGraphicsTransfer{true, false, true, 1};
constexpr QueueFamilyCapabilities kComputeTransfer{false, true, true, 1};
constexpr QueueFamilyCapabilities kTransferOnly{false, false, true, 1};

uint32_t select(const std::vector<QueueFamilyCapabilities>& families)
{
    return selectTransferQueueFamily(std::span<const QueueFamilyCapabilities>(families));
}

} // namespace

TEST_CASE("A dedicated transfer family is chosen", "[transfer-queue]")
{
    // This is what MoltenVK exposes as family 3 under
    // MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1, and the layout the selection
    // exists for.
    CHECK(select({kUniversal, kGraphicsTransfer, kComputeTransfer, kTransferOnly}) == 3);
    CHECK(select({kTransferOnly, kUniversal}) == 0);
}

TEST_CASE("Nothing is chosen when every family can also render", "[transfer-queue]")
{
    // MoltenVK's default: four identical GRAPHICS|COMPUTE|TRANSFER families. This
    // is the case the primary dev machine actually hits, so the fallback is the
    // one that most needs pinning -- uploads must stay on the graphics queue.
    CHECK(select({kUniversal, kUniversal, kUniversal, kUniversal}) == kNoQueueFamily);
    CHECK(select({kUniversal}) == kNoQueueFamily);
    CHECK(select({}) == kNoQueueFamily);
}

TEST_CASE("Graphics- or compute-capable families are rejected", "[transfer-queue]")
{
    // A second queue on a graphics or compute family is still that ring. Taking
    // one would add ownership-transfer machinery for no overlap, which is why
    // this selection has no such fallback even though async compute does.
    CHECK(select({kGraphicsTransfer, kComputeTransfer}) == kNoQueueFamily);

    constexpr QueueFamilyCapabilities manyGraphicsQueues{true, true, true, 8};
    CHECK(select({manyGraphicsQueues}) == kNoQueueFamily);
}

TEST_CASE("A transfer family with no queues is not usable", "[transfer-queue]")
{
    constexpr QueueFamilyCapabilities emptyTransfer{false, false, true, 0};
    CHECK(select({kUniversal, emptyTransfer}) == kNoQueueFamily);
    // ...but a real one later in the list still wins.
    CHECK(select({kUniversal, emptyTransfer, kTransferOnly}) == 2);
}

TEST_CASE("A family without the transfer bit is never chosen", "[transfer-queue]")
{
    // Defensive: the spec lets graphics/compute families omit the transfer bit
    // while still supporting transfer, but a family with none of the three is
    // not a candidate for anything.
    constexpr QueueFamilyCapabilities nothing{false, false, false, 4};
    CHECK(select({nothing, nothing}) == kNoQueueFamily);
}
