#include "renderer/TransientMemoryPlan.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

using Catch::Approx;
using ve::renderer::planTransientMemory;
using ve::renderer::TransientAllocationRequest;
using ve::renderer::TransientMemoryPlan;

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;

TransientAllocationRequest request(std::string name,
                                   uint64_t size,
                                   uint32_t firstPass,
                                   uint32_t lastPass,
                                   uint64_t alignment = 128)
{
    TransientAllocationRequest value{};
    value.name = std::move(name);
    value.size = size;
    value.alignment = alignment;
    value.firstPass = firstPass;
    value.lastPass = lastPass;
    return value;
}

// Every placed pair whose lifetimes overlap must occupy disjoint byte ranges.
// This is the invariant the whole feature rests on: a violation is a rendering
// corruption bug, so it is asserted structurally rather than by spot-checking
// offsets.
void requireNoOverlappingCollision(const TransientMemoryPlan& plan)
{
    for (size_t i = 0; i < plan.allocations.size(); ++i) {
        for (size_t j = i + 1; j < plan.allocations.size(); ++j) {
            const auto& a = plan.allocations[i];
            const auto& b = plan.allocations[j];
            if (!a.placed || !b.placed) {
                continue;
            }
            const bool lifetimeOverlap = !(a.lastPass < b.firstPass || b.lastPass < a.firstPass);
            if (!lifetimeOverlap) {
                continue;
            }
            const bool bytesOverlap = a.offset < b.offset + b.size && b.offset < a.offset + a.size;
            INFO("'" << a.name << "' [" << a.offset << "," << a.offset + a.size << ") vs '" << b.name << "' ["
                     << b.offset << "," << b.offset + b.size << ")");
            REQUIRE_FALSE(bytesOverlap);
        }
    }
}

} // namespace

TEST_CASE("An empty request set yields an empty pool")
{
    const TransientMemoryPlan plan = planTransientMemory({});

    REQUIRE(plan.allocations.empty());
    REQUIRE(plan.poolBytes == 0);
    REQUIRE(plan.unaliasedBytes == 0);
    REQUIRE(plan.reuseRatio() == Approx(0.0));
    REQUIRE(plan.savedBytes() == 0);
}

TEST_CASE("A single resource sits at offset zero")
{
    const std::vector<TransientAllocationRequest> requests = {request("only", 4 * kMiB, 0, 3)};

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.allocations.size() == 1);
    REQUIRE(plan.allocations[0].placed);
    REQUIRE(plan.allocations[0].offset == 0);
    REQUIRE(plan.poolBytes == 4 * kMiB);
    REQUIRE(plan.reuseRatio() == Approx(0.0));
}

TEST_CASE("Resources with disjoint lifetimes share the same bytes")
{
    const std::vector<TransientAllocationRequest> requests = {
        request("early", 8 * kMiB, 0, 2),
        request("late", 8 * kMiB, 3, 5),
    };

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.allocations[0].offset == plan.allocations[1].offset);
    REQUIRE(plan.poolBytes == 8 * kMiB);
    REQUIRE(plan.unaliasedBytes == 16 * kMiB);
    REQUIRE(plan.savedBytes() == 8 * kMiB);
    REQUIRE(plan.reuseRatio() == Approx(0.5));
}

TEST_CASE("Resources that share a single pass never share bytes")
{
    // Touching at one pass is still an overlap: both are live during it.
    const std::vector<TransientAllocationRequest> requests = {
        request("first", 8 * kMiB, 0, 3),
        request("second", 8 * kMiB, 3, 6),
    };

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.allocations[0].offset != plan.allocations[1].offset);
    REQUIRE(plan.poolBytes == 16 * kMiB);
    requireNoOverlappingCollision(plan);
}

TEST_CASE("A gap left by a small resource is reused by a later fitting one")
{
    // long spans everything; small and later are disjoint from each other, so
    // later must land back in small's bytes rather than extending the pool.
    const std::vector<TransientAllocationRequest> requests = {
        request("long", 16 * kMiB, 0, 9),
        request("small", 4 * kMiB, 0, 2),
        request("later", 4 * kMiB, 5, 9),
    };

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.allocations[1].offset == plan.allocations[2].offset);
    REQUIRE(plan.poolBytes == 20 * kMiB);
    requireNoOverlappingCollision(plan);
}

TEST_CASE("Alignment is honoured for every placed resource")
{
    const std::vector<TransientAllocationRequest> requests = {
        request("a", 100, 0, 5, 256),
        request("b", 100, 0, 5, 256),
        request("c", 100, 0, 5, 256),
    };

    const TransientMemoryPlan plan = planTransientMemory(requests);

    for (const auto& allocation : plan.allocations) {
        REQUIRE(allocation.placed);
        REQUIRE(allocation.offset % 256 == 0);
    }
    requireNoOverlappingCollision(plan);
}

TEST_CASE("Zero-size and empty-lifetime requests are dropped, not packed")
{
    const std::vector<TransientAllocationRequest> requests = {
        request("real", 4 * kMiB, 0, 4),
        request("zero-size", 0, 0, 4),
        // firstPass > lastPass: no surviving pass touches it.
        request("never-used", 4 * kMiB, 5, 4),
    };

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.allocations.size() == 3);
    REQUIRE(plan.allocations[0].placed);
    REQUIRE_FALSE(plan.allocations[1].placed);
    REQUIRE_FALSE(plan.allocations[2].placed);

    // A dropped resource must not inflate either number.
    REQUIRE(plan.poolBytes == 4 * kMiB);
    REQUIRE(plan.unaliasedBytes == 4 * kMiB);
}

TEST_CASE("The plan does not depend on the order requests arrive in")
{
    std::vector<TransientAllocationRequest> forward = {
        request("alpha", 8 * kMiB, 0, 2),
        request("beta", 8 * kMiB, 1, 4),
        request("gamma", 8 * kMiB, 5, 7),
        request("delta", 2 * kMiB, 0, 7),
    };
    std::vector<TransientAllocationRequest> reversed(forward.rbegin(), forward.rend());

    const TransientMemoryPlan forwardPlan = planTransientMemory(forward);
    const TransientMemoryPlan reversedPlan = planTransientMemory(reversed);

    REQUIRE(forwardPlan.poolBytes == reversedPlan.poolBytes);

    // The committed golden image depends on which resources share bytes, so the
    // per-resource offsets have to match too, not just the total.
    for (const auto& allocation : forwardPlan.allocations) {
        const auto match = std::find_if(reversedPlan.allocations.begin(),
                                        reversedPlan.allocations.end(),
                                        [&allocation](const auto& other) { return other.name == allocation.name; });
        REQUIRE(match != reversedPlan.allocations.end());
        INFO("resource '" << allocation.name << "'");
        REQUIRE(match->offset == allocation.offset);
    }
}

TEST_CASE("Repeated planning of the same input is byte-identical")
{
    const std::vector<TransientAllocationRequest> requests = {
        request("one", 7 * kMiB, 0, 3),
        request("two", 7 * kMiB, 4, 6),
        request("three", 3 * kMiB, 2, 5),
    };

    const TransientMemoryPlan first = planTransientMemory(requests);
    const TransientMemoryPlan second = planTransientMemory(requests);

    REQUIRE(first.poolBytes == second.poolBytes);
    for (size_t index = 0; index < first.allocations.size(); ++index) {
        REQUIRE(first.allocations[index].offset == second.allocations[index].offset);
    }
}

TEST_CASE("A fully sequential chain collapses onto the largest resource")
{
    // The best case for aliasing: nothing is ever live at the same time.
    std::vector<TransientAllocationRequest> requests;
    for (uint32_t index = 0; index < 6; ++index) {
        requests.push_back(request("step" + std::to_string(index), (index + 1) * kMiB, index * 2, index * 2 + 1));
    }

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.poolBytes == 6 * kMiB);
    REQUIRE(plan.unaliasedBytes == 21 * kMiB);
    requireNoOverlappingCollision(plan);
}

TEST_CASE("Fully overlapping resources cannot be aliased at all")
{
    // The worst case: everything live everywhere, so the pool is the sum and the
    // reuse ratio is zero. This must not be reported as a saving.
    std::vector<TransientAllocationRequest> requests;
    for (uint32_t index = 0; index < 4; ++index) {
        requests.push_back(request("wide" + std::to_string(index), 5 * kMiB, 0, 9));
    }

    const TransientMemoryPlan plan = planTransientMemory(requests);

    REQUIRE(plan.poolBytes == 20 * kMiB);
    REQUIRE(plan.savedBytes() == 0);
    REQUIRE(plan.reuseRatio() == Approx(0.0));
    requireNoOverlappingCollision(plan);
}
