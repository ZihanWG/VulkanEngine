#pragma once

// Offset assignment for the render graph's transient memory pool.
//
// Given each transient resource's size, alignment, and the inclusive range of
// passes it is live across, decide where in one shared allocation each one sits.
// Resources whose lifetimes do not overlap may share bytes; resources whose
// lifetimes do overlap must not.
//
// Deliberately free of Vulkan and of the graph's own types: this is arithmetic
// over intervals and byte ranges, which is exactly the kind of policy that
// belongs in VulkanEngineCore so it can be exercised without a device. The
// graph supplies the lifetimes (see computeTextureLifetimes in RenderGraph.h)
// and the RHI supplies the sizes.
//
// The result must be deterministic for a given input: the committed golden image
// depends on what ends up sharing memory with what.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ve::renderer {

struct TransientAllocationRequest {
    std::string name;
    uint64_t size = 0;
    uint64_t alignment = 1;

    // Inclusive pass range, in graph recording order. firstPass > lastPass marks
    // a resource no surviving pass touches; it is dropped rather than packed.
    uint32_t firstPass = 0;
    uint32_t lastPass = 0;
};

struct TransientAllocation {
    std::string name;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t firstPass = 0;
    uint32_t lastPass = 0;

    // False for requests that were dropped (zero size, or an empty lifetime).
    // Such an entry is still returned so the result stays parallel to the input.
    bool placed = false;
};

struct TransientMemoryPlan {
    // Parallel to the request span, so a caller can zip the two without
    // rebuilding an index.
    std::vector<TransientAllocation> allocations;

    // Bytes the pool must actually reserve.
    uint64_t poolBytes = 0;

    // Bytes the same resources cost with one allocation each -- the "before"
    // number this whole exercise is measured against.
    uint64_t unaliasedBytes = 0;

    [[nodiscard]] uint64_t savedBytes() const { return unaliasedBytes > poolBytes ? unaliasedBytes - poolBytes : 0; }

    // 0.0 when nothing is shared, approaching 1.0 as more is. Zero when there is
    // nothing to place, rather than a division by zero.
    [[nodiscard]] double reuseRatio() const;
};

// Greedy interval packing: largest resource first, each placed at the lowest
// offset where no lifetime-overlapping resource already sits. Ties on size break
// on name so the result does not depend on request order.
//
// Greedy is not optimal, but the optimal version of this (dynamic storage
// allocation) is NP-hard, and the resource count here is in the tens.
[[nodiscard]] TransientMemoryPlan planTransientMemory(std::span<const TransientAllocationRequest> requests);

} // namespace ve::renderer
