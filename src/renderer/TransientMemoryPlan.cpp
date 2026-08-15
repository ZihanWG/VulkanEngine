#include "renderer/TransientMemoryPlan.h"

#include <algorithm>
#include <numeric>

namespace ve::renderer {

namespace {

[[nodiscard]] uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

[[nodiscard]] bool lifetimesOverlap(const TransientAllocation& lhs, const TransientAllocationRequest& rhs)
{
    return !(lhs.lastPass < rhs.firstPass || rhs.lastPass < lhs.firstPass);
}

// Lowest offset at or above `alignment` boundaries where [offset, offset+size)
// clears every already-placed block it must avoid. `blocked` is sorted by offset.
[[nodiscard]] uint64_t lowestFreeOffset(const std::vector<const TransientAllocation*>& blocked,
                                        uint64_t size,
                                        uint64_t alignment)
{
    uint64_t candidate = 0;
    for (const TransientAllocation* placed : blocked) {
        candidate = alignUp(candidate, alignment);
        if (candidate + size <= placed->offset) {
            // Fits in the gap before this block.
            return candidate;
        }
        candidate = std::max(candidate, placed->offset + placed->size);
    }
    return alignUp(candidate, alignment);
}

} // namespace

double TransientMemoryPlan::reuseRatio() const
{
    if (unaliasedBytes == 0) {
        return 0.0;
    }
    return 1.0 - static_cast<double>(poolBytes) / static_cast<double>(unaliasedBytes);
}

TransientMemoryPlan planTransientMemory(std::span<const TransientAllocationRequest> requests)
{
    TransientMemoryPlan plan{};
    plan.allocations.resize(requests.size());

    // Seed every entry so the result stays parallel to the input even for the
    // requests that end up dropped.
    for (size_t index = 0; index < requests.size(); ++index) {
        const TransientAllocationRequest& request = requests[index];
        plan.allocations[index].name = request.name;
        plan.allocations[index].size = request.size;
        plan.allocations[index].firstPass = request.firstPass;
        plan.allocations[index].lastPass = request.lastPass;
    }

    // Largest first, then by name. The name tie-break is what makes the plan
    // independent of the order the graph happened to declare resources in.
    std::vector<size_t> order;
    order.reserve(requests.size());
    for (size_t index = 0; index < requests.size(); ++index) {
        const TransientAllocationRequest& request = requests[index];
        if (request.size == 0 || request.firstPass > request.lastPass) {
            continue;
        }
        plan.unaliasedBytes += alignUp(request.size, request.alignment);
        order.push_back(index);
    }

    std::sort(order.begin(), order.end(), [&requests](size_t lhs, size_t rhs) {
        if (requests[lhs].size != requests[rhs].size) {
            return requests[lhs].size > requests[rhs].size;
        }
        return requests[lhs].name < requests[rhs].name;
    });

    std::vector<const TransientAllocation*> blocked;
    for (const size_t index : order) {
        const TransientAllocationRequest& request = requests[index];

        blocked.clear();
        for (const size_t placedIndex : order) {
            const TransientAllocation& candidate = plan.allocations[placedIndex];
            if (candidate.placed && lifetimesOverlap(candidate, request)) {
                blocked.push_back(&candidate);
            }
        }
        std::sort(blocked.begin(), blocked.end(), [](const TransientAllocation* lhs, const TransientAllocation* rhs) {
            return lhs->offset < rhs->offset;
        });

        TransientAllocation& allocation = plan.allocations[index];
        allocation.offset = lowestFreeOffset(blocked, request.size, std::max<uint64_t>(request.alignment, 1));
        allocation.placed = true;
        plan.poolBytes = std::max(plan.poolBytes, allocation.offset + request.size);
    }

    return plan;
}

} // namespace ve::renderer
