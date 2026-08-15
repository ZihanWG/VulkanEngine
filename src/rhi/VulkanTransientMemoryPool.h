#pragma once

// One device-local allocation that several transient images are bound into at
// different offsets, so resources whose lifetimes do not overlap can share bytes.
//
// The pool owns memory and nothing else. Images bound into it are still created,
// owned, viewed, and described by whoever owned them before -- that separation is
// what keeps this from becoming a descriptor migration across PostProcessStack,
// ScreenSpaceReflections, and GroundTruthAmbientOcclusion.
//
// Offsets come from renderer::planTransientMemory, which guarantees that any two
// resources sharing bytes have disjoint lifetimes. The pool does not re-check
// that: it cannot, having no idea what a lifetime is. A wrong plan corrupts
// pixels without tripping validation, which is why the planner is the part with
// the exhaustive tests.
//
// Availability is not assumed anywhere. A driver that refuses the allocation, or
// a resource set with no memory type in common, leaves available() == false and
// every caller falls back to a private allocation per image -- which is what the
// engine did before this existed.

#include "rhi/VulkanCommon.h"
#include "rhi/VulkanMemory.h"

#include <cstdint>

namespace ve::rhi {

class VulkanContext;

class VulkanTransientMemoryPool final {
public:
    VulkanTransientMemoryPool() = default;
    ~VulkanTransientMemoryPool();

    VulkanTransientMemoryPool(const VulkanTransientMemoryPool&) = delete;
    VulkanTransientMemoryPool& operator=(const VulkanTransientMemoryPool&) = delete;
    VulkanTransientMemoryPool(VulkanTransientMemoryPool&&) = delete;
    VulkanTransientMemoryPool& operator=(VulkanTransientMemoryPool&&) = delete;

    // memoryTypeBits must be the intersection of every image's requirement. Zero
    // means no type accepts them all, and the pool declines rather than picking
    // one that only suits some -- a decline is a supported outcome, not an error.
    //
    // Returns available(). Safe to call on a live pool; the previous allocation
    // is released first, so every image bound into it must already be destroyed.
    bool create(VulkanContext& context, VkDeviceSize sizeBytes, uint32_t memoryTypeBits, VkDeviceSize alignment);
    void reset();

    [[nodiscard]] bool available() const { return allocation_ != VK_NULL_HANDLE; }
    [[nodiscard]] VmaAllocation allocation() const { return allocation_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }

    // Why the pool is unavailable, for the log line that explains a fallback.
    [[nodiscard]] const char* unavailableReason() const { return unavailableReason_; }

private:
    VulkanContext* context_ = nullptr;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    const char* unavailableReason_ = "not created";
};

} // namespace ve::rhi
