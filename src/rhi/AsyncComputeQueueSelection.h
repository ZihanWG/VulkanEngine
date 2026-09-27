#pragma once

// Which queue, if any, should carry the async compute passes.
//
// Split out from VulkanDevice for the same reason as TransferQueueSelection.h:
// the choice is a pure function of the family list, and each branch is real on
// some machine. The RTX 3080 Ti Laptop exposes a compute-only family; MoltenVK
// exposes one only under MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1 and otherwise
// has neither that nor a second graphics queue, so compute stays inline.

#include "rhi/TransferQueueSelection.h"

#include <cstdint>
#include <span>

namespace ve::rhi {

struct AsyncComputeQueueChoice {
    uint32_t family = kNoQueueFamily;
    uint32_t queueIndex = 0;
    // A compute-only family, which runs on the GPU's compute ring and overlaps
    // rasterization best. False for the second-graphics-queue fallback, which
    // still lets the driver interleave.
    bool dedicatedFamily = false;

    [[nodiscard]] bool available() const
    {
        return family != kNoQueueFamily;
    }
};

// The first compute-only family if there is one, else a second queue in the
// graphics family, else nothing -- and compute then stays on the graphics
// queue.
[[nodiscard]] AsyncComputeQueueChoice selectAsyncComputeQueue(std::span<const QueueFamilyCapabilities> families,
                                                              uint32_t graphicsFamily);

} // namespace ve::rhi
