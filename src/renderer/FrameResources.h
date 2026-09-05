#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>

namespace ve::renderer {

struct FrameResources {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    // Signaled when vkAcquireNextImageKHR has ownership-ready image data for this
    // frame. Still a binary semaphore: VK_KHR_swapchain does not accept a timeline
    // one on either acquire or present.
    VkSemaphore imageAvailable = VK_NULL_HANDLE;

    // The frame-timeline value this slot's last submission signals. Waiting for it
    // is what keeps the CPU from reusing the slot before the GPU finished with it,
    // and it replaces the per-slot fence.
    //
    // Zero means "never submitted", and a wait for zero on a timeline that starts
    // at zero returns immediately -- the same thing the fence's
    // VK_FENCE_CREATE_SIGNALED_BIT used to buy. The value is assigned only after a
    // successful submit, so a frame that returns early (an out-of-date acquire) or
    // throws leaves the slot on its previous, already-signalled value rather than
    // on one nothing will ever signal.
    uint64_t timelineValue = 0;
};

} // namespace ve::renderer
