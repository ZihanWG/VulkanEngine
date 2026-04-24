#pragma once

#include "rhi/VulkanCommon.h"

namespace ve::renderer {

struct FrameResources {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    // Signaled when vkAcquireNextImageKHR has ownership-ready image data for this frame.
    VkSemaphore imageAvailable = VK_NULL_HANDLE;

    // Signaled when graphics work has finished and presentation may read the swapchain image.
    VkSemaphore renderFinished = VK_NULL_HANDLE;

    // Keeps the CPU from reusing this frame slot before the GPU finished it.
    VkFence inFlightFence = VK_NULL_HANDLE;
};

} // namespace ve::renderer
