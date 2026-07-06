#pragma once

// Per-frame command buffers + binary semaphores for the async compute queue.
// Owns nothing when the device has no async queue (available() == false): the
// renderer then records its compute passes on the graphics queue as before.
//
// Synchronization model: the renderer records compute work into commandBuffer(f)
// and submits it to the async queue signaling semaphore(f) BEFORE the frame's
// graphics submission, which waits on semaphore(f) at the pipeline stage that
// first consumes the compute results. Command-buffer reuse is safe because the
// graphics submission's frame fence transitively covers the async work it
// waited on.

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <vector>

namespace ve::rhi {

class VulkanContext;

class VulkanAsyncCompute final {
public:
    VulkanAsyncCompute() = default;
    ~VulkanAsyncCompute();

    VulkanAsyncCompute(const VulkanAsyncCompute&) = delete;
    VulkanAsyncCompute& operator=(const VulkanAsyncCompute&) = delete;
    VulkanAsyncCompute(VulkanAsyncCompute&&) = delete;
    VulkanAsyncCompute& operator=(VulkanAsyncCompute&&) = delete;

    // No-op (available() stays false) when the context has no async queue.
    void initialize(const VulkanContext& context, uint32_t frameCount);
    void cleanup();

    [[nodiscard]] bool available() const { return available_; }
    [[nodiscard]] VkQueue queue() const { return queue_; }
    [[nodiscard]] VkCommandBuffer commandBuffer(uint32_t frameIndex) const
    {
        return commandBuffers_.at(frameIndex);
    }
    [[nodiscard]] VkSemaphore semaphore(uint32_t frameIndex) const
    {
        return semaphores_.at(frameIndex);
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore> semaphores_;
    bool available_ = false;
};

} // namespace ve::rhi
