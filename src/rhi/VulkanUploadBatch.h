#pragma once

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <vector>

// Batched host-to-device upload for load-time resources (docs/asset_system.md).
//
// Every texture used to submit its own command buffer and then call
// vkQueueWaitIdle, draining the whole queue before the next one started. On
// Sponza that was half the texture upload time across 77 textures. This batches
// the copies into one command buffer and waits once on a fence.
//
// Load time only. The steady-state frame path has its own command buffers and
// its own synchronization, and must not go through here.
namespace ve::rhi {

class VulkanCommandContext;
class VulkanContext;

// How much staging memory may be in flight before the batch submits. Retaining
// every staging allocation until the end would spike by the size of the scene's
// textures -- ~91 MiB for cooked Sponza and ~366 MiB uncooked -- so the batch
// flushes instead. A handful of submits captures essentially all of the win that
// one submit would, without a footprint proportional to scene size.
inline constexpr VkDeviceSize kUploadBatchStagingBudgetBytes = 64ULL * 1024ULL * 1024ULL;

// Whether adding `pendingBytes` to a batch already holding `retainedBytes`
// should submit first. Pure so the policy is unit tested rather than inferred
// from a memory graph.
//
// A single allocation larger than the whole budget still has to go through, so
// this never asks a caller to split one upload -- it only refuses to let it
// share a batch with anything else.
[[nodiscard]] bool uploadBatchShouldFlush(VkDeviceSize retainedBytes,
                                          VkDeviceSize pendingBytes,
                                          VkDeviceSize budgetBytes = kUploadBatchStagingBudgetBytes);

class VulkanUploadBatch final {
public:
    VulkanUploadBatch() = default;
    ~VulkanUploadBatch();

    VulkanUploadBatch(const VulkanUploadBatch&) = delete;
    VulkanUploadBatch& operator=(const VulkanUploadBatch&) = delete;
    VulkanUploadBatch(VulkanUploadBatch&&) = delete;
    VulkanUploadBatch& operator=(VulkanUploadBatch&&) = delete;

    void begin(VulkanContext& context, const VulkanCommandContext& commandContext);

    // Records into this until submitted. Callers add their own barriers; the
    // batch takes no view on what is being copied.
    [[nodiscard]] VkCommandBuffer commandBuffer() const { return commandBuffer_; }
    [[nodiscard]] bool recording() const { return commandBuffer_ != VK_NULL_HANDLE; }

    // Submits and waits if `pendingBytes` would push the batch over budget, then
    // reopens it. Call before recording an upload of that size.
    void flushIfOverBudget(VkDeviceSize pendingBytes);

    // Keeps a staging allocation alive until the submit completes. Passing
    // ownership here rather than letting it die at the end of the caller's scope
    // is the whole reason the batch exists.
    void retainStaging(VulkanBuffer&& staging, VkDeviceSize sizeBytes);

    // Submits whatever has been recorded and waits for it. Safe to call on an
    // empty batch. The destructor calls this, so an exception mid-load cannot
    // leave GPU work referencing freed staging.
    void submitAndWait();

    [[nodiscard]] VkDeviceSize retainedStagingBytes() const { return retainedStagingBytes_; }
    // Diagnostics: how many submits the batch actually needed, and the largest
    // amount of staging alive at once. Both are evidence for the budget.
    [[nodiscard]] uint32_t submitCount() const { return submitCount_; }
    [[nodiscard]] VkDeviceSize peakStagingBytes() const { return peakStagingBytes_; }

private:
    void openCommandBuffer();
    void releaseCommandBuffer();
    // Submits and waits for what is recorded, then releases the staging it read.
    // Keeps the fence and the counters, so a mid-batch flush stays one batch.
    void submitRecorded();

    VulkanContext* context_ = nullptr;
    const VulkanCommandContext* commandContext_ = nullptr;

    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    std::vector<VulkanBuffer> retainedStaging_;
    VkDeviceSize retainedStagingBytes_ = 0;
    VkDeviceSize peakStagingBytes_ = 0;
    uint32_t submitCount_ = 0;
    // Nothing was recorded, so a submit would be a wasted round trip.
    bool empty_ = true;
};

} // namespace ve::rhi
