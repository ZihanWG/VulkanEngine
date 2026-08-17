#include "rhi/VulkanUploadBatch.h"

#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <algorithm>
#include <utility>

namespace ve::rhi {

bool uploadBatchShouldFlush(VkDeviceSize retainedBytes, VkDeviceSize pendingBytes, VkDeviceSize budgetBytes)
{
    // An empty batch always accepts the next upload, however large. Refusing here
    // would ask the caller to split a single staging allocation, which it cannot
    // do -- one image copy needs its source contiguous.
    if (retainedBytes == 0) {
        return false;
    }

    return retainedBytes + pendingBytes > budgetBytes;
}

VulkanUploadBatch::~VulkanUploadBatch()
{
    // Submitting from a destructor is deliberate: an exception partway through a
    // scene load would otherwise drop the staging buffers while queued copies
    // still read them.
    submitAndWait();
}

void VulkanUploadBatch::begin(VulkanContext& context, const VulkanCommandContext& commandContext)
{
    submitAndWait();

    context_ = &context;
    commandContext_ = &commandContext;
    submitCount_ = 0;
    peakStagingBytes_ = 0;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(context_->vkDevice(), &fenceInfo, nullptr, &fence_));

    openCommandBuffer();
}

void VulkanUploadBatch::openCommandBuffer()
{
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext_->commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context_->vkDevice(), &allocateInfo, &commandBuffer_));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer_, &beginInfo));

    empty_ = true;
}

void VulkanUploadBatch::releaseCommandBuffer()
{
    if (commandBuffer_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(context_->vkDevice(), commandContext_->commandPool(), 1, &commandBuffer_);
        commandBuffer_ = VK_NULL_HANDLE;
    }
}

void VulkanUploadBatch::flushIfOverBudget(VkDeviceSize pendingBytes)
{
    if (!recording()) {
        return;
    }

    if (uploadBatchShouldFlush(retainedStagingBytes_, pendingBytes)) {
        // Submit and reopen, keeping the fence and the counters. Tearing the
        // batch down and re-beginning it would reset the very diagnostics that
        // justify the budget.
        submitRecorded();
        openCommandBuffer();
    }
}

void VulkanUploadBatch::submitRecorded()
{
    if (commandBuffer_ == VK_NULL_HANDLE) {
        return;
    }

    VK_CHECK(vkEndCommandBuffer(commandBuffer_));

    // An untouched command buffer still costs a submit and a fence wait, and
    // buys nothing.
    if (!empty_) {
        VkCommandBufferSubmitInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandBufferInfo.commandBuffer = commandBuffer_;

        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;

        VK_CHECK(vkResetFences(context_->vkDevice(), 1, &fence_));
        VK_CHECK(vkQueueSubmit2(context_->graphicsQueue(), 1, &submitInfo, fence_));
        // A fence rather than vkQueueWaitIdle: this waits for exactly this
        // submission instead of draining everything the queue holds.
        VK_CHECK(vkWaitForFences(context_->vkDevice(), 1, &fence_, VK_TRUE, UINT64_MAX));
        ++submitCount_;
    }

    releaseCommandBuffer();

    // Only now are the copies known to have read their sources.
    retainedStaging_.clear();
    retainedStagingBytes_ = 0;
    empty_ = true;
}

void VulkanUploadBatch::retainStaging(VulkanBuffer&& staging, VkDeviceSize sizeBytes)
{
    retainedStaging_.push_back(std::move(staging));
    retainedStagingBytes_ += sizeBytes;
    peakStagingBytes_ = std::max(peakStagingBytes_, retainedStagingBytes_);
    empty_ = false;
}

void VulkanUploadBatch::submitAndWait()
{
    if (context_ == nullptr) {
        return;
    }

    submitRecorded();

    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(context_->vkDevice(), fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }

    context_ = nullptr;
    commandContext_ = nullptr;
}

} // namespace ve::rhi
