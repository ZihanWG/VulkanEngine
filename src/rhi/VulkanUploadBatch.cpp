#include "rhi/VulkanUploadBatch.h"

#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <algorithm>
#include <utility>

namespace ve::rhi {

namespace {

// Both halves of a queue family ownership transfer, and the single-queue barrier
// that replaces them, differ only in the queue family indices and the scopes.
// Built in one place so the release and acquire cannot drift apart: the spec
// requires them to name identical layouts and identical family indices, and a
// mismatch is a VUID violation that may simply skip the transition.
VkImageMemoryBarrier2 shaderReadBarrier(VkImage image,
                                        uint32_t mipLevels,
                                        uint32_t srcQueueFamily,
                                        uint32_t dstQueueFamily,
                                        VkPipelineStageFlags2 srcStageMask,
                                        VkAccessFlags2 srcAccessMask,
                                        VkPipelineStageFlags2 dstStageMask,
                                        VkAccessFlags2 dstAccessMask)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = srcQueueFamily;
    barrier.dstQueueFamilyIndex = dstQueueFamily;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

void recordBarrier(VkCommandBuffer commandBuffer, const VkImageMemoryBarrier2& barrier)
{
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

} // namespace

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

    graphicsQueueFamily_ = context.queueFamilies().graphicsFamily.value();
    // Falling back to the graphics family makes usingTransferQueue() false and
    // leaves every path below on the single-submit form.
    transferQueueFamily_ =
        context.transferQueueAvailable() ? context.transferQueueFamily() : graphicsQueueFamily_;

    if (usingTransferQueue()) {
        // The transfer family needs its own pool: a command buffer may only be
        // submitted to the family its pool was created for.
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = transferQueueFamily_;
        VK_CHECK(vkCreateCommandPool(context_->vkDevice(), &poolInfo, nullptr, &transferPool_));

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(context_->vkDevice(), &semaphoreInfo, nullptr, &transferComplete_));
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(context_->vkDevice(), &fenceInfo, nullptr, &fence_));

    openCommandBuffer();
}

void VulkanUploadBatch::openCommandBuffer()
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = usingTransferQueue() ? transferPool_ : commandContext_->commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context_->vkDevice(), &allocateInfo, &commandBuffer_));
    VK_CHECK(vkBeginCommandBuffer(commandBuffer_, &beginInfo));

    if (usingTransferQueue()) {
        // Holds nothing but the acquire barriers, and must come from the
        // graphics pool because that is the queue it is submitted to.
        allocateInfo.commandPool = commandContext_->commandPool();
        VK_CHECK(vkAllocateCommandBuffers(context_->vkDevice(), &allocateInfo, &acquireCommandBuffer_));
        VK_CHECK(vkBeginCommandBuffer(acquireCommandBuffer_, &beginInfo));
    }

    pendingAcquires_.clear();
    empty_ = true;
}

void VulkanUploadBatch::releaseCommandBuffer()
{
    if (commandBuffer_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(context_->vkDevice(),
                             usingTransferQueue() ? transferPool_ : commandContext_->commandPool(),
                             1,
                             &commandBuffer_);
        commandBuffer_ = VK_NULL_HANDLE;
    }
    if (acquireCommandBuffer_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(context_->vkDevice(), commandContext_->commandPool(), 1, &acquireCommandBuffer_);
        acquireCommandBuffer_ = VK_NULL_HANDLE;
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
        // The acquire barriers go in first, before the graphics buffer is ended.
        for (const PendingAcquire& acquire : pendingAcquires_) {
            recordBarrier(acquireCommandBuffer_,
                          shaderReadBarrier(acquire.image,
                                            acquire.mipLevels,
                                            transferQueueFamily_,
                                            graphicsQueueFamily_,
                                            VK_PIPELINE_STAGE_2_NONE,
                                            VK_ACCESS_2_NONE,
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
        }

        VkCommandBufferSubmitInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandBufferInfo.commandBuffer = commandBuffer_;

        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;

        VK_CHECK(vkResetFences(context_->vkDevice(), 1, &fence_));

        if (usingTransferQueue()) {
            VK_CHECK(vkEndCommandBuffer(acquireCommandBuffer_));

            // The semaphore is not optional: a queue family ownership transfer
            // needs an execution dependency between the two submissions, or the
            // acquire can run before the release.
            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = transferComplete_;
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            submitInfo.signalSemaphoreInfoCount = 1;
            submitInfo.pSignalSemaphoreInfos = &signalInfo;
            VK_CHECK(vkQueueSubmit2(context_->transferQueue(), 1, &submitInfo, VK_NULL_HANDLE));

            VkSemaphoreSubmitInfo waitInfo = signalInfo;
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            VkCommandBufferSubmitInfo acquireInfo{};
            acquireInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            acquireInfo.commandBuffer = acquireCommandBuffer_;

            VkSubmitInfo2 acquireSubmit{};
            acquireSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            acquireSubmit.waitSemaphoreInfoCount = 1;
            acquireSubmit.pWaitSemaphoreInfos = &waitInfo;
            acquireSubmit.commandBufferInfoCount = 1;
            acquireSubmit.pCommandBufferInfos = &acquireInfo;
            // The fence hangs off the graphics submit, so waiting on it covers
            // both halves.
            VK_CHECK(vkQueueSubmit2(context_->graphicsQueue(), 1, &acquireSubmit, fence_));
        } else {
            VK_CHECK(vkQueueSubmit2(context_->graphicsQueue(), 1, &submitInfo, fence_));
        }

        // A fence rather than vkQueueWaitIdle: this waits for exactly this
        // submission instead of draining everything the queue holds.
        VK_CHECK(vkWaitForFences(context_->vkDevice(), 1, &fence_, VK_TRUE, UINT64_MAX));
        ++submitCount_;
    } else if (usingTransferQueue()) {
        VK_CHECK(vkEndCommandBuffer(acquireCommandBuffer_));
    }

    releaseCommandBuffer();

    // Only now are the copies known to have read their sources.
    retainedStaging_.clear();
    retainedStagingBytes_ = 0;
    empty_ = true;
}

void VulkanUploadBatch::releaseImageToGraphics(VkImage image, uint32_t mipLevels)
{
    if (!recording()) {
        return;
    }

    if (!usingTransferQueue()) {
        // One queue, one barrier: transfer writes become fragment-shader reads.
        recordBarrier(commandBuffer_,
                      shaderReadBarrier(image,
                                        mipLevels,
                                        VK_QUEUE_FAMILY_IGNORED,
                                        VK_QUEUE_FAMILY_IGNORED,
                                        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                        VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
        empty_ = false;
        return;
    }

    // Release half, on the transfer queue. It names no destination scope: the
    // ownership transfer, not this barrier, is what makes the data visible, and
    // the transfer queue cannot express a fragment-shader stage anyway.
    recordBarrier(commandBuffer_,
                  shaderReadBarrier(image,
                                    mipLevels,
                                    transferQueueFamily_,
                                    graphicsQueueFamily_,
                                    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_NONE,
                                    VK_ACCESS_2_NONE));

    // The acquire half is recorded at submit time, into the graphics command
    // buffer, after the semaphore wait.
    pendingAcquires_.push_back(PendingAcquire{image, mipLevels});
    empty_ = false;
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
    if (transferComplete_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(context_->vkDevice(), transferComplete_, nullptr);
        transferComplete_ = VK_NULL_HANDLE;
    }
    if (transferPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context_->vkDevice(), transferPool_, nullptr);
        transferPool_ = VK_NULL_HANDLE;
    }

    pendingAcquires_.clear();
    context_ = nullptr;
    commandContext_ = nullptr;
}

} // namespace ve::rhi
