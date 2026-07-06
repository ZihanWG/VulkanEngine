#include "rhi/VulkanAsyncCompute.h"

#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <string>
#include <vector>

namespace ve::rhi {

VulkanAsyncCompute::~VulkanAsyncCompute()
{
    cleanup();
}

void VulkanAsyncCompute::initialize(const VulkanContext& context, uint32_t frameCount)
{
    cleanup();

    if (!context.asyncComputeAvailable() || frameCount == 0) {
        return;
    }

    device_ = context.vkDevice();
    queue_ = context.asyncComputeQueue();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = context.asyncComputeQueueFamily();
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
    debug::setObjectName(device_, commandPool_, VK_OBJECT_TYPE_COMMAND_POOL, "AsyncComputeCommandPool");

    commandBuffers_.resize(frameCount, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = frameCount;
    VK_CHECK(vkAllocateCommandBuffers(device_, &allocateInfo, commandBuffers_.data()));

    semaphores_.resize(frameCount, VK_NULL_HANDLE);
    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &semaphores_[frameIndex]));
        debug::setObjectName(device_,
                             semaphores_[frameIndex],
                             VK_OBJECT_TYPE_SEMAPHORE,
                             "AsyncComputeFinished" + std::to_string(frameIndex));
    }

    available_ = true;
}

void VulkanAsyncCompute::cleanup()
{
    for (VkSemaphore semaphore : semaphores_) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }
    semaphores_.clear();

    commandBuffers_.clear();
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
    }

    queue_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    available_ = false;
}

} // namespace ve::rhi
