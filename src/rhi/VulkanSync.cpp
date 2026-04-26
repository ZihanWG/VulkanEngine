#include "rhi/VulkanSync.h"

#include "renderer/FrameResources.h"
#include "rhi/VulkanContext.h"

namespace ve::rhi {

namespace {

VkSemaphoreCreateInfo semaphoreCreateInfo()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    return semaphoreInfo;
}

} // namespace

VulkanSync::~VulkanSync()
{
    cleanup();
}

void VulkanSync::initialize(const VulkanContext& context, std::span<renderer::FrameResources> frames, uint32_t swapchainImageCount)
{
    cleanup();

    device_ = context.vkDevice();
    frames_ = frames;

    const VkSemaphoreCreateInfo semaphoreInfo = semaphoreCreateInfo();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (renderer::FrameResources& frame : frames_) {
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frame.imageAvailable));
        VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &frame.inFlightFence));
    }

    recreateRenderFinishedSemaphores(swapchainImageCount);
}

void VulkanSync::recreateRenderFinishedSemaphores(uint32_t swapchainImageCount)
{
    cleanupRenderFinishedSemaphores();

    if (!device_ || swapchainImageCount == 0) {
        return;
    }

    const VkSemaphoreCreateInfo semaphoreInfo = semaphoreCreateInfo();
    renderFinishedSemaphores_.resize(swapchainImageCount);

    for (VkSemaphore& semaphore : renderFinishedSemaphores_) {
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &semaphore));
    }
}

void VulkanSync::cleanup()
{
    if (!device_) {
        frames_ = {};
        return;
    }

    for (renderer::FrameResources& frame : frames_) {
        if (frame.imageAvailable) {
            vkDestroySemaphore(device_, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }

        if (frame.inFlightFence) {
            vkDestroyFence(device_, frame.inFlightFence, nullptr);
            frame.inFlightFence = VK_NULL_HANDLE;
        }
    }

    frames_ = {};
    cleanupRenderFinishedSemaphores();
    device_ = VK_NULL_HANDLE;
}

void VulkanSync::cleanupRenderFinishedSemaphores()
{
    if (!device_) {
        renderFinishedSemaphores_.clear();
        return;
    }

    for (VkSemaphore semaphore : renderFinishedSemaphores_) {
        if (semaphore) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }

    renderFinishedSemaphores_.clear();
}

} // namespace ve::rhi
