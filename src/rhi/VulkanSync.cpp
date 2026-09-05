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

    for (renderer::FrameResources& frame : frames_) {
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frame.imageAvailable));
        frame.timelineValue = 0;
    }

    // Starts at zero so every slot's initial timelineValue of zero is already
    // reached. The first submit signals 1.
    VkSemaphoreTypeCreateInfo timelineType{};
    timelineType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineType.initialValue = 0;

    VkSemaphoreCreateInfo timelineInfo = semaphoreCreateInfo();
    timelineInfo.pNext = &timelineType;
    VK_CHECK(vkCreateSemaphore(device_, &timelineInfo, nullptr, &frameTimeline_));

    recreateRenderFinishedSemaphores(swapchainImageCount);
}

void VulkanSync::waitForTimelineValue(uint64_t value) const
{
    if (frameTimeline_ == VK_NULL_HANDLE || value == 0) {
        return;
    }

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &frameTimeline_;
    waitInfo.pValues = &value;
    VK_CHECK(vkWaitSemaphores(device_, &waitInfo, UINT64_MAX));
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
        frame.timelineValue = 0;
    }

    if (frameTimeline_) {
        vkDestroySemaphore(device_, frameTimeline_, nullptr);
        frameTimeline_ = VK_NULL_HANDLE;
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
