#include "rhi/VulkanSync.h"

#include "renderer/FrameResources.h"
#include "rhi/VulkanContext.h"

namespace ve::rhi {

VulkanSync::~VulkanSync()
{
    cleanup();
}

void VulkanSync::initialize(const VulkanContext& context, std::span<renderer::FrameResources> frames)
{
    cleanup();

    device_ = context.vkDevice();
    frames_ = frames;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (renderer::FrameResources& frame : frames_) {
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frame.imageAvailable));
        VK_CHECK(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frame.renderFinished));
        VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &frame.inFlightFence));
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

        if (frame.renderFinished) {
            vkDestroySemaphore(device_, frame.renderFinished, nullptr);
            frame.renderFinished = VK_NULL_HANDLE;
        }

        if (frame.inFlightFence) {
            vkDestroyFence(device_, frame.inFlightFence, nullptr);
            frame.inFlightFence = VK_NULL_HANDLE;
        }
    }

    frames_ = {};
    device_ = VK_NULL_HANDLE;
}

} // namespace ve::rhi
