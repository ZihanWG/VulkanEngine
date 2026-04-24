#include "rhi/VulkanCommandContext.h"

#include "renderer/FrameResources.h"
#include "rhi/VulkanContext.h"

#include <vector>

namespace ve::rhi {

VulkanCommandContext::~VulkanCommandContext()
{
    cleanup();
}

void VulkanCommandContext::initialize(const VulkanContext& context, std::span<renderer::FrameResources> frames)
{
    cleanup();

    device_ = context.vkDevice();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = context.queueFamilies().graphicsFamily.value();
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));

    std::vector<VkCommandBuffer> commandBuffers(frames.size());
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    VK_CHECK(vkAllocateCommandBuffers(device_, &allocateInfo, commandBuffers.data()));

    for (size_t index = 0; index < frames.size(); ++index) {
        frames[index].commandBuffer = commandBuffers[index];
    }
}

void VulkanCommandContext::cleanup()
{
    if (commandPool_) {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
}

} // namespace ve::rhi
