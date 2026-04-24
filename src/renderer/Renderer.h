#pragma once

#include "renderer/FrameResources.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanPipeline.h"
#include "rhi/VulkanSwapchain.h"
#include "rhi/VulkanSync.h"

#include <vector>

namespace ve {

class Window;

class Renderer final {
public:
    explicit Renderer(Window& window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void drawFrame();
    void waitIdle();

private:
    void createPipeline();
    void recreateSwapchain();
    void recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void transitionSwapchainImage(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout);

    Window& window_;
    rhi::VulkanContext context_;
    std::vector<renderer::FrameResources> frames_;
    rhi::VulkanSwapchain swapchain_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanCommandContext commandContext_;
    rhi::VulkanSync sync_;
    std::vector<VkFence> imagesInFlight_;
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t currentFrame_ = 0;
    bool initialized_ = false;
};

} // namespace ve
