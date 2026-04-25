#pragma once

#include "renderer/FrameResources.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanPipeline.h"
#include "rhi/VulkanSwapchain.h"
#include "rhi/VulkanSync.h"

#include <chrono>
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
    void createMvpDescriptorSetLayout();
    void createPipeline();
    void createGeometryBuffers();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void updateFrameData(uint32_t frameIndex);
    void recreateSwapchain();
    void recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void transitionSwapchainImage(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout);
    void transitionDepthImage(VkCommandBuffer commandBuffer);

    Window& window_;
    rhi::VulkanContext context_;
    std::vector<renderer::FrameResources> frames_;
    rhi::VulkanSwapchain swapchain_;
    rhi::VulkanDescriptorSetLayout mvpDescriptorSetLayout_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanCommandContext commandContext_;
    rhi::VulkanSync sync_;
    rhi::VulkanBuffer vertexBuffer_;
    rhi::VulkanBuffer indexBuffer_;
    std::vector<rhi::VulkanBuffer> uniformBuffers_;
    rhi::VulkanDescriptorPool descriptorPool_;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<VkFence> imagesInFlight_;
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t indexCount_ = 0;
    uint32_t currentFrame_ = 0;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    bool initialized_ = false;
};

} // namespace ve
