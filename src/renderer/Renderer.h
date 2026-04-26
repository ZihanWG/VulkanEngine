#pragma once

#include "renderer/Camera.h"
#include "renderer/FrameResources.h"
#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/RenderObject.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanPipeline.h"
#include "rhi/VulkanSwapchain.h"
#include "rhi/VulkanSync.h"
#include "rhi/VulkanTexture.h"

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
    void createMaterialDescriptorSetLayout();
    void createPipeline();
    void createScene();
    void createCheckerboardTexture();
    void createMaterial();
    void createMaterialDescriptorSet(renderer::Material& material);
    void createObjectFrameDataBuffers();
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
    rhi::VulkanDescriptorSetLayout materialDescriptorSetLayout_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanCommandContext commandContext_;
    rhi::VulkanSync sync_;
    rhi::VulkanTexture checkerboardTexture_;
    rhi::VulkanDescriptorPool materialDescriptorPool_;
    renderer::Camera camera_;
    renderer::Mesh cubeMesh_;
    renderer::Material checkerboardMaterial_;
    std::vector<renderer::RenderObject> renderObjects_;
    std::vector<rhi::VulkanBuffer> frameObjectDataBuffers_;
    std::vector<VkFence> imagesInFlight_;
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t currentFrame_ = 0;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    bool initialized_ = false;
};

} // namespace ve
