#include "renderer/RenderGraph.h"

#include "rhi/VulkanShadowMap.h"
#include "rhi/VulkanSwapchain.h"

#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

constexpr RenderResourceHandle kShadowMapDepth{"ShadowMapDepth"};
constexpr RenderResourceHandle kSwapchainColor{"SwapchainColor"};
constexpr RenderResourceHandle kMainDepth{"MainDepth"};
constexpr RenderResourceHandle kMaterialTextures{"MaterialTextures"};
constexpr RenderResourceHandle kIblResources{"IBLResources"};

} // namespace

RenderGraph::RenderGraph()
    : passes_{
          RenderPassNode{
              "ShadowPass",
              RenderPassType::Shadow,
              {
                  {kShadowMapDepth, RenderResourceAccess::Write, "Writes the directional shadow map depth image."},
              }},
          RenderPassNode{
              "MainPass",
              RenderPassType::Main,
              {
                  {kShadowMapDepth,
                   RenderResourceAccess::Read,
                   "Samples the shadow map from material set 0 binding 1."},
                  {kSwapchainColor, RenderResourceAccess::Write, "Writes the acquired swapchain color image."},
                  {kMainDepth,
                   RenderResourceAccess::Write,
                   "Clears and writes the main Dynamic Rendering depth image."},
                  {kMaterialTextures,
                   RenderResourceAccess::Read,
                   "Samples material textures from descriptor set 0 bindings 0, 2, and 3."},
                  {kIblResources,
                   RenderResourceAccess::Read,
                   "Samples diffuse IBL, prefiltered specular IBL, and the BRDF LUT."},
              }},
      }
{}

void RenderGraph::beginFrame(VkCommandBuffer commandBuffer,
                             rhi::VulkanSwapchain& swapchain,
                             rhi::VulkanShadowMap& shadowMap,
                             uint32_t imageIndex)
{
    if (frameActive_) {
        throw std::logic_error("RenderGraph::beginFrame called while a frame is already active.");
    }
    if (commandBuffer == VK_NULL_HANDLE) {
        throw std::logic_error("RenderGraph::beginFrame requires a valid command buffer.");
    }

    frame_.commandBuffer = commandBuffer;
    frame_.swapchain = &swapchain;
    frame_.shadowMap = &shadowMap;
    frame_.imageIndex = imageIndex;
    frame_.swapchainImage = swapchain.image(imageIndex);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    frameActive_ = true;
}

void RenderGraph::beginShadowPass()
{
    requireFrameActive("RenderGraph::beginShadowPass");
    if (shadowPassActive_ || mainPassActive_) {
        throw std::logic_error("RenderGraph::beginShadowPass called while another pass is active.");
    }

    transitionShadowMapImage(frame_.shadowMap->layout(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    frame_.shadowMap->setLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkClearValue shadowDepthClear{};
    shadowDepthClear.depthStencil.depth = 1.0f;
    shadowDepthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo shadowDepthAttachment{};
    shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    shadowDepthAttachment.imageView = frame_.shadowMap->imageView();
    shadowDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    shadowDepthAttachment.clearValue = shadowDepthClear;

    VkRenderingInfo shadowRenderingInfo{};
    shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    shadowRenderingInfo.renderArea.offset = {0, 0};
    shadowRenderingInfo.renderArea.extent = frame_.shadowMap->extent();
    shadowRenderingInfo.layerCount = 1;
    shadowRenderingInfo.colorAttachmentCount = 0;
    shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &shadowRenderingInfo);
    shadowPassActive_ = true;
}

void RenderGraph::endShadowPass()
{
    requireFrameActive("RenderGraph::endShadowPass");
    if (!shadowPassActive_) {
        throw std::logic_error("RenderGraph::endShadowPass called without an active shadow pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);

    transitionShadowMapImage(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    frame_.shadowMap->setLayout(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    shadowPassActive_ = false;
}

void RenderGraph::beginMainPass()
{
    requireFrameActive("RenderGraph::beginMainPass");
    if (shadowPassActive_ || mainPassActive_) {
        throw std::logic_error("RenderGraph::beginMainPass called while another pass is active.");
    }

    transitionSwapchainImage(frame_.swapchainImage,
                             frame_.swapchain->imageLayout(frame_.imageIndex),
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    frame_.swapchain->setImageLayout(frame_.imageIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    transitionDepthImage();

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.03f;
    clearColor.color.float32[1] = 0.04f;
    clearColor.color.float32[2] = 0.07f;
    clearColor.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = frame_.swapchain->imageView(frame_.imageIndex);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearColor;

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = frame_.swapchain->depthImageView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = frame_.swapchain->extent();
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    mainPassActive_ = true;
}

void RenderGraph::endMainPass()
{
    requireFrameActive("RenderGraph::endMainPass");
    if (!mainPassActive_) {
        throw std::logic_error("RenderGraph::endMainPass called without an active main pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);

    transitionSwapchainImage(
        frame_.swapchainImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    frame_.swapchain->setImageLayout(frame_.imageIndex, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    mainPassActive_ = false;
}

void RenderGraph::endFrame()
{
    requireFrameActive("RenderGraph::endFrame");
    if (shadowPassActive_ || mainPassActive_) {
        throw std::logic_error("RenderGraph::endFrame called while a pass is still active.");
    }

    VK_CHECK(vkEndCommandBuffer(frame_.commandBuffer));

    frame_ = {};
    frameActive_ = false;
}

void RenderGraph::requireFrameActive(const char* operation) const
{
    if (!frameActive_) {
        throw std::logic_error(std::string(operation) + " requires an active frame.");
    }
}

void RenderGraph::transitionShadowMapImage(VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        srcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        srcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        dstAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dstAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = frame_.shadowMap->image();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);
}

void RenderGraph::transitionSwapchainImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        srcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);
}

void RenderGraph::transitionDepthImage()
{
    const VkImageLayout oldLayout = frame_.swapchain->depthImageLayout();
    constexpr VkImageLayout newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        srcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = frame_.swapchain->depthImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);
    frame_.swapchain->setDepthImageLayout(newLayout);
}

} // namespace ve::renderer
