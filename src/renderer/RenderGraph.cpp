#include "renderer/RenderGraph.h"

#include "rhi/VulkanShadowMap.h"
#include "rhi/VulkanSwapchain.h"

#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

constexpr RenderResourceHandle kShadowMapDepth{"ShadowMapDepth"};
constexpr RenderResourceHandle kSwapchainColor{"SwapchainColor"};
constexpr RenderResourceHandle kSceneColor{"SceneColor"};
constexpr RenderResourceHandle kBloomExtract{"BloomExtract"};
constexpr RenderResourceHandle kBloomPing{"BloomPing"};
constexpr RenderResourceHandle kBloomPong{"BloomPong"};
constexpr RenderResourceHandle kLuminancePartials{"LuminancePartials"};
constexpr RenderResourceHandle kLuminanceHistogram{"LuminanceHistogram"};
constexpr RenderResourceHandle kMainDepth{"MainDepth"};
constexpr RenderResourceHandle kMaterialTextures{"MaterialTextures"};
constexpr RenderResourceHandle kIblResources{"IBLResources"};

bool validImageResource(const RenderGraphImageResource& resource)
{
    return resource.image != VK_NULL_HANDLE && resource.imageView != VK_NULL_HANDLE && resource.layout != nullptr &&
           resource.extent.width > 0 && resource.extent.height > 0;
}

void requireImageResource(const RenderGraphImageResource& resource, const char* operation)
{
    if (!validImageResource(resource)) {
        throw std::logic_error(std::string(operation) + " requires a valid " + resource.name + " image resource.");
    }
}

} // namespace

RenderGraph::RenderGraph()
    : passes_{
          RenderPassNode{
              "CSMShadowPass",
              RenderPassType::Shadow,
              {
                  {kShadowMapDepth,
                   RenderResourceAccess::Write,
                   "Writes every cascaded shadow-map array layer with material-independent shadow-caster draws."},
              }},
          RenderPassNode{
              "MainHDRPass",
              RenderPassType::MainHdr,
              {
                  {kShadowMapDepth,
                   RenderResourceAccess::Read,
                   "Samples the cascaded shadow-map array from global/material set 0 binding 1."},
                  {kSceneColor,
                   RenderResourceAccess::Write,
                   "Writes linear HDR skybox and mesh lighting into an offscreen color target."},
                  {kMainDepth,
                   RenderResourceAccess::Write,
                   "Clears and writes the main Dynamic Rendering depth image."},
                  {kMaterialTextures,
                   RenderResourceAccess::Read,
                   "Samples material textures from bindless set 1 when available, otherwise descriptor set 0."},
                  {kIblResources,
                   RenderResourceAccess::Read,
                   "Samples diffuse IBL, prefiltered specular IBL, and the BRDF LUT."},
              }},
          RenderPassNode{"BloomExtractPass",
                         RenderPassType::BloomExtract,
                         {
                             {kSceneColor, RenderResourceAccess::Read, "Samples the HDR scene color target."},
                             {kBloomExtract,
                              RenderResourceAccess::Write,
                              "Writes bright pixels above the bloom threshold into the bloom extraction target."},
                         }},
          RenderPassNode{"BloomBlurPass",
                         RenderPassType::BloomBlur,
                         {
                             {kBloomExtract, RenderResourceAccess::Read, "Samples extracted bloom highlights."},
                             {kBloomPing, RenderResourceAccess::Write, "Writes the horizontal blur result."},
                             {kBloomPing, RenderResourceAccess::Read, "Samples the horizontal blur result."},
                             {kBloomPong, RenderResourceAccess::Write, "Writes the vertical blur result."},
                         }},
          RenderPassNode{"LuminancePass",
                         RenderPassType::Luminance,
                         {
                             {kSceneColor,
                              RenderResourceAccess::Read,
                              "Samples the HDR scene color target for log-average luminance reduction."},
                             {kLuminancePartials,
                              RenderResourceAccess::Write,
                              "Writes per-workgroup log-luminance partial sums for CPU readback."},
                         }},
          RenderPassNode{"HistogramExposurePass",
                         RenderPassType::HistogramExposure,
                         {
                             {kSceneColor,
                              RenderResourceAccess::Read,
                              "Samples the HDR scene color target for log2 luminance histogram binning."},
                             {kLuminanceHistogram,
                              RenderResourceAccess::Write,
                              "Writes 256 luminance histogram bin counts for CPU percentile readback."},
                         }},
          RenderPassNode{"CompositePass",
                         RenderPassType::Composite,
                         {
                             {kSceneColor, RenderResourceAccess::Read, "Samples the HDR scene color target."},
                             {kBloomPong, RenderResourceAccess::Read, "Samples the blurred bloom texture."},
                             {kSwapchainColor,
                              RenderResourceAccess::Write,
                              "Writes the exposed, tone-mapped final color to the acquired swapchain image."},
                         }},
          RenderPassNode{"ImGuiPass",
                         RenderPassType::ImGui,
                         {
                             {kSwapchainColor,
                              RenderResourceAccess::Write,
                              "Loads the composited swapchain color attachment and draws the debug UI overlay."},
                         }},
      }
{}

void RenderGraph::beginFrame(VkCommandBuffer commandBuffer,
                             rhi::VulkanSwapchain& swapchain,
                             rhi::VulkanShadowMap& shadowMap,
                             uint32_t imageIndex,
                             RenderGraphFrameResources frameResources)
{
    if (frameActive_) {
        throw std::logic_error("RenderGraph::beginFrame called while a frame is already active.");
    }
    if (commandBuffer == VK_NULL_HANDLE) {
        throw std::logic_error("RenderGraph::beginFrame requires a valid command buffer.");
    }
    requireImageResource(frameResources.sceneColor, "RenderGraph::beginFrame");
    requireImageResource(frameResources.bloomExtract, "RenderGraph::beginFrame");
    requireImageResource(frameResources.bloomPing, "RenderGraph::beginFrame");
    requireImageResource(frameResources.bloomPong, "RenderGraph::beginFrame");

    frame_.commandBuffer = commandBuffer;
    frame_.swapchain = &swapchain;
    frame_.shadowMap = &shadowMap;
    frame_.resources = frameResources;
    frame_.imageIndex = imageIndex;
    frame_.swapchainImage = swapchain.image(imageIndex);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    frameActive_ = true;
    activePass_ = ActivePass::None;
}

void RenderGraph::beginShadowPass(uint32_t cascadeLayer)
{
    requireFrameActive("RenderGraph::beginShadowPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginShadowPass called while another pass is active.");
    }
    if (cascadeLayer >= frame_.shadowMap->layerCount()) {
        throw std::out_of_range("RenderGraph::beginShadowPass cascade layer is out of range.");
    }

    if (frame_.shadowMap->layout() != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        transitionShadowMapImage(frame_.shadowMap->layout(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        frame_.shadowMap->setLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    }

    VkClearValue shadowDepthClear{};
    shadowDepthClear.depthStencil.depth = 1.0f;
    shadowDepthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo shadowDepthAttachment{};
    shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    shadowDepthAttachment.imageView = frame_.shadowMap->layerImageView(cascadeLayer);
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
    activePass_ = ActivePass::Shadow;
}

void RenderGraph::endShadowPass(bool finalCascade)
{
    requireFrameActive("RenderGraph::endShadowPass");
    if (activePass_ != ActivePass::Shadow) {
        throw std::logic_error("RenderGraph::endShadowPass called without an active shadow pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);

    if (finalCascade) {
        transitionShadowMapImage(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
        frame_.shadowMap->setLayout(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    }
    activePass_ = ActivePass::None;
}

void RenderGraph::beginMainHdrPass()
{
    requireFrameActive("RenderGraph::beginMainHdrPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginMainHdrPass called while another pass is active.");
    }

    transitionColorImage(frame_.resources.sceneColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    transitionDepthImage();

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.03f;
    clearColor.color.float32[1] = 0.04f;
    clearColor.color.float32[2] = 0.07f;
    clearColor.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = frame_.resources.sceneColor.imageView;
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
    renderingInfo.renderArea.extent = frame_.resources.sceneColor.extent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::MainHdr;
}

void RenderGraph::endMainHdrPass()
{
    requireFrameActive("RenderGraph::endMainHdrPass");
    if (activePass_ != ActivePass::MainHdr) {
        throw std::logic_error("RenderGraph::endMainHdrPass called without an active main HDR pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginBloomExtractPass()
{
    requireFrameActive("RenderGraph::beginBloomExtractPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginBloomExtractPass called while another pass is active.");
    }

    transitionColorImage(frame_.resources.sceneColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transitionColorImage(frame_.resources.bloomExtract, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginColorRendering(frame_.resources.bloomExtract, clearColor);
    activePass_ = ActivePass::BloomExtract;
}

void RenderGraph::endBloomExtractPass()
{
    requireFrameActive("RenderGraph::endBloomExtractPass");
    if (activePass_ != ActivePass::BloomExtract) {
        throw std::logic_error("RenderGraph::endBloomExtractPass called without an active bloom extract pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginBloomBlurPass(bool horizontal)
{
    requireFrameActive("RenderGraph::beginBloomBlurPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginBloomBlurPass called while another pass is active.");
    }

    RenderGraphImageResource& input = horizontal ? frame_.resources.bloomExtract : frame_.resources.bloomPing;
    RenderGraphImageResource& output = horizontal ? frame_.resources.bloomPing : frame_.resources.bloomPong;
    transitionColorImage(input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transitionColorImage(output, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginColorRendering(output, clearColor);
    activePass_ = ActivePass::BloomBlur;
}

void RenderGraph::endBloomBlurPass()
{
    requireFrameActive("RenderGraph::endBloomBlurPass");
    if (activePass_ != ActivePass::BloomBlur) {
        throw std::logic_error("RenderGraph::endBloomBlurPass called without an active bloom blur pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginLuminancePass()
{
    requireFrameActive("RenderGraph::beginLuminancePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginLuminancePass called while another pass is active.");
    }

    transitionColorImage(frame_.resources.sceneColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    activePass_ = ActivePass::Luminance;
}

void RenderGraph::endLuminancePass()
{
    requireFrameActive("RenderGraph::endLuminancePass");
    if (activePass_ != ActivePass::Luminance) {
        throw std::logic_error("RenderGraph::endLuminancePass called without an active luminance pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginHistogramExposurePass()
{
    requireFrameActive("RenderGraph::beginHistogramExposurePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginHistogramExposurePass called while another pass is active.");
    }

    transitionColorImage(frame_.resources.sceneColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    activePass_ = ActivePass::HistogramExposure;
}

void RenderGraph::endHistogramExposurePass()
{
    requireFrameActive("RenderGraph::endHistogramExposurePass");
    if (activePass_ != ActivePass::HistogramExposure) {
        throw std::logic_error(
            "RenderGraph::endHistogramExposurePass called without an active histogram exposure pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginCompositePass()
{
    requireFrameActive("RenderGraph::beginCompositePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginCompositePass called while another pass is active.");
    }

    transitionColorImage(frame_.resources.sceneColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transitionColorImage(frame_.resources.bloomPong, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transitionSwapchainImage(frame_.swapchainImage,
                             frame_.swapchain->imageLayout(frame_.imageIndex),
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    frame_.swapchain->setImageLayout(frame_.imageIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginSwapchainRendering(clearColor, VK_ATTACHMENT_LOAD_OP_CLEAR);
    activePass_ = ActivePass::Composite;
}

void RenderGraph::endCompositePass()
{
    requireFrameActive("RenderGraph::endCompositePass");
    if (activePass_ != ActivePass::Composite) {
        throw std::logic_error("RenderGraph::endCompositePass called without an active composite pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginImGuiPass()
{
    requireFrameActive("RenderGraph::beginImGuiPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginImGuiPass called while another pass is active.");
    }

    if (frame_.swapchain->imageLayout(frame_.imageIndex) != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        transitionSwapchainImage(frame_.swapchainImage,
                                 frame_.swapchain->imageLayout(frame_.imageIndex),
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        frame_.swapchain->setImageLayout(frame_.imageIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    } else {
        barrierSwapchainColorAttachment();
    }

    VkClearValue clearColor{};
    beginSwapchainRendering(clearColor, VK_ATTACHMENT_LOAD_OP_LOAD);
    activePass_ = ActivePass::ImGui;
}

void RenderGraph::endImGuiPass()
{
    requireFrameActive("RenderGraph::endImGuiPass");
    if (activePass_ != ActivePass::ImGui) {
        throw std::logic_error("RenderGraph::endImGuiPass called without an active ImGui pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    transitionSwapchainImage(
        frame_.swapchainImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    frame_.swapchain->setImageLayout(frame_.imageIndex, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    activePass_ = ActivePass::None;
}

void RenderGraph::endFrame()
{
    requireFrameActive("RenderGraph::endFrame");
    if (activePass_ != ActivePass::None) {
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
    barrier.subresourceRange.layerCount = frame_.shadowMap->layerCount();

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);
}

void RenderGraph::transitionSwapchainImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    if (oldLayout == newLayout) {
        return;
    }

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

void RenderGraph::barrierSwapchainColorAttachment()
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = frame_.swapchainImage;
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

void RenderGraph::transitionColorImage(RenderGraphImageResource& resource, VkImageLayout newLayout)
{
    requireImageResource(resource, "RenderGraph::transitionColorImage");

    const VkImageLayout oldLayout = *resource.layout;
    if (oldLayout == newLayout) {
        return;
    }

    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        srcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        srcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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
    barrier.image = resource.image;
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
    *resource.layout = newLayout;
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

void RenderGraph::beginColorRendering(const RenderGraphImageResource& resource, VkClearValue clearValue)
{
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = resource.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = resource.extent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
}

void RenderGraph::beginSwapchainRendering(VkClearValue clearValue, VkAttachmentLoadOp loadOp)
{
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = frame_.swapchain->imageView(frame_.imageIndex);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = loadOp;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = frame_.swapchain->extent();
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
}

} // namespace ve::renderer
