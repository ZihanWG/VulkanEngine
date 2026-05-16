#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <vector>

namespace ve::rhi {

class VulkanShadowMap;
class VulkanSwapchain;

} // namespace ve::rhi

namespace ve::renderer {

enum class RenderPassType {
    Shadow,
    MainHdr,
    BloomExtract,
    BloomBlur,
    Luminance,
    HistogramExposure,
    Composite,
    ImGui
};

enum class RenderPassExecutionType {
    Graphics,
    Compute,
    Transfer
};

enum class RenderResourceAccess {
    Read,
    Write
};

struct RenderResourceHandle {
    const char* name = "";
};

struct RenderResourceUsage {
    RenderResourceHandle resource{};
    RenderResourceAccess access = RenderResourceAccess::Read;
    const char* description = "";
};

struct RenderPassNode {
    const char* name = "";
    RenderPassType type = RenderPassType::MainHdr;
    std::vector<RenderResourceUsage> resourceUsages;
    RenderPassExecutionType executionType = RenderPassExecutionType::Graphics;
    const char* transitionSummary = "";
};

struct RenderGraphImageResource {
    const char* name = "";
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkImageLayout* layout = nullptr;
};

struct RenderGraphFrameResources {
    RenderGraphImageResource sceneColor;
    RenderGraphImageResource bloomExtract;
    RenderGraphImageResource bloomPing;
    RenderGraphImageResource bloomPong;
};

// This is a deliberately minimal frame graph. It documents the current pass
// order and resource usage, then centralizes the explicit Synchronization2
// transitions those passes need. It does not infer dependencies, allocate
// transient resources, alias attachments, cull passes, schedule async compute,
// or act as a production graph scheduler.
class RenderGraph final {
public:
    RenderGraph();

    void beginFrame(VkCommandBuffer commandBuffer,
                    rhi::VulkanSwapchain& swapchain,
                    rhi::VulkanShadowMap& shadowMap,
                    uint32_t imageIndex,
                    RenderGraphFrameResources frameResources);
    void beginShadowPass(uint32_t cascadeLayer);
    void endShadowPass(bool finalCascade);
    void beginMainHdrPass();
    void endMainHdrPass();
    void beginBloomExtractPass();
    void endBloomExtractPass();
    void beginBloomBlurPass(bool horizontal);
    void endBloomBlurPass();
    void beginLuminancePass();
    void endLuminancePass();
    void beginHistogramExposurePass();
    void endHistogramExposurePass();
    void beginCompositePass();
    void endCompositePass();
    void beginImGuiPass();
    void endImGuiPass();
    void endFrame();

    [[nodiscard]] const std::vector<RenderPassNode>& passes() const
    {
        return passes_;
    }

    [[nodiscard]] const std::vector<RenderPassNode>& debugPasses() const
    {
        return passes_;
    }

private:
    enum class ActivePass {
        None,
        Shadow,
        MainHdr,
        BloomExtract,
        BloomBlur,
        Luminance,
        HistogramExposure,
        Composite,
        ImGui
    };

    struct FrameState {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        rhi::VulkanSwapchain* swapchain = nullptr;
        rhi::VulkanShadowMap* shadowMap = nullptr;
        RenderGraphFrameResources resources{};
        uint32_t imageIndex = 0;
        VkImage swapchainImage = VK_NULL_HANDLE;
    };

    void requireFrameActive(const char* operation) const;
    void transitionShadowMapImage(VkImageLayout oldLayout, VkImageLayout newLayout);
    void transitionSwapchainImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);
    void barrierSwapchainColorAttachment();
    void transitionColorImage(RenderGraphImageResource& resource, VkImageLayout newLayout);
    void transitionDepthImage();
    void beginColorRendering(const RenderGraphImageResource& resource, VkClearValue clearValue);
    void beginSwapchainRendering(VkClearValue clearValue, VkAttachmentLoadOp loadOp);

    FrameState frame_{};
    std::vector<RenderPassNode> passes_;
    bool frameActive_ = false;
    ActivePass activePass_ = ActivePass::None;
};

[[nodiscard]] const char* renderPassTypeName(RenderPassType type);
[[nodiscard]] const char* renderPassExecutionTypeName(RenderPassExecutionType executionType);
[[nodiscard]] const char* renderResourceAccessName(RenderResourceAccess access);

} // namespace ve::renderer
