#pragma once

// PostProcessStack owns the HDR post-process subsystem extracted from the
// monolithic Renderer: the SceneColorHDR target, bloom (legacy + mip chain),
// TAA resolve/history/jitter, GPU auto-exposure (luminance/histogram/reduce),
// and the final composite (tone mapping + bloom + SSAO).
//
// Ownership boundary (Design B):
//   - Owned here: every post-process GPU resource (images, pipelines, compute
//     pipelines, descriptor pool/sets/layouts, sampler, per-frame buffers),
//     plus TAA history/jitter state and the exposure-readback plumbing.
//   - Borrowed by reference (still owned by Renderer): the rendering services
//     (context/render graph/GPU profiler/swapchain) and the *settings* structs
//     (tone mapping / bloom / TAA / SSAO) and exposure-output debug values.
//     These are referenced 90+ times across Renderer's serialization, portfolio
//     capture, scene I/O, and input code, so moving their ownership would force
//     a large invasive rewrite. The reference members keep the same names as the
//     former Renderer members so the relocated method bodies compile verbatim.
//   - Inputs passed in per (re)create / per frame: the main depth image lives on
//     the swapchain; createResources() also takes a fallback depth view, and the
//     composite/jitter calls take the per-frame jittered projection.
//
// SceneColorHDR is created here (it was created in createPostProcessResources),
// so the main HDR pass renders into PostProcessStack::sceneColor().

#include "renderer/RuntimeSettings.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"
#include "rhi/VulkanPipeline.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

namespace ve {

namespace rhi {
class VulkanContext;
class VulkanSwapchain;
} // namespace rhi

namespace renderer {

class RenderGraph;
class GpuProfiler;

class PostProcessStack final {
public:
    PostProcessStack(rhi::VulkanContext& context,
                     RenderGraph& renderGraph,
                     GpuProfiler& gpuProfiler,
                     rhi::VulkanSwapchain& swapchain,
                     ToneMappingSettings& toneMappingSettings,
                     BloomSettings& bloomSettings,
                     TaaSettings& taaSettings,
                     SsaoSettings& ssaoSettings,
                     float& currentExposure,
                     float& averageLuminance,
                     float& histogramClippedLuminance,
                     bool& ssaoAvailable);
    ~PostProcessStack();

    PostProcessStack(const PostProcessStack&) = delete;
    PostProcessStack& operator=(const PostProcessStack&) = delete;
    PostProcessStack(PostProcessStack&&) = delete;
    PostProcessStack& operator=(PostProcessStack&&) = delete;

    // Lifecycle (called from Renderer ctor / recreateSwapchain / createPipeline).
    // Names match the former Renderer members so relocated bodies compile verbatim.
    void createPostProcessSampler();
    void createPostProcessDescriptorSetLayouts();
    // Recreates SceneColorHDR + bloom + TAA + exposure resources and descriptor
    // sets. depthFallbackView is sampled by the composite when the real depth
    // image cannot be sampled (Renderer's checkerboard fallback).
    void createPostProcessResources(VkImageView depthFallbackView, uint32_t frameCount);
    void createExposureComputePipelines();
    void createBloomPipelines();
    void createTaaResolvePipeline();
    void createCompositePipeline();
    void invalidateTaaHistory();

    // Per-frame.
    void beginFrame(uint32_t frameIndex, bool taaActive);
    void updateAutoExposureFromReadback(uint32_t frameIndex);
    // Advances TAA jitter bookkeeping and returns the NDC jitter offset to add to
    // the main projection (zero when TAA jitter is inactive).
    [[nodiscard]] glm::vec2 advanceJitter(VkExtent2D extent);
    void recordTaaResolveCommands(VkCommandBuffer commandBuffer);
    void recordLegacyBloomCommands(VkCommandBuffer commandBuffer);
    void recordMipChainBloomCommands(VkCommandBuffer commandBuffer);
    void recordLuminanceCommands(VkCommandBuffer commandBuffer);
    void recordHistogramCommands(VkCommandBuffer commandBuffer);
    void recordCompositeCommands(VkCommandBuffer commandBuffer, const glm::mat4& jitteredProjection);
    void advanceTaaHistory();

    // State queries.
    [[nodiscard]] bool isAutoExposureActive() const;
    [[nodiscard]] bool isLogAverageExposureActive() const;
    [[nodiscard]] bool isHistogramExposureActive() const;
    [[nodiscard]] bool isGpuExposureActive() const;
    [[nodiscard]] bool isTaaActive() const;
    [[nodiscard]] bool isTaaJitterActive() const;
    [[nodiscard]] uint32_t taaHistoryReadIndex() const;
    [[nodiscard]] uint32_t taaHistoryWriteIndex() const;
    [[nodiscard]] float currentToneMappingExposure() const;

    [[nodiscard]] bool autoExposureAvailable() const { return autoExposureAvailable_; }
    [[nodiscard]] bool histogramExposureAvailable() const { return histogramExposureAvailable_; }
    [[nodiscard]] bool exposureReduceAvailable() const { return exposureReduceAvailable_; }

    // Resource accessors (main HDR pass, render graph wiring, debug-UI previews).
    [[nodiscard]] rhi::VulkanImage& sceneColor() { return sceneColor_; }
    [[nodiscard]] const rhi::VulkanImage& sceneColor() const { return sceneColor_; }
    [[nodiscard]] VkImageLayout& sceneColorLayout() { return sceneColorLayout_; }
    [[nodiscard]] const rhi::VulkanImage& bloomExtract() const { return bloomExtract_; }
    [[nodiscard]] const rhi::VulkanImage& bloomPing() const { return bloomPing_; }
    [[nodiscard]] const rhi::VulkanImage& bloomPong() const { return bloomPong_; }
    [[nodiscard]] const std::vector<rhi::VulkanImage>& bloomMipDownsampleImages() const
    {
        return bloomMipDownsampleImages_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanImage>& bloomMipUpsampleImages() const
    {
        return bloomMipUpsampleImages_;
    }
    [[nodiscard]] const std::array<rhi::VulkanImage, 2>& taaHistoryImages() const { return taaHistoryImages_; }
    [[nodiscard]] VkSampler sampler() const { return postProcessSampler_; }
    [[nodiscard]] VkExtent2D bloomExtent() const { return bloomExtent_; }
    [[nodiscard]] uint32_t taaPostProcessHistoryIndex() const { return taaPostProcessHistoryIndex_; }
    [[nodiscard]] bool taaHistoryValid() const { return taaHistoryValid_; }

    // Mutable layout/handle access used by render-graph transition tracking.
    [[nodiscard]] VkImageLayout& bloomExtractLayout() { return bloomExtractLayout_; }
    [[nodiscard]] VkImageLayout& bloomPingLayout() { return bloomPingLayout_; }
    [[nodiscard]] VkImageLayout& bloomPongLayout() { return bloomPongLayout_; }
    [[nodiscard]] std::array<VkImageLayout, 2>& taaHistoryLayouts() { return taaHistoryLayouts_; }
    [[nodiscard]] std::vector<VkImageLayout>& bloomMipDownsampleLayouts() { return bloomMipDownsampleLayouts_; }
    [[nodiscard]] std::vector<VkImageLayout>& bloomMipUpsampleLayouts() { return bloomMipUpsampleLayouts_; }

    // Pipeline presence / format accessors used by recreateSwapchain's
    // pipeline-recreate decision.
    [[nodiscard]] const rhi::VulkanPipeline& bloomExtractPipeline() const { return bloomExtractPipeline_; }
    [[nodiscard]] const rhi::VulkanPipeline& bloomBlurPipeline() const { return bloomBlurPipeline_; }
    [[nodiscard]] const rhi::VulkanPipeline& bloomDownsamplePipeline() const { return bloomDownsamplePipeline_; }
    [[nodiscard]] const rhi::VulkanPipeline& bloomUpsamplePipeline() const { return bloomUpsamplePipeline_; }
    [[nodiscard]] const rhi::VulkanPipeline& taaResolvePipeline() const { return taaResolvePipeline_; }
    [[nodiscard]] const rhi::VulkanPipeline& compositePipeline() const { return compositePipeline_; }
    [[nodiscard]] const rhi::VulkanComputePipeline& luminancePipeline() const { return luminancePipeline_; }
    [[nodiscard]] const rhi::VulkanComputePipeline& histogramPipeline() const { return histogramPipeline_; }
    [[nodiscard]] const rhi::VulkanComputePipeline& exposureReducePipeline() const { return exposureReducePipeline_; }
    [[nodiscard]] VkFormat bloomExtractPipelineColorFormat() const { return bloomExtractPipelineColorFormat_; }
    [[nodiscard]] VkFormat bloomBlurPipelineColorFormat() const { return bloomBlurPipelineColorFormat_; }
    [[nodiscard]] VkFormat bloomDownsamplePipelineColorFormat() const { return bloomDownsamplePipelineColorFormat_; }
    [[nodiscard]] VkFormat bloomUpsamplePipelineColorFormat() const { return bloomUpsamplePipelineColorFormat_; }
    [[nodiscard]] VkFormat taaResolvePipelineColorFormat() const { return taaResolvePipelineColorFormat_; }
    [[nodiscard]] VkFormat compositePipelineColorFormat() const { return compositePipelineColorFormat_; }

    [[nodiscard]] VkDescriptorSetLayout luminanceDescriptorSetLayoutHandle() const
    {
        return postProcessLuminanceDescriptorSetLayout_.handle();
    }

private:
    // Per-frame descriptor-set counts + which optional groups to create, computed
    // once and used to size the pool and gate the per-group descriptor writes.
    struct PostProcessDescriptorCounts {
        bool createLuminanceDescriptors = false;
        bool createHistogramDescriptors = false;
        bool createExposureReduceDescriptors = false;
        bool createCompositeDescriptors = false;
        uint32_t exposureDescriptorSetCount = 0;
        uint32_t compositeDescriptorSetCount = 0;
        uint32_t legacyBloomSetCount = 0;
        uint32_t bloomDownsampleSetCount = 0;
        uint32_t bloomUpsampleSetCount = 0;
        uint32_t taaResolveSetCount = 0;
        uint32_t taaBloomExtractSetCount = 0;
        uint32_t taaBloomDownsampleSetCount = 0;
        uint32_t taaCompositeDescriptorSetCount = 0;
        uint32_t taaLuminanceDescriptorSetCount = 0;
        uint32_t taaHistogramDescriptorSetCount = 0;
    };

    // Resource creation helpers (relocated verbatim from Renderer).
    void destroyPostProcessSampler();
    void createTaaResources();
    void destroyTaaResources();
    void createLuminanceResources();
    void destroyLuminanceResources();
    void createHistogramResources();
    void destroyHistogramResources();
    void createExposureResources();
    void destroyExposureResources();
    [[nodiscard]] PostProcessDescriptorCounts computePostProcessDescriptorCounts() const;
    void createPostProcessDescriptorPool(const PostProcessDescriptorCounts& counts);
    void createPostProcessDescriptorSets();
    void allocateLegacyBloomDescriptorSets();
    void createTaaResolveDescriptorSets();
    void createBloomMipDownsampleDescriptorSets();
    void createBloomMipUpsampleDescriptorSets();
    void createCompositeDescriptorSets(const PostProcessDescriptorCounts& counts);
    void createLuminanceDescriptorSets(const PostProcessDescriptorCounts& counts);
    void createHistogramDescriptorSets(const PostProcessDescriptorCounts& counts);
    void createExposureReduceDescriptorSets(const PostProcessDescriptorCounts& counts);
    [[nodiscard]] VkDescriptorImageInfo postProcessImageInfo(VkImageView imageView) const;
    [[nodiscard]] VkDescriptorImageInfo postProcessDepthInfo() const;
    void recordExposureReduceCommands(VkCommandBuffer commandBuffer);
    void disableAutoExposureFallback(std::string_view reason);
    void disableLogAverageExposureFallback(std::string_view reason);
    void disableHistogramExposureFallback(std::string_view reason);
    [[nodiscard]] VkDescriptorSet activeBloomExtractDescriptorSet() const;
    [[nodiscard]] VkDescriptorSet activeBloomMipDownsampleDescriptorSet(uint32_t level) const;
    [[nodiscard]] VkDescriptorSet activeCompositeDescriptorSet() const;
    [[nodiscard]] VkDescriptorSet activeLuminanceDescriptorSet() const;
    [[nodiscard]] VkDescriptorSet activeHistogramDescriptorSet() const;

    // ---- Borrowed services + settings (owned by Renderer) ------------------
    rhi::VulkanContext& context_;
    RenderGraph& renderGraph_;
    GpuProfiler& gpuProfiler_;
    rhi::VulkanSwapchain& swapchain_;
    ToneMappingSettings& toneMappingSettings_;
    BloomSettings& bloomSettings_;
    TaaSettings& taaSettings_;
    SsaoSettings& ssaoSettings_;
    float& currentExposure_;
    float& averageLuminance_;
    float& histogramClippedLuminance_;
    bool& ssaoAvailable_;

    // ---- Owned post-process resources --------------------------------------
    VkSampler postProcessSampler_ = VK_NULL_HANDLE;
    VkImageView depthFallbackView_ = VK_NULL_HANDLE;

    rhi::VulkanDescriptorSetLayout postProcessSingleImageDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessDualImageDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessCompositeDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessLuminanceDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessExposureReduceDescriptorSetLayout_;
    rhi::VulkanDescriptorPool postProcessDescriptorPool_;

    rhi::VulkanImage sceneColor_;
    rhi::VulkanImage bloomExtract_;
    rhi::VulkanImage bloomPing_;
    rhi::VulkanImage bloomPong_;
    std::array<rhi::VulkanImage, 2> taaHistoryImages_;
    std::vector<rhi::VulkanImage> bloomMipDownsampleImages_;
    std::vector<rhi::VulkanImage> bloomMipUpsampleImages_;

    rhi::VulkanPipeline bloomExtractPipeline_;
    rhi::VulkanPipeline bloomBlurPipeline_;
    rhi::VulkanPipeline bloomDownsamplePipeline_;
    rhi::VulkanPipeline bloomUpsamplePipeline_;
    rhi::VulkanPipeline taaResolvePipeline_;
    rhi::VulkanPipeline compositePipeline_;
    rhi::VulkanComputePipeline luminancePipeline_;
    rhi::VulkanComputePipeline histogramPipeline_;
    rhi::VulkanComputePipeline exposureReducePipeline_;

    VkDescriptorSet bloomExtractDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurHorizontalDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurVerticalDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet compositeDescriptorSet_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> taaResolveDescriptorSets_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDescriptorSet, 2> taaBloomExtractDescriptorSets_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDescriptorSet, 2> taaBloomMipDownsampleDescriptorSets_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> bloomMipDownsampleDescriptorSets_;
    std::vector<VkDescriptorSet> bloomMipUpsampleDescriptorSets_;
    std::vector<VkDescriptorSet> compositeDescriptorSets_;
    std::vector<VkDescriptorSet> luminanceDescriptorSets_;
    std::vector<VkDescriptorSet> histogramDescriptorSets_;
    std::array<std::vector<VkDescriptorSet>, 2> taaCompositeDescriptorSets_;
    std::array<std::vector<VkDescriptorSet>, 2> taaLuminanceDescriptorSets_;
    std::array<std::vector<VkDescriptorSet>, 2> taaHistogramDescriptorSets_;
    std::vector<VkDescriptorSet> exposureReduceDescriptorSets_;

    std::vector<rhi::VulkanBuffer> frameLuminanceBuffers_;
    std::vector<rhi::VulkanBuffer> frameLuminanceReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameHistogramBuffers_;
    std::vector<rhi::VulkanBuffer> frameHistogramReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameExposureBuffers_;
    std::vector<uint8_t> frameLuminanceReadbackReady_;
    std::vector<uint8_t> frameHistogramReadbackReady_;
    std::vector<uint8_t> frameExposureReadbackReady_;

    VkFormat bloomExtractPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomBlurPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomDownsamplePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomUpsamplePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat taaResolvePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat compositePipelineColorFormat_ = VK_FORMAT_UNDEFINED;

    VkExtent2D bloomExtent_{};
    VkImageLayout sceneColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomExtractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPingLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPongLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::array<VkImageLayout, 2> taaHistoryLayouts_{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    std::vector<VkImageLayout> bloomMipDownsampleLayouts_;
    std::vector<VkImageLayout> bloomMipUpsampleLayouts_;

    uint32_t luminanceGroupCountX_ = 0;
    uint32_t luminanceGroupCountY_ = 0;
    uint32_t luminancePartialCount_ = 0;
    uint32_t currentFrame_ = 0;
    // Frames-in-flight count, formerly read via Renderer::frames_.size().
    uint32_t frameCount_ = 0;

    bool autoExposureAvailable_ = false;
    bool histogramExposureAvailable_ = false;
    bool exposureReduceAvailable_ = false;
    bool autoExposureWarningLogged_ = false;
    bool logAverageExposureWarningLogged_ = false;
    bool histogramExposureWarningLogged_ = false;
    bool taaHistoryValid_ = false;

    std::chrono::steady_clock::time_point lastAutoExposureUpdate_ = std::chrono::steady_clock::now();

    glm::vec2 taaCurrentJitterPixels_{0.0f, 0.0f};
    glm::vec2 taaPreviousJitterPixels_{0.0f, 0.0f};
    glm::vec2 taaCurrentJitterNdc_{0.0f, 0.0f};
    glm::vec2 taaPreviousJitterNdc_{0.0f, 0.0f};
    uint32_t taaJitterIndex_ = 0;
    uint32_t taaHistoryWriteIndex_ = 0;
    uint32_t taaPostProcessHistoryIndex_ = 0;
};

} // namespace renderer
} // namespace ve
