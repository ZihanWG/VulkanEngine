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

#include "renderer/RenderResolution.h"
#include "renderer/RuntimeSettings.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"
#include "rhi/VulkanTransientMemoryPool.h"
#include "rhi/VulkanPipeline.h"

#include <array>
#include <cstdint>
#include <unordered_map>
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
                     const RenderResolution& renderResolution,
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
    // debugRawGain non-zero makes the composite output scene colour scaled by
    // it and skip ambient occlusion, bloom, exposure and tone mapping. Passed in
    // rather than read from settings here because it is the caller's debug
    // state, not a post-process one.
    // sharpness is the raw RenderScaleSettings value; this gates it on the frame
    // actually being upscaled, which it can do and the caller cannot -- the
    // render resolution is borrowed here.
    // sharpenDebugGain non-zero replaces the output with the sharpen filter's own
    // correction, amplified -- the only way to see what a filter this fine-scale
    // is actually doing without guessing from the final image.
    void recordCompositeCommands(VkCommandBuffer commandBuffer,
                                 const glm::mat4& jitteredProjection,
                                 float debugRawGain = 0.0f,
                                 float sharpness = 0.0f,
                                 float sharpenDebugGain = 0.0f);
    void advanceTaaHistory();
    // The AO target holds the previous frame's occlusion in the previous
    // sub-rect. A scale change makes it unusable without moving any storage, so
    // this exists separately from recreating the resources.
    void invalidateAmbientOcclusionHistory() { ambientOcclusionHistoryValid_ = false; }

    // State queries.
    [[nodiscard]] bool isAutoExposureActive() const;
    [[nodiscard]] bool isLogAverageExposureActive() const;
    // Whether recordLuminanceCommands will record anything this frame. The graph
    // declares LuminancePass only when this is true, and the recorder early-outs
    // on the same call, so the declaration and the recording cannot disagree
    // about whether the pass exists.
    [[nodiscard]] bool willRecordLuminancePass() const;
    // Which bloom chain this frame uses. The composite shader samples both
    // bindings and selects one, so the chain that loses the selection is work
    // thrown away; the graph declares the loser's output as a layout-only read,
    // culling removes its passes, and the recorders below return early. The
    // composite's bloomMethod push constant reads the same call, so the shader
    // cannot select a chain that was never filled.
    // Whether recordHistogramCommands will record anything this frame. The graph
    // declares HistogramExposurePass only when this is true, and the recorder
    // early-outs on the same call: manual and log-average exposure both return
    // before the pass begins, and declaring it anyway makes the endFrame backstop
    // report a mismatch on a frame where the renderer is behaving correctly.
    [[nodiscard]] bool willRecordHistogramPass() const;
    [[nodiscard]] bool willRecordMipChainBloom() const;
    [[nodiscard]] bool isHistogramExposureActive() const;
    [[nodiscard]] bool isGpuExposureActive() const;
    [[nodiscard]] bool isTaaActive() const;
    [[nodiscard]] bool isTaaJitterActive() const;
    [[nodiscard]] uint32_t taaHistoryReadIndex() const;
    [[nodiscard]] uint32_t taaHistoryWriteIndex() const;
    [[nodiscard]] float currentToneMappingExposure() const;

    [[nodiscard]] bool autoExposureAvailable() const
    {
        return autoExposureAvailable_;
    }
    [[nodiscard]] bool histogramExposureAvailable() const
    {
        return histogramExposureAvailable_;
    }
    [[nodiscard]] bool exposureReduceAvailable() const
    {
        return exposureReduceAvailable_;
    }

    // Resource accessors (main HDR pass, render graph wiring, debug-UI previews).
    [[nodiscard]] rhi::VulkanImage& sceneColor()
    {
        return sceneColor_;
    }
    [[nodiscard]] const rhi::VulkanImage& sceneColor() const
    {
        return sceneColor_;
    }
    [[nodiscard]] VkImageLayout& sceneColorLayout()
    {
        return sceneColorLayout_;
    }
    [[nodiscard]] const rhi::VulkanImage& velocity() const
    {
        return velocity_;
    }
    [[nodiscard]] VkImageLayout& velocityLayout()
    {
        return velocityLayout_;
    }
    [[nodiscard]] const rhi::VulkanImage& normalRoughness() const
    {
        return normalRoughness_;
    }
    [[nodiscard]] VkImageLayout& normalRoughnessLayout()
    {
        return normalRoughnessLayout_;
    }
    // GTAO visibility target: owned here alongside the G-buffer targets, written
    // by the GTAO subsystem's pass and sampled by the composite.
    [[nodiscard]] const rhi::VulkanImage& ambientOcclusion() const
    {
        return ambientOcclusion_;
    }
    [[nodiscard]] VkImageLayout& ambientOcclusionLayout()
    {
        return ambientOcclusionLayout_;
    }
    [[nodiscard]] const rhi::VulkanImage& bloomExtract() const
    {
        return bloomExtract_;
    }
    [[nodiscard]] const rhi::VulkanImage& bloomPing() const
    {
        return bloomPing_;
    }
    [[nodiscard]] const rhi::VulkanImage& bloomPong() const
    {
        return bloomPong_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanImage>& bloomMipDownsampleImages() const
    {
        return bloomMipDownsampleImages_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanImage>& bloomMipUpsampleImages() const
    {
        return bloomMipUpsampleImages_;
    }
    [[nodiscard]] const std::array<rhi::VulkanImage, 2>& taaHistoryImages() const
    {
        return taaHistoryImages_;
    }
    [[nodiscard]] VkSampler sampler() const
    {
        return postProcessSampler_;
    }
    // Rebuilds the bloom chain into `pool` at these offsets on the next resource
    // creation. Pass an empty map to go back to private allocations.
    void setBloomAliasPlan(std::unordered_map<std::string, VkDeviceSize> offsets,
                           VkDeviceSize poolBytes,
                           uint32_t memoryTypeBits,
                           VkDeviceSize alignment);

    // True when the bloom images currently share pool memory. Drives the graph's
    // alias-handoff barrier, so it must reflect reality, not intent.
    [[nodiscard]] bool bloomImagesAreAliased() const { return bloomAliased_; }
    [[nodiscard]] VkDeviceSize bloomPoolBytes() const { return bloomPool_.size(); }

    [[nodiscard]] VkExtent2D bloomExtent() const
    {
        return bloomExtent_;
    }
    [[nodiscard]] uint32_t taaPostProcessHistoryIndex() const
    {
        return taaPostProcessHistoryIndex_;
    }
    [[nodiscard]] bool taaHistoryValid() const
    {
        return taaHistoryValid_;
    }
    // False until the GTAO pass has written the ambient-occlusion target at
    // least once since it was created. The main pass samples that target for
    // the *previous* frame's occlusion, and a freshly created image holds
    // undefined contents -- reading them would darken the first frame by
    // whatever garbage the allocation happened to contain.
    [[nodiscard]] bool ambientOcclusionHistoryValid() const
    {
        return ambientOcclusionHistoryValid_;
    }
    void markAmbientOcclusionWritten()
    {
        ambientOcclusionHistoryValid_ = true;
    }
    // TAA jitter debug readouts (shown in the debug UI).
    [[nodiscard]] uint32_t taaJitterIndex() const
    {
        return taaJitterIndex_;
    }
    [[nodiscard]] glm::vec2 taaCurrentJitterPixels() const
    {
        return taaCurrentJitterPixels_;
    }
    [[nodiscard]] glm::vec2 taaCurrentJitterNdc() const
    {
        return taaCurrentJitterNdc_;
    }

    // Mutable layout/handle access used by render-graph transition tracking.
    [[nodiscard]] VkImageLayout& bloomExtractLayout()
    {
        return bloomExtractLayout_;
    }
    [[nodiscard]] VkImageLayout& bloomPingLayout()
    {
        return bloomPingLayout_;
    }
    [[nodiscard]] VkImageLayout& bloomPongLayout()
    {
        return bloomPongLayout_;
    }
    [[nodiscard]] std::array<VkImageLayout, 2>& taaHistoryLayouts()
    {
        return taaHistoryLayouts_;
    }
    [[nodiscard]] std::vector<VkImageLayout>& bloomMipDownsampleLayouts()
    {
        return bloomMipDownsampleLayouts_;
    }
    [[nodiscard]] std::vector<VkImageLayout>& bloomMipUpsampleLayouts()
    {
        return bloomMipUpsampleLayouts_;
    }

    // Pipeline presence / format accessors used by recreateSwapchain's
    // pipeline-recreate decision.
    [[nodiscard]] const rhi::VulkanPipeline& bloomExtractPipeline() const
    {
        return bloomExtractPipeline_;
    }
    [[nodiscard]] const rhi::VulkanPipeline& bloomBlurPipeline() const
    {
        return bloomBlurPipeline_;
    }
    [[nodiscard]] const rhi::VulkanPipeline& bloomDownsamplePipeline() const
    {
        return bloomDownsamplePipeline_;
    }
    [[nodiscard]] const rhi::VulkanPipeline& bloomUpsamplePipeline() const
    {
        return bloomUpsamplePipeline_;
    }
    [[nodiscard]] const rhi::VulkanPipeline& taaResolvePipeline() const
    {
        return taaResolvePipeline_;
    }
    [[nodiscard]] const rhi::VulkanPipeline& compositePipeline() const
    {
        return compositePipeline_;
    }
    [[nodiscard]] const rhi::VulkanComputePipeline& luminancePipeline() const
    {
        return luminancePipeline_;
    }
    [[nodiscard]] const rhi::VulkanComputePipeline& histogramPipeline() const
    {
        return histogramPipeline_;
    }
    [[nodiscard]] const rhi::VulkanComputePipeline& exposureReducePipeline() const
    {
        return exposureReducePipeline_;
    }
    [[nodiscard]] VkFormat bloomExtractPipelineColorFormat() const
    {
        return bloomExtractPipelineColorFormat_;
    }
    [[nodiscard]] VkFormat bloomBlurPipelineColorFormat() const
    {
        return bloomBlurPipelineColorFormat_;
    }
    [[nodiscard]] VkFormat bloomDownsamplePipelineColorFormat() const
    {
        return bloomDownsamplePipelineColorFormat_;
    }
    [[nodiscard]] VkFormat bloomUpsamplePipelineColorFormat() const
    {
        return bloomUpsamplePipelineColorFormat_;
    }
    [[nodiscard]] VkFormat taaResolvePipelineColorFormat() const
    {
        return taaResolvePipelineColorFormat_;
    }
    [[nodiscard]] VkFormat compositePipelineColorFormat() const
    {
        return compositePipelineColorFormat_;
    }

    [[nodiscard]] VkDescriptorSetLayout luminanceDescriptorSetLayoutHandle() const
    {
        return postProcessLuminanceDescriptorSetLayout_.handle();
    }

    // Per-frame exposure buffers, imported by the render graph for barriers.
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& luminanceBuffers() const
    {
        return frameLuminanceBuffers_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& luminanceReadbackBuffers() const
    {
        return frameLuminanceReadbackBuffers_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& histogramBuffers() const
    {
        return frameHistogramBuffers_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& histogramReadbackBuffers() const
    {
        return frameHistogramReadbackBuffers_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& exposureBuffers() const
    {
        return frameExposureBuffers_;
    }

    // The frame clock's elapsed time, pushed in by Renderer once per frame.
    // Exposure adaptation used to read steady_clock here, which made it a second
    // independent time source and defeated any attempt at a reproducible frame.
    void setFrameTimeSeconds(float seconds) { frameTimeSeconds_ = seconds; }

    // Resets the auto-exposure adaptation timer (called when settings change to
    // avoid a large delta-time spike on the next exposure reduce).
    void resetAutoExposureTimer() { lastAutoExposureUpdateSeconds_ = frameTimeSeconds_; }

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
    const RenderResolution& renderResolution_;

    // ---- The sub-rect invariant --------------------------------------------
    //
    // Targets are allocated at RenderResolution::allocationExtent() and only
    // their top-left extent() sub-rect is written, so three quantities that used
    // to be the same number no longer are. Every bug this design can produce is
    // a confusion between them:
    //
    //   texel size      = 1 / ALLOCATED. A texel is physical; the sub-rect does
    //                     not change how big one is, only how many are written.
    //   viewport,
    //   dispatch bounds = USED. What the frame actually covers.
    //   uv scale        = USED / ALLOCATED, per target. Derived targets (halves,
    //                     mips) round each side independently, so their ratio
    //                     drifts from the scene's and each needs its own.
    //
    // Sampling past uvScale reads texels this frame never wrote -- whatever the
    // last larger frame left there. It is stale image data, so it produces a
    // plausible-looking edge rather than an obvious failure, and nothing in the
    // validation layers will say a word about it.
    // What the post-process chain is currently reading, and how much of it is
    // written. Today both paths -- SceneColorHDR and the resolved TAA history --
    // are allocated at the maximum and written in the render sub-rect, so this is
    // one answer either way.
    //
    // It exists as its own type because temporal upsampling changes that: the
    // resolve would write the history at the *full* allocation, and everything
    // downstream would then be reading a fully written source while a TAA-off
    // frame still reads a sub-rected one. Threading the size and scale through
    // here rather than deriving them from renderResolution_ at each of the five
    // consumers is what makes that a local change instead of a hunt.
    struct ActivePostProcessSource {
        VkExtent2D allocatedExtent{};
        VkExtent2D writtenExtent{};
        glm::vec2 uvScale{1.0f, 1.0f};
    };
    [[nodiscard]] ActivePostProcessSource activePostProcessSource() const;

    [[nodiscard]] VkExtent2D sceneUsedExtent() const { return renderResolution_.extent(); }
    [[nodiscard]] VkExtent2D sceneAllocatedExtent() const { return renderResolution_.allocationExtent(); }
    // bloomExtent_ is the ALLOCATED bloom size; this is the half of what the
    // frame writes. Both halve their own base, so their ratio drifts from the
    // scene's uv scale and each bloom pass carries its own.
    // What the bloom chain's written region is derived from: whatever the chain
    // is reading, which temporal upsampling makes full resolution.
    [[nodiscard]] VkExtent2D bloomChainSourceExtent() const { return activePostProcessSource().writtenExtent; }
    [[nodiscard]] VkExtent2D bloomWrittenExtent() const
    {
        return RenderResolution::halved(bloomChainSourceExtent());
    }
    [[nodiscard]] glm::vec2 bloomUvScale() const
    {
        return RenderResolution::subRectUvScale(bloomWrittenExtent(), bloomExtent_);
    }
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
    // TAA resolve: current color, history color, velocity, main depth.
    rhi::VulkanDescriptorSetLayout postProcessTaaResolveDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessCompositeDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessLuminanceDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessExposureReduceDescriptorSetLayout_;
    rhi::VulkanDescriptorPool postProcessDescriptorPool_;

    // Offsets for the bloom chain inside the shared transient pool, keyed by the
    // name the render graph uses. Empty means "give every bloom image its own
    // allocation", which is what this did before the pool existed and remains
    // the fallback whenever the pool is unavailable.
    //
    // Applied by createPostProcessResources, so the images and the descriptor
    // writes that reference their views are rebuilt together by the one path
    // that already knows how to do both.
    rhi::VulkanTransientMemoryPool bloomPool_;
    std::unordered_map<std::string, VkDeviceSize> bloomAliasOffsets_;
    // The allocation extent the offsets were computed for. A resize invalidates
    // them: offsets and pool size are both extent-dependent.
    VkExtent2D bloomAliasPlanExtent_{};
    bool bloomAliased_ = false;

    rhi::VulkanImage sceneColor_;
    rhi::VulkanImage velocity_;
    rhi::VulkanImage normalRoughness_;
    rhi::VulkanImage ambientOcclusion_;
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
    VkImageLayout velocityLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout normalRoughnessLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout ambientOcclusionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomExtractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPingLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPongLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::array<VkImageLayout, 2> taaHistoryLayouts_{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    std::vector<VkImageLayout> bloomMipDownsampleLayouts_;
    std::vector<VkImageLayout> bloomMipUpsampleLayouts_;

    // Capacity, sized from the ALLOCATION so a render-scale change never has to
    // resize the partial buffer -- which is the last thing that stood between a
    // scale change and not rebuilding anything at all.
    uint32_t luminancePartialCapacity_ = 0;

    // How much of that capacity this frame uses. Derived from the written extent
    // at record time rather than stored, because storing it is exactly the bake-in
    // that forced a rebuild.
    struct LuminanceDispatch {
        uint32_t groupCountX = 0;
        uint32_t groupCountY = 0;
        uint32_t partialCount = 0;
    };
    [[nodiscard]] LuminanceDispatch luminanceDispatch() const;
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
    bool ambientOcclusionHistoryValid_ = false;

    // Both in frame-clock seconds, never wall-clock seconds.
    float frameTimeSeconds_ = 0.0f;
    float lastAutoExposureUpdateSeconds_ = 0.0f;

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
