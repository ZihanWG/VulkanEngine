#pragma once

// The ray-traced shadow mask: the sun's visibility traced once per pixel the
// depth prepass covered, in compute passes between DepthPrepass and
// MainHDRPass, so the main pass reads one texel instead of tracing per fragment.
// See docs/ray_traced_shadows.md#the-screen-space-mask.
//
// Two passes. RayTracedShadowTrace writes the trace image, R16G16B16A16_SFLOAT
// at the render-resolution allocation extent: per trace texel the visibility,
// the hit distance and the camera distance of the surface it traced from. At
// half resolution it fills one quarter of that image, one texel per 2x2 block.
// RayTracedShadowResolve turns it into the mask, R32G32_UINT at the same extent:
// x is the visibility's float bits (kRtMaskNotTraced where nothing speaks for
// the pixel), y the depth bits the prepass wrote there.
//
// A soft shadow's filter sits between the two, at the trace's resolution, with
// a third image, the filter image, in the trace image's format and extent: the
// radius pass reads the trace image into it, the blur's horizontal axis reads it
// back into the trace image, and the vertical axis writes it again, which the
// resolve then reads instead of the trace image.
//
// Temporal accumulation runs between the trace and the filter: two history
// images, in the trace image's format and extent, alternate as last frame's and
// this frame's. RayTracedShadowTemporal reads the trace image and last frame's
// history and writes this frame's, which the radius pass reads in place of the
// single ray. The histories are the one thing here that outlives a frame by
// design: history lifetime, reset by rayTracedShadowHistoryUsable.
//
// Ownership. All three images are written and read within one frame, so none is
// frame-indexed or history. They persist across frames only so their layouts
// can be tracked: the render graph imports them with those layouts, which is
// what orders the previous frame's reads before this frame's writes. Created
// with the ray-tracing scene's resources, recreated with every other
// render-resolution target, destroyed before them.
//
// The mask image exists whenever the ray-query variant of the main shader draws
// -- the TLAS and the bindless path both present -- pass or no pass: that
// variant declares it at set 0 binding 17 and runs on every frame, so the
// binding has to hold an image in SHADER_READ_ONLY_OPTIMAL whether or not this
// frame traced it. Without the bindless path nothing declares the binding, and
// the image is not allocated.

#include "renderer/RayTracedShadows.h"
#include "renderer/RenderResolution.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"
#include "rhi/VulkanPipelineStore.h"

#include <filesystem>

namespace ve {

namespace rhi {
class VulkanContext;
class VulkanSwapchain;
} // namespace rhi

namespace renderer {

class RayTracedShadowMask final {
public:
    RayTracedShadowMask(rhi::VulkanContext& context,
                        rhi::VulkanSwapchain& swapchain,
                        rhi::VulkanPipelineStore& pipelineStore,
                        const RenderResolution& renderResolution);
    ~RayTracedShadowMask();

    RayTracedShadowMask(const RayTracedShadowMask&) = delete;
    RayTracedShadowMask& operator=(const RayTracedShadowMask&) = delete;

    // On every pipeline rebuild when the device can trace: both pipelines --
    // the trace's second set is the bindless heap its cutout test samples -- and
    // on the first call the passes' own set layouts. Throws on failure; the
    // caller leaves the mask unavailable.
    void createPipelines(const std::filesystem::path& traceShaderPath,
                         const std::filesystem::path& resolveShaderPath,
                         const std::filesystem::path& radiusShaderPath,
                         const std::filesystem::path& blurShaderPath,
                         const std::filesystem::path& temporalShaderPath,
                         VkDescriptorSetLayout bindlessSetLayout);
    void destroyPipelines();

    // The images, their sampler and the passes' descriptor sets. At startup and
    // on every resize, after the swapchain's depth image. The TLAS handle never
    // changes -- the structure is rebuilt in place -- so it is written here.
    void createResources(VkAccelerationStructureKHR topLevel);
    void destroyResources();

    // Dispatches over push.traceExtent. The caller has begun the graph pass,
    // which put depth in DEPTH_READ_ONLY_OPTIMAL and the trace image in GENERAL.
    void recordTrace(VkCommandBuffer commandBuffer,
                     const RayTracedShadowTracePushConstants& push,
                     VkDescriptorSet bindlessSet) const;
    // The temporal pass over push.traceExtent. The graph pass put depth in
    // DEPTH_READ_ONLY_OPTIMAL, the trace image and last frame's history in
    // SHADER_READ_ONLY_OPTIMAL, and this frame's history in GENERAL.
    void recordTemporal(VkCommandBuffer commandBuffer, const RayTracedShadowTemporalPushConstants& push) const;
    // One of the filter's dispatches over push.traceExtent. The graph pass put
    // the image it reads in SHADER_READ_ONLY_OPTIMAL and the one it writes in
    // GENERAL: see RayTracedShadowFilterStage for which is which.
    void recordFilter(VkCommandBuffer commandBuffer,
                      RayTracedShadowFilterStage stage,
                      const RayTracedShadowFilterPushConstants& push) const;
    // Dispatches over push.fullExtent, reading the filter image when
    // push.filtered is set and the trace image otherwise. The graph pass put
    // that image in SHADER_READ_ONLY_OPTIMAL and the mask in GENERAL.
    void recordResolve(VkCommandBuffer commandBuffer, const RayTracedShadowResolvePushConstants& push) const;

    // Copies the mask's `extent` sub-rect to the host and writes its visibility
    // to a single-channel PFM: linear, the back-facing mark stripped, NaN where
    // nothing speaks for the pixel. A debug capture, never the frame path: it
    // idles the queue, like the virtual shadow map's pool dump it is modelled on.
    // Logs and returns false on any failure.
    [[nodiscard]] bool
    dumpVisibility(const std::filesystem::path& path, VkExtent2D extent, VkCommandPool commandPool, VkQueue queue);

    // The mask image exists, so binding 17 can be written.
    [[nodiscard]] bool imageReady() const
    {
        return mask_.image() != VK_NULL_HANDLE;
    }
    // The passes can run: both images, both pipelines, and a main depth image
    // that can be sampled and holds 32-bit floats.
    [[nodiscard]] bool available() const
    {
        // The pipelines are checked at use, not only at creation: a reference
        // the store reset since it was issued reads null until it is requested
        // again.
        return available_ && tracePipeline_.pipeline() != VK_NULL_HANDLE &&
               resolvePipeline_.pipeline() != VK_NULL_HANDLE && radiusPipeline_.pipeline() != VK_NULL_HANDLE &&
               blurPipeline_.pipeline() != VK_NULL_HANDLE && temporalPipeline_.pipeline() != VK_NULL_HANDLE;
    }

    [[nodiscard]] const rhi::VulkanImage& image() const
    {
        return mask_;
    }
    [[nodiscard]] VkImageLayout* layoutPtr()
    {
        return &maskLayout_;
    }
    [[nodiscard]] const rhi::VulkanImage& traceImage() const
    {
        return trace_;
    }
    [[nodiscard]] VkImageLayout* traceLayoutPtr()
    {
        return &traceLayout_;
    }
    [[nodiscard]] const rhi::VulkanImage& filterImage() const
    {
        return filter_;
    }
    [[nodiscard]] VkImageLayout* filterLayoutPtr()
    {
        return &filterLayout_;
    }

    // The history this frame writes is historyImage(historyParity()); last
    // frame's is the other. advanceHistory swaps them, once per frame that
    // accumulates, before anything is recorded.
    [[nodiscard]] uint32_t historyParity() const
    {
        return historyParity_;
    }
    void advanceHistory()
    {
        historyParity_ ^= 1u;
    }
    // True once after the history images are created: they hold nothing yet.
    [[nodiscard]] bool consumeHistoryFresh()
    {
        const bool fresh = historyFresh_;
        historyFresh_ = false;
        return fresh;
    }
    [[nodiscard]] const rhi::VulkanImage& historyImage(uint32_t index) const
    {
        return history_.at(index);
    }
    [[nodiscard]] VkImageLayout* historyLayoutPtr(uint32_t index)
    {
        return &historyLayout_.at(index);
    }
    [[nodiscard]] VkSampler sampler() const
    {
        return sampler_;
    }

    static constexpr VkFormat kFormat = VK_FORMAT_R32G32_UINT;
    static constexpr VkFormat kTraceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

private:
    void createDescriptorSetLayouts();

    rhi::VulkanContext& context_;
    rhi::VulkanSwapchain& swapchain_;
    rhi::VulkanPipelineStore& pipelineStore_;
    const RenderResolution& renderResolution_;

    // Trace: depth, TLAS, trace image (storage). Resolve: depth, trace or
    // filter image (sampled), mask (storage). Filter: one image sampled, the
    // other storage, and for the radius pass this frame's history (sampled).
    // Temporal: depth, trace image and last frame's history (sampled), this
    // frame's history (storage).
    rhi::VulkanDescriptorSetLayout traceSetLayout_;
    rhi::VulkanDescriptorSetLayout resolveSetLayout_;
    rhi::VulkanDescriptorSetLayout filterSetLayout_;
    rhi::VulkanDescriptorSetLayout temporalSetLayout_;
    rhi::ComputePipelineRef tracePipeline_;
    rhi::ComputePipelineRef resolvePipeline_;
    rhi::ComputePipelineRef radiusPipeline_;
    rhi::ComputePipelineRef blurPipeline_;
    rhi::ComputePipelineRef temporalPipeline_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkDescriptorSet traceSet_ = VK_NULL_HANDLE;
    VkDescriptorSet resolveSet_ = VK_NULL_HANDLE;
    VkDescriptorSet resolveFilteredSet_ = VK_NULL_HANDLE;
    // The radius pass and the vertical axis read the trace image and write the
    // filter image; the horizontal axis the other way round.
    VkDescriptorSet traceToFilterSet_ = VK_NULL_HANDLE;
    VkDescriptorSet filterToTraceSet_ = VK_NULL_HANDLE;
    // Indexed by the parity of the history this frame writes.
    std::array<VkDescriptorSet, 2> radiusSets_{};
    std::array<VkDescriptorSet, 2> temporalSets_{};
    rhi::VulkanImage trace_;
    VkImageLayout traceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    rhi::VulkanImage filter_;
    VkImageLayout filterLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::array<rhi::VulkanImage, 2> history_;
    std::array<VkImageLayout, 2> historyLayout_{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t historyParity_ = 0;
    bool historyFresh_ = true;
    rhi::VulkanImage mask_;
    VkImageLayout maskLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    // Nearest, clamped. Everything here is read with texelFetch, and the mask is
    // an integer format, which no filtering sampler may touch.
    VkSampler sampler_ = VK_NULL_HANDLE;
    bool available_ = false;
};

} // namespace renderer
} // namespace ve
