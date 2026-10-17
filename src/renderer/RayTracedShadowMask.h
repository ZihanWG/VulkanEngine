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
// the pixel), y the depth bits the prepass wrote there. A soft shadow's filter
// will sit between the two.
//
// Ownership. Both images are written and read within one frame, so neither is
// frame-indexed or history. They persist across frames only so their layouts
// can be tracked: the render graph imports them with those layouts, which is
// what orders the previous frame's reads before this frame's writes. Created
// with the ray-tracing scene's resources, recreated with every other
// render-resolution target, destroyed before them.
//
// The mask image exists whenever the device can trace, pass or no pass: the main
// shader's ray-query variant declares it at set 0 binding 17 and runs on every
// frame such a device draws, so the binding has to hold an image in
// SHADER_READ_ONLY_OPTIMAL whether or not this frame traced it.

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
    // Dispatches over push.fullExtent. The graph pass put the trace image in
    // SHADER_READ_ONLY_OPTIMAL and the mask in GENERAL.
    void recordResolve(VkCommandBuffer commandBuffer, const RayTracedShadowResolvePushConstants& push) const;

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
               resolvePipeline_.pipeline() != VK_NULL_HANDLE;
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

    // Trace: depth, TLAS, trace image (storage). Resolve: depth, trace image
    // (sampled), mask (storage).
    rhi::VulkanDescriptorSetLayout traceSetLayout_;
    rhi::VulkanDescriptorSetLayout resolveSetLayout_;
    rhi::ComputePipelineRef tracePipeline_;
    rhi::ComputePipelineRef resolvePipeline_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkDescriptorSet traceSet_ = VK_NULL_HANDLE;
    VkDescriptorSet resolveSet_ = VK_NULL_HANDLE;
    rhi::VulkanImage trace_;
    VkImageLayout traceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    rhi::VulkanImage mask_;
    VkImageLayout maskLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    // Nearest, clamped. Everything here is read with texelFetch, and the mask is
    // an integer format, which no filtering sampler may touch.
    VkSampler sampler_ = VK_NULL_HANDLE;
    bool available_ = false;
};

} // namespace renderer
} // namespace ve
