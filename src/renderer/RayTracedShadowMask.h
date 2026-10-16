#pragma once

// The ray-traced shadow mask: the sun's visibility traced once per pixel the
// depth prepass covered, in a compute pass between DepthPrepass and
// MainHDRPass, so the main pass reads one texel instead of tracing per fragment.
// See docs/ray_traced_shadows.md.
//
// Ownership. One image, R32G32_UINT at the render-resolution allocation extent:
// x is the visibility's float bits (kRtMaskNotTraced where nothing was traced),
// y the depth bits the prepass wrote there. Written and read within one frame,
// so it is neither frame-indexed nor history. It persists across frames only so
// its layout can be tracked: the render graph imports it with that layout, which
// is what orders the previous frame's fragment reads before this frame's write.
// Created with the ray-tracing scene's resources, recreated with every other
// render-resolution target, destroyed before them.
//
// The image exists whenever the device can trace, pass or no pass: the main
// shader's ray-query variant declares the mask at set 0 binding 17 and runs on
// every frame such a device draws, so the binding has to hold an image in
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

    // On every pipeline rebuild when the device can trace: the pipeline, whose
    // second set is the bindless heap the cutout test samples, and on the first
    // call the pass's own set layout. Throws on failure; the caller leaves the
    // mask unavailable.
    void createPipeline(const std::filesystem::path& shaderPath, VkDescriptorSetLayout bindlessSetLayout);
    void destroyPipeline();

    // The image, its sampler and the pass's descriptor set. At startup and on
    // every resize, after the swapchain's depth image. The TLAS handle never
    // changes -- the structure is rebuilt in place -- so it is written here.
    void createResources(VkAccelerationStructureKHR topLevel);
    void destroyResources();

    // Dispatches over push.extent. The caller has begun the graph pass, which
    // put depth in DEPTH_READ_ONLY_OPTIMAL and the mask in GENERAL.
    void record(VkCommandBuffer commandBuffer,
                const RayTracedShadowMaskPushConstants& push,
                VkDescriptorSet bindlessSet) const;

    // The image exists, so binding 17 can be written.
    [[nodiscard]] bool imageReady() const
    {
        return image_.image() != VK_NULL_HANDLE;
    }
    // The pass can run: the image, a pipeline, and a main depth image that can
    // be sampled.
    [[nodiscard]] bool available() const
    {
        // The pipeline is checked at use, not only at creation: a reference the
        // store reset since it was issued reads null until it is requested again.
        return available_ && pipeline_.pipeline() != VK_NULL_HANDLE;
    }

    [[nodiscard]] const rhi::VulkanImage& image() const
    {
        return image_;
    }
    [[nodiscard]] VkImageLayout* layoutPtr()
    {
        return &layout_;
    }
    [[nodiscard]] VkSampler sampler() const
    {
        return sampler_;
    }

    static constexpr VkFormat kFormat = VK_FORMAT_R32G32_UINT;

private:
    void createDescriptorSetLayout();

    rhi::VulkanContext& context_;
    rhi::VulkanSwapchain& swapchain_;
    rhi::VulkanPipelineStore& pipelineStore_;
    const RenderResolution& renderResolution_;

    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::ComputePipelineRef pipeline_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    rhi::VulkanImage image_;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    // Nearest, clamped. Depth is read with texelFetch and the mask is an integer
    // format, which no filtering sampler may touch.
    VkSampler sampler_ = VK_NULL_HANDLE;
    bool available_ = false;
};

} // namespace renderer
} // namespace ve
