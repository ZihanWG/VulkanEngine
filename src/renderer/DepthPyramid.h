#pragma once

// DepthPyramid owns the Hi-Z depth pyramid subsystem extracted from the
// monolithic Renderer: the R32_SFLOAT max-depth mip chain, its build compute
// pipeline, the per-mip build descriptor sets, the nearest-clamp sampler, and the
// image-layout tracking. Each frame, after the main depth pass, it downsamples
// the main depth buffer into the pyramid so the *next* frame's GPU culling pass
// can run conservative previous-frame Hi-Z occlusion tests.
//
// Ownership boundary (Design B, reference-borrowing): the rendering services
// (context / swapchain / render graph / GPU profiler) stay owned by Renderer and
// are borrowed here by reference. The pyramid image, sampler, and mip views are
// exposed through accessors so the GPU culling code (still in Renderer) can bind
// them into its compute descriptors, and so the debug UI can preview the mips.
//
// Disabling GPU occlusion when the device cannot build the pyramid is left to the
// caller: createResources() reports capability through buildAvailable(), and the
// Renderer flips its useGpuOcclusionCulling_ flag accordingly.

#include "renderer/RenderResolution.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve {

namespace rhi {
class VulkanContext;
class VulkanSwapchain;
} // namespace rhi

namespace renderer {

class RenderGraph;
class GpuProfiler;

class DepthPyramid final {
public:
    DepthPyramid(rhi::VulkanContext& context,
                 rhi::VulkanSwapchain& swapchain,
                 const RenderResolution& renderResolution,
                 RenderGraph& renderGraph,
                 GpuProfiler& gpuProfiler);
    ~DepthPyramid();

    DepthPyramid(const DepthPyramid&) = delete;
    DepthPyramid& operator=(const DepthPyramid&) = delete;

    // Created once at startup; survives swapchain recreation.
    void createDescriptorSetLayout();
    void createPipeline();

    // (Re)creates the pyramid image, mip views, sampler, and build descriptors for
    // the current swapchain extent. Tears down previous resources first, so it is
    // safe to call on swapchain recreation. Capability is reported afterwards via
    // buildAvailable() (false when the device cannot sample/store the format or the
    // main depth is not sampleable).
    void createResources();
    void destroyResources();

    // Marks the previous-frame contents stale (e.g. after a camera/scene change)
    // so the culling pass will not trust them for occlusion this frame.
    void invalidate();

    // midFrame: records into the DepthPyramidMidPass graph slot (the two-phase
    // occlusion rebuild between the phase-1 and phase-2 main passes) instead of
    // the end-of-frame DepthPyramidPass.
    void recordCommands(VkCommandBuffer commandBuffer,
                        uint32_t frameIndex,
                        const glm::mat4& frameViewProjection,
                        const glm::vec3& frameCameraPosition,
                        bool midFrame = false);
    void ensureShaderReadLayout(VkCommandBuffer commandBuffer);

    // --- accessors for the GPU culling code (binds image/sampler) and debug UI ---
    [[nodiscard]] VkImage image() const { return image_.image(); }
    [[nodiscard]] VkImageView imageView() const { return image_.imageView(); }
    [[nodiscard]] VkExtent3D extent() const { return image_.extent(); }
    [[nodiscard]] VkFormat format() const { return image_.format(); }
    [[nodiscard]] VkSampler sampler() const { return sampler_; }
    [[nodiscard]] VkImageLayout layout() const { return layout_; }
    // Mutable pointer to the tracked layout, handed to the render graph so its
    // barrier inference can keep the pyramid's layout in sync (matches the former
    // &depthPyramidLayout_ wiring).
    [[nodiscard]] VkImageLayout* layoutPtr() { return &layout_; }
    [[nodiscard]] uint32_t mipLevels() const { return mipLevels_; }
    [[nodiscard]] bool valid() const { return valid_; }
    [[nodiscard]] bool buildAvailable() const { return buildAvailable_; }
    [[nodiscard]] const glm::mat4& viewProjection() const { return viewProjection_; }
    [[nodiscard]] const glm::vec3& cameraPosition() const { return cameraPosition_; }
    [[nodiscard]] const std::vector<VkImageView>& mipImageViews() const { return mipImageViews_; }
    [[nodiscard]] uint32_t selectedDebugMip() const { return selectedDebugMip_; }
    void setSelectedDebugMip(uint32_t mip) { selectedDebugMip_ = mip; }

private:
    rhi::VulkanContext& context_;
    rhi::VulkanSwapchain& swapchain_;
    const RenderResolution& renderResolution_;
    RenderGraph& renderGraph_;
    GpuProfiler& gpuProfiler_;

    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::VulkanComputePipeline pipeline_;
    rhi::VulkanImage image_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<VkImageView> mipImageViews_;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t mipLevels_ = 0;
    uint32_t selectedDebugMip_ = 0;
    glm::mat4 viewProjection_{1.0f};
    glm::vec3 cameraPosition_{0.0f};
    bool valid_ = false;
    bool buildAvailable_ = false;
};

} // namespace renderer
} // namespace ve
