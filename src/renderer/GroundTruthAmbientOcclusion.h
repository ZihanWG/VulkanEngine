#pragma once

// GroundTruthAmbientOcclusion owns the GTAO subsystem: the horizon-search
// fullscreen pipeline, its per-frame parameter buffers, and the descriptor
// resources that read the main depth buffer and the thin G-buffer normal. The
// visibility target it writes into is owned by PostProcessStack (next to the
// velocity / normal-roughness G-buffer targets), so the composite pass can bind
// it without a cross-subsystem handoff; this class borrows that image's view
// through the render graph pass just like SSR borrows the scene-color target.
//
// Ownership boundary (Design B, reference-borrowing): the rendering services
// (context / swapchain / render graph / GPU profiler) and the SsaoSettings
// struct stay owned by Renderer and are borrowed by reference. Per-frame view
// and projection matrices arrive through uploadParams().
//
// Availability mirrors SSR: the horizon search reconstructs view positions from
// the main depth buffer, so a non-samplable depth image disables the subsystem
// (available() == false) and the renderer skips the pass.

#include "renderer/RuntimeSettings.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanPipeline.h"

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/mat4x4.hpp>

namespace ve {

namespace rhi {
class VulkanContext;
class VulkanSwapchain;
} // namespace rhi

namespace renderer {

class RenderGraph;
class GpuProfiler;

class GroundTruthAmbientOcclusion final {
public:
    GroundTruthAmbientOcclusion(rhi::VulkanContext& context,
                                rhi::VulkanSwapchain& swapchain,
                                RenderGraph& renderGraph,
                                GpuProfiler& gpuProfiler,
                                SsaoSettings& settings);
    ~GroundTruthAmbientOcclusion();

    GroundTruthAmbientOcclusion(const GroundTruthAmbientOcclusion&) = delete;
    GroundTruthAmbientOcclusion& operator=(const GroundTruthAmbientOcclusion&) = delete;

    // Created once at startup; survives swapchain recreation.
    void createDescriptorSetLayout();
    void createPipeline(const std::filesystem::path& vertexShaderPath,
                        const std::filesystem::path& fragmentShaderPath);

    // (Re)creates the params buffers and descriptor sets for the current
    // swapchain extent. normalRoughnessView is the thin G-buffer written by the
    // main pass (owned by PostProcessStack).
    void createResources(VkImageView normalRoughnessView, uint32_t frameCount);
    void destroyResources();

    // Host-visible write of this frame's matrices + horizon-search parameters.
    void uploadParams(uint32_t frameIndex,
                      const glm::mat4& view,
                      const glm::mat4& projection,
                      uint32_t frameCounter);

    // Records the GTAO pass (writes the visibility target owned by
    // PostProcessStack, set up by the render graph).
    void recordCommands(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent);

    [[nodiscard]] bool available() const { return available_; }

private:
    rhi::VulkanContext& context_;
    rhi::VulkanSwapchain& swapchain_;
    RenderGraph& renderGraph_;
    GpuProfiler& gpuProfiler_;
    SsaoSettings& settings_;

    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    rhi::VulkanPipeline pipeline_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<rhi::VulkanBuffer> frameParamsBuffers_;
    bool available_ = false;
};

} // namespace renderer
} // namespace ve
