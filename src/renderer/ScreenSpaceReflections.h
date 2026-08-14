#pragma once

// ScreenSpaceReflections owns the SSR subsystem: the scene-color copy image the
// trace samples (copying avoids a same-image read/write feedback loop), the
// per-frame trace-parameter buffers, the descriptor resources, and the
// additive-blend fullscreen trace pipeline. The thin G-buffer it reads
// (normal/roughness/metallic) is written by the main pass and owned by
// PostProcessStack next to the velocity target.
//
// Ownership boundary (Design B, reference-borrowing): the rendering services
// (context / swapchain / render graph / GPU profiler) and the SsrSettings
// struct stay owned by Renderer and are borrowed by reference. Per-frame view
// and projection matrices arrive through uploadParams().
//
// Availability mirrors SSAO: the trace marches the main depth buffer, so a
// non-samplable depth image disables the subsystem (available() == false) and
// the renderer skips the passes.

#include "renderer/RenderResolution.h"
#include "renderer/RuntimeSettings.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"
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

class ScreenSpaceReflections final {
public:
    ScreenSpaceReflections(rhi::VulkanContext& context,
                           rhi::VulkanSwapchain& swapchain,
                           const RenderResolution& renderResolution,
                           RenderGraph& renderGraph,
                           GpuProfiler& gpuProfiler,
                           SsrSettings& settings);
    ~ScreenSpaceReflections();

    ScreenSpaceReflections(const ScreenSpaceReflections&) = delete;
    ScreenSpaceReflections& operator=(const ScreenSpaceReflections&) = delete;

    // Created once at startup; survives swapchain recreation.
    void createDescriptorSetLayout();
    void createPipeline(const std::filesystem::path& vertexShaderPath,
                        const std::filesystem::path& fragmentShaderPath);

    // (Re)creates the scene-color copy image, params buffers, and descriptor
    // sets for the current swapchain extent. normalRoughnessView is the thin
    // G-buffer written by the main pass (owned by PostProcessStack).
    void createResources(VkImageView normalRoughnessView, uint32_t frameCount);
    // Bindings 4/5 (prefiltered environment + BRDF LUT) are written here rather
    // than in createResources, because the IBL resources are built after it at
    // startup. Safe to call repeatedly; a no-op until the handles are valid.
    void updateIblDescriptors(VkImageView prefilteredEnvView,
                              VkSampler prefilteredEnvSampler,
                              VkImageView brdfLutView,
                              VkSampler brdfLutSampler);
    // False until updateIblDescriptors has landed. While false the trace cannot
    // know what specular the main pass already wrote, so it stays purely additive.
    [[nodiscard]] bool isIblBound() const { return iblBound_; }
    void destroyResources();

    // Host-visible write of this frame's matrices + march/weight parameters.
    void uploadParams(uint32_t frameIndex,
                      const glm::mat4& view,
                      const glm::mat4& projection,
                      uint32_t frameCounter);

    // Records the copy pass (scene color -> copy) and the trace pass (additive
    // blend into scene color). sceneColorImage is the main HDR target.
    void recordCommands(VkCommandBuffer commandBuffer,
                        uint32_t frameIndex,
                        VkImage sceneColorImage,
                        VkExtent2D extent);

    [[nodiscard]] bool available() const { return available_; }

    // Render-graph wiring for the transient scene-color copy resource.
    [[nodiscard]] const rhi::VulkanImage& sceneColorCopy() const { return sceneColorCopy_; }
    [[nodiscard]] VkImageLayout* sceneColorCopyLayoutPtr() { return &sceneColorCopyLayout_; }

private:
    rhi::VulkanContext& context_;
    rhi::VulkanSwapchain& swapchain_;
    const RenderResolution& renderResolution_;
    RenderGraph& renderGraph_;
    GpuProfiler& gpuProfiler_;
    SsrSettings& settings_;

    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanImage sceneColorCopy_;
    VkImageLayout sceneColorCopyLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler sampler_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<rhi::VulkanBuffer> frameParamsBuffers_;
    bool available_ = false;
    bool iblBound_ = false;
};

} // namespace renderer
} // namespace ve
