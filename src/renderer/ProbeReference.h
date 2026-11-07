#pragma once

// The ray-traced reference for probe GI (gi.debugReferenceRays): per visible
// pixel, the indirect diffuse light the probes are meant to supply, traced from
// the pixel's own surface and accumulated over frames, beside what the probes
// actually supply there. A measuring instrument, off unless asked for; see
// src/shaders/probe_reference.comp and docs/irradiance_probes.md.
//
// Two images at the render-resolution allocation extent, R32G32B32A32_SFLOAT:
//   accumulation  rgb = running sum of traced radiance, a = rays it holds;
//   estimate      rgb = the probes' irradiance at the pixel, a = coverage.
// Both persist across frames so the accumulation can, and are imported into
// the render graph with their tracked layouts. Recreated with every other
// render-resolution target; the CPU restarts the accumulation whenever what it
// measures changed.

#include "renderer/IrradianceProbes.h"
#include "renderer/RenderResolution.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"
#include "rhi/VulkanPipelineStore.h"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>

namespace ve {

namespace rhi {
class VulkanContext;
class VulkanSwapchain;
} // namespace rhi

namespace renderer {

// Mirrors the push block in probe_reference.comp. Device addresses as uint64_t.
struct ProbeReferencePushConstants {
    glm::mat4 inverseViewProjection{1.0f};
    uint64_t frameConstantsAddress = 0;
    uint64_t objectFrameDataAddress = 0;
    uint64_t geometryTableAddress = 0;
    uint64_t lightBufferAddress = 0;
    uint64_t punctualShadowSlotAddress = 0;
    uint32_t lightCount = 0;
    float sunRayLength = 0.0f;
    uint32_t raysPerFrame = 0;
    // Bits 0-30 the accumulation's frame index, bit 31 set to start it over.
    uint32_t frameAndReset = 0;
    glm::uvec2 extent{0u};
};
static_assert(sizeof(ProbeReferencePushConstants) == 128);

class ProbeReference final {
public:
    static constexpr VkFormat kFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    static constexpr uint32_t kLocalSize = 8;

    ProbeReference(rhi::VulkanContext& context,
                   rhi::VulkanSwapchain& swapchain,
                   rhi::VulkanPipelineStore& pipelineStore,
                   const RenderResolution& renderResolution);
    ~ProbeReference();

    ProbeReference(const ProbeReference&) = delete;
    ProbeReference& operator=(const ProbeReference&) = delete;

    // The pipeline: set 0 the shared material set, set 1 the bindless heap, set
    // 2 this pass's own. On every pipeline rebuild when the device can trace.
    // Throws on failure; the caller leaves the reference unavailable.
    void createPipeline(const std::filesystem::path& shaderPath,
                        VkDescriptorSetLayout materialSetLayout,
                        VkDescriptorSetLayout bindlessSetLayout);
    void destroyPipeline();

    // The images and this pass's set, which samples the main depth and the thin
    // G-buffer's normals. At startup and on every resize, after both exist.
    void createResources(VkImageView normalRoughnessView);
    void destroyResources();

    [[nodiscard]] bool available() const
    {
        return pipeline_.pipeline() != VK_NULL_HANDLE && descriptorSet_ != VK_NULL_HANDLE;
    }

    // Dispatches over push.extent. The graph pass has put depth and normals in
    // a sampled layout and both images in GENERAL.
    void record(VkCommandBuffer commandBuffer,
                const ProbeReferencePushConstants& push,
                VkDescriptorSet materialSet,
                VkDescriptorSet bindlessSet) const;

    // Both images' `extent` sub-rect copied back and compared. A debug readback
    // that idles the queue, like the shadow-mask dump. Empty on failure.
    [[nodiscard]] std::optional<ProbeReferenceStats>
    readStats(VkExtent2D extent, VkCommandPool commandPool, VkQueue queue);

    [[nodiscard]] const rhi::VulkanImage& accumulation() const
    {
        return accumulation_;
    }
    [[nodiscard]] const rhi::VulkanImage& estimate() const
    {
        return estimate_;
    }
    [[nodiscard]] VkImageLayout* accumulationLayoutPtr()
    {
        return &accumulationLayout_;
    }
    [[nodiscard]] VkImageLayout* estimateLayoutPtr()
    {
        return &estimateLayout_;
    }

private:
    rhi::VulkanContext& context_;
    rhi::VulkanSwapchain& swapchain_;
    rhi::VulkanPipelineStore& pipelineStore_;
    const RenderResolution& renderResolution_;

    rhi::ComputePipelineRef pipeline_;
    rhi::VulkanDescriptorSetLayout setLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    rhi::VulkanImage accumulation_;
    rhi::VulkanImage estimate_;
    VkImageLayout accumulationLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout estimateLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace renderer
} // namespace ve
