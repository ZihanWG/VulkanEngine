#pragma once

// Vulkan side of irradiance-probe global illumination: owns the two octahedral
// atlases the probe grid stores through, the compute pipelines that maintain
// them, and the descriptor resources those need.
//
// Ownership boundary mirrors VolumetricFogPass: resources and recording live
// here, while the GPU-free grid, octahedral and atlas-addressing math lives in
// IrradianceProbes.h so it can be unit tested without a device.
//
// Two atlases rather than one, because irradiance and visibility want different
// resolutions and different formats:
//
//   irradiance  RGBA16F  8x8 core per probe   what the surface is lit by
//   depth       RG16F   16x16 core per probe  mean distance and mean distance
//                                             squared, which is what keeps light
//                                             from leaking through walls
//
// Both are tiny (roughly 200 and 330 KB), so the resolution split costs nothing
// and buys sharper occlusion where it matters.

#include "renderer/IrradianceProbes.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"

#include <cstdint>
#include <filesystem>

namespace ve::rhi {
class VulkanContext;
}

namespace ve::renderer {

// Mirrors the push constant block in probe_debug_fill.comp and
// probe_border.comp. One shader serves both atlases: they differ only in core
// resolution and in which image the dispatch targets, and the branch is uniform
// across the dispatch.
struct ProbeAtlasPushConstants {
    int32_t coreResolution = 0;
    // 0 = irradiance atlas, 1 = depth atlas.
    int32_t target = 0;
    // Non-zero writes the direction pattern; zero writes the neutral state an
    // atlas that has captured nothing should hold. Ignored by the border pass,
    // which copies whatever is there.
    int32_t debugPattern = 1;
};

static_assert(sizeof(ProbeAtlasPushConstants) == 12);

enum class ProbeAtlasTarget : int32_t {
    Irradiance = 0,
    Depth = 1
};

class IrradianceProbeVolume final {
public:
    IrradianceProbeVolume() = default;
    // The sampler is a raw VkSampler rather than an RAII wrapper, so unlike the
    // images and pipelines it needs an explicit destructor. Renderer declares
    // its VulkanContext before this member, so the device outlives it.
    ~IrradianceProbeVolume();

    IrradianceProbeVolume(const IrradianceProbeVolume&) = delete;
    IrradianceProbeVolume& operator=(const IrradianceProbeVolume&) = delete;
    IrradianceProbeVolume(IrradianceProbeVolume&&) = delete;
    IrradianceProbeVolume& operator=(IrradianceProbeVolume&&) = delete;

    void create(rhi::VulkanContext& context,
                const std::filesystem::path& debugFillShaderPath,
                const std::filesystem::path& borderShaderPath);
    void reset();

    // Whether the update pass has to be declared and recorded this frame.
    //
    // True while the atlases have never been written, whatever the toggle says.
    // That is the cold-start guard: the main pass declares its read of these
    // images unconditionally, so an atlas that was never written would be
    // sampled straight out of UNDEFINED. One dispatch pair over ~110k texels
    // costs nothing next to carrying that hazard.
    [[nodiscard]] bool needsUpdate(bool enabled) const
    {
        return available_ && (!atlasesInitialized_ || enabled);
    }

    // Requests exactly one more update, whatever the enable toggle says. For
    // changes to what the atlases should *contain*: with updates off the
    // contents are otherwise whatever the last update left, so a control that
    // changed them would appear to do nothing.
    void markDirty()
    {
        atlasesInitialized_ = false;
    }

    // Fills every probe tile and then wraps the octahedral seam into the border.
    // Two dispatches per atlas, in that order: the border copy reads core texels
    // the fill wrote, so they cannot be merged.
    //
    // debugPattern selects the direction pattern over the neutral empty-atlas
    // state. Until the capture phase lands those are the only two contents an
    // atlas can have.
    void recordUpdate(VkCommandBuffer commandBuffer, bool debugPattern);

    // The compute half works and the update pass can run.
    [[nodiscard]] bool available() const
    {
        return available_;
    }

    // The atlases exist and can be bound. True even when the compute half
    // failed, because the render graph and (from the shading phase on) the
    // material descriptors reference these images whether or not probes update.
    [[nodiscard]] bool hasAtlases() const
    {
        return irradianceAtlas_.imageView() != VK_NULL_HANDLE && depthAtlas_.imageView() != VK_NULL_HANDLE;
    }

    [[nodiscard]] const rhi::VulkanImage& irradianceAtlas() const
    {
        return irradianceAtlas_;
    }
    [[nodiscard]] const rhi::VulkanImage& depthAtlas() const
    {
        return depthAtlas_;
    }
    [[nodiscard]] VkImageLayout* irradianceAtlasLayoutPtr()
    {
        return &irradianceAtlasLayout_;
    }
    [[nodiscard]] VkImageLayout* depthAtlasLayoutPtr()
    {
        return &depthAtlasLayout_;
    }
    [[nodiscard]] VkSampler sampler() const
    {
        return sampler_;
    }

    // Where the grid sits in the world. Held here rather than in the math header
    // so it can be fitted to a scene at runtime; the addressing code takes it as
    // a parameter and never reads it back.
    [[nodiscard]] const ProbeGridBounds& bounds() const
    {
        return bounds_;
    }
    void setBounds(const ProbeGridBounds& bounds)
    {
        bounds_ = bounds;
    }

private:
    void createAtlases();
    void createSampler();
    void createDescriptorResources();
    void createPipelines(const std::filesystem::path& debugFillShaderPath,
                         const std::filesystem::path& borderShaderPath);
    void writeDescriptorSet();
    void dispatchAtlas(VkCommandBuffer commandBuffer,
                       const rhi::VulkanComputePipeline& pipeline,
                       ProbeAtlasTarget target,
                       bool debugPattern);

    rhi::VulkanContext* context_ = nullptr;
    bool available_ = false;
    bool atlasesInitialized_ = false;

    rhi::VulkanImage irradianceAtlas_;
    rhi::VulkanImage depthAtlas_;
    // Owned by this class but driven by the render graph, which transitions both
    // images and writes the layout back through these pointers.
    VkImageLayout irradianceAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout depthAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler sampler_ = VK_NULL_HANDLE;

    // One set, not one per frame: the atlases do not ping-pong and nothing in
    // the set changes between frames.
    rhi::VulkanDescriptorSetLayout setLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    rhi::VulkanComputePipeline debugFillPipeline_;
    rhi::VulkanComputePipeline borderPipeline_;

    ProbeGridBounds bounds_{};
};

} // namespace ve::renderer
