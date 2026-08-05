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
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

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

// Mirrors the ProbeShadingParams block in simple_bindless.frag.
//
// A uniform buffer rather than push constants or ObjectFrameData. The main
// pass's push-constant block has 24 bytes left and this needs 32; ObjectFrameData
// would fit, but its layout is duplicated in six shaders and the array stride
// depends on it, so a missed one reads every object past the first at the wrong
// offset -- silently, and only for objects that are not the first.
struct ProbeShadingParams {
    // xyz = grid origin. w = intensity, where zero disables the lookup outright:
    // the same "one value carries the off state" shape fogMaxDistance and the
    // punctual shadow slot sentinel use, so the shader needs no separate flag.
    glm::vec4 gridOrigin{0.0f};
    // xyz = spacing between probes. w = how far off the surface to sample from.
    glm::vec4 gridSpacing{1.0f, 1.0f, 1.0f, 0.0f};
    // x != 0 outputs the probe irradiance on its own instead of shaded colour.
    // A subtle indirect term is easy to mistake for no term at all, which is the
    // same reason the punctual shadow path has a visibility-only view.
    glm::vec4 debug{0.0f};
};

static_assert(sizeof(ProbeShadingParams) == 48);

enum class ProbeAtlasTarget : int32_t {
    Irradiance = 0,
    Depth = 1
};

// Mirrors the push constant block in probe_convolve.comp.
//
// The batch's probe indices travel explicitly rather than as a base index the
// shader adds its workgroup id to: the round-robin cursor wraps at the end of a
// cycle, so a contiguous run is exactly what the batch is not on those frames.
struct ProbeConvolvePushConstants {
    int32_t probeCount = 0;
    // Fraction of the previous tile kept. Zero on a probe's first capture, or
    // the neutral seed would be blended in forever.
    float hysteresis = 0.0f;
    // Sub-texel offset this capture was rasterised with; the reconstruction has
    // to move by the same amount.
    glm::vec2 jitter{0.0f};
    // Four to a vector because a push-constant array of scalars pads each
    // element to 16 bytes.
    std::array<glm::ivec4, kMaxProbesPerFrame / 4> probeIndices{};
};

static_assert(sizeof(ProbeConvolvePushConstants) == 80);
static_assert(sizeof(ProbeConvolvePushConstants) <= 128);
static_assert(offsetof(ProbeConvolvePushConstants, probeIndices) == 16);

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
                const std::filesystem::path& borderShaderPath,
                const std::filesystem::path& convolveShaderPath);
    void reset();

    // Picks the probes to capture this frame and returns them. Advances the
    // round-robin cursor and the jitter sequence, so it is called exactly once
    // per frame that captures.
    [[nodiscard]] const std::vector<uint32_t>& beginCaptureBatch(uint32_t probesPerFrame);

    // Sub-texel offset the current batch is being captured with. The capture
    // pass applies it to each face projection and the convolution subtracts it
    // again; both must use this same value.
    [[nodiscard]] glm::vec2 captureJitter() const
    {
        return captureJitter_;
    }

    // How much of a probe's previous value to keep. Held at zero until every
    // probe has been captured at least once: a probe blending against its
    // neutral seed would stay permanently too dark, and with a round-robin
    // cursor that is most of the grid for the first full cycle.
    void setHysteresis(float hysteresis)
    {
        hysteresis_ = hysteresis;
    }
    [[nodiscard]] bool gridConverged() const
    {
        return capturedProbeCount_ >= kProbeCount;
    }

    // The batch chosen by the last beginCaptureBatch, which is what the capture
    // pass rasterises and the convolution then reads back out of the capture
    // atlas in the same slot order.
    [[nodiscard]] const std::vector<uint32_t>& captureBatch() const
    {
        return captureBatch_;
    }

    // Marks this frame as capturing nothing, without touching the cursor -- so a
    // frame with probes disabled resumes where it left off instead of skipping a
    // stripe of the grid.
    void clearCaptureBatch()
    {
        captureBatch_.clear();
    }

    // Probes captured since startup, for the debug panel. Coverage over time is
    // the thing worth watching with an amortised update: it says how long the
    // grid takes to reflect a change rather than how fast one frame was.
    [[nodiscard]] uint64_t capturedProbeCount() const
    {
        return capturedProbeCount_;
    }
    [[nodiscard]] uint32_t updateCursor() const
    {
        return updateCursor_;
    }

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

    // Turns this frame's capture into probe tiles, then wraps the octahedral
    // seam into their borders. The border copy reads core texels the step before
    // it wrote, so the two cannot be merged.
    //
    // useDebugPattern replaces the convolution with the synthetic direction fill
    // from phase one, which is what makes an addressing or border mistake
    // legible; it also covers the case where the capture pipeline is
    // unavailable and there is nothing real to convolve.
    void recordUpdate(VkCommandBuffer commandBuffer, bool useDebugPattern);

    // True when the convolution pipeline came up, so a real capture can be
    // turned into probe tiles. False falls back to the debug pattern.
    [[nodiscard]] bool convolveAvailable() const
    {
        return convolveAvailable_;
    }

    [[nodiscard]] const rhi::VulkanImage& captureAtlas() const
    {
        return captureAtlas_;
    }
    [[nodiscard]] const rhi::VulkanImage& captureDepth() const
    {
        return captureDepth_;
    }
    [[nodiscard]] VkImageLayout* captureAtlasLayoutPtr()
    {
        return &captureAtlasLayout_;
    }
    [[nodiscard]] VkImageLayout* captureDepthLayoutPtr()
    {
        return &captureDepthLayout_;
    }

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

    // Buffer the material descriptor sets bind at the shading-params binding.
    // One buffer rather than one per frame: it is refreshed with a 48-byte
    // vkCmdUpdateBuffer inside the frame's own command buffer, so there is no
    // in-flight write for a descriptor to have to avoid.
    [[nodiscard]] VkBuffer shadingParamsBuffer() const
    {
        return shadingParamsBuffer_.buffer();
    }

    // Records this frame's parameters and the barrier making them visible to the
    // fragment stage. Must be called outside a rendering scope.
    void updateShadingParams(VkCommandBuffer commandBuffer, const ProbeShadingParams& params);

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
                         const std::filesystem::path& borderShaderPath,
                         const std::filesystem::path& convolveShaderPath);
    void writeDescriptorSet();
    void dispatchAtlas(VkCommandBuffer commandBuffer,
                       const rhi::VulkanComputePipeline& pipeline,
                       ProbeAtlasTarget target,
                       bool debugPattern);
    void recordConvolve(VkCommandBuffer commandBuffer);
    void recordDebugFill(VkCommandBuffer commandBuffer, bool debugPattern);
    void recordBorder(VkCommandBuffer commandBuffer);

    rhi::VulkanContext* context_ = nullptr;
    bool available_ = false;
    bool atlasesInitialized_ = false;

    bool convolveAvailable_ = false;

    rhi::VulkanImage irradianceAtlas_;
    rhi::VulkanImage depthAtlas_;
    // Scratch, not storage: it holds only the probes being captured right now,
    // one row per capture slot and one column per cube face, and is overwritten
    // every frame that captures.
    rhi::VulkanImage captureAtlas_;
    rhi::VulkanImage captureDepth_;
    VkImageLayout captureAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout captureDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    // Round-robin state. The cursor is what bounds any one probe's staleness;
    // see probeUpdateBatch.
    uint32_t updateCursor_ = 0;
    uint32_t updateIndex_ = 0;
    glm::vec2 captureJitter_{0.0f};
    float hysteresis_ = 0.0f;
    std::vector<uint32_t> captureBatch_;
    uint64_t capturedProbeCount_ = 0;
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
    rhi::VulkanComputePipeline convolvePipeline_;

    rhi::VulkanBuffer shadingParamsBuffer_;

    ProbeGridBounds bounds_{};
};

} // namespace ve::renderer
