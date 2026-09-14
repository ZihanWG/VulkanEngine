#pragma once

// Counts fragment shader invocations over the main opaque geometry, so the
// engine can report how many times an average covered pixel is shaded.
//
// It exists to answer one question the repository has been unable to answer:
// whether a depth prepass is worth building. docs/design_decisions.md records
// that the pass was never evaluated because no scene here had realistic depth
// complexity, and that the prerequisite was a scene rather than a pass. With
// `--scene sponza` that prerequisite is met and what remains is the number.
//
// **What is counted, and why it is the right number.** A depth prepass buys
// exactly the shading of fragments that are later overdrawn. Fragment shader
// invocations are what survives early depth testing, so this measures the
// shading that actually happens rather than the geometric layer count -- which
// is precisely the work a prepass would remove. A scene whose invocations per
// covered pixel sit near 1 has nothing for a prepass to save, however many
// surfaces overlap in world space, because the hardware is already rejecting
// them.
//
// Diagnostic only. `VkPhysicalDeviceFeatures::pipelineStatisticsQuery` is
// optional in Vulkan, the query is recorded only when `--overdraw` asks for it,
// and nothing in the frame path reads the result. A device without the feature
// loses one log line and nothing else.

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ve::rhi {

class VulkanContext;

} // namespace ve::rhi

namespace ve::renderer {

class OverdrawQuery final {
public:
    struct FrameResult {
        bool valid = false;
        // Fragment shader invocations recorded between begin() and end().
        uint64_t fragmentInvocations = 0;
        // Pixels in the render extent the scope was recorded at.
        uint64_t renderedPixels = 0;

        // Invocations per rendered pixel. This is an upper bound on what a depth
        // prepass could remove, and a loose one: the denominator is the whole
        // render extent, so any part of the frame the opaque geometry does not
        // cover (sky through an arch, the letterbox around a small scene) pulls
        // it down. Read 2.0 as "the average covered pixel is shaded about twice
        // or more", never as an exact layer count.
        [[nodiscard]] double invocationsPerPixel() const
        {
            return renderedPixels == 0 ? 0.0
                                       : static_cast<double>(fragmentInvocations) / static_cast<double>(renderedPixels);
        }
    };

    OverdrawQuery() = default;
    ~OverdrawQuery();

    OverdrawQuery(const OverdrawQuery&) = delete;
    OverdrawQuery& operator=(const OverdrawQuery&) = delete;
    OverdrawQuery(OverdrawQuery&&) = delete;
    OverdrawQuery& operator=(OverdrawQuery&&) = delete;

    // No-op when the device lacks pipelineStatisticsQuery; available() then
    // reports false and unavailableReason() says why.
    void initialize(const rhi::VulkanContext& context, uint32_t framesInFlight);
    void shutdown();

    // Resets this frame's query. Must be recorded OUTSIDE a render pass:
    // vkCmdResetQueryPool is not allowed inside one.
    void resetFrame(uint32_t frameIndex, VkCommandBuffer commandBuffer);

    // Brackets the draws to count. `renderedPixels` is stored with the result so
    // the ratio cannot be computed against a different frame's resolution --
    // render scale and dynamic resolution both move it.
    void begin(uint32_t frameIndex, VkCommandBuffer commandBuffer, uint64_t renderedPixels);
    void end(uint32_t frameIndex, VkCommandBuffer commandBuffer);

    void markFrameSubmitted(uint32_t frameIndex);

    // Reads back a submitted frame without blocking. Returns false while the
    // result is not ready, which is the normal case for the frames still in
    // flight.
    [[nodiscard]] bool readFrame(uint32_t frameIndex, FrameResult& result);

    [[nodiscard]] bool available() const
    {
        return available_;
    }
    [[nodiscard]] const std::string& unavailableReason() const
    {
        return unavailableReason_;
    }

private:
    struct FrameState {
        VkQueryPool queryPool = VK_NULL_HANDLE;
        uint64_t renderedPixels = 0;
        bool recorded = false;
        bool submitted = false;
        bool active = false;
    };

    [[nodiscard]] bool validFrameIndex(uint32_t frameIndex) const;
    void disable(std::string reason);

    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<FrameState> frames_;
    bool available_ = false;
    std::string unavailableReason_;
};

} // namespace ve::renderer
