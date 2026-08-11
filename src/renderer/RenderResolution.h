#pragma once

#include "renderer/RenderScale.h"
#include "rhi/VulkanCommon.h"

namespace ve::renderer {

// The two resolutions a frame has, in one borrowable place.
//
// `extent()` is what the scene is shaded at and what every screen-space target
// before the composite is sized to: main depth, SceneColorHDR, the thin
// G-buffer, velocity, GTAO, SSR, the TAA history, the bloom chain and the Hi-Z
// pyramid. `outputExtent()` is the swapchain, and only three things use it --
// the composite's own viewport, the ImGui overlay, and portfolio screenshots.
//
// Renderer owns one of these and every resolution-dependent subsystem borrows it
// by const reference, the same way they already borrow the swapchain. That is
// deliberate: the alternative (a setter per subsystem) has four places to forget
// on a resize, and a subsystem that missed the update allocates a target at the
// wrong size, which shows up as a validation error deep in a later pass rather
// than at the mistake.
class RenderResolution final {
public:
    // Called once per swapchain (re)creation, after the swapchain knows its new
    // extent and before anything sized off it is allocated.
    void update(VkExtent2D outputExtent, float scale)
    {
        outputExtent_ = outputExtent;
        scale_ = clampRenderScale(scale);
        const RenderDimensions dimensions =
            scaledRenderDimensions(outputExtent.width, outputExtent.height, scale_);
        extent_ = VkExtent2D{dimensions.width, dimensions.height};
    }

    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkExtent2D outputExtent() const { return outputExtent_; }
    [[nodiscard]] float scale() const { return scale_; }

    // True when no upscale happens, so callers can skip work that only a scaled
    // frame needs. Compares the extents rather than the scale, because rounding
    // can land a scale just under 1.0 on the native size anyway.
    [[nodiscard]] bool isNative() const
    {
        return extent_.width == outputExtent_.width && extent_.height == outputExtent_.height;
    }

private:
    VkExtent2D extent_{};
    VkExtent2D outputExtent_{};
    float scale_ = kMaxRenderScale;
};

} // namespace ve::renderer
