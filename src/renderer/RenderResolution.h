#pragma once

#include "renderer/RenderScale.h"
#include "rhi/VulkanCommon.h"

#include <algorithm>

#include <glm/vec2.hpp>

namespace ve::renderer {

// The three resolutions a frame has, in one borrowable place.
//
// - `allocationExtent()` is how big every screen-space target is *created*. It
//   does not change with the render scale, only with the window.
// - `extent()` is how much of those targets a frame actually *writes* -- the
//   top-left sub-rect. This is the render resolution.
// - `outputExtent()` is the swapchain. Only the composite's viewport, the ImGui
//   overlay and portfolio screenshots use it.
//
// Sizing targets to the maximum and rendering into a sub-rect is what makes a
// render-scale change free: nothing is reallocated, only the viewport moves.
// The cost is memory (a 0.5-scale frame owns four times the texels it writes)
// and the obligation on every consumer to scale its UVs by `uvScale()` and stay
// inside the written region -- outside it lies whatever the last larger frame
// left behind, which is stale image data rather than anything obviously wrong.
//
// Renderer owns one of these and every resolution-dependent subsystem borrows it
// by const reference, the same way they already borrow the swapchain. That is
// deliberate: the alternative (a setter per subsystem) has four places to forget
// on a resize, and a subsystem that missed the update allocates a target at the
// wrong size, which shows up as a validation error deep in a later pass rather
// than at the mistake.
class RenderResolution final {
public:
    // Called on swapchain (re)creation and on every render-scale change. Only
    // the former moves allocationExtent(), which is what decides whether
    // anything has to be reallocated.
    void update(VkExtent2D outputExtent, float scale)
    {
        outputExtent_ = outputExtent;
        // Targets are allocated for the largest scale the setting allows, which
        // clampRenderScale caps at 1.0 -- so that is the presentation size.
        allocationExtent_ = outputExtent;
        scale_ = clampRenderScale(scale);
        const RenderDimensions dimensions =
            scaledRenderDimensions(outputExtent.width, outputExtent.height, scale_);
        extent_ = VkExtent2D{dimensions.width, dimensions.height};
    }

    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] VkExtent2D allocationExtent() const { return allocationExtent_; }
    [[nodiscard]] VkExtent2D outputExtent() const { return outputExtent_; }
    [[nodiscard]] float scale() const { return scale_; }

    // UV scale for sampling a full-size target that only its top-left sub-rect
    // was written into.
    [[nodiscard]] glm::vec2 uvScale() const { return subRectUvScale(extent_, allocationExtent_); }

    // True when the whole allocation is written, so callers can skip work that
    // only a scaled frame needs. Compares the extents rather than the scale,
    // because rounding can land a scale just under 1.0 on the native size.
    [[nodiscard]] bool isNative() const
    {
        return extent_.width == allocationExtent_.width && extent_.height == allocationExtent_.height;
    }

    // Both sizes of a derived target -- a half-resolution one, a mip -- shrink by
    // the same rule but round independently, so their ratio drifts from
    // uvScale() and each one has to carry its own. Getting this wrong reads
    // texels the frame never wrote.
    [[nodiscard]] static glm::vec2 subRectUvScale(VkExtent2D usedExtent, VkExtent2D allocatedExtent)
    {
        if (allocatedExtent.width == 0 || allocatedExtent.height == 0) {
            return glm::vec2(1.0f);
        }
        return glm::vec2(static_cast<float>(usedExtent.width) / static_cast<float>(allocatedExtent.width),
                         static_cast<float>(usedExtent.height) / static_cast<float>(allocatedExtent.height));
    }

    // Halves an extent the way every derived target here does, never reaching
    // zero. Shared so the used and allocated sides cannot drift apart by using
    // different rounding.
    [[nodiscard]] static VkExtent2D halved(VkExtent2D extent)
    {
        return VkExtent2D{std::max(1u, extent.width / 2u), std::max(1u, extent.height / 2u)};
    }

private:
    VkExtent2D extent_{};
    VkExtent2D allocationExtent_{};
    VkExtent2D outputExtent_{};
    float scale_ = kMaxRenderScale;
};

} // namespace ve::renderer
