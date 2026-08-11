#pragma once

#include <cstdint>

namespace ve::renderer {

// Render scale decouples the resolution the scene is *shaded* at from the
// resolution it is *presented* at. Everything up to and including TAA and the
// bloom chain runs at the scaled extent; the composite pass upsamples to the
// swapchain and the ImGui overlay stays native.
//
// This is the one knob that trades fragment cost close to linearly, which is
// what a fragment-bound frame needs -- on this engine the main HDR pass is
// 56-78% of the frame and its punctual light loop alone is ~64% of that, so
// halving the pixel count is worth far more than any local shader change left.
//
// The math is here rather than in the renderer so it stays free of Vulkan and
// unit-testable without a device, like the rest of VulkanEngineCore.

// Below a quarter the image is unusable even with TAA, and above 1.0 this would
// be supersampling -- a different feature with different plumbing (the composite
// would need a proper downsample filter, not a bilinear stretch).
inline constexpr float kMinRenderScale = 0.25f;
inline constexpr float kMaxRenderScale = 1.0f;

[[nodiscard]] float clampRenderScale(float scale);

struct RenderDimensions {
    uint32_t width = 0;
    uint32_t height = 0;
};

// Scales a presentation resolution into the internal render resolution. Rounds
// to nearest so a scale of 0.5 on an odd dimension does not systematically lose
// a pixel, and never returns zero in a dimension -- a zero-sized attachment is
// invalid, and every caller here would otherwise have to guard for it.
//
// The caller is expected to have clamped `scale` already; this clamps again
// rather than trusting it, because a stray value reaches Vulkan as an image
// create info and fails much less legibly there.
[[nodiscard]] RenderDimensions scaledRenderDimensions(uint32_t width, uint32_t height, float scale);

} // namespace ve::renderer
