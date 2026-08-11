#include "renderer/RenderScale.h"

#include <algorithm>
#include <cmath>

namespace ve::renderer {

float clampRenderScale(float scale)
{
    // NaN fails every comparison, so std::clamp would propagate it into the
    // extent math. Treat it as "no scaling" rather than letting it through.
    if (!(scale == scale)) {
        return kMaxRenderScale;
    }

    return std::clamp(scale, kMinRenderScale, kMaxRenderScale);
}

RenderDimensions scaledRenderDimensions(uint32_t width, uint32_t height, float scale)
{
    const float clamped = clampRenderScale(scale);
    if (width == 0 || height == 0) {
        // A zero-sized swapchain (minimised window) has no meaningful scaled
        // size; hand back the input so the caller's existing zero-extent guard
        // is the one that fires.
        return {width, height};
    }

    const auto scaleDimension = [clamped](uint32_t dimension) {
        const float scaled = std::round(static_cast<float>(dimension) * clamped);
        return std::max(1u, static_cast<uint32_t>(scaled));
    };

    return {scaleDimension(width), scaleDimension(height)};
}

} // namespace ve::renderer
