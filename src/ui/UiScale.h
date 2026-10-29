#pragma once

#include <algorithm>

namespace ve::ui {

// The user's multiplier on top of the automatic debug-UI scale.
inline constexpr float kMinDebugUiScale = 0.5f;
inline constexpr float kMaxDebugUiScale = 2.0f;

// ImGui's default 13 px font and its paddings are sized for a display about
// 1080 rows tall. On a 1440p or 4K window that left the panels unreadably small
// and every slider a thin target, so the base scale follows the window height
// (never shrinking below 1) and the user's multiplier rides on top of it.
//
// displayHeight is in ImGui's logical units (io.DisplaySize.y), not framebuffer
// pixels: on a high-density display ImGui already renders a logical unit at the
// framebuffer scale, and counting those pixels again would double the size.
[[nodiscard]] inline float effectiveDebugUiScale(float displayHeight, float userScale)
{
    const float base = displayHeight > 0.0f ? std::clamp(displayHeight / 1080.0f, 1.0f, 3.0f) : 1.0f;
    return base * std::clamp(userScale, kMinDebugUiScale, kMaxDebugUiScale);
}

} // namespace ve::ui
