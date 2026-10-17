#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace ve {

// Writes a single-channel portable float map ("Pf"): 32-bit little-endian floats,
// linear, rows stored bottom to top as the format requires. For values a PNG
// would quantise or a tone curve would bend -- the ray-traced shadow mask's
// visibility, compared against a reference by tools/dev/compare_shadow_visibility.py.
// `rowsTopToBottom` holds width * height values in screen order. Returns false,
// having logged why, when the file cannot be written.
[[nodiscard]] bool writePfmGray(const std::filesystem::path& path,
                                uint32_t width,
                                uint32_t height,
                                std::span<const float> rowsTopToBottom);

} // namespace ve
