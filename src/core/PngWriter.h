#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace ve {

void writePngRgba8(const std::filesystem::path& path,
                   uint32_t width,
                   uint32_t height,
                   std::span<const uint8_t> rgbaPixels,
                   uint32_t rowStrideBytes);

} // namespace ve
