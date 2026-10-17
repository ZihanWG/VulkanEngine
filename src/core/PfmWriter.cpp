#include "core/PfmWriter.h"

#include "core/Logger.h"

#include <bit>
#include <cstddef>
#include <fstream>
#include <string>

namespace ve {

bool writePfmGray(const std::filesystem::path& path,
                  uint32_t width,
                  uint32_t height,
                  std::span<const float> rowsTopToBottom)
{
    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (width == 0 || height == 0 || rowsTopToBottom.size() != texelCount) {
        Logger::error("PFM write: " + std::to_string(rowsTopToBottom.size()) + " values do not fill " +
                      std::to_string(width) + "x" + std::to_string(height) + ".");
        return false;
    }
    // The scale's sign names the byte order: negative is little-endian, which is
    // what every platform this builds on stores, and the static_assert keeps it so.
    static_assert(std::endian::native == std::endian::little, "PFM output assumes a little-endian host.");

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        Logger::error("PFM write: cannot open " + path.string() + ".");
        return false;
    }
    file << "Pf\n" << width << ' ' << height << "\n-1.0\n";
    for (uint32_t row = height; row-- > 0;) {
        const std::span<const float> line = rowsTopToBottom.subspan(static_cast<size_t>(row) * width, width);
        file.write(reinterpret_cast<const char*>(line.data()), static_cast<std::streamsize>(line.size_bytes()));
    }
    if (!file) {
        Logger::error("PFM write: writing " + path.string() + " failed.");
        return false;
    }
    return true;
}

} // namespace ve
