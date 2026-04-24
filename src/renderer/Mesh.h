#pragma once

#include <cstdint>

namespace ve::renderer {

struct Mesh {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t vertexOffset = 0;
};

} // namespace ve::renderer
