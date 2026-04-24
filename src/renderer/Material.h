#pragma once

#include <cstdint>

namespace ve::renderer {

struct Material {
    uint32_t baseColorTextureIndex = UINT32_MAX;
};

} // namespace ve::renderer
