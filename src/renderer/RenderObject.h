#pragma once

#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/Transform.h"

#include <cstddef>
#include <string>

namespace ve::renderer {

struct RenderObject {
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    const Material* materialTable = nullptr;
    size_t materialCount = 0;
    Transform transform{};
    std::string debugName;
    bool animateTransform = false;
};

} // namespace ve::renderer
