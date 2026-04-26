#pragma once

#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/Transform.h"

#include <string>

namespace ve::renderer {

struct RenderObject {
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    Transform transform{};
    std::string debugName;
};

} // namespace ve::renderer
