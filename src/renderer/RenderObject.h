#pragma once

#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/Transform.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ve::renderer {

enum class RenderObjectSourceType {
    BuiltInFallbackCube,
    ImportedGltf,
    PortfolioShowcase
};

struct RenderObject {
    uint32_t debugId = 0;
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    const Material* materialTable = nullptr;
    size_t materialCount = 0;
    Transform transform{};
    std::string debugName;
    RenderObjectSourceType sourceType = RenderObjectSourceType::BuiltInFallbackCube;
    bool animateTransform = false;
    bool portfolioOnly = false;
    bool hideInPortfolio = false;

    [[nodiscard]] Aabb worldBounds() const
    {
        if (!mesh) {
            return {};
        }

        return mesh->localBounds().transform(transform.modelMatrix());
    }
};

} // namespace ve::renderer
