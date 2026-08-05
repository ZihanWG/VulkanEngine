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
    PortfolioShowcase,
    OcclusionTest,
    // A closed, coloured room. The one scene here that can actually show
    // indirect light: colour bleeding needs saturated walls, and a second
    // bounce needs somewhere for light to be trapped.
    CornellBox
};

struct RenderObject {
    uint32_t debugId = 0;
    uint32_t sceneObjectId = 0;
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    const Material* materialTable = nullptr;
    size_t materialCount = 0;
    Transform transform{};
    // Previous-frame model matrix captured at the end of each frame update and
    // used to build per-object motion vectors; previousModelValid is false until
    // the object has been through one frame (velocity then sees zero object motion).
    glm::mat4 previousModelMatrix{1.0f};
    bool previousModelValid = false;
    std::string debugName;
    RenderObjectSourceType sourceType = RenderObjectSourceType::BuiltInFallbackCube;
    bool visible = true;
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
