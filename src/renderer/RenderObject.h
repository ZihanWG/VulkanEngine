#pragma once

#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/Transform.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
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
    CornellBox,
    // Thousands of objects behind occluder slabs. Exists to give the culling,
    // batching, LOD and frame-prep paths a load the default scene cannot.
    Stress,
    // Overlapping full-frame slabs. Loads the fragment path, which the geometry
    // stress scene deliberately does not.
    FragmentStress,
    // A ground plane under a strong low sun, with tall casters and one spot
    // light. The one scene here whose directional shadows are actually legible:
    // every other scene is ambient-dominated, with an umbra at 31.8/255 against
    // a lit floor at 80, which is why shadow work has had to be judged from
    // patch means rather than from the picture.
    SunlitYard
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

    // Which of the mesh's primitives this object draws. -1, the default, is the
    // whole mesh, which is every procedural scene in the repository.
    //
    // A glTF scene can instead import one object PER PRIMITIVE. Sponza is one
    // node holding 103 primitives, so as a single object it presented the culling
    // and shadow-caching paths with exactly one thing to consider, whatever its
    // real content was -- a frustum test against the bounds of an entire building
    // rejects nothing, and the per-object shadow caches had one key. Draw
    // submission was already per primitive (collectDrawItemsForObject), so this
    // changes the granularity of the decisions ABOUT objects, not of the draws.
    int primitiveIndex = -1;

    // Half-open range of indices into mesh->primitives() that this object draws:
    // [0, n) for a whole-mesh object, [k, k+1) for one primitive of one.
    //
    // The index matters, not just the primitive: DrawItem::submeshIndex addresses
    // the mesh's own table, and meshLocalLodRange looks the LOD chain up through
    // it. One accessor rather than the test repeated at each site, because a site
    // that iterated mesh->primitives() directly would draw, light or label the
    // whole building for an object that is one arch of it.
    [[nodiscard]] size_t firstPrimitiveIndex() const
    {
        return primitiveIndex < 0 ? 0 : static_cast<size_t>(primitiveIndex);
    }

    [[nodiscard]] size_t primitiveEndIndex() const
    {
        if (!mesh) {
            return 0;
        }
        const size_t available = mesh->primitives().size();
        if (primitiveIndex < 0) {
            return available;
        }
        // Clamped rather than trusted: a stale index would otherwise read past
        // the span.
        return std::min(available, firstPrimitiveIndex() + 1);
    }

    // The local extent this object is culled and lit against: the mesh's, or the
    // single primitive's when it draws one.
    //
    // Every bounds this object has goes through here. The alternative -- each
    // caller transforming mesh->localBounds() itself -- is what the per-primitive
    // split had to repair: updateObjectTransformCache re-derived bounds that way
    // for a good reason (it already had the matrix composed and did not want a
    // second one), and so it kept handing the GPU cull the bounds of the whole
    // building for every object. The frustum then rejected nothing, which looks
    // exactly like a scene that happens to be fully visible. Hence the overload
    // below: a caller with a matrix in hand can pass it in instead of opting out.
    [[nodiscard]] Aabb localBounds() const
    {
        if (!mesh) {
            return {};
        }

        if (primitiveIndex >= 0) {
            const std::span<const MeshPrimitive> primitives = mesh->primitives();
            const size_t index = firstPrimitiveIndex();
            if (index < primitives.size() && primitives[index].localBounds.valid()) {
                return primitives[index].localBounds;
            }
            // Falls through to the mesh bounds when the primitive has none -- an
            // older cook, or geometry whose accessor carried no min/max. Too large
            // a box over-includes rather than wrongly culling, which is the only
            // safe direction to be wrong in here.
        }

        return mesh->localBounds();
    }

    [[nodiscard]] Aabb worldBounds(const glm::mat4& model) const
    {
        if (!mesh) {
            return {};
        }
        return localBounds().transform(model);
    }

    [[nodiscard]] Aabb worldBounds() const
    {
        return worldBounds(transform.modelMatrix());
    }
};

} // namespace ve::renderer
