#pragma once

// SceneBuilder constructs the renderer's CPU-side scene object lists (the default
// portfolio showcase, the built-in cube fallback, and the procedural occlusion
// test scene) from already-created shared meshes and materials.
//
// It performs no GPU work: it only appends RenderObject records and reads back
// their layout, which keeps the scene-layout logic free of Vulkan device state
// and isolated from the rest of the monolithic Renderer.
//
// Ownership boundary (Design B, reference-borrowing): the shared meshes and the
// material array stay owned by Renderer and are borrowed here by const reference;
// debug-id allocation also stays in Renderer and is injected as a callback so the
// ids handed out remain globally unique across every scene the renderer builds.

#include "renderer/RenderObject.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <glm/vec3.hpp>

namespace ve::renderer {

// Material slot indices within Renderer's materialVariants_ array. Material
// creation (Renderer::createMaterial) populates the array in this order, and the
// scene-building functions index into it by these names.
constexpr size_t kPortfolioGroundMaterialIndex = 4;
constexpr size_t kPortfolioMatteGrayMaterialIndex = 5;
constexpr size_t kPortfolioGlossyBlueMaterialIndex = 6;
constexpr size_t kPortfolioRoughMetalMaterialIndex = 7;
constexpr size_t kPortfolioPolishedMetalSmallMaterialIndex = 8;
constexpr size_t kPortfolioHeroCeramicMaterialIndex = 9;
constexpr size_t kPortfolioBackdropMaterialIndex = 10;
// glTF MASK material for the perforated-panel cutout demo. Its holes must show up
// in the shadow map too, which is what the alpha-tested shadow pipeline adds.
constexpr size_t kPortfolioCutoutLatticeMaterialIndex = 11;
// glTF BLEND material for the transparency demo. Sorted back to front and drawn
// in the dedicated transparent pass, after every screen-space effect.
constexpr size_t kPortfolioGlassMaterialIndex = 12;
// Cornell box surfaces. Saturated and matte on purpose: colour bleeding is
// proportional to how saturated the bouncing surface is, and any specular would
// be lost in a convolution over 8x8 directions anyway.
constexpr size_t kCornellWhiteMaterialIndex = 13;
constexpr size_t kCornellRedMaterialIndex = 14;
constexpr size_t kCornellGreenMaterialIndex = 15;
// Floor, ceiling, back wall, two coloured side walls, two blocks.
constexpr int kCornellBoxObjectCount = 7;
// Interior half-extent. The room spans [-half, half] in X and Z and [0, 2*half]
// in Y, open toward +Z so a camera can see in.
constexpr float kCornellBoxHalfExtent = 5.0f;
constexpr int kOcclusionTestGridColumns = 12;
constexpr int kOcclusionTestGridRows = 10;
constexpr int kOcclusionTestOccluderCount = 5;
constexpr int kOcclusionTestObjectCount =
    1 + kOcclusionTestOccluderCount + (kOcclusionTestGridColumns * kOcclusionTestGridRows);

class SceneBuilder {
public:
    SceneBuilder(const Mesh& cubeMesh,
                 const Mesh& sphereMesh,
                 const std::vector<Material>& materials,
                 std::function<uint32_t()> allocateDebugId);

    SceneBuilder(const SceneBuilder&) = delete;
    SceneBuilder& operator=(const SceneBuilder&) = delete;

    // Appends the default portfolio showcase (studio floor/backdrop/plinth plus
    // the hero ceramic sphere and the PBR material-sample spheres). No-op when the
    // showcase is already present or the portfolio materials are unavailable.
    void appendPortfolioShowcase(std::vector<RenderObject>& objects) const;

    // Appends the four-cube built-in fallback scene.
    void appendCubeFallback(std::vector<RenderObject>& objects) const;

    // Appends the procedural occlusion-test scene (ground, occluder walls, and a
    // grid of mostly-hidden cubes). Returns false and writes `status` if the cube
    // mesh or runtime materials are unavailable.
    bool appendOcclusionTest(std::vector<RenderObject>& objects, std::string& status) const;

    // Appends a closed Cornell-style room: white floor, ceiling and back wall,
    // one red and one green side wall, and two blocks. Returns false and writes
    // `status` when the cube mesh or the Cornell materials are unavailable.
    bool appendCornellBox(std::vector<RenderObject>& objects, std::string& status) const;

    // Restores the showcase objects' transforms/visibility to their authored
    // preset. Pure operation on an existing object list (the caller is responsible
    // for any GPU-state invalidation that should follow).
    static void resetPortfolioShowcaseToPreset(std::vector<RenderObject>& objects);

    [[nodiscard]] static bool hasPortfolioShowcase(const std::vector<RenderObject>& objects);
    [[nodiscard]] static bool hasOcclusionTest(const std::vector<RenderObject>& objects);
    [[nodiscard]] static bool hasCornellBox(const std::vector<RenderObject>& objects);

private:
    const Mesh& cubeMesh_;
    const Mesh& sphereMesh_;
    const std::vector<Material>& materials_;
    std::function<uint32_t()> allocateDebugId_;
};

} // namespace ve::renderer
