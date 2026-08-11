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

// Top surface of the portfolio studio floor: position -0.56 + half of its 0.08
// thickness. The cube mesh spans [-0.5, 0.5], so an object's scale is its full
// extent and its bottom face is position.y - scale.y * 0.5.
constexpr float kPortfolioFloorTopY = -0.52f;

// How far an object that "stands on" the floor is sunk into it.
//
// Landing a bottom face exactly on kPortfolioFloorTopY is coplanar z-fighting,
// and TAA turns that from a cosmetic problem into a visible one: jitter moves
// the rasterized position by a fraction of a pixel every frame, so the
// interpolated depth at each pixel centre changes and the depth test flips
// between frames. The glass panes showed it worst -- they only depth-test, so a
// flip toggles whether the pixel has glass on it at all -- as a strip along the
// floor line that flickered with TAA on and froze the moment jitter was
// disabled.
//
// Intersecting rather than kissing removes the ambiguity: the bottom face ends
// up inside the floor slab (which spans 0.08), so it is unambiguously occluded.
constexpr float kPortfolioFloorSinkDepth = 0.02f;
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

// Stress scene. The default scene is eleven draw items, which is too few to
// exercise anything this renderer was built for: GPU culling rejects nothing,
// the parallel frame-prep loops never clear their 64-item chunk threshold, and
// LOD selection has no distance range to select over.
//
// The grid alternates cube and sphere so both batches are non-trivial, and the
// sphere carries the LOD chain. Occluder slabs stand between the camera and the
// far half so occlusion culling has something real to reject -- without them a
// large scene still reports zero rejections and proves nothing.
constexpr int kStressGridColumns = 48;
constexpr int kStressGridRows = 48;
constexpr int kStressOccluderCount = 6;
constexpr int kStressObjectCount =
    1 + kStressOccluderCount + (kStressGridColumns * kStressGridRows);
constexpr float kStressGridSpacing = 2.2f;

// Fragment stress. The geometry stress scene above loads culling and CPU frame
// prep, but it runs *faster* than the default scene because its objects are
// small on screen -- MainHDRPass is fragment-bound, so screen coverage is what
// costs, not object count.
//
// This one is the opposite: a handful of large slabs stacked in depth, each
// filling the frame, so every pixel is shaded several times over. Paired with a
// dense light cluster (see updateDemoLights) it drives the per-fragment light
// loop, which ablation put at ~64% of MainHDRPass.
constexpr int kFragmentStressLayerCount = 6;
constexpr int kFragmentStressObjectCount = kFragmentStressLayerCount + 1;
constexpr int kFragmentStressLightCount = 192;
constexpr float kFragmentStressLightRange = 14.0f;

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

    // Appends the stress scene: a ground slab, a row of occluder slabs, and a
    // kStressGridColumns x kStressGridRows grid alternating cube and sphere.
    bool appendStressScene(std::vector<RenderObject>& objects, std::string& status) const;

    // Appends the fragment stress scene: overlapping full-frame slabs whose
    // whole purpose is overdraw and screen coverage.
    bool appendFragmentStressScene(std::vector<RenderObject>& objects, std::string& status) const;

    // Restores the showcase objects' transforms/visibility to their authored
    // preset. Pure operation on an existing object list (the caller is responsible
    // for any GPU-state invalidation that should follow).
    static void resetPortfolioShowcaseToPreset(std::vector<RenderObject>& objects);

    [[nodiscard]] static bool hasPortfolioShowcase(const std::vector<RenderObject>& objects);
    [[nodiscard]] static bool hasOcclusionTest(const std::vector<RenderObject>& objects);
    [[nodiscard]] static bool hasCornellBox(const std::vector<RenderObject>& objects);
    [[nodiscard]] static bool hasStressScene(const std::vector<RenderObject>& objects);
    [[nodiscard]] static bool hasFragmentStressScene(const std::vector<RenderObject>& objects);

private:
    const Mesh& cubeMesh_;
    const Mesh& sphereMesh_;
    const std::vector<Material>& materials_;
    std::function<uint32_t()> allocateDebugId_;
};

} // namespace ve::renderer
