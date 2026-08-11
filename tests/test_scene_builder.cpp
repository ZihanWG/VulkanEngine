#include "renderer/SceneBuilder.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

using ve::renderer::Material;
using ve::renderer::Mesh;
using ve::renderer::RenderObject;
using ve::renderer::RenderObjectSourceType;
using ve::renderer::SceneBuilder;

namespace {

// A debug-id allocator that mirrors Renderer's: hands out a monotonically
// increasing id starting at 1. SceneBuilder takes this as a callback so the ids
// stay unique without SceneBuilder owning the counter.
struct IdAllocator {
    uint32_t next = 1;
    uint32_t operator()() { return next++; }
};

// Builds a material array sized so every portfolio slot index (up to
// kPortfolioCutoutLatticeMaterialIndex == 11) is valid, tagging the PBR-sample
// slots with the debug names SceneBuilder keys its has-checks off.
std::vector<Material> makePortfolioMaterials()
{
    std::vector<Material> materials(ve::renderer::kPortfolioGlassMaterialIndex + 1);
    materials[ve::renderer::kPortfolioHeroCeramicMaterialIndex].debugName = "Portfolio_HeroCeramic";
    materials[ve::renderer::kPortfolioMatteGrayMaterialIndex].debugName = "Portfolio_MatteGray";
    materials[ve::renderer::kPortfolioGlossyBlueMaterialIndex].debugName = "Portfolio_GlossyBlue";
    materials[ve::renderer::kPortfolioRoughMetalMaterialIndex].debugName = "Portfolio_RoughMetal";
    materials[ve::renderer::kPortfolioPolishedMetalSmallMaterialIndex].debugName = "Portfolio_PolishedMetalSmall";
    return materials;
}

std::vector<std::string> objectNames(const std::vector<RenderObject>& objects)
{
    std::vector<std::string> names;
    names.reserve(objects.size());
    for (const RenderObject& object : objects) {
        names.push_back(object.debugName);
    }
    return names;
}

} // namespace

TEST_CASE("SceneBuilder cube fallback produces the four authored cubes", "[scene]")
{
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials(4);
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    builder.appendCubeFallback(objects);

    REQUIRE(objectNames(objects) ==
            std::vector<std::string>{"Center Cube", "Left Cube", "Right Cube", "Elevated Cube"});

    for (size_t index = 0; index < objects.size(); ++index) {
        const RenderObject& object = objects[index];
        CHECK(object.sourceType == RenderObjectSourceType::BuiltInFallbackCube);
        CHECK(object.mesh == &cubeMesh);
        CHECK(object.material == &materials.at(index));
        CHECK(object.animateTransform);
        CHECK(object.hideInPortfolio);
        CHECK(object.sceneObjectId == object.debugId);
    }

    // Spot-check one authored transform so a future edit to the layout is caught.
    CHECK(objects.front().transform.position.x == Catch::Approx(0.0f));
    CHECK(objects.front().transform.position.y == Catch::Approx(-0.1f));
}

TEST_CASE("SceneBuilder portfolio showcase lays out floor, backdrop, plinth, and spheres", "[scene]")
{
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials = makePortfolioMaterials();
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    builder.appendPortfolioShowcase(objects);

    REQUIRE(objectNames(objects) == std::vector<std::string>{
                                        "Portfolio Studio Floor",
                                        "Portfolio Studio Backdrop",
                                        "Portfolio Side Plinth",
                                        "Portfolio Hero Ceramic",
                                        "Portfolio Matte Gray",
                                        "Portfolio Glossy Blue",
                                        "Portfolio Rough Metal",
                                        "Portfolio Polished Metal Small",
                                        "Portfolio Cutout Panel",
                                        "Portfolio Glass Pane Near",
                                        "Portfolio Glass Pane Far",
                                    });

    for (const RenderObject& object : objects) {
        CHECK(object.sourceType == RenderObjectSourceType::PortfolioShowcase);
        CHECK_FALSE(object.animateTransform);
        CHECK_FALSE(object.portfolioOnly);
    }

    // First three are cubes (floor/backdrop/plinth), then five spheres, and the
    // cutout panel is a cube again.
    CHECK(objects[0].mesh == &cubeMesh);
    CHECK(objects[1].mesh == &cubeMesh);
    CHECK(objects[2].mesh == &cubeMesh);
    for (size_t index = 3; index < 8; ++index) {
        CHECK(objects[index].mesh == &sphereMesh);
    }
    CHECK(objects[8].mesh == &cubeMesh);
    CHECK(objects[9].mesh == &cubeMesh);
    CHECK(objects[10].mesh == &cubeMesh);

    // Material slot wiring matches the named indices in SceneBuilder.h.
    CHECK(objects[0].material == &materials[ve::renderer::kPortfolioGroundMaterialIndex]);
    CHECK(objects[1].material == &materials[ve::renderer::kPortfolioBackdropMaterialIndex]);
    CHECK(objects[2].material == &materials[ve::renderer::kPortfolioGroundMaterialIndex]);
    CHECK(objects[3].material == &materials[ve::renderer::kPortfolioHeroCeramicMaterialIndex]);
    CHECK(objects[7].material == &materials[ve::renderer::kPortfolioPolishedMetalSmallMaterialIndex]);
    CHECK(objects[8].material == &materials[ve::renderer::kPortfolioCutoutLatticeMaterialIndex]);
    CHECK(objects[9].material == &materials[ve::renderer::kPortfolioGlassMaterialIndex]);
    CHECK(objects[10].material == &materials[ve::renderer::kPortfolioGlassMaterialIndex]);
}

TEST_CASE("SceneBuilder omits the alpha demo objects when their material slots are missing", "[scene]")
{
    // The showcase only needs slots through kPortfolioBackdropMaterialIndex, so a
    // material array that stops there must still build — just without the cutout
    // demo — rather than indexing past the end.
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials(ve::renderer::kPortfolioBackdropMaterialIndex + 1);
    materials[ve::renderer::kPortfolioHeroCeramicMaterialIndex].debugName = "Portfolio_HeroCeramic";
    materials[ve::renderer::kPortfolioMatteGrayMaterialIndex].debugName = "Portfolio_MatteGray";
    materials[ve::renderer::kPortfolioGlossyBlueMaterialIndex].debugName = "Portfolio_GlossyBlue";
    materials[ve::renderer::kPortfolioRoughMetalMaterialIndex].debugName = "Portfolio_RoughMetal";
    materials[ve::renderer::kPortfolioPolishedMetalSmallMaterialIndex].debugName = "Portfolio_PolishedMetalSmall";
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    builder.appendPortfolioShowcase(objects);

    CHECK(objects.size() == 8);
    const std::vector<std::string> names = objectNames(objects);
    CHECK(std::find(names.begin(), names.end(), "Portfolio Cutout Panel") == names.end());
    CHECK(std::find(names.begin(), names.end(), "Portfolio Glass Pane Near") == names.end());
}

TEST_CASE("SceneBuilder skips the portfolio showcase without portfolio materials", "[scene]")
{
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials(5); // fewer than kPortfolioBackdropMaterialIndex + 1
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    builder.appendPortfolioShowcase(objects);

    CHECK(objects.empty());
}

TEST_CASE("SceneBuilder occlusion test refuses to build without a valid mesh", "[scene]")
{
    // Mesh::valid() is false for a default mesh (no GPU buffers), so the occlusion
    // guard should reject the build and report why. The success path needs a real
    // GPU mesh and is exercised by the on-device run, not this headless test.
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials(11);
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    std::string status;
    const bool built = builder.appendOcclusionTest(objects, status);

    CHECK_FALSE(built);
    CHECK(objects.empty());
    CHECK(status.find("unavailable") != std::string::npos);
}

TEST_CASE("SceneBuilder reset restores showcase objects to their authored preset", "[scene]")
{
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials = makePortfolioMaterials();
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    builder.appendPortfolioShowcase(objects);

    // Perturb the hero sphere the way the editor/gizmo would.
    RenderObject& hero = objects.at(3);
    REQUIRE(hero.debugName == "Portfolio Hero Ceramic");
    hero.transform.position = {5.0f, 5.0f, 5.0f};
    hero.visible = false;
    hero.animateTransform = true;

    SceneBuilder::resetPortfolioShowcaseToPreset(objects);

    CHECK(hero.transform.position.x == Catch::Approx(0.0f));
    CHECK(hero.transform.position.y == Catch::Approx(-0.11f));
    CHECK(hero.transform.position.z == Catch::Approx(0.08f));
    CHECK(hero.visible);
    CHECK_FALSE(hero.animateTransform);
}

TEST_CASE("SceneBuilder hands out unique debug ids across builds", "[scene]")
{
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials = makePortfolioMaterials();
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    builder.appendCubeFallback(objects);
    builder.appendPortfolioShowcase(objects);

    std::set<uint32_t> seen;
    for (const RenderObject& object : objects) {
        CHECK(object.sceneObjectId == object.debugId);
        const bool inserted = seen.insert(object.debugId).second;
        CHECK(inserted); // every debug id is unique
    }
    CHECK(seen.size() == objects.size());
}

TEST_CASE("Stress scene reports unavailable without valid meshes", "[scene][stress]")
{
    // Same boundary the occlusion-test case checks: building real geometry needs
    // GPU meshes, so the headless test can only pin the refusal path.
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials(11);
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    std::string status;
    const bool built = builder.appendStressScene(objects, status);

    CHECK_FALSE(built);
    CHECK(objects.empty());
    CHECK(status.find("unavailable") != std::string::npos);
}

TEST_CASE("hasStressScene only matches stress objects with a mesh", "[scene][stress]")
{
    Mesh mesh;
    std::vector<RenderObject> objects;
    CHECK_FALSE(SceneBuilder::hasStressScene(objects));

    // Another preset's objects must not be mistaken for the stress scene, or
    // loading one would silently skip rebuilding the other.
    RenderObject cornell{};
    cornell.sourceType = ve::renderer::RenderObjectSourceType::CornellBox;
    cornell.mesh = &mesh;
    objects.push_back(cornell);
    CHECK_FALSE(SceneBuilder::hasStressScene(objects));

    // A stress entry that never got a mesh is not a loaded scene either.
    RenderObject meshless{};
    meshless.sourceType = ve::renderer::RenderObjectSourceType::Stress;
    objects.push_back(meshless);
    CHECK_FALSE(SceneBuilder::hasStressScene(objects));

    RenderObject stress{};
    stress.sourceType = ve::renderer::RenderObjectSourceType::Stress;
    stress.mesh = &mesh;
    objects.push_back(stress);
    CHECK(SceneBuilder::hasStressScene(objects));
}

TEST_CASE("Stress object count matches its grid arithmetic", "[scene][stress]")
{
    // reserve() uses this constant; if it drifts from the loops the vector
    // reallocates mid-build, which is silent but defeats the point of reserving.
    CHECK(ve::renderer::kStressObjectCount ==
          1 + ve::renderer::kStressOccluderCount +
              (ve::renderer::kStressGridColumns * ve::renderer::kStressGridRows));
    CHECK(ve::renderer::kStressObjectCount > 2000);
}

TEST_CASE("Fragment stress scene reports unavailable without a valid mesh", "[scene][stress]")
{
    Mesh cubeMesh;
    Mesh sphereMesh;
    std::vector<Material> materials(11);
    IdAllocator ids;
    SceneBuilder builder(cubeMesh, sphereMesh, materials, std::ref(ids));

    std::vector<RenderObject> objects;
    std::string status;
    CHECK_FALSE(builder.appendFragmentStressScene(objects, status));
    CHECK(objects.empty());
    CHECK(status.find("unavailable") != std::string::npos);
}

TEST_CASE("The two stress scenes are told apart by their source type", "[scene][stress]")
{
    // They are mutually exclusive at load time, so a has-check that matched both
    // would make loading one skip rebuilding the other.
    Mesh mesh;
    std::vector<RenderObject> objects;

    RenderObject geometry{};
    geometry.sourceType = ve::renderer::RenderObjectSourceType::Stress;
    geometry.mesh = &mesh;
    objects.push_back(geometry);

    CHECK(SceneBuilder::hasStressScene(objects));
    CHECK_FALSE(SceneBuilder::hasFragmentStressScene(objects));

    RenderObject fragment{};
    fragment.sourceType = ve::renderer::RenderObjectSourceType::FragmentStress;
    fragment.mesh = &mesh;
    objects.push_back(fragment);

    CHECK(SceneBuilder::hasStressScene(objects));
    CHECK(SceneBuilder::hasFragmentStressScene(objects));
}
