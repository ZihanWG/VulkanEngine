#include "renderer/SceneBuilder.h"

#include "core/Logger.h"

#include <array>
#include <sstream>
#include <string_view>
#include <utility>

#include <glm/common.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

SceneBuilder::SceneBuilder(const Mesh& cubeMesh,
                           const Mesh& sphereMesh,
                           const std::vector<Material>& materials,
                           std::function<uint32_t()> allocateDebugId)
    : cubeMesh_(cubeMesh),
      sphereMesh_(sphereMesh),
      materials_(materials),
      allocateDebugId_(std::move(allocateDebugId))
{
}

void SceneBuilder::appendPortfolioShowcase(std::vector<RenderObject>& objects) const
{
    if (hasPortfolioShowcase(objects)) {
        return;
    }
    if (materials_.size() <= kPortfolioBackdropMaterialIndex) {
        Logger::warn("Portfolio showcase scene was not added because portfolio materials are unavailable.");
        return;
    }

    const auto addPortfolioCube = [this, &objects](std::string debugName,
                                                   const Material* material,
                                                   const glm::vec3& position,
                                                   const glm::vec3& rotationRadians,
                                                   const glm::vec3& scale) {
        RenderObject cube{};
        cube.debugId = allocateDebugId_();
        cube.sceneObjectId = cube.debugId;
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.sourceType = RenderObjectSourceType::PortfolioShowcase;
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = false;
        // Visible in both the editor and F11 portfolio capture (the showcase is the
        // default scene now); portfolio detection keys off sourceType, not this flag.
        cube.portfolioOnly = false;
        objects.push_back(std::move(cube));
    };

    const auto addPortfolioSphere = [this, &objects](std::string debugName,
                                                     const Material* material,
                                                     const glm::vec3& position,
                                                     const glm::vec3& scale) {
        RenderObject sphere{};
        sphere.debugId = allocateDebugId_();
        sphere.sceneObjectId = sphere.debugId;
        sphere.mesh = &sphereMesh_;
        sphere.material = material;
        sphere.debugName = std::move(debugName);
        sphere.sourceType = RenderObjectSourceType::PortfolioShowcase;
        sphere.transform.position = position;
        sphere.transform.scale = scale;
        sphere.animateTransform = false;
        sphere.portfolioOnly = false;
        objects.push_back(std::move(sphere));
    };

    objects.reserve(objects.size() + 8);
    addPortfolioCube("Portfolio Studio Floor",
                     &materials_.at(kPortfolioGroundMaterialIndex),
                     {0.0f, -0.56f, 0.24f},
                     {0.0f, 0.0f, 0.0f},
                     {11.0f, 0.08f, 6.4f});
    addPortfolioCube("Portfolio Studio Backdrop",
                     &materials_.at(kPortfolioBackdropMaterialIndex),
                     {0.0f, 2.08f, -2.82f},
                     {0.0f, 0.0f, 0.0f},
                     {60.0f, 9.0f, 0.08f});
    addPortfolioCube("Portfolio Side Plinth",
                     &materials_.at(kPortfolioGroundMaterialIndex),
                     {1.02f, -0.42f, -0.18f},
                     {0.0f, -0.16f, 0.0f},
                     {0.96f, 0.28f, 0.70f});
    addPortfolioSphere("Portfolio Hero Ceramic",
                       &materials_.at(kPortfolioHeroCeramicMaterialIndex),
                       {0.0f, -0.11f, 0.08f},
                       {0.82f, 0.82f, 0.82f});
    addPortfolioSphere("Portfolio Matte Gray",
                       &materials_.at(kPortfolioMatteGrayMaterialIndex),
                       {-0.92f, -0.24f, 0.06f},
                       {0.56f, 0.56f, 0.56f});
    addPortfolioSphere("Portfolio Glossy Blue",
                       &materials_.at(kPortfolioGlossyBlueMaterialIndex),
                       {-0.62f, -0.33f, 0.66f},
                       {0.38f, 0.38f, 0.38f});
    addPortfolioSphere("Portfolio Rough Metal",
                       &materials_.at(kPortfolioRoughMetalMaterialIndex),
                       {0.96f, -0.29f, 0.42f},
                       {0.46f, 0.46f, 0.46f});
    addPortfolioSphere("Portfolio Polished Metal Small",
                       &materials_.at(kPortfolioPolishedMetalSmallMaterialIndex),
                       {1.04f, -0.09f, -0.18f},
                       {0.38f, 0.38f, 0.38f});

    // Alpha-mask demo. Stands on the floor, angled into the key light so the
    // perforations read both in the panel itself and in the shadow it throws.
    if (materials_.size() > kPortfolioCutoutLatticeMaterialIndex) {
        addPortfolioCube("Portfolio Cutout Panel",
                         &materials_.at(kPortfolioCutoutLatticeMaterialIndex),
                         {-1.55f, -0.21f, 0.34f},
                         {0.0f, 0.38f, 0.0f},
                         {0.92f, 0.62f, 0.05f});
    }

    // Alpha-blend demo: two overlapping glass panes at different depths in front
    // of the material spheres. Two of them on purpose -- a single transparent
    // surface looks correct even with the sort broken, while an overlapping pair
    // immediately shows whether back-to-front ordering actually holds.
    if (materials_.size() > kPortfolioGlassMaterialIndex) {
        addPortfolioCube("Portfolio Glass Pane Near",
                         &materials_.at(kPortfolioGlassMaterialIndex),
                         {0.34f, -0.16f, 1.28f},
                         {0.0f, -0.22f, 0.0f},
                         {0.86f, 0.72f, 0.04f});
        addPortfolioCube("Portfolio Glass Pane Far",
                         &materials_.at(kPortfolioGlassMaterialIndex),
                         {-0.22f, -0.20f, 0.94f},
                         {0.0f, 0.30f, 0.0f},
                         {0.70f, 0.64f, 0.04f});
    }
}

void SceneBuilder::appendCubeFallback(std::vector<RenderObject>& objects) const
{
    const auto addCube = [this, &objects](std::string debugName,
                                          const Material* material,
                                          const glm::vec3& position,
                                          const glm::vec3& rotationRadians,
                                          const glm::vec3& scale,
                                          bool animateTransform = true,
                                          bool portfolioOnly = false,
                                          bool hideInPortfolio = false,
                                          RenderObjectSourceType sourceType =
                                              RenderObjectSourceType::BuiltInFallbackCube) {
        RenderObject cube{};
        cube.debugId = allocateDebugId_();
        cube.sceneObjectId = cube.debugId;
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.sourceType = sourceType;
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = animateTransform;
        cube.portfolioOnly = portfolioOnly;
        cube.hideInPortfolio = hideInPortfolio;
        objects.push_back(std::move(cube));
    };

    objects.reserve(objects.size() + 4);
    addCube("Center Cube",
            &materials_.at(0),
            {0.0f, -0.1f, 0.0f},
            {0.2f, 0.0f, 0.0f},
            {0.7f, 0.7f, 0.7f},
            true,
            false,
            true);
    addCube(
        "Left Cube",
        &materials_.at(1),
        {-1.35f, -0.15f, -0.35f},
        {0.0f, 0.35f, 0.2f},
        {0.5f, 0.5f, 0.5f},
        true,
        false,
        true);
    addCube("Right Cube",
            &materials_.at(2),
            {1.35f, -0.05f, -0.25f},
            {0.25f, -0.35f, 0.0f},
            {0.55f, 0.8f, 0.55f},
            true,
            false,
            true);
    addCube("Elevated Cube",
            &materials_.at(3),
            {0.0f, 1.0f, -0.7f},
            {-0.3f, 0.2f, 0.45f},
            {0.45f, 0.45f, 0.45f},
            true,
            false,
            true);
}

bool SceneBuilder::appendOcclusionTest(std::vector<RenderObject>& objects, std::string& status) const
{
    if (!cubeMesh_.valid() || materials_.empty()) {
        status = "Occlusion test scene is unavailable: cube mesh or runtime materials are not initialized.";
        Logger::warn(status);
        return false;
    }

    const auto materialAt = [this](size_t materialIndex) -> const Material* {
        return &materials_.at(materialIndex % materials_.size());
    };

    const auto addCube = [this, &objects](std::string debugName,
                                          const Material* material,
                                          const glm::vec3& position,
                                          const glm::vec3& rotationRadians,
                                          const glm::vec3& scale) {
        RenderObject cube{};
        cube.debugId = allocateDebugId_();
        cube.sceneObjectId = cube.debugId;
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.sourceType = RenderObjectSourceType::OcclusionTest;
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = false;
        cube.portfolioOnly = false;
        cube.hideInPortfolio = true;
        objects.push_back(std::move(cube));
    };

    objects.reserve(objects.size() + static_cast<size_t>(kOcclusionTestObjectCount));

    const size_t groundMaterial = materials_.size() > kPortfolioGroundMaterialIndex
                                      ? kPortfolioGroundMaterialIndex
                                      : 0;
    addCube("Occlusion Test Ground",
            materialAt(groundMaterial),
            {0.0f, -0.10f, -4.0f},
            {0.0f, 0.0f, 0.0f},
            {13.0f, 0.12f, 22.0f});

    const std::array<float, kOcclusionTestOccluderCount> occluderX = {-2.6f, -1.3f, 0.0f, 1.3f, 2.6f};
    for (size_t occluderIndex = 0; occluderIndex < occluderX.size(); ++occluderIndex) {
        std::ostringstream name;
        name << "Occlusion Test Wall " << (occluderIndex + 1);
        const float zOffset = (occluderIndex % 2 == 0) ? 0.15f : -0.08f;
        addCube(name.str(),
                materialAt(occluderIndex),
                {occluderX[occluderIndex], 1.35f, 0.35f + zOffset},
                {0.0f, 0.0f, 0.0f},
                {0.98f, 3.10f, 0.55f});
    }

    for (int row = 0; row < kOcclusionTestGridRows; ++row) {
        for (int column = 0; column < kOcclusionTestGridColumns; ++column) {
            const float x = -5.5f + static_cast<float>(column) * 1.0f;
            const float z = -2.4f - static_cast<float>(row) * 1.05f;
            const bool topWitnessRow = row == kOcclusionTestGridRows - 1;
            const bool sideWitnessColumn = column == 0 || column == kOcclusionTestGridColumns - 1;
            const float y = topWitnessRow ? 3.05f : 0.22f + 0.34f * static_cast<float>((row + column) % 3);
            const float uniformScale = sideWitnessColumn ? 0.40f : 0.32f + 0.035f * static_cast<float>((row + column) % 4);

            std::ostringstream name;
            name << "Occlusion Test Hidden Cube r" << row << " c" << column;
            if (topWitnessRow || sideWitnessColumn) {
                name << " visible-edge";
            }

            addCube(name.str(),
                    materialAt(static_cast<size_t>((row + column) % 4)),
                    {x, y, z},
                    {0.0f, 0.15f * static_cast<float>((row + column) % 5), 0.0f},
                    {uniformScale, uniformScale, uniformScale});
        }
    }

    return true;
}

void SceneBuilder::resetPortfolioShowcaseToPreset(std::vector<RenderObject>& objects)
{
    const auto resetObject = [&objects](std::string_view debugName,
                                        const glm::vec3& position,
                                        const glm::vec3& rotationRadians,
                                        const glm::vec3& scale) {
        for (RenderObject& object : objects) {
            if (object.sourceType != RenderObjectSourceType::PortfolioShowcase ||
                object.debugName != debugName) {
                continue;
            }

            object.transform.position = position;
            object.transform.rotationRadians = rotationRadians;
            object.transform.scale = scale;
            object.transform.matrixOverride = glm::mat4{1.0f};
            object.transform.useMatrixOverride = false;
            object.visible = true;
            object.animateTransform = false;
            object.portfolioOnly = false;
            object.hideInPortfolio = false;
            return;
        }
    };

    resetObject("Portfolio Studio Floor", {0.0f, -0.56f, 0.24f}, {0.0f, 0.0f, 0.0f}, {11.0f, 0.08f, 6.4f});
    resetObject("Portfolio Studio Backdrop", {0.0f, 2.08f, -2.82f}, {0.0f, 0.0f, 0.0f}, {60.0f, 9.0f, 0.08f});
    resetObject("Portfolio Side Plinth", {1.02f, -0.42f, -0.18f}, {0.0f, -0.16f, 0.0f}, {0.96f, 0.28f, 0.70f});
    resetObject("Portfolio Hero Ceramic", {0.0f, -0.11f, 0.08f}, {0.0f, 0.0f, 0.0f}, {0.82f, 0.82f, 0.82f});
    resetObject("Portfolio Matte Gray", {-0.92f, -0.24f, 0.06f}, {0.0f, 0.0f, 0.0f}, {0.56f, 0.56f, 0.56f});
    resetObject("Portfolio Glossy Blue", {-0.62f, -0.33f, 0.66f}, {0.0f, 0.0f, 0.0f}, {0.38f, 0.38f, 0.38f});
    resetObject("Portfolio Rough Metal", {0.96f, -0.29f, 0.42f}, {0.0f, 0.0f, 0.0f}, {0.46f, 0.46f, 0.46f});
    resetObject("Portfolio Polished Metal Small", {1.04f, -0.09f, -0.18f}, {0.0f, 0.0f, 0.0f}, {0.38f, 0.38f, 0.38f});
    resetObject("Portfolio Cutout Panel", {-1.55f, -0.21f, 0.34f}, {0.0f, 0.38f, 0.0f}, {0.92f, 0.62f, 0.05f});
    resetObject("Portfolio Glass Pane Near", {0.34f, -0.16f, 1.28f}, {0.0f, -0.22f, 0.0f}, {0.86f, 0.72f, 0.04f});
    resetObject("Portfolio Glass Pane Far", {-0.22f, -0.20f, 0.94f}, {0.0f, 0.30f, 0.0f}, {0.70f, 0.64f, 0.04f});
}

bool SceneBuilder::hasPortfolioShowcase(const std::vector<RenderObject>& objects)
{
    bool hasFloor = false;
    bool hasBackdrop = false;
    bool hasHero = false;
    size_t materialSampleCount = 0;

    for (const RenderObject& object : objects) {
        if (object.sourceType != RenderObjectSourceType::PortfolioShowcase || !object.mesh ||
            !object.mesh->valid() || !object.material) {
            continue;
        }

        if (object.debugName == "Portfolio Studio Floor") {
            hasFloor = true;
            continue;
        }
        if (object.debugName == "Portfolio Studio Backdrop") {
            hasBackdrop = true;
            continue;
        }
        if (object.material->debugName == "Portfolio_HeroCeramic") {
            hasHero = true;
            continue;
        }
        if (object.material->debugName == "Portfolio_MatteGray" ||
            object.material->debugName == "Portfolio_GlossyBlue" ||
            object.material->debugName == "Portfolio_RoughMetal" ||
            object.material->debugName == "Portfolio_PolishedMetalSmall") {
            ++materialSampleCount;
        }
    }

    return hasFloor && hasBackdrop && hasHero && materialSampleCount >= 4;
}

bool SceneBuilder::appendCornellBox(std::vector<RenderObject>& objects, std::string& status) const
{
    if (!cubeMesh_.valid() || materials_.size() <= kCornellGreenMaterialIndex) {
        status = "Cornell box is unavailable: cube mesh or Cornell materials are not initialized.";
        Logger::warn(status);
        return false;
    }

    const auto addSlab = [this, &objects](std::string debugName,
                                          size_t materialIndex,
                                          const glm::vec3& position,
                                          const glm::vec3& rotationRadians,
                                          const glm::vec3& scale) {
        RenderObject slab{};
        slab.debugId = allocateDebugId_();
        slab.sceneObjectId = slab.debugId;
        slab.mesh = &cubeMesh_;
        slab.material = &materials_.at(materialIndex);
        slab.debugName = std::move(debugName);
        slab.sourceType = RenderObjectSourceType::CornellBox;
        slab.transform.position = position;
        slab.transform.rotationRadians = rotationRadians;
        slab.transform.scale = scale;
        slab.animateTransform = false;
        slab.portfolioOnly = false;
        slab.hideInPortfolio = true;
        objects.push_back(std::move(slab));
    };

    constexpr float kHalf = kCornellBoxHalfExtent;
    constexpr float kSize = kHalf * 2.0f;
    // Thin enough to read as a surface, thick enough that the probe visibility
    // moments see a wall rather than a plane they can average across.
    constexpr float kWallThickness = 0.25f;
    constexpr float kHalfThickness = kWallThickness * 0.5f;

    objects.reserve(objects.size() + static_cast<size_t>(kCornellBoxObjectCount));

    // The room. Walls sit just outside the interior so the interior really is
    // kSize on a side, which is what the probe grid is fitted to.
    addSlab("Cornell Floor", kCornellWhiteMaterialIndex, {0.0f, -kHalfThickness, 0.0f}, {}, {kSize, kWallThickness, kSize});
    addSlab("Cornell Ceiling",
            kCornellWhiteMaterialIndex,
            {0.0f, kSize + kHalfThickness, 0.0f},
            {},
            {kSize, kWallThickness, kSize});
    addSlab("Cornell Back Wall",
            kCornellWhiteMaterialIndex,
            {0.0f, kHalf, -kHalf - kHalfThickness},
            {},
            {kSize, kSize, kWallThickness});
    // The two coloured walls are the whole point: everything white in this room
    // picks up their colour from one bounce, and that tint is the most legible
    // evidence that indirect light is being transported at all.
    addSlab("Cornell Left Wall (red)",
            kCornellRedMaterialIndex,
            {-kHalf - kHalfThickness, kHalf, 0.0f},
            {},
            {kWallThickness, kSize, kSize});
    addSlab("Cornell Right Wall (green)",
            kCornellGreenMaterialIndex,
            {kHalf + kHalfThickness, kHalf, 0.0f},
            {},
            {kWallThickness, kSize, kSize});
    // Open toward +Z: without a missing wall there is nothing to look through,
    // and a probe grid inside a sealed box would never see the camera's side.

    // Two blocks, rotated off-axis the way the original has them, so the floor
    // between them is lit almost entirely by bounce rather than directly.
    addSlab("Cornell Tall Block",
            kCornellWhiteMaterialIndex,
            {-1.7f, 3.0f, -1.6f},
            {0.0f, -0.30f, 0.0f},
            {2.8f, 6.0f, 2.8f});
    addSlab("Cornell Short Block",
            kCornellWhiteMaterialIndex,
            {1.9f, 1.5f, 1.1f},
            {0.0f, 0.28f, 0.0f},
            {2.8f, 3.0f, 2.8f});

    status = "Cornell box active: closed room with red and green side walls, lit by one overhead light.";
    return true;
}

bool SceneBuilder::appendStressScene(std::vector<RenderObject>& objects, std::string& status) const
{
    if (!cubeMesh_.valid() || !sphereMesh_.valid() || materials_.empty()) {
        status = "Stress scene is unavailable: cube/sphere mesh or runtime materials are not initialized.";
        Logger::warn(status);
        return false;
    }

    const auto materialAt = [this](size_t materialIndex) -> const Material* {
        return &materials_.at(materialIndex % materials_.size());
    };

    const auto add = [this, &objects](std::string debugName,
                                      const Mesh& mesh,
                                      const Material* material,
                                      const glm::vec3& position,
                                      const glm::vec3& scale) {
        RenderObject object{};
        object.debugId = allocateDebugId_();
        object.sceneObjectId = object.debugId;
        object.mesh = &mesh;
        object.material = material;
        object.debugName = std::move(debugName);
        object.sourceType = RenderObjectSourceType::Stress;
        object.transform.position = position;
        object.transform.scale = scale;
        // Static on purpose: an animated transform invalidates the depth
        // pyramid every frame, which would leave occlusion culling with nothing
        // to test against -- the opposite of what this scene is for.
        object.animateTransform = false;
        object.portfolioOnly = false;
        object.hideInPortfolio = true;
        objects.push_back(std::move(object));
    };

    objects.reserve(objects.size() + static_cast<size_t>(kStressObjectCount));

    constexpr float kHalfSpanX = 0.5f * kStressGridSpacing * static_cast<float>(kStressGridColumns - 1);
    constexpr float kHalfSpanZ = 0.5f * kStressGridSpacing * static_cast<float>(kStressGridRows - 1);

    const size_t groundMaterial =
        materials_.size() > kPortfolioGroundMaterialIndex ? kPortfolioGroundMaterialIndex : 0;
    add("Stress Ground",
        cubeMesh_,
        materialAt(groundMaterial),
        {0.0f, -0.6f, -kHalfSpanZ},
        {kHalfSpanX * 2.4f, 0.4f, kHalfSpanZ * 2.4f});

    // Slabs across the near half. The camera preset sits in front of them, so
    // most of the grid behind is occluded and the Hi-Z test can reject it.
    for (int occluder = 0; occluder < kStressOccluderCount; ++occluder) {
        const float t = static_cast<float>(occluder) / static_cast<float>(kStressOccluderCount - 1);
        const float x = -kHalfSpanX * 0.75f + t * kHalfSpanX * 1.5f;
        add("Stress Occluder " + std::to_string(occluder),
            cubeMesh_,
            materialAt(static_cast<size_t>(occluder)),
            {x, 6.0f, -kHalfSpanZ * 0.35f},
            {kHalfSpanX * 0.22f, 12.0f, 1.2f});
    }

    for (int row = 0; row < kStressGridRows; ++row) {
        for (int column = 0; column < kStressGridColumns; ++column) {
            const float x = -kHalfSpanX + static_cast<float>(column) * kStressGridSpacing;
            const float z = -static_cast<float>(row) * kStressGridSpacing;
            const bool useSphere = ((row + column) % 2) == 0;
            // Varied scale so LOD selection spans levels rather than pinning
            // every item to one.
            const float scale = 0.45f + 0.35f * static_cast<float>((row * 7 + column * 3) % 5) / 4.0f;

            add("Stress " + std::string(useSphere ? "Sphere" : "Cube") + " r" + std::to_string(row) + " c" +
                    std::to_string(column),
                useSphere ? sphereMesh_ : cubeMesh_,
                materialAt(static_cast<size_t>((row + column) % materials_.size())),
                {x, scale, z},
                {scale, scale, scale});
        }
    }

    return true;
}

bool SceneBuilder::hasStressScene(const std::vector<RenderObject>& objects)
{
    for (const RenderObject& object : objects) {
        if (object.sourceType == RenderObjectSourceType::Stress && object.mesh != nullptr) {
            return true;
        }
    }

    return false;
}

bool SceneBuilder::hasCornellBox(const std::vector<RenderObject>& objects)
{
    size_t count = 0;
    for (const RenderObject& object : objects) {
        if (object.sourceType == RenderObjectSourceType::CornellBox && object.mesh != nullptr) {
            ++count;
        }
    }
    return count >= static_cast<size_t>(kCornellBoxObjectCount);
}

bool SceneBuilder::hasOcclusionTest(const std::vector<RenderObject>& objects)
{
    size_t occlusionObjectCount = 0;
    bool hasGround = false;
    bool hasWall = false;
    bool hasHiddenObject = false;

    for (const RenderObject& object : objects) {
        if (object.sourceType != RenderObjectSourceType::OcclusionTest || !object.mesh ||
            !object.mesh->valid() || !object.material) {
            continue;
        }

        ++occlusionObjectCount;
        if (object.debugName == "Occlusion Test Ground") {
            hasGround = true;
        } else if (object.debugName.find("Occlusion Test Wall") == 0) {
            hasWall = true;
        } else if (object.debugName.find("Occlusion Test Hidden Cube") == 0) {
            hasHiddenObject = true;
        }
    }

    return hasGround && hasWall && hasHiddenObject &&
           occlusionObjectCount >= static_cast<size_t>(kOcclusionTestObjectCount);
}

} // namespace ve::renderer
