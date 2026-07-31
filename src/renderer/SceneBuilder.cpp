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
