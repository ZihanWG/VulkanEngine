// Scene contents: building the demo scenes, creating and editing materials, and
// the material-asset and scene JSON round trips.
//
// Split out of Renderer.cpp. Definitions only -- these remain Renderer member
// functions, so no call site changed.
#include "renderer/Renderer.h"
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>


namespace ve {

uint32_t Renderer::allocateRenderObjectDebugId()
{
    const uint32_t debugId = nextRenderObjectDebugId_;
    ++nextRenderObjectDebugId_;
    if (nextRenderObjectDebugId_ == 0) {
        nextRenderObjectDebugId_ = 1;
    }
    return debugId;
}

renderer::SceneBuilder Renderer::makeSceneBuilder()
{
    return renderer::SceneBuilder(
        cubeMesh_, portfolioSphereMesh_, materialVariants_, [this] { return allocateRenderObjectDebugId(); });
}

void Renderer::createScene()
{
    resetSceneState();
    createSceneSharedResources();

    // The portfolio sphere showcase is the default editor scene. The glTF import
    // (tryLoadGltfScene) and the cube fallback remain available through the
    // scene-loading UI; they are just no longer the startup default.
    makeSceneBuilder().appendPortfolioShowcase(renderObjects_);
    if (renderObjects_.empty()) {
        Logger::warn("Portfolio showcase scene unavailable; using built-in cube fallback scene.");
        makeSceneBuilder().appendCubeFallback(renderObjects_);
    }
}

void Renderer::resetSceneState()
{
    imguiLayer_.clearTexturePreviewDescriptors();
    renderObjects_.clear();
    selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
    nextRenderObjectDebugId_ = 1;
    occlusionTestSceneActive_ = false;
    occlusionTestSceneStatus_ = "Occlusion test scene not loaded.";
    allDrawItems_.clear();
    visibleDrawItems_.clear();
    shadowDrawItems_.clear();
    shadowMeshDrawBatches_.clear();
    gpuShadowMeshDrawBatches_.clear();
    for (std::vector<DrawItem>& cascadeDrawItems : shadowCascadeDrawItems_) {
        cascadeDrawItems.clear();
    }
    for (std::vector<MeshDrawBatch>& cascadeBatches : shadowCascadeMeshDrawBatches_) {
        cascadeBatches.clear();
    }
    shadowVisibleDrawItemsPerCascade_.fill(0);
    shadowBatchCountPerCascade_.fill(0);
    cullingStats_ = {};
    shadowCullingStats_ = {};
    importedMeshes_.clear();
    importedMaterials_.clear();
    importedBaseColorTextures_.clear();
    importedNormalTextures_.clear();
    importedMetallicRoughnessTextures_.clear();
    materialAssetTextures_.clear();
}

void Renderer::createSceneSharedResources()
{
    cubeMesh_ = renderer::Mesh::createCube(context_, commandContext_);
    portfolioSphereMesh_ = renderer::Mesh::createUvSphere(context_, commandContext_);
    const std::filesystem::path builtinAssetDir = assetDirectory();
    builtinTextureFactory_.createCheckerboardBaseColor(context_, commandContext_, builtinAssetDir, checkerboardTexture_);
    builtinTextureFactory_.createPortfolioBaseColor(context_, commandContext_, portfolioBaseColorTexture_);
    builtinTextureFactory_.createPortfolioBackdrop(context_, commandContext_, portfolioBackdropTexture_);
    builtinTextureFactory_.createCutoutLattice(context_, commandContext_, cutoutLatticeTexture_);
    builtinTextureFactory_.createNormal(
        context_, commandContext_, builtinAssetDir, normalMapTexture_, flatNormalTexture_, normalMapAssetLoaded_);
    builtinTextureFactory_.createMetallicRoughness(context_,
                                                   commandContext_,
                                                   builtinAssetDir,
                                                   metallicRoughnessTexture_,
                                                   neutralMetallicRoughnessTexture_,
                                                   metallicRoughnessMapAssetLoaded_);
    createEnvironmentMap();
    createMaterial();

    // Frame the portfolio sphere showcase, which is now the default editor scene.
    camera_ = portfolioCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;
}

bool Renderer::tryLoadGltfScene()
{
    const std::array<std::filesystem::path, 2> modelCandidates = {
        assetPath("models/test_mesh.gltf"),
        assetPath("models/test_mesh.glb"),
    };

    for (const std::filesystem::path& modelPath : modelCandidates) {
        if (!std::filesystem::exists(modelPath)) {
            continue;
        }

        try {
            renderer::LoadedGltfAsset loadedAsset =
                renderer::Mesh::createFromGltf(context_, commandContext_, modelPath);
            createImportedGltfTextures(loadedAsset.textures, loadedAsset.materials);
            createImportedGltfMaterials(loadedAsset.materials);
            importedMeshes_ = std::move(loadedAsset.meshes);

            renderObjects_.reserve(loadedAsset.nodeMeshInstances.size() + 8);
            for (const renderer::GltfNodeMeshInstance& instance : loadedAsset.nodeMeshInstances) {
                if (instance.meshIndex >= importedMeshes_.size() || !importedMeshes_[instance.meshIndex].valid()) {
                    Logger::warn("Skipping imported glTF RenderObject with invalid mesh index " +
                                 std::to_string(instance.meshIndex) + ".");
                    continue;
                }

                renderer::RenderObject importedObject{};
                importedObject.debugId = allocateRenderObjectDebugId();
                importedObject.sceneObjectId = importedObject.debugId;
                importedObject.mesh = &importedMeshes_[instance.meshIndex];
                importedObject.material =
                    importedMaterials_.empty() ? &materialVariants_.at(0) : &importedMaterials_.front();
                if (!importedMaterials_.empty()) {
                    importedObject.materialTable = importedMaterials_.data();
                    importedObject.materialCount = importedMaterials_.size();
                }
                importedObject.debugName = instance.debugName.empty() ? "Imported glTF Node" : instance.debugName;
                importedObject.sourceType = renderer::RenderObjectSourceType::ImportedGltf;
                importedObject.transform = renderer::Transform::fromMatrix(instance.transform);
                importedObject.hideInPortfolio = true;
                renderObjects_.push_back(std::move(importedObject));
            }

            if (renderObjects_.empty()) {
                throw std::runtime_error("Loaded glTF asset did not produce any valid RenderObjects.");
            }

            Logger::info("Loaded glTF scene: " + modelPath.string() + " with " +
                         std::to_string(importedMeshes_.size()) + " mesh slot(s), " +
                         std::to_string(renderObjects_.size()) + " render object(s), and " +
                         std::to_string(importedMaterials_.size()) + " material(s).");
            makeSceneBuilder().appendPortfolioShowcase(renderObjects_);
            return true;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load glTF mesh '" + modelPath.string() + "': " + error.what());
        }
    }

    return false;
}

void Renderer::resetPortfolioShowcaseObjectsToPreset()
{
    renderer::SceneBuilder::resetPortfolioShowcaseToPreset(renderObjects_);
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

bool Renderer::currentFrameHasPortfolioShowcaseDrawItems() const
{
    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex >= renderObjects_.size()) {
            continue;
        }

        const renderer::RenderObject& object = renderObjects_[drawItem.objectIndex];
        if (object.sourceType == renderer::RenderObjectSourceType::PortfolioShowcase && !object.hideInPortfolio) {
            return true;
        }
    }

    return false;
}

bool Renderer::ensurePortfolioShowcaseSceneReady()
{
    if (renderer::SceneBuilder::hasPortfolioShowcase(renderObjects_)) {
        return true;
    }

    if (!cubeMesh_.valid() || !portfolioSphereMesh_.valid() ||
        materialVariants_.size() <= renderer::kPortfolioBackdropMaterialIndex) {
        screenshotCapture_.setStatus(
            "Portfolio showcase scene is unavailable: required meshes or materials are not initialized.");
        Logger::warn(screenshotCapture_.status());
        return false;
    }

    Logger::warn("Portfolio showcase scene was missing; rebuilding portfolio-only showcase objects.");
    makeSceneBuilder().appendPortfolioShowcase(renderObjects_);
    if (renderer::SceneBuilder::hasPortfolioShowcase(renderObjects_)) {
        return true;
    }

    screenshotCapture_.setStatus("Portfolio showcase scene is unavailable after rebuild.");
    Logger::warn(screenshotCapture_.status());
    return false;
}

void Renderer::resetOcclusionTestSceneToPreset()
{
    const auto firstRemoved = std::remove_if(renderObjects_.begin(), renderObjects_.end(), [](const auto& object) {
        return object.sourceType == renderer::RenderObjectSourceType::OcclusionTest;
    });

    if (firstRemoved != renderObjects_.end()) {
        const size_t firstRemovedIndex = static_cast<size_t>(firstRemoved - renderObjects_.begin());
        if (selectedRenderObjectIndex_ >= firstRemovedIndex) {
            selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
        }
        renderObjects_.erase(firstRemoved, renderObjects_.end());
    }

    makeSceneBuilder().appendOcclusionTest(renderObjects_, occlusionTestSceneStatus_);
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

void Renderer::resetCornellBoxSceneToPreset()
{
    const auto firstRemoved = std::remove_if(renderObjects_.begin(), renderObjects_.end(), [](const auto& object) {
        return object.sourceType == renderer::RenderObjectSourceType::CornellBox;
    });

    if (firstRemoved != renderObjects_.end()) {
        const size_t firstRemovedIndex = static_cast<size_t>(firstRemoved - renderObjects_.begin());
        if (selectedRenderObjectIndex_ >= firstRemovedIndex) {
            selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
        }
        renderObjects_.erase(firstRemoved, renderObjects_.end());
    }

    makeSceneBuilder().appendCornellBox(renderObjects_, cornellBoxSceneStatus_);
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

const rhi::VulkanTexture* Renderer::loadMaterialAssetTextureOrFallback(
    const std::filesystem::path& materialPath,
    const std::filesystem::path& texturePath,
    rhi::TextureColorSpace colorSpace,
    std::string_view slotName,
    const rhi::VulkanTexture& fallbackTexture,
    bool& fallbackUsed)
{
    fallbackUsed = false;
    if (texturePath.empty()) {
        return &fallbackTexture;
    }

    const std::filesystem::path resolvedTexturePath = resolveMaterialTexturePath(materialPath, texturePath);
    if (!std::filesystem::exists(resolvedTexturePath)) {
        fallbackUsed = true;
        Logger::warn("Material asset texture is missing for " + std::string(slotName) +
                     "; using fallback texture: " + resolvedTexturePath.string());
        return &fallbackTexture;
    }

    auto texture = std::make_unique<rhi::VulkanTexture>();
    try {
        texture->createFromFile(context_, commandContext_, resolvedTexturePath, colorSpace, true);
        texture->setDebugMetadata(rhi::TextureDebugMetadata{
            "Material asset " + std::string(slotName) + " texture",
            resolvedTexturePath.string(),
            colorSpace,
            rhi::TextureDebugSource::LoadedFromDisk,
            false,
        });
        nameTextureResources(*texture, "MaterialAssetTexture_" + std::string(slotName));
        (void)assetManager_.registerTextureAsset(resolvedTexturePath, std::string(slotName));
        const rhi::VulkanTexture* texturePointer = texture.get();
        materialAssetTextures_.push_back(std::move(texture));
        Logger::info("Loaded material asset " + std::string(slotName) + " texture as " +
                     std::string(colorSpaceName(colorSpace)) + ": " + resolvedTexturePath.string());
        return texturePointer;
    } catch (const std::exception& error) {
        fallbackUsed = true;
        Logger::warn("Failed to load material asset " + std::string(slotName) + " texture '" +
                     resolvedTexturePath.string() + "'; using fallback texture: " + error.what());
        return &fallbackTexture;
    }
}

renderer::Material Renderer::createMaterialFromAsset(const assets::MaterialAsset& materialAsset,
                                                     const rhi::VulkanTexture& baseColorFallback,
                                                     const rhi::VulkanTexture& normalFallback,
                                                     const rhi::VulkanTexture& metallicRoughnessFallback,
                                                     float multiScatterStrength,
                                                     renderer::MaterialSource fallbackSource)
{
    bool baseColorLoadFallback = false;
    bool normalLoadFallback = false;
    bool metallicRoughnessLoadFallback = false;

    renderer::Material material{};
    material.debugName = materialAsset.name.empty() ? "Material Asset" : materialAsset.name;
    material.assetName = materialAsset.name;
    material.sourceAssetPath = materialAsset.sourcePath;
    material.shader = materialAsset.shader.empty() ? "pbr_opaque" : materialAsset.shader;
    material.baseColorTexturePath = materialAsset.textures.baseColor;
    material.normalTexturePath = materialAsset.textures.normal;
    material.metallicRoughnessTexturePath = materialAsset.textures.metallicRoughness;
    material.alphaMode = materialAsset.alphaMode.empty() ? "OPAQUE" : materialAsset.alphaMode;
    material.baseColorTexture = loadMaterialAssetTextureOrFallback(materialAsset.sourcePath,
                                                                   materialAsset.textures.baseColor,
                                                                   rhi::TextureColorSpace::SRGB,
                                                                   "base color",
                                                                   baseColorFallback,
                                                                   baseColorLoadFallback);
    material.normalTexture = loadMaterialAssetTextureOrFallback(materialAsset.sourcePath,
                                                                materialAsset.textures.normal,
                                                                rhi::TextureColorSpace::Linear,
                                                                "normal",
                                                                normalFallback,
                                                                normalLoadFallback);
    material.metallicRoughnessTexture =
        loadMaterialAssetTextureOrFallback(materialAsset.sourcePath,
                                           materialAsset.textures.metallicRoughness,
                                           rhi::TextureColorSpace::Linear,
                                           "metallic-roughness",
                                           metallicRoughnessFallback,
                                           metallicRoughnessLoadFallback);
    material.baseColorFactor = materialAsset.baseColorFactor;
    material.emissiveFactor = glm::max(materialAsset.emissiveFactor, glm::vec3(0.0f));
    material.metallic = std::clamp(materialAsset.metallicFactor, 0.0f, 1.0f);
    material.roughness = std::clamp(materialAsset.roughnessFactor, 0.0f, 1.0f);
    material.multiScatterStrength = multiScatterStrength;
    material.alphaCutoff = std::max(materialAsset.alphaCutoff, 0.0f);
    material.doubleSided = materialAsset.doubleSided;
    material.source = materialAsset.fallback ? fallbackSource : renderer::MaterialSource::MaterialAsset;
    material.baseColorTextureFallback =
        baseColorLoadFallback || (material.baseColorTexture && material.baseColorTexture->debugMetadata().fallback);
    material.normalTextureFallback =
        normalLoadFallback || (material.normalTexture && material.normalTexture->debugMetadata().fallback);
    material.metallicRoughnessTextureFallback =
        metallicRoughnessLoadFallback ||
        (material.metallicRoughnessTexture && material.metallicRoughnessTexture->debugMetadata().fallback);
    material.hasNormalMap = !material.normalTextureFallback;
    material.hasMetallicRoughnessMap = !material.metallicRoughnessTextureFallback;

    assignBindlessTextureIndices(material);
    createMaterialDescriptorSet(material);
    return material;
}

assets::MaterialAsset Renderer::runtimeMaterialToAsset(const renderer::Material& material) const
{
    assets::MaterialAsset materialAsset{};
    materialAsset.sourcePath = material.sourceAssetPath;
    materialAsset.name = !material.assetName.empty() ? material.assetName : material.debugName;
    if (materialAsset.name.empty()) {
        materialAsset.name = "Material";
    }
    materialAsset.shader = material.shader.empty() ? "pbr_opaque" : material.shader;
    materialAsset.baseColorFactor = material.baseColorFactor;
    materialAsset.emissiveFactor = material.emissiveFactor;
    materialAsset.metallicFactor = material.metallic;
    materialAsset.roughnessFactor = material.roughness;
    materialAsset.textures.baseColor = material.baseColorTexturePath;
    materialAsset.textures.normal = material.normalTexturePath;
    materialAsset.textures.metallicRoughness = material.metallicRoughnessTexturePath;
    materialAsset.alphaMode = material.alphaMode.empty() ? "OPAQUE" : material.alphaMode;
    materialAsset.alphaCutoff = material.alphaCutoff;
    materialAsset.doubleSided = material.doubleSided;
    materialAsset.fallback = false;
    return materialAsset;
}

namespace {

// Turns a free-form material name into a filesystem-safe slug for a new asset
// file (lowercase, alphanumerics preserved, runs of other characters collapsed
// to single underscores).
std::string slugifyMaterialName(std::string_view name)
{
    std::string slug;
    slug.reserve(name.size());
    bool pendingSeparator = false;
    for (const char ch : name) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch)) {
            if (pendingSeparator && !slug.empty()) {
                slug.push_back('_');
            }
            pendingSeparator = false;
            slug.push_back(static_cast<char>(std::tolower(uch)));
        } else {
            pendingSeparator = true;
        }
    }
    if (slug.empty()) {
        slug = "material";
    }
    return slug;
}

} // namespace

std::filesystem::path Renderer::makeNewMaterialAssetPath(const renderer::Material& material) const
{
    std::string_view name = !material.assetName.empty() ? std::string_view(material.assetName)
                                                        : std::string_view(material.debugName);
    const std::string slug = slugifyMaterialName(name);

    // Avoid clobbering an existing file on disk by appending a numeric suffix.
    std::filesystem::path candidate = materialAssetPath(slug + ".material.json");
    for (int index = 2; std::filesystem::exists(candidate); ++index) {
        candidate = materialAssetPath(slug + "_" + std::to_string(index) + ".material.json");
    }
    return candidate;
}

bool Renderer::saveMaterialAssetFromUi(renderer::Material& material)
{
    if (material.sourceAssetPath.empty()) {
        // "Save As": this material was created/edited in the UI without a backing
        // file (e.g. a glTF or procedural material). Synthesize a new asset path
        // under assets/materials/ so it can be persisted and reloaded later.
        material.sourceAssetPath = makeNewMaterialAssetPath(material);
    }

    std::string errorMessage;
    assets::MaterialAsset materialAsset = runtimeMaterialToAsset(material);
    if (!assetManager_.saveMaterialAsset(material.sourceAssetPath, materialAsset, &errorMessage)) {
        lastMaterialAssetStatus_ = errorMessage;
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    material.source = renderer::MaterialSource::MaterialAsset;
    material.assetName = materialAsset.name;
    lastMaterialAssetStatus_ = "Saved material asset: " + material.sourceAssetPath.string();
    Logger::info(lastMaterialAssetStatus_);
    invalidateTaaHistory();
    return true;
}

bool Renderer::reloadMaterialAssetFromUi(renderer::Material& material)
{
    if (material.sourceAssetPath.empty()) {
        lastMaterialAssetStatus_ = "Reload Material skipped: selected material has no source asset path.";
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    std::string errorMessage;
    const assets::MaterialAssetHandle handle = assetManager_.loadMaterialAsset(material.sourceAssetPath, &errorMessage);
    if (!handle) {
        lastMaterialAssetStatus_ = errorMessage;
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    const assets::MaterialAsset* materialAsset = assetManager_.materialAsset(handle);
    if (!materialAsset) {
        lastMaterialAssetStatus_ = "Reload Material failed: loaded material asset was not registered.";
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    material.debugName = materialAsset->name.empty() ? material.debugName : materialAsset->name;
    material.assetName = materialAsset->name;
    material.shader = materialAsset->shader.empty() ? "pbr_opaque" : materialAsset->shader;
    material.baseColorFactor = materialAsset->baseColorFactor;
    material.emissiveFactor = glm::max(materialAsset->emissiveFactor, glm::vec3(0.0f));
    material.metallic = std::clamp(materialAsset->metallicFactor, 0.0f, 1.0f);
    material.roughness = std::clamp(materialAsset->roughnessFactor, 0.0f, 1.0f);
    material.baseColorTexturePath = materialAsset->textures.baseColor;
    material.normalTexturePath = materialAsset->textures.normal;
    material.metallicRoughnessTexturePath = materialAsset->textures.metallicRoughness;
    material.alphaMode = materialAsset->alphaMode.empty() ? "OPAQUE" : materialAsset->alphaMode;
    material.alphaCutoff = std::max(materialAsset->alphaCutoff, 0.0f);
    material.doubleSided = materialAsset->doubleSided;
    material.source = renderer::MaterialSource::MaterialAsset;

    lastMaterialAssetStatus_ =
        "Reloaded material scalar/metadata fields from " + material.sourceAssetPath.string() +
        ". Texture rebinding is not hot-reloaded in Phase 3.";
    Logger::info(lastMaterialAssetStatus_);
    invalidateTaaHistory();
    return true;
}

void Renderer::createMaterial()
{
    materialVariants_.clear();
    materialVariants_.reserve(11);

    if (isBindlessMaterialTextureActive()) {
        bindlessBaseColorFallbackIndex_ = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::BaseColor, checkerboardTexture_);
        bindlessNormalFallbackIndex_ = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::Normal, flatNormalTexture_);
        bindlessMetallicRoughnessFallbackIndex_ = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::MetallicRoughness, neutralMetallicRoughnessTexture_);
    }

    createBuiltInMaterialVariants();
    createPortfolioMaterialVariants();

    checkerboardMaterial_ = materialVariants_.front();
}

void Renderer::createBuiltInMaterialVariants()
{
    const auto addMaterial = [this](std::string debugName,
                                    const glm::vec4& baseColorFactor,
                                    float metallic,
                                    float roughness,
                                    float multiScatterStrength) {
        renderer::Material material{};
        material.debugName = std::move(debugName);
        material.baseColorTexture = &checkerboardTexture_;
        material.normalTexture = &normalMapTexture_;
        material.metallicRoughnessTexture = &metallicRoughnessTexture_;
        material.baseColorFactor = baseColorFactor;
        material.metallic = metallic;
        material.roughness = roughness;
        material.multiScatterStrength = multiScatterStrength;
        material.source = renderer::MaterialSource::BuiltIn;
        material.hasNormalMap = normalMapAssetLoaded_;
        material.hasMetallicRoughnessMap = metallicRoughnessMapAssetLoaded_;
        material.baseColorTextureFallback = checkerboardTexture_.debugMetadata().fallback;
        material.normalTextureFallback = !normalMapAssetLoaded_;
        material.metallicRoughnessTextureFallback = !metallicRoughnessMapAssetLoaded_;
        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        materialVariants_.push_back(std::move(material));
    };

    addMaterial("Checkerboard Matte", {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.75f, 0.0f);
    addMaterial("Checkerboard Warm Semi-Metal", {1.0f, 0.82f, 0.65f, 1.0f}, 0.35f, 0.38f, 0.5f);
    addMaterial("Checkerboard Cool Rough Metal", {0.72f, 0.84f, 1.0f, 1.0f}, 0.85f, 0.62f, 1.0f);
    addMaterial("Checkerboard Glossy Dielectric", {0.9f, 1.0f, 0.78f, 1.0f}, 0.0f, 0.18f, 0.25f);
}

void Renderer::createPortfolioMaterialVariants()
{
    const auto addPortfolioMaterial = [this](std::string debugName,
                                             const rhi::VulkanTexture* baseColorTexture,
                                             const glm::vec4& baseColorFactor,
                                             float metallic,
                                             float roughness,
                                             float multiScatterStrength) {
        renderer::Material material{};
        material.debugName = std::move(debugName);
        material.assetName = material.debugName;
        material.shader = "pbr_opaque";
        material.alphaMode = "OPAQUE";
        material.baseColorTexture = baseColorTexture;
        material.normalTexture = &flatNormalTexture_;
        material.metallicRoughnessTexture = &neutralMetallicRoughnessTexture_;
        material.baseColorFactor = baseColorFactor;
        material.metallic = metallic;
        material.roughness = roughness;
        material.multiScatterStrength = multiScatterStrength;
        material.alphaCutoff = 0.5f;
        material.source = renderer::MaterialSource::BuiltIn;
        material.hasNormalMap = false;
        material.hasMetallicRoughnessMap = false;
        material.doubleSided = false;
        material.baseColorTextureFallback = false;
        material.normalTextureFallback = true;
        material.metallicRoughnessTextureFallback = true;
        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        materialVariants_.push_back(std::move(material));
    };

    const auto portfolioMaterialAssetOrFallback =
        [this](std::string_view filename,
               std::string debugName,
               const glm::vec4& baseColorFactor,
               float metallic,
               float roughness) {
            const std::filesystem::path path = materialAssetPath(filename);
            std::string errorMessage;
            const assets::MaterialAssetHandle handle = assetManager_.loadMaterialAsset(path, &errorMessage);
            if (handle) {
                const assets::MaterialAsset* materialAsset = assetManager_.materialAsset(handle);
                if (materialAsset) {
                    Logger::info("Loaded portfolio material asset: " + path.string());
                    return *materialAsset;
                }
            }

            Logger::warn(errorMessage.empty() ? "Portfolio material asset failed to load; using fallback values: " +
                                                    path.string()
                                              : errorMessage + "; using fallback values.");
            assets::MaterialAsset fallback = assets::AssetManager::fallbackMaterialAsset(std::move(debugName));
            fallback.sourcePath = path;
            fallback.baseColorFactor = baseColorFactor;
            fallback.metallicFactor = metallic;
            fallback.roughnessFactor = roughness;
            return fallback;
        };

    const auto addPortfolioMaterialAsset =
        [this, &portfolioMaterialAssetOrFallback](std::string_view filename,
                                                  std::string debugName,
                                                  const glm::vec4& baseColorFactor,
                                                  float metallic,
                                                  float roughness,
                                                  float multiScatterStrength) {
            const assets::MaterialAsset materialAsset =
                portfolioMaterialAssetOrFallback(filename, debugName, baseColorFactor, metallic, roughness);
            renderer::Material material = createMaterialFromAsset(materialAsset,
                                                                  portfolioBaseColorTexture_,
                                                                  flatNormalTexture_,
                                                                  neutralMetallicRoughnessTexture_,
                                                                  multiScatterStrength,
                                                                  renderer::MaterialSource::Fallback);
            materialVariants_.push_back(std::move(material));
        };

    addPortfolioMaterial(
        "Portfolio_Ground", &portfolioBaseColorTexture_, {0.30f, 0.32f, 0.32f, 1.0f}, 0.0f, 0.86f, 0.0f);
    addPortfolioMaterialAsset("portfolio_matte_gray.material.json",
                              "Portfolio_MatteGray",
                              {0.66f, 0.66f, 0.62f, 1.0f},
                              0.0f,
                              0.85f,
                              0.0f);
    addPortfolioMaterialAsset("portfolio_glossy_blue.material.json",
                              "Portfolio_GlossyBlue",
                              {0.18f, 0.43f, 0.88f, 1.0f},
                              0.0f,
                              0.30f,
                              0.2f);
    addPortfolioMaterialAsset("portfolio_rough_metal.material.json",
                              "Portfolio_RoughMetal",
                              {0.76f, 0.74f, 0.70f, 1.0f},
                              1.0f,
                              0.60f,
                              0.70f);
    addPortfolioMaterialAsset("portfolio_polished_metal_small.material.json",
                              "Portfolio_PolishedMetalSmall",
                              {0.82f, 0.85f, 0.88f, 1.0f},
                              1.0f,
                              0.23f,
                              0.40f);
    addPortfolioMaterialAsset("portfolio_hero_ceramic.material.json",
                              "Portfolio_HeroCeramic",
                              {0.66f, 0.72f, 0.76f, 1.0f},
                              0.0f,
                              0.55f,
                              0.05f);
    addPortfolioMaterial(
        "Portfolio_Backdrop", &portfolioBackdropTexture_, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.94f, 0.0f);
    addPortfolioMaterial(
        "Portfolio_CutoutLattice", &cutoutLatticeTexture_, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.58f, 0.0f);
    // baseColorFactor.a is the glass opacity; the blend pipeline reads it straight
    // out of the fragment shader's existing alpha output.
    addPortfolioMaterial(
        "Portfolio_Glass", &portfolioBaseColorTexture_, {0.42f, 0.68f, 0.78f, 0.38f}, 0.0f, 0.08f, 0.0f);

    // Cornell box surfaces. Matte and saturated: colour bleeding scales with how
    // saturated the bouncing surface is, and specular would not survive the probe
    // convolution anyway. The white is deliberately not 1.0 -- a perfectly white
    // room is exactly the case where the bounce feedback series stops converging.
    //
    // portfolioBaseColorTexture_ is already a flat white 4x4, so the surface
    // colour here is exactly the factor with nothing modulating it.
    addPortfolioMaterial("Cornell_White", &portfolioBaseColorTexture_, {0.72f, 0.71f, 0.68f, 1.0f}, 0.0f, 0.95f, 0.0f);
    addPortfolioMaterial("Cornell_Red", &portfolioBaseColorTexture_, {0.63f, 0.07f, 0.05f, 1.0f}, 0.0f, 0.95f, 0.0f);
    addPortfolioMaterial("Cornell_Green", &portfolioBaseColorTexture_, {0.09f, 0.48f, 0.10f, 1.0f}, 0.0f, 0.95f, 0.0f);

    // Give the hero ceramic a soft warm emissive so factor-only emissive is
    // visible in the default scene and reads through bloom. Editable per material
    // from the Material Inspector.
    if (renderer::kPortfolioHeroCeramicMaterialIndex < materialVariants_.size()) {
        materialVariants_[renderer::kPortfolioHeroCeramicMaterialIndex].emissiveFactor = glm::vec3(0.9f, 0.45f, 0.15f);
    }

    // addPortfolioMaterial builds opaque materials; promote this one to glTF MASK
    // so it lands in the Mask render bucket and clips against the lattice alpha.
    // Double-sided because a perforated panel is visible through its own holes.
    if (renderer::kPortfolioCutoutLatticeMaterialIndex < materialVariants_.size()) {
        renderer::Material& cutoutMaterial = materialVariants_[renderer::kPortfolioCutoutLatticeMaterialIndex];
        cutoutMaterial.alphaMode = "MASK";
        cutoutMaterial.alphaCutoff = 0.5f;
        cutoutMaterial.doubleSided = true;
    }

    // Same promotion for the glass: addPortfolioMaterial builds opaque materials,
    // so the BLEND mode is applied afterwards.
    if (renderer::kPortfolioGlassMaterialIndex < materialVariants_.size()) {
        renderer::Material& glassMaterial = materialVariants_[renderer::kPortfolioGlassMaterialIndex];
        glassMaterial.alphaMode = "BLEND";
        glassMaterial.doubleSided = true;
    }
}

void Renderer::assignBindlessTextureIndices(renderer::Material& material)
{
    if (!isBindlessMaterialTextureActive()) {
        return;
    }

    material.baseColorTextureIndex =
        material.baseColorTexture && material.baseColorTexture->valid()
            ? bindlessTextureHeap_.registerTexture(renderer::BindlessTextureHeap::TextureKind::BaseColor,
                                                   *material.baseColorTexture)
            : bindlessBaseColorFallbackIndex_;
    material.normalTextureIndex = material.normalTexture && material.normalTexture->valid()
                                      ? bindlessTextureHeap_.registerTexture(
                                            renderer::BindlessTextureHeap::TextureKind::Normal, *material.normalTexture)
                                      : bindlessNormalFallbackIndex_;
    material.metallicRoughnessTextureIndex =
        material.metallicRoughnessTexture && material.metallicRoughnessTexture->valid()
            ? bindlessTextureHeap_.registerTexture(renderer::BindlessTextureHeap::TextureKind::MetallicRoughness,
                                                   *material.metallicRoughnessTexture)
            : bindlessMetallicRoughnessFallbackIndex_;
    // Emissive maps are sRGB color, so they share the base-color bindless array.
    // Without one, the index falls back and the shader skips sampling it
    // (hasEmissiveTexture stays false, so emissive uses the factor only).
    if (material.hasEmissiveTexture && material.emissiveTexture && material.emissiveTexture->valid()) {
        material.emissiveTextureIndex = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::BaseColor, *material.emissiveTexture);
    } else {
        material.emissiveTextureIndex = bindlessBaseColorFallbackIndex_;
        material.hasEmissiveTexture = false;
    }
}

void Renderer::createImportedGltfTextures(const std::vector<renderer::GltfTextureInfo>& textureInfos,
                                          const std::vector<renderer::GltfMaterialInfo>& materialInfos)
{
    importedBaseColorTextures_.clear();
    importedNormalTextures_.clear();
    importedMetallicRoughnessTextures_.clear();
    importedBaseColorTextures_.resize(textureInfos.size());
    importedNormalTextures_.resize(textureInfos.size());
    importedMetallicRoughnessTextures_.resize(textureInfos.size());

    std::vector<uint8_t> baseColorNeeded(textureInfos.size(), 0);
    std::vector<uint8_t> normalNeeded(textureInfos.size(), 0);
    std::vector<uint8_t> metallicRoughnessNeeded(textureInfos.size(), 0);

    const auto markNeeded = [textureCount = textureInfos.size()](int textureIndex, std::vector<uint8_t>& needed) {
        if (textureIndex >= 0 && static_cast<size_t>(textureIndex) < textureCount) {
            needed[static_cast<size_t>(textureIndex)] = 1;
        }
    };

    for (const renderer::GltfMaterialInfo& materialInfo : materialInfos) {
        markNeeded(materialInfo.baseColorTextureIndex, baseColorNeeded);
        markNeeded(materialInfo.normalTextureIndex, normalNeeded);
        markNeeded(materialInfo.metallicRoughnessTextureIndex, metallicRoughnessNeeded);
        // Emissive maps decode as sRGB into the base-color texture array.
        markNeeded(materialInfo.emissiveTextureIndex, baseColorNeeded);
    }

    // Decode the needed textures on worker threads, then upload them here on the
    // device-owning thread. PNG/JPEG decoding is the expensive part and is safe to
    // parallelize; the Vulkan uploads stay serial on this thread.
    struct PendingTextureUpload {
        size_t textureIndex = 0;
        rhi::TextureColorSpace colorSpace = rhi::TextureColorSpace::Linear;
        std::string_view slotName;
        std::string debugPrefix;
        std::vector<rhi::VulkanTexture>* textures = nullptr;
        const renderer::GltfTextureInfo* info = nullptr;
        std::future<rhi::DecodedImage> decode;
    };

    std::vector<PendingTextureUpload> pendingUploads;

    const auto enqueueDecode = [this, &textureInfos, &pendingUploads](size_t textureIndex,
                                                                      rhi::TextureColorSpace colorSpace,
                                                                      std::string_view slotName,
                                                                      std::string_view debugPrefix,
                                                                      std::vector<rhi::VulkanTexture>& textures) {
        const renderer::GltfTextureInfo& textureInfo = textureInfos[textureIndex];
        if (textureInfo.path.empty() && textureInfo.encodedData.empty()) {
            return;
        }
        if (!textureInfo.path.empty() && !std::filesystem::exists(textureInfo.path)) {
            Logger::warn("glTF texture image is missing; material fallback will be used: " +
                         textureInfo.path.string());
            return;
        }

        const renderer::GltfTextureInfo* infoPtr = &textureInfo;
        std::future<rhi::DecodedImage> decode = jobSystem_.enqueue([infoPtr]() -> rhi::DecodedImage {
            if (!infoPtr->path.empty()) {
                return rhi::VulkanTexture::decodeImageFile(infoPtr->path);
            }
            return rhi::VulkanTexture::decodeImageBytes(
                std::span<const uint8_t>(infoPtr->encodedData.data(), infoPtr->encodedData.size()));
        });

        pendingUploads.push_back(PendingTextureUpload{
            textureIndex, colorSpace, slotName, std::string(debugPrefix), &textures, infoPtr, std::move(decode)});
    };

    for (size_t textureIndex = 0; textureIndex < textureInfos.size(); ++textureIndex) {
        if (baseColorNeeded[textureIndex] != 0) {
            enqueueDecode(textureIndex,
                          rhi::TextureColorSpace::SRGB,
                          "base color",
                          "GltfBaseColorTexture",
                          importedBaseColorTextures_);
        }
        if (normalNeeded[textureIndex] != 0) {
            enqueueDecode(
                textureIndex, rhi::TextureColorSpace::Linear, "normal", "GltfNormalTexture", importedNormalTextures_);
        }
        if (metallicRoughnessNeeded[textureIndex] != 0) {
            enqueueDecode(textureIndex,
                          rhi::TextureColorSpace::Linear,
                          "metallic-roughness",
                          "GltfMetallicRoughnessTexture",
                          importedMetallicRoughnessTextures_);
        }
    }

    for (PendingTextureUpload& pending : pendingUploads) {
        std::vector<rhi::VulkanTexture>& textures = *pending.textures;
        const renderer::GltfTextureInfo& textureInfo = *pending.info;
        try {
            const rhi::DecodedImage decoded = pending.decode.get();
            textures[pending.textureIndex].createFromRgba8(context_,
                                                           commandContext_,
                                                           decoded.width,
                                                           decoded.height,
                                                           decoded.pixels,
                                                           rhi::rgba8FormatForColorSpace(pending.colorSpace),
                                                           true);
            if (!textureInfo.path.empty()) {
                Logger::info("Loaded glTF " + std::string(pending.slotName) + " texture as " +
                             std::string(colorSpaceName(pending.colorSpace)) + ": " + textureInfo.path.string());
            } else {
                Logger::info("Loaded embedded glTF " + std::string(pending.slotName) + " texture as " +
                             std::string(colorSpaceName(pending.colorSpace)) + ": " + textureInfo.debugName);
            }

            textures[pending.textureIndex].setDebugMetadata(rhi::TextureDebugMetadata{
                textureInfo.debugName.empty() ? pending.debugPrefix + std::to_string(pending.textureIndex)
                                              : textureInfo.debugName,
                textureInfo.path.empty() ? std::string{} : textureInfo.path.string(),
                pending.colorSpace,
                textureInfo.embedded ? rhi::TextureDebugSource::GltfEmbeddedData
                                     : rhi::TextureDebugSource::GltfExternalFile,
                false,
            });
            nameTextureResources(textures[pending.textureIndex],
                                 pending.debugPrefix + std::to_string(pending.textureIndex));
        } catch (const std::exception& error) {
            const std::string textureName =
                !textureInfo.path.empty() ? textureInfo.path.string() : textureInfo.debugName;
            Logger::warn("Failed to load glTF " + std::string(pending.slotName) + " texture '" + textureName +
                         "'; material fallback will be used: " + error.what());
        }
    }
}

void Renderer::createImportedGltfMaterials(const std::vector<renderer::GltfMaterialInfo>& materialInfos)
{
    std::vector<renderer::GltfMaterialInfo> defaultMaterialInfos;
    const std::vector<renderer::GltfMaterialInfo>* sourceMaterialInfos = &materialInfos;
    if (materialInfos.empty()) {
        renderer::GltfMaterialInfo defaultMaterial{};
        defaultMaterial.debugName = "Default glTF Material";
        defaultMaterial.fallback = true;
        defaultMaterialInfos.push_back(std::move(defaultMaterial));
        sourceMaterialInfos = &defaultMaterialInfos;
    }

    importedMaterials_.clear();
    importedMaterials_.reserve(sourceMaterialInfos->size());

    const auto textureOrFallback = [](int textureIndex,
                                      const std::vector<rhi::VulkanTexture>& textures,
                                      const rhi::VulkanTexture& fallbackTexture) -> const rhi::VulkanTexture* {
        if (textureIndex >= 0 && static_cast<size_t>(textureIndex) < textures.size() &&
            textures[static_cast<size_t>(textureIndex)].valid()) {
            return &textures[static_cast<size_t>(textureIndex)];
        }
        return &fallbackTexture;
    };

    const auto textureLoaded = [](int textureIndex, const std::vector<rhi::VulkanTexture>& textures) {
        return textureIndex >= 0 && static_cast<size_t>(textureIndex) < textures.size() &&
               textures[static_cast<size_t>(textureIndex)].valid();
    };

    for (const renderer::GltfMaterialInfo& materialInfo : *sourceMaterialInfos) {
        renderer::Material material{};
        material.debugName = materialInfo.debugName.empty() ? "glTF Material" : materialInfo.debugName;
        material.baseColorTexture =
            textureOrFallback(materialInfo.baseColorTextureIndex, importedBaseColorTextures_, checkerboardTexture_);
        material.normalTexture =
            textureOrFallback(materialInfo.normalTextureIndex, importedNormalTextures_, flatNormalTexture_);
        material.metallicRoughnessTexture = textureOrFallback(materialInfo.metallicRoughnessTextureIndex,
                                                              importedMetallicRoughnessTextures_,
                                                              neutralMetallicRoughnessTexture_);
        material.baseColorFactor = materialInfo.baseColorFactor;
        material.emissiveFactor = materialInfo.emissiveFactor;
        material.hasEmissiveTexture = textureLoaded(materialInfo.emissiveTextureIndex, importedBaseColorTextures_);
        material.emissiveTexture = material.hasEmissiveTexture
                                       ? &importedBaseColorTextures_[static_cast<size_t>(materialInfo.emissiveTextureIndex)]
                                       : nullptr;
        material.metallic = materialInfo.metallic;
        material.roughness = materialInfo.roughness;
        material.multiScatterStrength = 1.0f;
        material.alphaMode = materialInfo.alphaMode;
        material.alphaCutoff = std::max(materialInfo.alphaCutoff, 0.0f);
        material.doubleSided = materialInfo.doubleSided;
        material.source =
            materialInfo.fallback ? renderer::MaterialSource::Fallback : renderer::MaterialSource::Gltf;
        material.hasNormalMap = textureLoaded(materialInfo.normalTextureIndex, importedNormalTextures_);
        material.hasMetallicRoughnessMap =
            textureLoaded(materialInfo.metallicRoughnessTextureIndex, importedMetallicRoughnessTextures_);
        material.baseColorTextureFallback = !textureLoaded(materialInfo.baseColorTextureIndex, importedBaseColorTextures_);
        material.normalTextureFallback = !material.hasNormalMap;
        material.metallicRoughnessTextureFallback = !material.hasMetallicRoughnessMap;

        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        importedMaterials_.push_back(std::move(material));
    }
}

std::vector<const renderer::Material*> Renderer::materialsForObject(const renderer::RenderObject& object) const
{
    std::vector<const renderer::Material*> materials;
    if (object.mesh && object.mesh->hasSubMeshes()) {
        const std::span<const renderer::MeshPrimitive> primitives = object.mesh->primitives();
        materials.reserve(primitives.size());
        for (const renderer::MeshPrimitive& primitive : primitives) {
            const renderer::Material* material = resolveMaterial(object, &primitive);
            if (!material) {
                continue;
            }
            if (std::find(materials.begin(), materials.end(), material) == materials.end()) {
                materials.push_back(material);
            }
        }
    }

    if (materials.empty() && object.material) {
        materials.push_back(object.material);
    }

    return materials;
}

const renderer::Material* Renderer::primaryMaterialForObject(const renderer::RenderObject& object) const
{
    const std::vector<const renderer::Material*> materials = materialsForObject(object);
    return materials.empty() ? nullptr : materials.front();
}

renderer::Material* Renderer::mutableMaterialFromPointer(const renderer::Material* material)
{
    if (!material) {
        return nullptr;
    }

    for (renderer::Material& candidate : materialVariants_) {
        if (&candidate == material) {
            return &candidate;
        }
    }
    for (renderer::Material& candidate : importedMaterials_) {
        if (&candidate == material) {
            return &candidate;
        }
    }
    if (&checkerboardMaterial_ == material) {
        return &checkerboardMaterial_;
    }

    return nullptr;
}

renderer::Material* Renderer::primaryMutableMaterialForObject(renderer::RenderObject& object)
{
    return mutableMaterialFromPointer(primaryMaterialForObject(object));
}

renderer::Material* Renderer::findRuntimeMaterialByAssetPath(const std::filesystem::path& path)
{
    if (path.empty()) {
        return nullptr;
    }

    const std::string key = path.lexically_normal().generic_string();
    const auto matchesPath = [&key](const renderer::Material& material) {
        return !material.sourceAssetPath.empty() && material.sourceAssetPath.lexically_normal().generic_string() == key;
    };

    for (renderer::Material& material : materialVariants_) {
        if (matchesPath(material)) {
            return &material;
        }
    }
    for (renderer::Material& material : importedMaterials_) {
        if (matchesPath(material)) {
            return &material;
        }
    }
    if (matchesPath(checkerboardMaterial_)) {
        return &checkerboardMaterial_;
    }

    return nullptr;
}

void Renderer::loadOcclusionTestScene()
{
    if (portfolioCaptureMode_) {
        setPortfolioCaptureMode(false);
    }

    resetOcclusionTestSceneToPreset();
    if (!renderer::SceneBuilder::hasOcclusionTest(renderObjects_)) {
        occlusionTestSceneActive_ = false;
        return;
    }

    occlusionTestSceneActive_ = true;
    resetCameraToOcclusionTestPreset();
    resetDirectionalLightToDefault();
    debugUiSettings_.showCullingStats = true;
    debugUiSettings_.showGpuTimingGraphs = true;
    debugUiSettings_.showRenderGraphPanel = true;
    occlusionTestSceneStatus_ = "Occlusion test scene active: " +
                                std::to_string(renderer::kOcclusionTestObjectCount) +
                                " procedural cube objects, including 5 occluder walls and 120 hidden/edge cubes.";
    Logger::info(occlusionTestSceneStatus_);
}

void Renderer::loadCornellBoxScene()
{
    if (portfolioCaptureMode_) {
        setPortfolioCaptureMode(false);
    }
    occlusionTestSceneActive_ = false;

    resetCornellBoxSceneToPreset();
    if (!renderer::SceneBuilder::hasCornellBox(renderObjects_)) {
        cornellBoxSceneActive_ = false;
        cornellBoxSceneStatus_ = "Cornell box objects are unavailable; see the startup log.";
        Logger::warn(cornellBoxSceneStatus_);
        return;
    }

    cornellBoxSceneActive_ = true;

    // Looking in through the open side. Far enough back that both blocks and
    // both coloured walls are in frame, which is what makes the bleed readable.
    camera_.position = {0.0f, 5.0f, 17.5f};
    camera_.target = {0.0f, 4.6f, 0.0f};
    camera_.up = {0.0f, 1.0f, 0.0f};
    editorCamera_.syncFromCamera(camera_);

    // The sun is switched off rather than dimmed. It would flood the room
    // through the open side and swamp the one thing this scene exists to show.
    directionalLightSettings_.color = glm::vec3{0.0f};
    directionalLightSettings_.intensity = 0.0f;

    // Fit the probe grid to the room's interior, inset so no probe lands inside
    // a wall -- a probe in solid geometry sees nothing but that geometry and
    // contributes it to everything nearby.
    constexpr float kHalf = renderer::kCornellBoxHalfExtent;
    constexpr float kInset = 0.8f;
    const float spanXZ = (kHalf - kInset) * 2.0f;
    const float spanY = kHalf * 2.0f - kInset * 2.0f;
    giSettings_.gridOrigin[0] = -(kHalf - kInset);
    giSettings_.gridOrigin[1] = kInset;
    giSettings_.gridOrigin[2] = -(kHalf - kInset);
    giSettings_.gridSpacing[0] = spanXZ / static_cast<float>(renderer::kProbeGridX - 1);
    giSettings_.gridSpacing[1] = spanY / static_cast<float>(renderer::kProbeGridY - 1);
    giSettings_.gridSpacing[2] = spanXZ / static_cast<float>(renderer::kProbeGridZ - 1);

    // The whole point of the scene, so it arrives switched on.
    giSettings_.enabled = true;
    giSettings_.debugPattern = false;
    debugUiSettings_.showIrradianceProbePanel = true;
    clampRuntimeSettings();

    cornellBoxSceneStatus_ =
        "Cornell box active: closed room, one overhead light, sun disabled, probe grid fitted to the interior.";
    Logger::info(cornellBoxSceneStatus_);
}

void Renderer::saveSceneFromUi()
{
    try {
        const std::filesystem::path parentPath = sceneDocumentPath_.parent_path();
        if (!parentPath.empty()) {
            std::error_code createError;
            std::filesystem::create_directories(parentPath, createError);
            if (createError) {
                throw std::runtime_error("could not create scene directory '" + parentPath.string() +
                                         "': " + createError.message());
            }
        }

        Json cameraJson = Json{{"position", vec3ToJson(camera_.position)},
                               {"target", vec3ToJson(camera_.target)},
                               {"up", vec3ToJson(camera_.up)},
                               {"verticalFovDegrees", glm::degrees(camera_.verticalFovRadians)},
                               {"nearPlane", camera_.nearPlane},
                               {"farPlane", camera_.farPlane}};

        Json lightJson = Json{{"direction", vec3ToJson(directionalLightSettings_.direction)},
                              {"color", vec3ToJson(directionalLightSettings_.color)},
                              {"intensity", directionalLightSettings_.intensity},
                              {"portfolioPresetActive", portfolioCaptureMode_}};

        Json objectsJson = Json::array();
        for (size_t objectIndex = 0; objectIndex < renderObjects_.size(); ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            const uint32_t objectId = renderObjectEditorId(object);
            const renderer::Material* material = primaryMaterialForObject(object);
            const ObjectDrawDebugInfo debugInfo = objectDrawDebugInfo(static_cast<uint32_t>(objectIndex));

            Json objectJson = Json{{"id", objectId},
                                   {"debugId", object.debugId},
                                   {"name", object.debugName},
                                   {"visible", object.visible},
                                   {"source", renderObjectSourceTypeName(object.sourceType)},
                                   {"portfolioOnly", object.portfolioOnly},
                                   {"hideInPortfolio", object.hideInPortfolio},
                                   {"transform", transformToJson(object.transform)},
                                   {"drawItemCount", debugInfo.drawItemCount}};

            objectJson["mesh"] = Json{{"name", object.mesh ? object.mesh->debugName() : std::string{}},
                                      {"pointer", pointerString(object.mesh)},
                                      {"submeshCount", meshSubmeshCount(object.mesh)}};
            objectJson["material"] =
                Json{{"name", material ? material->debugName : std::string{}},
                     {"assetName", material ? material->assetName : std::string{}},
                     {"assetPath", material ? stableProjectPathString(material->sourceAssetPath) : std::string{}},
                     {"shader", material ? material->shader : std::string{}},
                     {"primaryLabel", materialDebugLabel(object)},
                     {"pointer", pointerString(material)},
                     {"slotCount", object.materialCount},
                     {"source", material ? std::string(materialSourceName(material->source)) : std::string{"none"}},
                     {"materialAssetRebinding", object.materialTable ? "metadata-only for material tables"
                                                                      : "restored by assetPath when available"}};

            objectsJson.push_back(std::move(objectJson));
        }

        const Json sceneJson = Json{{"schemaVersion", 1},
                                    {"sceneName", portfolioCaptureMode_ ? "Portfolio Runtime Scene"
                                                  : (occlusionTestSceneActive_ ? "Occlusion Test Runtime Scene"
                                                                               : "Default Runtime Scene")},
                                    {"camera", std::move(cameraJson)},
                                    {"directionalLight", std::move(lightJson)},
                                    {"objects", std::move(objectsJson)},
                                     {"limitations",
                                     Json::array({"Mesh references and glTF material-table references are saved as "
                                                  "debug metadata only.",
                                                  "Simple object material asset paths are restored when they match a "
                                                  "loaded runtime material.",
                                                  "glTF material-table assignments remain runtime data and are not "
                                                  "rebuilt from scene JSON.",
                                                  "Load preserves current runtime mesh/material pointers and restores "
                                                  "matching object transforms, names, visibility, camera, and light."})}};

        std::ofstream output(sceneDocumentPath_);
        if (!output) {
            throw std::runtime_error("could not open scene file for writing");
        }
        output << sceneJson.dump(4) << '\n';
        if (!output) {
            throw std::runtime_error("failed while writing scene file");
        }

        lastSceneSaveStatus_ = "Saved scene to " + sceneDocumentPath_.string() + ".";
        Logger::info(lastSceneSaveStatus_);
    } catch (const std::exception& error) {
        lastSceneSaveStatus_ = "Scene save failed: " + std::string(error.what());
        Logger::warn(lastSceneSaveStatus_);
    }
}

void Renderer::loadSceneFromUi()
{
    try {
        std::error_code existsError;
        if (!std::filesystem::exists(sceneDocumentPath_, existsError)) {
            if (existsError) {
                throw std::runtime_error("could not check scene file: " + existsError.message());
            }
            lastSceneLoadStatus_ =
                std::string(kNoSavedSceneFoundMessage) + " Scene path: " + sceneDocumentPath_.string() + ".";
            Logger::warn(lastSceneLoadStatus_);
            return;
        }

        std::ifstream input(sceneDocumentPath_);
        if (!input) {
            throw std::runtime_error("could not open scene file for reading");
        }

        const Json sceneJson = Json::parse(input);
        if (!sceneJson.is_object()) {
            throw std::runtime_error("scene root must be a JSON object");
        }

        if (const Json* cameraJson = jsonObjectMember(sceneJson, "camera")) {
            readJsonVec3(*cameraJson, "position", camera_.position);
            readJsonVec3(*cameraJson, "target", camera_.target);
            readJsonVec3(*cameraJson, "up", camera_.up);

            float fovDegrees = glm::degrees(camera_.verticalFovRadians);
            if (readJsonFloat(*cameraJson, "verticalFovDegrees", fovDegrees)) {
                camera_.verticalFovRadians = glm::radians(std::clamp(fovDegrees, 1.0f, 160.0f));
            } else {
                readJsonFloat(*cameraJson, "verticalFovRadians", camera_.verticalFovRadians);
                camera_.verticalFovRadians = std::clamp(camera_.verticalFovRadians,
                                                        glm::radians(1.0f),
                                                        glm::radians(160.0f));
            }

            readJsonFloat(*cameraJson, "nearPlane", camera_.nearPlane);
            readJsonFloat(*cameraJson, "farPlane", camera_.farPlane);
            camera_.nearPlane = std::max(camera_.nearPlane, 0.001f);
            camera_.farPlane = std::max(camera_.farPlane, camera_.nearPlane + 0.001f);
            camera_.up = normalizedOrFallback(camera_.up, {0.0f, 1.0f, 0.0f});
            if (glm::length(camera_.target - camera_.position) <= 0.001f) {
                camera_.target = camera_.position + glm::vec3{0.0f, 0.0f, -1.0f};
            }
            csmSettings_.nearPlane = camera_.nearPlane;
            csmSettings_.farPlane = camera_.farPlane;
        }

        if (const Json* lightJson = jsonObjectMember(sceneJson, "directionalLight")) {
            readJsonVec3(*lightJson, "direction", directionalLightSettings_.direction);
            readJsonVec3(*lightJson, "color", directionalLightSettings_.color);
            readJsonFloat(*lightJson, "intensity", directionalLightSettings_.intensity);
            directionalLightSettings_.direction =
                normalizedOrFallback(directionalLightSettings_.direction,
                                     {kDirectionalLightDirection.x,
                                      kDirectionalLightDirection.y,
                                      kDirectionalLightDirection.z});
            directionalLightSettings_.color = glm::max(directionalLightSettings_.color, glm::vec3{0.0f});
            directionalLightSettings_.intensity = std::max(directionalLightSettings_.intensity, 0.0f);
        }

        size_t matchedObjects = 0;
        size_t skippedObjects = 0;
        size_t restoredMaterialAssets = 0;
        size_t skippedMaterialAssets = 0;
        std::vector<uint8_t> objectUsed(renderObjects_.size(), 0);
        if (const auto objectsIt = sceneJson.find("objects"); objectsIt != sceneJson.end()) {
            if (!objectsIt->is_array()) {
                throw std::runtime_error("Expected array member 'objects'.");
            }

            for (const Json& objectJson : *objectsIt) {
                if (!objectJson.is_object()) {
                    ++skippedObjects;
                    continue;
                }

                uint32_t objectId = 0;
                readJsonUint32(objectJson, "id", objectId);
                if (objectId == 0) {
                    readJsonUint32(objectJson, "sceneObjectId", objectId);
                }

                std::string objectName;
                readJsonString(objectJson, "name", objectName);

                size_t objectIndex = kInvalidRenderObjectIndex;
                if (objectId != 0) {
                    for (size_t candidateIndex = 0; candidateIndex < renderObjects_.size(); ++candidateIndex) {
                        if (objectUsed[candidateIndex]) {
                            continue;
                        }
                        const renderer::RenderObject& candidate = renderObjects_[candidateIndex];
                        if (renderObjectEditorId(candidate) == objectId || candidate.debugId == objectId) {
                            objectIndex = candidateIndex;
                            break;
                        }
                    }
                }

                if (objectIndex == kInvalidRenderObjectIndex && !objectName.empty()) {
                    for (size_t candidateIndex = 0; candidateIndex < renderObjects_.size(); ++candidateIndex) {
                        if (!objectUsed[candidateIndex] && renderObjects_[candidateIndex].debugName == objectName) {
                            objectIndex = candidateIndex;
                            break;
                        }
                    }
                }

                if (objectIndex == kInvalidRenderObjectIndex) {
                    ++skippedObjects;
                    continue;
                }

                objectUsed[objectIndex] = 1;
                renderer::RenderObject& object = renderObjects_[objectIndex];
                if (objectId != 0) {
                    object.sceneObjectId = objectId;
                    object.debugId = objectId;
                } else if (object.sceneObjectId == 0) {
                    object.sceneObjectId = object.debugId;
                }
                if (!objectName.empty()) {
                    object.debugName = objectName;
                }
                readJsonBool(objectJson, "visible", object.visible);

                if (const Json* materialJson = jsonObjectMember(objectJson, "material")) {
                    std::string materialAssetPathString;
                    readJsonString(*materialJson, "assetPath", materialAssetPathString);
                    if (!materialAssetPathString.empty()) {
                        const std::filesystem::path materialPath = resolveProjectPath(materialAssetPathString);
                        if (renderer::Material* material = findRuntimeMaterialByAssetPath(materialPath)) {
                            if (!object.materialTable) {
                                object.material = material;
                                ++restoredMaterialAssets;
                            } else {
                                ++skippedMaterialAssets;
                            }
                        } else {
                            ++skippedMaterialAssets;
                            Logger::warn("Scene material asset path did not match a runtime material: " +
                                         materialAssetPathString);
                        }
                    }
                }

                if (const Json* transformJson = jsonObjectMember(objectJson, "transform")) {
                    std::string mode = "trs";
                    readJsonString(*transformJson, "mode", mode);

                    glm::mat4 matrix{1.0f};
                    const bool hasMatrix = readJsonMat4(*transformJson, "matrix", matrix);
                    if (mode == "matrix" && hasMatrix) {
                        object.transform = renderer::Transform::fromMatrix(matrix);
                    } else {
                        renderer::Transform editableTransform = object.transform;
                        convertMatrixOverrideToEditableTrs(editableTransform);
                        editableTransform.useMatrixOverride = false;
                        editableTransform.matrixOverride = glm::mat4{1.0f};

                        readJsonVec3(*transformJson, "position", editableTransform.position);

                        glm::vec3 rotationDegrees = glm::degrees(editableTransform.rotationRadians);
                        if (readJsonVec3(*transformJson, "rotationDegrees", rotationDegrees)) {
                            editableTransform.rotationRadians = glm::radians(rotationDegrees);
                        } else {
                            readJsonVec3(*transformJson, "rotationRadians", editableTransform.rotationRadians);
                        }

                        readJsonVec3(*transformJson, "scale", editableTransform.scale);
                        object.transform = editableTransform;
                    }
                }

                object.animateTransform = false;
                ++matchedObjects;
            }
        }

        uint32_t maxObjectId = 0;
        for (renderer::RenderObject& object : renderObjects_) {
            if (object.sceneObjectId == 0) {
                object.sceneObjectId = object.debugId;
            }
            maxObjectId = std::max(maxObjectId, renderObjectEditorId(object));
        }
        if (maxObjectId < std::numeric_limits<uint32_t>::max()) {
            nextRenderObjectDebugId_ = std::max(nextRenderObjectDebugId_, maxObjectId + 1);
        }

        clampRuntimeSettings();
        invalidateDepthPyramid();
        invalidateTaaHistory();
        lastSceneLoadStatus_ = "Loaded scene from " + sceneDocumentPath_.string() + ". Matched " +
                               std::to_string(matchedObjects) + " object(s), skipped " +
                               std::to_string(skippedObjects) + ", restored " +
                               std::to_string(restoredMaterialAssets) + " material asset assignment(s), skipped " +
                               std::to_string(skippedMaterialAssets) + ".";
        Logger::info(lastSceneLoadStatus_);
    } catch (const std::exception& error) {
        lastSceneLoadStatus_ = "Scene load failed: " + std::string(error.what());
        Logger::warn(lastSceneLoadStatus_);
    }
}

} // namespace ve
