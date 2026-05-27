#pragma once

#include <cstdint>
#include <filesystem>
#include <glm/vec4.hpp>
#include <string>
#include <unordered_map>

namespace ve::assets {

using AssetHandle = uint64_t;

struct MaterialAssetHandle {
    AssetHandle id = 0;

    [[nodiscard]] explicit operator bool() const { return id != 0; }
};

struct TextureAssetHandle {
    AssetHandle id = 0;

    [[nodiscard]] explicit operator bool() const { return id != 0; }
};

struct MaterialTexturePaths {
    std::filesystem::path baseColor;
    std::filesystem::path normal;
    std::filesystem::path metallicRoughness;
};

struct MaterialAsset {
    MaterialAssetHandle handle{};
    std::filesystem::path sourcePath;
    std::string name = "Default Material";
    std::string shader = "pbr_opaque";
    glm::vec4 baseColorFactor{1.0f};
    float metallicFactor = 0.0f;
    float roughnessFactor = 0.5f;
    MaterialTexturePaths textures{};
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
    bool fallback = false;
};

struct TextureAsset {
    TextureAssetHandle handle{};
    std::filesystem::path sourcePath;
    std::string debugName;
};

class AssetManager final {
public:
    AssetManager() = default;

    [[nodiscard]] MaterialAssetHandle loadMaterialAsset(const std::filesystem::path& path,
                                                        std::string* errorMessage = nullptr);
    [[nodiscard]] bool saveMaterialAsset(const std::filesystem::path& path,
                                         const MaterialAsset& material,
                                         std::string* errorMessage = nullptr);

    [[nodiscard]] const MaterialAsset* materialAsset(MaterialAssetHandle handle) const;
    [[nodiscard]] MaterialAsset* materialAsset(MaterialAssetHandle handle);
    [[nodiscard]] const MaterialAsset* findMaterialAssetByPath(const std::filesystem::path& path) const;

    [[nodiscard]] TextureAssetHandle registerTextureAsset(const std::filesystem::path& path,
                                                          std::string debugName = {});
    [[nodiscard]] const TextureAsset* textureAsset(TextureAssetHandle handle) const;

    [[nodiscard]] static MaterialAsset fallbackMaterialAsset(std::string name = "Fallback Material");

private:
    [[nodiscard]] MaterialAssetHandle stableMaterialHandleForPath(const std::filesystem::path& path);
    [[nodiscard]] TextureAssetHandle stableTextureHandleForPath(const std::filesystem::path& path);

    std::unordered_map<AssetHandle, MaterialAsset> materialAssets_;
    std::unordered_map<std::string, AssetHandle> materialPathToHandle_;
    std::unordered_map<AssetHandle, TextureAsset> textureAssets_;
    std::unordered_map<std::string, AssetHandle> texturePathToHandle_;
};

} // namespace ve::assets
