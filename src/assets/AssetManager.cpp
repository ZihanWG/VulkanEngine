#include "assets/AssetManager.h"

#include <json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace ve::assets {

namespace {

using Json = nlohmann::json;

[[nodiscard]] std::string normalizedPathKey(const std::filesystem::path& path)
{
    return path.lexically_normal().generic_string();
}

[[nodiscard]] AssetHandle fnv1a64(std::string_view text)
{
    uint64_t hash = 14695981039346656037ull;
    for (const char character : text) {
        hash ^= static_cast<unsigned char>(character);
        hash *= 1099511628211ull;
    }
    return hash == 0 ? 14695981039346656037ull : hash;
}

[[nodiscard]] bool isFinite(float value)
{
    return std::isfinite(value);
}

[[nodiscard]] float readRequiredFloat(const Json& value, std::string_view fieldName)
{
    if (!value.is_number()) {
        throw std::runtime_error("Expected numeric material field '" + std::string(fieldName) + "'.");
    }

    const float parsed = value.get<float>();
    if (!isFinite(parsed)) {
        throw std::runtime_error("Material field '" + std::string(fieldName) + "' must be finite.");
    }
    return parsed;
}

bool readOptionalString(const Json& object, const char* fieldName, std::string& output)
{
    const auto it = object.find(fieldName);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string("Expected string material field '") + fieldName + "'.");
    }
    output = it->get<std::string>();
    return true;
}

bool readOptionalFloat(const Json& object, const char* fieldName, float& output)
{
    const auto it = object.find(fieldName);
    if (it == object.end()) {
        return false;
    }
    output = readRequiredFloat(*it, fieldName);
    return true;
}

bool readOptionalBool(const Json& object, const char* fieldName, bool& output)
{
    const auto it = object.find(fieldName);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_boolean()) {
        throw std::runtime_error(std::string("Expected boolean material field '") + fieldName + "'.");
    }
    output = it->get<bool>();
    return true;
}

bool readOptionalVec4(const Json& object, const char* fieldName, glm::vec4& output)
{
    const auto it = object.find(fieldName);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_array() || it->size() != 4) {
        throw std::runtime_error(std::string("Expected vec4 array material field '") + fieldName + "'.");
    }

    glm::vec4 parsed{1.0f};
    for (size_t componentIndex = 0; componentIndex < 4; ++componentIndex) {
        parsed[static_cast<glm::vec4::length_type>(componentIndex)] =
            readRequiredFloat((*it)[componentIndex], fieldName);
    }
    output = parsed;
    return true;
}

std::filesystem::path readTexturePath(const Json& textures, const char* fieldName)
{
    const auto it = textures.find(fieldName);
    if (it == textures.end()) {
        return {};
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string("Expected string material texture field 'textures.") + fieldName + "'.");
    }
    return std::filesystem::path(it->get<std::string>());
}

[[nodiscard]] MaterialAsset parseMaterialAssetJson(const Json& root, const std::filesystem::path& sourcePath)
{
    if (!root.is_object()) {
        throw std::runtime_error("Material asset root must be a JSON object.");
    }

    MaterialAsset material{};
    material.sourcePath = sourcePath.lexically_normal();
    if (!sourcePath.empty()) {
        material.name = sourcePath.stem().string();
    }

    readOptionalString(root, "name", material.name);
    readOptionalString(root, "shader", material.shader);
    readOptionalVec4(root, "baseColorFactor", material.baseColorFactor);
    readOptionalFloat(root, "metallicFactor", material.metallicFactor);
    readOptionalFloat(root, "roughnessFactor", material.roughnessFactor);
    readOptionalString(root, "alphaMode", material.alphaMode);
    readOptionalFloat(root, "alphaCutoff", material.alphaCutoff);
    readOptionalBool(root, "doubleSided", material.doubleSided);

    const auto texturesIt = root.find("textures");
    if (texturesIt != root.end()) {
        if (!texturesIt->is_object()) {
            throw std::runtime_error("Expected object material field 'textures'.");
        }
        material.textures.baseColor = readTexturePath(*texturesIt, "baseColor");
        material.textures.normal = readTexturePath(*texturesIt, "normal");
        material.textures.metallicRoughness = readTexturePath(*texturesIt, "metallicRoughness");
    }

    if (material.name.empty()) {
        material.name = "Material";
    }
    if (material.shader.empty()) {
        material.shader = "pbr_opaque";
    }
    if (material.alphaMode.empty()) {
        material.alphaMode = "OPAQUE";
    }

    return material;
}

[[nodiscard]] Json materialAssetToJson(const MaterialAsset& material)
{
    Json texturesJson = Json{{"baseColor", material.textures.baseColor.generic_string()},
                             {"normal", material.textures.normal.generic_string()},
                             {"metallicRoughness", material.textures.metallicRoughness.generic_string()}};

    return Json{{"name", material.name},
                {"shader", material.shader},
                {"baseColorFactor",
                 Json::array({material.baseColorFactor.r,
                              material.baseColorFactor.g,
                              material.baseColorFactor.b,
                              material.baseColorFactor.a})},
                {"metallicFactor", material.metallicFactor},
                {"roughnessFactor", material.roughnessFactor},
                {"textures", std::move(texturesJson)},
                {"alphaMode", material.alphaMode},
                {"alphaCutoff", material.alphaCutoff},
                {"doubleSided", material.doubleSided}};
}

} // namespace

MaterialAssetHandle AssetManager::loadMaterialAsset(const std::filesystem::path& path, std::string* errorMessage)
{
    if (errorMessage) {
        errorMessage->clear();
    }

    try {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("could not open material asset for reading");
        }

        const Json root = Json::parse(input);
        MaterialAsset material = parseMaterialAssetJson(root, path);
        material.handle = stableMaterialHandleForPath(path);
        materialAssets_[material.handle.id] = std::move(material);
        return MaterialAssetHandle{material.handle.id};
    } catch (const std::exception& error) {
        if (errorMessage) {
            *errorMessage = "Failed to load material asset '" + path.string() + "': " + error.what();
        }
        return {};
    }
}

bool AssetManager::saveMaterialAsset(const std::filesystem::path& path,
                                     const MaterialAsset& material,
                                     std::string* errorMessage)
{
    if (errorMessage) {
        errorMessage->clear();
    }

    try {
        const std::filesystem::path parentPath = path.parent_path();
        if (!parentPath.empty()) {
            std::error_code createError;
            std::filesystem::create_directories(parentPath, createError);
            if (createError) {
                throw std::runtime_error("could not create material directory '" + parentPath.string() +
                                         "': " + createError.message());
            }
        }

        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("could not open material asset for writing");
        }

        output << materialAssetToJson(material).dump(4) << '\n';
        if (!output) {
            throw std::runtime_error("failed while writing material asset");
        }

        MaterialAsset stored = material;
        stored.sourcePath = path.lexically_normal();
        stored.handle = stableMaterialHandleForPath(path);
        materialAssets_[stored.handle.id] = std::move(stored);
        return true;
    } catch (const std::exception& error) {
        if (errorMessage) {
            *errorMessage = "Failed to save material asset '" + path.string() + "': " + error.what();
        }
        return false;
    }
}

const MaterialAsset* AssetManager::materialAsset(MaterialAssetHandle handle) const
{
    const auto found = materialAssets_.find(handle.id);
    return found == materialAssets_.end() ? nullptr : &found->second;
}

MaterialAsset* AssetManager::materialAsset(MaterialAssetHandle handle)
{
    const auto found = materialAssets_.find(handle.id);
    return found == materialAssets_.end() ? nullptr : &found->second;
}

const MaterialAsset* AssetManager::findMaterialAssetByPath(const std::filesystem::path& path) const
{
    const auto foundHandle = materialPathToHandle_.find(normalizedPathKey(path));
    if (foundHandle == materialPathToHandle_.end()) {
        return nullptr;
    }
    return materialAsset(MaterialAssetHandle{foundHandle->second});
}

TextureAssetHandle AssetManager::registerTextureAsset(const std::filesystem::path& path, std::string debugName)
{
    if (path.empty()) {
        return {};
    }

    TextureAssetHandle handle = stableTextureHandleForPath(path);
    TextureAsset texture{};
    texture.handle = handle;
    texture.sourcePath = path.lexically_normal();
    texture.debugName = std::move(debugName);
    textureAssets_[handle.id] = std::move(texture);
    return handle;
}

const TextureAsset* AssetManager::textureAsset(TextureAssetHandle handle) const
{
    const auto found = textureAssets_.find(handle.id);
    return found == textureAssets_.end() ? nullptr : &found->second;
}

MaterialAsset AssetManager::fallbackMaterialAsset(std::string name)
{
    MaterialAsset material{};
    material.name = std::move(name);
    material.shader = "pbr_opaque";
    material.baseColorFactor = glm::vec4(1.0f);
    material.metallicFactor = 0.0f;
    material.roughnessFactor = 0.5f;
    material.alphaMode = "OPAQUE";
    material.alphaCutoff = 0.5f;
    material.doubleSided = false;
    material.fallback = true;
    return material;
}

MaterialAssetHandle AssetManager::stableMaterialHandleForPath(const std::filesystem::path& path)
{
    const std::string key = normalizedPathKey(path);
    const auto found = materialPathToHandle_.find(key);
    if (found != materialPathToHandle_.end()) {
        return MaterialAssetHandle{found->second};
    }

    AssetHandle handle = fnv1a64(key);
    while (handle == 0 || materialAssets_.find(handle) != materialAssets_.end()) {
        ++handle;
        if (handle == 0) {
            handle = 1;
        }
    }

    materialPathToHandle_[key] = handle;
    return MaterialAssetHandle{handle};
}

TextureAssetHandle AssetManager::stableTextureHandleForPath(const std::filesystem::path& path)
{
    const std::string key = normalizedPathKey(path);
    const auto found = texturePathToHandle_.find(key);
    if (found != texturePathToHandle_.end()) {
        return TextureAssetHandle{found->second};
    }

    AssetHandle handle = fnv1a64(key);
    while (handle == 0 || textureAssets_.find(handle) != textureAssets_.end()) {
        ++handle;
        if (handle == 0) {
            handle = 1;
        }
    }

    texturePathToHandle_[key] = handle;
    return TextureAssetHandle{handle};
}

} // namespace ve::assets
