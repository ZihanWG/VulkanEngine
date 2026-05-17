#include "renderer/Mesh.h"

#include "core/Logger.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ve::renderer {

namespace {

const std::array<Vertex, 24> kCubeVertices = {{
    // Front (+Z)
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 0.95f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},

    // Back (-Z)
    {{0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.95f, 1.0f, 0.95f}, {0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f}},

    // Left (-X)
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 0.95f, 1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.95f, 0.95f, 1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.95f, 0.95f, 1.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.95f, 0.95f, 1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},

    // Right (+X)
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.9f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 0.9f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.9f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 0.9f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f}},

    // Top (+Y)
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 1.0f}, {0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.95f, 1.0f}, {1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.95f, 1.0f}, {1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.95f, 1.0f}, {0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},

    // Bottom (-Y)
    {{-0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.95f, 1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.95f, 1.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.95f, 1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}
}};

const std::array<uint32_t, 36> kCubeIndices = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
};

struct GltfAccessorView {
    const tinygltf::Accessor* accessor = nullptr;
    const unsigned char* data = nullptr;
    size_t stride = 0;
    size_t componentSize = 0;
    int componentCount = 0;
};

[[nodiscard]] int componentCountForType(int type)
{
    switch (type) {
    case TINYGLTF_TYPE_SCALAR:
        return 1;
    case TINYGLTF_TYPE_VEC2:
        return 2;
    case TINYGLTF_TYPE_VEC3:
        return 3;
    case TINYGLTF_TYPE_VEC4:
        return 4;
    default:
        return 0;
    }
}

[[nodiscard]] size_t componentSizeForType(int componentType)
{
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2;
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return 4;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return 8;
    default:
        return 0;
    }
}

[[nodiscard]] bool addWouldOverflow(size_t lhs, size_t rhs)
{
    return lhs > std::numeric_limits<size_t>::max() - rhs;
}

[[nodiscard]] GltfAccessorView makeAccessorView(
    const tinygltf::Model& model,
    int accessorIndex,
    std::string_view name,
    int minimumComponentCount)
{
    if (accessorIndex < 0 || static_cast<size_t>(accessorIndex) >= model.accessors.size()) {
        throw std::runtime_error("glTF accessor index is out of range for " + std::string(name) + ".");
    }

    const tinygltf::Accessor& accessor = model.accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.sparse.isSparse) {
        throw std::runtime_error("Sparse glTF accessors are not supported yet for " + std::string(name) + ".");
    }
    if (accessor.bufferView < 0 || static_cast<size_t>(accessor.bufferView) >= model.bufferViews.size()) {
        throw std::runtime_error("glTF accessor has no valid bufferView for " + std::string(name) + ".");
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    if (bufferView.buffer < 0 || static_cast<size_t>(bufferView.buffer) >= model.buffers.size()) {
        throw std::runtime_error("glTF bufferView references an invalid buffer for " + std::string(name) + ".");
    }

    const int componentCount = componentCountForType(accessor.type);
    const size_t componentSize = componentSizeForType(accessor.componentType);
    if (componentCount < minimumComponentCount || componentSize == 0) {
        throw std::runtime_error("glTF accessor has an unsupported type for " + std::string(name) + ".");
    }

    const int byteStride = accessor.ByteStride(bufferView);
    if (byteStride <= 0) {
        throw std::runtime_error("glTF accessor has an invalid byte stride for " + std::string(name) + ".");
    }

    const size_t baseOffset = bufferView.byteOffset + accessor.byteOffset;
    const size_t elementByteSize = componentSize * static_cast<size_t>(componentCount);
    if (static_cast<size_t>(byteStride) < elementByteSize) {
        throw std::runtime_error("glTF accessor stride is smaller than its element size for " + std::string(name) + ".");
    }

    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView.buffer)];
    if (accessor.count > 0) {
        const size_t lastElementOffset = static_cast<size_t>(byteStride) * (accessor.count - 1);
        if (addWouldOverflow(baseOffset, lastElementOffset) ||
            addWouldOverflow(baseOffset + lastElementOffset, elementByteSize) ||
            baseOffset + lastElementOffset + elementByteSize > buffer.data.size()) {
            throw std::runtime_error("glTF accessor data exceeds its buffer for " + std::string(name) + ".");
        }
    }

    return {&accessor, buffer.data.data() + baseOffset, static_cast<size_t>(byteStride), componentSize, componentCount};
}

[[nodiscard]] float readAccessorComponent(const unsigned char* data, int componentType, bool normalized)
{
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
        int8_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? std::max(static_cast<float>(value) / 127.0f, -1.0f) : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        uint8_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? static_cast<float>(value) / 255.0f : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
        int16_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? std::max(static_cast<float>(value) / 32767.0f, -1.0f) : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? static_cast<float>(value) / 65535.0f : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_INT: {
        int32_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        uint32_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return normalized ? static_cast<float>(static_cast<double>(value) / 4294967295.0) : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        float value = 0.0f;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_DOUBLE: {
        double value = 0.0;
        std::memcpy(&value, data, sizeof(value));
        return static_cast<float>(value);
    }
    default:
        throw std::runtime_error("Unsupported glTF accessor component type.");
    }
}

[[nodiscard]] glm::vec4 readAccessorVec4(const GltfAccessorView& view, size_t elementIndex, const glm::vec4& fallback)
{
    if (!view.accessor || elementIndex >= view.accessor->count) {
        return fallback;
    }

    glm::vec4 result = fallback;
    const unsigned char* elementData = view.data + elementIndex * view.stride;
    const int count = std::min(view.componentCount, 4);
    for (int component = 0; component < count; ++component) {
        result[component] = readAccessorComponent(elementData + static_cast<size_t>(component) * view.componentSize,
                                                  view.accessor->componentType,
                                                  view.accessor->normalized);
    }

    return result;
}

[[nodiscard]] Aabb accessorMinMaxBounds(const tinygltf::Accessor& accessor)
{
    Aabb bounds{};
    if (accessor.minValues.size() < 3 || accessor.maxValues.size() < 3) {
        return bounds;
    }

    const glm::vec3 minBounds{static_cast<float>(accessor.minValues[0]),
                              static_cast<float>(accessor.minValues[1]),
                              static_cast<float>(accessor.minValues[2])};
    const glm::vec3 maxBounds{static_cast<float>(accessor.maxValues[0]),
                              static_cast<float>(accessor.maxValues[1]),
                              static_cast<float>(accessor.maxValues[2])};
    if (!std::isfinite(minBounds.x) || !std::isfinite(minBounds.y) || !std::isfinite(minBounds.z) ||
        !std::isfinite(maxBounds.x) || !std::isfinite(maxBounds.y) || !std::isfinite(maxBounds.z)) {
        return bounds;
    }

    bounds.expand(minBounds);
    bounds.expand(maxBounds);
    return bounds;
}

[[nodiscard]] uint32_t readIndexValue(const GltfAccessorView& view, size_t elementIndex)
{
    const unsigned char* elementData = view.data + elementIndex * view.stride;
    switch (view.accessor->componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        uint8_t value = 0;
        std::memcpy(&value, elementData, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t value = 0;
        std::memcpy(&value, elementData, sizeof(value));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        uint32_t value = 0;
        std::memcpy(&value, elementData, sizeof(value));
        return value;
    }
    default:
        throw std::runtime_error("glTF indices must use UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT.");
    }
}

[[nodiscard]] int findAttribute(const tinygltf::Primitive& primitive, const char* name)
{
    const auto attribute = primitive.attributes.find(name);
    return attribute == primitive.attributes.end() ? -1 : attribute->second;
}

[[nodiscard]] GltfAccessorView makeOptionalAttributeView(
    const tinygltf::Model& model,
    const tinygltf::Primitive& primitive,
    const char* name,
    int minimumComponentCount,
    size_t vertexCount)
{
    const int accessorIndex = findAttribute(primitive, name);
    if (accessorIndex < 0) {
        return {};
    }

    try {
        GltfAccessorView view = makeAccessorView(model, accessorIndex, name, minimumComponentCount);
        if (view.accessor->count < vertexCount) {
            Logger::warn("Ignoring glTF attribute " + std::string(name) +
                         " because it has fewer elements than POSITION.");
            return {};
        }
        return view;
    } catch (const std::exception& error) {
        Logger::warn("Ignoring glTF attribute " + std::string(name) + ": " + error.what());
        return {};
    }
}

[[nodiscard]] bool isTrianglePrimitive(const tinygltf::Primitive& primitive)
{
    return primitive.mode == -1 || primitive.mode == TINYGLTF_MODE_TRIANGLES;
}

bool copyEncodedImageData(tinygltf::Image* image,
                          const int imageIndex,
                          std::string* error,
                          std::string* warning,
                          int requestedWidth,
                          int requestedHeight,
                          const unsigned char* bytes,
                          int size,
                          void* userData)
{
    (void)imageIndex;
    (void)warning;
    (void)userData;

    if (!image || !bytes || size <= 0) {
        if (error) {
            *error += "Embedded glTF image data is empty.\n";
        }
        return false;
    }

    image->image.assign(bytes, bytes + size);
    image->as_is = true;
    if (requestedWidth > 0) {
        image->width = requestedWidth;
    }
    if (requestedHeight > 0) {
        image->height = requestedHeight;
    }
    return true;
}

[[nodiscard]] std::filesystem::path resolveImagePath(const std::filesystem::path& gltfPath, const std::string& uri)
{
    std::string decodedUri = uri;
    if (!tinygltf::URIDecode(uri, &decodedUri, nullptr)) {
        Logger::warn("Failed to decode glTF image URI '" + uri + "'; using it as written.");
        decodedUri = uri;
    }

    std::filesystem::path imagePath(decodedUri);
    if (imagePath.is_relative()) {
        imagePath = gltfPath.parent_path() / imagePath;
    }
    return imagePath.lexically_normal();
}

[[nodiscard]] std::string textureDebugName(const tinygltf::Texture& texture,
                                           const tinygltf::Image& image,
                                           size_t textureIndex)
{
    if (!texture.name.empty()) {
        return texture.name;
    }
    if (!image.name.empty()) {
        return image.name;
    }
    return "glTF Texture " + std::to_string(textureIndex);
}

[[nodiscard]] std::vector<GltfTextureInfo> loadGltfTextureInfos(const tinygltf::Model& model,
                                                                const std::filesystem::path& gltfPath)
{
    std::vector<GltfTextureInfo> textures(model.textures.size());

    for (size_t textureIndex = 0; textureIndex < model.textures.size(); ++textureIndex) {
        const tinygltf::Texture& sourceTexture = model.textures[textureIndex];
        if (sourceTexture.source < 0 || static_cast<size_t>(sourceTexture.source) >= model.images.size()) {
            Logger::warn("glTF texture " + std::to_string(textureIndex) + " does not reference a valid image.");
            continue;
        }

        const tinygltf::Image& sourceImage = model.images[static_cast<size_t>(sourceTexture.source)];
        GltfTextureInfo& texture = textures[textureIndex];
        texture.debugName = textureDebugName(sourceTexture, sourceImage, textureIndex);

        if (!sourceImage.image.empty()) {
            texture.encodedData = sourceImage.image;
            texture.embedded = true;
            continue;
        }

        if (!sourceImage.uri.empty()) {
            texture.path = resolveImagePath(gltfPath, sourceImage.uri);
            continue;
        }

        Logger::warn("glTF image for texture " + std::to_string(textureIndex) +
                     " has neither an external URI nor embedded encoded data.");
    }

    return textures;
}

[[nodiscard]] int validTextureIndex(const tinygltf::Model& model,
                                    int textureIndex,
                                    std::string_view materialName,
                                    std::string_view slotName)
{
    if (textureIndex < 0) {
        return -1;
    }
    if (static_cast<size_t>(textureIndex) >= model.textures.size()) {
        Logger::warn("glTF material '" + std::string(materialName) + "' references an out-of-range " +
                     std::string(slotName) + " texture index.");
        return -1;
    }

    const tinygltf::Texture& texture = model.textures[static_cast<size_t>(textureIndex)];
    if (texture.source < 0 || static_cast<size_t>(texture.source) >= model.images.size()) {
        Logger::warn("glTF material '" + std::string(materialName) + "' references a " + std::string(slotName) +
                     " texture without a valid image source.");
        return -1;
    }

    return textureIndex;
}

[[nodiscard]] GltfMaterialInfo makeDefaultGltfMaterial(std::string debugName)
{
    GltfMaterialInfo material{};
    material.debugName = std::move(debugName);
    return material;
}

[[nodiscard]] GltfMaterialInfo loadGltfMaterialInfo(const tinygltf::Model& model,
                                                    const tinygltf::Material& sourceMaterial,
                                                    size_t materialIndex)
{
    GltfMaterialInfo material{};
    material.debugName =
        sourceMaterial.name.empty() ? "glTF Material " + std::to_string(materialIndex) : sourceMaterial.name;

    const tinygltf::PbrMetallicRoughness& pbr = sourceMaterial.pbrMetallicRoughness;
    if (pbr.baseColorFactor.size() >= 4) {
        material.baseColorFactor = {static_cast<float>(pbr.baseColorFactor[0]),
                                    static_cast<float>(pbr.baseColorFactor[1]),
                                    static_cast<float>(pbr.baseColorFactor[2]),
                                    static_cast<float>(pbr.baseColorFactor[3])};
    }
    material.metallic = static_cast<float>(pbr.metallicFactor);
    material.roughness = static_cast<float>(pbr.roughnessFactor);
    material.baseColorTextureIndex =
        validTextureIndex(model, pbr.baseColorTexture.index, material.debugName, "base color");
    material.normalTextureIndex =
        validTextureIndex(model, sourceMaterial.normalTexture.index, material.debugName, "normal");
    material.metallicRoughnessTextureIndex =
        validTextureIndex(model, pbr.metallicRoughnessTexture.index, material.debugName, "metallic-roughness");

    if (pbr.baseColorTexture.index >= 0 && pbr.baseColorTexture.texCoord != 0) {
        Logger::warn("glTF material '" + material.debugName +
                     "' uses a base color texCoord set other than TEXCOORD_0; TEXCOORD_0 will be sampled.");
    }
    if (sourceMaterial.normalTexture.index >= 0 && sourceMaterial.normalTexture.texCoord != 0) {
        Logger::warn("glTF material '" + material.debugName +
                     "' uses a normal texCoord set other than TEXCOORD_0; TEXCOORD_0 will be sampled.");
    }
    if (pbr.metallicRoughnessTexture.index >= 0 && pbr.metallicRoughnessTexture.texCoord != 0) {
        Logger::warn("glTF material '" + material.debugName +
                     "' uses a metallic-roughness texCoord set other than TEXCOORD_0; TEXCOORD_0 will be sampled.");
    }

    return material;
}

[[nodiscard]] std::vector<GltfMaterialInfo> loadGltfMaterialInfos(const tinygltf::Model& model)
{
    std::vector<GltfMaterialInfo> materials;
    materials.reserve(model.materials.size());

    for (size_t materialIndex = 0; materialIndex < model.materials.size(); ++materialIndex) {
        materials.push_back(loadGltfMaterialInfo(model, model.materials[materialIndex], materialIndex));
    }

    return materials;
}

[[nodiscard]] glm::mat4 loadNodeLocalTransform(const tinygltf::Node& node)
{
    if (!node.matrix.empty()) {
        if (node.matrix.size() != 16) {
            throw std::runtime_error("glTF node matrix must contain 16 values.");
        }

        glm::mat4 transform{1.0f};
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                transform[column][row] = static_cast<float>(node.matrix[static_cast<size_t>(column * 4 + row)]);
            }
        }
        return transform;
    }

    glm::vec3 translation{0.0f};
    if (!node.translation.empty()) {
        if (node.translation.size() != 3) {
            throw std::runtime_error("glTF node translation must contain 3 values.");
        }
        translation = {static_cast<float>(node.translation[0]),
                       static_cast<float>(node.translation[1]),
                       static_cast<float>(node.translation[2])};
    }

    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    if (!node.rotation.empty()) {
        if (node.rotation.size() != 4) {
            throw std::runtime_error("glTF node rotation must contain 4 values.");
        }
        rotation = glm::quat{static_cast<float>(node.rotation[3]),
                             static_cast<float>(node.rotation[0]),
                             static_cast<float>(node.rotation[1]),
                             static_cast<float>(node.rotation[2])};
    }

    glm::vec3 scale{1.0f};
    if (!node.scale.empty()) {
        if (node.scale.size() != 3) {
            throw std::runtime_error("glTF node scale must contain 3 values.");
        }
        scale = {static_cast<float>(node.scale[0]),
                 static_cast<float>(node.scale[1]),
                 static_cast<float>(node.scale[2])};
    }

    return glm::translate(glm::mat4{1.0f}, translation) * glm::mat4_cast(rotation) *
           glm::scale(glm::mat4{1.0f}, scale);
}

[[nodiscard]] std::string meshDebugName(const tinygltf::Mesh& mesh, size_t meshIndex)
{
    return mesh.name.empty() ? "glTF Mesh " + std::to_string(meshIndex) : mesh.name;
}

[[nodiscard]] std::string nodeMeshInstanceDebugName(const tinygltf::Model& model,
                                                    const tinygltf::Node& node,
                                                    size_t nodeIndex)
{
    if (!node.name.empty()) {
        return node.name;
    }

    if (node.mesh >= 0 && static_cast<size_t>(node.mesh) < model.meshes.size()) {
        const std::string meshName = meshDebugName(model.meshes[static_cast<size_t>(node.mesh)],
                                                  static_cast<size_t>(node.mesh));
        return "glTF Node " + std::to_string(nodeIndex) + " (" + meshName + ")";
    }

    return "glTF Node " + std::to_string(nodeIndex);
}

} // namespace

VkVertexInputBindingDescription vertexBindingDescription()
{
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = static_cast<uint32_t>(sizeof(Vertex));
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding;
}

std::array<VkVertexInputAttributeDescription, 5> vertexAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 5> attributes{};

    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = static_cast<uint32_t>(offsetof(Vertex, position));

    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = static_cast<uint32_t>(offsetof(Vertex, color));

    attributes[2].location = 2;
    attributes[2].binding = 0;
    attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[2].offset = static_cast<uint32_t>(offsetof(Vertex, uv));

    attributes[3].location = 3;
    attributes[3].binding = 0;
    attributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[3].offset = static_cast<uint32_t>(offsetof(Vertex, normal));

    attributes[4].location = 4;
    attributes[4].binding = 0;
    attributes[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[4].offset = static_cast<uint32_t>(offsetof(Vertex, tangent));

    return attributes;
}

Mesh Mesh::createCube(rhi::VulkanContext& context, const rhi::VulkanCommandContext& commandContext)
{
    Mesh mesh;
    mesh.debugName_ = "Built-in Cube Mesh";
    for (const Vertex& vertex : kCubeVertices) {
        mesh.localBounds_.expand(vertex.position);
    }

    mesh.vertexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const Vertex>(kCubeVertices.data(), kCubeVertices.size())),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    mesh.indexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const uint32_t>(kCubeIndices.data(), kCubeIndices.size())),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    mesh.indexCount_ = static_cast<uint32_t>(kCubeIndices.size());
    return mesh;
}

LoadedGltfAsset Mesh::createFromGltf(
    rhi::VulkanContext& context,
    const rhi::VulkanCommandContext& commandContext,
    const std::filesystem::path& path)
{
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(copyEncodedImageData, nullptr);

    tinygltf::Model model;
    std::string error;
    std::string warning;

    const std::string filename = path.string();
    bool loaded = false;
    if (path.extension() == ".glb") {
        loaded = loader.LoadBinaryFromFile(&model, &error, &warning, filename);
    } else {
        loaded = loader.LoadASCIIFromFile(&model, &error, &warning, filename);
    }

    if (!warning.empty()) {
        Logger::warn("tinygltf warning while loading '" + filename + "': " + warning);
    }
    if (!loaded) {
        throw std::runtime_error("Failed to load glTF file '" + filename + "': " + error);
    }
    if (model.meshes.empty()) {
        throw std::runtime_error("glTF file contains no meshes: " + filename);
    }

    std::vector<GltfTextureInfo> textureInfos = loadGltfTextureInfos(model, path);
    std::vector<GltfMaterialInfo> materialInfos = loadGltfMaterialInfos(model);
    uint32_t defaultMaterialIndex = 0;
    bool hasDefaultMaterial = false;

    const auto ensureDefaultMaterial = [&materialInfos, &defaultMaterialIndex, &hasDefaultMaterial]() {
        if (!hasDefaultMaterial) {
            defaultMaterialIndex = static_cast<uint32_t>(materialInfos.size());
            materialInfos.push_back(makeDefaultGltfMaterial("Default glTF Material"));
            hasDefaultMaterial = true;
        }
        return defaultMaterialIndex;
    };

    const auto resolvePrimitiveMaterialIndex = [&model, &ensureDefaultMaterial](const tinygltf::Primitive& primitive,
                                                                                size_t primitiveIndex,
                                                                                const std::string& filename) {
        if (primitive.material < 0) {
            return ensureDefaultMaterial();
        }

        if (static_cast<size_t>(primitive.material) >= model.materials.size()) {
            Logger::warn("glTF primitive " + std::to_string(primitiveIndex) + " in " + filename +
                         " references an out-of-range material; using the default material.");
            return ensureDefaultMaterial();
        }

        return static_cast<uint32_t>(primitive.material);
    };

    const auto createMeshFromSource = [&](size_t meshIndex) {
        const tinygltf::Mesh& sourceMesh = model.meshes[meshIndex];
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<MeshPrimitive> subMeshes;
        Aabb localBounds{};

        for (size_t primitiveIndex = 0; primitiveIndex < sourceMesh.primitives.size(); ++primitiveIndex) {
            const tinygltf::Primitive& primitive = sourceMesh.primitives[primitiveIndex];
            if (!isTrianglePrimitive(primitive)) {
                Logger::warn("Skipping non-triangle glTF primitive " + std::to_string(primitiveIndex) + " in " +
                             filename);
                continue;
            }

            const int positionAccessorIndex = findAttribute(primitive, "POSITION");
            if (positionAccessorIndex < 0) {
                Logger::warn("Skipping glTF primitive without POSITION attribute in " + filename);
                continue;
            }

            const GltfAccessorView positions = makeAccessorView(model, positionAccessorIndex, "POSITION", 3);
            const size_t vertexCount = positions.accessor->count;
            if (vertexCount == 0) {
                Logger::warn("Skipping empty glTF primitive in " + filename);
                continue;
            }
            if (vertices.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - vertexCount) {
                throw std::runtime_error("glTF mesh has too many vertices for a uint32_t index buffer.");
            }

            const GltfAccessorView normals = makeOptionalAttributeView(model, primitive, "NORMAL", 3, vertexCount);
            const GltfAccessorView texcoords =
                makeOptionalAttributeView(model, primitive, "TEXCOORD_0", 2, vertexCount);
            const GltfAccessorView tangents = makeOptionalAttributeView(model, primitive, "TANGENT", 4, vertexCount);
            const Aabb positionAccessorBounds = accessorMinMaxBounds(*positions.accessor);
            const bool hasPositionAccessorBounds = positionAccessorBounds.valid();
            if (hasPositionAccessorBounds) {
                localBounds.merge(positionAccessorBounds);
            }

            const uint32_t baseVertex = static_cast<uint32_t>(vertices.size());
            const uint32_t firstIndex = static_cast<uint32_t>(indices.size());
            const uint32_t materialIndex = resolvePrimitiveMaterialIndex(primitive, primitiveIndex, filename);
            vertices.reserve(vertices.size() + vertexCount);
            for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
                const glm::vec4 position = readAccessorVec4(positions, vertexIndex, glm::vec4(0.0f));
                const glm::vec4 normal = readAccessorVec4(normals, vertexIndex, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
                const glm::vec4 uv = readAccessorVec4(texcoords, vertexIndex, glm::vec4(0.0f));
                // Missing tangents use a stable axis fallback. Proper imported-mesh tangent
                // generation is future work.
                const glm::vec4 tangent =
                    readAccessorVec4(tangents, vertexIndex, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

                Vertex vertex{};
                vertex.position = glm::vec3(position);
                vertex.color = glm::vec3(1.0f);
                vertex.uv = glm::vec2(uv);
                vertex.normal = glm::vec3(normal);
                vertex.tangent = tangent;
                if (!hasPositionAccessorBounds) {
                    localBounds.expand(vertex.position);
                }
                vertices.push_back(vertex);
            }

            if (primitive.indices >= 0) {
                const GltfAccessorView sourceIndices = makeAccessorView(model, primitive.indices, "indices", 1);
                if (sourceIndices.accessor->type != TINYGLTF_TYPE_SCALAR) {
                    throw std::runtime_error("glTF index accessor must be SCALAR.");
                }

                indices.reserve(indices.size() + sourceIndices.accessor->count);
                for (size_t indexElement = 0; indexElement < sourceIndices.accessor->count; ++indexElement) {
                    const uint32_t localIndex = readIndexValue(sourceIndices, indexElement);
                    if (localIndex >= vertexCount) {
                        throw std::runtime_error("glTF index references a vertex outside the primitive.");
                    }
                    indices.push_back(baseVertex + localIndex);
                }
            } else {
                indices.reserve(indices.size() + vertexCount);
                for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
                    indices.push_back(baseVertex + static_cast<uint32_t>(vertexIndex));
                }
            }

            const uint32_t indexCount = static_cast<uint32_t>(indices.size()) - firstIndex;
            if (indexCount > 0) {
                subMeshes.push_back({firstIndex, indexCount, materialIndex});
            }
        }

        if (vertices.empty() || indices.empty()) {
            throw std::runtime_error("glTF mesh contains no supported triangle geometry.");
        }
        if (indices.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("glTF mesh has too many indices for this renderer.");
        }
        if ((indices.size() % 3) != 0) {
            Logger::warn("Loaded glTF triangle mesh has an index count that is not divisible by 3: " + filename);
        }

        // Milestone 26 preserves glTF positions and node transforms as authored.
        // No handedness or up-axis conversion is applied yet.
        Mesh mesh;
        mesh.vertexBuffer_.createDeviceLocal(
            context,
            commandContext,
            std::as_bytes(std::span<const Vertex>(vertices.data(), vertices.size())),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        mesh.indexBuffer_.createDeviceLocal(
            context,
            commandContext,
            std::as_bytes(std::span<const uint32_t>(indices.data(), indices.size())),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        mesh.indexCount_ = static_cast<uint32_t>(indices.size());
        mesh.subMeshes_ = std::move(subMeshes);
        mesh.localBounds_ = localBounds;
        mesh.debugName_ = meshDebugName(sourceMesh, meshIndex);

        const std::string debugName = path.stem().string() + "Mesh" + std::to_string(meshIndex);
        rhi::debug::setObjectName(
            context.vkDevice(), mesh.vertexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, debugName + "VertexBuffer");
        rhi::debug::setObjectName(
            context.vkDevice(), mesh.indexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, debugName + "IndexBuffer");

        return mesh;
    };

    std::vector<Mesh> meshes(model.meshes.size());
    std::vector<bool> meshLoaded(model.meshes.size(), false);
    size_t loadedMeshCount = 0;
    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
        try {
            meshes[meshIndex] = createMeshFromSource(meshIndex);
            meshLoaded[meshIndex] = true;
            ++loadedMeshCount;
        } catch (const std::exception& error) {
            Logger::warn("Skipping glTF mesh " + std::to_string(meshIndex) + " (" +
                         meshDebugName(model.meshes[meshIndex], meshIndex) + "): " + error.what());
        }
    }

    if (loadedMeshCount == 0) {
        throw std::runtime_error("glTF file contains no supported triangle geometry: " + filename);
    }

    std::vector<GltfNodeMeshInstance> nodeMeshInstances;
    nodeMeshInstances.reserve(model.nodes.size());

    const auto addFallbackFirstMeshInstance = [&nodeMeshInstances, &meshLoaded]() {
        for (size_t meshIndex = 0; meshIndex < meshLoaded.size(); ++meshIndex) {
            if (meshLoaded[meshIndex]) {
                GltfNodeMeshInstance instance{};
                instance.meshIndex = static_cast<uint32_t>(meshIndex);
                instance.transform = glm::mat4{1.0f};
                instance.debugName = "Imported glTF Mesh " + std::to_string(meshIndex);
                nodeMeshInstances.push_back(std::move(instance));
                return true;
            }
        }

        return false;
    };

    const auto traverseNode = [&](auto&& self, int nodeIndex, const glm::mat4& parentTransform, size_t depth) -> void {
        if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= model.nodes.size()) {
            Logger::warn("Skipping out-of-range glTF scene node index " + std::to_string(nodeIndex) + ".");
            return;
        }
        if (depth > model.nodes.size()) {
            Logger::warn("Stopping glTF node traversal because the hierarchy appears cyclic.");
            return;
        }

        const tinygltf::Node& node = model.nodes[static_cast<size_t>(nodeIndex)];
        const glm::mat4 worldTransform = parentTransform * loadNodeLocalTransform(node);

        if (node.mesh >= 0) {
            if (static_cast<size_t>(node.mesh) >= meshLoaded.size()) {
                Logger::warn("Skipping glTF node mesh reference " + std::to_string(node.mesh) +
                             " because it is out of range.");
            } else if (!meshLoaded[static_cast<size_t>(node.mesh)]) {
                Logger::warn("Skipping glTF node mesh reference " + std::to_string(node.mesh) +
                             " because the mesh has no supported geometry.");
            } else {
                GltfNodeMeshInstance instance{};
                instance.meshIndex = static_cast<uint32_t>(node.mesh);
                instance.transform = worldTransform;
                instance.debugName = nodeMeshInstanceDebugName(model, node, static_cast<size_t>(nodeIndex));
                nodeMeshInstances.push_back(std::move(instance));
            }
        }

        for (int childNodeIndex : node.children) {
            self(self, childNodeIndex, worldTransform, depth + 1);
        }
    };

    if (!model.scenes.empty()) {
        const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        if (sceneIndex < 0 || static_cast<size_t>(sceneIndex) >= model.scenes.size()) {
            Logger::warn("glTF default scene index is out of range; using the first loaded mesh at identity.");
        } else {
            const tinygltf::Scene& scene = model.scenes[static_cast<size_t>(sceneIndex)];
            for (int rootNodeIndex : scene.nodes) {
                traverseNode(traverseNode, rootNodeIndex, glm::mat4{1.0f}, 0);
            }
        }
    } else {
        Logger::warn("glTF file has no scenes; using the first loaded mesh at identity.");
    }

    if (nodeMeshInstances.empty()) {
        Logger::warn("glTF scene contains no supported mesh nodes; using the first loaded mesh at identity.");
        if (!addFallbackFirstMeshInstance()) {
            throw std::runtime_error("glTF file contains no supported mesh nodes: " + filename);
        }
    }

    LoadedGltfAsset loadedAsset{};
    loadedAsset.meshes = std::move(meshes);
    loadedAsset.nodeMeshInstances = std::move(nodeMeshInstances);
    loadedAsset.materials = std::move(materialInfos);
    loadedAsset.textures = std::move(textureInfos);
    return loadedAsset;
}

} // namespace ve::renderer
