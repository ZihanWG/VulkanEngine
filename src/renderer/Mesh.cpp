#include "renderer/Mesh.h"

#include "core/Logger.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
            Logger::warn("Ignoring glTF attribute " + std::string(name) + " because it has fewer elements than POSITION.");
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

Mesh Mesh::createFromGltf(
    rhi::VulkanContext& context,
    const rhi::VulkanCommandContext& commandContext,
    const std::filesystem::path& path)
{
    tinygltf::TinyGLTF loader;
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

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    const tinygltf::Mesh& sourceMesh = model.meshes.front();
    for (size_t primitiveIndex = 0; primitiveIndex < sourceMesh.primitives.size(); ++primitiveIndex) {
        const tinygltf::Primitive& primitive = sourceMesh.primitives[primitiveIndex];
        if (!isTrianglePrimitive(primitive)) {
            Logger::warn("Skipping non-triangle glTF primitive " + std::to_string(primitiveIndex) + " in " + filename);
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
        const GltfAccessorView texcoords = makeOptionalAttributeView(model, primitive, "TEXCOORD_0", 2, vertexCount);
        const GltfAccessorView tangents = makeOptionalAttributeView(model, primitive, "TANGENT", 4, vertexCount);

        const uint32_t baseVertex = static_cast<uint32_t>(vertices.size());
        vertices.reserve(vertices.size() + vertexCount);
        for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
            const glm::vec4 position = readAccessorVec4(positions, vertexIndex, glm::vec4(0.0f));
            const glm::vec4 normal = readAccessorVec4(normals, vertexIndex, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
            const glm::vec4 uv = readAccessorVec4(texcoords, vertexIndex, glm::vec4(0.0f));
            // Missing tangents use a stable axis fallback. Proper imported-mesh tangent
            // generation is future work.
            const glm::vec4 tangent = readAccessorVec4(tangents, vertexIndex, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            Vertex vertex{};
            vertex.position = glm::vec3(position);
            vertex.color = glm::vec3(1.0f);
            vertex.uv = glm::vec2(uv);
            vertex.normal = glm::vec3(normal);
            vertex.tangent = tangent;
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
    }

    if (vertices.empty() || indices.empty()) {
        throw std::runtime_error("glTF file contains no supported triangle geometry: " + filename);
    }
    if (indices.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("glTF mesh has too many indices for this renderer.");
    }
    if ((indices.size() % 3) != 0) {
        Logger::warn("Loaded glTF triangle mesh has an index count that is not divisible by 3: " + filename);
    }

    // Milestone 24 preserves glTF positions as authored. No handedness, up-axis, or
    // node transform conversion is applied until scene hierarchy support exists.
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

    const std::string debugName = path.stem().string();
    rhi::debug::setObjectName(
        context.vkDevice(), mesh.vertexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, debugName + "VertexBuffer");
    rhi::debug::setObjectName(
        context.vkDevice(), mesh.indexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, debugName + "IndexBuffer");

    return mesh;
}

} // namespace ve::renderer
