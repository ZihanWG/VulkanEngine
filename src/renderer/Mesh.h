#pragma once

#include "renderer/Bounds.h"
#include "renderer/MeshLod.h"
#include "rhi/VulkanBuffer.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <span>
#include <string>
#include <vector>

namespace ve::renderer {

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 uv;
    glm::vec3 normal;
    glm::vec4 tangent;
};

struct MeshPrimitive {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = 0;
    // Half-open range inside Mesh::lods(). lodCount is at least 1 for any
    // primitive that produced geometry, because level 0 always exists.
    uint32_t lodBase = 0;
    uint32_t lodCount = 0;
};

struct GltfTextureInfo {
    std::string debugName;
    std::filesystem::path path;
    std::vector<uint8_t> encodedData;
    bool embedded = false;
};

struct GltfMaterialInfo {
    std::string debugName;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    glm::vec3 emissiveFactor = glm::vec3(0.0f);
    float metallic = 1.0f;
    float roughness = 1.0f;
    int baseColorTextureIndex = -1;
    int normalTextureIndex = -1;
    int metallicRoughnessTextureIndex = -1;
    int emissiveTextureIndex = -1;
    bool fallback = false;
};

struct GltfNodeMeshInstance {
    uint32_t meshIndex = 0;
    glm::mat4 transform{1.0f};
    std::string debugName;
};

struct LoadedGltfAsset;

[[nodiscard]] VkVertexInputBindingDescription vertexBindingDescription();
[[nodiscard]] std::array<VkVertexInputAttributeDescription, 5> vertexAttributeDescriptions();

class Mesh final {
public:
    Mesh() = default;
    ~Mesh() = default;

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;

    [[nodiscard]] static Mesh createCube(
        rhi::VulkanContext& context,
        const rhi::VulkanCommandContext& commandContext);

    [[nodiscard]] static Mesh createUvSphere(
        rhi::VulkanContext& context,
        const rhi::VulkanCommandContext& commandContext,
        uint32_t segments = 48,
        uint32_t rings = 24);

    [[nodiscard]] static LoadedGltfAsset createFromGltf(
        rhi::VulkanContext& context,
        const rhi::VulkanCommandContext& commandContext,
        const std::filesystem::path& path);

    [[nodiscard]] VkBuffer vertexBuffer() const { return vertexBuffer_.buffer(); }
    [[nodiscard]] VkBuffer indexBuffer() const { return indexBuffer_.buffer(); }
    // Index count of LOD 0, i.e. the authored geometry. Simplified levels live
    // past it in the same buffer and are addressed through lods().
    [[nodiscard]] uint32_t indexCount() const { return indexCount_; }
    [[nodiscard]] std::span<const MeshPrimitive> primitives() const { return subMeshes_; }
    [[nodiscard]] bool hasSubMeshes() const { return !subMeshes_.empty(); }

    // Flat LOD table. Sub-meshed meshes address it through MeshPrimitive::lodBase
    // / lodCount; meshes without sub-meshes use lodBase() / lodCount() below.
    [[nodiscard]] std::span<const MeshLod> lods() const { return lods_; }
    [[nodiscard]] uint32_t lodBase() const { return lodBase_; }
    [[nodiscard]] uint32_t lodCount() const { return lodCount_; }
    [[nodiscard]] const Aabb& localBounds() const { return localBounds_; }
    [[nodiscard]] const std::string& debugName() const { return debugName_; }
    [[nodiscard]] bool valid() const { return vertexBuffer_.buffer() != VK_NULL_HANDLE && indexCount_ > 0; }

private:
    // Mesh owns the GPU-local buffers for one drawable piece of geometry.
    rhi::VulkanBuffer vertexBuffer_;
    rhi::VulkanBuffer indexBuffer_;
    uint32_t indexCount_ = 0;
    std::vector<MeshPrimitive> subMeshes_;
    std::vector<MeshLod> lods_;
    // LOD range for the whole-mesh path; unused when subMeshes_ is populated.
    uint32_t lodBase_ = 0;
    uint32_t lodCount_ = 0;
    Aabb localBounds_{};
    std::string debugName_;
};

struct LoadedGltfAsset {
    std::vector<Mesh> meshes;
    std::vector<GltfNodeMeshInstance> nodeMeshInstances;
    std::vector<GltfMaterialInfo> materials;
    std::vector<GltfTextureInfo> textures;
};

} // namespace ve::renderer
