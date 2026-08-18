#include "renderer/Mesh.h"

#include "core/Logger.h"
#include "renderer/MeshCache.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

// The tinygltf implementation lives in GltfSkinnedImport.cpp (VulkanEngineCore);
// here we only need the declarations.
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
#include <fstream>
#include <future>
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

// Returns empty on any problem: a missing cook is a normal miss, and
// meshCacheStatus() reports an empty blob as Missing.
std::vector<std::byte> readFileBytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }

    const std::streamsize size = input.tellg();
    if (size <= 0) {
        return {};
    }

    input.seekg(0);
    std::vector<std::byte> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) {
        return {};
    }

    return bytes;
}

constexpr float kPi = 3.14159265358979323846f;


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

    mesh.indexCount_ = static_cast<uint32_t>(kCubeIndices.size());

    // 12 triangles is far under kMinLodIndexCount, so this produces a level-0-only
    // chain. It still runs so every valid mesh has a LOD table and the cull shader
    // never has to special-case a missing one.
    std::vector<uint32_t> indices(kCubeIndices.begin(), kCubeIndices.end());
    mesh.lods_ = buildLodChain(indices,
                               0,
                               mesh.indexCount_,
                               &kCubeVertices[0].position.x,
                               kCubeVertices.size(),
                               sizeof(Vertex),
                               mesh.debugName_);
    mesh.lodBase_ = 0;
    mesh.lodCount_ = static_cast<uint32_t>(mesh.lods_.size());

    mesh.indexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const uint32_t>(indices.data(), indices.size())),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    return mesh;
}

Mesh Mesh::createUvSphere(
    rhi::VulkanContext& context,
    const rhi::VulkanCommandContext& commandContext,
    uint32_t segments,
    uint32_t rings)
{
    segments = std::max(segments, 8U);
    rings = std::max(rings, 4U);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(static_cast<size_t>(segments + 1U) * static_cast<size_t>(rings + 1U));
    indices.reserve(static_cast<size_t>(segments) * static_cast<size_t>(rings - 1U) * 6U);

    for (uint32_t ring = 0; ring <= rings; ++ring) {
        const float v = static_cast<float>(ring) / static_cast<float>(rings);
        const float phi = v * kPi;
        const float y = std::cos(phi) * 0.5f;
        const float radius = std::sin(phi) * 0.5f;

        for (uint32_t segment = 0; segment <= segments; ++segment) {
            const float u = static_cast<float>(segment) / static_cast<float>(segments);
            const float theta = u * kPi * 2.0f;
            const glm::vec3 normal = glm::normalize(glm::vec3{
                std::cos(theta) * radius,
                y,
                std::sin(theta) * radius,
            });
            const glm::vec3 tangent = glm::normalize(glm::vec3{-std::sin(theta), 0.0f, std::cos(theta)});

            Vertex vertex{};
            vertex.position = normal * 0.5f;
            vertex.color = glm::vec3(1.0f);
            vertex.uv = {u, 1.0f - v};
            vertex.normal = normal;
            vertex.tangent = {tangent, 1.0f};
            vertices.push_back(vertex);
        }
    }

    const uint32_t rowStride = segments + 1U;
    for (uint32_t ring = 0; ring < rings; ++ring) {
        for (uint32_t segment = 0; segment < segments; ++segment) {
            const uint32_t a = ring * rowStride + segment;
            const uint32_t b = (ring + 1U) * rowStride + segment;
            const uint32_t c = b + 1U;
            const uint32_t d = a + 1U;

            if (ring > 0) {
                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(d);
            }
            if (ring + 1U < rings) {
                indices.push_back(d);
                indices.push_back(b);
                indices.push_back(c);
            }
        }
    }

    Mesh mesh;
    mesh.debugName_ = "Built-in UV Sphere Mesh";
    for (const Vertex& vertex : vertices) {
        mesh.localBounds_.expand(vertex.position);
    }

    mesh.vertexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const Vertex>(vertices.data(), vertices.size())),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    mesh.indexCount_ = static_cast<uint32_t>(indices.size());
    mesh.lods_ = buildLodChain(indices,
                               0,
                               mesh.indexCount_,
                               &vertices[0].position.x,
                               vertices.size(),
                               sizeof(Vertex),
                               mesh.debugName_);
    mesh.lodBase_ = 0;
    mesh.lodCount_ = static_cast<uint32_t>(mesh.lods_.size());

    mesh.indexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const uint32_t>(indices.data(), indices.size())),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    rhi::debug::setObjectName(
        context.vkDevice(), mesh.vertexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, "PortfolioSphereVertexBuffer");
    rhi::debug::setObjectName(
        context.vkDevice(), mesh.indexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, "PortfolioSphereIndexBuffer");
    return mesh;
}

Mesh Mesh::createFromGeometry(rhi::VulkanContext& context,
                              const rhi::VulkanCommandContext& commandContext,
                              CpuMeshData&& geometry,
                              std::string_view debugNamePrefix)
{
    Mesh mesh;
    mesh.debugName_ = std::move(geometry.debugName);
    mesh.indexCount_ = geometry.indexCount;
    mesh.subMeshes_ = std::move(geometry.primitives);
    mesh.lods_ = std::move(geometry.lods);
    mesh.localBounds_ = geometry.localBounds;

    mesh.vertexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const Vertex>(geometry.vertices.data(), geometry.vertices.size())),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.indexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const uint32_t>(geometry.indices.data(), geometry.indices.size())),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    const std::string debugName(debugNamePrefix);
    rhi::debug::setObjectName(
        context.vkDevice(), mesh.vertexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, debugName + "VertexBuffer");
    rhi::debug::setObjectName(
        context.vkDevice(), mesh.indexBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, debugName + "IndexBuffer");

    return mesh;
}

LoadedGltfAsset Mesh::createFromGltf(rhi::VulkanContext& context,
                                     const rhi::VulkanCommandContext& commandContext,
                                     const std::filesystem::path& path,
                                     JobSystem* jobSystem)
{
    // A cooked sidecar removes assembly and LOD construction -- ~333 ms of a
    // ~349 ms Sponza import. Any reason it does not match falls back to the glTF
    // and says why: "your cook is stale, re-run it" and an unexplained slow
    // startup must not look the same in a log.
    std::vector<CpuMeshData> cookedMeshes;
    bool haveCooked = false;

    MeshCacheExpectation expectation{};
    if (makeMeshCacheExpectation(path, expectation)) {
        const std::filesystem::path cookedPath = meshCacheSidecarPath(path);
        const std::vector<std::byte> blob = readFileBytes(cookedPath);
        const MeshCacheStatus status = meshCacheStatus(blob, expectation);
        if (status == MeshCacheStatus::Usable) {
            try {
                cookedMeshes = readMeshCache(blob);
                haveCooked = true;
                Logger::info("Using cooked mesh geometry: " + cookedPath.string());
            } catch (const std::exception& error) {
                Logger::warn("Cooked mesh geometry '" + cookedPath.string()
                             + "' could not be read; loading the glTF instead: " + error.what());
            }
        } else if (status != MeshCacheStatus::Missing) {
            Logger::warn("Ignoring cooked mesh geometry '" + cookedPath.string()
                         + "': " + std::string(meshCacheStatusName(status)) + ". Re-run vemeshcook.");
        }
    }

    // Everything expensive happens without a device; this is only the upload.
    GltfGeometry geometry = loadGltfGeometry(path, jobSystem, haveCooked ? &cookedMeshes : nullptr);

    LoadedGltfAsset loadedAsset{};
    loadedAsset.meshes.reserve(geometry.meshes.size());
    for (size_t meshIndex = 0; meshIndex < geometry.meshes.size(); ++meshIndex) {
        loadedAsset.meshes.push_back(createFromGeometry(context,
                                                        commandContext,
                                                        std::move(geometry.meshes[meshIndex]),
                                                        path.stem().string() + "Mesh" + std::to_string(meshIndex)));
    }

    loadedAsset.nodeMeshInstances = std::move(geometry.nodeMeshInstances);
    loadedAsset.materials = std::move(geometry.materials);
    loadedAsset.textures = std::move(geometry.textures);
    return loadedAsset;
}

} // namespace ve::renderer
