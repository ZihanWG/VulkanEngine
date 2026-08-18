#pragma once

// GPU-free glTF geometry loading: parse, vertex/index assembly, and LOD chain
// construction, with no Vulkan anywhere.
//
// Split out of Mesh.cpp so the same work can run offline. A measurement is the
// reason: on Sponza, glTF import is ~349 ms of which ~296 ms is LOD
// simplification and ~35 ms is vertex assembly, against ~2 ms of actual JSON
// parsing. Baking that into a cooked file removes nearly all of it, and nothing
// could cook it while it lived behind a signature that takes a VulkanContext.
//
// The same split as renderer/MeshLod.h and renderer/BrdfIntegration.h: the
// expensive, testable half has no device, and the Vulkan half is only the buffer
// upload (Mesh::createFromGeometry).

#include "renderer/Bounds.h"
#include "renderer/MeshLod.h"

#include <cstdint>
#include <filesystem>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <vector>

namespace ve {
class JobSystem;
}

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
    // glTF alphaMode / alphaCutoff / doubleSided, kept in the source spelling so
    // Material can round-trip them. They decide the render bucket, so importing a
    // cutout or transparent glTF without them would silently render it opaque.
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
    bool fallback = false;
};

struct GltfNodeMeshInstance {
    uint32_t meshIndex = 0;
    glm::mat4 transform{1.0f};
    std::string debugName;
};

// One mesh's geometry before it reaches a device. This is exactly what a cooked
// mesh file has to store, and exactly what Mesh::createFromGeometry uploads.
struct CpuMeshData {
    std::string debugName;
    std::vector<Vertex> vertices;
    // Authored indices first, then every simplified LOD level appended after
    // them, which is why indexCount (level 0's total) is stored separately.
    std::vector<uint32_t> indices;
    uint32_t indexCount = 0;
    std::vector<MeshPrimitive> primitives;
    std::vector<MeshLod> lods;
    Aabb localBounds{};
};

struct GltfGeometry {
    std::vector<CpuMeshData> meshes;
    std::vector<GltfNodeMeshInstance> nodeMeshInstances;
    std::vector<GltfMaterialInfo> materials;
    std::vector<GltfTextureInfo> textures;
};

// Parses `path`, assembles vertices and indices, and builds every primitive's LOD
// chain. `jobSystem` spreads the simplification across the pool; nullptr keeps it
// serial and produces a byte-identical index buffer either way, because only the
// serial append decides layout.
//
// Must not be called from a JobSystem worker. Throws std::runtime_error when the
// file cannot be read or holds no supported triangle geometry.
[[nodiscard]] GltfGeometry loadGltfGeometry(const std::filesystem::path& path, JobSystem* jobSystem = nullptr);

} // namespace ve::renderer
