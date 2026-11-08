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
    // This primitive's own extent, in mesh-local space.
    //
    // The mesh-wide bounds are the union of these, which is all a whole-mesh
    // draw needs. A scene that imports one RenderObject per primitive needs them
    // apart: an object culled against the union of an entire building is an
    // object that is never culled. Costs nothing to record -- the import already
    // computed each primitive's extent on its way to the union.
    //
    // Last in the struct so the positional initialisers that predate it still
    // mean what they say. That is a readability choice, not a compatibility one:
    // the cooked .vemesh stores this struct verbatim and MeshCacheExpectation
    // carries sizeof(MeshPrimitive) as primitiveStride, so growing it invalidates
    // every existing cook rather than reading one at the wrong stride.
    Aabb localBounds{};
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

// One KHR_lights_punctual light, placed by the node that references it. Kept in
// glTF's own terms -- candela for point and spot, lux for directional, a range of
// 0 meaning unbounded -- because what those units mean in this renderer is a
// per-scene decision the importer has no basis for. See importedLightRange.
struct GltfPunctualLight {
    enum class Type : uint8_t {
        Point,
        Spot,
        Directional,
    };

    Type type = Type::Point;
    std::string debugName;
    glm::vec3 position{0.0f};
    // Where the light points: the node's local -Z, as the extension defines it.
    // Normalised; meaningless for a point light, which still carries it.
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
    float range = 0.0f;
    // Half-angles from the axis, in radians -- the same convention as
    // ClusteredLighting::addSpotLight, so they pass through unchanged.
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7853981634f;
};

// Where a glTF perspective camera stands and looks, placed by its node. Only the
// pose and the vertical field of view: the clip planes an authoring tool writes
// (Blender's 1 mm near plane, say) are a poor fit for a depth buffer sized to the
// scene, so the renderer keeps deriving those from the scene bounds.
struct GltfCameraPose {
    std::string debugName;
    glm::vec3 position{0.0f};
    // The node's local -Z and +Y, as the glTF camera model defines them.
    glm::vec3 forward{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float verticalFovRadians = 0.0f;
};

// The range the clustered light cull uses for an imported light. An authored
// range wins. An unbounded one (0) is cut where inverse-square illuminance falls
// to `cutoffIlluminance`, since a light with no range touches every froxel and the
// cull would put it in every list. Clamped to `maxRange` so one very bright light
// cannot do the same thing by being bright instead of unbounded.
[[nodiscard]] float importedLightRange(float intensity, float authoredRange, float cutoffIlluminance, float maxRange);

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
    // Every level's meshlets, and which of them each level owns (parallel to
    // `lods`). Both empty on a mesh that was not meshletized, which consumers
    // read as "draw each level whole".
    std::vector<Meshlet> meshlets;
    std::vector<glm::uvec2> meshletRangesPerLod;
    Aabb localBounds{};
};

struct GltfGeometry {
    std::vector<CpuMeshData> meshes;
    std::vector<GltfNodeMeshInstance> nodeMeshInstances;
    std::vector<GltfMaterialInfo> materials;
    std::vector<GltfTextureInfo> textures;
    // Read from the glTF on every load, cooked or not: a .vemesh replaces mesh
    // assembly only, and the scene graph that places these is always parsed.
    std::vector<GltfPunctualLight> lights;
    // Perspective cameras, in scene-graph order. Orthographic ones are skipped.
    std::vector<GltfCameraPose> cameras;
};

// Parses `path`, assembles vertices and indices, and builds every primitive's LOD
// chain. `jobSystem` spreads the simplification across the pool; nullptr keeps it
// serial and produces a byte-identical index buffer either way, because only the
// serial append decides layout.
//
// `cookedMeshes`, when non-null, supplies geometry that was built earlier and
// skips assembly and simplification entirely -- ~333 ms of a ~349 ms Sponza
// import. The glTF is still parsed either way: materials, textures and node
// transforms come from it and cost ~2 ms, so cooking them too would buy nothing
// and add a second staleness surface. The vector is consumed.
//
// Must not be called from a JobSystem worker. Throws std::runtime_error when the
// file cannot be read, holds no supported triangle geometry, or (with
// `cookedMeshes`) disagrees with the cooked data about how many meshes it has.
// `buildMeshletTable` groups each LOD level's triangles into meshlets. Off by
// default because it REORDERS triangles inside each level -- a permutation, so
// the geometry is identical, but not the rasterization order. See
// RendererStartupOverrides::buildMeshlets.
//
// `lodBuildSettings` decides what the LOD chains contain, normal weight
// included; vemeshcook cooks with the defaults, and the cook's fingerprint
// records them.
[[nodiscard]] GltfGeometry loadGltfGeometry(const std::filesystem::path& path,
                                            JobSystem* jobSystem = nullptr,
                                            std::vector<CpuMeshData>* cookedMeshes = nullptr,
                                            bool buildMeshletTable = false,
                                            const LodBuildSettings& lodBuildSettings = {});

} // namespace ve::renderer
