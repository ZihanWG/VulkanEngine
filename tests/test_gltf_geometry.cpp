// Per-primitive bounds from the GPU-free glTF importer.
//
// A glTF mesh holding many primitives used to yield one set of bounds -- the
// union of all of them. That is everything a whole-mesh draw needs, and it is
// wrong for a scene that imports one RenderObject per primitive: an object
// culled against the union of every primitive in the file is an object the
// frustum never rejects, and its projected size, which is what selects a LOD, is
// the size of the whole file. Sponza showed both symptoms at once -- 0 of 103
// draw items culled and every one of them pinned to LOD 0.

#include "renderer/GltfGeometry.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

using Catch::Approx;
using ve::renderer::CpuMeshData;
using ve::renderer::GltfGeometry;
using ve::renderer::loadGltfGeometry;
using ve::renderer::MeshPrimitive;

namespace {

// Two triangles ten units apart, one mesh, one node. Authored for this test
// because every other model in assets/ has a single primitive, and a
// single-primitive mesh cannot tell per-primitive bounds apart from mesh-wide
// ones -- they are equal. The bug this covers is invisible on such a fixture.
std::filesystem::path fixturePath()
{
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR) / "models" / "two_primitive_bounds.gltf";
}

} // namespace

TEST_CASE("Each glTF primitive carries its own bounds", "[gltf][geometry]")
{
    const GltfGeometry geometry = loadGltfGeometry(fixturePath());
    REQUIRE(geometry.meshes.size() == 1);

    const CpuMeshData& mesh = geometry.meshes.front();
    REQUIRE(mesh.primitives.size() == 2);

    // The mesh spans both triangles.
    CHECK(mesh.localBounds.min.x == Approx(0.0f));
    CHECK(mesh.localBounds.max.x == Approx(11.0f));

    // Each primitive spans only its own, which is the whole point: if these
    // reported the mesh's extent, both would read 0..11 and pass a test that only
    // checked validity.
    CHECK(mesh.primitives[0].localBounds.min.x == Approx(0.0f));
    CHECK(mesh.primitives[0].localBounds.max.x == Approx(1.0f));
    CHECK(mesh.primitives[1].localBounds.min.x == Approx(10.0f));
    CHECK(mesh.primitives[1].localBounds.max.x == Approx(11.0f));
}

TEST_CASE("Primitive bounds are filled in whether or not the accessor carries min/max", "[gltf][geometry]")
{
    // The fixture's first POSITION accessor declares min/max and its second does
    // not, so one primitive takes the declared-bounds path and the other the
    // per-vertex fallback. Both have to end up with real bounds: the fallback
    // used to expand only the mesh-wide box, which left such a primitive with an
    // empty one -- and an empty box reads as "cull me" or "bound me at infinity"
    // depending on the consumer, neither of which is the geometry.
    const GltfGeometry geometry = loadGltfGeometry(fixturePath());
    REQUIRE(geometry.meshes.size() == 1);

    for (const MeshPrimitive& primitive : geometry.meshes.front().primitives) {
        CHECK(primitive.localBounds.valid());
        CHECK(primitive.localBounds.max.x > primitive.localBounds.min.x);
        CHECK(primitive.localBounds.max.y > primitive.localBounds.min.y);
    }
}

TEST_CASE("The mesh bounds are the union of the primitive bounds", "[gltf][geometry]")
{
    // The invariant that keeps the two consistent. A whole-mesh object and the
    // set of per-primitive objects built from the same mesh must occupy the same
    // space, or switching between them would move geometry.
    const GltfGeometry geometry = loadGltfGeometry(fixturePath());
    REQUIRE(geometry.meshes.size() == 1);

    const CpuMeshData& mesh = geometry.meshes.front();
    ve::renderer::Aabb united{};
    for (const MeshPrimitive& primitive : mesh.primitives) {
        united.merge(primitive.localBounds);
    }

    CHECK(united.min.x == Approx(mesh.localBounds.min.x));
    CHECK(united.min.y == Approx(mesh.localBounds.min.y));
    CHECK(united.min.z == Approx(mesh.localBounds.min.z));
    CHECK(united.max.x == Approx(mesh.localBounds.max.x));
    CHECK(united.max.y == Approx(mesh.localBounds.max.y));
    CHECK(united.max.z == Approx(mesh.localBounds.max.z));
}

namespace {

std::filesystem::path lightsFixturePath()
{
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR) / "models" / "punctual_lights.gltf";
}

const ve::renderer::GltfPunctualLight* findLight(const GltfGeometry& geometry, const std::string& name)
{
    for (const ve::renderer::GltfPunctualLight& light : geometry.lights) {
        if (light.debugName == name) {
            return &light;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("KHR_lights_punctual lights are placed by their node's world transform", "[gltf][lights]")
{
    using Type = ve::renderer::GltfPunctualLight::Type;
    const GltfGeometry geometry = loadGltfGeometry(lightsFixturePath());
    REQUIRE(geometry.lights.size() == 3);

    // Both children of a node translated by +5 on X: a light that ignored its
    // parent would sit at x = 0 and still look plausible.
    const ve::renderer::GltfPunctualLight* lamp = findLight(geometry, "Lamp");
    REQUIRE(lamp != nullptr);
    CHECK(lamp->type == Type::Point);
    CHECK(lamp->position.x == Approx(5.0f));
    CHECK(lamp->position.y == Approx(2.0f));
    CHECK(lamp->color.g == Approx(0.5f));
    CHECK(lamp->intensity == Approx(40.0f));
    CHECK(lamp->range == Approx(0.0f));

    // Rotated -90 degrees about X, so local -Z ends up pointing straight down.
    const ve::renderer::GltfPunctualLight* cone = findLight(geometry, "Cone");
    REQUIRE(cone != nullptr);
    CHECK(cone->type == Type::Spot);
    CHECK(cone->position.x == Approx(5.0f));
    CHECK(cone->position.y == Approx(4.0f));
    CHECK(cone->direction.x == Approx(0.0f).margin(1e-5));
    CHECK(cone->direction.y == Approx(-1.0f));
    CHECK(cone->direction.z == Approx(0.0f).margin(1e-5));
    CHECK(cone->range == Approx(12.0f));
    CHECK(cone->innerConeAngle == Approx(0.2f));
    CHECK(cone->outerConeAngle == Approx(0.5f));

    // Unrotated: the extension's default direction, local -Z.
    const ve::renderer::GltfPunctualLight* sun = findLight(geometry, "Sun");
    REQUIRE(sun != nullptr);
    CHECK(sun->type == Type::Directional);
    CHECK(sun->direction.z == Approx(-1.0f));
}

TEST_CASE("A glTF without KHR_lights_punctual imports no lights", "[gltf][lights]")
{
    CHECK(loadGltfGeometry(fixturePath()).lights.empty());
}

TEST_CASE("An imported light's cull range", "[gltf][lights]")
{
    using ve::renderer::importedLightRange;
    // Authored ranges are kept, up to the cap.
    CHECK(importedLightRange(100.0f, 12.0f, 0.01f, 50.0f) == Approx(12.0f));
    CHECK(importedLightRange(100.0f, 80.0f, 0.01f, 50.0f) == Approx(50.0f));
    // Unbounded: cut where intensity / d^2 reaches the cutoff -- 100 / 0.04 = 50^2.
    CHECK(importedLightRange(100.0f, 0.0f, 0.04f, 1000.0f) == Approx(50.0f));
    CHECK(importedLightRange(1.0e6f, 0.0f, 0.01f, 40.0f) == Approx(40.0f));
    // A dark light reaches nothing.
    CHECK(importedLightRange(0.0f, 0.0f, 0.01f, 40.0f) == Approx(0.0f));
}

TEST_CASE("A glTF camera is placed by its node, scale and all", "[gltf][camera]")
{
    // Under the +5 X rig, at +10 Z, with a 0.01 scale on its own node -- the shape
    // an FBX conversion produces. A pose that read the scaled axes as unit vectors
    // would come back a hundredth of its length.
    const GltfGeometry geometry = loadGltfGeometry(lightsFixturePath());
    REQUIRE(geometry.cameras.size() == 1);
    const ve::renderer::GltfCameraPose& camera = geometry.cameras.front();
    CHECK(camera.debugName == "Shot");
    CHECK(camera.position.x == Approx(5.0f));
    CHECK(camera.position.z == Approx(10.0f));
    CHECK(camera.forward.z == Approx(-1.0f));
    CHECK(camera.up.y == Approx(1.0f));
    CHECK(camera.verticalFovRadians == Approx(0.6f));
    CHECK(loadGltfGeometry(fixturePath()).cameras.empty());
}
