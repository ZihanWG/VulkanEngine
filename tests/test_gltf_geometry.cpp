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
