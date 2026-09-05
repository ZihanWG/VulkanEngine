#pragma once

// GPU-free construction of the built-in cube and UV sphere.
//
// Split out of Mesh.cpp for the same reason renderer/GltfGeometry.h was, but
// with a different payoff: not speed, testability. Triangle winding is exactly
// the kind of fact whose failure mode is silence -- every pipeline in this
// renderer rasterised with VK_CULL_MODE_NONE, so a mesh wound the wrong way
// drew identically to one wound correctly, and nothing in the engine or the
// tests could tell them apart.
//
// It was not hypothetical. The UV sphere was wound clockwise-from-outside for
// the life of the renderer, and it only surfaced when back-face culling was
// tried on an immediate-mode GPU (docs/design_decisions.md): the spheres
// rendered their own interiors while every other object was unaffected. Behind
// a VulkanContext that could not be asserted on; here it can, and
// tests/test_primitive_geometry.cpp does.
//
// Convention, stated once because both builders and the test depend on it:
// **counter-clockwise when viewed from outside, under a right-handed cross
// product**, matching VK_FRONT_FACE_COUNTER_CLOCKWISE in VulkanPipeline.cpp.
// For a triangle (v0, v1, v2) that means
// `cross(v1 - v0, v2 - v0)` points away from the surface, i.e. along the
// authored vertex normal rather than against it.

#include "renderer/GltfGeometry.h"

#include <cstdint>
#include <vector>

namespace ve::renderer {

// Vertices and indices of one built-in primitive, before any device sees them.
// Deliberately not CpuMeshData: the LOD chain and bounds are Mesh's job, and
// building them here would drag meshoptimizer into a header that exists to be
// cheap to test.
struct PrimitiveGeometry {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

// A unit cube centred on the origin, 24 vertices so each face carries its own
// normal, UV and tangent.
[[nodiscard]] PrimitiveGeometry buildCubeGeometry();

// A unit-diameter sphere centred on the origin. `segments` and `rings` are
// clamped to the minimums that still close the surface.
[[nodiscard]] PrimitiveGeometry buildUvSphereGeometry(uint32_t segments, uint32_t rings);

} // namespace ve::renderer
