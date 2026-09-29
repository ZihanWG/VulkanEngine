#pragma once

#include "renderer/Bounds.h"
#include "renderer/Transform.h"

#include <glm/mat4x4.hpp>

namespace ve::renderer {

// One render object's model matrix and world AABB, kept together with the inputs
// they were derived from. Composing the matrix from TRS (three sin/cos pairs and
// five mat4 multiplies) and transforming eight corners costs ~75 ns per object,
// and a static object pays it every frame for the same answer.
struct CachedObjectTransform {
    Transform transform{};
    Aabb localBounds{};
    glm::mat4 model{1.0f};
    Aabb worldBounds{};
    bool valid = false;
};

// Brings `entry` up to date for these inputs; returns true when it recomputed.
//
// Keyed on the inputs' values, not on a dirty flag: every place that writes a
// transform (animation, the gizmo, scene edits, scene loads) is covered without
// any of them knowing the cache exists, and a reused slot that now holds a
// different object simply misses. The comparison is bitwise rather than `==`, so
// -0.0 and 0.0 are different keys: a cached result is then always exactly the
// bits a fresh transform.modelMatrix() and localBounds.transform(model) would
// produce, which is what lets the renderer's output stay byte-identical.
bool refreshCachedObjectTransform(CachedObjectTransform& entry, const Transform& transform, const Aabb& localBounds);

} // namespace ve::renderer
