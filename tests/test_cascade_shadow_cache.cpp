// Cascaded-shadow-map cache key.
//
// The key is what decides whether a cascade is redrawn, so the interesting
// assertions are not "the hash is stable" but "every input that changes the
// image changes the key". A missed input does not fail loudly: it freezes a
// cascade at stale content, which reads as a shading bug far from its cause.
// Each case below therefore isolates one input and asserts the key moves.

#include "renderer/CascadeShadowCache.h"

#include <catch2/catch_test_macros.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <vector>

using ve::renderer::CascadeShadowCaster;
using ve::renderer::CascadeShadowPassState;
using ve::renderer::computeCascadeShadowKey;
using ve::renderer::ShadowCacheKey;

namespace {

// Two distinct addresses to stand in for mesh/material identity. Their values
// are never dereferenced.
int meshStorageA = 0;
int meshStorageB = 0;
int materialStorageA = 0;
int materialStorageB = 0;

CascadeShadowPassState makeState()
{
    CascadeShadowPassState state{};
    state.lightViewProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 100.0f) *
                                glm::lookAt(glm::vec3(5.0f, 10.0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    state.rasterDepthBiasConstantFactor = 1.25f;
    state.rasterDepthBiasSlopeFactor = 1.75f;
    state.shadowResolution = 2048;
    state.gpuLodSelectionActive = true;
    return state;
}

std::vector<CascadeShadowCaster> makeCasters()
{
    CascadeShadowCaster first{};
    first.mesh = &meshStorageA;
    first.material = &materialStorageA;
    first.firstIndex = 0;
    first.indexCount = 300;
    first.lodLevel = 0;
    first.bucket = 0;
    first.alphaCutoff = -1.0f;
    first.modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    CascadeShadowCaster second{};
    second.mesh = &meshStorageB;
    second.material = &materialStorageB;
    second.firstIndex = 300;
    second.indexCount = 150;
    second.lodLevel = 1;
    second.bucket = 1;
    second.alphaCutoff = 0.5f;
    second.modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f));

    return {first, second};
}

uint64_t keyOf(const CascadeShadowPassState& state, const std::vector<CascadeShadowCaster>& casters)
{
    return computeCascadeShadowKey(state, std::span<const CascadeShadowCaster>(casters.data(), casters.size()));
}

} // namespace

TEST_CASE("Identical cascade inputs hash identically", "[cascade-shadow-cache]")
{
    const CascadeShadowPassState state = makeState();
    const std::vector<CascadeShadowCaster> casters = makeCasters();

    REQUIRE(keyOf(state, casters) == keyOf(state, casters));

    // A freshly constructed key must equal an explicitly reset one, or the
    // first frame's hash would depend on construction order.
    ShadowCacheKey fresh;
    ShadowCacheKey resetKey;
    resetKey.reset();
    REQUIRE(fresh.value() == resetKey.value());
}

TEST_CASE("Every pass-state input moves the cascade key", "[cascade-shadow-cache]")
{
    const CascadeShadowPassState base = makeState();
    const std::vector<CascadeShadowCaster> casters = makeCasters();
    const uint64_t baseKey = keyOf(base, casters);

    SECTION("light view-projection")
    {
        CascadeShadowPassState state = base;
        state.lightViewProjection[3][0] += 0.001f;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("raster depth bias constant factor")
    {
        CascadeShadowPassState state = base;
        state.rasterDepthBiasConstantFactor += 0.01f;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("raster depth bias slope factor")
    {
        CascadeShadowPassState state = base;
        state.rasterDepthBiasSlopeFactor += 0.01f;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("shadow resolution")
    {
        CascadeShadowPassState state = base;
        state.shadowResolution = 1024;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("GPU LOD selection toggle")
    {
        CascadeShadowPassState state = base;
        state.gpuLodSelectionActive = !state.gpuLodSelectionActive;
        REQUIRE(keyOf(state, casters) != baseKey);
    }
}

TEST_CASE("Every per-caster input moves the cascade key", "[cascade-shadow-cache]")
{
    const CascadeShadowPassState state = makeState();
    const std::vector<CascadeShadowCaster> base = makeCasters();
    const uint64_t baseKey = keyOf(state, base);

    SECTION("mesh identity")
    {
        std::vector<CascadeShadowCaster> casters = base;
        casters[0].mesh = &meshStorageB;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("material identity")
    {
        std::vector<CascadeShadowCaster> casters = base;
        casters[1].material = &materialStorageA;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("index range")
    {
        std::vector<CascadeShadowCaster> casters = base;
        casters[0].firstIndex += 3;
        REQUIRE(keyOf(state, casters) != baseKey);

        casters = base;
        casters[0].indexCount -= 3;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("selected LOD level")
    {
        // The GPU cull picks this from the projected screen radius, so it can
        // move with nothing else in the caster changing.
        std::vector<CascadeShadowCaster> casters = base;
        casters[0].lodLevel = 2;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("render bucket")
    {
        std::vector<CascadeShadowCaster> casters = base;
        casters[0].bucket = 1;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("alpha cutoff")
    {
        std::vector<CascadeShadowCaster> casters = base;
        casters[1].alphaCutoff = 0.25f;
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("model matrix")
    {
        std::vector<CascadeShadowCaster> casters = base;
        casters[1].modelMatrix = glm::translate(casters[1].modelMatrix, glm::vec3(0.0f, 0.0f, 0.001f));
        REQUIRE(keyOf(state, casters) != baseKey);
    }
}

TEST_CASE("Caster list structure is part of the cascade key", "[cascade-shadow-cache]")
{
    const CascadeShadowPassState state = makeState();
    const std::vector<CascadeShadowCaster> base = makeCasters();
    const uint64_t baseKey = keyOf(state, base);

    SECTION("a caster leaving the cascade")
    {
        // This is the case per-cascade caching exists for: an object leaving one
        // cascade's frustum must dirty that cascade even though every remaining
        // caster is untouched.
        std::vector<CascadeShadowCaster> casters = base;
        casters.pop_back();
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("an empty cascade differs from a populated one")
    {
        const std::vector<CascadeShadowCaster> empty;
        REQUIRE(keyOf(state, empty) != baseKey);
    }

    SECTION("draw order")
    {
        // Shadow depth is order-independent in the image, but order is also the
        // cheapest proxy for "the batching changed", and re-sorting is not free
        // to assume equivalent. Hashing it is the conservative direction.
        std::vector<CascadeShadowCaster> casters = {base[1], base[0]};
        REQUIRE(keyOf(state, casters) != baseKey);
    }

    SECTION("a prefix cannot collide with the longer list")
    {
        std::vector<CascadeShadowCaster> extended = base;
        extended.push_back(base[0]);
        REQUIRE(keyOf(state, extended) != baseKey);
    }
}

TEST_CASE("Cascades with different fits do not share a key", "[cascade-shadow-cache]")
{
    // Four cascades over the same casters differ only in their fitted matrix.
    // If that did not separate them, a cached cascade could be mistaken for a
    // different one and never redraw.
    const std::vector<CascadeShadowCaster> casters = makeCasters();
    std::array<uint64_t, 4> keys{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < keys.size(); ++cascadeIndex) {
        CascadeShadowPassState state = makeState();
        const float extent = 10.0f * static_cast<float>(cascadeIndex + 1);
        state.lightViewProjection =
            glm::ortho(-extent, extent, -extent, extent, 0.1f, 100.0f) *
            glm::lookAt(glm::vec3(5.0f, 10.0f, 5.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        keys[cascadeIndex] = keyOf(state, casters);
    }

    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            REQUIRE(keys[i] != keys[j]);
        }
    }
}
