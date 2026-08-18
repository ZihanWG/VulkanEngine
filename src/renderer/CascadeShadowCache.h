#pragma once

// GPU-free content hashing for shadow-pass caching.
//
// Two things live here: the FNV-1a hasher both shadow paths share, and the
// cascaded-shadow-map key built on top of it. Neither touches Vulkan, so both
// are unit-testable on the CPU the same way ClusterGrid.h, CascadeMath.h and
// MeshLod.h are.
//
// Caching a shadow render is only safe if nothing that could change its
// contents has changed, and the failure mode of getting that wrong is a stale
// shadow -- which shows up far from its cause and reads as a rendering bug
// rather than as a cache bug. So instead of tracking individual dirty flags and
// hoping the list stays complete, each pass hashes its actual inputs and
// re-renders whenever the hash moves.
//
// That concentrates the risk: a forgotten input is a correctness bug either
// way, but here it is one bug in one place rather than one per input that can
// change. Floats are hashed by exact bit pattern, so this is a strict
// "identical inputs" test and never an approximate one.

#include <cstddef>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <span>

namespace ve::renderer {

class ShadowCacheKey {
public:
    void reset()
    {
        // FNV-1a 64-bit offset basis.
        hash_ = 1469598103934665603ULL;
    }

    void addBytes(const void* data, size_t size);

    void add(float value)
    {
        addBytes(&value, sizeof(value));
    }
    void add(uint32_t value)
    {
        addBytes(&value, sizeof(value));
    }
    void add(const glm::mat4& value)
    {
        addBytes(&value, sizeof(value));
    }
    // Pointer identity stands in for "which mesh", which is what the draw
    // actually selects. Within one scene that is exact: an index range names
    // geometry in that mesh's own buffer, so a reallocation reusing an address
    // would have changed the range hashed alongside it. Across a scene switch it
    // is not -- the buffer behind the address is replaced wholesale -- which is
    // why Renderer::resetSceneState drops the caches outright rather than
    // trusting this.
    void add(const void* pointer)
    {
        addBytes(&pointer, sizeof(pointer));
    }

    [[nodiscard]] uint64_t value() const
    {
        return hash_;
    }

private:
    uint64_t hash_ = 1469598103934665603ULL;
};

// One caster exactly as the cascade will rasterize it.
//
// `lodLevel` is the level the *GPU* cull would select for this item, mirrored on
// the CPU. It has to be here rather than being implied by firstIndex/indexCount:
// on the GPU-culled path those CPU fields are not what gets drawn, because the
// cull dispatch picks a level from the projected screen radius and writes it
// straight into the indirect command. A camera move that changes only the
// selected level would otherwise leave this key unchanged and freeze a cascade
// at the wrong detail.
struct CascadeShadowCaster {
    const void* mesh = nullptr;
    const void* material = nullptr;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t lodLevel = 0;
    // Which shadow pipeline the item binds. Opaque and cutout casters draw the
    // same geometry through different fragment shaders.
    uint32_t bucket = 0;
    // Negative sentinel when the item is not alpha tested. Hashed because it is
    // runtime-editable from the material inspector and changes the silhouette
    // without changing anything else here.
    float alphaCutoff = -1.0f;
    glm::mat4 modelMatrix{1.0f};
};

// Everything about the pass itself that a cascade's image depends on.
struct CascadeShadowPassState {
    glm::mat4 lightViewProjection{1.0f};
    // Raster state the tile is rendered with.
    float rasterDepthBiasConstantFactor = 0.0f;
    float rasterDepthBiasSlopeFactor = 0.0f;
    // Changes the texel grid the cascade is rendered into.
    uint32_t shadowResolution = 0;
    // Whether the drawn index ranges come from the GPU cull's LOD selection or
    // straight from the CPU draw items. Hashed so that toggling the path
    // invalidates rather than silently reinterpreting every caster's lodLevel.
    bool gpuLodSelectionActive = false;
};

// Content hash of one cascade: the pass state plus every caster that cascade
// actually draws, in order.
//
// The caster count is folded in explicitly so that a shorter list can never
// hash to the same value as a longer one that happens to start with it.
[[nodiscard]] uint64_t computeCascadeShadowKey(const CascadeShadowPassState& state,
                                               std::span<const CascadeShadowCaster> casters);

} // namespace ve::renderer
