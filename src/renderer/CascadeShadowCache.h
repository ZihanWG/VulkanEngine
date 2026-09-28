#pragma once

// GPU-free content hashing for shadow-pass caching.
//
// Two things live here: the content hasher both shadow paths share, and the
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

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <glm/mat4x4.hpp>
#include <span>

namespace ve::renderer {

// A streaming 64-bit hash over the bytes it is given, in order: adding A then B
// hashes the same as adding the concatenation AB (on the little-endian hosts
// this builds for).
//
// Eight bytes per step, using xxHash64's tail round and final avalanche. It was
// byte-wise FNV-1a, which spends a dependent 64-bit multiply on every byte; a
// caster is 80-100 bytes of mesh, index range and model matrix, and the
// punctual and cascade keys hash thousands of them per frame on --scene
// stress. Values only ever meet other values from the same run -- nothing here
// is persisted -- so the algorithm can change without migrating anything.
class ShadowCacheKey {
public:
    void reset()
    {
        hash_ = kSeed;
        pending_ = 0;
        pendingBytes_ = 0;
        length_ = 0;
    }

    void addBytes(const void* data, size_t size)
    {
        const auto* bytes = static_cast<const unsigned char*>(data);
        length_ += size;
        // Whole words straight from the input. With a partial word pending, each
        // input word completes it and its high bytes become the next partial
        // word, so an unaligned stream still takes one step per eight bytes.
        while (size >= 8) {
            uint64_t word = 0;
            std::memcpy(&word, bytes, sizeof(word));
            if (pendingBytes_ == 0) {
                mixWord(word);
            } else {
                mixWord(pending_ | (word << (8u * pendingBytes_)));
                pending_ = word >> (64u - 8u * pendingBytes_);
            }
            bytes += 8;
            size -= 8;
        }
        for (; size > 0; --size, ++bytes) {
            pending_ |= static_cast<uint64_t>(*bytes) << (8u * pendingBytes_);
            if (++pendingBytes_ == 8) {
                mixWord(pending_);
                pending_ = 0;
                pendingBytes_ = 0;
            }
        }
    }

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
        // The partial word is zero-padded; the length folded in after it is what
        // keeps a stream ending in zero bytes apart from a shorter one.
        uint64_t hash = pendingBytes_ != 0 ? mixed(hash_, pending_) : hash_;
        hash += length_;
        hash ^= hash >> 33;
        hash *= kPrime2;
        hash ^= hash >> 29;
        hash *= kPrime3;
        hash ^= hash >> 32;
        return hash;
    }

private:
    static constexpr uint64_t kPrime1 = 0x9E3779B185EBCA87ULL;
    static constexpr uint64_t kPrime2 = 0xC2B2AE3D27D4EB4FULL;
    static constexpr uint64_t kPrime3 = 0x165667B19E3779F9ULL;
    static constexpr uint64_t kPrime4 = 0x85EBCA77C2B2AE63ULL;
    static constexpr uint64_t kSeed = 0x27D4EB2F165667C5ULL;

    [[nodiscard]] static uint64_t mixed(uint64_t hash, uint64_t word)
    {
        hash ^= std::rotl(word * kPrime2, 31) * kPrime1;
        return std::rotl(hash, 27) * kPrime1 + kPrime4;
    }
    void mixWord(uint64_t word)
    {
        hash_ = mixed(hash_, word);
    }

    uint64_t hash_ = kSeed;
    uint64_t pending_ = 0;
    uint64_t length_ = 0;
    uint32_t pendingBytes_ = 0;
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
    // The skinned caster's pose, or 0 when it does not reach this cascade.
    //
    // It cannot be a CascadeShadowCaster: those are identified by mesh pointer
    // and index range, all of which a skinned mesh holds constant while its
    // vertices move. A skinned caster's content lives in the joint palette
    // instead, so what gets hashed is a digest of that palette. Without it the
    // cache would hold a cascade whose caster is visibly somewhere else -- a
    // stale shadow, which reads as a rendering bug rather than a cache bug.
    uint64_t skinnedCasterPose = 0;
};

// Content hash of one cascade: the pass state plus every caster that cascade
// actually draws, in order.
//
// The caster count is folded in explicitly so that a shorter list can never
// hash to the same value as a longer one that happens to start with it.
[[nodiscard]] uint64_t computeCascadeShadowKey(const CascadeShadowPassState& state,
                                               std::span<const CascadeShadowCaster> casters);

} // namespace ve::renderer
