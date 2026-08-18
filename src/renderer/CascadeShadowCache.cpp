#include "renderer/CascadeShadowCache.h"

namespace ve::renderer {

void ShadowCacheKey::addBytes(const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash_ ^= bytes[i];
        // FNV-1a 64-bit prime.
        hash_ *= 1099511628211ULL;
    }
}

uint64_t computeCascadeShadowKey(const CascadeShadowPassState& state, std::span<const CascadeShadowCaster> casters)
{
    ShadowCacheKey key;
    key.reset();

    key.add(state.lightViewProjection);
    key.add(state.rasterDepthBiasConstantFactor);
    key.add(state.rasterDepthBiasSlopeFactor);
    key.add(state.shadowResolution);
    key.add(static_cast<uint32_t>(state.gpuLodSelectionActive ? 1u : 0u));
    key.add(static_cast<uint32_t>(casters.size()));

    for (const CascadeShadowCaster& caster : casters) {
        key.add(caster.mesh);
        key.add(caster.material);
        key.add(caster.firstIndex);
        key.add(caster.indexCount);
        key.add(caster.lodLevel);
        key.add(caster.bucket);
        key.add(caster.alphaCutoff);
        key.add(caster.modelMatrix);
    }

    return key.value();
}

} // namespace ve::renderer
