#pragma once

// Punctual (point/spot) shadow subsystem: owns the shadow atlas image, the
// per-frame slot buffer the fragment shader reads through a device address, and
// the per-frame CPU bookkeeping that decides which lights get a tile.
//
// Ownership boundary mirrors ClusteredLighting: this class owns the resources
// and the CPU-side math, while the actual caster draws stay in Renderer, which
// is where the draw-item lists, pipelines, and the render graph live. The
// atlas pass is recorded as one dynamic-rendering pass over the whole atlas,
// with a viewport/scissor per allocated slot.
//
// The GPU-free allocator and projection math live in PunctualShadowAtlas.h so
// they can be unit tested without a device.

#include "renderer/Bounds.h"
#include "renderer/PunctualShadowAtlas.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanShadowMap.h"

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve::rhi {
class VulkanContext;
}

namespace ve::renderer {

class PunctualShadows final {
public:
    void create(rhi::VulkanContext& context, uint32_t frameCount);
    void reset();

    // Drops every slot from the previous frame. Called before the per-light
    // assignment walk so slot indices are stable for exactly one frame.
    void beginFrame();

    // Assigns an atlas tile to one shadow-casting spot light and records the
    // projection its casters render with. Returns the slot index, or
    // kInvalidPunctualShadowSlot when the atlas is out of tiles -- callers treat
    // that as "unshadowed this frame" rather than an error.
    uint32_t addSpotLight(const glm::vec3& position,
                          const glm::vec3& direction,
                          float outerAngleRadians,
                          float range);

    // Assigns six consecutive tiles to one shadow-casting point light, one per
    // cube face, and returns the base slot. The shader adds the face index it
    // derives from the light-to-fragment direction, which is why the six have to
    // be consecutive. Returns kInvalidPunctualShadowSlot when six do not remain.
    uint32_t addPointLight(const glm::vec3& position, float range);

    // Writes the frame's slot records into the per-frame device-address buffer.
    void upload(uint32_t frameIndex);

    [[nodiscard]] bool valid() const
    {
        return atlas_.valid();
    }

    [[nodiscard]] const rhi::VulkanShadowMap& atlas() const
    {
        return atlas_;
    }
    [[nodiscard]] rhi::VulkanShadowMap& atlas()
    {
        return atlas_;
    }

    [[nodiscard]] VkDeviceAddress slotBufferAddress(uint32_t frameIndex) const;

    [[nodiscard]] uint32_t slotCount() const
    {
        return static_cast<uint32_t>(slots_.size());
    }
    [[nodiscard]] const std::vector<GpuShadowSlot>& slots() const
    {
        return slots_;
    }

    // Light-space view-projection and cull frustum for one allocated slot.
    [[nodiscard]] const glm::mat4& slotViewProjection(uint32_t slot) const;
    [[nodiscard]] const Frustum& slotFrustum(uint32_t slot) const;

    // Depth bias applied when the shader compares against the atlas. Exposed so
    // the debug UI can tune it against real geometry the way the CSM bias is.
    void setDepthBias(float constantBias, float normalBias)
    {
        constantBias_ = constantBias;
        normalBias_ = normalBias;
    }
    [[nodiscard]] float constantBias() const
    {
        return constantBias_;
    }
    [[nodiscard]] float normalBias() const
    {
        return normalBias_;
    }

private:
    rhi::VulkanContext* context_ = nullptr;
    rhi::VulkanShadowMap atlas_;
    // Per-frame, host-visible, device-address. Sized for the whole grid so a
    // frame that fills the atlas never reallocates mid-flight.
    std::vector<rhi::VulkanBuffer> slotBuffers_;

    PunctualShadowAtlasAllocator allocator_;
    std::vector<GpuShadowSlot> slots_;
    // Parallel to slots_: the cull frustum for each, kept off the GPU record
    // because the shader never needs it.
    std::vector<Frustum> slotFrustums_;

    // Defaults tuned against the demo scene's spot cone; both are in the same
    // units the CSM path uses so the two shadow types stay comparable.
    float constantBias_ = 0.0015f;
    float normalBias_ = 0.035f;
};

} // namespace ve::renderer
