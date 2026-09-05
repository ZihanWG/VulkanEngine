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
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanShadowMap.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve::rhi {
class VulkanContext;
}

namespace ve::renderer {

class PunctualShadows final {
public:
    // Slots beyond this fall back to CPU culling. The indirect buffer is sized
    // slots * kMaxDrawItems commands, so the cap is what keeps that from growing
    // to megabytes for an atlas that in practice holds a couple of dozen tiles.
    static constexpr uint32_t kMaxGpuCulledSlots = 64;
    // Mesh batches the GPU cull can address. Survivors are compacted per
    // (slot, batch), so the counter buffer is slots x batches; 64 x 256 is 64 KiB
    // per frame, where sizing it for the 8192-batch worst case would be 2 MiB.
    // Scenes here run under twenty batches, and a frame that exceeds this falls
    // back to CPU caster culling rather than drawing something wrong -- see
    // Renderer::isGpuPunctualShadowCullingActive.
    static constexpr uint32_t kMaxGpuCulledBatches = 256;

    void create(rhi::VulkanContext& context, uint32_t frameCount);

    // Optional GPU caster culling. Failure leaves cullAvailable() false and the
    // renderer keeps its CPU frustum tests, matching how every other optional
    // subsystem here degrades.
    void createCullResources(uint32_t frameCount,
                             const std::filesystem::path& cullShaderPath,
                             uint32_t maxDrawItems);

    [[nodiscard]] bool cullAvailable() const
    {
        return cullAvailable_;
    }

    // Uploads this frame's per-slot frustum planes for the cull to read.
    void uploadSlotFrustums(uint32_t frameIndex);

    // One dispatch over every (slot, draw item) pair. Must be recorded *outside*
    // the atlas rendering scope -- compute cannot run inside one -- which is why
    // every slot is culled up front rather than interleaved with its draws the
    // way the CSM path does per cascade.
    // casterFlags is one entry per draw item, zero for anything that must not
    // cast. It is separate from the shared cull input because that buffer serves
    // the main and CSM paths too and carries no bucket information.
    // batchCount is gpuShadowMeshDrawBatches_.size(); it addresses the
    // per-(slot, batch) survivor counters the draw side reads back as an
    // indirect count.
    void recordCull(VkCommandBuffer commandBuffer,
                    uint32_t frameIndex,
                    VkBuffer cullInputBuffer,
                    VkDeviceSize cullInputSize,
                    uint32_t drawItemCount,
                    uint32_t batchCount,
                    std::span<const uint32_t> casterFlags);

    [[nodiscard]] VkBuffer cullIndirectBuffer(uint32_t frameIndex) const;
    // Survivor count per (slot, batch), consumed by vkCmdDrawIndexedIndirectCount.
    [[nodiscard]] VkBuffer cullVisibleCountBuffer(uint32_t frameIndex) const;
    // Commands reserved per slot; also the offset multiplier into the buffer.
    [[nodiscard]] uint32_t cullSlotCommandStride() const
    {
        return cullSlotCommandStride_;
    }
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
                          float range,
                          uint32_t sizeClass);

    // Assigns six consecutive tiles to one shadow-casting point light, one per
    // cube face, and returns the base slot. The shader adds the face index it
    // derives from the light-to-fragment direction, which is why the six have to
    // be consecutive. Returns kInvalidPunctualShadowSlot when six do not remain.
    uint32_t addPointLight(const glm::vec3& position, float range, uint32_t sizeClass);

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

    // Light-space view-projection, cull frustum, and atlas rect for one slot.
    // The rect no longer follows from the index -- tiles vary in size, so it is
    // stored per slot and the atlas pass reads it back for viewport/scissor.
    [[nodiscard]] const glm::mat4& slotViewProjection(uint32_t slot) const;
    [[nodiscard]] const Frustum& slotFrustum(uint32_t slot) const;
    [[nodiscard]] ShadowAtlasRect slotRect(uint32_t slot) const;

    // Fraction of the atlas area handed out this frame. With mixed tile sizes a
    // slot count says nothing about how full the atlas actually is.
    [[nodiscard]] float occupancy() const
    {
        return allocator_.occupancy();
    }

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
    // Parallel to slots_: the cull frustum and pixel rect for each, kept off the
    // GPU record because the shader needs neither (it reads the UV rect that is
    // already inside GpuShadowSlot).
    std::vector<Frustum> slotFrustums_;
    std::vector<ShadowAtlasRect> slotRects_;
    // Scratch for the all-or-nothing cube-face reservation.
    std::vector<ShadowAtlasRect> pendingFaceRects_;

    // --- optional GPU caster culling ---
    bool cullAvailable_ = false;
    uint32_t cullSlotCommandStride_ = 0;
    rhi::VulkanComputePipeline cullPipeline_;
    rhi::VulkanDescriptorSetLayout cullSetLayout_;
    rhi::VulkanDescriptorPool cullDescriptorPool_;
    std::vector<VkDescriptorSet> cullSets_;
    // Per frame: six planes per slot, host-visible since they change every frame.
    std::vector<rhi::VulkanBuffer> slotFrustumBuffers_;
    // Per frame: compacted commands, one region of cullSlotCommandStride_ per slot.
    std::vector<rhi::VulkanBuffer> cullIndirectBuffers_;
    std::vector<rhi::VulkanBuffer> cullVisibleCountBuffers_;
    // Per frame: one visible count per slot, reset each dispatch.
    // Per frame: one flag per draw item, host-visible since buckets change with
    // the scene.
    std::vector<rhi::VulkanBuffer> casterFlagBuffers_;
    // Scratch for the plane upload.
    std::vector<glm::vec4> slotFrustumPlanes_;

    // Defaults tuned against the demo scene's spot cone; both are in the same
    // units the CSM path uses so the two shadow types stay comparable.
    float constantBias_ = 0.0015f;
    float normalBias_ = 0.035f;
};

} // namespace ve::renderer
