#pragma once

// The skinned demo mesh as a ray-traced shadow caster (docs/ray_traced_shadows.md).
//
// The raster passes skin in their vertex stage and keep nothing, so a ray has no
// deformed triangles to hit. This owns what it takes to give it some: a buffer of
// skinned positions written by a compute pass from the frame's joint palette, and
// a bottom-level structure built from that buffer with ALLOW_UPDATE, built in
// full the first time and refitted after that.
//
// Ownership follows the TLAS (renderer/RayTracingScene.h): one positions buffer
// and one BLAS, rewritten in place by a frame that traces with the mesh visible
// and its pose moved (renderer::RayTracingTopLevelTracker decides). Earlier
// frames' refits read the positions and their rays read the BLAS;
// RayTracingScene's entry barrier orders both against this frame's writes.
// What this records itself is the edge inside the build: the dispatch's position
// writes against the build's reads of them.

#include "rhi/VulkanAccelerationStructure.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanComputePipeline.h"

#include <cstdint>
#include <filesystem>

namespace ve::rhi {
class VulkanContext;
} // namespace ve::rhi

namespace ve::renderer {

// Device addresses of the skinned mesh's bind-pose streams.
struct RayTracingSkinnedSource {
    // renderer::Vertex, bind pose.
    VkDeviceAddress vertices = 0;
    // renderer::SkinningVertex, parallel to the vertices.
    VkDeviceAddress skinning = 0;
    VkDeviceAddress indices = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
};

class RayTracingSkinnedCaster final {
public:
    RayTracingSkinnedCaster() = default;
    ~RayTracingSkinnedCaster();

    RayTracingSkinnedCaster(const RayTracingSkinnedCaster&) = delete;
    RayTracingSkinnedCaster& operator=(const RayTracingSkinnedCaster&) = delete;
    RayTracingSkinnedCaster(RayTracingSkinnedCaster&&) = delete;
    RayTracingSkinnedCaster& operator=(RayTracingSkinnedCaster&&) = delete;

    // The positions buffer, the BLAS storage sized for a build, and the
    // pre-skin pipeline. Nothing is built: there is no pose until a frame has
    // uploaded a palette. Throws on failure and leaves nothing allocated.
    void create(rhi::VulkanContext& context,
                const RayTracingSkinnedSource& source,
                const std::filesystem::path& preSkinShaderPath);
    void destroy();

    [[nodiscard]] bool valid() const
    {
        return bottomLevel_.valid();
    }
    // Scratch the next record() needs: a full build's the first time, a refit's
    // after. The caller's scratch buffer is sized for the larger once.
    [[nodiscard]] VkDeviceSize buildScratchSize() const
    {
        return buildScratchSize_;
    }
    [[nodiscard]] VkDeviceSize updateScratchSize() const
    {
        return updateScratchSize_;
    }
    // What a TLAS instance references. Stable for the caster's lifetime: the
    // structure is rewritten in place, never replaced.
    [[nodiscard]] VkDeviceAddress bottomLevelAddress() const
    {
        return bottomLevel_.deviceAddress();
    }
    [[nodiscard]] VkDeviceAddress positionsAddress() const
    {
        return positions_.deviceAddress();
    }
    [[nodiscard]] VkDeviceAddress indexAddress() const
    {
        return source_.indices;
    }
    [[nodiscard]] uint64_t builds() const
    {
        return builds_;
    }
    [[nodiscard]] uint64_t refits() const
    {
        return refits_;
    }

    // The pre-skin dispatch, the barrier from its writes to the build's reads,
    // then a full build (first call) or a refit. Leaves the BLAS written and not
    // yet visible to anything: the caller's barrier after it makes it visible to
    // the TLAS build and to the rays.
    void record(VkCommandBuffer commandBuffer, VkDeviceAddress jointPalette, VkDeviceAddress scratch);

private:
    rhi::VulkanContext* context_ = nullptr;
    RayTracingSkinnedSource source_{};
    rhi::VulkanBuffer positions_;
    rhi::VulkanAccelerationStructure bottomLevel_;
    rhi::VulkanComputePipeline preSkinPipeline_;
    VkDeviceSize buildScratchSize_ = 0;
    VkDeviceSize updateScratchSize_ = 0;
    bool built_ = false;
    uint64_t builds_ = 0;
    uint64_t refits_ = 0;
};

} // namespace ve::renderer
