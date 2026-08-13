#pragma once

// GpuCulling owns the GPU-driven visibility-culling subsystem extracted from the
// monolithic Renderer: the shared cull.comp compute pipeline + descriptor layout,
// the per-frame main and shadow descriptor sets, the cull input / frame-param /
// compacted-indirect-count / visible-count(+readback) buffers, and the per-frame
// readback bookkeeping. It records the main frustum/occlusion cull and the
// per-cascade shadow cull dispatches, and reads back the visible-count results.
//
// Ownership boundary (Design B, reference-borrowing): the rendering services
// (context / depth pyramid / render graph / GPU profiler) and the *shared*
// indirect-draw output buffers (owned by Renderer's frame resources, because the
// main/shadow passes also read them) are borrowed by reference. The per-frame
// draw-data marshalling stays in Renderer (it reads scene draw items / batches /
// occlusion settings) and writes into the cull input/param buffers through the
// accessors below; the small frame bookkeeping it produces is pushed via the
// set*FrameInfo setters. The use-GPU-culling toggles and the CPU-side culling
// stats also stay in Renderer; GpuCulling reports only resource readiness, so the
// Renderer predicates combine the toggles with mainResourcesReady()/
// shadowResourcesReady().

#include "renderer/MeshLod.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"

#include <array>
#include <cstdint>
#include <vector>

#include <glm/vec4.hpp>

namespace ve {

namespace rhi {
class VulkanContext;
} // namespace rhi

namespace renderer {

class DepthPyramid;
class RenderGraph;
class GpuProfiler;

// Result of reading back the main cull pass's per-frame visible/culled counts.
// phase2RescuedDrawItems counts phase-1 occlusion candidates that the two-phase
// re-test emitted after all; the effective occlusion-culled count is
// occlusionCulledDrawItems - phase2RescuedDrawItems.
struct GpuCullCounters {
    uint32_t totalDrawItems = 0;
    uint32_t visibleDrawItems = 0;
    uint32_t frustumCulledDrawItems = 0;
    uint32_t occlusionCulledDrawItems = 0;
    uint32_t phase2RescuedDrawItems = 0;
    // Emitted draws per LOD level, counted by the cull shader as it selects.
    // Meshes with no chain are not counted, so this sums to <= visibleDrawItems.
    std::array<uint32_t, kMaxMeshLods> lodDrawItems{};
};

class GpuCulling final {
public:
    GpuCulling(rhi::VulkanContext& context,
               DepthPyramid& depthPyramid,
               RenderGraph& renderGraph,
               GpuProfiler& gpuProfiler,
               const std::vector<rhi::VulkanBuffer>& frameIndirectDrawBuffers,
               const std::vector<rhi::VulkanBuffer>& frameShadowIndirectDrawBuffers);
    ~GpuCulling();

    GpuCulling(const GpuCulling&) = delete;
    GpuCulling& operator=(const GpuCulling&) = delete;

    // (Re)creates the shared cull pipeline + main and (optionally) shadow cull
    // resources for frameCount frames-in-flight. wantMainCull / wantShadowCull are
    // the user toggles; shadowIndirectActive is the Renderer predicate gating the
    // shadow indirect path. Reports capability afterwards via available() /
    // shadowAvailable(). Requires the borrowed indirect-draw buffers and a valid
    // depth pyramid descriptor resource to already exist.
    void createResources(uint32_t frameCount, bool wantMainCull, bool wantShadowCull, bool shadowIndirectActive);
    void destroyResources();

    // Rebinds the depth-pyramid image/sampler into the main + shadow cull
    // descriptor sets (called after the pyramid is (re)created).
    void updateDepthPyramidDescriptors();

    // --- per-frame inputs produced by the Renderer's draw-data marshalling ---
    [[nodiscard]] rhi::VulkanBuffer& cullInputBuffer(uint32_t frameIndex);
    [[nodiscard]] rhi::VulkanBuffer& shadowCullInputBuffer(uint32_t frameIndex);
    [[nodiscard]] rhi::VulkanBuffer& paramBuffer(uint32_t frameIndex);
    // Flat LOD table the Renderer refills each frame; draw items index it with
    // their lodBase/lodCount.
    [[nodiscard]] rhi::VulkanBuffer& meshLodBuffer(uint32_t frameIndex);
    void resetFrameCounters(uint32_t frameIndex, uint32_t mainTotalDrawItems);
    void setMainCullFrameInfo(uint32_t frameIndex, uint32_t batchCount, bool indirectCountPath);
    void setShadowCullFrameInfo(uint32_t frameIndex,
                                uint32_t totalDrawItems,
                                uint32_t batchCount,
                                bool indirectCountPath);

    // --- recording (called from the frame loop) ---
    // active = Renderer's isGpuCullingActive(); frustumPlanes / drawItemCount come
    // from the frame; mainPassMultiDrawIndirect = isMainPassMultiDrawIndirectActive().
    // copyReadback: copies the stats counters to the readback buffer at the end
    // of this dispatch. Pass false when a phase-2 dispatch follows (the phase-2
    // recording copies the combined counters instead).
    void recordMainCull(VkCommandBuffer commandBuffer,
                        uint32_t frameIndex,
                        bool active,
                        uint32_t drawItemCount,
                        const std::array<glm::vec4, 6>& frustumPlanes,
                        bool mainPassMultiDrawIndirect,
                        bool copyReadback);
    // Two-phase occlusion phase 2: resets the per-batch visible counts (stats
    // counters persist), re-tests the phase-1 occlusion candidates against the
    // freshly built current-frame pyramid, and copies the combined counters to
    // the readback buffer. Recorded between the phase-1 main pass and the
    // phase-2 main pass.
    void recordMainCullPhase2(VkCommandBuffer commandBuffer,
                              uint32_t frameIndex,
                              bool active,
                              uint32_t drawItemCount,
                              const std::array<glm::vec4, 6>& frustumPlanes,
                              bool mainPassMultiDrawIndirect);
    // One dispatch for every cascade. The visible set is the union of the
    // cascade frusta (read from the frame-params buffer, not pushed), because a
    // single indirect list feeds all of them.
    void recordShadowCull(VkCommandBuffer commandBuffer,
                          uint32_t frameIndex,
                          bool active,
                          uint32_t cascadeCount,
                          uint32_t drawItemCount);

    // --- readback (results consumed by the debug UI / stats in Renderer) ---
    [[nodiscard]] bool readMainVisibleCount(bool active, uint32_t frameIndex, uint32_t& visibleCount);
    [[nodiscard]] bool readMainCounters(bool active, uint32_t frameIndex, GpuCullCounters& counters);
    [[nodiscard]] bool readShadowVisibleCount(bool active, uint32_t frameIndex, uint32_t& visibleCount);

    // --- resource readiness (Renderer combines with the use* toggles) ---
    [[nodiscard]] bool available() const { return gpuCullingAvailable_; }
    [[nodiscard]] bool shadowAvailable() const { return gpuShadowCullingAvailable_; }
    [[nodiscard]] bool mainResourcesReady(uint32_t frameCount) const;
    [[nodiscard]] bool shadowResourcesReady(uint32_t frameCount) const;
    [[nodiscard]] bool frameIndirectCountPathActive(uint32_t frameIndex) const;
    [[nodiscard]] bool frameShadowIndirectCountPathActive(uint32_t frameIndex) const;
    [[nodiscard]] uint32_t mainTotalDrawItems(uint32_t frameIndex) const;
    [[nodiscard]] uint32_t shadowTotalDrawItems(uint32_t frameIndex) const;

    // --- buffers the main draw / render graph bind (owned here) ---
    [[nodiscard]] const rhi::VulkanBuffer& visibleCountBuffer(uint32_t frameIndex) const;
    [[nodiscard]] const rhi::VulkanBuffer& shadowVisibleCountBuffer(uint32_t frameIndex) const;
    [[nodiscard]] uint32_t mainBatchCount(uint32_t frameIndex) const;
    [[nodiscard]] uint32_t shadowBatchCount(uint32_t frameIndex) const;

    // Buffer vectors registered as render-graph resources (mirrors how
    // PostProcessStack exposes exposureBuffers()).
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& cullInputBuffers() const { return frameCullInputBuffers_; }
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& visibleCountBuffers() const
    {
        return frameBatchVisibleCountBuffers_;
    }
    [[nodiscard]] const std::vector<rhi::VulkanBuffer>& visibleCountReadbackBuffers() const
    {
        return frameBatchVisibleCountReadbackBuffers_;
    }

private:
    // Barrier + copy of the visible-count/stats buffer into the host-readable
    // readback buffer, shared by the single-phase and phase-2 recording paths.
    void recordMainVisibleCountReadback(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    void createCullDescriptorLayout();
    void createCullPipeline();
    void createCullBuffers(uint32_t frameCount);
    void createCullDescriptorSets(uint32_t frameCount);
    void createShadowCullingResources(uint32_t frameCount, bool shadowIndirectActive);
    void createShadowCullBuffers(uint32_t frameCount);
    void createShadowCullDescriptorSets(uint32_t frameCount);
    void destroyShadowCullingResources();

    rhi::VulkanContext& context_;
    DepthPyramid& depthPyramid_;
    RenderGraph& renderGraph_;
    GpuProfiler& gpuProfiler_;
    const std::vector<rhi::VulkanBuffer>& frameIndirectDrawBuffers_;
    const std::vector<rhi::VulkanBuffer>& frameShadowIndirectDrawBuffers_;

    rhi::VulkanDescriptorSetLayout gpuCullDescriptorSetLayout_;
    rhi::VulkanComputePipeline gpuCullPipeline_;
    rhi::VulkanDescriptorPool gpuCullDescriptorPool_;
    rhi::VulkanDescriptorPool shadowCullDescriptorPool_;
    std::vector<VkDescriptorSet> gpuCullDescriptorSets_;
    std::vector<VkDescriptorSet> shadowCullDescriptorSets_;
    std::vector<rhi::VulkanBuffer> frameCullInputBuffers_;
    std::vector<rhi::VulkanBuffer> framePhaseResultBuffers_;
    // Per-frame LOD table (binding 6), shared by the main and shadow cull sets.
    std::vector<rhi::VulkanBuffer> frameMeshLodBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowCullInputBuffers_;
    std::vector<rhi::VulkanBuffer> frameGpuCullParamBuffers_;
    std::vector<rhi::VulkanBuffer> frameBatchVisibleCountBuffers_;
    std::vector<rhi::VulkanBuffer> frameBatchVisibleCountReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowBatchVisibleCountBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowBatchVisibleCountReadbackBuffers_;
    std::vector<uint32_t> frameGpuCullTotalDrawItems_;
    std::vector<uint32_t> frameGpuCullBatchCounts_;
    std::vector<uint32_t> frameGpuShadowCullTotalDrawItems_;
    std::vector<uint32_t> frameGpuShadowCullBatchCounts_;
    std::vector<uint8_t> frameGpuCullReadbackReady_;
    std::vector<uint8_t> frameGpuCullIndirectCountPath_;
    std::vector<uint8_t> frameGpuShadowCullReadbackReady_;
    std::vector<uint8_t> frameGpuShadowCullIndirectCountPath_;
    bool gpuCullingAvailable_ = false;
    bool gpuShadowCullingAvailable_ = false;
};

} // namespace renderer
} // namespace ve
