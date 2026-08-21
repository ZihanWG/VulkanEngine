#pragma once

// VirtualShadowMapPass owns the Vulkan side of the virtual shadow map.
//
// Phase 1 scope: page MARKING only. The dispatch works out which clipmap pages
// this frame's visible surfaces need and writes a bitmask, which is copied into
// a host-readable buffer and decoded a frame later. Nothing here changes what
// the renderer draws -- the point of this phase is to measure how many pages a
// real frame asks for before committing to a physical pool, a per-frame page
// budget, or a sampling path.
//
// Ownership boundary (Design B, reference-borrowing): the rendering services
// (context / depth pyramid / render graph / GPU profiler) stay owned by Renderer
// and are borrowed here by reference, matching DepthPyramid and GpuCulling. The
// per-frame request, readback, and parameter buffers are owned here.
//
// The page-request buffers are deliberately NOT render-graph resources. The only
// graph-visible dependency this pass has is the depth pyramid it samples, which
// the VsmPageMarkPass declaration covers; the buffer clear, the dispatch, and the
// readback copy are ordered by explicit barriers, which is the same split
// GpuCulling uses for its own counter readback. That is also why no RenderGraph
// reference is held here: the renderer opens and closes the declared pass around
// recordMarkPass, exactly as it does for the other subsystem recorders.

#include "renderer/VirtualShadowMap.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanShadowMap.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace ve {

namespace rhi {
class VulkanContext;
} // namespace rhi

namespace renderer {

class DepthPyramid;
class GpuProfiler;

// Everything the marking dispatch needs about this frame. Assembled by the
// renderer, which is the only place that knows both the camera and the light.
struct VsmMarkFrameInput {
    // The view-projection the depth pyramid was built with, NOT this frame's.
    // The dispatch inverts it to reconstruct world positions from that depth.
    glm::mat4 depthViewProjection{1.0f};
    // Camera position that produced that depth, for the same reason.
    glm::vec3 depthCameraPosition{0.0f};
    // viewportHeight * 0.5 * abs(proj[1][1]). The abs() is the caller's job.
    float projScaleY = 0.0f;
    // THIS frame's camera: it decides which pages are addressable, which is a
    // separate question from where the depth came from.
    glm::vec3 cameraPosition{0.0f};
    glm::vec3 lightDirection{0.0f, -1.0f, 0.0f};
    VsmClipmapSettings clipmap{};
    // Written sub-rect of the depth pyramid, not its allocation. The pyramid is
    // allocated at the maximum render resolution and only partly written when
    // render scale is below 1.
    VkExtent2D depthExtent{};
    // Pixels per marking thread along each axis.
    uint32_t blockStride = 4;
    // False when the previous frame's depth cannot be trusted (first frame,
    // resize, camera cut). The dispatch is skipped and the request set stays
    // empty rather than marking pages from garbage.
    bool depthValid = false;
};

// One page the residency update decided has to be drawn this frame.
struct VsmDirtyPage {
    uint32_t pageId = 0;
    uint32_t physicalPage = 0;
    uint32_t level = 0;
    glm::ivec2 absolutePage{0, 0};
};

// What one residency update did. Every one of these is surfaced, because the
// interesting failures here are quiet: pages refused, pages over budget, and
// pages that scrolled out between being requested and being allocated all look
// identical in the image (a coarser shadow) and identical to each other.
struct VsmResidencyStats {
    // Pages the marking pass asked for, some frames ago.
    uint32_t requestedPages = 0;
    // Of those, the ones still inside this frame's window. The difference is the
    // cost of the readback latency under camera motion.
    uint32_t addressablePages = 0;
    // Pages holding physical space after the update.
    uint32_t residentPages = 0;
    // Pages whose depth is already correct -- the whole point of the page grid
    // being absolute.
    uint32_t cachedPages = 0;
    // Pages queued for drawing this frame.
    uint32_t dirtyPages = 0;
    // Needed drawing but hit kMaxVsmPagesPerFrame. They stay allocated and
    // unrendered, and are picked up next frame.
    uint32_t overBudgetPages = 0;
    // The allocator had nothing to give: every physical page was already claimed
    // by this same frame. Non-zero means the pool is genuinely too small.
    uint32_t refusedPages = 0;
    uint32_t evictions = 0;
    // Pages whose depth a moved caster dropped since the last update. Distinct
    // from dirtyPages: these are redraws caused by the scene changing, not by
    // the camera reaching somewhere new.
    uint32_t casterInvalidatedPages = 0;
};

class VirtualShadowMapPass final {
public:
    VirtualShadowMapPass(rhi::VulkanContext& context, DepthPyramid& depthPyramid, GpuProfiler& gpuProfiler);
    ~VirtualShadowMapPass();

    VirtualShadowMapPass(const VirtualShadowMapPass&) = delete;
    VirtualShadowMapPass& operator=(const VirtualShadowMapPass&) = delete;

    // The page pool and its page table. Fixed-size and independent of both the
    // swapchain and the depth pyramid -- pages are a light-space quantity -- so
    // this is created once alongside the cascaded shadow map and, critically,
    // BEFORE the graphics pipelines, which need the pool's depth format.
    void createPagePoolResources(uint32_t frameCount);

    // (Re)creates the marking pipeline and its per-frame buffers/descriptors.
    // Separate from the pool because this half needs the depth pyramid, which is
    // swapchain-sized and recreated on resize. Safe to call again; it leaves the
    // pool and the pages in it alone. Capability is reported through
    // available(); a failure leaves the subsystem inert rather than throwing, so
    // the renderer keeps running on CSM.
    void createResources(uint32_t frameCount);
    void destroyResources();

    // Descriptor sets point at the depth pyramid's image view and sampler, both
    // of which are recreated on resize. Call after DepthPyramid::createResources.
    void refreshDepthPyramidBinding();

    [[nodiscard]] bool available() const { return available_; }

    // Clears the request bitmask, dispatches the marking shader, and copies the
    // result into this frame slot's readback buffer. Must be called inside the
    // graph's VsmPageMarkPass scope so the pyramid is in a sampled layout.
    void recordMarkPass(VkCommandBuffer commandBuffer, uint32_t frameIndex, const VsmMarkFrameInput& input);

    // Decodes the bitmask this frame slot produced. False until the slot has
    // actually recorded a marking pass at least once, matching how GpuCulling
    // gates its own readback -- the buffer holds whatever it held before, and
    // reporting that as a measurement would be worse than reporting nothing.
    [[nodiscard]] bool readRequestStats(uint32_t frameIndex, uint32_t levelCount, VsmPageRequestStats& stats);

    // Threads the last dispatch covered, for the debug UI. Purely informational.
    [[nodiscard]] uint32_t lastMarkThreadCount() const { return lastMarkThreadCount_; }

    // --- residency ---------------------------------------------------------

    // Runs the allocator over the request set this frame slot read back, binds
    // physical pages, and uploads the page table. Returns what it did.
    //
    // `cameraLightSpaceXy` is THIS frame's camera, which decides what is still
    // addressable; the request set was marked against an older window, whose
    // origins are remembered per frame slot so a requested slot can be turned
    // back into the absolute page it meant.
    //
    // `lightDirection` is not used to place anything -- the caller has already
    // folded it into cameraLightSpaceXy -- but it is checked. Moving the light
    // changes every page's world rect while leaving every page's identity
    // untouched, which is precisely the case the identity check cannot catch, and
    // there is no single edit site to hook: the debug UI drags the direction
    // every frame. So residency hashes its own inputs and drops itself when they
    // move, the same way the cascade cache does rather than tracking dirty flags.
    VsmResidencyStats updateResidency(uint32_t frameIndex,
                                      const VsmClipmapSettings& clipmap,
                                      const glm::vec2& cameraLightSpaceXy,
                                      const glm::vec3& lightDirection,
                                      uint64_t frameCounter);

    // Pages the last updateResidency queued for drawing.
    [[nodiscard]] const std::vector<VsmDirtyPage>& dirtyPages() const { return dirtyPages_; }

    // Drops the depth of every addressable page a world AABB touches, at every
    // active level, so the next residency update redraws them in place.
    //
    // This is how a caster that MOVED dirties its pages. The page table's
    // identity check cannot see that: the object moved, the page kept both its
    // coordinates and its physical page, and its depth silently still describes
    // where the object used to be. The caller invalidates both where the object
    // was and where it now is.
    //
    // Returns the number of pages that actually held depth, so repeated requests
    // and never-drawn pages do not inflate the counter.
    uint32_t invalidatePagesForBounds(const VsmClipmapSettings& clipmap,
                                      const glm::mat4& lightView,
                                      const glm::vec2& cameraLightSpaceXy,
                                      const glm::vec3& boundsMin,
                                      const glm::vec3& boundsMax);

    // Pages invalidated by moved casters since the last residency update, and
    // the running total. Reset by updateResidency, which consumes them.
    [[nodiscard]] uint32_t pagesInvalidatedByCasters() const { return pagesInvalidatedByCasters_; }

    // Records that every queued page has been drawn. Called after the page pass
    // actually recorded them, not when they were queued -- until the draws are
    // in the command buffer the pool still holds the previous occupant's depth.
    void markDirtyPagesRendered(uint32_t frameIndex);

    // Drops all residency. For anything that changes what a page would contain
    // without changing its identity: a scene switch, the light moving, a clipmap
    // settings change.
    void invalidateResidency();

    [[nodiscard]] const rhi::VulkanShadowMap& pagePool() const { return pagePool_; }
    [[nodiscard]] rhi::VulkanShadowMap& pagePool() { return pagePool_; }
    [[nodiscard]] bool pagePoolValid() const { return pagePool_.valid(); }
    // First frame after the pool is created: its contents are undefined, so the
    // whole image is cleared rather than only the pages being drawn.
    [[nodiscard]] bool pagePoolNeedsFullClear() const { return pagePoolNeedsFullClear_; }
    void setPagePoolFullClearDone() { pagePoolNeedsFullClear_ = false; }
    [[nodiscard]] VkDeviceAddress pageTableAddress(uint32_t frameIndex) const;

    // The page table as the allocator holds it, for the residency overlay. CPU
    // side: it is the same data the GPU reads, so the overlay shows what the
    // sampler will actually find rather than a separate bookkeeping copy that
    // could drift from it.
    [[nodiscard]] std::span<const VsmPageTableEntry> pageTable() const { return allocator_.entries(); }

    // --- per-page caster culling ------------------------------------------

    // (Re)creates the cull pipeline and its per-frame buffers. Separate again
    // because it needs the draw-item capacity, which the renderer owns. Requires
    // the page pool. Reports capability through cullAvailable().
    void createCullResources(uint32_t frameCount, uint32_t maxDrawItems);
    [[nodiscard]] bool cullAvailable() const { return cullAvailable_; }
    [[nodiscard]] uint32_t pageCommandStride() const { return kMaxVsmCastersPerPage; }
    [[nodiscard]] VkBuffer cullIndirectBuffer(uint32_t frameIndex) const;

    // One dispatch over every (dirty page, draw item) pair, recorded before the
    // page pass opens its rendering scope -- compute cannot run inside one, and
    // the scope has to stay single so untouched pages keep their depth.
    //
    // `casterFlags` is 1 for draw items that cast. The shared cull input carries
    // no bucket and an indirect per-page draw cannot skip blended geometry at
    // replay time, so the filter has to happen in the dispatch.
    void recordPageCull(VkCommandBuffer commandBuffer,
                        uint32_t frameIndex,
                        VkBuffer cullInputBuffer,
                        VkDeviceSize cullInputSize,
                        uint32_t drawItemCount,
                        const VsmClipmapSettings& clipmap,
                        const glm::mat4& lightView,
                        std::span<const uint32_t> casterFlags);

    // Casters the per-page command cap refused, read back a frame later. Counted
    // rather than silently dropped, matching the FrameCapacity contract.
    [[nodiscard]] bool readPageCullOverflow(uint32_t frameIndex, uint32_t& overflowCount);

private:
    void createDescriptorSetLayout();
    void createPipeline();
    void createBuffers(uint32_t frameCount);
    void createDescriptorSets(uint32_t frameCount);
    void createPagePool();
    void destroyMarkResources();
    [[nodiscard]] bool readRequestWords(uint32_t frameIndex, std::array<uint32_t, kVsmPageRequestWordCount>& words);
    void uploadPageTable(uint32_t frameIndex);
    void destroyCullResources();

    rhi::VulkanContext& context_;
    DepthPyramid& depthPyramid_;
    GpuProfiler& gpuProfiler_;

    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::VulkanComputePipeline pipeline_;
    rhi::VulkanDescriptorPool descriptorPool_;
    std::vector<VkDescriptorSet> descriptorSets_;

    // Per frame in flight.
    std::vector<rhi::VulkanBuffer> requestBuffers_;
    std::vector<rhi::VulkanBuffer> requestReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> paramsBuffers_;
    std::vector<rhi::VulkanBuffer> pageTableBuffers_;
    std::vector<uint8_t> readbackReady_;
    // Window origin per level that each frame slot's request set was marked
    // against. Without it a read-back slot index is ambiguous -- the same slot
    // names a different absolute page once the window has scrolled.
    std::vector<std::array<glm::ivec2, kVsmMaxClipmapLevels>> markWindowOrigins_;

    rhi::VulkanShadowMap pagePool_;
    VsmPageAllocator allocator_;
    std::vector<VsmDirtyPage> dirtyPages_;
    std::vector<uint32_t> requestedPageIds_;
    // Reused across calls so per-object invalidation does not allocate.
    std::vector<uint32_t> scratchPageIds_;
    uint32_t pagesInvalidatedByCasters_ = 0;
    // Inputs the current residency was built against. Compared by exact bit
    // pattern, so this is a strict "identical inputs" test and never an
    // approximate one -- same rule as ShadowCacheKey.
    glm::vec3 residencyLightDirection_{0.0f};
    VsmClipmapSettings residencyClipmap_{};
    bool residencyValid_ = false;

    // Per-page caster culling. Modelled on PunctualShadows: same one-dispatch,
    // one-region-per-slot shape, and the same reason for it.
    rhi::VulkanDescriptorSetLayout cullSetLayout_;
    rhi::VulkanComputePipeline cullPipeline_;
    rhi::VulkanDescriptorPool cullDescriptorPool_;
    std::vector<VkDescriptorSet> cullSets_;
    std::vector<rhi::VulkanBuffer> pageFrustumBuffers_;
    std::vector<rhi::VulkanBuffer> cullIndirectBuffers_;
    std::vector<rhi::VulkanBuffer> cullVisibleCountBuffers_;
    std::vector<rhi::VulkanBuffer> cullOverflowReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> casterFlagBuffers_;
    std::vector<uint8_t> cullReadbackReady_;
    std::vector<glm::vec4> pageFrustumPlanes_;
    uint32_t cullDrawItemCapacity_ = 0;
    bool cullAvailable_ = false;

    bool available_ = false;
    bool pagePoolNeedsFullClear_ = true;
    uint32_t lastMarkThreadCount_ = 0;
};

} // namespace renderer
} // namespace ve
