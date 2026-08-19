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

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
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

class VirtualShadowMapPass final {
public:
    VirtualShadowMapPass(rhi::VulkanContext& context, DepthPyramid& depthPyramid, GpuProfiler& gpuProfiler);
    ~VirtualShadowMapPass();

    VirtualShadowMapPass(const VirtualShadowMapPass&) = delete;
    VirtualShadowMapPass& operator=(const VirtualShadowMapPass&) = delete;

    // (Re)creates the marking pipeline and the per-frame buffers/descriptors.
    // Safe to call again: existing resources are torn down first. Capability is
    // reported afterwards through available(); a failure leaves the subsystem
    // inert rather than throwing, so the renderer keeps running on CSM.
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

private:
    void createDescriptorSetLayout();
    void createPipeline();
    void createBuffers(uint32_t frameCount);
    void createDescriptorSets(uint32_t frameCount);

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
    std::vector<uint8_t> readbackReady_;

    bool available_ = false;
    uint32_t lastMarkThreadCount_ = 0;
};

} // namespace renderer
} // namespace ve
