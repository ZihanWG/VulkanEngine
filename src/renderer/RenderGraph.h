#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <vector>

namespace ve::rhi {

class VulkanShadowMap;
class VulkanSwapchain;

} // namespace ve::rhi

namespace ve::renderer {

enum class RenderPassType {
    Shadow,
    Main
};

enum class RenderResourceAccess {
    Read,
    Write
};

struct RenderResourceHandle {
    const char* name = "";
};

struct RenderResourceUsage {
    RenderResourceHandle resource{};
    RenderResourceAccess access = RenderResourceAccess::Read;
    const char* description = "";
};

struct RenderPassNode {
    const char* name = "";
    RenderPassType type = RenderPassType::Main;
    std::vector<RenderResourceUsage> resourceUsages;
};

// This is a deliberately minimal frame graph. It documents the current pass
// order and resource usage, then centralizes the explicit Synchronization2
// transitions those passes need. It does not schedule passes automatically,
// allocate transient resources, alias attachments, cull passes, or use async
// compute yet.
class RenderGraph final {
public:
    RenderGraph();

    void beginFrame(
        VkCommandBuffer commandBuffer,
        rhi::VulkanSwapchain& swapchain,
        rhi::VulkanShadowMap& shadowMap,
        uint32_t imageIndex);
    void beginShadowPass();
    void endShadowPass();
    void beginMainPass();
    void endMainPass();
    void endFrame();

    [[nodiscard]] const std::vector<RenderPassNode>& passes() const { return passes_; }

private:
    struct FrameState {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        rhi::VulkanSwapchain* swapchain = nullptr;
        rhi::VulkanShadowMap* shadowMap = nullptr;
        uint32_t imageIndex = 0;
        VkImage swapchainImage = VK_NULL_HANDLE;
    };

    void requireFrameActive(const char* operation) const;
    void transitionShadowMapImage(VkImageLayout oldLayout, VkImageLayout newLayout);
    void transitionSwapchainImage(
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout);
    void transitionDepthImage();

    FrameState frame_{};
    std::vector<RenderPassNode> passes_;
    bool frameActive_ = false;
    bool shadowPassActive_ = false;
    bool mainPassActive_ = false;
};

} // namespace ve::renderer
