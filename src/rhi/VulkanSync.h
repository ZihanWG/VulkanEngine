#pragma once

#include "rhi/VulkanCommon.h"

#include <span>
#include <vector>

namespace ve::renderer {
struct FrameResources;
}

namespace ve::rhi {

class VulkanContext;

class VulkanSync final {
public:
    VulkanSync() = default;
    ~VulkanSync();

    VulkanSync(const VulkanSync&) = delete;
    VulkanSync& operator=(const VulkanSync&) = delete;
    VulkanSync(VulkanSync&&) = delete;
    VulkanSync& operator=(VulkanSync&&) = delete;

    void initialize(const VulkanContext& context, std::span<renderer::FrameResources> frames, uint32_t swapchainImageCount);
    void recreateRenderFinishedSemaphores(uint32_t swapchainImageCount);
    void cleanup();

    [[nodiscard]] VkSemaphore renderFinishedSemaphore(uint32_t swapchainImageIndex) const
    {
        return renderFinishedSemaphores_.at(swapchainImageIndex);
    }

    // One monotonic counter for the whole device, signalled by every graphics
    // submit. It replaces the per-slot fences: a slot records the value its last
    // submission signals, and waiting for that value is equivalent to waiting on
    // that slot's fence -- with the difference that the value is also meaningful
    // to anything else that needs to know how far the GPU has got, which a fence
    // owned by a frame slot is not.
    [[nodiscard]] VkSemaphore frameTimeline() const { return frameTimeline_; }

    // Block until the timeline reaches `value`. A value of zero returns
    // immediately, which is what makes a never-submitted slot free to use.
    void waitForTimelineValue(uint64_t value) const;

private:
    void cleanupRenderFinishedSemaphores();

    VkDevice device_ = VK_NULL_HANDLE;

    // Frame-scoped acquire semaphores are stored on FrameResources but owned here.
    std::span<renderer::FrameResources> frames_{};

    VkSemaphore frameTimeline_ = VK_NULL_HANDLE;

    // Presentation can retain a render-finished semaphore until that swapchain image
    // is acquired again, so these semaphores are owned per swapchain image.
    std::vector<VkSemaphore> renderFinishedSemaphores_;
};

} // namespace ve::rhi
