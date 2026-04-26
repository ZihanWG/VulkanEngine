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

private:
    void cleanupRenderFinishedSemaphores();

    VkDevice device_ = VK_NULL_HANDLE;

    // Frame-scoped acquire semaphores and fences are stored on FrameResources but owned here.
    std::span<renderer::FrameResources> frames_{};

    // Presentation can retain a render-finished semaphore until that swapchain image
    // is acquired again, so these semaphores are owned per swapchain image.
    std::vector<VkSemaphore> renderFinishedSemaphores_;
};

} // namespace ve::rhi
