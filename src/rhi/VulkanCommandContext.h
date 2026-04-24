#pragma once

#include "rhi/VulkanCommon.h"

#include <span>

namespace ve::renderer {
struct FrameResources;
}

namespace ve::rhi {

class VulkanContext;

class VulkanCommandContext final {
public:
    VulkanCommandContext() = default;
    ~VulkanCommandContext();

    VulkanCommandContext(const VulkanCommandContext&) = delete;
    VulkanCommandContext& operator=(const VulkanCommandContext&) = delete;
    VulkanCommandContext(VulkanCommandContext&&) = delete;
    VulkanCommandContext& operator=(VulkanCommandContext&&) = delete;

    void initialize(const VulkanContext& context, std::span<renderer::FrameResources> frames);
    void cleanup();

    [[nodiscard]] VkCommandPool commandPool() const { return commandPool_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;

    // One resettable graphics command pool owns all per-frame primary command buffers.
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
