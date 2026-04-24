#pragma once

#include "rhi/VulkanCommon.h"

#include <span>

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

    void initialize(const VulkanContext& context, std::span<renderer::FrameResources> frames);
    void cleanup();

private:
    VkDevice device_ = VK_NULL_HANDLE;

    // Synchronization handles are stored on FrameResources but owned and destroyed here.
    std::span<renderer::FrameResources> frames_{};
};

} // namespace ve::rhi
