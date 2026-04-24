#include "rhi/VulkanPipeline.h"

#include <utility>

namespace ve::rhi {

VulkanPipeline::~VulkanPipeline()
{
    reset();
}

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept
{
    moveFrom(other);
}

VulkanPipeline& VulkanPipeline::operator=(VulkanPipeline&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }

    return *this;
}

void VulkanPipeline::reset()
{
    if (pipeline_) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }

    if (layout_) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
}

void VulkanPipeline::moveFrom(VulkanPipeline& other) noexcept
{
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
}

} // namespace ve::rhi
