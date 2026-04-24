#pragma once

#include "rhi/VulkanCommon.h"

#include <filesystem>

namespace ve::rhi {

struct VulkanPipelineCreateInfo {
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    bool enableDepth = false;
};

class VulkanPipeline final {
public:
    VulkanPipeline() = default;
    ~VulkanPipeline();

    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&& other) noexcept;

    void create(VkDevice device, const VulkanPipelineCreateInfo& createInfo);
    void reset();

    [[nodiscard]] VkPipeline pipeline() const { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const { return layout_; }

private:
    [[nodiscard]] VkShaderModule createShaderModule(const std::filesystem::path& path) const;
    void moveFrom(VulkanPipeline& other) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule vertexShaderModule_ = VK_NULL_HANDLE;
    VkShaderModule fragmentShaderModule_ = VK_NULL_HANDLE;

    // A pipeline layout is required even before descriptors or push constants exist.
    // It is the shader resource contract that future descriptor sets will plug into.
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    // Graphics pipeline state is immutable; resize-sensitive state is kept dynamic.
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
