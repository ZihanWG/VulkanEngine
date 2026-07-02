#pragma once

#include "rhi/VulkanCommon.h"

#include <filesystem>
#include <span>

namespace ve::rhi {

struct VulkanComputePipelineCreateInfo {
    std::filesystem::path shaderPath;
    std::span<const VkDescriptorSetLayout> descriptorSetLayouts;
    std::span<const VkPushConstantRange> pushConstantRanges;
    // Optional shared pipeline cache; defaults to VK_NULL_HANDLE (cold compile).
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
};

class VulkanComputePipeline final {
public:
    VulkanComputePipeline() = default;
    ~VulkanComputePipeline();

    VulkanComputePipeline(const VulkanComputePipeline&) = delete;
    VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;
    VulkanComputePipeline(VulkanComputePipeline&& other) noexcept;
    VulkanComputePipeline& operator=(VulkanComputePipeline&& other) noexcept;

    void create(VkDevice device, const VulkanComputePipelineCreateInfo& createInfo);
    void reset();

    [[nodiscard]] VkPipeline pipeline() const { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const { return layout_; }

private:
    [[nodiscard]] VkShaderModule createShaderModule(const std::filesystem::path& path) const;
    void moveFrom(VulkanComputePipeline& other) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
