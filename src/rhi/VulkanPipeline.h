#pragma once

#include "rhi/VulkanCommon.h"

#include <filesystem>
#include <span>

namespace ve::rhi {

struct VulkanPipelineCreateInfo {
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    std::span<const VkVertexInputBindingDescription> vertexBindings;
    std::span<const VkVertexInputAttributeDescription> vertexAttributes;
    std::span<const VkDescriptorSetLayout> descriptorSetLayouts;
    std::span<const VkPushConstantRange> pushConstantRanges;
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

    // The pipeline layout is the shader resource contract. For this milestone it
    // declares the push constant range carrying the MVP buffer device address.
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    // Graphics pipeline state is immutable; resize-sensitive state is kept dynamic.
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
