#pragma once

#include "rhi/VulkanCommon.h"

#include <filesystem>
#include <span>

namespace ve::rhi {

struct VulkanPipelineCreateInfo {
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    // Multi-attachment override: when non-empty this span replaces colorFormat,
    // and the pipeline renders into one color attachment per listed format.
    std::span<const VkFormat> colorFormats;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    std::span<const VkVertexInputBindingDescription> vertexBindings;
    std::span<const VkVertexInputAttributeDescription> vertexAttributes;
    std::span<const VkDescriptorSetLayout> descriptorSetLayouts;
    std::span<const VkPushConstantRange> pushConstantRanges;
    // Optional shared pipeline cache; defaults to VK_NULL_HANDLE (cold compile).
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    bool enableColorAttachment = true;
    // ONE + ONE additive blending on every color attachment (the shader
    // pre-multiplies its weights). Used by the SSR trace pass to accumulate
    // reflections into the existing scene color.
    bool enableAdditiveBlend = false;
    // Standard SRC_ALPHA / ONE_MINUS_SRC_ALPHA "over" blending for glTF BLEND
    // materials. Mutually exclusive with enableAdditiveBlend.
    //
    // Unlike enableAdditiveBlend, this applies to attachment 0 only; the rest are
    // plain overwrites. The transparent pass renders into the same MRT set as the
    // main pass, and blending a velocity vector or an octahedral normal would be
    // meaningless -- only the colour is composited.
    bool enableAlphaBlend = false;
    // Set from VulkanDevice::independentBlendEnabled(). Without it Vulkan requires
    // every attachment to share one blend state, so enableAlphaBlend falls back to
    // blending all of them.
    bool independentBlendAvailable = false;
    bool enableDepth = false;
    bool depthWriteEnable = true;
    bool enableDepthBias = false;
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    float depthBiasConstantFactor = 0.0f;
    float depthBiasClamp = 0.0f;
    float depthBiasSlopeFactor = 0.0f;
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

    // The pipeline layout is the shader resource contract: descriptor sets plus
    // any push constant ranges required by that pipeline.
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    // Graphics pipeline state is immutable; resize-sensitive state is kept dynamic.
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace ve::rhi
