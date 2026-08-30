#pragma once

#include "rhi/VulkanPipeline.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

// An owning, comparable, hashable value form of VulkanPipelineCreateInfo.
//
// VulkanPipelineCreateInfo cannot be a key itself: its four array members are
// non-owning std::spans, usually pointing at stack arrays inside a
// createXxxPipeline() function, so they dangle the moment that function returns.
// PipelineKey copies them into vectors and is otherwise a field-for-field mirror.
//
// It exists so pipeline state can be looked up by value rather than by member
// name. Two things follow from that, and both are the point:
//
//  - Identical state resolves to one VkPipeline. Today the same depth-only caster
//    state is compiled three times under three member names (see
//    RendererResources.cpp: the punctual atlas, the VSM page pool and the CSM
//    cascades all rasterise shadow_punctual.vert / shadow_skinned.vert into an
//    rhi::VulkanShadowMap, and every VulkanShadowMap resolves its format through
//    the same chooseShadowMapFormat(), so the formats are always equal).
//  - A format change is a cache miss by construction, rather than by a clause
//    somebody remembered to add to Renderer::pipelineNeedsRecreate.
//
// ## What is normalized, and why so little
//
// The two failure modes are not symmetric:
//
//  - Over-normalizing (treating state that matters as irrelevant) returns a
//    pipeline built for a *different* configuration. That is a real bug, and a
//    quiet one -- the wrong blend or the wrong attachment format does not
//    necessarily trip validation.
//  - Under-normalizing (treating irrelevant state as significant) compiles one
//    extra pipeline. That is a wasted object and nothing else.
//
// So normalization is limited to the cases VulkanPipeline::create() proves cannot
// reach the driver, and everything else is kept verbatim:
//
//  1. Color formats. create() resolves enableColorAttachment / colorFormat /
//     colorFormats into a single list -- empty when there is no color attachment,
//     otherwise `colorFormats` if non-empty and `{colorFormat}` if not. Call sites
//     set *both* fields (createMainGraphicsPipeline sets colorFormat and a 3-format
//     MRT list, then reads colorFormat back for its own bookkeeping), so two infos
//     that build a byte-identical pipeline can disagree on a field creation ignored.
//     This is the one normalization the cache actually needs to work; without it
//     the main and transparent pipelines would miss against themselves.
//     `enableColorAttachment` is not stored separately because it is exactly
//     `!colorFormats.empty()` after resolution.
//  2. depthFormat and depthWriteEnable, both gated on enableDepth. create() writes
//     `enableDepth ? depthFormat : VK_FORMAT_UNDEFINED` and
//     `enableDepth && depthWriteEnable`, so neither raw value can reach the driver
//     with the depth test off.
//
// Deliberately *not* normalized, even though each is arguably inert in some
// configuration: depthCompareOp (create() writes it unconditionally, whatever
// depthTestEnable says), the three depthBias factors (also written
// unconditionally, independent of enableDepthBias), and the blend flags when there
// is no color attachment. Each would be a guess about what the driver ignores, and
// the payoff would be at most one fewer pipeline.
//
// `pipelineCache` is excluded outright. It is the VkPipelineCache compile hint
// passed to vkCreateGraphicsPipelines, not pipeline state: two pipelines that
// differ only in it are the same pipeline, and keying on it would be a pure
// false miss.
//
// `independentBlendAvailable` *is* kept, despite being a device capability rather
// than a caller decision. It changes how many attachments enableAlphaBlend turns
// on, so it genuinely selects between two different pipelines.
//
// Vulkan handles (VkDescriptorSetLayout) are compared as handles. They are stable
// for as long as the layout lives, which makes the store's lifetime rule
// load-bearing -- see VulkanPipelineStore.h.
//
// GPU-free: this header names Vulkan types but calls no Vulkan entry point, so the
// key can be unit-tested without a device, the same way VulkanPipelineCache's
// header validation is (tests/test_pipeline_cache_header.cpp).
namespace ve::rhi {

struct PipelineKey {
    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;

    // Resolved per rule 1 above: empty means "no color attachment".
    std::vector<VkFormat> colorFormats;
    std::vector<VkVertexInputBindingDescription> vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    std::vector<VkPushConstantRange> pushConstantRanges;

    // VK_FORMAT_UNDEFINED whenever enableDepth is false (rule 2).
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    uint32_t viewMask = 0;

    // Compared and hashed by bit pattern, not by float equality: the key is a
    // pure value lookup, so it must have no tolerance semantics. -0.0f and 0.0f
    // are therefore distinct keys (one extra pipeline at worst), and a NaN bias
    // compares equal to the identical NaN instead of to nothing.
    float depthBiasConstantFactor = 0.0f;
    float depthBiasClamp = 0.0f;
    float depthBiasSlopeFactor = 0.0f;

    bool enableAdditiveBlend = false;
    bool enableAlphaBlend = false;
    bool independentBlendAvailable = false;
    bool enableDepth = false;
    bool depthWriteEnable = false;
    bool enableDepthBias = false;

    // Copies createInfo's spans and applies the normalization described above.
    [[nodiscard]] static PipelineKey from(const VulkanPipelineCreateInfo& createInfo);
};

[[nodiscard]] bool operator==(const PipelineKey& lhs, const PipelineKey& rhs);

[[nodiscard]] inline bool operator!=(const PipelineKey& lhs, const PipelineKey& rhs)
{
    return !(lhs == rhs);
}

} // namespace ve::rhi

template <> struct std::hash<ve::rhi::PipelineKey> {
    [[nodiscard]] std::size_t operator()(const ve::rhi::PipelineKey& key) const noexcept;
};
