#include "rhi/PipelineKey.h"

#include <bit>
#include <type_traits>

namespace ve::rhi {

namespace {

// Field-wise rather than memcmp: these are C structs whose padding bytes are
// indeterminate, so a byte comparison could report two identical descriptions as
// different depending on how the caller's stack happened to be initialized.
[[nodiscard]] bool equal(const VkVertexInputBindingDescription& lhs, const VkVertexInputBindingDescription& rhs)
{
    return lhs.binding == rhs.binding && lhs.stride == rhs.stride && lhs.inputRate == rhs.inputRate;
}

[[nodiscard]] bool equal(const VkVertexInputAttributeDescription& lhs, const VkVertexInputAttributeDescription& rhs)
{
    return lhs.location == rhs.location && lhs.binding == rhs.binding && lhs.format == rhs.format &&
           lhs.offset == rhs.offset;
}

[[nodiscard]] bool equal(const VkPushConstantRange& lhs, const VkPushConstantRange& rhs)
{
    return lhs.stageFlags == rhs.stageFlags && lhs.offset == rhs.offset && lhs.size == rhs.size;
}

[[nodiscard]] bool equal(VkFormat lhs, VkFormat rhs)
{
    return lhs == rhs;
}

[[nodiscard]] bool equal(VkDescriptorSetLayout lhs, VkDescriptorSetLayout rhs)
{
    return lhs == rhs;
}

template <typename T> [[nodiscard]] bool equalSequence(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (!equal(lhs[index], rhs[index])) {
            return false;
        }
    }

    return true;
}

// Order matters: the same values combined in a different order must produce a
// different digest, otherwise (say) swapping a binding and a stride would collide.
void hashCombine(std::size_t& seed, std::size_t value)
{
    constexpr std::size_t kGoldenRatio = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kGoldenRatio + (seed << 6) + (seed >> 2);
}

void hashValue(std::size_t& seed, uint32_t value)
{
    hashCombine(seed, static_cast<std::size_t>(value));
}

void hashValue(std::size_t& seed, bool value)
{
    hashCombine(seed, value ? 1U : 0U);
}

// Bit pattern, matching how operator== compares these -- see PipelineKey.h.
void hashValue(std::size_t& seed, float value)
{
    hashCombine(seed, static_cast<std::size_t>(std::bit_cast<uint32_t>(value)));
}

// Explicit, because an unscoped enum converts to uint32_t, bool and float alike:
// without this overload hashSequence<VkFormat> is ambiguous.
void hashValue(std::size_t& seed, VkFormat value)
{
    hashCombine(seed, static_cast<std::size_t>(static_cast<uint32_t>(value)));
}

void hashValue(std::size_t& seed, const std::filesystem::path& value)
{
    // std::filesystem::hash_value is the portable spelling; std::hash<path> was
    // only required from C++20 and is still missing on some standard libraries.
    hashCombine(seed, std::filesystem::hash_value(value));
}

void hashValue(std::size_t& seed, VkDescriptorSetLayout value)
{
    // Vulkan non-dispatchable handles are a pointer type on 64-bit targets and a
    // bare uint64_t on 32-bit ones, which no single cast spells portably. Every
    // target this project builds for (MSVC x64, x86-64 Linux, arm64 macOS) is the
    // pointer form, so pin that rather than let a 32-bit build fail obscurely.
    static_assert(std::is_pointer_v<VkDescriptorSetLayout>,
                  "PipelineKey hashes descriptor set layout handles as pointers; this target defines Vulkan "
                  "non-dispatchable handles as integers instead.");
    hashCombine(seed, std::hash<const void*>{}(static_cast<const void*>(value)));
}

void hashValue(std::size_t& seed, const VkVertexInputBindingDescription& value)
{
    hashValue(seed, value.binding);
    hashValue(seed, value.stride);
    hashValue(seed, static_cast<uint32_t>(value.inputRate));
}

void hashValue(std::size_t& seed, const VkVertexInputAttributeDescription& value)
{
    hashValue(seed, value.location);
    hashValue(seed, value.binding);
    hashValue(seed, static_cast<uint32_t>(value.format));
    hashValue(seed, value.offset);
}

void hashValue(std::size_t& seed, const VkPushConstantRange& value)
{
    hashValue(seed, static_cast<uint32_t>(value.stageFlags));
    hashValue(seed, value.offset);
    hashValue(seed, value.size);
}

template <typename T> void hashSequence(std::size_t& seed, const std::vector<T>& values)
{
    // The size is folded in so that {a} and {a, a} cannot collide.
    hashCombine(seed, values.size());
    for (const T& value : values) {
        hashValue(seed, value);
    }
}

} // namespace

PipelineKey PipelineKey::from(const VulkanPipelineCreateInfo& createInfo)
{
    PipelineKey key{};

    key.vertexShaderPath = createInfo.vertexShaderPath;
    key.fragmentShaderPath = createInfo.fragmentShaderPath;

    // Rule 1 (PipelineKey.h): mirror VulkanPipeline::create()'s resolution exactly.
    // Diverging here would not fail loudly -- it would hand back a pipeline whose
    // attachment set does not match what the caller described.
    if (createInfo.enableColorAttachment) {
        if (!createInfo.colorFormats.empty()) {
            key.colorFormats.assign(createInfo.colorFormats.begin(), createInfo.colorFormats.end());
        } else {
            key.colorFormats.push_back(createInfo.colorFormat);
        }
    }

    key.vertexBindings.assign(createInfo.vertexBindings.begin(), createInfo.vertexBindings.end());
    key.vertexAttributes.assign(createInfo.vertexAttributes.begin(), createInfo.vertexAttributes.end());
    key.descriptorSetLayouts.assign(createInfo.descriptorSetLayouts.begin(), createInfo.descriptorSetLayouts.end());
    key.pushConstantRanges.assign(createInfo.pushConstantRanges.begin(), createInfo.pushConstantRanges.end());

    key.enableDepth = createInfo.enableDepth;
    // Rule 2: neither of these reaches the driver with the depth test off.
    key.depthFormat = createInfo.enableDepth ? createInfo.depthFormat : VK_FORMAT_UNDEFINED;
    key.depthWriteEnable = createInfo.enableDepth && createInfo.depthWriteEnable;

    key.cullMode = createInfo.cullMode;
    key.depthCompareOp = createInfo.depthCompareOp;
    key.viewMask = createInfo.viewMask;
    key.depthBiasConstantFactor = createInfo.depthBiasConstantFactor;
    key.depthBiasClamp = createInfo.depthBiasClamp;
    key.depthBiasSlopeFactor = createInfo.depthBiasSlopeFactor;
    key.enableAdditiveBlend = createInfo.enableAdditiveBlend;
    key.enableAlphaBlend = createInfo.enableAlphaBlend;
    key.independentBlendAvailable = createInfo.independentBlendAvailable;
    key.enableDepthBias = createInfo.enableDepthBias;

    // createInfo.pipelineCache is deliberately not copied -- see PipelineKey.h.
    return key;
}

ComputePipelineKey ComputePipelineKey::from(const VulkanComputePipelineCreateInfo& createInfo)
{
    ComputePipelineKey key{};
    key.shaderPath = createInfo.shaderPath;
    key.descriptorSetLayouts.assign(createInfo.descriptorSetLayouts.begin(), createInfo.descriptorSetLayouts.end());
    key.pushConstantRanges.assign(createInfo.pushConstantRanges.begin(), createInfo.pushConstantRanges.end());
    // createInfo.pipelineCache is deliberately not copied -- see PipelineKey.h.
    return key;
}

bool operator==(const ComputePipelineKey& lhs, const ComputePipelineKey& rhs)
{
    return lhs.shaderPath == rhs.shaderPath && equalSequence(lhs.descriptorSetLayouts, rhs.descriptorSetLayouts)
        && equalSequence(lhs.pushConstantRanges, rhs.pushConstantRanges);
}

bool operator==(const PipelineKey& lhs, const PipelineKey& rhs)
{
    // Scalars first: they are cheap and reject most mismatches before any
    // sequence walk.
    return lhs.depthFormat == rhs.depthFormat && lhs.cullMode == rhs.cullMode &&
           lhs.depthCompareOp == rhs.depthCompareOp && lhs.viewMask == rhs.viewMask &&
           std::bit_cast<uint32_t>(lhs.depthBiasConstantFactor) ==
               std::bit_cast<uint32_t>(rhs.depthBiasConstantFactor) &&
           std::bit_cast<uint32_t>(lhs.depthBiasClamp) == std::bit_cast<uint32_t>(rhs.depthBiasClamp) &&
           std::bit_cast<uint32_t>(lhs.depthBiasSlopeFactor) == std::bit_cast<uint32_t>(rhs.depthBiasSlopeFactor) &&
           lhs.enableAdditiveBlend == rhs.enableAdditiveBlend && lhs.enableAlphaBlend == rhs.enableAlphaBlend &&
           lhs.independentBlendAvailable == rhs.independentBlendAvailable && lhs.enableDepth == rhs.enableDepth &&
           lhs.depthWriteEnable == rhs.depthWriteEnable && lhs.enableDepthBias == rhs.enableDepthBias &&
           lhs.vertexShaderPath == rhs.vertexShaderPath && lhs.fragmentShaderPath == rhs.fragmentShaderPath &&
           equalSequence(lhs.colorFormats, rhs.colorFormats) && equalSequence(lhs.vertexBindings, rhs.vertexBindings) &&
           equalSequence(lhs.vertexAttributes, rhs.vertexAttributes) &&
           equalSequence(lhs.descriptorSetLayouts, rhs.descriptorSetLayouts) &&
           equalSequence(lhs.pushConstantRanges, rhs.pushConstantRanges);
}

} // namespace ve::rhi

std::size_t std::hash<ve::rhi::PipelineKey>::operator()(const ve::rhi::PipelineKey& key) const noexcept
{
    std::size_t seed = 0;

    ve::rhi::hashValue(seed, key.vertexShaderPath);
    ve::rhi::hashValue(seed, key.fragmentShaderPath);
    ve::rhi::hashSequence(seed, key.colorFormats);
    ve::rhi::hashSequence(seed, key.vertexBindings);
    ve::rhi::hashSequence(seed, key.vertexAttributes);
    ve::rhi::hashSequence(seed, key.descriptorSetLayouts);
    ve::rhi::hashSequence(seed, key.pushConstantRanges);
    ve::rhi::hashValue(seed, static_cast<uint32_t>(key.depthFormat));
    ve::rhi::hashValue(seed, static_cast<uint32_t>(key.cullMode));
    ve::rhi::hashValue(seed, static_cast<uint32_t>(key.depthCompareOp));
    ve::rhi::hashValue(seed, key.viewMask);
    ve::rhi::hashValue(seed, key.depthBiasConstantFactor);
    ve::rhi::hashValue(seed, key.depthBiasClamp);
    ve::rhi::hashValue(seed, key.depthBiasSlopeFactor);
    ve::rhi::hashValue(seed, key.enableAdditiveBlend);
    ve::rhi::hashValue(seed, key.enableAlphaBlend);
    ve::rhi::hashValue(seed, key.independentBlendAvailable);
    ve::rhi::hashValue(seed, key.enableDepth);
    ve::rhi::hashValue(seed, key.depthWriteEnable);
    ve::rhi::hashValue(seed, key.enableDepthBias);

    return seed;
}

std::size_t std::hash<ve::rhi::ComputePipelineKey>::operator()(const ve::rhi::ComputePipelineKey& key) const noexcept
{
    std::size_t seed = 0;

    ve::rhi::hashValue(seed, key.shaderPath);
    ve::rhi::hashSequence(seed, key.descriptorSetLayouts);
    ve::rhi::hashSequence(seed, key.pushConstantRanges);

    return seed;
}
