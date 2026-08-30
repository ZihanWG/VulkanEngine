#include "rhi/PipelineKey.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// GPU-free by construction: these cases build VulkanPipelineCreateInfo values and
// compare the keys derived from them. No Vulkan entry point is called and no
// device exists, the same way tests/test_pipeline_cache_header.cpp covers the
// on-disk cache validation.

using ve::rhi::PipelineKey;
using ve::rhi::VulkanPipelineCreateInfo;

namespace {

// Descriptor set layouts are compared as handles, so the tests only need distinct
// non-null values. Casting integers to a handle type is what a real device would
// hand back; nothing here dereferences them.
VkDescriptorSetLayout fakeLayout(std::uintptr_t value)
{
    return reinterpret_cast<VkDescriptorSetLayout>(value);
}

constexpr VkVertexInputBindingDescription kBinding{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
constexpr std::array<VkVertexInputAttributeDescription, 2> kAttributes{
    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, 12},
};
constexpr VkPushConstantRange kPushRange{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};

// A fully populated info, so that every keyed field has a non-default value to
// mutate away from. A baseline of all-defaults would let a field missing from
// operator== pass by accident.
VulkanPipelineCreateInfo baselineInfo()
{
    VulkanPipelineCreateInfo info{};
    info.vertexShaderPath = "shaders/simple.vert.spv";
    info.fragmentShaderPath = "shaders/simple.frag.spv";
    info.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    info.depthFormat = VK_FORMAT_D32_SFLOAT;
    info.vertexBindings = std::span<const VkVertexInputBindingDescription>(&kBinding, 1);
    info.vertexAttributes = std::span<const VkVertexInputAttributeDescription>(kAttributes.data(), kAttributes.size());
    info.pushConstantRanges = std::span<const VkPushConstantRange>(&kPushRange, 1);
    info.enableColorAttachment = true;
    info.enableDepth = true;
    info.depthWriteEnable = true;
    info.enableDepthBias = true;
    info.cullMode = VK_CULL_MODE_BACK_BIT;
    info.depthCompareOp = VK_COMPARE_OP_GREATER;
    info.viewMask = 0x3;
    info.depthBiasConstantFactor = 1.25f;
    info.depthBiasClamp = 0.5f;
    info.depthBiasSlopeFactor = 2.75f;
    return info;
}

PipelineKey baselineKey()
{
    return PipelineKey::from(baselineInfo());
}

// Equality and hashing must agree: unequal-but-same-hash is legal for a hash map,
// but equal-with-different-hash is a bug that shows up as a duplicate pipeline.
void requireSameKey(const PipelineKey& lhs, const PipelineKey& rhs)
{
    CHECK(lhs == rhs);
    CHECK(std::hash<PipelineKey>{}(lhs) == std::hash<PipelineKey>{}(rhs));
}

} // namespace

TEST_CASE("Identical descriptions produce equal keys", "[pipeline-key]")
{
    requireSameKey(baselineKey(), baselineKey());
}

TEST_CASE("colorFormat and colorFormats normalize to the same key", "[pipeline-key]")
{
    // VulkanPipeline::create() resolves a single colorFormat and a one-element
    // colorFormats list to the same attachment set, so the key must too.
    // Without this, a caller that sets both fields (as the main pass does) could
    // miss against a caller that set only one.
    VulkanPipelineCreateInfo scalarForm = baselineInfo();
    scalarForm.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    const std::array<VkFormat, 1> single{VK_FORMAT_R16G16B16A16_SFLOAT};
    VulkanPipelineCreateInfo listForm = baselineInfo();
    listForm.colorFormat = VK_FORMAT_UNDEFINED;
    listForm.colorFormats = std::span<const VkFormat>(single.data(), single.size());

    requireSameKey(PipelineKey::from(scalarForm), PipelineKey::from(listForm));
}

TEST_CASE("A non-empty colorFormats list overrides colorFormat", "[pipeline-key]")
{
    // The override direction matters: colorFormats wins, and the ignored
    // colorFormat must not leak into the key. The main pass relies on this --
    // it sets colorFormat for its own bookkeeping and an MRT list for the pipeline.
    const std::array<VkFormat, 3> mrt{
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_A2B10G10R10_UNORM_PACK32};

    VulkanPipelineCreateInfo withScalar = baselineInfo();
    withScalar.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    withScalar.colorFormats = std::span<const VkFormat>(mrt.data(), mrt.size());

    VulkanPipelineCreateInfo withDifferentScalar = withScalar;
    withDifferentScalar.colorFormat = VK_FORMAT_R8G8B8A8_UNORM;

    requireSameKey(PipelineKey::from(withScalar), PipelineKey::from(withDifferentScalar));

    // ...but the list itself is significant.
    const std::array<VkFormat, 2> shorterMrt{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT};
    VulkanPipelineCreateInfo shorter = withScalar;
    shorter.colorFormats = std::span<const VkFormat>(shorterMrt.data(), shorterMrt.size());
    CHECK(PipelineKey::from(withScalar) != PipelineKey::from(shorter));
}

TEST_CASE("Color formats are irrelevant with no color attachment", "[pipeline-key]")
{
    // Every depth-only shadow pipeline takes this path.
    const std::array<VkFormat, 1> ignored{VK_FORMAT_R8G8B8A8_UNORM};

    VulkanPipelineCreateInfo depthOnly = baselineInfo();
    depthOnly.enableColorAttachment = false;
    depthOnly.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    VulkanPipelineCreateInfo depthOnlyOtherFormats = baselineInfo();
    depthOnlyOtherFormats.enableColorAttachment = false;
    depthOnlyOtherFormats.colorFormat = VK_FORMAT_UNDEFINED;
    depthOnlyOtherFormats.colorFormats = std::span<const VkFormat>(ignored.data(), ignored.size());

    requireSameKey(PipelineKey::from(depthOnly), PipelineKey::from(depthOnlyOtherFormats));

    // Turning the attachment back on is a different pipeline.
    CHECK(PipelineKey::from(depthOnly) != baselineKey());
}

TEST_CASE("Depth format and depth write are irrelevant with depth off", "[pipeline-key]")
{
    // VulkanPipeline::create() writes VK_FORMAT_UNDEFINED and depthWriteEnable
    // VK_FALSE when enableDepth is false, so neither raw value reaches the driver.
    VulkanPipelineCreateInfo noDepth = baselineInfo();
    noDepth.enableDepth = false;

    VulkanPipelineCreateInfo noDepthDifferentIgnoredFields = noDepth;
    noDepthDifferentIgnoredFields.depthFormat = VK_FORMAT_D16_UNORM;
    noDepthDifferentIgnoredFields.depthWriteEnable = false;

    requireSameKey(PipelineKey::from(noDepth), PipelineKey::from(noDepthDifferentIgnoredFields));
}

TEST_CASE("The pipeline cache handle is not part of the key", "[pipeline-key]")
{
    // It is the vkCreateGraphicsPipelines compile hint, not pipeline state.
    // Keying on it would make every lookup a miss the moment the handle changed.
    VulkanPipelineCreateInfo withoutCache = baselineInfo();
    VulkanPipelineCreateInfo withCache = baselineInfo();
    withCache.pipelineCache = reinterpret_cast<VkPipelineCache>(std::uintptr_t{0xCAFE});

    requireSameKey(PipelineKey::from(withoutCache), PipelineKey::from(withCache));
}

TEST_CASE("Every keyed field changes the key", "[pipeline-key]")
{
    // The guard against a field added to VulkanPipelineCreateInfo and copied into
    // PipelineKey but forgotten in operator== or in the hash. A miss here is a
    // silently shared pipeline built for a different configuration.
    const std::array<VkFormat, 2> otherColorFormats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT};
    constexpr VkVertexInputBindingDescription otherBinding{1, 16, VK_VERTEX_INPUT_RATE_INSTANCE};
    constexpr VkPushConstantRange otherPushRange{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};
    const VkDescriptorSetLayout layout = fakeLayout(0x1000);

    const std::vector<std::pair<std::string, std::function<void(VulkanPipelineCreateInfo&)>>> mutations{
        {"vertexShaderPath", [](VulkanPipelineCreateInfo& i) { i.vertexShaderPath = "shaders/other.vert.spv"; }},
        {"fragmentShaderPath", [](VulkanPipelineCreateInfo& i) { i.fragmentShaderPath = "shaders/other.frag.spv"; }},
        {"colorFormat", [](VulkanPipelineCreateInfo& i) { i.colorFormat = VK_FORMAT_R8G8B8A8_UNORM; }},
        {"colorFormats",
         [&](VulkanPipelineCreateInfo& i) {
             i.colorFormats = std::span<const VkFormat>(otherColorFormats.data(), otherColorFormats.size());
         }},
        {"depthFormat", [](VulkanPipelineCreateInfo& i) { i.depthFormat = VK_FORMAT_D16_UNORM; }},
        {"vertexBindings",
         [&](VulkanPipelineCreateInfo& i) {
             i.vertexBindings = std::span<const VkVertexInputBindingDescription>(&otherBinding, 1);
         }},
        {"vertexAttributes",
         [](VulkanPipelineCreateInfo& i) {
             i.vertexAttributes = std::span<const VkVertexInputAttributeDescription>(kAttributes.data(), 1);
         }},
        {"descriptorSetLayouts",
         [&](VulkanPipelineCreateInfo& i) {
             i.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&layout, 1);
         }},
        {"pushConstantRanges",
         [&](VulkanPipelineCreateInfo& i) {
             i.pushConstantRanges = std::span<const VkPushConstantRange>(&otherPushRange, 1);
         }},
        {"enableColorAttachment", [](VulkanPipelineCreateInfo& i) { i.enableColorAttachment = false; }},
        {"enableAdditiveBlend", [](VulkanPipelineCreateInfo& i) { i.enableAdditiveBlend = true; }},
        {"enableAlphaBlend", [](VulkanPipelineCreateInfo& i) { i.enableAlphaBlend = true; }},
        {"independentBlendAvailable", [](VulkanPipelineCreateInfo& i) { i.independentBlendAvailable = true; }},
        {"enableDepth", [](VulkanPipelineCreateInfo& i) { i.enableDepth = false; }},
        {"depthWriteEnable", [](VulkanPipelineCreateInfo& i) { i.depthWriteEnable = false; }},
        {"enableDepthBias", [](VulkanPipelineCreateInfo& i) { i.enableDepthBias = false; }},
        {"viewMask", [](VulkanPipelineCreateInfo& i) { i.viewMask = 0xF; }},
        {"cullMode", [](VulkanPipelineCreateInfo& i) { i.cullMode = VK_CULL_MODE_NONE; }},
        {"depthCompareOp", [](VulkanPipelineCreateInfo& i) { i.depthCompareOp = VK_COMPARE_OP_LESS; }},
        {"depthBiasConstantFactor", [](VulkanPipelineCreateInfo& i) { i.depthBiasConstantFactor = 4.0f; }},
        {"depthBiasClamp", [](VulkanPipelineCreateInfo& i) { i.depthBiasClamp = 4.0f; }},
        {"depthBiasSlopeFactor", [](VulkanPipelineCreateInfo& i) { i.depthBiasSlopeFactor = 4.0f; }},
    };

    const PipelineKey baseline = baselineKey();
    for (const auto& [fieldName, mutate] : mutations) {
        VulkanPipelineCreateInfo mutated = baselineInfo();
        mutate(mutated);

        INFO("field: " << fieldName);
        CHECK(PipelineKey::from(mutated) != baseline);
    }
}

TEST_CASE("Depth bias factors compare by bit pattern", "[pipeline-key]")
{
    // A pure value lookup must not carry float tolerance semantics.
    VulkanPipelineCreateInfo positiveZero = baselineInfo();
    positiveZero.depthBiasConstantFactor = 0.0f;

    VulkanPipelineCreateInfo negativeZero = baselineInfo();
    negativeZero.depthBiasConstantFactor = -0.0f;

    // 0.0f == -0.0f as floats, so a naive comparison would call these equal.
    // Distinct keys cost at most one extra pipeline; the point is that the key is
    // a bit pattern, with no comparison that could also swallow a real difference.
    CHECK(PipelineKey::from(positiveZero) != PipelineKey::from(negativeZero));

    // The mirror case: a NaN bias is equal to the identical NaN, where float
    // comparison would report "not equal" and rebuild the pipeline every lookup.
    VulkanPipelineCreateInfo nanBias = baselineInfo();
    nanBias.depthBiasConstantFactor = std::nanf("");
    requireSameKey(PipelineKey::from(nanBias), PipelineKey::from(nanBias));
}

TEST_CASE("Sequence length and order are significant", "[pipeline-key]")
{
    const std::array<VkFormat, 1> one{VK_FORMAT_R16G16B16A16_SFLOAT};
    const std::array<VkFormat, 2> twice{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
    const std::array<VkFormat, 2> forward{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT};
    const std::array<VkFormat, 2> reversed{VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};

    const auto keyFor = [](const auto& formats) {
        VulkanPipelineCreateInfo info = baselineInfo();
        info.colorFormats = std::span<const VkFormat>(formats.data(), formats.size());
        return PipelineKey::from(info);
    };

    // Repeating an attachment is a different MRT set, not the same one.
    CHECK(keyFor(one) != keyFor(twice));
    // Attachment order is the shader's output locations, so it cannot be a set.
    CHECK(keyFor(forward) != keyFor(reversed));
}

TEST_CASE("Descriptor set layout handles are compared by identity", "[pipeline-key]")
{
    const std::array<VkDescriptorSetLayout, 2> materialThenBindless{fakeLayout(0x10), fakeLayout(0x20)};
    const std::array<VkDescriptorSetLayout, 2> bindlessThenMaterial{fakeLayout(0x20), fakeLayout(0x10)};
    const std::array<VkDescriptorSetLayout, 1> materialOnly{fakeLayout(0x10)};

    const auto keyFor = [](const auto& layouts) {
        VulkanPipelineCreateInfo info = baselineInfo();
        info.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(layouts.data(), layouts.size());
        return PipelineKey::from(info);
    };

    // Set order is the binding index, so a swap is a different pipeline layout.
    CHECK(keyFor(materialThenBindless) != keyFor(bindlessThenMaterial));
    // The bindless fallback drops set 1; that must not alias the bindless path.
    CHECK(keyFor(materialThenBindless) != keyFor(materialOnly));
}

TEST_CASE("Depth-only caster pipelines built for the same target collapse", "[pipeline-key]")
{
    // The concrete claim this whole change rests on. Renderer::createPunctualShadowPipeline
    // and Renderer::createVsmPagePipeline describe the same operation -- one
    // depth-only draw per rect with that rect's projection pushed -- and differ
    // only in which rhi::VulkanShadowMap they target. Every VulkanShadowMap
    // resolves its format through the same chooseShadowMapFormat() on the same
    // physical device, so those formats are always equal and the two descriptions
    // are the same pipeline.
    //
    // If this case ever fails, the two call sites have genuinely diverged and the
    // duplicate is real -- fix the caller, do not relax the key.
    const auto casterInfo = [](VkFormat depthFormat) {
        VulkanPipelineCreateInfo info{};
        info.vertexShaderPath = "shaders/shadow_punctual.vert.spv";
        info.depthFormat = depthFormat;
        info.vertexBindings = std::span<const VkVertexInputBindingDescription>(&kBinding, 1);
        info.vertexAttributes = std::span<const VkVertexInputAttributeDescription>(kAttributes.data(), 1);
        info.pushConstantRanges = std::span<const VkPushConstantRange>(&kPushRange, 1);
        info.enableColorAttachment = false;
        info.enableDepth = true;
        info.depthWriteEnable = true;
        info.enableDepthBias = true;
        info.cullMode = VK_CULL_MODE_NONE;
        info.depthBiasConstantFactor = 1.25f;
        info.depthBiasSlopeFactor = 1.75f;
        return info;
    };

    requireSameKey(PipelineKey::from(casterInfo(VK_FORMAT_D32_SFLOAT)),
                   PipelineKey::from(casterInfo(VK_FORMAT_D32_SFLOAT)));

    // A device that fell back to D16 for one target and not the other would be a
    // genuine difference, and the key must still see it.
    CHECK(PipelineKey::from(casterInfo(VK_FORMAT_D32_SFLOAT)) != PipelineKey::from(casterInfo(VK_FORMAT_D16_UNORM)));
}

TEST_CASE("The key works as an unordered_map key", "[pipeline-key]")
{
    // What the store actually does with it.
    std::unordered_map<PipelineKey, std::string> pipelines;

    pipelines.emplace(baselineKey(), "Main");
    REQUIRE(pipelines.size() == 1);

    // A second identical description finds the first entry rather than inserting.
    const auto reused = pipelines.emplace(baselineKey(), "MainAgain");
    CHECK_FALSE(reused.second);
    CHECK(pipelines.size() == 1);
    CHECK(pipelines.at(baselineKey()) == "Main");

    VulkanPipelineCreateInfo different = baselineInfo();
    different.cullMode = VK_CULL_MODE_NONE;
    pipelines.emplace(PipelineKey::from(different), "MainNoCull");
    CHECK(pipelines.size() == 2);
}
