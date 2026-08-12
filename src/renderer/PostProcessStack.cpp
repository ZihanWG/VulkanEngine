// PostProcessStack: HDR post-process subsystem extracted from Renderer.
//
// The bulk of this file is relocated verbatim from Renderer.cpp (bloom, TAA,
// auto-exposure, composite, descriptor/resource creation). Injected services
// and borrowed settings keep the same member names as the former Renderer
// members so the moved bodies are unchanged. See PostProcessStack.h for the
// ownership boundary.

#include "renderer/PostProcessStack.h"

#include "renderer/RendererInternal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>

namespace ve {
namespace renderer {

PostProcessStack::PostProcessStack(rhi::VulkanContext& context,
                                   RenderGraph& renderGraph,
                                   GpuProfiler& gpuProfiler,
                                   rhi::VulkanSwapchain& swapchain,
                                   const RenderResolution& renderResolution,
                                   ToneMappingSettings& toneMappingSettings,
                                   BloomSettings& bloomSettings,
                                   TaaSettings& taaSettings,
                                   SsaoSettings& ssaoSettings,
                                   float& currentExposure,
                                   float& averageLuminance,
                                   float& histogramClippedLuminance,
                                   bool& ssaoAvailable)
    : context_(context), renderGraph_(renderGraph), gpuProfiler_(gpuProfiler), swapchain_(swapchain),
      renderResolution_(renderResolution), toneMappingSettings_(toneMappingSettings), bloomSettings_(bloomSettings), taaSettings_(taaSettings),
      ssaoSettings_(ssaoSettings), currentExposure_(currentExposure), averageLuminance_(averageLuminance),
      histogramClippedLuminance_(histogramClippedLuminance), ssaoAvailable_(ssaoAvailable)
{}

PostProcessStack::~PostProcessStack()
{
    destroyPostProcessSampler();
}

void PostProcessStack::beginFrame(uint32_t frameIndex, bool taaActive)
{
    currentFrame_ = frameIndex;
    taaPostProcessHistoryIndex_ = taaActive ? taaHistoryWriteIndex() : 0u;
}

void PostProcessStack::advanceTaaHistory()
{
    taaHistoryValid_ = true;
    taaHistoryWriteIndex_ = (taaHistoryWriteIndex_ + 1u) % kTaaHistoryCount;
}

// Advances TAA jitter bookkeeping (relocated from Renderer::updateFrameData) and
// returns the NDC offset the caller adds to the main projection. Returns {0,0}
// when TAA jitter is inactive, so the caller's add is a no-op.
glm::vec2 PostProcessStack::advanceJitter(VkExtent2D extent)
{
    taaPreviousJitterPixels_ = taaCurrentJitterPixels_;
    taaPreviousJitterNdc_ = taaCurrentJitterNdc_;
    taaCurrentJitterPixels_ = {0.0f, 0.0f};
    taaCurrentJitterNdc_ = {0.0f, 0.0f};
    if (isTaaJitterActive() && extent.width > 0 && extent.height > 0) {
        const uint32_t sampleIndex = (taaJitterIndex_ % kTaaJitterSampleCount) + 1u;
        taaCurrentJitterPixels_ = {
            halton(sampleIndex, 2u) - 0.5f,
            halton(sampleIndex, 3u) - 0.5f,
        };
        taaCurrentJitterNdc_ = {
            2.0f * taaCurrentJitterPixels_.x / static_cast<float>(extent.width),
            2.0f * taaCurrentJitterPixels_.y / static_cast<float>(extent.height),
        };
        ++taaJitterIndex_;
    }
    return taaCurrentJitterNdc_;
}

// Post-process slice of the former Renderer::createPostProcessResources. The
// depth-pyramid creation, ImGui preview-descriptor reset, and bloom-mip debug
// selection that were interleaved here stay in Renderer's wrapper.
void PostProcessStack::createPostProcessResources(VkImageView depthFallbackView, uint32_t frameCount)
{
    depthFallbackView_ = depthFallbackView;
    frameCount_ = frameCount;

    // Allocated at the maximum render resolution; only the top-left sub-rect is
    // written. Every consumer scales its UVs by RenderResolution::uvScale and
    // clamps inside the written region -- see sub_rect.glsl.
    const VkExtent2D extent = sceneAllocatedExtent();
    if (extent.width == 0 || extent.height == 0) {
        throw std::runtime_error("Cannot create post-process resources for a zero-sized render extent.");
    }

    postProcessDescriptorPool_.reset();
    bloomExtractDescriptorSet_ = VK_NULL_HANDLE;
    bloomBlurHorizontalDescriptorSet_ = VK_NULL_HANDLE;
    bloomBlurVerticalDescriptorSet_ = VK_NULL_HANDLE;
    compositeDescriptorSet_ = VK_NULL_HANDLE;
    bloomMipDownsampleDescriptorSets_.clear();
    bloomMipUpsampleDescriptorSets_.clear();
    compositeDescriptorSets_.clear();
    luminanceDescriptorSets_.clear();
    histogramDescriptorSets_.clear();
    exposureReduceDescriptorSets_.clear();
    destroyTaaResources();
    bloomMipDownsampleImages_.clear();
    bloomMipUpsampleImages_.clear();
    bloomMipDownsampleLayouts_.clear();
    bloomMipUpsampleLayouts_.clear();
    destroyLuminanceResources();
    destroyHistogramResources();
    destroyExposureResources();

    rhi::VulkanImageCreateInfo sceneColorInfo{};
    sceneColorInfo.width = extent.width;
    sceneColorInfo.height = extent.height;
    sceneColorInfo.format = kSceneColorFormat;
    sceneColorInfo.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sceneColorInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sceneColorInfo.debugName = "SceneColorHDR";
    sceneColor_.create(context_, sceneColorInfo);
    sceneColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    rhi::VulkanImageCreateInfo velocityInfo{};
    velocityInfo.width = extent.width;
    velocityInfo.height = extent.height;
    velocityInfo.format = kVelocityFormat;
    velocityInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    velocityInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    velocityInfo.debugName = "VelocityBuffer";
    velocity_.create(context_, velocityInfo);
    velocityLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    rhi::VulkanImageCreateInfo normalRoughnessInfo{};
    normalRoughnessInfo.width = extent.width;
    normalRoughnessInfo.height = extent.height;
    normalRoughnessInfo.format = kNormalRoughnessFormat;
    normalRoughnessInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    normalRoughnessInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    normalRoughnessInfo.debugName = "NormalRoughnessGBuffer";
    normalRoughness_.create(context_, normalRoughnessInfo);
    normalRoughnessLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    rhi::VulkanImageCreateInfo ambientOcclusionInfo{};
    ambientOcclusionInfo.width = extent.width;
    ambientOcclusionInfo.height = extent.height;
    ambientOcclusionInfo.format = kAmbientOcclusionFormat;
    ambientOcclusionInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ambientOcclusionInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ambientOcclusionInfo.debugName = "AmbientOcclusion";
    ambientOcclusion_.create(context_, ambientOcclusionInfo);
    ambientOcclusionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    // Contents are undefined until GTAO writes it, and the main pass samples it
    // for the previous frame's occlusion, so it stays unusable for exactly one
    // frame after every (re)creation -- including swapchain resize.
    ambientOcclusionHistoryValid_ = false;

    createTaaResources();

    // Allocated half. bloomWrittenExtent() is the half of what the frame writes,
    // and the two round independently -- which is why every bloom pass carries
    // its own source uv scale rather than the scene's.
    bloomExtent_ = RenderResolution::halved(sceneAllocatedExtent());

    rhi::VulkanImageCreateInfo bloomInfo{};
    bloomInfo.width = bloomExtent_.width;
    bloomInfo.height = bloomExtent_.height;
    bloomInfo.format = kBloomColorFormat;
    bloomInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bloomInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    bloomInfo.debugName = "BloomExtract";
    bloomExtract_.create(context_, bloomInfo);
    bloomExtractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    bloomInfo.debugName = "BloomPing";
    bloomPing_.create(context_, bloomInfo);
    bloomPingLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    bloomInfo.debugName = "BloomPong";
    bloomPong_.create(context_, bloomInfo);
    bloomPongLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    const uint32_t bloomMipCount = calculateBloomMipChainLevels(sceneAllocatedExtent());
    bloomMipDownsampleImages_.resize(bloomMipCount);
    bloomMipDownsampleLayouts_.assign(bloomMipCount, VK_IMAGE_LAYOUT_UNDEFINED);
    for (uint32_t level = 0; level < bloomMipCount; ++level) {
        const VkExtent2D mipSize = bloomMipExtent(sceneAllocatedExtent(), level);
        bloomInfo.width = mipSize.width;
        bloomInfo.height = mipSize.height;
        bloomInfo.debugName = "BloomMipDownsample" + std::to_string(level);
        bloomMipDownsampleImages_[level].create(context_, bloomInfo);
    }

    const uint32_t bloomUpsampleCount = bloomMipCount > 1u ? bloomMipCount - 1u : 0u;
    bloomMipUpsampleImages_.resize(bloomUpsampleCount);
    bloomMipUpsampleLayouts_.assign(bloomUpsampleCount, VK_IMAGE_LAYOUT_UNDEFINED);
    for (uint32_t level = 0; level < bloomUpsampleCount; ++level) {
        const VkExtent2D mipSize = bloomMipExtent(sceneAllocatedExtent(), level);
        bloomInfo.width = mipSize.width;
        bloomInfo.height = mipSize.height;
        bloomInfo.debugName = "BloomMipUpsample" + std::to_string(level);
        bloomMipUpsampleImages_[level].create(context_, bloomInfo);
    }

    try {
        createLuminanceResources();
    } catch (const std::exception& error) {
        disableLogAverageExposureFallback(std::string("Log-average exposure luminance resources unavailable: ") +
                                          error.what());
    }

    try {
        createHistogramResources();
    } catch (const std::exception& error) {
        disableHistogramExposureFallback(std::string("Histogram exposure resources unavailable: ") + error.what());
    }

    createExposureResources();
    createPostProcessDescriptorSets();
}

// Exposure compute pipelines (luminance / histogram / reduce). The depth-pyramid
// compute pipeline that shared the former Renderer::createComputePipelines stays
// in Renderer.
void PostProcessStack::createExposureComputePipelines()
{
    const VkDescriptorSetLayout postProcessExposureReduceDescriptorSetLayout =
        postProcessExposureReduceDescriptorSetLayout_.handle();

    luminancePipeline_.reset();
    histogramPipeline_.reset();
    exposureReducePipeline_.reset();

    if (toneMappingSettings_.enableAutoExposure &&
        postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE) {
        const VkDescriptorSetLayout exposureDescriptorSetLayout = postProcessLuminanceDescriptorSetLayout_.handle();
        try {
            const VkPushConstantRange luminancePushConstantRange{
                VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(LuminancePushConstants))};

            rhi::VulkanComputePipelineCreateInfo luminancePipelineInfo{};
            luminancePipelineInfo.shaderPath = shaderPath("luminance.comp.spv");
            luminancePipelineInfo.descriptorSetLayouts =
                std::span<const VkDescriptorSetLayout>(&exposureDescriptorSetLayout, 1);
            luminancePipelineInfo.pushConstantRanges =
                std::span<const VkPushConstantRange>(&luminancePushConstantRange, 1);
            luminancePipelineInfo.pipelineCache = context_.pipelineCache();
            luminancePipeline_.create(context_.vkDevice(), luminancePipelineInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      luminancePipeline_.pipeline(),
                                      VK_OBJECT_TYPE_PIPELINE,
                                      "AutoExposureComputePipeline");
            rhi::debug::setObjectName(context_.vkDevice(),
                                      luminancePipeline_.layout(),
                                      VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                      "AutoExposurePipelineLayout");
        } catch (const std::exception& error) {
            disableLogAverageExposureFallback(std::string("Log-average exposure compute pipeline creation failed: ") +
                                              error.what());
        }

        try {
            const VkPushConstantRange histogramPushConstantRange{
                VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(HistogramPushConstants))};

            rhi::VulkanComputePipelineCreateInfo histogramPipelineInfo{};
            histogramPipelineInfo.shaderPath = shaderPath("luminance_histogram.comp.spv");
            histogramPipelineInfo.descriptorSetLayouts =
                std::span<const VkDescriptorSetLayout>(&exposureDescriptorSetLayout, 1);
            histogramPipelineInfo.pushConstantRanges =
                std::span<const VkPushConstantRange>(&histogramPushConstantRange, 1);
            histogramPipelineInfo.pipelineCache = context_.pipelineCache();
            histogramPipeline_.create(context_.vkDevice(), histogramPipelineInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      histogramPipeline_.pipeline(),
                                      VK_OBJECT_TYPE_PIPELINE,
                                      "HistogramExposureComputePipeline");
            rhi::debug::setObjectName(context_.vkDevice(),
                                      histogramPipeline_.layout(),
                                      VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                      "HistogramExposurePipelineLayout");
        } catch (const std::exception& error) {
            disableHistogramExposureFallback(std::string("Histogram exposure compute pipeline creation failed: ") +
                                             error.what());
        }

        if (postProcessExposureReduceDescriptorSetLayout != VK_NULL_HANDLE) {
            try {
                const VkPushConstantRange exposureReducePushConstantRange{
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(ExposureReducePushConstants))};

                rhi::VulkanComputePipelineCreateInfo exposureReducePipelineInfo{};
                exposureReducePipelineInfo.shaderPath = shaderPath("exposure_reduce.comp.spv");
                exposureReducePipelineInfo.descriptorSetLayouts =
                    std::span<const VkDescriptorSetLayout>(&postProcessExposureReduceDescriptorSetLayout, 1);
                exposureReducePipelineInfo.pushConstantRanges =
                    std::span<const VkPushConstantRange>(&exposureReducePushConstantRange, 1);
                exposureReducePipelineInfo.pipelineCache = context_.pipelineCache();
                exposureReducePipeline_.create(context_.vkDevice(), exposureReducePipelineInfo);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          exposureReducePipeline_.pipeline(),
                                          VK_OBJECT_TYPE_PIPELINE,
                                          "ExposureReduceComputePipeline");
                rhi::debug::setObjectName(context_.vkDevice(),
                                          exposureReducePipeline_.layout(),
                                          VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                          "ExposureReducePipelineLayout");
            } catch (const std::exception& error) {
                disableAutoExposureFallback(std::string("GPU exposure reduce compute pipeline creation failed: ") +
                                            error.what());
            }
        }
    }
}

// Composite pass (relocated from Renderer::recordRenderCommands). The jittered
// projection is passed in (it is Renderer frame state); ssaoAvailable_ is a
// borrowed reference written by createCompositeDescriptorSets.
void PostProcessStack::recordCompositeCommands(VkCommandBuffer commandBuffer,
                                               const glm::mat4& jitteredProjection,
                                               float debugRawGain,
                                               float sharpness,
                                               float sharpenDebugGain)
{
    rhi::debug::beginLabel(commandBuffer, "CompositePass");
    const bool compositeProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "CompositePass");
    renderGraph_.beginCompositePass();
    // The one pass that runs at presentation resolution: it samples the scene
    // colour with a linear sampler over normalised UVs, so a reduced render
    // extent upscales here for free.
    setViewportAndScissor(commandBuffer, renderResolution_.outputExtent());
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_.pipeline());
    const VkDescriptorSet compositeDescriptorSet = activeCompositeDescriptorSet();
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            compositePipeline_.layout(),
                            0,
                            1,
                            &compositeDescriptorSet,
                            0,
                            nullptr);
    CompositePushConstants compositePushConstants{
        currentToneMappingExposure(),
        bloomSettings_.enabled ? std::max(bloomSettings_.intensity, 0.0f) : 0.0f,
        toneMappingOperatorValue(toneMappingSettings_.operatorType),
        bloomSettings_.enabled ? 1u : 0u,
        bloomSettings_.useMipChain && (!bloomMipUpsampleImages_.empty() || !bloomMipDownsampleImages_.empty()) ? 1u
                                                                                                               : 0u,
        isGpuExposureActive() ? 1u : 0u};
    // GTAO is computed in its own pass now; the composite only multiplies the
    // precomputed visibility term, gated by this enable flag. invProjection is
    // retained in the push block (still fed from the jittered projection) for
    // future depth-driven composite effects.
    // Only the reference path multiplies here. With ambientOnly set -- the
    // default -- the main pass has already applied occlusion to the ambient
    // term, and multiplying again would both double-apply it and put it back on
    // the direct lighting this change exists to spare.
    const bool ssaoActive = ssaoSettings_.enabled && ssaoAvailable_ && !ssaoSettings_.ambientOnly;
    compositePushConstants.invProjection = glm::inverse(jitteredProjection);
    // zw is the scene/AO uv scale: both are sub-rected and share an allocation,
    // so one pair covers them. Bloom is not sub-rected and is sampled unscaled.
    compositePushConstants.debugParams =
        glm::vec4(std::max(sharpenDebugGain, 0.0f), 0.0f, renderResolution_.uvScale());
    // zw is the bloom uv scale: the bloom chain is sub-rected too, and its halves
    // round independently of the scene's, so it cannot share debugParams.zw.
    compositePushConstants.ssaoParams1 = glm::vec4(ssaoActive ? 1.0f : 0.0f, 0.0f, bloomUvScale());
    compositePushConstants.debugRawGain = std::max(debugRawGain, 0.0f);
    // Sharpening exists to undo the softness of the upscale, so a native frame
    // gets none of it and stays bit-identical to what it rendered before the
    // filter existed. The shader reads a zero here as "off" and skips the four
    // extra taps entirely.
    compositePushConstants.sharpness = renderResolution_.isNative() ? 0.0f : std::clamp(sharpness, 0.0f, 1.0f);
    vkCmdPushConstants(commandBuffer,
                       compositePipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(CompositePushConstants)),
                       &compositePushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endCompositePass();
    if (compositeProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
}

void PostProcessStack::createPostProcessDescriptorSetLayouts()
{
    VkDescriptorSetLayoutBinding singleImageBinding{};
    singleImageBinding.binding = 0;
    singleImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    singleImageBinding.descriptorCount = 1;
    singleImageBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    postProcessSingleImageDescriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(&singleImageBinding, 1));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessSingleImageDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessSingleImageDescriptorSetLayout");

    std::array<VkDescriptorSetLayoutBinding, 2> dualImageBindings{};
    dualImageBindings[0] = singleImageBinding;
    dualImageBindings[1] = singleImageBinding;
    dualImageBindings[1].binding = 1;

    postProcessDualImageDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(dualImageBindings.data(), dualImageBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessDualImageDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessDualImageDescriptorSetLayout");

    // TAA resolve: current color, history color, velocity, main depth.
    std::array<VkDescriptorSetLayoutBinding, 4> taaResolveBindings{};
    for (uint32_t bindingIndex = 0; bindingIndex < taaResolveBindings.size(); ++bindingIndex) {
        taaResolveBindings[bindingIndex] = singleImageBinding;
        taaResolveBindings[bindingIndex].binding = bindingIndex;
    }

    postProcessTaaResolveDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(taaResolveBindings.data(), taaResolveBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessTaaResolveDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessTaaResolveDescriptorSetLayout");

    std::array<VkDescriptorSetLayoutBinding, 5> compositeBindings{};
    compositeBindings[0] = singleImageBinding;
    compositeBindings[1] = singleImageBinding;
    compositeBindings[1].binding = 1;
    compositeBindings[2] = singleImageBinding;
    compositeBindings[2].binding = 2;
    compositeBindings[3].binding = 3;
    compositeBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compositeBindings[3].descriptorCount = 1;
    compositeBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    compositeBindings[4] = singleImageBinding; // GTAO visibility term, multiplied into scene color
    compositeBindings[4].binding = 4;

    postProcessCompositeDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(compositeBindings.data(), compositeBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessCompositeDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessCompositeDescriptorSetLayout");

    std::array<VkDescriptorSetLayoutBinding, 2> luminanceBindings{};
    luminanceBindings[0].binding = 0;
    luminanceBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    luminanceBindings[0].descriptorCount = 1;
    luminanceBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    luminanceBindings[1].binding = 1;
    luminanceBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    luminanceBindings[1].descriptorCount = 1;
    luminanceBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    try {
        postProcessLuminanceDescriptorSetLayout_.create(
            context_.vkDevice(),
            std::span<const VkDescriptorSetLayoutBinding>(luminanceBindings.data(), luminanceBindings.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  postProcessLuminanceDescriptorSetLayout_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                  "PostProcessLuminanceDescriptorSetLayout");
    } catch (const std::exception& error) {
        disableAutoExposureFallback(std::string("Auto exposure descriptor layout creation failed: ") + error.what());
    }

    std::array<VkDescriptorSetLayoutBinding, 3> exposureReduceBindings{};
    for (uint32_t bindingIndex = 0; bindingIndex < exposureReduceBindings.size(); ++bindingIndex) {
        exposureReduceBindings[bindingIndex].binding = bindingIndex;
        exposureReduceBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureReduceBindings[bindingIndex].descriptorCount = 1;
        exposureReduceBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    try {
        postProcessExposureReduceDescriptorSetLayout_.create(
            context_.vkDevice(),
            std::span<const VkDescriptorSetLayoutBinding>(exposureReduceBindings.data(),
                                                          exposureReduceBindings.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  postProcessExposureReduceDescriptorSetLayout_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                  "PostProcessExposureReduceDescriptorSetLayout");
    } catch (const std::exception& error) {
        disableAutoExposureFallback(std::string("Exposure reduce descriptor layout creation failed: ") + error.what());
    }
}

void PostProcessStack::createPostProcessSampler()
{
    destroyPostProcessSampler();

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &postProcessSampler_));
    rhi::debug::setObjectName(
        context_.vkDevice(), postProcessSampler_, VK_OBJECT_TYPE_SAMPLER, "PostProcessLinearClampSampler");
}

void PostProcessStack::destroyPostProcessSampler()
{
    if (postProcessSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), postProcessSampler_, nullptr);
        postProcessSampler_ = VK_NULL_HANDLE;
    }
}

void PostProcessStack::createTaaResources()
{
    destroyTaaResources();

    // Follows the scene targets above.
    const VkExtent2D extent = sceneAllocatedExtent();
    if (extent.width == 0 || extent.height == 0) {
        throw std::runtime_error("Cannot create TAA history resources for a zero-sized render extent.");
    }

    rhi::VulkanImageCreateInfo historyInfo{};
    historyInfo.width = extent.width;
    historyInfo.height = extent.height;
    historyInfo.format = kSceneColorFormat;
    historyInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    historyInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    for (uint32_t index = 0; index < kTaaHistoryCount; ++index) {
        historyInfo.debugName = "TAAHistory" + std::to_string(index);
        taaHistoryImages_[index].create(context_, historyInfo);
        taaHistoryLayouts_[index] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    invalidateTaaHistory();
}

void PostProcessStack::destroyTaaResources()
{
    taaResolveDescriptorSets_.fill(VK_NULL_HANDLE);
    taaBloomExtractDescriptorSets_.fill(VK_NULL_HANDLE);
    taaBloomMipDownsampleDescriptorSets_.fill(VK_NULL_HANDLE);
    for (auto& descriptorSets : taaCompositeDescriptorSets_) {
        descriptorSets.clear();
    }
    for (auto& descriptorSets : taaLuminanceDescriptorSets_) {
        descriptorSets.clear();
    }
    for (auto& descriptorSets : taaHistogramDescriptorSets_) {
        descriptorSets.clear();
    }

    for (rhi::VulkanImage& image : taaHistoryImages_) {
        image.reset();
    }
    taaHistoryLayouts_.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    invalidateTaaHistory();
}

void PostProcessStack::invalidateTaaHistory()
{
    taaHistoryValid_ = false;
    taaHistoryWriteIndex_ = 0;
    taaPostProcessHistoryIndex_ = 0;
    taaJitterIndex_ = 0;
    taaCurrentJitterPixels_ = {0.0f, 0.0f};
    taaPreviousJitterPixels_ = {0.0f, 0.0f};
    taaCurrentJitterNdc_ = {0.0f, 0.0f};
    taaPreviousJitterNdc_ = {0.0f, 0.0f};
    // The jittered/previous view-projection matrices stay in Renderer (they are
    // consumed by the main pass); Renderer::invalidateTaaHistory resets them and
    // then delegates here for the TAA-owned state.
}

VkDescriptorImageInfo PostProcessStack::postProcessImageInfo(VkImageView imageView) const
{
    VkDescriptorImageInfo info{};
    info.sampler = postProcessSampler_;
    info.imageView = imageView;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return info;
}

VkDescriptorImageInfo PostProcessStack::postProcessDepthInfo() const
{
    VkDescriptorImageInfo info{};
    info.sampler = postProcessSampler_;
    if (swapchain_.depthSupportsSampling()) {
        info.imageView = swapchain_.depthImageView();
        info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    } else {
        info.imageView = depthFallbackView_;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    return info;
}

PostProcessStack::PostProcessDescriptorCounts PostProcessStack::computePostProcessDescriptorCounts() const
{
    PostProcessDescriptorCounts counts{};
    counts.createLuminanceDescriptors = autoExposureAvailable_ && !frameLuminanceBuffers_.empty() &&
                                        frameLuminanceBuffers_.size() == frameCount_ &&
                                        postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    counts.createHistogramDescriptors = histogramExposureAvailable_ && !frameHistogramBuffers_.empty() &&
                                        frameHistogramBuffers_.size() == frameCount_ &&
                                        postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    counts.createExposureReduceDescriptors = counts.createLuminanceDescriptors && counts.createHistogramDescriptors &&
                                             exposureReduceAvailable_ && frameExposureBuffers_.size() == frameCount_ &&
                                             postProcessExposureReduceDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    counts.createCompositeDescriptors = !frameExposureBuffers_.empty() && frameExposureBuffers_.size() == frameCount_ &&
                                        postProcessCompositeDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    counts.exposureDescriptorSetCount =
        (counts.createLuminanceDescriptors ? static_cast<uint32_t>(frameCount_) : 0u) +
        (counts.createHistogramDescriptors ? static_cast<uint32_t>(frameCount_) : 0u) +
        (counts.createExposureReduceDescriptors ? static_cast<uint32_t>(frameCount_) : 0u);
    counts.compositeDescriptorSetCount = counts.createCompositeDescriptors ? static_cast<uint32_t>(frameCount_) : 0u;
    counts.legacyBloomSetCount = 3u;
    counts.bloomDownsampleSetCount = static_cast<uint32_t>(bloomMipDownsampleImages_.size());
    counts.bloomUpsampleSetCount = static_cast<uint32_t>(bloomMipUpsampleImages_.size());
    counts.taaResolveSetCount = kTaaHistoryCount;
    counts.taaBloomExtractSetCount = kTaaHistoryCount;
    counts.taaBloomDownsampleSetCount = bloomMipDownsampleImages_.empty() ? 0u : kTaaHistoryCount;
    counts.taaCompositeDescriptorSetCount =
        counts.createCompositeDescriptors ? kTaaHistoryCount * static_cast<uint32_t>(frameCount_) : 0u;
    counts.taaLuminanceDescriptorSetCount =
        counts.createLuminanceDescriptors ? kTaaHistoryCount * static_cast<uint32_t>(frameCount_) : 0u;
    counts.taaHistogramDescriptorSetCount =
        counts.createHistogramDescriptors ? kTaaHistoryCount * static_cast<uint32_t>(frameCount_) : 0u;
    return counts;
}

void PostProcessStack::createPostProcessDescriptorPool(const PostProcessDescriptorCounts& counts)
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = counts.legacyBloomSetCount + counts.bloomDownsampleSetCount +
                                   (2u * counts.bloomUpsampleSetCount) + (4u * counts.compositeDescriptorSetCount) +
                                   (counts.createLuminanceDescriptors ? static_cast<uint32_t>(frameCount_) : 0u) +
                                   (counts.createHistogramDescriptors ? static_cast<uint32_t>(frameCount_) : 0u) +
                                   (4u * counts.taaResolveSetCount) + counts.taaBloomExtractSetCount +
                                   counts.taaBloomDownsampleSetCount + (4u * counts.taaCompositeDescriptorSetCount) +
                                   counts.taaLuminanceDescriptorSetCount + counts.taaHistogramDescriptorSetCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount =
        (counts.createLuminanceDescriptors ? static_cast<uint32_t>(frameCount_) : 0u) +
        (counts.createHistogramDescriptors ? static_cast<uint32_t>(frameCount_) : 0u) +
        (3u * (counts.createExposureReduceDescriptors ? static_cast<uint32_t>(frameCount_) : 0u)) +
        counts.compositeDescriptorSetCount + counts.taaCompositeDescriptorSetCount +
        counts.taaLuminanceDescriptorSetCount + counts.taaHistogramDescriptorSetCount;
    const uint32_t poolSizeCount = poolSizes[1].descriptorCount > 0 ? 2u : 1u;
    const uint32_t maxSets =
        counts.legacyBloomSetCount + counts.bloomDownsampleSetCount + counts.bloomUpsampleSetCount +
        counts.compositeDescriptorSetCount + counts.exposureDescriptorSetCount + counts.taaResolveSetCount +
        counts.taaBloomExtractSetCount + counts.taaBloomDownsampleSetCount + counts.taaCompositeDescriptorSetCount +
        counts.taaLuminanceDescriptorSetCount + counts.taaHistogramDescriptorSetCount;

    postProcessDescriptorPool_.create(
        context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizeCount), maxSets);
    rhi::debug::setObjectName(
        context_.vkDevice(), postProcessDescriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "PostProcessPool");
}

void PostProcessStack::createPostProcessDescriptorSets()
{
    if (postProcessSampler_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot create post-process descriptors without a sampler.");
    }

    const PostProcessDescriptorCounts counts = computePostProcessDescriptorCounts();
    createPostProcessDescriptorPool(counts);

    allocateLegacyBloomDescriptorSets();
    createTaaResolveDescriptorSets();
    createBloomMipDownsampleDescriptorSets();
    createBloomMipUpsampleDescriptorSets();
    createCompositeDescriptorSets(counts);
    createLuminanceDescriptorSets(counts);
    createHistogramDescriptorSets(counts);
    createExposureReduceDescriptorSets(counts);
}

void PostProcessStack::allocateLegacyBloomDescriptorSets()
{
    std::array<VkDescriptorSetLayout, 3> legacyDescriptorSetLayouts{
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessSingleImageDescriptorSetLayout_.handle(),
    };
    std::array<VkDescriptorSet, 3> legacyDescriptorSets{};

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(legacyDescriptorSetLayouts.size());
    allocateInfo.pSetLayouts = legacyDescriptorSetLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, legacyDescriptorSets.data()));

    bloomExtractDescriptorSet_ = legacyDescriptorSets[0];
    bloomBlurHorizontalDescriptorSet_ = legacyDescriptorSets[1];
    bloomBlurVerticalDescriptorSet_ = legacyDescriptorSets[2];

    rhi::debug::setObjectName(
        context_.vkDevice(), bloomExtractDescriptorSet_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "BloomExtractDescriptorSet");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomBlurHorizontalDescriptorSet_,
                              VK_OBJECT_TYPE_DESCRIPTOR_SET,
                              "BloomBlurHorizontalDescriptorSet");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomBlurVerticalDescriptorSet_,
                              VK_OBJECT_TYPE_DESCRIPTOR_SET,
                              "BloomBlurVerticalDescriptorSet");

    std::array<VkDescriptorImageInfo, 3> legacyImageInfos{
        postProcessImageInfo(sceneColor_.imageView()),
        postProcessImageInfo(bloomExtract_.imageView()),
        postProcessImageInfo(bloomPing_.imageView()),
    };

    std::array<VkWriteDescriptorSet, 3> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = bloomExtractDescriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &legacyImageInfos[0];

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = bloomBlurHorizontalDescriptorSet_;
    writes[1].dstBinding = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &legacyImageInfos[1];

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = bloomBlurVerticalDescriptorSet_;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &legacyImageInfos[2];

    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PostProcessStack::createTaaResolveDescriptorSets()
{
    if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE && taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
        std::array<VkDescriptorSetLayout, kTaaHistoryCount> taaResolveLayouts{
            postProcessTaaResolveDescriptorSetLayout_.handle(),
            postProcessTaaResolveDescriptorSetLayout_.handle(),
        };
        VkDescriptorSetAllocateInfo taaResolveAllocateInfo{};
        taaResolveAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        taaResolveAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        taaResolveAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaResolveLayouts.size());
        taaResolveAllocateInfo.pSetLayouts = taaResolveLayouts.data();
        VK_CHECK(
            vkAllocateDescriptorSets(context_.vkDevice(), &taaResolveAllocateInfo, taaResolveDescriptorSets_.data()));

        std::array<VkDescriptorSetLayout, kTaaHistoryCount> taaSingleImageLayouts{
            postProcessSingleImageDescriptorSetLayout_.handle(),
            postProcessSingleImageDescriptorSetLayout_.handle(),
        };
        VkDescriptorSetAllocateInfo taaBloomAllocateInfo{};
        taaBloomAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        taaBloomAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        taaBloomAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaSingleImageLayouts.size());
        taaBloomAllocateInfo.pSetLayouts = taaSingleImageLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &taaBloomAllocateInfo, taaBloomExtractDescriptorSets_.data()));

        std::array<std::array<VkDescriptorImageInfo, 4>, kTaaHistoryCount> taaResolveImageInfos{};
        std::array<VkDescriptorImageInfo, kTaaHistoryCount> taaBloomImageInfos{};
        std::array<VkWriteDescriptorSet, kTaaHistoryCount * 5u> taaWrites{};
        for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
            taaResolveImageInfos[historyIndex][0] = postProcessImageInfo(sceneColor_.imageView());
            taaResolveImageInfos[historyIndex][1] = postProcessImageInfo(taaHistoryImages_[historyIndex].imageView());
            taaResolveImageInfos[historyIndex][2] = postProcessImageInfo(velocity_.imageView());
            taaResolveImageInfos[historyIndex][3] = postProcessDepthInfo();
            taaBloomImageInfos[historyIndex] = postProcessImageInfo(taaHistoryImages_[historyIndex].imageView());

            const uint32_t writeBase = historyIndex * 5u;
            for (uint32_t binding = 0; binding < 4u; ++binding) {
                taaWrites[writeBase + binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                taaWrites[writeBase + binding].dstSet = taaResolveDescriptorSets_[historyIndex];
                taaWrites[writeBase + binding].dstBinding = binding;
                taaWrites[writeBase + binding].descriptorCount = 1;
                taaWrites[writeBase + binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                taaWrites[writeBase + binding].pImageInfo = &taaResolveImageInfos[historyIndex][binding];
            }

            taaWrites[writeBase + 4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            taaWrites[writeBase + 4].dstSet = taaBloomExtractDescriptorSets_[historyIndex];
            taaWrites[writeBase + 4].dstBinding = 0;
            taaWrites[writeBase + 4].descriptorCount = 1;
            taaWrites[writeBase + 4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            taaWrites[writeBase + 4].pImageInfo = &taaBloomImageInfos[historyIndex];

            rhi::debug::setObjectName(context_.vkDevice(),
                                      taaResolveDescriptorSets_[historyIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "TAAResolveDescriptorSet" + std::to_string(historyIndex));
            rhi::debug::setObjectName(context_.vkDevice(),
                                      taaBloomExtractDescriptorSets_[historyIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "TAABloomExtractDescriptorSet" + std::to_string(historyIndex));
        }
        vkUpdateDescriptorSets(
            context_.vkDevice(), static_cast<uint32_t>(taaWrites.size()), taaWrites.data(), 0, nullptr);

        if (!bloomMipDownsampleImages_.empty()) {
            VkDescriptorSetAllocateInfo taaDownsampleAllocateInfo{};
            taaDownsampleAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            taaDownsampleAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            taaDownsampleAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaSingleImageLayouts.size());
            taaDownsampleAllocateInfo.pSetLayouts = taaSingleImageLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(
                context_.vkDevice(), &taaDownsampleAllocateInfo, taaBloomMipDownsampleDescriptorSets_.data()));

            std::array<VkDescriptorImageInfo, kTaaHistoryCount> taaDownsampleImageInfos{};
            std::array<VkWriteDescriptorSet, kTaaHistoryCount> taaDownsampleWrites{};
            for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                taaDownsampleImageInfos[historyIndex] =
                    postProcessImageInfo(taaHistoryImages_[historyIndex].imageView());
                taaDownsampleWrites[historyIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                taaDownsampleWrites[historyIndex].dstSet = taaBloomMipDownsampleDescriptorSets_[historyIndex];
                taaDownsampleWrites[historyIndex].dstBinding = 0;
                taaDownsampleWrites[historyIndex].descriptorCount = 1;
                taaDownsampleWrites[historyIndex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                taaDownsampleWrites[historyIndex].pImageInfo = &taaDownsampleImageInfos[historyIndex];
                rhi::debug::setObjectName(context_.vkDevice(),
                                          taaBloomMipDownsampleDescriptorSets_[historyIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "TAABloomMipDownsampleDescriptorSet" + std::to_string(historyIndex));
            }
            vkUpdateDescriptorSets(context_.vkDevice(),
                                   static_cast<uint32_t>(taaDownsampleWrites.size()),
                                   taaDownsampleWrites.data(),
                                   0,
                                   nullptr);
        }
    }
}

void PostProcessStack::createBloomMipDownsampleDescriptorSets()
{
    if (!bloomMipDownsampleImages_.empty()) {
        bloomMipDownsampleDescriptorSets_.assign(bloomMipDownsampleImages_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> downsampleLayouts(bloomMipDownsampleImages_.size(),
                                                             postProcessSingleImageDescriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo downsampleAllocateInfo{};
        downsampleAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        downsampleAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        downsampleAllocateInfo.descriptorSetCount = static_cast<uint32_t>(downsampleLayouts.size());
        downsampleAllocateInfo.pSetLayouts = downsampleLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &downsampleAllocateInfo, bloomMipDownsampleDescriptorSets_.data()));

        std::vector<VkDescriptorImageInfo> downsampleImageInfos(bloomMipDownsampleImages_.size());
        std::vector<VkWriteDescriptorSet> downsampleWrites(bloomMipDownsampleImages_.size());
        for (size_t level = 0; level < bloomMipDownsampleImages_.size(); ++level) {
            const VkImageView sourceView =
                level == 0 ? sceneColor_.imageView() : bloomMipDownsampleImages_[level - 1].imageView();
            downsampleImageInfos[level] = postProcessImageInfo(sourceView);
            downsampleWrites[level].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            downsampleWrites[level].dstSet = bloomMipDownsampleDescriptorSets_[level];
            downsampleWrites[level].dstBinding = 0;
            downsampleWrites[level].descriptorCount = 1;
            downsampleWrites[level].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            downsampleWrites[level].pImageInfo = &downsampleImageInfos[level];
            rhi::debug::setObjectName(context_.vkDevice(),
                                      bloomMipDownsampleDescriptorSets_[level],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "BloomMipDownsampleDescriptorSet" + std::to_string(level));
        }
        vkUpdateDescriptorSets(
            context_.vkDevice(), static_cast<uint32_t>(downsampleWrites.size()), downsampleWrites.data(), 0, nullptr);
    }
}

void PostProcessStack::createBloomMipUpsampleDescriptorSets()
{
    if (!bloomMipUpsampleImages_.empty()) {
        bloomMipUpsampleDescriptorSets_.assign(bloomMipUpsampleImages_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> upsampleLayouts(bloomMipUpsampleImages_.size(),
                                                           postProcessDualImageDescriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo upsampleAllocateInfo{};
        upsampleAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        upsampleAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        upsampleAllocateInfo.descriptorSetCount = static_cast<uint32_t>(upsampleLayouts.size());
        upsampleAllocateInfo.pSetLayouts = upsampleLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &upsampleAllocateInfo, bloomMipUpsampleDescriptorSets_.data()));

        std::vector<std::array<VkDescriptorImageInfo, 2>> upsampleImageInfos(bloomMipUpsampleImages_.size());
        std::vector<VkWriteDescriptorSet> upsampleWrites(bloomMipUpsampleImages_.size() * 2u);
        for (size_t level = 0; level < bloomMipUpsampleImages_.size(); ++level) {
            const VkImageView lowerView = level + 1 == bloomMipDownsampleImages_.size() - 1
                                              ? bloomMipDownsampleImages_[level + 1].imageView()
                                              : bloomMipUpsampleImages_[level + 1].imageView();
            upsampleImageInfos[level][0] = postProcessImageInfo(bloomMipDownsampleImages_[level].imageView());
            upsampleImageInfos[level][1] = postProcessImageInfo(lowerView);

            const size_t writeBase = level * 2u;
            upsampleWrites[writeBase].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            upsampleWrites[writeBase].dstSet = bloomMipUpsampleDescriptorSets_[level];
            upsampleWrites[writeBase].dstBinding = 0;
            upsampleWrites[writeBase].descriptorCount = 1;
            upsampleWrites[writeBase].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            upsampleWrites[writeBase].pImageInfo = &upsampleImageInfos[level][0];

            upsampleWrites[writeBase + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            upsampleWrites[writeBase + 1].dstSet = bloomMipUpsampleDescriptorSets_[level];
            upsampleWrites[writeBase + 1].dstBinding = 1;
            upsampleWrites[writeBase + 1].descriptorCount = 1;
            upsampleWrites[writeBase + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            upsampleWrites[writeBase + 1].pImageInfo = &upsampleImageInfos[level][1];

            rhi::debug::setObjectName(context_.vkDevice(),
                                      bloomMipUpsampleDescriptorSets_[level],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "BloomMipUpsampleDescriptorSet" + std::to_string(level));
        }
        vkUpdateDescriptorSets(
            context_.vkDevice(), static_cast<uint32_t>(upsampleWrites.size()), upsampleWrites.data(), 0, nullptr);
    }
}

void PostProcessStack::createCompositeDescriptorSets(const PostProcessDescriptorCounts& counts)
{
    if (counts.createCompositeDescriptors) {
        compositeDescriptorSets_.assign(frameCount_, VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> compositeLayouts(frameCount_,
                                                            postProcessCompositeDescriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo compositeAllocateInfo{};
        compositeAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        compositeAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        compositeAllocateInfo.descriptorSetCount = static_cast<uint32_t>(compositeDescriptorSets_.size());
        compositeAllocateInfo.pSetLayouts = compositeLayouts.data();
        VK_CHECK(
            vkAllocateDescriptorSets(context_.vkDevice(), &compositeAllocateInfo, compositeDescriptorSets_.data()));
        compositeDescriptorSet_ = compositeDescriptorSets_.empty() ? VK_NULL_HANDLE : compositeDescriptorSets_.front();

        std::vector<std::array<VkDescriptorImageInfo, 4>> compositeImageInfos(frameCount_);
        std::vector<VkDescriptorBufferInfo> compositeExposureInfos(frameCount_);
        std::vector<VkWriteDescriptorSet> compositeWrites(frameCount_ * 5u);
        VkImageView mipBloomView = bloomPong_.imageView();
        if (!bloomMipUpsampleImages_.empty()) {
            mipBloomView = bloomMipUpsampleImages_.front().imageView();
        } else if (!bloomMipDownsampleImages_.empty()) {
            mipBloomView = bloomMipDownsampleImages_.front().imageView();
        }

        // The composite multiplies the GTAO visibility term into scene color. The
        // subsystem gates on a samplable depth image (same requirement the GTAO
        // horizon search has), so mirror that availability flag here.
        ssaoAvailable_ = swapchain_.depthSupportsSampling();
        const VkDescriptorImageInfo compositeAmbientOcclusionInfo = postProcessImageInfo(ambientOcclusion_.imageView());
        for (size_t frameIndex = 0; frameIndex < frameCount_; ++frameIndex) {
            compositeImageInfos[frameIndex][0] = postProcessImageInfo(sceneColor_.imageView());
            compositeImageInfos[frameIndex][1] = postProcessImageInfo(bloomPong_.imageView());
            compositeImageInfos[frameIndex][2] = postProcessImageInfo(mipBloomView);
            compositeImageInfos[frameIndex][3] = compositeAmbientOcclusionInfo;

            compositeExposureInfos[frameIndex].buffer = frameExposureBuffers_[frameIndex].buffer();
            compositeExposureInfos[frameIndex].offset = 0;
            compositeExposureInfos[frameIndex].range = sizeof(ExposureState);

            const size_t writeBase = frameIndex * 5u;
            for (uint32_t binding = 0; binding < 3; ++binding) {
                compositeWrites[writeBase + binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                compositeWrites[writeBase + binding].dstSet = compositeDescriptorSets_[frameIndex];
                compositeWrites[writeBase + binding].dstBinding = binding;
                compositeWrites[writeBase + binding].descriptorCount = 1;
                compositeWrites[writeBase + binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                compositeWrites[writeBase + binding].pImageInfo = &compositeImageInfos[frameIndex][binding];
            }

            compositeWrites[writeBase + 3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compositeWrites[writeBase + 3].dstSet = compositeDescriptorSets_[frameIndex];
            compositeWrites[writeBase + 3].dstBinding = 3;
            compositeWrites[writeBase + 3].descriptorCount = 1;
            compositeWrites[writeBase + 3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            compositeWrites[writeBase + 3].pBufferInfo = &compositeExposureInfos[frameIndex];

            compositeWrites[writeBase + 4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compositeWrites[writeBase + 4].dstSet = compositeDescriptorSets_[frameIndex];
            compositeWrites[writeBase + 4].dstBinding = 4;
            compositeWrites[writeBase + 4].descriptorCount = 1;
            compositeWrites[writeBase + 4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            compositeWrites[writeBase + 4].pImageInfo = &compositeImageInfos[frameIndex][3];

            rhi::debug::setObjectName(context_.vkDevice(),
                                      compositeDescriptorSets_[frameIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "CompositeDescriptorSet" + std::to_string(frameIndex));
        }

        vkUpdateDescriptorSets(
            context_.vkDevice(), static_cast<uint32_t>(compositeWrites.size()), compositeWrites.data(), 0, nullptr);

        if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE && taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
            for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                auto& taaDescriptorSets = taaCompositeDescriptorSets_[historyIndex];
                taaDescriptorSets.assign(frameCount_, VK_NULL_HANDLE);
                std::vector<VkDescriptorSetLayout> taaCompositeLayouts(
                    frameCount_, postProcessCompositeDescriptorSetLayout_.handle());
                VkDescriptorSetAllocateInfo taaCompositeAllocateInfo{};
                taaCompositeAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                taaCompositeAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
                taaCompositeAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaDescriptorSets.size());
                taaCompositeAllocateInfo.pSetLayouts = taaCompositeLayouts.data();
                VK_CHECK(
                    vkAllocateDescriptorSets(context_.vkDevice(), &taaCompositeAllocateInfo, taaDescriptorSets.data()));

                std::vector<std::array<VkDescriptorImageInfo, 4>> taaCompositeImageInfos(frameCount_);
                std::vector<VkDescriptorBufferInfo> taaCompositeExposureInfos(frameCount_);
                std::vector<VkWriteDescriptorSet> taaCompositeWrites(frameCount_ * 5u);
                const VkDescriptorImageInfo taaCompositeAmbientOcclusionInfo =
                    postProcessImageInfo(ambientOcclusion_.imageView());
                for (size_t frameIndex = 0; frameIndex < frameCount_; ++frameIndex) {
                    taaCompositeImageInfos[frameIndex][0] =
                        postProcessImageInfo(taaHistoryImages_[historyIndex].imageView());
                    taaCompositeImageInfos[frameIndex][1] = postProcessImageInfo(bloomPong_.imageView());
                    taaCompositeImageInfos[frameIndex][2] = postProcessImageInfo(mipBloomView);
                    taaCompositeImageInfos[frameIndex][3] = taaCompositeAmbientOcclusionInfo;

                    taaCompositeExposureInfos[frameIndex].buffer = frameExposureBuffers_[frameIndex].buffer();
                    taaCompositeExposureInfos[frameIndex].offset = 0;
                    taaCompositeExposureInfos[frameIndex].range = sizeof(ExposureState);

                    const size_t writeBase = frameIndex * 5u;
                    for (uint32_t binding = 0; binding < 3; ++binding) {
                        taaCompositeWrites[writeBase + binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaCompositeWrites[writeBase + binding].dstSet = taaDescriptorSets[frameIndex];
                        taaCompositeWrites[writeBase + binding].dstBinding = binding;
                        taaCompositeWrites[writeBase + binding].descriptorCount = 1;
                        taaCompositeWrites[writeBase + binding].descriptorType =
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        taaCompositeWrites[writeBase + binding].pImageInfo =
                            &taaCompositeImageInfos[frameIndex][binding];
                    }

                    taaCompositeWrites[writeBase + 3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    taaCompositeWrites[writeBase + 3].dstSet = taaDescriptorSets[frameIndex];
                    taaCompositeWrites[writeBase + 3].dstBinding = 3;
                    taaCompositeWrites[writeBase + 3].descriptorCount = 1;
                    taaCompositeWrites[writeBase + 3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    taaCompositeWrites[writeBase + 3].pBufferInfo = &taaCompositeExposureInfos[frameIndex];

                    taaCompositeWrites[writeBase + 4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    taaCompositeWrites[writeBase + 4].dstSet = taaDescriptorSets[frameIndex];
                    taaCompositeWrites[writeBase + 4].dstBinding = 4;
                    taaCompositeWrites[writeBase + 4].descriptorCount = 1;
                    taaCompositeWrites[writeBase + 4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    taaCompositeWrites[writeBase + 4].pImageInfo = &taaCompositeImageInfos[frameIndex][3];

                    rhi::debug::setObjectName(context_.vkDevice(),
                                              taaDescriptorSets[frameIndex],
                                              VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                              "TAACompositeDescriptorSet" + std::to_string(historyIndex) + "_" +
                                                  std::to_string(frameIndex));
                }

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(taaCompositeWrites.size()),
                                       taaCompositeWrites.data(),
                                       0,
                                       nullptr);
            }
        }
    }
}

void PostProcessStack::createLuminanceDescriptorSets(const PostProcessDescriptorCounts& counts)
{
    const VkDescriptorImageInfo sceneColorInfo = postProcessImageInfo(sceneColor_.imageView());

    if (counts.createLuminanceDescriptors) {
        try {
            luminanceDescriptorSets_.assign(frameCount_, VK_NULL_HANDLE);
            std::vector<VkDescriptorSetLayout> luminanceLayouts(frameCount_,
                                                                postProcessLuminanceDescriptorSetLayout_.handle());
            VkDescriptorSetAllocateInfo luminanceAllocateInfo{};
            luminanceAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            luminanceAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            luminanceAllocateInfo.descriptorSetCount = static_cast<uint32_t>(luminanceDescriptorSets_.size());
            luminanceAllocateInfo.pSetLayouts = luminanceLayouts.data();
            VK_CHECK(
                vkAllocateDescriptorSets(context_.vkDevice(), &luminanceAllocateInfo, luminanceDescriptorSets_.data()));

            for (size_t frameIndex = 0; frameIndex < luminanceDescriptorSets_.size(); ++frameIndex) {
                VkDescriptorBufferInfo luminanceBufferInfo{};
                luminanceBufferInfo.buffer = frameLuminanceBuffers_[frameIndex].buffer();
                luminanceBufferInfo.offset = 0;
                luminanceBufferInfo.range = frameLuminanceBuffers_[frameIndex].size();

                std::array<VkWriteDescriptorSet, 2> luminanceWrites{};
                luminanceWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                luminanceWrites[0].dstSet = luminanceDescriptorSets_[frameIndex];
                luminanceWrites[0].dstBinding = 0;
                luminanceWrites[0].descriptorCount = 1;
                luminanceWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                luminanceWrites[0].pImageInfo = &sceneColorInfo;

                luminanceWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                luminanceWrites[1].dstSet = luminanceDescriptorSets_[frameIndex];
                luminanceWrites[1].dstBinding = 1;
                luminanceWrites[1].descriptorCount = 1;
                luminanceWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                luminanceWrites[1].pBufferInfo = &luminanceBufferInfo;

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(luminanceWrites.size()),
                                       luminanceWrites.data(),
                                       0,
                                       nullptr);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          luminanceDescriptorSets_[frameIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "LuminanceDescriptorSet" + std::to_string(frameIndex));
            }

            if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE &&
                taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
                for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                    auto& taaDescriptorSets = taaLuminanceDescriptorSets_[historyIndex];
                    taaDescriptorSets.assign(frameCount_, VK_NULL_HANDLE);
                    std::vector<VkDescriptorSetLayout> taaLuminanceLayouts(
                        frameCount_, postProcessLuminanceDescriptorSetLayout_.handle());
                    VkDescriptorSetAllocateInfo taaLuminanceAllocateInfo{};
                    taaLuminanceAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    taaLuminanceAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
                    taaLuminanceAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaDescriptorSets.size());
                    taaLuminanceAllocateInfo.pSetLayouts = taaLuminanceLayouts.data();
                    VK_CHECK(vkAllocateDescriptorSets(
                        context_.vkDevice(), &taaLuminanceAllocateInfo, taaDescriptorSets.data()));

                    for (size_t frameIndex = 0; frameIndex < taaDescriptorSets.size(); ++frameIndex) {
                        VkDescriptorImageInfo taaSceneColorInfo =
                            postProcessImageInfo(taaHistoryImages_[historyIndex].imageView());
                        VkDescriptorBufferInfo luminanceBufferInfo{};
                        luminanceBufferInfo.buffer = frameLuminanceBuffers_[frameIndex].buffer();
                        luminanceBufferInfo.offset = 0;
                        luminanceBufferInfo.range = frameLuminanceBuffers_[frameIndex].size();

                        std::array<VkWriteDescriptorSet, 2> taaLuminanceWrites{};
                        taaLuminanceWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaLuminanceWrites[0].dstSet = taaDescriptorSets[frameIndex];
                        taaLuminanceWrites[0].dstBinding = 0;
                        taaLuminanceWrites[0].descriptorCount = 1;
                        taaLuminanceWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        taaLuminanceWrites[0].pImageInfo = &taaSceneColorInfo;

                        taaLuminanceWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaLuminanceWrites[1].dstSet = taaDescriptorSets[frameIndex];
                        taaLuminanceWrites[1].dstBinding = 1;
                        taaLuminanceWrites[1].descriptorCount = 1;
                        taaLuminanceWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        taaLuminanceWrites[1].pBufferInfo = &luminanceBufferInfo;

                        vkUpdateDescriptorSets(context_.vkDevice(),
                                               static_cast<uint32_t>(taaLuminanceWrites.size()),
                                               taaLuminanceWrites.data(),
                                               0,
                                               nullptr);
                        rhi::debug::setObjectName(context_.vkDevice(),
                                                  taaDescriptorSets[frameIndex],
                                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                                  "TAALuminanceDescriptorSet" + std::to_string(historyIndex) + "_" +
                                                      std::to_string(frameIndex));
                    }
                }
            }
        } catch (const std::exception& error) {
            luminanceDescriptorSets_.clear();
            disableLogAverageExposureFallback(std::string("Log-average exposure descriptor allocation failed: ") +
                                              error.what());
        }
    }
}

void PostProcessStack::createHistogramDescriptorSets(const PostProcessDescriptorCounts& counts)
{
    const VkDescriptorImageInfo sceneColorInfo = postProcessImageInfo(sceneColor_.imageView());

    if (counts.createHistogramDescriptors) {
        try {
            histogramDescriptorSets_.assign(frameCount_, VK_NULL_HANDLE);
            std::vector<VkDescriptorSetLayout> histogramLayouts(frameCount_,
                                                                postProcessLuminanceDescriptorSetLayout_.handle());
            VkDescriptorSetAllocateInfo histogramAllocateInfo{};
            histogramAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            histogramAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            histogramAllocateInfo.descriptorSetCount = static_cast<uint32_t>(histogramDescriptorSets_.size());
            histogramAllocateInfo.pSetLayouts = histogramLayouts.data();
            VK_CHECK(
                vkAllocateDescriptorSets(context_.vkDevice(), &histogramAllocateInfo, histogramDescriptorSets_.data()));

            for (size_t frameIndex = 0; frameIndex < histogramDescriptorSets_.size(); ++frameIndex) {
                VkDescriptorBufferInfo histogramBufferInfo{};
                histogramBufferInfo.buffer = frameHistogramBuffers_[frameIndex].buffer();
                histogramBufferInfo.offset = 0;
                histogramBufferInfo.range = frameHistogramBuffers_[frameIndex].size();

                std::array<VkWriteDescriptorSet, 2> histogramWrites{};
                histogramWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                histogramWrites[0].dstSet = histogramDescriptorSets_[frameIndex];
                histogramWrites[0].dstBinding = 0;
                histogramWrites[0].descriptorCount = 1;
                histogramWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                histogramWrites[0].pImageInfo = &sceneColorInfo;

                histogramWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                histogramWrites[1].dstSet = histogramDescriptorSets_[frameIndex];
                histogramWrites[1].dstBinding = 1;
                histogramWrites[1].descriptorCount = 1;
                histogramWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                histogramWrites[1].pBufferInfo = &histogramBufferInfo;

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(histogramWrites.size()),
                                       histogramWrites.data(),
                                       0,
                                       nullptr);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          histogramDescriptorSets_[frameIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "HistogramDescriptorSet" + std::to_string(frameIndex));
            }

            if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE &&
                taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
                for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                    auto& taaDescriptorSets = taaHistogramDescriptorSets_[historyIndex];
                    taaDescriptorSets.assign(frameCount_, VK_NULL_HANDLE);
                    std::vector<VkDescriptorSetLayout> taaHistogramLayouts(
                        frameCount_, postProcessLuminanceDescriptorSetLayout_.handle());
                    VkDescriptorSetAllocateInfo taaHistogramAllocateInfo{};
                    taaHistogramAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    taaHistogramAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
                    taaHistogramAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaDescriptorSets.size());
                    taaHistogramAllocateInfo.pSetLayouts = taaHistogramLayouts.data();
                    VK_CHECK(vkAllocateDescriptorSets(
                        context_.vkDevice(), &taaHistogramAllocateInfo, taaDescriptorSets.data()));

                    for (size_t frameIndex = 0; frameIndex < taaDescriptorSets.size(); ++frameIndex) {
                        VkDescriptorImageInfo taaSceneColorInfo =
                            postProcessImageInfo(taaHistoryImages_[historyIndex].imageView());
                        VkDescriptorBufferInfo histogramBufferInfo{};
                        histogramBufferInfo.buffer = frameHistogramBuffers_[frameIndex].buffer();
                        histogramBufferInfo.offset = 0;
                        histogramBufferInfo.range = frameHistogramBuffers_[frameIndex].size();

                        std::array<VkWriteDescriptorSet, 2> taaHistogramWrites{};
                        taaHistogramWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaHistogramWrites[0].dstSet = taaDescriptorSets[frameIndex];
                        taaHistogramWrites[0].dstBinding = 0;
                        taaHistogramWrites[0].descriptorCount = 1;
                        taaHistogramWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        taaHistogramWrites[0].pImageInfo = &taaSceneColorInfo;

                        taaHistogramWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaHistogramWrites[1].dstSet = taaDescriptorSets[frameIndex];
                        taaHistogramWrites[1].dstBinding = 1;
                        taaHistogramWrites[1].descriptorCount = 1;
                        taaHistogramWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        taaHistogramWrites[1].pBufferInfo = &histogramBufferInfo;

                        vkUpdateDescriptorSets(context_.vkDevice(),
                                               static_cast<uint32_t>(taaHistogramWrites.size()),
                                               taaHistogramWrites.data(),
                                               0,
                                               nullptr);
                        rhi::debug::setObjectName(context_.vkDevice(),
                                                  taaDescriptorSets[frameIndex],
                                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                                  "TAAHistogramDescriptorSet" + std::to_string(historyIndex) + "_" +
                                                      std::to_string(frameIndex));
                    }
                }
            }
        } catch (const std::exception& error) {
            histogramDescriptorSets_.clear();
            disableHistogramExposureFallback(std::string("Histogram exposure descriptor allocation failed: ") +
                                             error.what());
        }
    }
}

void PostProcessStack::createExposureReduceDescriptorSets(const PostProcessDescriptorCounts& counts)
{
    if (counts.createExposureReduceDescriptors) {
        try {
            exposureReduceDescriptorSets_.assign(frameCount_, VK_NULL_HANDLE);
            std::vector<VkDescriptorSetLayout> exposureLayouts(frameCount_,
                                                               postProcessExposureReduceDescriptorSetLayout_.handle());
            VkDescriptorSetAllocateInfo exposureAllocateInfo{};
            exposureAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            exposureAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            exposureAllocateInfo.descriptorSetCount = static_cast<uint32_t>(exposureReduceDescriptorSets_.size());
            exposureAllocateInfo.pSetLayouts = exposureLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(
                context_.vkDevice(), &exposureAllocateInfo, exposureReduceDescriptorSets_.data()));

            for (size_t frameIndex = 0; frameIndex < exposureReduceDescriptorSets_.size(); ++frameIndex) {
                std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
                bufferInfos[0].buffer = frameLuminanceBuffers_[frameIndex].buffer();
                bufferInfos[0].range = frameLuminanceBuffers_[frameIndex].size();
                bufferInfos[1].buffer = frameHistogramBuffers_[frameIndex].buffer();
                bufferInfos[1].range = frameHistogramBuffers_[frameIndex].size();
                bufferInfos[2].buffer = frameExposureBuffers_[frameIndex].buffer();
                bufferInfos[2].range = sizeof(ExposureState);

                std::array<VkWriteDescriptorSet, 3> exposureWrites{};
                for (uint32_t binding = 0; binding < exposureWrites.size(); ++binding) {
                    exposureWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    exposureWrites[binding].dstSet = exposureReduceDescriptorSets_[frameIndex];
                    exposureWrites[binding].dstBinding = binding;
                    exposureWrites[binding].descriptorCount = 1;
                    exposureWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    exposureWrites[binding].pBufferInfo = &bufferInfos[binding];
                }

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(exposureWrites.size()),
                                       exposureWrites.data(),
                                       0,
                                       nullptr);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          exposureReduceDescriptorSets_[frameIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "ExposureReduceDescriptorSet" + std::to_string(frameIndex));
            }
        } catch (const std::exception& error) {
            exposureReduceDescriptorSets_.clear();
            disableAutoExposureFallback(std::string("GPU exposure reduce descriptor allocation failed: ") +
                                        error.what());
        }
    }
}

void PostProcessStack::createLuminanceResources()
{
    destroyLuminanceResources();

    if (!toneMappingSettings_.enableAutoExposure) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        return;
    }
    if (postProcessLuminanceDescriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("missing luminance descriptor set layout");
    }

    // USED, not allocated: the reduction covers exactly the texels the frame
    // writes, and the partial buffer is sized to match. Both are recomputed on
    // every recreate, which a scale change still triggers -- when it stops doing
    // so, the buffer has to be sized for the allocation and the partial count
    // recomputed per frame instead.
    const VkExtent2D sceneExtent = sceneUsedExtent();
    if (sceneExtent.width == 0 || sceneExtent.height == 0) {
        throw std::runtime_error("scene color extent is zero");
    }

    luminanceGroupCountX_ = (sceneExtent.width + kLuminanceLocalSizeX - 1) / kLuminanceLocalSizeX;
    luminanceGroupCountY_ = (sceneExtent.height + kLuminanceLocalSizeY - 1) / kLuminanceLocalSizeY;
    luminancePartialCount_ = luminanceGroupCountX_ * luminanceGroupCountY_;
    if (luminancePartialCount_ == 0) {
        throw std::runtime_error("luminance reduction produced zero workgroups");
    }

    const VkDeviceSize luminanceBufferSize =
        static_cast<VkDeviceSize>(luminancePartialCount_) * sizeof(LuminancePartial);

    frameLuminanceBuffers_.resize(frameCount_);
    frameLuminanceReadbackBuffers_.clear();
    frameLuminanceReadbackReady_.clear();

    for (size_t frameIndex = 0; frameIndex < frameCount_; ++frameIndex) {
        rhi::VulkanBufferCreateInfo luminanceInfo{};
        luminanceInfo.size = luminanceBufferSize;
        luminanceInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        luminanceInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameLuminanceBuffers_[frameIndex].createBuffer(context_, luminanceInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameLuminanceBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminancePartialBuffer" + std::to_string(frameIndex));
    }

    autoExposureAvailable_ = true;
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
}

void PostProcessStack::destroyLuminanceResources()
{
    autoExposureAvailable_ = false;
    luminanceDescriptorSets_.clear();
    frameLuminanceReadbackReady_.clear();
    frameLuminanceReadbackBuffers_.clear();
    frameLuminanceBuffers_.clear();
    luminancePartialCount_ = 0;
    luminanceGroupCountX_ = 0;
    luminanceGroupCountY_ = 0;
}

void PostProcessStack::createHistogramResources()
{
    destroyHistogramResources();

    if (!toneMappingSettings_.enableAutoExposure) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        return;
    }
    if (postProcessLuminanceDescriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("missing exposure descriptor set layout");
    }

    const VkExtent2D sceneExtent = sceneUsedExtent();
    if (sceneExtent.width == 0 || sceneExtent.height == 0) {
        throw std::runtime_error("scene color extent is zero");
    }

    const VkDeviceSize histogramBufferSize = static_cast<VkDeviceSize>(kHistogramBinCount * sizeof(uint32_t));

    frameHistogramBuffers_.resize(frameCount_);
    frameHistogramReadbackBuffers_.clear();
    frameHistogramReadbackReady_.clear();

    for (size_t frameIndex = 0; frameIndex < frameCount_; ++frameIndex) {
        rhi::VulkanBufferCreateInfo histogramInfo{};
        histogramInfo.size = histogramBufferSize;
        histogramInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        histogramInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameHistogramBuffers_[frameIndex].createBuffer(context_, histogramInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameHistogramBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminanceHistogramBuffer" + std::to_string(frameIndex));
    }

    histogramExposureAvailable_ = true;
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
}

void PostProcessStack::destroyHistogramResources()
{
    histogramExposureAvailable_ = false;
    histogramDescriptorSets_.clear();
    frameHistogramReadbackReady_.clear();
    frameHistogramReadbackBuffers_.clear();
    frameHistogramBuffers_.clear();
}

void PostProcessStack::createExposureResources()
{
    destroyExposureResources();

    if ((frameCount_ == 0u)) {
        return;
    }

    frameExposureBuffers_.resize(frameCount_);
    frameExposureReadbackReady_.assign(frameCount_, 0);

    const ExposureMode mode =
        exposureModeValue(toneMappingSettings_.enableAutoExposure ? toneMappingSettings_.exposureMode
                                                                  : static_cast<int>(ExposureMode::Manual));
    const ExposureState initialState{
        toneMappingExposureValue(toneMappingSettings_.enableAutoExposure ? currentExposure_
                                                                         : toneMappingSettings_.manualExposure),
        averageLuminance_,
        histogramClippedLuminance_,
        static_cast<uint32_t>(mode),
    };

    for (size_t frameIndex = 0; frameIndex < frameCount_; ++frameIndex) {
        rhi::VulkanBufferCreateInfo exposureInfo{};
        exposureInfo.size = sizeof(ExposureState);
        exposureInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        exposureInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        exposureInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        frameExposureBuffers_[frameIndex].createBuffer(context_, exposureInfo);
        frameExposureBuffers_[frameIndex].upload(std::as_bytes(std::span<const ExposureState>(&initialState, 1)));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameExposureBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "ExposureStateBuffer" + std::to_string(frameIndex));
    }

    exposureReduceAvailable_ = true;
}

void PostProcessStack::destroyExposureResources()
{
    exposureReduceAvailable_ = false;
    exposureReduceDescriptorSets_.clear();
    frameExposureReadbackReady_.clear();
    frameExposureBuffers_.clear();
}

void PostProcessStack::disableAutoExposureFallback(std::string_view reason)
{
    if (!autoExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; disabling auto exposure and using manual exposure.");
        autoExposureWarningLogged_ = true;
    }

    toneMappingSettings_.enableAutoExposure = false;
    currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    destroyLuminanceResources();
    destroyHistogramResources();
    luminancePipeline_.reset();
    histogramPipeline_.reset();
    exposureReducePipeline_.reset();
    exposureReduceAvailable_ = false;
}

void PostProcessStack::disableLogAverageExposureFallback(std::string_view reason)
{
    if (!logAverageExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; log-average exposure fallback unavailable.");
        logAverageExposureWarningLogged_ = true;
    }

    destroyLuminanceResources();
    luminancePipeline_.reset();
    exposureReducePipeline_.reset();
    exposureReduceAvailable_ = false;
    if (!isHistogramExposureActive()) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }
}

void PostProcessStack::disableHistogramExposureFallback(std::string_view reason)
{
    if (!histogramExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; falling back to log-average exposure when available.");
        histogramExposureWarningLogged_ = true;
    }

    destroyHistogramResources();
    histogramPipeline_.reset();
    exposureReducePipeline_.reset();
    exposureReduceAvailable_ = false;
    if (!isLogAverageExposureActive()) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }
}

void PostProcessStack::createBloomPipelines()
{
    const VkDescriptorSetLayout postProcessSingleImageDescriptorSetLayout =
        postProcessSingleImageDescriptorSetLayout_.handle();
    const VkDescriptorSetLayout postProcessDualImageDescriptorSetLayout =
        postProcessDualImageDescriptorSetLayout_.handle();
    const VkPushConstantRange bloomExtractPushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomExtractPushConstants))};
    const VkPushConstantRange bloomBlurPushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomBlurPushConstants))};
    const VkPushConstantRange bloomDownsamplePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomDownsamplePushConstants))};
    const VkPushConstantRange bloomUpsamplePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomUpsamplePushConstants))};

    rhi::VulkanPipelineCreateInfo bloomExtractPipelineInfo{};
    bloomExtractPipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomExtractPipelineInfo.fragmentShaderPath = shaderPath("bloom_extract.frag.spv");
    bloomExtractPipelineInfo.colorFormat = kBloomColorFormat;
    bloomExtractPipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessSingleImageDescriptorSetLayout, 1);
    bloomExtractPipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&bloomExtractPushConstantRange, 1);

    bloomExtractPipelineInfo.pipelineCache = context_.pipelineCache();
    bloomExtractPipeline_.create(context_.vkDevice(), bloomExtractPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomExtractPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "BloomExtractPipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomExtractPipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "BloomExtractPipelineLayout");
    bloomExtractPipelineColorFormat_ = bloomExtractPipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo bloomBlurPipelineInfo{};
    bloomBlurPipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomBlurPipelineInfo.fragmentShaderPath = shaderPath("bloom_blur.frag.spv");
    bloomBlurPipelineInfo.colorFormat = kBloomColorFormat;
    bloomBlurPipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessSingleImageDescriptorSetLayout, 1);
    bloomBlurPipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&bloomBlurPushConstantRange, 1);

    bloomBlurPipelineInfo.pipelineCache = context_.pipelineCache();
    bloomBlurPipeline_.create(context_.vkDevice(), bloomBlurPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomBlurPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "BloomBlurPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomBlurPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "BloomBlurPipelineLayout");
    bloomBlurPipelineColorFormat_ = bloomBlurPipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo bloomDownsamplePipelineInfo{};
    bloomDownsamplePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomDownsamplePipelineInfo.fragmentShaderPath = shaderPath("bloom_downsample.frag.spv");
    bloomDownsamplePipelineInfo.colorFormat = kBloomColorFormat;
    bloomDownsamplePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessSingleImageDescriptorSetLayout, 1);
    bloomDownsamplePipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&bloomDownsamplePushConstantRange, 1);

    bloomDownsamplePipelineInfo.pipelineCache = context_.pipelineCache();
    bloomDownsamplePipeline_.create(context_.vkDevice(), bloomDownsamplePipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomDownsamplePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "BloomDownsamplePipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomDownsamplePipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "BloomDownsamplePipelineLayout");
    bloomDownsamplePipelineColorFormat_ = bloomDownsamplePipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo bloomUpsamplePipelineInfo{};
    bloomUpsamplePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomUpsamplePipelineInfo.fragmentShaderPath = shaderPath("bloom_upsample.frag.spv");
    bloomUpsamplePipelineInfo.colorFormat = kBloomColorFormat;
    bloomUpsamplePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessDualImageDescriptorSetLayout, 1);
    bloomUpsamplePipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&bloomUpsamplePushConstantRange, 1);

    bloomUpsamplePipelineInfo.pipelineCache = context_.pipelineCache();
    bloomUpsamplePipeline_.create(context_.vkDevice(), bloomUpsamplePipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomUpsamplePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "BloomUpsamplePipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomUpsamplePipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "BloomUpsamplePipelineLayout");
    bloomUpsamplePipelineColorFormat_ = bloomUpsamplePipelineInfo.colorFormat;
}

void PostProcessStack::createTaaResolvePipeline()
{
    const VkDescriptorSetLayout taaResolveDescriptorSetLayout = postProcessTaaResolveDescriptorSetLayout_.handle();
    const VkPushConstantRange taaResolvePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(TaaResolvePushConstants))};

    rhi::VulkanPipelineCreateInfo taaResolvePipelineInfo{};
    taaResolvePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    taaResolvePipelineInfo.fragmentShaderPath = shaderPath("taa_resolve.frag.spv");
    taaResolvePipelineInfo.colorFormat = kSceneColorFormat;
    taaResolvePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&taaResolveDescriptorSetLayout, 1);
    taaResolvePipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&taaResolvePushConstantRange, 1);

    taaResolvePipelineInfo.pipelineCache = context_.pipelineCache();
    taaResolvePipeline_.create(context_.vkDevice(), taaResolvePipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), taaResolvePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "TAAResolvePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), taaResolvePipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "TAAResolvePipelineLayout");
    taaResolvePipelineColorFormat_ = taaResolvePipelineInfo.colorFormat;
}

void PostProcessStack::createCompositePipeline()
{
    const VkDescriptorSetLayout postProcessCompositeDescriptorSetLayout =
        postProcessCompositeDescriptorSetLayout_.handle();
    const VkPushConstantRange compositePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(CompositePushConstants))};

    rhi::VulkanPipelineCreateInfo compositePipelineInfo{};
    compositePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    compositePipelineInfo.fragmentShaderPath = shaderPath("composite.frag.spv");
    compositePipelineInfo.colorFormat = swapchain_.colorFormat();
    compositePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessCompositeDescriptorSetLayout, 1);
    compositePipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&compositePushConstantRange, 1);

    compositePipelineInfo.pipelineCache = context_.pipelineCache();
    compositePipeline_.create(context_.vkDevice(), compositePipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), compositePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "CompositePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), compositePipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "CompositePipelineLayout");
    compositePipelineColorFormat_ = compositePipelineInfo.colorFormat;
}

void PostProcessStack::updateAutoExposureFromReadback(uint32_t frameIndex)
{
    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (!toneMappingSettings_.enableAutoExposure || mode == ExposureMode::Manual) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        if (frameIndex < frameExposureBuffers_.size() && frameExposureBuffers_[frameIndex].valid()) {
            const ExposureState manualState{
                currentExposure_,
                averageLuminance_,
                histogramClippedLuminance_,
                static_cast<uint32_t>(ExposureMode::Manual),
            };
            frameExposureBuffers_[frameIndex].upload(std::as_bytes(std::span<const ExposureState>(&manualState, 1)));
            if (frameIndex < frameExposureReadbackReady_.size()) {
                frameExposureReadbackReady_[frameIndex] = 1;
            }
        }
        return;
    }

    if (frameIndex >= frameExposureReadbackReady_.size() || frameExposureReadbackReady_[frameIndex] == 0 ||
        frameIndex >= frameExposureBuffers_.size() || !frameExposureBuffers_[frameIndex].valid()) {
        return;
    }

    ExposureState state{};
    frameExposureBuffers_[frameIndex].download(std::as_writable_bytes(std::span<ExposureState>(&state, 1)));
    if (std::isfinite(state.exposure) && state.exposure > 0.0f) {
        currentExposure_ = toneMappingExposureValue(state.exposure);
    }
    if (std::isfinite(state.averageLuminance) && state.averageLuminance > 0.0f) {
        averageLuminance_ = std::max(state.averageLuminance, kMinAverageLuminance);
    }
    if (std::isfinite(state.histogramLuminance) && state.histogramLuminance > 0.0f) {
        histogramClippedLuminance_ = std::max(state.histogramLuminance, kMinAverageLuminance);
    }
}

bool PostProcessStack::isAutoExposureActive() const
{
    if (!toneMappingSettings_.enableAutoExposure) {
        return false;
    }

    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (mode == ExposureMode::Manual) {
        return false;
    }
    if (mode == ExposureMode::LogAverage) {
        return isLogAverageExposureActive();
    }

    return isHistogramExposureActive() || isLogAverageExposureActive();
}

bool PostProcessStack::isLogAverageExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && autoExposureAvailable_ &&
           luminancePipeline_.pipeline() != VK_NULL_HANDLE && luminancePipeline_.layout() != VK_NULL_HANDLE &&
           luminanceDescriptorSets_.size() == frameCount_ && frameLuminanceBuffers_.size() == frameCount_ &&
           luminancePartialCount_ > 0 && luminanceGroupCountX_ > 0 && luminanceGroupCountY_ > 0;
}

bool PostProcessStack::isHistogramExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && histogramExposureAvailable_ &&
           histogramPipeline_.pipeline() != VK_NULL_HANDLE && histogramPipeline_.layout() != VK_NULL_HANDLE &&
           histogramDescriptorSets_.size() == frameCount_ && frameHistogramBuffers_.size() == frameCount_ &&
           isLogAverageExposureActive();
}

bool PostProcessStack::isGpuExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && exposureReduceAvailable_ &&
           exposureReducePipeline_.pipeline() != VK_NULL_HANDLE && exposureReducePipeline_.layout() != VK_NULL_HANDLE &&
           exposureReduceDescriptorSets_.size() == frameCount_ && frameExposureBuffers_.size() == frameCount_;
}

bool PostProcessStack::isTaaActive() const
{
    return taaSettings_.enabled && taaResolvePipeline_.pipeline() != VK_NULL_HANDLE &&
           taaResolvePipeline_.layout() != VK_NULL_HANDLE && taaHistoryImages_[0].imageView() != VK_NULL_HANDLE &&
           taaHistoryImages_[1].imageView() != VK_NULL_HANDLE && taaResolveDescriptorSets_[0] != VK_NULL_HANDLE &&
           taaResolveDescriptorSets_[1] != VK_NULL_HANDLE;
}

bool PostProcessStack::isTaaJitterActive() const
{
    return isTaaActive() && taaSettings_.jitterEnabled;
}

uint32_t PostProcessStack::taaHistoryReadIndex() const
{
    return (taaHistoryWriteIndex_ + 1u) % kTaaHistoryCount;
}

uint32_t PostProcessStack::taaHistoryWriteIndex() const
{
    return taaHistoryWriteIndex_ % kTaaHistoryCount;
}

VkDescriptorSet PostProcessStack::activeBloomExtractDescriptorSet() const
{
    if (isTaaActive()) {
        const VkDescriptorSet descriptorSet =
            taaBloomExtractDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (descriptorSet != VK_NULL_HANDLE) {
            return descriptorSet;
        }
    }
    return bloomExtractDescriptorSet_;
}

VkDescriptorSet PostProcessStack::activeBloomMipDownsampleDescriptorSet(uint32_t level) const
{
    if (level == 0 && isTaaActive()) {
        const VkDescriptorSet descriptorSet =
            taaBloomMipDownsampleDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (descriptorSet != VK_NULL_HANDLE) {
            return descriptorSet;
        }
    }
    return level < bloomMipDownsampleDescriptorSets_.size() ? bloomMipDownsampleDescriptorSets_[level] : VK_NULL_HANDLE;
}

VkDescriptorSet PostProcessStack::activeCompositeDescriptorSet() const
{
    if (isTaaActive()) {
        const auto& descriptorSets = taaCompositeDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (currentFrame_ < descriptorSets.size() && descriptorSets[currentFrame_] != VK_NULL_HANDLE) {
            return descriptorSets[currentFrame_];
        }
    }
    return currentFrame_ < compositeDescriptorSets_.size() ? compositeDescriptorSets_[currentFrame_]
                                                           : compositeDescriptorSet_;
}

VkDescriptorSet PostProcessStack::activeLuminanceDescriptorSet() const
{
    if (isTaaActive()) {
        const auto& descriptorSets = taaLuminanceDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (currentFrame_ < descriptorSets.size() && descriptorSets[currentFrame_] != VK_NULL_HANDLE) {
            return descriptorSets[currentFrame_];
        }
    }
    return currentFrame_ < luminanceDescriptorSets_.size() ? luminanceDescriptorSets_[currentFrame_] : VK_NULL_HANDLE;
}

VkDescriptorSet PostProcessStack::activeHistogramDescriptorSet() const
{
    if (isTaaActive()) {
        const auto& descriptorSets = taaHistogramDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (currentFrame_ < descriptorSets.size() && descriptorSets[currentFrame_] != VK_NULL_HANDLE) {
            return descriptorSets[currentFrame_];
        }
    }
    return currentFrame_ < histogramDescriptorSets_.size() ? histogramDescriptorSets_[currentFrame_] : VK_NULL_HANDLE;
}

float PostProcessStack::currentToneMappingExposure() const
{
    if (!toneMappingSettings_.enableAutoExposure) {
        return toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }

    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (mode == ExposureMode::LogAverage && isLogAverageExposureActive()) {
        return toneMappingExposureValue(currentExposure_);
    }
    if (mode == ExposureMode::Histogram && (isHistogramExposureActive() || isLogAverageExposureActive())) {
        return toneMappingExposureValue(currentExposure_);
    }

    return toneMappingExposureValue(toneMappingSettings_.manualExposure);
}

void PostProcessStack::recordLuminanceCommands(VkCommandBuffer commandBuffer)
{
    if (!isLogAverageExposureActive() || exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual ||
        currentFrame_ >= frameLuminanceBuffers_.size()) {
        return;
    }

    VkBuffer luminanceBuffer = frameLuminanceBuffers_[currentFrame_].buffer();
    if (luminanceBuffer == VK_NULL_HANDLE) {
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, "LuminancePass");
    renderGraph_.beginLuminancePass();

    rhi::debug::beginLabel(commandBuffer, "LuminancePass");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminancePipeline_.pipeline());

    const VkDescriptorSet descriptorSet = activeLuminanceDescriptorSet();
    if (descriptorSet == VK_NULL_HANDLE) {
        renderGraph_.endLuminancePass();
        rhi::debug::endLabel(commandBuffer);
        return;
    }
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminancePipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    const VkExtent2D sceneExtent = sceneUsedExtent();
    const LuminancePushConstants pushConstants{
        glm::uvec4(sceneExtent.width, sceneExtent.height, luminanceGroupCountX_, 0)};
    vkCmdPushConstants(commandBuffer,
                       luminancePipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(LuminancePushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "AutoExposureCompute");
    vkCmdDispatch(commandBuffer, luminanceGroupCountX_, luminanceGroupCountY_, 1);
    rhi::debug::endLabel(commandBuffer);

    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endLuminancePass();
}

void PostProcessStack::recordHistogramCommands(VkCommandBuffer commandBuffer)
{
    if (!isHistogramExposureActive() || exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual ||
        currentFrame_ >= frameHistogramBuffers_.size() || currentFrame_ >= frameExposureReadbackReady_.size()) {
        return;
    }

    VkBuffer histogramBuffer = frameHistogramBuffers_[currentFrame_].buffer();
    if (histogramBuffer == VK_NULL_HANDLE) {
        return;
    }

    const VkExtent2D sceneExtent = sceneUsedExtent();
    const uint32_t groupCountX = (sceneExtent.width + kHistogramLocalSizeX - 1) / kHistogramLocalSizeX;
    const uint32_t groupCountY = (sceneExtent.height + kHistogramLocalSizeY - 1) / kHistogramLocalSizeY;
    if (groupCountX == 0 || groupCountY == 0) {
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, "Histogram Exposure");
    renderGraph_.beginHistogramExposurePass();

    rhi::debug::beginLabel(commandBuffer, "HistogramExposurePass");
    const VkDeviceSize histogramBufferSize = frameHistogramBuffers_[currentFrame_].size();
    vkCmdFillBuffer(commandBuffer, histogramBuffer, 0, histogramBufferSize, 0);

    VkBufferMemoryBarrier2 resetBarrier{};
    resetBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    resetBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    resetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    resetBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    resetBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarrier.buffer = histogramBuffer;
    resetBarrier.offset = 0;
    resetBarrier.size = histogramBufferSize;

    VkDependencyInfo resetDependencyInfo{};
    resetDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    resetDependencyInfo.bufferMemoryBarrierCount = 1;
    resetDependencyInfo.pBufferMemoryBarriers = &resetBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &resetDependencyInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline_.pipeline());

    const VkDescriptorSet descriptorSet = activeHistogramDescriptorSet();
    if (descriptorSet == VK_NULL_HANDLE) {
        renderGraph_.endHistogramExposurePass();
        rhi::debug::endLabel(commandBuffer);
        return;
    }
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    const auto [minLogLuminance, maxLogLuminance] = sanitizedHistogramLogRange(
        toneMappingSettings_.histogramMinLogLuminance, toneMappingSettings_.histogramMaxLogLuminance);
    const HistogramPushConstants pushConstants{glm::uvec4(sceneExtent.width, sceneExtent.height, kHistogramBinCount, 0),
                                               glm::vec4(minLogLuminance, maxLogLuminance, kMinAverageLuminance, 0.0f)};
    vkCmdPushConstants(commandBuffer,
                       histogramPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(HistogramPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "HistogramCompute");
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
    rhi::debug::endLabel(commandBuffer);

    VkBufferMemoryBarrier2 computeBarrier{};
    computeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    computeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarrier.buffer = histogramBuffer;
    computeBarrier.offset = 0;
    computeBarrier.size = histogramBufferSize;

    VkDependencyInfo computeDependencyInfo{};
    computeDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    computeDependencyInfo.bufferMemoryBarrierCount = 1;
    computeDependencyInfo.pBufferMemoryBarriers = &computeBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &computeDependencyInfo);

    recordExposureReduceCommands(commandBuffer);
    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endHistogramExposurePass();
}

void PostProcessStack::recordExposureReduceCommands(VkCommandBuffer commandBuffer)
{
    if (isGpuExposureActive() && currentFrame_ < exposureReduceDescriptorSets_.size() &&
        currentFrame_ < frameExposureBuffers_.size()) {
        VkBuffer exposureBuffer = frameExposureBuffers_[currentFrame_].buffer();
        if (exposureBuffer != VK_NULL_HANDLE) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposureReducePipeline_.pipeline());

            const VkDescriptorSet exposureDescriptorSet = exposureReduceDescriptorSets_[currentFrame_];
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    exposureReducePipeline_.layout(),
                                    0,
                                    1,
                                    &exposureDescriptorSet,
                                    0,
                                    nullptr);

            const auto now = std::chrono::steady_clock::now();
            const float deltaTime = std::max(0.0f, std::chrono::duration<float>(now - lastAutoExposureUpdate_).count());
            lastAutoExposureUpdate_ = now;
            const auto [lowPercentile, highPercentile] =
                sanitizedPercentileRange(toneMappingSettings_.lowPercentile, toneMappingSettings_.highPercentile);
            const auto [reduceMinLogLuminance, reduceMaxLogLuminance] = sanitizedHistogramLogRange(
                toneMappingSettings_.histogramMinLogLuminance, toneMappingSettings_.histogramMaxLogLuminance);
            const float minExposure = std::max(toneMappingSettings_.minExposure, 0.0f);
            const float maxExposure = std::max(toneMappingSettings_.maxExposure, minExposure);
            const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
            const ExposureReducePushConstants exposurePushConstants{
                glm::uvec4(static_cast<uint32_t>(mode), luminancePartialCount_, kHistogramBinCount, 0),
                glm::vec4(toneMappingExposureValue(toneMappingSettings_.manualExposure),
                          std::max(toneMappingSettings_.targetLuminance, kMinAverageLuminance),
                          minExposure,
                          maxExposure),
                glm::vec4(
                    deltaTime, std::max(toneMappingSettings_.adaptationRate, 0.0f), lowPercentile, highPercentile),
                glm::vec4(reduceMinLogLuminance, reduceMaxLogLuminance, kMinAverageLuminance, 0.0f)};
            vkCmdPushConstants(commandBuffer,
                               exposureReducePipeline_.layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(ExposureReducePushConstants)),
                               &exposurePushConstants);

            rhi::debug::beginLabel(commandBuffer, "ExposureReduce");
            vkCmdDispatch(commandBuffer, 1, 1, 1);
            rhi::debug::endLabel(commandBuffer);

            VkBufferMemoryBarrier2 exposureBarrier{};
            exposureBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            exposureBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            exposureBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            exposureBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            exposureBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
            exposureBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            exposureBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            exposureBarrier.buffer = exposureBuffer;
            exposureBarrier.offset = 0;
            exposureBarrier.size = sizeof(ExposureState);

            VkDependencyInfo exposureDependencyInfo{};
            exposureDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            exposureDependencyInfo.bufferMemoryBarrierCount = 1;
            exposureDependencyInfo.pBufferMemoryBarriers = &exposureBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &exposureDependencyInfo);

            frameExposureReadbackReady_[currentFrame_] = 1;
        }
    }
}

void PostProcessStack::recordTaaResolveCommands(VkCommandBuffer commandBuffer)
{
    if (!isTaaActive()) {
        return;
    }

    const uint32_t readIndex = taaHistoryReadIndex();
    const VkDescriptorSet descriptorSet = taaResolveDescriptorSets_[readIndex];
    if (descriptorSet == VK_NULL_HANDLE) {
        return;
    }

    // Allocated size: the resolve's tap offsets are one *texel*, and the
    // sub-rect changes how many are written, not how big one is.
    const VkExtent3D sceneExtent = sceneColor_.extent();
    if (sceneExtent.width == 0 || sceneExtent.height == 0) {
        return;
    }
    const VkExtent2D usedExtent = sceneUsedExtent();
    const glm::vec2 sceneUvScale = renderResolution_.uvScale();

    rhi::debug::beginLabel(commandBuffer, "TAAResolvePass");
    const bool taaProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "TAAResolvePass");
    renderGraph_.beginTaaResolvePass();
    setViewportAndScissor(commandBuffer, usedExtent);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, taaResolvePipeline_.pipeline());
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, taaResolvePipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    const TaaResolvePushConstants pushConstants{
        glm::vec2{1.0f / static_cast<float>(sceneExtent.width), 1.0f / static_cast<float>(sceneExtent.height)},
        taaSettings_.feedback,
        taaHistoryValid_ ? 1u : 0u,
        taaSettings_.neighborhoodClampEnabled ? 1u : 0u,
        taaSettings_.reprojectionEnabled ? 1u : 0u,
        // Depth dilation needs a samplable main depth; otherwise binding 3 holds
        // the checkerboard fallback and the shader must skip the depth reads.
        swapchain_.depthSupportsSampling() ? 1u : 0u,
        0u,
        sceneUvScale,
        // Depth is still sized to what the frame writes, so its taps step by its
        // own texel, not scene colour's.
        glm::vec2{1.0f / static_cast<float>(usedExtent.width), 1.0f / static_cast<float>(usedExtent.height)}};
    vkCmdPushConstants(commandBuffer,
                       taaResolvePipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(TaaResolvePushConstants)),
                       &pushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endTaaResolvePass();
    if (taaProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
}

void PostProcessStack::recordLegacyBloomCommands(VkCommandBuffer commandBuffer)
{
    rhi::debug::beginLabel(commandBuffer, "BloomExtractPass");
    const bool bloomExtractProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "BloomExtractPass");
    renderGraph_.beginBloomExtractPass();
    setViewportAndScissor(commandBuffer, bloomWrittenExtent());
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomExtractPipeline_.pipeline());
    const VkDescriptorSet bloomExtractDescriptorSet = activeBloomExtractDescriptorSet();
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomExtractPipeline_.layout(),
                            0,
                            1,
                            &bloomExtractDescriptorSet,
                            0,
                            nullptr);
    const BloomExtractPushConstants bloomExtractPushConstants{
        bloomSettings_.threshold, 0.0f, renderResolution_.uvScale()};
    vkCmdPushConstants(commandBuffer,
                       bloomExtractPipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(BloomExtractPushConstants)),
                       &bloomExtractPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endBloomExtractPass();
    if (bloomExtractProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    const BloomBlurPushConstants horizontalBlurPushConstants{
        glm::vec2{1.0f / static_cast<float>(bloomExtent_.width), 1.0f / static_cast<float>(bloomExtent_.height)},
        1u,
        0u,
        bloomUvScale()};
    rhi::debug::beginLabel(commandBuffer, "BloomBlurHorizontal");
    const bool bloomBlurHorizontalProfileScope =
        gpuProfiler_.beginScope(currentFrame_, commandBuffer, "BloomBlurHorizontal");
    renderGraph_.beginBloomBlurPass(true);
    setViewportAndScissor(commandBuffer, bloomWrittenExtent());
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomBlurPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomBlurPipeline_.layout(),
                            0,
                            1,
                            &bloomBlurHorizontalDescriptorSet_,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       bloomBlurPipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(BloomBlurPushConstants)),
                       &horizontalBlurPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endBloomBlurPass();
    if (bloomBlurHorizontalProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    const BloomBlurPushConstants verticalBlurPushConstants{
        glm::vec2{1.0f / static_cast<float>(bloomExtent_.width), 1.0f / static_cast<float>(bloomExtent_.height)},
        0u,
        0u,
        bloomUvScale()};
    rhi::debug::beginLabel(commandBuffer, "BloomBlurVertical");
    const bool bloomBlurVerticalProfileScope =
        gpuProfiler_.beginScope(currentFrame_, commandBuffer, "BloomBlurVertical");
    renderGraph_.beginBloomBlurPass(false);
    setViewportAndScissor(commandBuffer, bloomWrittenExtent());
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomBlurPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomBlurPipeline_.layout(),
                            0,
                            1,
                            &bloomBlurVerticalDescriptorSet_,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       bloomBlurPipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(BloomBlurPushConstants)),
                       &verticalBlurPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endBloomBlurPass();
    if (bloomBlurVerticalProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
}

void PostProcessStack::recordMipChainBloomCommands(VkCommandBuffer commandBuffer)
{
    if (bloomMipDownsampleImages_.empty() ||
        bloomMipDownsampleDescriptorSets_.size() != bloomMipDownsampleImages_.size() ||
        bloomDownsamplePipeline_.pipeline() == VK_NULL_HANDLE || bloomDownsamplePipeline_.layout() == VK_NULL_HANDLE) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "BloomMipChain");

    const bool downsampleProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "Bloom Downsample Chain");
    rhi::debug::beginLabel(commandBuffer, "Bloom Downsample Chain");
    for (uint32_t level = 0; level < bloomMipDownsampleImages_.size(); ++level) {
        // Level 0 reads the sub-rected scene colour, so its source texel is one
        // texel of the *allocation* and its UVs need scaling; deeper levels read a
        // mip that was written in full.
        const bool readsSceneColor = level == 0;
        const VkExtent2D sourceExtent = readsSceneColor
                                            ? sceneAllocatedExtent()
                                            : VkExtent2D{bloomMipDownsampleImages_[level - 1].extent().width,
                                                         bloomMipDownsampleImages_[level - 1].extent().height};
        const glm::vec2 sourceUvScale =
            readsSceneColor ? renderResolution_.uvScale()
                            : RenderResolution::subRectUvScale(bloomMipExtent(sceneUsedExtent(), level - 1),
                                                               bloomMipExtent(sceneAllocatedExtent(), level - 1));
        // Viewport is the level's *written* size; the image is the allocated one.
        const VkExtent2D outputSize = bloomMipExtent(sceneUsedExtent(), level);

        rhi::debug::beginLabel(commandBuffer, "BloomDownsampleMip" + std::to_string(level));
        renderGraph_.beginBloomDownsamplePass(level);
        setViewportAndScissor(commandBuffer, outputSize);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomDownsamplePipeline_.pipeline());
        const VkDescriptorSet descriptorSet = activeBloomMipDownsampleDescriptorSet(level);
        if (descriptorSet == VK_NULL_HANDLE) {
            renderGraph_.endBloomDownsamplePass();
            rhi::debug::endLabel(commandBuffer);
            continue;
        }
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                bloomDownsamplePipeline_.layout(),
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);
        const BloomDownsamplePushConstants pushConstants{
            glm::vec2{1.0f / static_cast<float>(sourceExtent.width), 1.0f / static_cast<float>(sourceExtent.height)},
            bloomSettings_.threshold,
            level == 0 ? 1u : 0u,
            sourceUvScale};
        vkCmdPushConstants(commandBuffer,
                           bloomDownsamplePipeline_.layout(),
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(BloomDownsamplePushConstants)),
                           &pushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        renderGraph_.endBloomDownsamplePass();
        rhi::debug::endLabel(commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
    if (downsampleProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    if (!bloomMipUpsampleImages_.empty() && bloomMipUpsampleDescriptorSets_.size() == bloomMipUpsampleImages_.size() &&
        bloomUpsamplePipeline_.pipeline() != VK_NULL_HANDLE && bloomUpsamplePipeline_.layout() != VK_NULL_HANDLE) {
        const bool upsampleProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "Bloom Upsample Chain");
        rhi::debug::beginLabel(commandBuffer, "Bloom Upsample Chain");
        for (uint32_t reverseIndex = 0; reverseIndex < bloomMipUpsampleImages_.size(); ++reverseIndex) {
            const uint32_t level = static_cast<uint32_t>(bloomMipUpsampleImages_.size() - 1u - reverseIndex);
            const VkExtent2D outputSize = bloomMipExtent(sceneUsedExtent(), level);
            const VkExtent3D lowerExtent = level + 1u == bloomMipDownsampleImages_.size() - 1u
                                               ? bloomMipDownsampleImages_[level + 1u].extent()
                                               : bloomMipUpsampleImages_[level + 1u].extent();
            // The two sources are different mips, so different written fractions.
            const glm::vec2 currentUvScale =
                RenderResolution::subRectUvScale(bloomMipExtent(sceneUsedExtent(), level),
                                                 bloomMipExtent(sceneAllocatedExtent(), level));
            const glm::vec2 lowerUvScale =
                RenderResolution::subRectUvScale(bloomMipExtent(sceneUsedExtent(), level + 1u),
                                                 bloomMipExtent(sceneAllocatedExtent(), level + 1u));

            rhi::debug::beginLabel(commandBuffer, "BloomUpsampleMip" + std::to_string(level));
            renderGraph_.beginBloomUpsamplePass(level);
            setViewportAndScissor(commandBuffer, outputSize);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomUpsamplePipeline_.pipeline());
            const VkDescriptorSet descriptorSet = bloomMipUpsampleDescriptorSets_[level];
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    bloomUpsamplePipeline_.layout(),
                                    0,
                                    1,
                                    &descriptorSet,
                                    0,
                                    nullptr);
            const BloomUpsamplePushConstants pushConstants{
                glm::vec2{1.0f / static_cast<float>(lowerExtent.width), 1.0f / static_cast<float>(lowerExtent.height)},
                bloomSettings_.radius,
                0.0f,
                currentUvScale,
                lowerUvScale};
            vkCmdPushConstants(commandBuffer,
                               bloomUpsamplePipeline_.layout(),
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(BloomUpsamplePushConstants)),
                               &pushConstants);
            vkCmdDraw(commandBuffer, 3, 1, 0, 0);
            renderGraph_.endBloomUpsamplePass();
            rhi::debug::endLabel(commandBuffer);
        }
        rhi::debug::endLabel(commandBuffer);
        if (upsampleProfileScope) {
            gpuProfiler_.endScope(currentFrame_, commandBuffer);
        }
    }

    rhi::debug::endLabel(commandBuffer);
}

} // namespace renderer
} // namespace ve
