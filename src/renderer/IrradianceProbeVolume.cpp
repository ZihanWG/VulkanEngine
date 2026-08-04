#include "renderer/IrradianceProbeVolume.h"

#include "core/Logger.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <array>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

constexpr uint32_t kBindingIrradianceAtlas = 0;
constexpr uint32_t kBindingDepthAtlas = 1;
constexpr uint32_t kBindingCaptureAtlas = 2;
constexpr uint32_t kBindingCount = 3;

// Matches the local_size in probe_debug_fill.comp and probe_border.comp.
constexpr uint32_t kProbeLocalSize = 8;

[[nodiscard]] uint32_t dispatchCount(uint32_t total, uint32_t localSize)
{
    return (total + localSize - 1) / localSize;
}

[[nodiscard]] uint32_t coreResolutionFor(ProbeAtlasTarget target)
{
    return target == ProbeAtlasTarget::Irradiance ? kProbeIrradianceResolution : kProbeDepthResolution;
}

VkImageMemoryBarrier2 atlasBarrier(VkImage image,
                                   VkPipelineStageFlags2 srcStage,
                                   VkAccessFlags2 srcAccess,
                                   VkPipelineStageFlags2 dstStage,
                                   VkAccessFlags2 dstAccess)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    // Both sides are storage-image access, so this orders memory without moving
    // the image: the render graph owns the layout and has already put it in
    // GENERAL for the update pass.
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return barrier;
}

} // namespace

IrradianceProbeVolume::~IrradianceProbeVolume()
{
    reset();
}

void IrradianceProbeVolume::create(rhi::VulkanContext& context,
                                   const std::filesystem::path& debugFillShaderPath,
                                   const std::filesystem::path& borderShaderPath,
                                   const std::filesystem::path& convolveShaderPath)
{
    reset();
    context_ = &context;

    // The atlases are created outside the try, like the fog volumes: the render
    // graph imports them and the main pass declares a read on them whether or
    // not the compute half came up, so they cannot be left dangling. Only the
    // pipelines below are optional.
    createAtlases();
    createSampler();

    try {
        createDescriptorResources();
        createPipelines(debugFillShaderPath, borderShaderPath, convolveShaderPath);
        writeDescriptorSet();

        available_ = true;
        Logger::info("Irradiance probes enabled with a " + std::to_string(kProbeGridX) + "x" +
                     std::to_string(kProbeGridY) + "x" + std::to_string(kProbeGridZ) + " probe grid (" +
                     std::to_string(kProbeIrradianceAtlasWidth) + "x" + std::to_string(kProbeIrradianceAtlasHeight) +
                     " irradiance, " + std::to_string(kProbeDepthAtlasWidth) + "x" +
                     std::to_string(kProbeDepthAtlasHeight) + " depth).");
    } catch (const std::exception& error) {
        // Optional subsystem: a failure renders the scene without probe GI
        // rather than failing device creation, matching clustered lighting,
        // punctual shadows and volumetric fog. The atlases deliberately survive
        // so everything that references them still has a valid image.
        debugFillPipeline_.reset();
        borderPipeline_.reset();
        convolvePipeline_.reset();
        descriptorSet_ = VK_NULL_HANDLE;
        descriptorPool_.reset();
        setLayout_.reset();
        available_ = false;
        convolveAvailable_ = false;
        Logger::warn(std::string("Irradiance probes unavailable: ") + error.what());
    }
}

const std::vector<uint32_t>& IrradianceProbeVolume::beginCaptureBatch(uint32_t probesPerFrame)
{
    updateCursor_ = probeUpdateBatch(updateCursor_, probesPerFrame, captureBatch_);
    capturedProbeCount_ += captureBatch_.size();
    return captureBatch_;
}

void IrradianceProbeVolume::reset()
{
    if (context_ != nullptr && sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_->vkDevice(), sampler_, nullptr);
    }
    sampler_ = VK_NULL_HANDLE;

    debugFillPipeline_.reset();
    borderPipeline_.reset();
    convolvePipeline_.reset();
    descriptorSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    setLayout_.reset();
    irradianceAtlas_.reset();
    depthAtlas_.reset();
    captureAtlas_.reset();
    captureDepth_.reset();
    shadingParamsBuffer_.reset();
    irradianceAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    captureAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    captureDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    updateCursor_ = 0;
    captureBatch_.clear();
    capturedProbeCount_ = 0;
    atlasesInitialized_ = false;
    available_ = false;
    convolveAvailable_ = false;
    context_ = nullptr;
}

void IrradianceProbeVolume::createAtlases()
{
    rhi::VulkanImageCreateInfo irradianceInfo{};
    irradianceInfo.width = kProbeIrradianceAtlasWidth;
    irradianceInfo.height = kProbeIrradianceAtlasHeight;
    // Half float rather than a packed format: irradiance is HDR, and
    // R16G16B16A16_SFLOAT is one of the formats Vulkan requires storage-image
    // support for, so there is no format fallback path to get wrong.
    irradianceInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    irradianceInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    irradianceInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    irradianceInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    irradianceInfo.debugName = "ProbeIrradianceAtlas";
    irradianceAtlas_.create(*context_, irradianceInfo);

    rhi::VulkanImageCreateInfo depthInfo{};
    depthInfo.width = kProbeDepthAtlasWidth;
    depthInfo.height = kProbeDepthAtlasHeight;
    // Distance and distance squared, for the Chebyshev visibility bound. Also a
    // mandatory storage format. Distances are clamped to kProbeMaxDistance so
    // the squared channel stays inside the half float's range.
    depthInfo.format = VK_FORMAT_R16G16_SFLOAT;
    depthInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    depthInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    depthInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthInfo.debugName = "ProbeDepthAtlas";
    depthAtlas_.create(*context_, depthInfo);

    const glm::uvec2 captureSize = probeCaptureAtlasSize();

    rhi::VulkanImageCreateInfo captureInfo{};
    captureInfo.width = captureSize.x;
    captureInfo.height = captureSize.y;
    // rgb = outgoing radiance, a = distance from the probe. One attachment
    // rather than two: the convolution reads both for every texel it visits, so
    // splitting them would double the fetch count in the one loop that matters.
    captureInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    captureInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
    captureInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    captureInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    captureInfo.debugName = "ProbeCaptureAtlas";
    captureAtlas_.create(*context_, captureInfo);

    rhi::VulkanImageCreateInfo captureDepthInfo{};
    captureDepthInfo.width = captureSize.x;
    captureDepthInfo.height = captureSize.y;
    // Only ever a depth attachment: distance is written explicitly by the
    // capture fragment shader rather than reconstructed from this, which is why
    // it needs no sampled usage and no particular precision beyond resolving
    // which surface is nearest.
    captureDepthInfo.format = VK_FORMAT_D32_SFLOAT;
    captureDepthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    captureDepthInfo.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    captureDepthInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    captureDepthInfo.debugName = "ProbeCaptureDepth";
    captureDepth_.create(*context_, captureDepthInfo);

    // Device-local and refreshed by vkCmdUpdateBuffer inside the frame that
    // reads it, rather than host-visible and written per frame. That is what
    // makes one buffer enough: a host write would race the previous frame's
    // fragment shader still reading it, and avoiding that would mean a buffer
    // and a descriptor set per frame in flight.
    rhi::VulkanBufferCreateInfo paramsInfo{};
    paramsInfo.size = sizeof(ProbeShadingParams);
    paramsInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    paramsInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    shadingParamsBuffer_.createBuffer(*context_, paramsInfo);
    rhi::debug::setObjectName(
        context_->vkDevice(), shadingParamsBuffer_.buffer(), VK_OBJECT_TYPE_BUFFER, "ProbeShadingParams");
}

void IrradianceProbeVolume::updateShadingParams(VkCommandBuffer commandBuffer, const ProbeShadingParams& params)
{
    if (shadingParamsBuffer_.buffer() == VK_NULL_HANDLE) {
        return;
    }

    vkCmdUpdateBuffer(commandBuffer, shadingParamsBuffer_.buffer(), 0, sizeof(ProbeShadingParams), &params);

    // The buffer is not a render-graph resource, so the barrier is written by
    // hand the way VolumetricFogPass does for its own volumes.
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = shadingParamsBuffer_.buffer();
    barrier.offset = 0;
    barrier.size = sizeof(ProbeShadingParams);

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void IrradianceProbeVolume::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // Linear is the entire reason the tiles carry a border: probe lookups land
    // between texels and have to blend, including across the octahedral seam.
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Clamp rather than repeat, though neither should ever be reached: the UV
    // mapping keeps every tap inside the probe's own tile. Clamp is the mode
    // that degrades least if that ever stops being true.
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_->vkDevice(), &samplerInfo, nullptr, &sampler_));
    rhi::debug::setObjectName(context_->vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "ProbeAtlasSampler");
}

void IrradianceProbeVolume::createDescriptorResources()
{
    const VkDevice device = context_->vkDevice();

    std::array<VkDescriptorSetLayoutBinding, kBindingCount> bindings{};
    bindings[kBindingIrradianceAtlas].binding = kBindingIrradianceAtlas;
    bindings[kBindingIrradianceAtlas].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[kBindingIrradianceAtlas].descriptorCount = 1;
    bindings[kBindingIrradianceAtlas].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[kBindingDepthAtlas].binding = kBindingDepthAtlas;
    bindings[kBindingDepthAtlas].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[kBindingDepthAtlas].descriptorCount = 1;
    bindings[kBindingDepthAtlas].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // Only the convolution reads this one; the fill and border shaders simply do
    // not declare it. One layout for all three keeps a single descriptor set for
    // the whole subsystem, and an unused binding costs nothing.
    bindings[kBindingCaptureAtlas].binding = kBindingCaptureAtlas;
    bindings[kBindingCaptureAtlas].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[kBindingCaptureAtlas].descriptorCount = 1;
    bindings[kBindingCaptureAtlas].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    setLayout_.create(device, std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));

    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = kBindingCount;
    descriptorPool_.create(device, std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), 1);

    const VkDescriptorSetLayout layout = setLayout_.handle();
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool_.handle();
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &layout;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, &descriptorSet_));
}

void IrradianceProbeVolume::createPipelines(const std::filesystem::path& debugFillShaderPath,
                                            const std::filesystem::path& borderShaderPath,
                                            const std::filesystem::path& convolveShaderPath)
{
    const VkDevice device = context_->vkDevice();
    const VkDescriptorSetLayout layout = setLayout_.handle();

    // Same descriptor set layout for all three, different push-constant ranges:
    // the convolution carries this frame's probe indices, which the per-atlas
    // shaders have no use for.
    const auto makePipeline = [&](rhi::VulkanComputePipeline& pipeline,
                                  const std::filesystem::path& shaderPath,
                                  uint32_t pushConstantSize,
                                  const char* debugName) {
        const VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize};
        rhi::VulkanComputePipelineCreateInfo info{};
        info.shaderPath = shaderPath;
        info.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&layout, 1);
        info.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
        info.pipelineCache = context_->pipelineCache();
        pipeline.create(device, info);
        rhi::debug::setObjectName(device, pipeline.pipeline(), VK_OBJECT_TYPE_PIPELINE, debugName);
    };

    makePipeline(
        debugFillPipeline_, debugFillShaderPath, sizeof(ProbeAtlasPushConstants), "ProbeDebugFillPipeline");
    makePipeline(borderPipeline_, borderShaderPath, sizeof(ProbeAtlasPushConstants), "ProbeBorderPipeline");

    // The convolution is what turns a capture into probe tiles, and it is the
    // only piece here that can fail without taking the rest with it: the debug
    // fill still produces a valid, inspectable atlas. So it degrades separately
    // rather than failing the whole subsystem.
    try {
        makePipeline(
            convolvePipeline_, convolveShaderPath, sizeof(ProbeConvolvePushConstants), "ProbeConvolvePipeline");
        convolveAvailable_ = true;
    } catch (const std::exception& error) {
        convolvePipeline_.reset();
        convolveAvailable_ = false;
        Logger::warn(std::string("Irradiance probe convolution unavailable; probes will hold the debug "
                                 "pattern rather than captured radiance: ") +
                     error.what());
    }
}

void IrradianceProbeVolume::writeDescriptorSet()
{
    VkDescriptorImageInfo irradianceInfo{};
    irradianceInfo.imageView = irradianceAtlas_.imageView();
    irradianceInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo depthInfo{};
    depthInfo.imageView = depthAtlas_.imageView();
    depthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo captureInfo{};
    captureInfo.imageView = captureAtlas_.imageView();
    captureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, kBindingCount> writes{};
    for (uint32_t binding = 0; binding < kBindingCount; ++binding) {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = descriptorSet_;
        writes[binding].dstBinding = binding;
        writes[binding].dstArrayElement = 0;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    writes[kBindingIrradianceAtlas].pImageInfo = &irradianceInfo;
    writes[kBindingDepthAtlas].pImageInfo = &depthInfo;
    writes[kBindingCaptureAtlas].pImageInfo = &captureInfo;

    vkUpdateDescriptorSets(
        context_->vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void IrradianceProbeVolume::dispatchAtlas(VkCommandBuffer commandBuffer,
                                          const rhi::VulkanComputePipeline& pipeline,
                                          ProbeAtlasTarget target,
                                          bool debugPattern)
{
    const uint32_t core = coreResolutionFor(target);
    const glm::uvec2 atlasSize = probeAtlasSize(core);

    ProbeAtlasPushConstants pushConstants{};
    pushConstants.coreResolution = static_cast<int32_t>(core);
    pushConstants.target = static_cast<int32_t>(target);
    pushConstants.debugPattern = debugPattern ? 1 : 0;

    vkCmdPushConstants(commandBuffer,
                       pipeline.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(ProbeAtlasPushConstants),
                       &pushConstants);
    vkCmdDispatch(commandBuffer,
                  dispatchCount(atlasSize.x, kProbeLocalSize),
                  dispatchCount(atlasSize.y, kProbeLocalSize),
                  1);
}

void IrradianceProbeVolume::recordDebugFill(VkCommandBuffer commandBuffer, bool debugPattern)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, debugFillPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            debugFillPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet_,
                            0,
                            nullptr);
    dispatchAtlas(commandBuffer, debugFillPipeline_, ProbeAtlasTarget::Irradiance, debugPattern);
    dispatchAtlas(commandBuffer, debugFillPipeline_, ProbeAtlasTarget::Depth, debugPattern);
}

void IrradianceProbeVolume::recordConvolve(VkCommandBuffer commandBuffer)
{
    ProbeConvolvePushConstants pushConstants{};
    pushConstants.probeCount = static_cast<int32_t>(captureBatch_.size());
    for (size_t slot = 0; slot < captureBatch_.size() && slot < kMaxProbesPerFrame; ++slot) {
        pushConstants.probeIndices[slot / 4][slot % 4] = static_cast<int32_t>(captureBatch_[slot]);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, convolvePipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            convolvePipeline_.layout(),
                            0,
                            1,
                            &descriptorSet_,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       convolvePipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       sizeof(ProbeConvolvePushConstants),
                       &pushConstants);
    // One workgroup per probe in the batch. The workgroup is 16x16, sized to the
    // depth tile, and its first 8x8 threads also produce the irradiance tile.
    vkCmdDispatch(commandBuffer, static_cast<uint32_t>(captureBatch_.size()), 1, 1);
}

void IrradianceProbeVolume::recordBorder(VkCommandBuffer commandBuffer)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, borderPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            borderPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet_,
                            0,
                            nullptr);
    // Over the whole atlas rather than just the probes that changed. The border
    // is a texel-for-texel copy over ~110k texels; restricting it to this
    // frame's tiles would save a fraction of a very small pass and add a second
    // dispatch shape to get wrong.
    dispatchAtlas(commandBuffer, borderPipeline_, ProbeAtlasTarget::Irradiance, false);
    dispatchAtlas(commandBuffer, borderPipeline_, ProbeAtlasTarget::Depth, false);
}

void IrradianceProbeVolume::recordUpdate(VkCommandBuffer commandBuffer, bool useDebugPattern)
{
    if (!available_) {
        return;
    }

    // The convolution needs something to convolve. Without a batch (probes per
    // frame set to zero, or the capture pipeline missing) the debug fill is what
    // keeps the atlases holding something defined.
    const bool convolve = !useDebugPattern && convolveAvailable_ && !captureBatch_.empty();

    rhi::debug::beginLabel(commandBuffer, convolve ? "IrradianceProbeConvolve" : "IrradianceProbeFill");

    // An amortised update only ever writes the tiles of the probes it captured,
    // and the atlases are never cleared, so on the first update every probe the
    // batch did not include would still hold whatever its allocation contained.
    // Seeding all of them once is what stops an uncaptured probe from
    // contributing garbage -- and the seed is the neutral "gathered nothing"
    // state rather than the direction pattern, because that is the honest value
    // for a probe that has not been captured yet.
    const bool seedEveryTile = !atlasesInitialized_;
    if (seedEveryTile || !convolve) {
        // A fill standing in for a failed convolution uses the pattern instead:
        // a blank atlas would be indistinguishable from a working one.
        recordDebugFill(commandBuffer, convolve ? false : (useDebugPattern || !convolveAvailable_));
    }

    if (convolve) {
        if (seedEveryTile) {
            // Both dispatches write the same tiles; without this the seed could
            // land after the convolution and wipe it.
            const std::array<VkImageMemoryBarrier2, 2> seedToConvolve{
                atlasBarrier(irradianceAtlas_.image(),
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
                atlasBarrier(depthAtlas_.image(),
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)};
            VkDependencyInfo seedDependency{};
            seedDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            seedDependency.imageMemoryBarrierCount = static_cast<uint32_t>(seedToConvolve.size());
            seedDependency.pImageMemoryBarriers = seedToConvolve.data();
            vkCmdPipelineBarrier2(commandBuffer, &seedDependency);
        }
        recordConvolve(commandBuffer);
    }

    // The border pass reads core texels the step above just wrote. Both touch
    // the same images inside one graph pass, so this is a read-after-write the
    // graph cannot infer -- it sees a single pass.
    const std::array<VkImageMemoryBarrier2, 2> writeToBorder{
        atlasBarrier(irradianceAtlas_.image(),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        atlasBarrier(depthAtlas_.image(),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)};

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = static_cast<uint32_t>(writeToBorder.size());
    dependency.pImageMemoryBarriers = writeToBorder.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    recordBorder(commandBuffer);

    atlasesInitialized_ = true;

    rhi::debug::endLabel(commandBuffer);
}

} // namespace ve::renderer
