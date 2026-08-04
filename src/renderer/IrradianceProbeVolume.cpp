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
constexpr uint32_t kBindingCount = 2;

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
                                   const std::filesystem::path& borderShaderPath)
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
        createPipelines(debugFillShaderPath, borderShaderPath);
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
        descriptorSet_ = VK_NULL_HANDLE;
        descriptorPool_.reset();
        setLayout_.reset();
        available_ = false;
        Logger::warn(std::string("Irradiance probes unavailable: ") + error.what());
    }
}

void IrradianceProbeVolume::reset()
{
    if (context_ != nullptr && sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_->vkDevice(), sampler_, nullptr);
    }
    sampler_ = VK_NULL_HANDLE;

    debugFillPipeline_.reset();
    borderPipeline_.reset();
    descriptorSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    setLayout_.reset();
    irradianceAtlas_.reset();
    depthAtlas_.reset();
    irradianceAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtlasLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    atlasesInitialized_ = false;
    available_ = false;
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
                                            const std::filesystem::path& borderShaderPath)
{
    const VkDevice device = context_->vkDevice();
    const VkDescriptorSetLayout layout = setLayout_.handle();
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(ProbeAtlasPushConstants))};

    const auto makePipeline = [&](rhi::VulkanComputePipeline& pipeline,
                                  const std::filesystem::path& shaderPath,
                                  const char* debugName) {
        rhi::VulkanComputePipelineCreateInfo info{};
        info.shaderPath = shaderPath;
        info.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&layout, 1);
        info.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
        info.pipelineCache = context_->pipelineCache();
        pipeline.create(device, info);
        rhi::debug::setObjectName(device, pipeline.pipeline(), VK_OBJECT_TYPE_PIPELINE, debugName);
    };

    makePipeline(debugFillPipeline_, debugFillShaderPath, "ProbeDebugFillPipeline");
    makePipeline(borderPipeline_, borderShaderPath, "ProbeBorderPipeline");
}

void IrradianceProbeVolume::writeDescriptorSet()
{
    VkDescriptorImageInfo irradianceInfo{};
    irradianceInfo.imageView = irradianceAtlas_.imageView();
    irradianceInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo depthInfo{};
    depthInfo.imageView = depthAtlas_.imageView();
    depthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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

void IrradianceProbeVolume::recordUpdate(VkCommandBuffer commandBuffer, bool debugPattern)
{
    if (!available_) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "IrradianceProbeUpdate");

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

    // The border pass reads core texels the fill just wrote. Both dispatches
    // touch the same images, so this is a genuine read-after-write and not
    // something the render graph can infer -- the graph sees one pass.
    const std::array<VkImageMemoryBarrier2, 2> fillToBorder{
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
    dependency.imageMemoryBarrierCount = static_cast<uint32_t>(fillToBorder.size());
    dependency.pImageMemoryBarriers = fillToBorder.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, borderPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            borderPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet_,
                            0,
                            nullptr);
    dispatchAtlas(commandBuffer, borderPipeline_, ProbeAtlasTarget::Irradiance, debugPattern);
    dispatchAtlas(commandBuffer, borderPipeline_, ProbeAtlasTarget::Depth, debugPattern);

    atlasesInitialized_ = true;

    rhi::debug::endLabel(commandBuffer);
}

} // namespace ve::renderer
