#include "renderer/DepthPyramid.h"

// RendererInternal.h provides the shared anonymous-namespace helpers and structs
// the relocated bodies rely on: shaderPath/mipExtent/imageBarrier/
// recordImageBarrier, depthPyramidFormatSupports/calculateDepthPyramidMipLevels,
// the kDepthPyramid* constants, and DepthPyramidPushConstants. This mirrors how
// PostProcessStack reuses the same internals.
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/GpuProfiler.h"
#include "renderer/RenderGraph.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <algorithm>
#include <array>
#include <span>
#include <string>

namespace ve::renderer {

DepthPyramid::DepthPyramid(rhi::VulkanContext& context,
                           rhi::VulkanSwapchain& swapchain,
                           RenderGraph& renderGraph,
                           GpuProfiler& gpuProfiler)
    : context_(context), swapchain_(swapchain), renderGraph_(renderGraph), gpuProfiler_(gpuProfiler)
{
}

DepthPyramid::~DepthPyramid()
{
    destroyResources();
}

void DepthPyramid::createDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              descriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "DepthPyramidDescriptorSetLayout");
}

void DepthPyramid::createPipeline()
{
    pipeline_.reset();
    if (descriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        return;
    }

    const VkDescriptorSetLayout depthPyramidDescriptorSetLayout = descriptorSetLayout_.handle();
    const VkPushConstantRange depthPyramidPushConstantRange{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(DepthPyramidPushConstants))};

    rhi::VulkanComputePipelineCreateInfo depthPyramidPipelineInfo{};
    depthPyramidPipelineInfo.shaderPath = shaderPath("depth_pyramid.comp.spv");
    depthPyramidPipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&depthPyramidDescriptorSetLayout, 1);
    depthPyramidPipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&depthPyramidPushConstantRange, 1);
    pipeline_.create(context_.vkDevice(), depthPyramidPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "DepthPyramidComputePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "DepthPyramidPipelineLayout");
}

void DepthPyramid::destroyResources()
{
    valid_ = false;
    buildAvailable_ = false;
    descriptorSets_.clear();
    descriptorPool_.reset();

    for (VkImageView imageView : mipImageViews_) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(context_.vkDevice(), imageView, nullptr);
        }
    }
    mipImageViews_.clear();

    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    image_.reset();
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    mipLevels_ = 0;
    selectedDebugMip_ = 0;
    viewProjection_ = glm::mat4{1.0f};
    cameraPosition_ = glm::vec3{0.0f};
}

void DepthPyramid::invalidate()
{
    valid_ = false;
    viewProjection_ = glm::mat4{1.0f};
    cameraPosition_ = glm::vec3{0.0f};
}

void DepthPyramid::createResources()
{
    destroyResources();

    if (descriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot create depth pyramid resources without a descriptor set layout.");
    }

    const bool sampledFormatSupported =
        depthPyramidFormatSupports(context_.physicalDevice(), VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
    const bool storageFormatSupported =
        depthPyramidFormatSupports(context_.physicalDevice(), VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
    if (!sampledFormatSupported) {
        Logger::warn("Depth pyramid disabled: VK_FORMAT_R32_SFLOAT is not sampleable on this device.");
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    buildAvailable_ = swapchain_.depthSupportsSampling() && storageFormatSupported;
    mipLevels_ = buildAvailable_ ? calculateDepthPyramidMipLevels(extent) : 1U;

    rhi::VulkanImageCreateInfo imageInfo{};
    imageInfo.width = extent.width;
    imageInfo.height = extent.height;
    imageInfo.format = kDepthPyramidFormat;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | (buildAvailable_ ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    imageInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageInfo.mipLevels = mipLevels_;
    imageInfo.debugName = "DepthPyramidHiZ";
    image_.create(context_, imageInfo);
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    mipImageViews_.resize(mipLevels_, VK_NULL_HANDLE);
    for (uint32_t mipLevel = 0; mipLevel < mipLevels_; ++mipLevel) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image_.image();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kDepthPyramidFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = mipLevel;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(context_.vkDevice(), &viewInfo, nullptr, &mipImageViews_[mipLevel]));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  mipImageViews_[mipLevel],
                                  VK_OBJECT_TYPE_IMAGE_VIEW,
                                  "DepthPyramidHiZMip" + std::to_string(mipLevel) + "View");
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(std::max(mipLevels_, 1u) - 1u);
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &sampler_));
    rhi::debug::setObjectName(
        context_.vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "DepthPyramidNearestClampSampler");

    if (!buildAvailable_) {
        if (!swapchain_.depthSupportsSampling()) {
            Logger::warn("Depth pyramid generation disabled: selected main depth format is not sampleable.");
        }
        if (!storageFormatSupported) {
            Logger::warn("Depth pyramid generation disabled: VK_FORMAT_R32_SFLOAT storage images are unsupported.");
        }
        return;
    }

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = mipLevels_;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = mipLevels_;
    descriptorPool_.create(
        context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), mipLevels_);
    rhi::debug::setObjectName(context_.vkDevice(),
                              descriptorPool_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                              "DepthPyramidDescriptorPool");

    descriptorSets_.resize(mipLevels_, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(mipLevels_, descriptorSetLayout_.handle());
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(descriptorSets_.size());
    allocateInfo.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, descriptorSets_.data()));

    for (uint32_t mipLevel = 0; mipLevel < mipLevels_; ++mipLevel) {
        VkDescriptorImageInfo sourceInfo{};
        sourceInfo.sampler = sampler_;
        sourceInfo.imageView = mipLevel == 0 ? swapchain_.depthImageView() : mipImageViews_[mipLevel - 1];
        sourceInfo.imageLayout = mipLevel == 0 ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = mipImageViews_[mipLevel];
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[mipLevel];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &sourceInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[mipLevel];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;

        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  descriptorSets_[mipLevel],
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                  "DepthPyramidDescriptorSetMip" + std::to_string(mipLevel));
    }
}

void DepthPyramid::ensureShaderReadLayout(VkCommandBuffer commandBuffer)
{
    if (image_.image() == VK_NULL_HANDLE || mipLevels_ == 0 ||
        layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        return;
    }

    const bool undefined = layout_ == VK_IMAGE_LAYOUT_UNDEFINED;
    const VkImageMemoryBarrier2 shaderReadBarrier =
        imageBarrier(image_.image(),
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     layout_,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     undefined ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     undefined ? VK_ACCESS_2_NONE
                               : (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     0,
                     mipLevels_);
    recordImageBarrier(commandBuffer, shaderReadBarrier);
    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void DepthPyramid::recordCommands(VkCommandBuffer commandBuffer,
                                  uint32_t frameIndex,
                                  const glm::mat4& frameViewProjection,
                                  const glm::vec3& frameCameraPosition)
{
    if (!buildAvailable_ || image_.image() == VK_NULL_HANDLE || pipeline_.pipeline() == VK_NULL_HANDLE ||
        pipeline_.layout() == VK_NULL_HANDLE || descriptorSets_.empty() || mipImageViews_.empty() ||
        mipLevels_ == 0) {
        valid_ = false;
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, frameIndex, commandBuffer, "DepthPyramid");
    renderGraph_.beginDepthPyramidPass();
    rhi::debug::beginLabel(commandBuffer, "DepthPyramid");

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());

    const VkExtent2D baseExtent = swapchain_.extent();
    for (uint32_t mipLevel = 0; mipLevel < mipLevels_; ++mipLevel) {
        const VkExtent2D sourceExtent = mipLevel == 0 ? baseExtent : mipExtent(baseExtent, mipLevel - 1);
        const VkExtent2D destinationExtent = mipExtent(baseExtent, mipLevel);
        const VkDescriptorSet descriptorSet = descriptorSets_[mipLevel];
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_.layout(),
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);

        const DepthPyramidPushConstants pushConstants{
            glm::uvec4(sourceExtent.width, sourceExtent.height, destinationExtent.width, destinationExtent.height)};
        vkCmdPushConstants(commandBuffer,
                           pipeline_.layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(DepthPyramidPushConstants)),
                           &pushConstants);

        const uint32_t groupCountX = (destinationExtent.width + kDepthPyramidLocalSizeX - 1) / kDepthPyramidLocalSizeX;
        const uint32_t groupCountY =
            (destinationExtent.height + kDepthPyramidLocalSizeY - 1) / kDepthPyramidLocalSizeY;
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

        const VkImageMemoryBarrier2 mipWriteToRead = imageBarrier(image_.image(),
                                                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                                                  mipLevel,
                                                                  1);
        recordImageBarrier(commandBuffer, mipWriteToRead);
    }

    const VkImageMemoryBarrier2 pyramidToShaderRead =
        imageBarrier(image_.image(),
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     0,
                     mipLevels_);
    recordImageBarrier(commandBuffer, pyramidToShaderRead);
    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    valid_ = true;
    viewProjection_ = frameViewProjection;
    cameraPosition_ = frameCameraPosition;

    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endDepthPyramidPass();
}

} // namespace ve::renderer
