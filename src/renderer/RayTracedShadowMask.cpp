#include "renderer/RayTracedShadowMask.h"

#include "core/Logger.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <array>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

VkDescriptorSetLayoutBinding computeBinding(uint32_t binding, VkDescriptorType type)
{
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = type;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return layoutBinding;
}

// The main pass's test of whether a mask texel is about its fragment compares
// depth bits exactly, which only a 32-bit float depth buffer stores unrounded.
bool depthHoldsFragmentDepthExactly(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

} // namespace

RayTracedShadowMask::RayTracedShadowMask(rhi::VulkanContext& context,
                                         rhi::VulkanSwapchain& swapchain,
                                         rhi::VulkanPipelineStore& pipelineStore,
                                         const RenderResolution& renderResolution)
    : context_(context), swapchain_(swapchain), pipelineStore_(pipelineStore), renderResolution_(renderResolution)
{}

RayTracedShadowMask::~RayTracedShadowMask()
{
    destroyResources();
    destroyPipelines();
}

void RayTracedShadowMask::createPipelines(const std::filesystem::path& traceShaderPath,
                                          const std::filesystem::path& resolveShaderPath,
                                          VkDescriptorSetLayout bindlessSetLayout)
{
    if (bindlessSetLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("the shadow mask's cutout test needs the bindless texture heap");
    }

    // Re-requested on every pipeline rebuild, because the store's reset drops
    // every reference it issued; the set layouts are created once, because the
    // descriptor sets allocated against them outlive a rebuild that is not also
    // a resize.
    if (traceSetLayout_.handle() == VK_NULL_HANDLE) {
        createDescriptorSetLayouts();
    }

    // The trace's set 1 is the heap at the same set number the main pass binds
    // it at, which is what lets the shared trace include declare it once.
    const std::array<VkDescriptorSetLayout, 2> traceLayouts{traceSetLayout_.handle(), bindlessSetLayout};
    const VkPushConstantRange tracePush{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(RayTracedShadowTracePushConstants))};
    rhi::VulkanComputePipelineCreateInfo traceInfo{};
    traceInfo.shaderPath = traceShaderPath;
    traceInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(traceLayouts.data(), traceLayouts.size());
    traceInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&tracePush, 1);
    traceInfo.pipelineCache = context_.pipelineCache();
    tracePipeline_ = pipelineStore_.get(context_.vkDevice(), traceInfo, "RayTracedShadowTracePipeline");

    const VkDescriptorSetLayout resolveLayout = resolveSetLayout_.handle();
    const VkPushConstantRange resolvePush{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(RayTracedShadowResolvePushConstants))};
    rhi::VulkanComputePipelineCreateInfo resolveInfo{};
    resolveInfo.shaderPath = resolveShaderPath;
    resolveInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&resolveLayout, 1);
    resolveInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&resolvePush, 1);
    resolveInfo.pipelineCache = context_.pipelineCache();
    resolvePipeline_ = pipelineStore_.get(context_.vkDevice(), resolveInfo, "RayTracedShadowResolvePipeline");
}

void RayTracedShadowMask::createDescriptorSetLayouts()
{
    const std::array<VkDescriptorSetLayoutBinding, 3> traceBindings{
        computeBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        computeBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR),
        computeBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
    };
    traceSetLayout_.create(context_.vkDevice(),
                           std::span<const VkDescriptorSetLayoutBinding>(traceBindings.data(), traceBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              traceSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "RayTracedShadowTraceSetLayout");

    const std::array<VkDescriptorSetLayoutBinding, 3> resolveBindings{
        computeBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        computeBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        computeBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
    };
    resolveSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(resolveBindings.data(), resolveBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              resolveSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "RayTracedShadowResolveSetLayout");
}

void RayTracedShadowMask::destroyPipelines()
{
    available_ = false;
    tracePipeline_.reset();
    resolvePipeline_.reset();
    traceSetLayout_.reset();
    resolveSetLayout_.reset();
}

void RayTracedShadowMask::createResources(VkAccelerationStructureKHR topLevel)
{
    destroyResources();

    try {
        const VkExtent2D extent = renderResolution_.allocationExtent();
        rhi::VulkanImageCreateInfo maskInfo{};
        maskInfo.width = extent.width;
        maskInfo.height = extent.height;
        maskInfo.format = kFormat;
        maskInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        maskInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        maskInfo.debugName = "RayTracedShadowMask";
        mask_.create(context_, maskInfo);
        maskLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &sampler_));
        rhi::debug::setObjectName(
            context_.vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "RayTracedShadowMaskNearestSampler");

        // The passes need everything below; the mask image alone is enough for
        // binding 17, which is why a failure from here on leaves it in place.
        if (tracePipeline_.pipeline() == VK_NULL_HANDLE || resolvePipeline_.pipeline() == VK_NULL_HANDLE ||
            traceSetLayout_.handle() == VK_NULL_HANDLE || topLevel == VK_NULL_HANDLE) {
            return;
        }
        if (!swapchain_.depthSupportsSampling()) {
            Logger::warn("Ray-traced shadow mask unavailable: the main depth image cannot be sampled; "
                         "every fragment traces inline.");
            return;
        }
        if (!depthHoldsFragmentDepthExactly(swapchain_.depthFormat())) {
            Logger::warn("Ray-traced shadow mask unavailable: the main depth format is not 32-bit float, so a "
                         "fragment cannot tell whether a texel is about its surface; every fragment traces inline.");
            return;
        }

        rhi::VulkanImageCreateInfo traceInfo = maskInfo;
        traceInfo.format = kTraceFormat;
        traceInfo.debugName = "RayTracedShadowTrace";
        trace_.create(context_, traceInfo);
        traceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        const std::array<VkDescriptorPoolSize, 3> poolSizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        }};
        descriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), 2);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  descriptorPool_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                  "RayTracedShadowMaskDescriptorPool");

        const std::array<VkDescriptorSetLayout, 2> setLayouts{traceSetLayout_.handle(), resolveSetLayout_.handle()};
        std::array<VkDescriptorSet, 2> sets{};
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
        allocateInfo.pSetLayouts = setLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, sets.data()));
        traceSet_ = sets[0];
        resolveSet_ = sets[1];

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = sampler_;
        depthInfo.imageView = swapchain_.depthImageView();
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSetAccelerationStructureKHR topLevelInfo{};
        topLevelInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        topLevelInfo.accelerationStructureCount = 1;
        topLevelInfo.pAccelerationStructures = &topLevel;

        VkDescriptorImageInfo traceStorageInfo{};
        traceStorageInfo.imageView = trace_.imageView();
        traceStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo traceSampledInfo{};
        traceSampledInfo.sampler = sampler_;
        traceSampledInfo.imageView = trace_.imageView();
        traceSampledInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo maskStorageInfo{};
        maskStorageInfo.imageView = mask_.imageView();
        maskStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 6> writes{};
        for (uint32_t index = 0; index < writes.size(); ++index) {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = index < 3 ? traceSet_ : resolveSet_;
            writes[index].dstBinding = index % 3;
            writes[index].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &depthInfo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1].pNext = &topLevelInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &traceStorageInfo;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &depthInfo;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].pImageInfo = &traceSampledInfo;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[5].pImageInfo = &maskStorageInfo;
        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        available_ = true;
    } catch (const std::exception& error) {
        Logger::warn(std::string("Ray-traced shadow mask unavailable: ") + error.what());
        destroyResources();
    }
}

void RayTracedShadowMask::destroyResources()
{
    available_ = false;
    traceSet_ = VK_NULL_HANDLE;
    resolveSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    trace_.reset();
    traceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    mask_.reset();
    maskLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void RayTracedShadowMask::recordTrace(VkCommandBuffer commandBuffer,
                                      const RayTracedShadowTracePushConstants& push,
                                      VkDescriptorSet bindlessSet) const
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, tracePipeline_.pipeline());
    const std::array<VkDescriptorSet, 2> sets{traceSet_, bindlessSet};
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            tracePipeline_.layout(),
                            0,
                            static_cast<uint32_t>(sets.size()),
                            sets.data(),
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       tracePipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(push)),
                       &push);
    vkCmdDispatch(commandBuffer,
                  rayTracedShadowMaskGroupCount(push.traceExtent.x),
                  rayTracedShadowMaskGroupCount(push.traceExtent.y),
                  1);
}

void RayTracedShadowMask::recordResolve(VkCommandBuffer commandBuffer,
                                        const RayTracedShadowResolvePushConstants& push) const
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_.pipeline());
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_.layout(), 0, 1, &resolveSet_, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
                       resolvePipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(push)),
                       &push);
    vkCmdDispatch(commandBuffer,
                  rayTracedShadowMaskGroupCount(push.fullExtent.x),
                  rayTracedShadowMaskGroupCount(push.fullExtent.y),
                  1);
}

} // namespace ve::renderer
