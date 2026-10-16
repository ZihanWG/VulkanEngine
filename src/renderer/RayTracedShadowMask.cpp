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

RayTracedShadowMask::RayTracedShadowMask(rhi::VulkanContext& context,
                                         rhi::VulkanSwapchain& swapchain,
                                         rhi::VulkanPipelineStore& pipelineStore,
                                         const RenderResolution& renderResolution)
    : context_(context), swapchain_(swapchain), pipelineStore_(pipelineStore), renderResolution_(renderResolution)
{}

RayTracedShadowMask::~RayTracedShadowMask()
{
    destroyResources();
    destroyPipeline();
}

void RayTracedShadowMask::createPipeline(const std::filesystem::path& shaderPath,
                                         VkDescriptorSetLayout bindlessSetLayout)
{
    if (bindlessSetLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("the shadow mask's cutout test needs the bindless texture heap");
    }

    // Re-requested on every pipeline rebuild, because the store's reset drops
    // every reference it issued; the set layout is created once, because the
    // descriptor set allocated against it outlives a rebuild that is not also a
    // resize.
    if (descriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        createDescriptorSetLayout();
    }
    // Set 1 is the heap at the same set number the main pass binds it at, which
    // is what lets the shared trace include declare it once.
    const std::array<VkDescriptorSetLayout, 2> setLayouts{descriptorSetLayout_.handle(), bindlessSetLayout};
    const VkPushConstantRange pushRange{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(RayTracedShadowMaskPushConstants))};
    rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.shaderPath = shaderPath;
    pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(setLayouts.data(), setLayouts.size());
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushRange, 1);
    pipelineInfo.pipelineCache = context_.pipelineCache();
    pipeline_ = pipelineStore_.get(context_.vkDevice(), pipelineInfo, "RayTracedShadowMaskPipeline");
}

void RayTracedShadowMask::createDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    descriptorSetLayout_.create(context_.vkDevice(),
                                std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              descriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "RayTracedShadowMaskSetLayout");
}

void RayTracedShadowMask::destroyPipeline()
{
    available_ = false;
    pipeline_.reset();
    descriptorSetLayout_.reset();
}

void RayTracedShadowMask::createResources(VkAccelerationStructureKHR topLevel)
{
    destroyResources();

    try {
        const VkExtent2D extent = renderResolution_.allocationExtent();
        rhi::VulkanImageCreateInfo imageInfo{};
        imageInfo.width = extent.width;
        imageInfo.height = extent.height;
        imageInfo.format = kFormat;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageInfo.debugName = "RayTracedShadowMask";
        image_.create(context_, imageInfo);
        layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

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

        // The pass needs all three; the image alone is enough for binding 17,
        // which is why a failure below still leaves it in place.
        if (pipeline_.pipeline() == VK_NULL_HANDLE || descriptorSetLayout_.handle() == VK_NULL_HANDLE) {
            return;
        }
        if (!swapchain_.depthSupportsSampling()) {
            Logger::warn("Ray-traced shadow mask unavailable: the main depth image cannot be sampled; "
                         "every fragment traces inline.");
            return;
        }
        if (topLevel == VK_NULL_HANDLE) {
            return;
        }

        const std::array<VkDescriptorPoolSize, 3> poolSizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        }};
        descriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), 1);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  descriptorPool_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                  "RayTracedShadowMaskDescriptorPool");

        const VkDescriptorSetLayout setLayout = descriptorSetLayout_.handle();
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &setLayout;
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, &descriptorSet_));

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = sampler_;
        depthInfo.imageView = swapchain_.depthImageView();
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSetAccelerationStructureKHR topLevelInfo{};
        topLevelInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        topLevelInfo.accelerationStructureCount = 1;
        topLevelInfo.pAccelerationStructures = &topLevel;

        VkDescriptorImageInfo maskInfo{};
        maskInfo.imageView = image_.imageView();
        maskInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (uint32_t index = 0; index < writes.size(); ++index) {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptorSet_;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &depthInfo;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1].pNext = &topLevelInfo;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &maskInfo;
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
    descriptorSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    image_.reset();
    layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void RayTracedShadowMask::record(VkCommandBuffer commandBuffer,
                                 const RayTracedShadowMaskPushConstants& push,
                                 VkDescriptorSet bindlessSet) const
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());
    const std::array<VkDescriptorSet, 2> sets{descriptorSet_, bindlessSet};
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_.layout(),
                            0,
                            static_cast<uint32_t>(sets.size()),
                            sets.data(),
                            0,
                            nullptr);
    vkCmdPushConstants(
        commandBuffer, pipeline_.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(push)), &push);
    vkCmdDispatch(
        commandBuffer, rayTracedShadowMaskGroupCount(push.extent.x), rayTracedShadowMaskGroupCount(push.extent.y), 1);
}

} // namespace ve::renderer
