#include "renderer/RayTracedShadowMask.h"

#include "core/Logger.h"
#include "core/PfmWriter.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <array>
#include <bit>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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

VkWriteDescriptorSet
imageWrite(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* imageInfo)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = imageInfo;
    return write;
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
                                          const std::filesystem::path& radiusShaderPath,
                                          const std::filesystem::path& blurShaderPath,
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

    // The filter's two shaders share a set layout and a push block; the blur
    // runs both axes from one pipeline, with the axis in the push block.
    const VkDescriptorSetLayout filterLayout = filterSetLayout_.handle();
    const VkPushConstantRange filterPush{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(RayTracedShadowFilterPushConstants))};
    rhi::VulkanComputePipelineCreateInfo filterInfo{};
    filterInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&filterLayout, 1);
    filterInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&filterPush, 1);
    filterInfo.pipelineCache = context_.pipelineCache();
    filterInfo.shaderPath = radiusShaderPath;
    radiusPipeline_ = pipelineStore_.get(context_.vkDevice(), filterInfo, "RayTracedShadowRadiusPipeline");
    filterInfo.shaderPath = blurShaderPath;
    blurPipeline_ = pipelineStore_.get(context_.vkDevice(), filterInfo, "RayTracedShadowBlurPipeline");
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

    const std::array<VkDescriptorSetLayoutBinding, 2> filterBindings{
        computeBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
        computeBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
    };
    filterSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(filterBindings.data(), filterBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              filterSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "RayTracedShadowFilterSetLayout");
}

void RayTracedShadowMask::destroyPipelines()
{
    available_ = false;
    tracePipeline_.reset();
    resolvePipeline_.reset();
    radiusPipeline_.reset();
    blurPipeline_.reset();
    traceSetLayout_.reset();
    resolveSetLayout_.reset();
    filterSetLayout_.reset();
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
        // Transfer source for dumpVisibility, the debug readback.
        maskInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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
            radiusPipeline_.pipeline() == VK_NULL_HANDLE || blurPipeline_.pipeline() == VK_NULL_HANDLE ||
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
        traceInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        traceInfo.format = kTraceFormat;
        traceInfo.debugName = "RayTracedShadowTrace";
        trace_.create(context_, traceInfo);
        traceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        // Allocated with the trace image whether or not the filter runs, so
        // turning the filter on is a branch, not a reallocation.
        traceInfo.debugName = "RayTracedShadowFilter";
        filter_.create(context_, traceInfo);
        filterLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        // Five sets: trace; resolve from the trace image and from the filter
        // image; the filter's two directions between the images.
        const std::array<VkDescriptorPoolSize, 3> poolSizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5},
        }};
        descriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), 5);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  descriptorPool_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                  "RayTracedShadowMaskDescriptorPool");

        const std::array<VkDescriptorSetLayout, 5> setLayouts{traceSetLayout_.handle(),
                                                              resolveSetLayout_.handle(),
                                                              resolveSetLayout_.handle(),
                                                              filterSetLayout_.handle(),
                                                              filterSetLayout_.handle()};
        std::array<VkDescriptorSet, 5> sets{};
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
        allocateInfo.pSetLayouts = setLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, sets.data()));
        traceSet_ = sets[0];
        resolveSet_ = sets[1];
        resolveFilteredSet_ = sets[2];
        traceToFilterSet_ = sets[3];
        filterToTraceSet_ = sets[4];

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

        VkDescriptorImageInfo filterStorageInfo{};
        filterStorageInfo.imageView = filter_.imageView();
        filterStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo filterSampledInfo{};
        filterSampledInfo.sampler = sampler_;
        filterSampledInfo.imageView = filter_.imageView();
        filterSampledInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo maskStorageInfo{};
        maskStorageInfo.imageView = mask_.imageView();
        maskStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        constexpr VkDescriptorType kSampled = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        constexpr VkDescriptorType kStorage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        std::array<VkWriteDescriptorSet, 13> writes{
            imageWrite(traceSet_, 0, kSampled, &depthInfo),
            imageWrite(traceSet_, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr),
            imageWrite(traceSet_, 2, kStorage, &traceStorageInfo),
            imageWrite(resolveSet_, 0, kSampled, &depthInfo),
            imageWrite(resolveSet_, 1, kSampled, &traceSampledInfo),
            imageWrite(resolveSet_, 2, kStorage, &maskStorageInfo),
            imageWrite(resolveFilteredSet_, 0, kSampled, &depthInfo),
            imageWrite(resolveFilteredSet_, 1, kSampled, &filterSampledInfo),
            imageWrite(resolveFilteredSet_, 2, kStorage, &maskStorageInfo),
            imageWrite(traceToFilterSet_, 0, kSampled, &traceSampledInfo),
            imageWrite(traceToFilterSet_, 1, kStorage, &filterStorageInfo),
            imageWrite(filterToTraceSet_, 0, kSampled, &filterSampledInfo),
            imageWrite(filterToTraceSet_, 1, kStorage, &traceStorageInfo),
        };
        writes[1].pNext = &topLevelInfo;
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
    resolveFilteredSet_ = VK_NULL_HANDLE;
    traceToFilterSet_ = VK_NULL_HANDLE;
    filterToTraceSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    filter_.reset();
    filterLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    trace_.reset();
    traceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    mask_.reset();
    maskLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

bool RayTracedShadowMask::dumpVisibility(const std::filesystem::path& path,
                                         VkExtent2D extent,
                                         VkCommandPool commandPool,
                                         VkQueue queue)
{
    if (!imageReady() || extent.width == 0 || extent.height == 0) {
        Logger::error("Ray-traced shadow dump: no mask image to read.");
        return false;
    }
    if (commandPool == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) {
        Logger::error("Ray-traced shadow dump: no command pool or queue to submit the copy on.");
        return false;
    }

    const VkDevice device = context_.vkDevice();
    const size_t texelCount = static_cast<size_t>(extent.width) * static_cast<size_t>(extent.height);
    // R32G32_UINT: visibility bits, then depth bits.
    const VkDeviceSize byteCount = static_cast<VkDeviceSize>(texelCount) * 2u * sizeof(uint32_t);

    vkDeviceWaitIdle(device);

    rhi::VulkanBufferCreateInfo bufferInfo{};
    bufferInfo.size = byteCount;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    rhi::VulkanBuffer staging;
    staging.createBuffer(context_, bufferInfo);
    if (!staging.valid()) {
        Logger::error("Ray-traced shadow dump: could not allocate the readback buffer.");
        return false;
    }

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
        Logger::error("Ray-traced shadow dump: could not allocate a command buffer.");
        return false;
    }

    // Back to the layout the graph tracks, so the next frame's first-use barrier
    // starts from what the image is actually in.
    const VkImageLayout restoreLayout =
        maskLayout_ == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : maskLayout_;
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barrier.oldLayout = maskLayout_;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = mask_.image();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(
        commandBuffer, mask_.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer(), 1, &region);
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = restoreLayout;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    const VkResult submitted = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (submitted == VK_SUCCESS) {
        vkQueueWaitIdle(queue);
        maskLayout_ = restoreLayout;
    }
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    if (submitted != VK_SUCCESS) {
        Logger::error("Ray-traced shadow dump: the copy submission failed.");
        return false;
    }

    std::vector<uint32_t> raw(texelCount * 2u);
    staging.download(std::as_writable_bytes(std::span<uint32_t>(raw.data(), raw.size())));
    std::vector<float> visibility(texelCount);
    size_t traced = 0;
    for (size_t index = 0; index < texelCount; ++index) {
        float value = std::bit_cast<float>(raw[index * 2u]);
        // The mask's two encodings: below zero nothing speaks for the pixel, and
        // two or more is a visibility traced from a surface facing away.
        if (value < 0.0f) {
            value = std::numeric_limits<float>::quiet_NaN();
        } else {
            if (value >= kRtMaskBackFacingBias) {
                value -= kRtMaskBackFacingBias;
            }
            ++traced;
        }
        visibility[index] = value;
    }
    if (!writePfmGray(path, extent.width, extent.height, visibility)) {
        return false;
    }
    Logger::info("Ray-traced shadow dump: wrote " + std::to_string(extent.width) + "x" + std::to_string(extent.height) +
                 " visibility to " + path.string() + " (" + std::to_string(traced) + " pixels traced).");
    return true;
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

void RayTracedShadowMask::recordFilter(VkCommandBuffer commandBuffer,
                                       RayTracedShadowFilterStage stage,
                                       const RayTracedShadowFilterPushConstants& push) const
{
    const rhi::ComputePipelineRef& pipeline =
        stage == RayTracedShadowFilterStage::Radius ? radiusPipeline_ : blurPipeline_;
    const VkDescriptorSet set =
        stage == RayTracedShadowFilterStage::BlurHorizontal ? filterToTraceSet_ : traceToFilterSet_;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(push)), &push);
    vkCmdDispatch(commandBuffer,
                  rayTracedShadowMaskGroupCount(push.traceExtent.x),
                  rayTracedShadowMaskGroupCount(push.traceExtent.y),
                  1);
}

void RayTracedShadowMask::recordResolve(VkCommandBuffer commandBuffer,
                                        const RayTracedShadowResolvePushConstants& push) const
{
    const VkDescriptorSet set = push.filtered != 0 ? resolveFilteredSet_ : resolveSet_;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_.pipeline());
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_.layout(), 0, 1, &set, 0, nullptr);
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
