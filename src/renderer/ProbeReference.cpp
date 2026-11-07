#include "renderer/ProbeReference.h"

#include "core/Logger.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <array>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve::renderer {

namespace {

constexpr uint32_t kBindingDepth = 0;
constexpr uint32_t kBindingNormals = 1;
constexpr uint32_t kBindingAccumulation = 2;
constexpr uint32_t kBindingEstimate = 3;

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

} // namespace

ProbeReference::ProbeReference(rhi::VulkanContext& context,
                               rhi::VulkanSwapchain& swapchain,
                               rhi::VulkanPipelineStore& pipelineStore,
                               const RenderResolution& renderResolution)
    : context_(context), swapchain_(swapchain), pipelineStore_(pipelineStore), renderResolution_(renderResolution)
{}

ProbeReference::~ProbeReference()
{
    destroyResources();
}

void ProbeReference::createPipeline(const std::filesystem::path& shaderPath,
                                    VkDescriptorSetLayout materialSetLayout,
                                    VkDescriptorSetLayout bindlessSetLayout)
{
    if (materialSetLayout == VK_NULL_HANDLE || bindlessSetLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("the reference needs the material set and the bindless heap");
    }
    // Created once: the set allocated against it outlives a pipeline rebuild that
    // is not also a resize.
    if (setLayout_.handle() == VK_NULL_HANDLE) {
        const std::array<VkDescriptorSetLayoutBinding, 4> bindings{
            computeBinding(kBindingDepth, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
            computeBinding(kBindingNormals, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
            computeBinding(kBindingAccumulation, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
            computeBinding(kBindingEstimate, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
        };
        setLayout_.create(context_.vkDevice(),
                          std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
        rhi::debug::setObjectName(
            context_.vkDevice(), setLayout_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "ProbeReferenceSetLayout");
    }
    const std::array<VkDescriptorSetLayout, 3> layouts{materialSetLayout, bindlessSetLayout, setLayout_.handle()};
    const VkPushConstantRange push{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(ProbeReferencePushConstants))};
    rhi::VulkanComputePipelineCreateInfo info{};
    info.shaderPath = shaderPath;
    info.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(layouts.data(), layouts.size());
    info.pushConstantRanges = std::span<const VkPushConstantRange>(&push, 1);
    info.pipelineCache = context_.pipelineCache();
    pipeline_ = pipelineStore_.get(context_.vkDevice(), info, "ProbeReferencePipeline");
}

void ProbeReference::destroyPipeline()
{
    pipeline_.reset();
    setLayout_.reset();
}

void ProbeReference::createResources(VkImageView normalRoughnessView)
{
    destroyResources();
    if (setLayout_.handle() == VK_NULL_HANDLE || normalRoughnessView == VK_NULL_HANDLE ||
        !swapchain_.depthSupportsSampling()) {
        return;
    }
    try {
        const VkExtent2D extent = renderResolution_.allocationExtent();
        rhi::VulkanImageCreateInfo imageInfo{};
        imageInfo.width = extent.width;
        imageInfo.height = extent.height;
        imageInfo.format = kFormat;
        // Transfer source for readStats.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageInfo.debugName = "ProbeReferenceAccumulation";
        accumulation_.create(context_, imageInfo);
        imageInfo.debugName = "ProbeReferenceEstimate";
        estimate_.create(context_, imageInfo);
        accumulationLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        estimateLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &sampler_));
        rhi::debug::setObjectName(context_.vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "ProbeReferenceSampler");

        const std::array<VkDescriptorPoolSize, 2> poolSizes{{
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        }};
        descriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), 1);
        const VkDescriptorSetLayout layout = setLayout_.handle();
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &layout;
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, &descriptorSet_));

        VkDescriptorImageInfo depthInfo{};
        depthInfo.sampler = sampler_;
        depthInfo.imageView = swapchain_.depthImageView();
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo normalInfo{};
        normalInfo.sampler = sampler_;
        normalInfo.imageView = normalRoughnessView;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo accumulationInfo{};
        accumulationInfo.imageView = accumulation_.imageView();
        accumulationInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo estimateInfo{};
        estimateInfo.imageView = estimate_.imageView();
        estimateInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        const std::array<VkWriteDescriptorSet, 4> writes{
            imageWrite(descriptorSet_, kBindingDepth, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo),
            imageWrite(descriptorSet_, kBindingNormals, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalInfo),
            imageWrite(descriptorSet_, kBindingAccumulation, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumulationInfo),
            imageWrite(descriptorSet_, kBindingEstimate, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &estimateInfo),
        };
        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    } catch (const std::exception& error) {
        Logger::warn(std::string("Probe GI reference unavailable: ") + error.what());
        destroyResources();
    }
}

void ProbeReference::destroyResources()
{
    descriptorSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    accumulation_.reset();
    estimate_.reset();
    accumulationLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    estimateLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void ProbeReference::record(VkCommandBuffer commandBuffer,
                            const ProbeReferencePushConstants& push,
                            VkDescriptorSet materialSet,
                            VkDescriptorSet bindlessSet) const
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());
    const std::array<VkDescriptorSet, 3> sets{materialSet, bindlessSet, descriptorSet_};
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
        commandBuffer, (push.extent.x + kLocalSize - 1) / kLocalSize, (push.extent.y + kLocalSize - 1) / kLocalSize, 1);
}

std::optional<ProbeReferenceStats>
ProbeReference::readStats(VkExtent2D extent, VkCommandPool commandPool, VkQueue queue)
{
    if (!available() || extent.width == 0 || extent.height == 0 || commandPool == VK_NULL_HANDLE ||
        queue == VK_NULL_HANDLE) {
        return std::nullopt;
    }
    const VkDevice device = context_.vkDevice();
    const size_t texelCount = static_cast<size_t>(extent.width) * extent.height;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(texelCount) * 4u * sizeof(float);
    vkDeviceWaitIdle(device);

    rhi::VulkanBufferCreateInfo bufferInfo{};
    bufferInfo.size = imageBytes * 2u;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    rhi::VulkanBuffer staging;
    staging.createBuffer(context_, bufferInfo);
    if (!staging.valid()) {
        return std::nullopt;
    }
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
        return std::nullopt;
    }
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Each image to TRANSFER_SRC, copied, and back to the layout the graph
    // tracks, so the next frame's first barrier starts from the truth.
    const std::array<std::pair<const rhi::VulkanImage*, VkImageLayout*>, 2> images{{
        {&accumulation_, &accumulationLayout_},
        {&estimate_, &estimateLayout_},
    }};
    std::array<VkImageLayout, 2> restore{};
    for (size_t index = 0; index < images.size(); ++index) {
        const VkImageLayout current = *images[index].second;
        restore[index] = current == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : current;
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout = current;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = images[index].first->image();
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);

        VkBufferImageCopy region{};
        region.bufferOffset = imageBytes * index;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer,
                               images[index].first->image(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.buffer(),
                               1,
                               &region);

        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = restore[index];
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    const VkResult submitted = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (submitted == VK_SUCCESS) {
        vkQueueWaitIdle(queue);
        accumulationLayout_ = restore[0];
        estimateLayout_ = restore[1];
    }
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    if (submitted != VK_SUCCESS) {
        return std::nullopt;
    }

    std::vector<float> raw(texelCount * 8u);
    staging.download(std::as_writable_bytes(std::span<float>(raw.data(), raw.size())));
    return probeReferenceStats(std::span<const float>(raw.data(), texelCount * 4u),
                               std::span<const float>(raw.data() + texelCount * 4u, texelCount * 4u));
}

} // namespace ve::renderer
