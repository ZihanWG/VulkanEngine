#include "renderer/GpuCulling.h"

// RendererInternal.h supplies the shared cull constants/structs the relocated
// bodies use: GpuCullDrawItem / GpuCullFrameParams / GpuCullPushConstants,
// kGpuCull* sizes, kMaxDrawItems, the shaderPath helper, and VK_CHECK. This
// mirrors how PostProcessStack / DepthPyramid reuse the same internals.
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/DepthPyramid.h"
#include "renderer/GpuProfiler.h"
#include "renderer/RenderGraph.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve::renderer {

GpuCulling::GpuCulling(rhi::VulkanContext& context,
                       DepthPyramid& depthPyramid,
                       RenderGraph& renderGraph,
                       GpuProfiler& gpuProfiler,
                       const std::vector<rhi::VulkanBuffer>& frameIndirectDrawBuffers,
                       const std::vector<rhi::VulkanBuffer>& frameShadowIndirectDrawBuffers)
    : context_(context),
      depthPyramid_(depthPyramid),
      renderGraph_(renderGraph),
      gpuProfiler_(gpuProfiler),
      frameIndirectDrawBuffers_(frameIndirectDrawBuffers),
      frameShadowIndirectDrawBuffers_(frameShadowIndirectDrawBuffers)
{
}

GpuCulling::~GpuCulling()
{
    destroyResources();
}

void GpuCulling::createResources(uint32_t frameCount,
                                 bool wantMainCull,
                                 bool wantShadowCull,
                                 bool shadowIndirectActive)
{
    destroyResources();

    if (!wantMainCull) {
        if (wantShadowCull) {
            Logger::warn("GPU shadow culling unavailable because the shared GPU culling pipeline is disabled.");
        }
        return;
    }

    try {
        if (frameIndirectDrawBuffers_.size() != frameCount) {
            throw std::runtime_error("GPU culling requires one indirect output buffer per frame.");
        }

        if (depthPyramid_.image() == VK_NULL_HANDLE || depthPyramid_.sampler() == VK_NULL_HANDLE) {
            throw std::runtime_error("GPU culling requires a valid depth pyramid descriptor resource.");
        }

        createCullDescriptorLayout();
        createCullPipeline();
        createCullBuffers(frameCount);
        createCullDescriptorSets(frameCount);

        gpuCullingAvailable_ = true;
        Logger::info("GPU frustum culling enabled for main-pass indirect command generation and per-batch visible "
                     "count readback.");
        if (context_.device().drawIndexedIndirectCountAvailable()) {
            Logger::info("vkCmdDrawIndexedIndirectCount support detected; compacted per-batch indirect-count drawing "
                         "will be used when the main pass can use bindless multi-draw indirect.");
        }

        if (wantShadowCull) {
            createShadowCullingResources(frameCount, shadowIndirectActive);
        }
    } catch (const std::exception& error) {
        Logger::warn(std::string("GPU frustum culling unavailable; falling back to CPU culling: ") + error.what());
        destroyResources();
    }
}

void GpuCulling::createCullDescriptorLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Two-phase occlusion phase-result buffer (written in phase 1, read in
    // phase 2). Shadow culling binds it too (shared layout) but never uses it.
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    gpuCullDescriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              gpuCullDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "GpuCullDescriptorSetLayout");
}

void GpuCulling::createCullPipeline()
{
    const VkDescriptorSetLayout cullDescriptorSetLayout = gpuCullDescriptorSetLayout_.handle();
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(GpuCullPushConstants))};

    rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.shaderPath = shaderPath("cull.comp.spv");
    pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&cullDescriptorSetLayout, 1);
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.pipelineCache = context_.pipelineCache();
    gpuCullPipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), gpuCullPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "GpuCullComputePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), gpuCullPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "GpuCullPipelineLayout");
}

void GpuCulling::createCullBuffers(uint32_t frameCount)
{
    frameCullInputBuffers_.resize(frameCount);
    for (size_t frameIndex = 0; frameIndex < frameCullInputBuffers_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxDrawItems * sizeof(GpuCullDrawItem));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        frameCullInputBuffers_[frameIndex].createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameCullInputBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuCullInputBuffer" + std::to_string(frameIndex));
    }

    frameGpuCullParamBuffers_.resize(frameCount);
    for (size_t frameIndex = 0; frameIndex < frameGpuCullParamBuffers_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(sizeof(GpuCullFrameParams));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        frameGpuCullParamBuffers_[frameIndex].createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameGpuCullParamBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuCullFrameParamsBuffer" + std::to_string(frameIndex));
    }

    framePhaseResultBuffers_.resize(frameCount);
    for (size_t frameIndex = 0; frameIndex < framePhaseResultBuffers_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxDrawItems * sizeof(uint32_t));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        framePhaseResultBuffers_[frameIndex].createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  framePhaseResultBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuCullPhaseResultBuffer" + std::to_string(frameIndex));
    }

    frameBatchVisibleCountBuffers_.resize(frameCount);
    frameBatchVisibleCountReadbackBuffers_.resize(frameCount);
    frameGpuCullTotalDrawItems_.assign(frameCount, 0);
    frameGpuCullBatchCounts_.assign(frameCount, 0);
    frameGpuCullReadbackReady_.assign(frameCount, 0);
    frameGpuCullIndirectCountPath_.assign(frameCount, 0);
    for (size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        rhi::VulkanBufferCreateInfo gpuCountInfo{};
        gpuCountInfo.size = kGpuCullCountBufferSize;
        gpuCountInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        gpuCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameBatchVisibleCountBuffers_[frameIndex].createBuffer(context_, gpuCountInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameBatchVisibleCountBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuBatchVisibleCountBuffer" + std::to_string(frameIndex));

        rhi::VulkanBufferCreateInfo readbackCountInfo{};
        readbackCountInfo.size = kGpuCullCountBufferSize;
        readbackCountInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        readbackCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        readbackCountInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        frameBatchVisibleCountReadbackBuffers_[frameIndex].createBuffer(context_, readbackCountInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameBatchVisibleCountReadbackBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuBatchVisibleCountReadbackBuffer" + std::to_string(frameIndex));
    }
}

void GpuCulling::createCullDescriptorSets(uint32_t frameCount)
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(frameCount * 5);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(frameCount);

    gpuCullDescriptorPool_.create(context_.vkDevice(),
                                  std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()),
                                  static_cast<uint32_t>(frameCount));
    rhi::debug::setObjectName(context_.vkDevice(),
                              gpuCullDescriptorPool_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                              "GpuCullDescriptorPool");

    gpuCullDescriptorSets_.resize(frameCount, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> setLayouts(frameCount, gpuCullDescriptorSetLayout_.handle());
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = gpuCullDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(gpuCullDescriptorSets_.size());
    allocateInfo.pSetLayouts = setLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, gpuCullDescriptorSets_.data()));

    for (size_t frameIndex = 0; frameIndex < gpuCullDescriptorSets_.size(); ++frameIndex) {
        VkDescriptorBufferInfo inputBufferInfo{};
        inputBufferInfo.buffer = frameCullInputBuffers_[frameIndex].buffer();
        inputBufferInfo.offset = 0;
        inputBufferInfo.range = frameCullInputBuffers_[frameIndex].size();

        VkDescriptorBufferInfo outputBufferInfo{};
        outputBufferInfo.buffer = frameIndirectDrawBuffers_[frameIndex].buffer();
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = frameIndirectDrawBuffers_[frameIndex].size();

        VkDescriptorBufferInfo visibleCountBufferInfo{};
        visibleCountBufferInfo.buffer = frameBatchVisibleCountBuffers_[frameIndex].buffer();
        visibleCountBufferInfo.offset = 0;
        visibleCountBufferInfo.range = kGpuCullCountBufferSize;

        VkDescriptorImageInfo depthPyramidInfo{};
        depthPyramidInfo.sampler = depthPyramid_.sampler();
        depthPyramidInfo.imageView = depthPyramid_.imageView();
        depthPyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo frameParamsBufferInfo{};
        frameParamsBufferInfo.buffer = frameGpuCullParamBuffers_[frameIndex].buffer();
        frameParamsBufferInfo.offset = 0;
        frameParamsBufferInfo.range = frameGpuCullParamBuffers_[frameIndex].size();

        VkDescriptorBufferInfo phaseResultBufferInfo{};
        phaseResultBufferInfo.buffer = framePhaseResultBuffers_[frameIndex].buffer();
        phaseResultBufferInfo.offset = 0;
        phaseResultBufferInfo.range = framePhaseResultBuffers_[frameIndex].size();

        std::array<VkWriteDescriptorSet, 6> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = gpuCullDescriptorSets_[frameIndex];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &inputBufferInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = gpuCullDescriptorSets_[frameIndex];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &outputBufferInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = gpuCullDescriptorSets_[frameIndex];
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &visibleCountBufferInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = gpuCullDescriptorSets_[frameIndex];
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &depthPyramidInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = gpuCullDescriptorSets_[frameIndex];
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &frameParamsBufferInfo;

        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = gpuCullDescriptorSets_[frameIndex];
        writes[5].dstBinding = 5;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].pBufferInfo = &phaseResultBufferInfo;

        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  gpuCullDescriptorSets_[frameIndex],
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                  "GpuCullDescriptorSet" + std::to_string(frameIndex));
    }
}

void GpuCulling::destroyResources()
{
    destroyShadowCullingResources();

    gpuCullingAvailable_ = false;
    gpuCullDescriptorSets_.clear();
    gpuCullDescriptorPool_.reset();
    frameGpuCullIndirectCountPath_.clear();
    frameGpuCullReadbackReady_.clear();
    frameGpuCullBatchCounts_.clear();
    frameGpuCullTotalDrawItems_.clear();
    frameBatchVisibleCountReadbackBuffers_.clear();
    frameBatchVisibleCountBuffers_.clear();
    framePhaseResultBuffers_.clear();
    frameGpuCullParamBuffers_.clear();
    frameCullInputBuffers_.clear();
    gpuCullPipeline_.reset();
    gpuCullDescriptorSetLayout_.reset();
}

void GpuCulling::createShadowCullingResources(uint32_t frameCount, bool shadowIndirectActive)
{
    destroyShadowCullingResources();

    try {
        if (gpuCullDescriptorSetLayout_.handle() == VK_NULL_HANDLE ||
            gpuCullPipeline_.pipeline() == VK_NULL_HANDLE || gpuCullPipeline_.layout() == VK_NULL_HANDLE) {
            throw std::runtime_error("GPU shadow culling requires the shared cull.comp pipeline.");
        }
        if (!shadowIndirectActive) {
            throw std::runtime_error("GPU shadow culling requires the shadow indirect draw path.");
        }

        createShadowCullBuffers(frameCount);
        createShadowCullDescriptorSets(frameCount);

        gpuShadowCullingAvailable_ = true;
        Logger::info("GPU shadow culling preparation enabled with per-frame shadow cull input, compacted indirect "
                     "output, and per-batch visible count buffers.");
    } catch (const std::exception& error) {
        Logger::warn(std::string("GPU shadow culling unavailable; using CPU shadow culling fallback: ") +
                     error.what());
        destroyShadowCullingResources();
    }
}

void GpuCulling::createShadowCullBuffers(uint32_t frameCount)
{
    frameShadowCullInputBuffers_.resize(frameCount);
    for (size_t frameIndex = 0; frameIndex < frameShadowCullInputBuffers_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxDrawItems * sizeof(GpuCullDrawItem));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        frameShadowCullInputBuffers_[frameIndex].createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameShadowCullInputBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuShadowCullInputBuffer" + std::to_string(frameIndex));
    }

    frameShadowBatchVisibleCountBuffers_.resize(frameCount);
    frameShadowBatchVisibleCountReadbackBuffers_.resize(frameCount);
    frameGpuShadowCullTotalDrawItems_.assign(frameCount, 0);
    frameGpuShadowCullBatchCounts_.assign(frameCount, 0);
    frameGpuShadowCullReadbackReady_.assign(frameCount, 0);
    frameGpuShadowCullIndirectCountPath_.assign(frameCount, 0);
    for (size_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        rhi::VulkanBufferCreateInfo gpuCountInfo{};
        gpuCountInfo.size = kGpuCullCountBufferSize;
        gpuCountInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        gpuCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameShadowBatchVisibleCountBuffers_[frameIndex].createBuffer(context_, gpuCountInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameShadowBatchVisibleCountBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuShadowBatchVisibleCountBuffer" + std::to_string(frameIndex));

        rhi::VulkanBufferCreateInfo readbackCountInfo{};
        readbackCountInfo.size = kGpuCullCountBufferSize;
        readbackCountInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        readbackCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        readbackCountInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        frameShadowBatchVisibleCountReadbackBuffers_[frameIndex].createBuffer(context_, readbackCountInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameShadowBatchVisibleCountReadbackBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "GpuShadowBatchVisibleCountReadbackBuffer" + std::to_string(frameIndex));
    }
}

void GpuCulling::createShadowCullDescriptorSets(uint32_t frameCount)
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(frameCount * 5);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(frameCount);

    shadowCullDescriptorPool_.create(context_.vkDevice(),
                                     std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()),
                                     static_cast<uint32_t>(frameCount));
    rhi::debug::setObjectName(context_.vkDevice(),
                              shadowCullDescriptorPool_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                              "GpuShadowCullDescriptorPool");

    shadowCullDescriptorSets_.resize(frameCount, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> setLayouts(frameCount, gpuCullDescriptorSetLayout_.handle());
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = shadowCullDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(shadowCullDescriptorSets_.size());
    allocateInfo.pSetLayouts = setLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, shadowCullDescriptorSets_.data()));

    for (size_t frameIndex = 0; frameIndex < shadowCullDescriptorSets_.size(); ++frameIndex) {
        VkDescriptorBufferInfo inputBufferInfo{};
        inputBufferInfo.buffer = frameShadowCullInputBuffers_[frameIndex].buffer();
        inputBufferInfo.offset = 0;
        inputBufferInfo.range = frameShadowCullInputBuffers_[frameIndex].size();

        VkDescriptorBufferInfo outputBufferInfo{};
        outputBufferInfo.buffer = frameShadowIndirectDrawBuffers_[frameIndex].buffer();
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = frameShadowIndirectDrawBuffers_[frameIndex].size();

        VkDescriptorBufferInfo visibleCountBufferInfo{};
        visibleCountBufferInfo.buffer = frameShadowBatchVisibleCountBuffers_[frameIndex].buffer();
        visibleCountBufferInfo.offset = 0;
        visibleCountBufferInfo.range = kGpuCullCountBufferSize;

        VkDescriptorImageInfo depthPyramidInfo{};
        depthPyramidInfo.sampler = depthPyramid_.sampler();
        depthPyramidInfo.imageView = depthPyramid_.imageView();
        depthPyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo frameParamsBufferInfo{};
        frameParamsBufferInfo.buffer = frameGpuCullParamBuffers_[frameIndex].buffer();
        frameParamsBufferInfo.offset = 0;
        frameParamsBufferInfo.range = frameGpuCullParamBuffers_[frameIndex].size();

        VkDescriptorBufferInfo phaseResultBufferInfo{};
        phaseResultBufferInfo.buffer = framePhaseResultBuffers_[frameIndex].buffer();
        phaseResultBufferInfo.offset = 0;
        phaseResultBufferInfo.range = framePhaseResultBuffers_[frameIndex].size();

        std::array<VkWriteDescriptorSet, 6> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = shadowCullDescriptorSets_[frameIndex];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &inputBufferInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = shadowCullDescriptorSets_[frameIndex];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &outputBufferInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = shadowCullDescriptorSets_[frameIndex];
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &visibleCountBufferInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = shadowCullDescriptorSets_[frameIndex];
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &depthPyramidInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = shadowCullDescriptorSets_[frameIndex];
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &frameParamsBufferInfo;

        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = shadowCullDescriptorSets_[frameIndex];
        writes[5].dstBinding = 5;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].pBufferInfo = &phaseResultBufferInfo;

        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  shadowCullDescriptorSets_[frameIndex],
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                  "GpuShadowCullDescriptorSet" + std::to_string(frameIndex));
    }
}

void GpuCulling::destroyShadowCullingResources()
{
    gpuShadowCullingAvailable_ = false;
    shadowCullDescriptorSets_.clear();
    shadowCullDescriptorPool_.reset();
    frameGpuShadowCullIndirectCountPath_.clear();
    frameGpuShadowCullReadbackReady_.clear();
    frameGpuShadowCullBatchCounts_.clear();
    frameGpuShadowCullTotalDrawItems_.clear();
    frameShadowBatchVisibleCountReadbackBuffers_.clear();
    frameShadowBatchVisibleCountBuffers_.clear();
    frameShadowCullInputBuffers_.clear();
}

void GpuCulling::updateDepthPyramidDescriptors()
{
    if (depthPyramid_.imageView() == VK_NULL_HANDLE || depthPyramid_.sampler() == VK_NULL_HANDLE) {
        return;
    }

    VkDescriptorImageInfo depthPyramidInfo{};
    depthPyramidInfo.sampler = depthPyramid_.sampler();
    depthPyramidInfo.imageView = depthPyramid_.imageView();
    depthPyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const auto updateSets = [this, &depthPyramidInfo](const std::vector<VkDescriptorSet>& descriptorSets) {
        for (VkDescriptorSet descriptorSet : descriptorSets) {
            if (descriptorSet == VK_NULL_HANDLE) {
                continue;
            }

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
            write.dstBinding = 3;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &depthPyramidInfo;
            vkUpdateDescriptorSets(context_.vkDevice(), 1, &write, 0, nullptr);
        }
    };

    updateSets(gpuCullDescriptorSets_);
    updateSets(shadowCullDescriptorSets_);
}

rhi::VulkanBuffer& GpuCulling::cullInputBuffer(uint32_t frameIndex)
{
    return frameCullInputBuffers_.at(frameIndex);
}

rhi::VulkanBuffer& GpuCulling::shadowCullInputBuffer(uint32_t frameIndex)
{
    return frameShadowCullInputBuffers_.at(frameIndex);
}

rhi::VulkanBuffer& GpuCulling::paramBuffer(uint32_t frameIndex)
{
    return frameGpuCullParamBuffers_.at(frameIndex);
}

void GpuCulling::resetFrameCounters(uint32_t frameIndex, uint32_t mainTotalDrawItems)
{
    if (frameIndex < frameGpuCullTotalDrawItems_.size()) {
        frameGpuCullTotalDrawItems_[frameIndex] = mainTotalDrawItems;
    }
    if (frameIndex < frameGpuCullReadbackReady_.size()) {
        frameGpuCullReadbackReady_[frameIndex] = 0;
    }
    if (frameIndex < frameGpuCullBatchCounts_.size()) {
        frameGpuCullBatchCounts_[frameIndex] = 0;
    }
    if (frameIndex < frameGpuCullIndirectCountPath_.size()) {
        frameGpuCullIndirectCountPath_[frameIndex] = 0;
    }
    if (frameIndex < frameGpuShadowCullTotalDrawItems_.size()) {
        frameGpuShadowCullTotalDrawItems_[frameIndex] = 0;
    }
    if (frameIndex < frameGpuShadowCullReadbackReady_.size()) {
        frameGpuShadowCullReadbackReady_[frameIndex] = 0;
    }
    if (frameIndex < frameGpuShadowCullBatchCounts_.size()) {
        frameGpuShadowCullBatchCounts_[frameIndex] = 0;
    }
    if (frameIndex < frameGpuShadowCullIndirectCountPath_.size()) {
        frameGpuShadowCullIndirectCountPath_[frameIndex] = 0;
    }
}

void GpuCulling::setMainCullFrameInfo(uint32_t frameIndex, uint32_t batchCount, bool indirectCountPath)
{
    if (frameIndex < frameGpuCullBatchCounts_.size()) {
        frameGpuCullBatchCounts_[frameIndex] = batchCount;
    }
    if (frameIndex < frameGpuCullIndirectCountPath_.size()) {
        frameGpuCullIndirectCountPath_[frameIndex] = indirectCountPath ? 1 : 0;
    }
}

void GpuCulling::setShadowCullFrameInfo(uint32_t frameIndex,
                                        uint32_t totalDrawItems,
                                        uint32_t batchCount,
                                        bool indirectCountPath)
{
    if (frameIndex < frameGpuShadowCullTotalDrawItems_.size()) {
        frameGpuShadowCullTotalDrawItems_[frameIndex] = totalDrawItems;
    }
    if (frameIndex < frameGpuShadowCullBatchCounts_.size()) {
        frameGpuShadowCullBatchCounts_[frameIndex] = batchCount;
    }
    if (frameIndex < frameGpuShadowCullIndirectCountPath_.size()) {
        frameGpuShadowCullIndirectCountPath_[frameIndex] = indirectCountPath ? 1 : 0;
    }
}

void GpuCulling::recordMainCull(VkCommandBuffer commandBuffer,
                                uint32_t frameIndex,
                                bool active,
                                uint32_t drawItemCount,
                                const std::array<glm::vec4, 6>& frustumPlanes,
                                bool mainPassMultiDrawIndirect,
                                bool copyReadback)
{
    if (!active || drawItemCount == 0) {
        return;
    }
    if (frameIndex >= gpuCullDescriptorSets_.size() || frameIndex >= frameBatchVisibleCountBuffers_.size() ||
        frameIndex >= frameBatchVisibleCountReadbackBuffers_.size() || frameIndex >= frameGpuCullReadbackReady_.size()) {
        return;
    }

    VkBuffer visibleCountBuffer = frameBatchVisibleCountBuffers_.at(frameIndex).buffer();
    VkBuffer visibleCountReadbackBuffer = frameBatchVisibleCountReadbackBuffers_.at(frameIndex).buffer();
    if (visibleCountBuffer == VK_NULL_HANDLE || visibleCountReadbackBuffer == VK_NULL_HANDLE) {
        return;
    }

    const bool indirectCountPathActive = frameIndirectCountPathActive(frameIndex);

    const renderer::GpuProfileScope profileScope(gpuProfiler_, frameIndex, commandBuffer, "MainGpuCullingPass");
    renderGraph_.beginMainGpuCullingPass();
    rhi::debug::beginLabel(commandBuffer, "GpuCulling");
    depthPyramid_.ensureShaderReadLayout(commandBuffer);
    vkCmdFillBuffer(commandBuffer, visibleCountBuffer, 0, kGpuCullCountBufferSize, 0);

    VkBufferMemoryBarrier2 resetCountBarrier{};
    resetCountBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    resetCountBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    resetCountBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    resetCountBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resetCountBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    resetCountBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetCountBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetCountBarrier.buffer = visibleCountBuffer;
    resetCountBarrier.offset = 0;
    resetCountBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo resetDependencyInfo{};
    resetDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    resetDependencyInfo.bufferMemoryBarrierCount = 1;
    resetDependencyInfo.pBufferMemoryBarriers = &resetCountBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &resetDependencyInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.pipeline());

    const VkDescriptorSet descriptorSet = gpuCullDescriptorSets_[frameIndex];
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    GpuCullPushConstants pushConstants{};
    pushConstants.frustumPlanes = frustumPlanes;
    pushConstants.params = glm::uvec4(drawItemCount,
                                      mainPassMultiDrawIndirect ? 1U : 0U,
                                      indirectCountPathActive ? 1U : 0U,
                                      1U);
    vkCmdPushConstants(commandBuffer,
                       gpuCullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(GpuCullPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "ComputeCullDispatch");
    const uint32_t groupCount = (drawItemCount + kGpuCullLocalSize - 1) / kGpuCullLocalSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
    rhi::debug::endLabel(commandBuffer);

    if (copyReadback) {
        recordMainVisibleCountReadback(commandBuffer, frameIndex);
    }
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endMainGpuCullingPass();
}

void GpuCulling::recordMainVisibleCountReadback(VkCommandBuffer commandBuffer, uint32_t frameIndex)
{
    VkBuffer visibleCountBuffer = frameBatchVisibleCountBuffers_.at(frameIndex).buffer();
    VkBuffer visibleCountReadbackBuffer = frameBatchVisibleCountReadbackBuffers_.at(frameIndex).buffer();

    VkBufferMemoryBarrier2 visibleCountCopyBarrier{};
    visibleCountCopyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    visibleCountCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    visibleCountCopyBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    visibleCountCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    visibleCountCopyBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    visibleCountCopyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    visibleCountCopyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    visibleCountCopyBarrier.buffer = visibleCountBuffer;
    visibleCountCopyBarrier.offset = 0;
    visibleCountCopyBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo visibleCountCopyDependency{};
    visibleCountCopyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    visibleCountCopyDependency.bufferMemoryBarrierCount = 1;
    visibleCountCopyDependency.pBufferMemoryBarriers = &visibleCountCopyBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &visibleCountCopyDependency);

    VkBufferCopy visibleCountCopy{};
    visibleCountCopy.size = kGpuCullCountBufferSize;
    vkCmdCopyBuffer(commandBuffer, visibleCountBuffer, visibleCountReadbackBuffer, 1, &visibleCountCopy);

    VkBufferMemoryBarrier2 readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    readbackBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    readbackBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    readbackBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.buffer = visibleCountReadbackBuffer;
    readbackBarrier.offset = 0;
    readbackBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameGpuCullReadbackReady_[frameIndex] = 1;
}

void GpuCulling::recordMainCullPhase2(VkCommandBuffer commandBuffer,
                                      uint32_t frameIndex,
                                      bool active,
                                      uint32_t drawItemCount,
                                      const std::array<glm::vec4, 6>& frustumPlanes,
                                      bool mainPassMultiDrawIndirect)
{
    if (!active || drawItemCount == 0) {
        return;
    }
    if (frameIndex >= gpuCullDescriptorSets_.size() || frameIndex >= frameBatchVisibleCountBuffers_.size() ||
        frameIndex >= frameBatchVisibleCountReadbackBuffers_.size() || frameIndex >= frameGpuCullReadbackReady_.size()) {
        return;
    }

    VkBuffer visibleCountBuffer = frameBatchVisibleCountBuffers_.at(frameIndex).buffer();
    if (visibleCountBuffer == VK_NULL_HANDLE) {
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, frameIndex, commandBuffer, "MainGpuCullingPhase2");
    renderGraph_.beginMainGpuCullingPhase2Pass();
    rhi::debug::beginLabel(commandBuffer, "GpuCullingPhase2");
    depthPyramid_.ensureShaderReadLayout(commandBuffer);

    // Compacted path: reset only the per-batch visible counts, because phase 2
    // compacts its rescued draws into the same batch regions the phase-1 draws
    // already consumed. The stats counters at kGpuCullStatsCounterOffset persist
    // so phase 2 can append the rescued count to phase 1's totals. The
    // non-compacted path overwrites every fixed command slot instead, so no
    // count reset is needed there.
    if (frameIndirectCountPathActive(frameIndex)) {
        vkCmdFillBuffer(commandBuffer, visibleCountBuffer, 0, kBatchVisibleCountBufferSize, 0);

        VkBufferMemoryBarrier2 resetCountBarrier{};
        resetCountBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        resetCountBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        resetCountBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        resetCountBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        resetCountBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        resetCountBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resetCountBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resetCountBarrier.buffer = visibleCountBuffer;
        resetCountBarrier.offset = 0;
        resetCountBarrier.size = kBatchVisibleCountBufferSize;

        VkDependencyInfo resetDependencyInfo{};
        resetDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        resetDependencyInfo.bufferMemoryBarrierCount = 1;
        resetDependencyInfo.pBufferMemoryBarriers = &resetCountBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &resetDependencyInfo);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.pipeline());

    const VkDescriptorSet descriptorSet = gpuCullDescriptorSets_[frameIndex];
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    GpuCullPushConstants pushConstants{};
    pushConstants.frustumPlanes = frustumPlanes;
    // w = occlusion allowed (bit 0) + phase 2 (bit 1).
    pushConstants.params = glm::uvec4(drawItemCount,
                                      mainPassMultiDrawIndirect ? 1U : 0U,
                                      frameIndirectCountPathActive(frameIndex) ? 1U : 0U,
                                      3U);
    vkCmdPushConstants(commandBuffer,
                       gpuCullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(GpuCullPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "ComputeCullPhase2Dispatch");
    const uint32_t groupCount = (drawItemCount + kGpuCullLocalSize - 1) / kGpuCullLocalSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
    rhi::debug::endLabel(commandBuffer);

    recordMainVisibleCountReadback(commandBuffer, frameIndex);
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endMainGpuCullingPhase2Pass();
}

void GpuCulling::recordShadowCull(VkCommandBuffer commandBuffer,
                                  uint32_t frameIndex,
                                  bool active,
                                  uint32_t cascadeIndex,
                                  uint32_t cascadeCount,
                                  uint32_t drawItemCount,
                                  const std::array<glm::vec4, 6>& cascadeFrustumPlanes)
{
    if (!active || drawItemCount == 0) {
        return;
    }
    if (cascadeIndex >= cascadeCount) {
        return;
    }
    if (frameIndex >= shadowCullDescriptorSets_.size() ||
        frameIndex >= frameShadowBatchVisibleCountBuffers_.size() ||
        frameIndex >= frameShadowBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameShadowIndirectDrawBuffers_.size() ||
        frameIndex >= frameGpuShadowCullReadbackReady_.size()) {
        return;
    }

    VkBuffer visibleCountBuffer = frameShadowBatchVisibleCountBuffers_.at(frameIndex).buffer();
    VkBuffer visibleCountReadbackBuffer = frameShadowBatchVisibleCountReadbackBuffers_.at(frameIndex).buffer();
    VkBuffer shadowIndirectDrawBuffer = frameShadowIndirectDrawBuffers_.at(frameIndex).buffer();
    if (visibleCountBuffer == VK_NULL_HANDLE || visibleCountReadbackBuffer == VK_NULL_HANDLE ||
        shadowIndirectDrawBuffer == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize shadowIndirectBufferSize =
        std::min<VkDeviceSize>(frameShadowIndirectDrawBuffers_.at(frameIndex).size(),
                               static_cast<VkDeviceSize>(drawItemCount * sizeof(VkDrawIndexedIndirectCommand)));
    if (shadowIndirectBufferSize == 0) {
        return;
    }

    const std::string profileName = "ShadowGpuCullingCascade" + std::to_string(cascadeIndex);
    const renderer::GpuProfileScope profileScope(gpuProfiler_, frameIndex, commandBuffer, profileName);
    rhi::debug::beginLabel(commandBuffer, "GpuShadowCullingCascade" + std::to_string(cascadeIndex));
    depthPyramid_.ensureShaderReadLayout(commandBuffer);
    vkCmdFillBuffer(commandBuffer, visibleCountBuffer, 0, kGpuCullCountBufferSize, 0);
    vkCmdFillBuffer(commandBuffer, shadowIndirectDrawBuffer, 0, shadowIndirectBufferSize, 0);

    std::array<VkBufferMemoryBarrier2, 2> resetBarriers{};
    resetBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    resetBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    resetBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    resetBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resetBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    resetBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarriers[0].buffer = visibleCountBuffer;
    resetBarriers[0].offset = 0;
    resetBarriers[0].size = kGpuCullCountBufferSize;

    resetBarriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    resetBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    resetBarriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    resetBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resetBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    resetBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarriers[1].buffer = shadowIndirectDrawBuffer;
    resetBarriers[1].offset = 0;
    resetBarriers[1].size = shadowIndirectBufferSize;

    VkDependencyInfo resetDependencyInfo{};
    resetDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    resetDependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(resetBarriers.size());
    resetDependencyInfo.pBufferMemoryBarriers = resetBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &resetDependencyInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.pipeline());

    const VkDescriptorSet descriptorSet = shadowCullDescriptorSets_[frameIndex];
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    GpuCullPushConstants pushConstants{};
    pushConstants.frustumPlanes = cascadeFrustumPlanes;
    pushConstants.params = glm::uvec4(drawItemCount, 1U, 1U, 0U);
    vkCmdPushConstants(commandBuffer,
                       gpuCullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(GpuCullPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "ShadowCullDispatchCascade" + std::to_string(cascadeIndex));
    const uint32_t groupCount = (drawItemCount + kGpuCullLocalSize - 1) / kGpuCullLocalSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
    rhi::debug::endLabel(commandBuffer);

    std::array<VkBufferMemoryBarrier2, 2> computeBarriers{};
    computeBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    computeBarriers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    computeBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[0].buffer = shadowIndirectDrawBuffer;
    computeBarriers[0].offset = 0;
    computeBarriers[0].size = shadowIndirectBufferSize;

    computeBarriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    computeBarriers[1].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    computeBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[1].buffer = visibleCountBuffer;
    computeBarriers[1].offset = 0;
    computeBarriers[1].size = kGpuCullCountBufferSize;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(computeBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = computeBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    VkBufferCopy visibleCountCopy{};
    visibleCountCopy.size = kGpuCullCountBufferSize;
    vkCmdCopyBuffer(commandBuffer, visibleCountBuffer, visibleCountReadbackBuffer, 1, &visibleCountCopy);

    VkBufferMemoryBarrier2 readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    readbackBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    readbackBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    readbackBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.buffer = visibleCountReadbackBuffer;
    readbackBarrier.offset = 0;
    readbackBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameGpuShadowCullReadbackReady_[frameIndex] = 1;
    rhi::debug::endLabel(commandBuffer);
}

bool GpuCulling::readMainVisibleCount(bool active, uint32_t frameIndex, uint32_t& visibleCount)
{
    visibleCount = 0;
    if (!active || frameIndex >= frameBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameGpuCullReadbackReady_.size() || frameGpuCullReadbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = frameBatchVisibleCountReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    const bool indirectCountPathActive = frameIndirectCountPathActive(frameIndex);
    const uint32_t countEntryCount = indirectCountPathActive && frameIndex < frameGpuCullBatchCounts_.size()
                                         ? std::max(frameGpuCullBatchCounts_[frameIndex], 1U)
                                         : 1U;
    std::vector<uint32_t> visibleCounts(countEntryCount, 0);
    readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(visibleCounts.data(), visibleCounts.size())));

    uint64_t totalVisibleCount = 0;
    for (uint32_t count : visibleCounts) {
        totalVisibleCount += count;
    }

    visibleCount = static_cast<uint32_t>(std::min<uint64_t>(totalVisibleCount, kMaxDrawItems));
    if (frameIndex < frameGpuCullTotalDrawItems_.size()) {
        visibleCount = std::min(visibleCount, frameGpuCullTotalDrawItems_[frameIndex]);
    }
    return true;
}

bool GpuCulling::readMainCounters(bool active, uint32_t frameIndex, GpuCullCounters& counters)
{
    counters = {};
    if (!active || frameIndex >= frameBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameGpuCullReadbackReady_.size() || frameGpuCullReadbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = frameBatchVisibleCountReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    std::array<uint32_t, kGpuCullStatsCounterCount> values{};
    readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(values.data(), values.size())),
                            static_cast<VkDeviceSize>(kGpuCullStatsCounterOffset * sizeof(uint32_t)));

    counters.totalDrawItems = values[0];
    counters.visibleDrawItems = values[1];
    counters.frustumCulledDrawItems = values[2];
    counters.occlusionCulledDrawItems = values[3];
    counters.phase2RescuedDrawItems = values[4];
    if (frameIndex < frameGpuCullTotalDrawItems_.size()) {
        counters.totalDrawItems = std::min(counters.totalDrawItems, frameGpuCullTotalDrawItems_[frameIndex]);
        counters.visibleDrawItems = std::min(counters.visibleDrawItems, counters.totalDrawItems);
        counters.frustumCulledDrawItems = std::min(counters.frustumCulledDrawItems, counters.totalDrawItems);
        counters.occlusionCulledDrawItems = std::min(counters.occlusionCulledDrawItems, counters.totalDrawItems);
        counters.phase2RescuedDrawItems = std::min(counters.phase2RescuedDrawItems, counters.occlusionCulledDrawItems);
    }
    return true;
}

bool GpuCulling::readShadowVisibleCount(bool active, uint32_t frameIndex, uint32_t& visibleCount)
{
    visibleCount = 0;
    if (!active || frameIndex >= frameShadowBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameGpuShadowCullReadbackReady_.size() || frameGpuShadowCullReadbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = frameShadowBatchVisibleCountReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    const uint32_t countEntryCount = frameIndex < frameGpuShadowCullBatchCounts_.size()
                                         ? std::max(frameGpuShadowCullBatchCounts_[frameIndex], 1U)
                                         : 1U;
    std::vector<uint32_t> visibleCounts(countEntryCount, 0);
    readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(visibleCounts.data(), visibleCounts.size())));

    uint64_t totalVisibleCount = 0;
    for (uint32_t count : visibleCounts) {
        totalVisibleCount += count;
    }

    visibleCount = static_cast<uint32_t>(std::min<uint64_t>(totalVisibleCount, kMaxDrawItems));
    if (frameIndex < frameGpuShadowCullTotalDrawItems_.size()) {
        visibleCount = std::min(visibleCount, frameGpuShadowCullTotalDrawItems_[frameIndex]);
    }
    return true;
}

bool GpuCulling::mainResourcesReady(uint32_t frameCount) const
{
    return gpuCullingAvailable_ && gpuCullPipeline_.pipeline() != VK_NULL_HANDLE &&
           gpuCullPipeline_.layout() != VK_NULL_HANDLE && !gpuCullDescriptorSets_.empty() &&
           frameCullInputBuffers_.size() == frameCount && frameBatchVisibleCountBuffers_.size() == frameCount &&
           frameBatchVisibleCountReadbackBuffers_.size() == frameCount &&
           frameGpuCullParamBuffers_.size() == frameCount;
}

bool GpuCulling::shadowResourcesReady(uint32_t frameCount) const
{
    return gpuShadowCullingAvailable_ && gpuCullPipeline_.pipeline() != VK_NULL_HANDLE &&
           gpuCullPipeline_.layout() != VK_NULL_HANDLE && !shadowCullDescriptorSets_.empty() &&
           frameShadowCullInputBuffers_.size() == frameCount &&
           frameShadowBatchVisibleCountBuffers_.size() == frameCount &&
           frameShadowBatchVisibleCountReadbackBuffers_.size() == frameCount &&
           frameGpuCullParamBuffers_.size() == frameCount;
}

bool GpuCulling::frameIndirectCountPathActive(uint32_t frameIndex) const
{
    return frameIndex < frameGpuCullIndirectCountPath_.size() && frameGpuCullIndirectCountPath_[frameIndex] != 0;
}

bool GpuCulling::frameShadowIndirectCountPathActive(uint32_t frameIndex) const
{
    return frameIndex < frameGpuShadowCullIndirectCountPath_.size() &&
           frameGpuShadowCullIndirectCountPath_[frameIndex] != 0;
}

uint32_t GpuCulling::mainTotalDrawItems(uint32_t frameIndex) const
{
    return frameIndex < frameGpuCullTotalDrawItems_.size() ? frameGpuCullTotalDrawItems_[frameIndex] : 0;
}

uint32_t GpuCulling::shadowTotalDrawItems(uint32_t frameIndex) const
{
    return frameIndex < frameGpuShadowCullTotalDrawItems_.size() ? frameGpuShadowCullTotalDrawItems_[frameIndex] : 0;
}

const rhi::VulkanBuffer& GpuCulling::visibleCountBuffer(uint32_t frameIndex) const
{
    return frameBatchVisibleCountBuffers_.at(frameIndex);
}

const rhi::VulkanBuffer& GpuCulling::shadowVisibleCountBuffer(uint32_t frameIndex) const
{
    return frameShadowBatchVisibleCountBuffers_.at(frameIndex);
}

uint32_t GpuCulling::mainBatchCount(uint32_t frameIndex) const
{
    return frameIndex < frameGpuCullBatchCounts_.size() ? frameGpuCullBatchCounts_[frameIndex] : 0;
}

uint32_t GpuCulling::shadowBatchCount(uint32_t frameIndex) const
{
    return frameIndex < frameGpuShadowCullBatchCounts_.size() ? frameGpuShadowCullBatchCounts_[frameIndex] : 0;
}

} // namespace ve::renderer
