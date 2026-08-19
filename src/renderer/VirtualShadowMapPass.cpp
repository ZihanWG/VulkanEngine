#include "renderer/VirtualShadowMapPass.h"

// RendererInternal.h provides shaderPath() and the other internal-linkage
// helpers the renderer translation units share, the same way DepthPyramid.cpp
// and PostProcessStack.cpp reuse them.
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/DepthPyramid.h"
#include "renderer/GpuProfiler.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <string>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

namespace ve::renderer {
namespace {

// Mirrors the VsmMarkParamsBuffer block in vsm_page_mark.comp. std430 puts each
// mat4 on a 16-byte boundary and each vec4 after it, so the layout is the
// natural one -- but it is pinned here anyway, because a binary-layout mismatch
// is exactly what validation layers cannot see.
struct VsmMarkParams {
    glm::mat4 depthViewProjectionInverse{1.0f};
    glm::mat4 lightView{1.0f};
    glm::vec4 depthCameraPosition{0.0f};
    glm::vec4 clipmapParams{0.0f};
    glm::uvec4 sizesAndFlags{0u};
};

static_assert(offsetof(VsmMarkParams, depthViewProjectionInverse) == 0);
static_assert(offsetof(VsmMarkParams, lightView) == 64);
static_assert(offsetof(VsmMarkParams, depthCameraPosition) == 128);
static_assert(offsetof(VsmMarkParams, clipmapParams) == 144);
static_assert(offsetof(VsmMarkParams, sizesAndFlags) == 160);
static_assert(sizeof(VsmMarkParams) == 176);

constexpr VkDeviceSize kRequestBufferSize = static_cast<VkDeviceSize>(kVsmPageRequestWordCount) * sizeof(uint32_t);
constexpr uint32_t kMarkWorkgroupSize = 8;

void bufferBarrier(VkCommandBuffer commandBuffer,
                   VkBuffer buffer,
                   VkDeviceSize size,
                   VkPipelineStageFlags2 srcStage,
                   VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage,
                   VkAccessFlags2 dstAccess)
{
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = size;

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace

VirtualShadowMapPass::VirtualShadowMapPass(rhi::VulkanContext& context,
                                           DepthPyramid& depthPyramid,
                                           GpuProfiler& gpuProfiler)
    : context_(context), depthPyramid_(depthPyramid), gpuProfiler_(gpuProfiler)
{
}

VirtualShadowMapPass::~VirtualShadowMapPass()
{
    destroyResources();
}

void VirtualShadowMapPass::createResources(uint32_t frameCount)
{
    destroyResources();
    if (frameCount == 0) {
        return;
    }

    // The image and sampler have to exist for the descriptors to name them.
    // Deliberately not DepthPyramid::valid(), which means something else: that
    // the pyramid has been BUILT and its contents can be trusted. That is a
    // per-frame question, answered by VsmMarkFrameInput::depthValid, and it is
    // false at startup because no frame has run yet. Reported rather than
    // thrown: the renderer keeps drawing with CSM either way.
    if (depthPyramid_.imageView() == VK_NULL_HANDLE || depthPyramid_.sampler() == VK_NULL_HANDLE) {
        Logger::info("Virtual shadow map page marking is unavailable: no depth pyramid image to mark from.");
        return;
    }

    try {
        createDescriptorSetLayout();
        createPipeline();
        createBuffers(frameCount);
        createDescriptorSets(frameCount);
        available_ = pipeline_.pipeline() != VK_NULL_HANDLE;
    } catch (const std::exception& error) {
        Logger::warn(std::string("Virtual shadow map page marking failed to initialise: ") + error.what());
        destroyResources();
        return;
    }

    if (available_) {
        Logger::info("Virtual shadow map page marking is available. " + std::to_string(kVsmMaxVirtualPages) +
                     " virtual pages across " + std::to_string(kVsmMaxClipmapLevels) + " levels, " +
                     std::to_string(kVsmPageSize) + "px pages.");
    }
}

void VirtualShadowMapPass::destroyResources()
{
    available_ = false;
    lastMarkThreadCount_ = 0;
    readbackReady_.clear();
    descriptorSets_.clear();
    descriptorPool_.reset();
    paramsBuffers_.clear();
    requestReadbackBuffers_.clear();
    requestBuffers_.clear();
    pipeline_.reset();
    descriptorSetLayout_.reset();
}

void VirtualShadowMapPass::createDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
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

    descriptorSetLayout_.create(context_.vkDevice(),
                                std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              descriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "VsmPageMarkDescriptorSetLayout");
}

void VirtualShadowMapPass::createPipeline()
{
    const VkDescriptorSetLayout setLayout = descriptorSetLayout_.handle();
    if (setLayout == VK_NULL_HANDLE) {
        return;
    }

    rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.shaderPath = shaderPath("vsm_page_mark.comp.spv");
    pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&setLayout, 1);
    pipelineInfo.pipelineCache = context_.pipelineCache();
    pipeline_.create(context_.vkDevice(), pipelineInfo);

    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "VsmPageMarkComputePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "VsmPageMarkPipelineLayout");
}

void VirtualShadowMapPass::createBuffers(uint32_t frameCount)
{
    requestBuffers_.resize(frameCount);
    requestReadbackBuffers_.resize(frameCount);
    paramsBuffers_.resize(frameCount);
    readbackReady_.assign(frameCount, 0);

    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        rhi::VulkanBufferCreateInfo requestInfo{};
        requestInfo.size = kRequestBufferSize;
        requestInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        requestInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        requestBuffers_[frameIndex].createBuffer(context_, requestInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  requestBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "VsmPageRequestBuffer" + std::to_string(frameIndex));

        rhi::VulkanBufferCreateInfo readbackInfo{};
        readbackInfo.size = kRequestBufferSize;
        readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        readbackInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        readbackInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        requestReadbackBuffers_[frameIndex].createBuffer(context_, readbackInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  requestReadbackBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "VsmPageRequestReadbackBuffer" + std::to_string(frameIndex));

        rhi::VulkanBufferCreateInfo paramsInfo{};
        paramsInfo.size = sizeof(VsmMarkParams);
        paramsInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        paramsInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        paramsInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        paramsBuffers_[frameIndex].createBuffer(context_, paramsInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  paramsBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "VsmMarkParamsBuffer" + std::to_string(frameIndex));
    }
}

void VirtualShadowMapPass::createDescriptorSets(uint32_t frameCount)
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = frameCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = frameCount * 2;
    descriptorPool_.create(
        context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), frameCount);

    std::vector<VkDescriptorSetLayout> layouts(frameCount, descriptorSetLayout_.handle());
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool_.handle();
    allocateInfo.descriptorSetCount = frameCount;
    allocateInfo.pSetLayouts = layouts.data();

    descriptorSets_.resize(frameCount, VK_NULL_HANDLE);
    if (vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, descriptorSets_.data()) != VK_SUCCESS) {
        descriptorSets_.clear();
        throw std::runtime_error("Failed to allocate VSM page-mark descriptor sets.");
    }

    refreshDepthPyramidBinding();
}

void VirtualShadowMapPass::refreshDepthPyramidBinding()
{
    if (descriptorSets_.empty() || depthPyramid_.imageView() == VK_NULL_HANDLE ||
        depthPyramid_.sampler() == VK_NULL_HANDLE) {
        return;
    }

    for (uint32_t frameIndex = 0; frameIndex < descriptorSets_.size(); ++frameIndex) {
        VkDescriptorImageInfo pyramidInfo{};
        pyramidInfo.sampler = depthPyramid_.sampler();
        pyramidInfo.imageView = depthPyramid_.imageView();
        pyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorBufferInfo requestInfo{requestBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo paramsInfo{paramsBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};

        std::array<VkWriteDescriptorSet, 3> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[frameIndex];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &pyramidInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[frameIndex];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &requestInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptorSets_[frameIndex];
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &paramsInfo;

        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void VirtualShadowMapPass::recordMarkPass(VkCommandBuffer commandBuffer,
                                          uint32_t frameIndex,
                                          const VsmMarkFrameInput& input)
{
    if (!available_ || frameIndex >= descriptorSets_.size()) {
        return;
    }

    const VkBuffer requestBuffer = requestBuffers_[frameIndex].buffer();
    const VkBuffer readbackBuffer = requestReadbackBuffers_[frameIndex].buffer();
    if (requestBuffer == VK_NULL_HANDLE || readbackBuffer == VK_NULL_HANDLE) {
        return;
    }

    const GpuProfileScope profileScope(gpuProfiler_, frameIndex, commandBuffer, "VsmPageMark");
    rhi::debug::beginLabel(commandBuffer, "VsmPageMark");

    const VsmClipmapSettings clipmap = clampVsmClipmapSettings(input.clipmap);
    const glm::mat4 lightView = vsmLightView(input.lightDirection);
    const glm::vec2 cameraLightSpaceXy = glm::vec2(lightView * glm::vec4(input.cameraPosition, 1.0f));

    const uint32_t stride = std::max(input.blockStride, 1u);
    const uint32_t depthWidth = input.depthExtent.width;
    const uint32_t depthHeight = input.depthExtent.height;
    // A skipped dispatch still clears and copies, so the readback reports an
    // honest zero rather than the previous frame's page set.
    const bool dispatchThisFrame = input.depthValid && depthWidth > 0 && depthHeight > 0;

    VsmMarkParams params{};
    params.depthViewProjectionInverse = glm::inverse(input.depthViewProjection);
    params.lightView = lightView;
    params.depthCameraPosition = glm::vec4(input.depthCameraPosition, input.projScaleY);
    params.clipmapParams =
        glm::vec4(cameraLightSpaceXy.x, cameraLightSpaceXy.y, clipmap.level0Extent, clipmap.texelsPerPixel);
    params.sizesAndFlags = glm::uvec4(clipmap.levelCount, stride, depthWidth, depthHeight);
    paramsBuffers_[frameIndex].upload(std::as_bytes(std::span<const VsmMarkParams>(&params, 1)));

    // The request set is rebuilt from scratch every frame: a page needed last
    // frame is not evidence it is needed now, and carrying stale bits over would
    // make the measurement monotonically grow toward "everything".
    vkCmdFillBuffer(commandBuffer, requestBuffer, 0, kRequestBufferSize, 0);
    bufferBarrier(commandBuffer,
                  requestBuffer,
                  kRequestBufferSize,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    uint32_t threadCount = 0;
    if (dispatchThisFrame) {
        const uint32_t blocksX = (depthWidth + stride - 1u) / stride;
        const uint32_t blocksY = (depthHeight + stride - 1u) / stride;
        threadCount = blocksX * blocksY;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.pipeline());
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_.layout(),
                                0,
                                1,
                                &descriptorSets_[frameIndex],
                                0,
                                nullptr);
        vkCmdDispatch(commandBuffer,
                      (blocksX + kMarkWorkgroupSize - 1u) / kMarkWorkgroupSize,
                      (blocksY + kMarkWorkgroupSize - 1u) / kMarkWorkgroupSize,
                      1);

        bufferBarrier(commandBuffer,
                      requestBuffer,
                      kRequestBufferSize,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT);
    } else {
        // Nothing wrote it, but the fill did; the copy below still needs the
        // fill's write visible to the transfer read.
        bufferBarrier(commandBuffer,
                      requestBuffer,
                      kRequestBufferSize,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    lastMarkThreadCount_ = threadCount;

    VkBufferCopy copyRegion{};
    copyRegion.size = kRequestBufferSize;
    vkCmdCopyBuffer(commandBuffer, requestBuffer, readbackBuffer, 1, &copyRegion);

    // Host visibility is never inferred by the graph -- this buffer is not a
    // graph resource, and the CPU read happens a frame later once the fence has
    // proved the copy retired.
    bufferBarrier(commandBuffer,
                  readbackBuffer,
                  kRequestBufferSize,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_HOST_BIT,
                  VK_ACCESS_2_HOST_READ_BIT);

    if (frameIndex < readbackReady_.size()) {
        readbackReady_[frameIndex] = 1;
    }

    rhi::debug::endLabel(commandBuffer);
}

bool VirtualShadowMapPass::readRequestStats(uint32_t frameIndex, uint32_t levelCount, VsmPageRequestStats& stats)
{
    stats = {};
    if (!available_ || frameIndex >= requestReadbackBuffers_.size() || frameIndex >= readbackReady_.size() ||
        readbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = requestReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    std::array<uint32_t, kVsmPageRequestWordCount> words{};
    readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(words.data(), words.size())));
    stats = vsmDecodeRequestStats(words.data(), levelCount);
    return true;
}

} // namespace ve::renderer
