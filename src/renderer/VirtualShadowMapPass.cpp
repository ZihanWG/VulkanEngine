#include "renderer/VirtualShadowMapPass.h"

// RendererInternal.h provides shaderPath() and the other internal-linkage
// helpers the renderer translation units share, the same way DepthPyramid.cpp
// and PostProcessStack.cpp reuse them.
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/Bounds.h"
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
constexpr VkDeviceSize kPageTableBufferSize = static_cast<VkDeviceSize>(kVsmMaxVirtualPages) * sizeof(VsmPageTableEntry);
constexpr uint32_t kMarkWorkgroupSize = 8;

// Stand-in when a frame slot has no remembered window, which only happens before
// that slot has recorded a marking pass -- and in that case there is no request
// set to interpret either.
const std::array<glm::ivec2, kVsmMaxClipmapLevels> kZeroOrigins{};

// One counter per (page, bucket) plus a trailing over-cap counter.
constexpr uint32_t kCullCounterCount = kMaxVsmPagesPerFrame * kVsmCasterBucketCount + 1u;
constexpr uint32_t kCullOverflowCounterIndex = kMaxVsmPagesPerFrame * kVsmCasterBucketCount;
constexpr VkDeviceSize kCullCounterBufferSize = static_cast<VkDeviceSize>(kCullCounterCount) * sizeof(uint32_t);
constexpr uint32_t kCullWorkgroupSize = 64;

// Mirrors the push block in vsm_page_cull.comp.
struct VsmPageCullPushConstants {
    uint32_t pageCount = 0;
    uint32_t drawItemCount = 0;
    uint32_t pageCommandStride = 0;
    uint32_t overflowCounterIndex = 0;
};

static_assert(sizeof(VsmPageCullPushConstants) == 16);

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

void VirtualShadowMapPass::createPagePoolResources(uint32_t frameCount)
{
    pagePool_.reset();
    pageTableBuffers_.clear();
    allocator_.reset();
    residencyValid_ = false;
    dirtyPages_.clear();
    if (frameCount == 0) {
        return;
    }

    try {
        createPagePool();

        pageTableBuffers_.resize(frameCount);
        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            // Device address as well as a storage-buffer binding: the page pass
            // and its cull dispatch bind it as a descriptor, but the main pass
            // fragment shader reaches it by address, because the main pass
            // push-constant block is already full at its 128-byte guaranteed
            // size.
            rhi::VulkanBufferCreateInfo tableInfo{};
            tableInfo.size = kPageTableBufferSize;
            tableInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            tableInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            tableInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            tableInfo.requestDeviceAddress = true;
            pageTableBuffers_[frameIndex].createBuffer(context_, tableInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      pageTableBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "VsmPageTableBuffer" + std::to_string(frameIndex));
        }
    } catch (const std::exception& error) {
        Logger::warn(std::string("Virtual shadow map page pool failed to initialise: ") + error.what());
        pagePool_.reset();
        pageTableBuffers_.clear();
        return;
    }

    Logger::info("Virtual shadow map page pool is available. " + std::to_string(kVsmMaxVirtualPages) +
                 " virtual pages across " + std::to_string(kVsmMaxClipmapLevels) + " levels, " +
                 std::to_string(kVsmPageSize) + "px pages, into a " + std::to_string(kVsmPagePoolSize) + "x" +
                 std::to_string(kVsmPagePoolSize) + " pool of " + std::to_string(kVsmPagePoolPageCount) +
                 " physical pages.");
}

void VirtualShadowMapPass::createResources(uint32_t frameCount)
{
    destroyMarkResources();
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
        destroyMarkResources();
        return;
    }

    if (available_) {
        Logger::info("Virtual shadow map page marking is available.");
    }
}

void VirtualShadowMapPass::destroyCullResources()
{
    cullAvailable_ = false;
    cullDrawItemCapacity_ = 0;
    cullReadbackReady_.clear();
    cullSets_.clear();
    cullDescriptorPool_.reset();
    casterFlagBuffers_.clear();
    cullOverflowReadbackBuffers_.clear();
    cullVisibleCountBuffers_.clear();
    cullIndirectBuffers_.clear();
    pageFrustumBuffers_.clear();
    pageFrustumPlanes_.clear();
    cullPipeline_.reset();
    cullSetLayout_.reset();
}

void VirtualShadowMapPass::destroyMarkResources()
{
    available_ = false;
    lastMarkThreadCount_ = 0;
    markWindowOrigins_.clear();
    readbackReady_.clear();
    descriptorSets_.clear();
    descriptorPool_.reset();
    paramsBuffers_.clear();
    requestReadbackBuffers_.clear();
    requestBuffers_.clear();
    pipeline_.reset();
    descriptorSetLayout_.reset();
}

void VirtualShadowMapPass::destroyResources()
{
    destroyCullResources();
    destroyMarkResources();
    pagePoolNeedsFullClear_ = true;
    allocator_.reset();
    residencyValid_ = false;
    dirtyPages_.clear();
    requestedPageIds_.clear();
    pagePool_.reset();
    pageTableBuffers_.clear();
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
    markWindowOrigins_.assign(frameCount, {});

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

void VirtualShadowMapPass::createPagePool()
{
    // One depth image, exactly like the punctual shadow atlas: a single
    // descriptor and a single layout to track, whatever the page count. The pool
    // is fixed-size and independent of the swapchain -- pages are a light-space
    // quantity, so nothing about it follows the window extent.
    pagePool_.create(context_,
                     kVsmPagePoolSize,
                     kVsmPagePoolSize,
                     /*layerCount=*/1,
                     rhi::VulkanShadowMap::ViewKind::Single,
                     "VsmPagePool");
    pagePoolNeedsFullClear_ = true;
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

    // The window this slot's request set is about to be marked against. A slot
    // index alone is ambiguous once the window scrolls -- the same slot names a
    // different absolute page -- so the origin travels with the request set to
    // the frame that reads it back.
    if (frameIndex < markWindowOrigins_.size()) {
        auto& origins = markWindowOrigins_[frameIndex];
        origins.fill(glm::ivec2{0, 0});
        for (uint32_t level = 0; level < clipmap.levelCount; ++level) {
            origins[level] = vsmWindowOrigin(clipmap, level, cameraLightSpaceXy);
        }
    }

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

bool VirtualShadowMapPass::readRequestWords(uint32_t frameIndex, std::array<uint32_t, kVsmPageRequestWordCount>& words)
{
    words.fill(0u);
    if (!available_ || frameIndex >= requestReadbackBuffers_.size() || frameIndex >= readbackReady_.size() ||
        readbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = requestReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(words.data(), words.size())));
    return true;
}

bool VirtualShadowMapPass::readRequestStats(uint32_t frameIndex, uint32_t levelCount, VsmPageRequestStats& stats)
{
    stats = {};
    std::array<uint32_t, kVsmPageRequestWordCount> words{};
    if (!readRequestWords(frameIndex, words)) {
        return false;
    }

    stats = vsmDecodeRequestStats(words.data(), levelCount);
    return true;
}

VkDeviceAddress VirtualShadowMapPass::pageTableAddress(uint32_t frameIndex) const
{
    if (frameIndex >= pageTableBuffers_.size() || !pageTableBuffers_[frameIndex].valid()) {
        return 0;
    }
    return pageTableBuffers_[frameIndex].deviceAddress();
}

void VirtualShadowMapPass::invalidateResidency()
{
    allocator_.reset();
    dirtyPages_.clear();
    residencyValid_ = false;
    // Nothing in the pool is referenced any more, so its contents are
    // meaningless; the next page pass clears it outright rather than trusting a
    // partial clear over depth no entry points at.
    pagePoolNeedsFullClear_ = true;
}

VsmResidencyStats VirtualShadowMapPass::updateResidency(uint32_t frameIndex,
                                                        const VsmClipmapSettings& clipmap,
                                                        const glm::vec2& cameraLightSpaceXy,
                                                        const glm::vec3& lightDirection,
                                                        uint64_t frameCounter)
{
    VsmResidencyStats stats{};
    dirtyPages_.clear();
    if (!available_) {
        return stats;
    }

    const VsmClipmapSettings clamped = clampVsmClipmapSettings(clipmap);

    // Everything the pool's contents depend on but a page's identity does not.
    // Exact equality on purpose: a light that moved by a hair still moved every
    // shadow in the pool, and "close enough" here is a stale-shadow bug that
    // shows up far from its cause.
    const bool inputsMoved = !residencyValid_ || residencyLightDirection_ != lightDirection ||
                             residencyClipmap_.levelCount != clamped.levelCount ||
                             residencyClipmap_.level0Extent != clamped.level0Extent ||
                             residencyClipmap_.depthRange != clamped.depthRange;
    if (inputsMoved) {
        invalidateResidency();
        residencyLightDirection_ = lightDirection;
        residencyClipmap_ = clamped;
        residencyValid_ = true;
    }

    std::array<uint32_t, kVsmPageRequestWordCount> words{};
    if (!readRequestWords(frameIndex, words)) {
        // No request set yet. Residency is left exactly as it was rather than
        // cleared: the pages already in the pool are still valid for their
        // absolute coordinates, and dropping them would redraw everything on the
        // first frame that happens to have no readback.
        stats.residentPages = allocator_.residentPages();
        return stats;
    }

    const std::array<glm::ivec2, kVsmMaxClipmapLevels>& markOrigins =
        frameIndex < markWindowOrigins_.size() ? markWindowOrigins_[frameIndex] : kZeroOrigins;

    allocator_.beginFrame(frameCounter);

    const uint32_t activePages = std::min(clamped.levelCount, kVsmMaxClipmapLevels) * kVsmPagesPerLevel;
    for (uint32_t pageId = 0; pageId < activePages; ++pageId) {
        if ((words[vsmRequestWordIndex(pageId)] & vsmRequestBitMask(pageId)) == 0u) {
            continue;
        }
        ++stats.requestedPages;

        const uint32_t level = vsmPageLevel(pageId);
        const uint32_t slot = vsmPageSlot(pageId);

        // Recover the page this slot meant when it was marked, then check it is
        // still reachable now. A page that scrolled out in the meantime is not
        // worth a physical page or a draw -- nothing can sample it this frame.
        const glm::ivec2 absolutePage = vsmAbsolutePageForSlot(markOrigins[level], slot);
        const glm::ivec2 currentOrigin = vsmWindowOrigin(clamped, level, cameraLightSpaceXy);
        if (!vsmPageInWindow(absolutePage, currentOrigin)) {
            continue;
        }
        ++stats.addressablePages;

        const VsmPageAcquireResult acquired = allocator_.acquire(pageId, absolutePage);
        if (acquired.evicted) {
            ++stats.evictions;
        }
        if (acquired.physicalPage == kVsmInvalidPhysicalPage) {
            ++stats.refusedPages;
            continue;
        }
        if (acquired.alreadyRendered) {
            ++stats.cachedPages;
            continue;
        }

        // Allocated but with no depth for these coordinates. Queue it, unless
        // the frame's draw budget is spent -- in which case it stays allocated
        // and unrendered, and the next frame picks it up.
        if (dirtyPages_.size() >= kMaxVsmPagesPerFrame) {
            ++stats.overBudgetPages;
            continue;
        }
        dirtyPages_.push_back(VsmDirtyPage{pageId, acquired.physicalPage, level, absolutePage});
    }

    stats.residentPages = allocator_.residentPages();
    stats.dirtyPages = static_cast<uint32_t>(dirtyPages_.size());
    // Consumed here: the count belongs to the update that acts on it, and the
    // caller invalidates before calling this.
    stats.casterInvalidatedPages = pagesInvalidatedByCasters_;
    pagesInvalidatedByCasters_ = 0;

    uploadPageTable(frameIndex);
    return stats;
}

void VirtualShadowMapPass::uploadPageTable(uint32_t frameIndex)
{
    if (frameIndex >= pageTableBuffers_.size() || !pageTableBuffers_[frameIndex].valid()) {
        return;
    }

    const std::vector<VsmPageTableEntry>& entries = allocator_.entries();
    pageTableBuffers_[frameIndex].upload(
        std::as_bytes(std::span<const VsmPageTableEntry>(entries.data(), entries.size())));
}

uint32_t VirtualShadowMapPass::invalidatePagesForBounds(const VsmClipmapSettings& clipmap,
                                                        const glm::mat4& lightView,
                                                        const glm::vec2& cameraLightSpaceXy,
                                                        const glm::vec3& boundsMin,
                                                        const glm::vec3& boundsMax)
{
    if (!available_) {
        return 0;
    }

    const VsmClipmapSettings clamped = clampVsmClipmapSettings(clipmap);
    uint32_t invalidated = 0;

    // Every level, not just the one the object's distance would select: the
    // marking pass requests one level coarser as a fallback, so an object's
    // shadow can be resident at more than one level at once and dropping only
    // the finest would leave a stale coarse copy for the walk to find.
    for (uint32_t level = 0; level < clamped.levelCount; ++level) {
        scratchPageIds_.clear();
        const glm::ivec2 windowOrigin = vsmWindowOrigin(clamped, level, cameraLightSpaceXy);
        vsmPagesOverlappingBounds(
            clamped, lightView, level, windowOrigin, boundsMin, boundsMax, scratchPageIds_);
        for (const uint32_t pageId : scratchPageIds_) {
            if (allocator_.invalidate(pageId)) {
                ++invalidated;
            }
        }
    }

    pagesInvalidatedByCasters_ += invalidated;
    return invalidated;
}

void VirtualShadowMapPass::markDirtyPagesRendered(uint32_t frameIndex)
{
    if (dirtyPages_.empty()) {
        return;
    }
    for (const VsmDirtyPage& page : dirtyPages_) {
        allocator_.markRendered(page.pageId);
    }
    // The table has to go back up: `rendered` is what the sampling path reads to
    // decide whether a page can be trusted, and it only became true just now.
    uploadPageTable(frameIndex);
}

void VirtualShadowMapPass::createCullResources(uint32_t frameCount, uint32_t maxDrawItems)
{
    destroyCullResources();
    if (!pagePool_.valid() || frameCount == 0 || maxDrawItems == 0) {
        return;
    }

    try {
        cullDrawItemCapacity_ = maxDrawItems;

        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
            bindings[binding].binding = binding;
            bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[binding].descriptorCount = 1;
            bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        cullSetLayout_.create(context_.vkDevice(),
                              std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));

        const VkDescriptorSetLayout setLayout = cullSetLayout_.handle();
        const VkPushConstantRange pushRange{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(VsmPageCullPushConstants))};
        rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.shaderPath = shaderPath("vsm_page_cull.comp.spv");
        pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&setLayout, 1);
        pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushRange, 1);
        pipelineInfo.pipelineCache = context_.pipelineCache();
        cullPipeline_.create(context_.vkDevice(), pipelineInfo);
        rhi::debug::setObjectName(
            context_.vkDevice(), cullPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "VsmPageCullComputePipeline");

        std::array<VkDescriptorPoolSize, 1> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = frameCount * static_cast<uint32_t>(bindings.size());
        cullDescriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), frameCount);

        pageFrustumBuffers_.resize(frameCount);
        cullIndirectBuffers_.resize(frameCount);
        cullVisibleCountBuffers_.resize(frameCount);
        cullOverflowReadbackBuffers_.resize(frameCount);
        casterFlagBuffers_.resize(frameCount);
        cullReadbackReady_.assign(frameCount, 0);
        pageFrustumPlanes_.assign(static_cast<size_t>(kMaxVsmPagesPerFrame) * 6u, glm::vec4{0.0f});

        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo frustumInfo{};
            frustumInfo.size = static_cast<VkDeviceSize>(kMaxVsmPagesPerFrame) * 6 * sizeof(glm::vec4);
            frustumInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            frustumInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            frustumInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            pageFrustumBuffers_[frameIndex].createBuffer(context_, frustumInfo);

            // Capped per page rather than sized by the draw-item count. At
            // kMaxDrawItems this would be 128 x 8192 x 20 bytes = 21 MiB per
            // frame in flight; a page covers a small world rect, so the cap is
            // generous, and going over is counted rather than hidden.
            rhi::VulkanBufferCreateInfo indirectInfo{};
            indirectInfo.size = static_cast<VkDeviceSize>(kMaxVsmPagesPerFrame) * kVsmCasterBucketCount *
                                kMaxVsmCastersPerPage * sizeof(VkDrawIndexedIndirectCommand);
            indirectInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            indirectInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            cullIndirectBuffers_[frameIndex].createBuffer(context_, indirectInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      cullIndirectBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "VsmPageCullIndirectBuffer" + std::to_string(frameIndex));

            rhi::VulkanBufferCreateInfo countInfo{};
            countInfo.size = kCullCounterBufferSize;
            countInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            countInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            cullVisibleCountBuffers_[frameIndex].createBuffer(context_, countInfo);

            rhi::VulkanBufferCreateInfo overflowInfo{};
            overflowInfo.size = kCullCounterBufferSize;
            overflowInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            overflowInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            overflowInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            cullOverflowReadbackBuffers_[frameIndex].createBuffer(context_, overflowInfo);

            rhi::VulkanBufferCreateInfo flagInfo{};
            flagInfo.size = static_cast<VkDeviceSize>(maxDrawItems) * sizeof(uint32_t);
            flagInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            flagInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            flagInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            casterFlagBuffers_[frameIndex].createBuffer(context_, flagInfo);
        }

        std::vector<VkDescriptorSetLayout> layouts(frameCount, cullSetLayout_.handle());
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = cullDescriptorPool_.handle();
        allocateInfo.descriptorSetCount = frameCount;
        allocateInfo.pSetLayouts = layouts.data();
        cullSets_.resize(frameCount, VK_NULL_HANDLE);
        if (vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, cullSets_.data()) != VK_SUCCESS) {
            cullSets_.clear();
            throw std::runtime_error("Failed to allocate VSM page-cull descriptor sets.");
        }

        cullAvailable_ = cullPipeline_.pipeline() != VK_NULL_HANDLE;
    } catch (const std::exception& error) {
        Logger::warn(std::string("Virtual shadow map page culling failed to initialise: ") + error.what());
        destroyCullResources();
        return;
    }

    Logger::info("Virtual shadow map page culling is available. Up to " + std::to_string(kMaxVsmPagesPerFrame) +
                 " pages per frame, " + std::to_string(kMaxVsmCastersPerPage) + " casters per page.");
}

VkBuffer VirtualShadowMapPass::cullIndirectBuffer(uint32_t frameIndex) const
{
    if (frameIndex >= cullIndirectBuffers_.size()) {
        return VK_NULL_HANDLE;
    }
    return cullIndirectBuffers_[frameIndex].buffer();
}

void VirtualShadowMapPass::recordPageCull(VkCommandBuffer commandBuffer,
                                          uint32_t frameIndex,
                                          VkBuffer cullInputBuffer,
                                          VkDeviceSize cullInputSize,
                                          uint32_t drawItemCount,
                                          const VsmClipmapSettings& clipmap,
                                          const glm::mat4& lightView,
                                          std::span<const uint32_t> casterFlags)
{
    if (!cullAvailable_ || frameIndex >= cullSets_.size() || dirtyPages_.empty() || drawItemCount == 0 ||
        cullInputBuffer == VK_NULL_HANDLE) {
        return;
    }

    const uint32_t pageCount = static_cast<uint32_t>(std::min<size_t>(dirtyPages_.size(), kMaxVsmPagesPerFrame));
    const uint32_t items = std::min(drawItemCount, cullDrawItemCapacity_);

    const GpuProfileScope profileScope(gpuProfiler_, frameIndex, commandBuffer, "VsmPageCull");
    rhi::debug::beginLabel(commandBuffer, "VsmPageCull");

    // Six planes per dirty page, from that page's own orthographic projection --
    // the same extraction the punctual atlas uses for its per-slot frusta.
    const VsmClipmapSettings clamped = clampVsmClipmapSettings(clipmap);
    std::fill(pageFrustumPlanes_.begin(), pageFrustumPlanes_.end(), glm::vec4{0.0f});
    for (uint32_t pageIndex = 0; pageIndex < pageCount; ++pageIndex) {
        const VsmDirtyPage& page = dirtyPages_[pageIndex];
        const glm::mat4 pageViewProjection = vsmPageViewProjection(clamped, lightView, page.level, page.absolutePage);
        const Frustum frustum = Frustum::fromViewProjection(pageViewProjection);
        for (size_t planeIndex = 0; planeIndex < frustum.planes.size(); ++planeIndex) {
            const FrustumPlane& plane = frustum.planes[planeIndex];
            pageFrustumPlanes_[static_cast<size_t>(pageIndex) * 6u + planeIndex] =
                glm::vec4(plane.normal, plane.distance);
        }
    }
    pageFrustumBuffers_[frameIndex].upload(
        std::as_bytes(std::span<const glm::vec4>(pageFrustumPlanes_.data(), pageFrustumPlanes_.size())));

    if (!casterFlags.empty()) {
        casterFlagBuffers_[frameIndex].upload(
            std::as_bytes(casterFlags.subspan(0, std::min<size_t>(casterFlags.size(), cullDrawItemCapacity_))));
    }

    std::array<VkDescriptorBufferInfo, 5> bufferInfos{};
    bufferInfos[0] = {cullInputBuffer, 0, cullInputSize};
    bufferInfos[1] = {cullIndirectBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
    bufferInfos[2] = {cullVisibleCountBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
    bufferInfos[3] = {pageFrustumBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
    bufferInfos[4] = {casterFlagBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};

    std::array<VkWriteDescriptorSet, 5> writes{};
    for (uint32_t binding = 0; binding < writes.size(); ++binding) {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = cullSets_[frameIndex];
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &bufferInfos[binding];
    }
    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Only the command regions this frame's pages will draw from are cleared.
    // The counters are zeroed in full, because the over-cap counter lives past
    // the per-page ones.
    const VkDeviceSize clearedCommandBytes = static_cast<VkDeviceSize>(pageCount) * kVsmCasterBucketCount *
                                             kMaxVsmCastersPerPage * sizeof(VkDrawIndexedIndirectCommand);
    vkCmdFillBuffer(commandBuffer, cullVisibleCountBuffers_[frameIndex].buffer(), 0, kCullCounterBufferSize, 0);
    vkCmdFillBuffer(commandBuffer, cullIndirectBuffers_[frameIndex].buffer(), 0, clearedCommandBytes, 0);

    std::array<VkBufferMemoryBarrier2, 2> resetBarriers{};
    for (VkBufferMemoryBarrier2& barrier : resetBarriers) {
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset = 0;
    }
    resetBarriers[0].buffer = cullVisibleCountBuffers_[frameIndex].buffer();
    resetBarriers[0].size = kCullCounterBufferSize;
    resetBarriers[1].buffer = cullIndirectBuffers_[frameIndex].buffer();
    resetBarriers[1].size = clearedCommandBytes;

    VkDependencyInfo resetDependency{};
    resetDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    resetDependency.bufferMemoryBarrierCount = static_cast<uint32_t>(resetBarriers.size());
    resetDependency.pBufferMemoryBarriers = resetBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &resetDependency);

    VsmPageCullPushConstants push{};
    push.pageCount = pageCount;
    push.drawItemCount = items;
    push.pageCommandStride = kMaxVsmCastersPerPage;
    push.overflowCounterIndex = kCullOverflowCounterIndex;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            cullPipeline_.layout(),
                            0,
                            1,
                            &cullSets_[frameIndex],
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       cullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(push)),
                       &push);

    const uint32_t threadCount = pageCount * items;
    vkCmdDispatch(commandBuffer, (threadCount + kCullWorkgroupSize - 1u) / kCullWorkgroupSize, 1, 1);

    // The page pass reads these as indirect commands and the copy below reads
    // the counters. Neither buffer is a graph resource, so both edges are
    // explicit.
    std::array<VkBufferMemoryBarrier2, 2> toConsumers{};
    for (VkBufferMemoryBarrier2& barrier : toConsumers) {
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset = 0;
    }
    toConsumers[0].buffer = cullIndirectBuffers_[frameIndex].buffer();
    toConsumers[0].size = clearedCommandBytes;
    toConsumers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    toConsumers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    toConsumers[1].buffer = cullVisibleCountBuffers_[frameIndex].buffer();
    toConsumers[1].size = kCullCounterBufferSize;
    toConsumers[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toConsumers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo consumerDependency{};
    consumerDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    consumerDependency.bufferMemoryBarrierCount = static_cast<uint32_t>(toConsumers.size());
    consumerDependency.pBufferMemoryBarriers = toConsumers.data();
    vkCmdPipelineBarrier2(commandBuffer, &consumerDependency);

    VkBufferCopy copyRegion{};
    copyRegion.size = kCullCounterBufferSize;
    vkCmdCopyBuffer(commandBuffer,
                    cullVisibleCountBuffers_[frameIndex].buffer(),
                    cullOverflowReadbackBuffers_[frameIndex].buffer(),
                    1,
                    &copyRegion);

    bufferBarrier(commandBuffer,
                  cullOverflowReadbackBuffers_[frameIndex].buffer(),
                  kCullCounterBufferSize,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_HOST_BIT,
                  VK_ACCESS_2_HOST_READ_BIT);

    if (frameIndex < cullReadbackReady_.size()) {
        cullReadbackReady_[frameIndex] = 1;
    }

    rhi::debug::endLabel(commandBuffer);
}

bool VirtualShadowMapPass::readPageCullOverflow(uint32_t frameIndex, uint32_t& overflowCount)
{
    overflowCount = 0;
    if (!cullAvailable_ || frameIndex >= cullOverflowReadbackBuffers_.size() ||
        frameIndex >= cullReadbackReady_.size() || cullReadbackReady_[frameIndex] == 0) {
        return false;
    }

    std::array<uint32_t, kCullCounterCount> counters{};
    cullOverflowReadbackBuffers_[frameIndex].download(
        std::as_writable_bytes(std::span<uint32_t>(counters.data(), counters.size())));
    overflowCount = counters[kCullOverflowCounterIndex];
    return true;
}

} // namespace ve::renderer
