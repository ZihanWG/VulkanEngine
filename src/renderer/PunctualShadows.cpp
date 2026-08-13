#include "renderer/PunctualShadows.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <array>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

// Identity slot handed back for out-of-range queries so a bookkeeping slip
// degrades into "no shadow" instead of reading past the end of the vectors.
const glm::mat4 kIdentityViewProjection{1.0f};
const Frustum kEmptyFrustum{};

} // namespace

void PunctualShadows::create(rhi::VulkanContext& context, uint32_t frameCount)
{
    reset();
    context_ = &context;

    try {
        atlas_.create(context,
                      kPunctualShadowAtlasSize,
                      kPunctualShadowAtlasSize,
                      /*layerCount=*/1,
                      rhi::VulkanShadowMap::ViewKind::Single,
                      "PunctualShadowAtlas");

        slotBuffers_.resize(frameCount);
        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = static_cast<VkDeviceSize>(kMaxPunctualShadowSlots) * sizeof(GpuShadowSlot);
            bufferInfo.usage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            bufferInfo.requestDeviceAddress = true;
            slotBuffers_[frameIndex].createBuffer(context, bufferInfo);
            rhi::debug::setObjectName(context.vkDevice(),
                                      slotBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "PunctualShadowSlotBuffer" + std::to_string(frameIndex));
        }

        slots_.reserve(kMaxPunctualShadowSlots);
        slotFrustums_.reserve(kMaxPunctualShadowSlots);

        Logger::info("Punctual shadow atlas enabled: " + std::to_string(kPunctualShadowAtlasSize) + "x" +
                     std::to_string(kPunctualShadowAtlasSize) + ", tiles " +
                     std::to_string(kPunctualShadowMinTileSize) + "-" +
                     std::to_string(kPunctualShadowMaxTileSize) + "px.");
    } catch (const std::exception& error) {
        // Punctual lights still shade, they just do not cast. Matching the
        // clustered path's policy: a missing optional subsystem degrades the
        // frame instead of failing device creation.
        reset();
        Logger::warn(std::string("Punctual shadow atlas unavailable; point/spot lights will not cast shadows: ") +
                     error.what());
    }
}

void PunctualShadows::reset()
{
    slotBuffers_.clear();
    atlas_.reset();
    slots_.clear();
    slotFrustums_.clear();
    slotRects_.clear();
    allocator_.reset();

    cullPipeline_.reset();
    cullSets_.clear();
    cullDescriptorPool_.reset();
    cullSetLayout_.reset();
    slotFrustumBuffers_.clear();
    cullIndirectBuffers_.clear();
    cullVisibleCountBuffers_.clear();
    casterFlagBuffers_.clear();
    cullAvailable_ = false;
    cullSlotCommandStride_ = 0;

    context_ = nullptr;
}

void PunctualShadows::beginFrame()
{
    allocator_.reset();
    slots_.clear();
    slotFrustums_.clear();
    slotRects_.clear();
}

uint32_t PunctualShadows::addSpotLight(const glm::vec3& position,
                                       const glm::vec3& direction,
                                       float outerAngleRadians,
                                       float range,
                                       uint32_t sizeClass)
{
    if (!valid()) {
        return kInvalidPunctualShadowSlot;
    }

    ShadowAtlasRect rect{};
    if (!allocator_.allocate(sizeClass, rect)) {
        return kInvalidPunctualShadowSlot;
    }

    // Slot indices stay sequential even though rects no longer do, which is what
    // keeps the float encoding and the consecutive cube-face property intact.
    const auto slot = static_cast<uint32_t>(slots_.size());
    const glm::mat4 viewProjection =
        computeSpotShadowViewProjection(position, direction, outerAngleRadians, range);

    GpuShadowSlot record{};
    record.viewProjection = viewProjection;
    record.atlasUvOffsetScale = shadowAtlasRectUvOffsetScale(rect);
    record.params = glm::vec4(constantBias_,
                              normalBias_,
                              // One atlas texel in UV: the PCF tap step. Derived
                              // from the atlas (not the tile) because the shader
                              // steps in atlas UV after the tile remap.
                              1.0f / static_cast<float>(kPunctualShadowAtlasSize),
                              0.0f);

    slots_.push_back(record);
    slotFrustums_.push_back(computeSpotShadowFrustum(viewProjection));
    slotRects_.push_back(rect);

    return slot;
}

uint32_t PunctualShadows::addPointLight(const glm::vec3& position, float range, uint32_t sizeClass)
{
    if (!valid()) {
        return kInvalidPunctualShadowSlot;
    }

    // All six or none: a light missing two faces would sample cleared tiles
    // there, which reads as light leaking through solid geometry.
    if (!allocator_.allocateRange(sizeClass, kPointShadowFaceCount, pendingFaceRects_)) {
        return kInvalidPunctualShadowSlot;
    }

    const auto baseSlot = static_cast<uint32_t>(slots_.size());
    for (uint32_t face = 0; face < kPointShadowFaceCount; ++face) {
        const glm::mat4 viewProjection = computePointShadowFaceViewProjection(position, face, range);

        GpuShadowSlot record{};
        record.viewProjection = viewProjection;
        record.atlasUvOffsetScale = shadowAtlasRectUvOffsetScale(pendingFaceRects_[face]);
        record.params = glm::vec4(
            constantBias_, normalBias_, 1.0f / static_cast<float>(kPunctualShadowAtlasSize), 0.0f);

        slots_.push_back(record);
        slotFrustums_.push_back(computeSpotShadowFrustum(viewProjection));
        slotRects_.push_back(pendingFaceRects_[face]);
    }

    return baseSlot;
}

void PunctualShadows::createCullResources(uint32_t frameCount,
                                          const std::filesystem::path& cullShaderPath,
                                          uint32_t maxDrawItems)
{
    cullAvailable_ = false;
    if (context_ == nullptr || !valid()) {
        return;
    }

    try {
        cullSlotCommandStride_ = std::max(maxDrawItems, 1u);
        const VkDevice device = context_->vkDevice();

        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
            bindings[binding].binding = binding;
            bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[binding].descriptorCount = 1;
            bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        cullSetLayout_.create(device, std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));

        std::array<VkDescriptorPoolSize, 1> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = frameCount * static_cast<uint32_t>(bindings.size());
        cullDescriptorPool_.create(
            device, std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), frameCount);

        slotFrustumBuffers_.resize(frameCount);
        cullIndirectBuffers_.resize(frameCount);
        cullVisibleCountBuffers_.resize(frameCount);
        casterFlagBuffers_.resize(frameCount);
        cullSets_.resize(frameCount);

        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo frustumInfo{};
            frustumInfo.size = static_cast<VkDeviceSize>(kMaxGpuCulledSlots) * 6 * sizeof(glm::vec4);
            frustumInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            frustumInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            frustumInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            slotFrustumBuffers_[frameIndex].createBuffer(*context_, frustumInfo);

            rhi::VulkanBufferCreateInfo indirectInfo{};
            indirectInfo.size = static_cast<VkDeviceSize>(kMaxGpuCulledSlots) * cullSlotCommandStride_ *
                                sizeof(VkDrawIndexedIndirectCommand);
            indirectInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            indirectInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            cullIndirectBuffers_[frameIndex].createBuffer(*context_, indirectInfo);

            rhi::VulkanBufferCreateInfo flagInfo{};
            flagInfo.size = static_cast<VkDeviceSize>(cullSlotCommandStride_) * sizeof(uint32_t);
            flagInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            flagInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            flagInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            casterFlagBuffers_[frameIndex].createBuffer(*context_, flagInfo);

            rhi::VulkanBufferCreateInfo countInfo{};
            countInfo.size = static_cast<VkDeviceSize>(kMaxGpuCulledSlots) * sizeof(uint32_t);
            countInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            countInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            cullVisibleCountBuffers_[frameIndex].createBuffer(*context_, countInfo);

            const VkDescriptorSetLayout layout = cullSetLayout_.handle();
            VkDescriptorSetAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocateInfo.descriptorPool = cullDescriptorPool_.handle();
            allocateInfo.descriptorSetCount = 1;
            allocateInfo.pSetLayouts = &layout;
            VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, &cullSets_[frameIndex]));
        }

        const VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4};
        const VkDescriptorSetLayout cullLayout = cullSetLayout_.handle();
        rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.shaderPath = cullShaderPath;
        pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&cullLayout, 1);
        pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushRange, 1);
        pipelineInfo.pipelineCache = context_->pipelineCache();
        cullPipeline_.create(context_->vkDevice(), pipelineInfo);

        cullAvailable_ = true;
        Logger::info("Punctual shadow GPU caster culling enabled for up to " +
                     std::to_string(kMaxGpuCulledSlots) + " slots.");
    } catch (const std::exception& error) {
        cullAvailable_ = false;
        Logger::warn(std::string("Punctual shadow GPU caster culling unavailable; CPU culling stays in use: ") +
                     error.what());
    }
}

void PunctualShadows::uploadSlotFrustums(uint32_t frameIndex)
{
    if (!cullAvailable_ || frameIndex >= slotFrustumBuffers_.size()) {
        return;
    }

    const uint32_t count = std::min<uint32_t>(slotCount(), kMaxGpuCulledSlots);
    slotFrustumPlanes_.assign(static_cast<size_t>(kMaxGpuCulledSlots) * 6, glm::vec4{0.0f});
    for (uint32_t slot = 0; slot < count; ++slot) {
        const Frustum& frustum = slotFrustum(slot);
        for (size_t planeIndex = 0; planeIndex < frustum.planes.size(); ++planeIndex) {
            slotFrustumPlanes_[static_cast<size_t>(slot) * 6 + planeIndex] =
                glm::vec4(frustum.planes[planeIndex].normal, frustum.planes[planeIndex].distance);
        }
    }

    slotFrustumBuffers_[frameIndex].upload(
        std::as_bytes(std::span<const glm::vec4>(slotFrustumPlanes_.data(), slotFrustumPlanes_.size())));
}

void PunctualShadows::recordCull(VkCommandBuffer commandBuffer,
                                 uint32_t frameIndex,
                                 VkBuffer cullInputBuffer,
                                 VkDeviceSize cullInputSize,
                                 uint32_t drawItemCount,
                                 std::span<const uint32_t> casterFlags)
{
    const uint32_t culledSlots = std::min<uint32_t>(slotCount(), kMaxGpuCulledSlots);
    if (!cullAvailable_ || frameIndex >= cullSets_.size() || culledSlots == 0 || drawItemCount == 0 ||
        cullInputBuffer == VK_NULL_HANDLE) {
        return;
    }

    // Descriptors are rewritten each frame because the cull input buffer belongs
    // to GpuCulling and is chosen per frame there.
    // Upload the caster mask before the descriptors reference it.
    if (!casterFlags.empty()) {
        casterFlagBuffers_[frameIndex].upload(std::as_bytes(
            casterFlags.subspan(0, std::min<size_t>(casterFlags.size(), cullSlotCommandStride_))));
    }

    std::array<VkDescriptorBufferInfo, 5> bufferInfos{};
    bufferInfos[0] = {cullInputBuffer, 0, cullInputSize};
    bufferInfos[1] = {cullIndirectBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
    bufferInfos[2] = {cullVisibleCountBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
    bufferInfos[3] = {slotFrustumBuffers_[frameIndex].buffer(), 0, VK_WHOLE_SIZE};
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
    vkUpdateDescriptorSets(context_->vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Zero the counts, and zero the command region so a slot that culls
    // everything draws nothing rather than replaying last frame's commands --
    // this platform has no indirect-count, so the draw side always submits the
    // full stride and relies on zeroed indexCount to be a no-op.
    const VkDeviceSize indirectBytes = static_cast<VkDeviceSize>(culledSlots) * cullSlotCommandStride_ *
                                       sizeof(VkDrawIndexedIndirectCommand);
    vkCmdFillBuffer(commandBuffer, cullVisibleCountBuffers_[frameIndex].buffer(), 0,
                    static_cast<VkDeviceSize>(culledSlots) * sizeof(uint32_t), 0);
    vkCmdFillBuffer(commandBuffer, cullIndirectBuffers_[frameIndex].buffer(), 0, indirectBytes, 0);

    std::array<VkBufferMemoryBarrier2, 2> resetBarriers{};
    for (size_t i = 0; i < resetBarriers.size(); ++i) {
        resetBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        resetBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        resetBarriers[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        resetBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        resetBarriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        resetBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resetBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        resetBarriers[i].offset = 0;
    }
    resetBarriers[0].buffer = cullVisibleCountBuffers_[frameIndex].buffer();
    resetBarriers[0].size = VK_WHOLE_SIZE;
    resetBarriers[1].buffer = cullIndirectBuffers_[frameIndex].buffer();
    resetBarriers[1].size = VK_WHOLE_SIZE;

    VkDependencyInfo resetDependency{};
    resetDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    resetDependency.bufferMemoryBarrierCount = static_cast<uint32_t>(resetBarriers.size());
    resetDependency.pBufferMemoryBarriers = resetBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &resetDependency);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            cullPipeline_.layout(),
                            0,
                            1,
                            &cullSets_[frameIndex],
                            0,
                            nullptr);

    const std::array<uint32_t, 4> pushConstants{culledSlots, drawItemCount, cullSlotCommandStride_, 0};
    vkCmdPushConstants(commandBuffer,
                       cullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(pushConstants.size() * sizeof(uint32_t)),
                       pushConstants.data());

    // One thread per (slot, draw item).
    constexpr uint32_t kLocalSize = 64;
    const uint32_t threadCount = culledSlots * drawItemCount;
    vkCmdDispatch(commandBuffer, (threadCount + kLocalSize - 1) / kLocalSize, 1, 1);

    // The atlas pass consumes these as indirect draw commands.
    VkBufferMemoryBarrier2 toIndirect{};
    toIndirect.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    toIndirect.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    toIndirect.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    toIndirect.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    toIndirect.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    toIndirect.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toIndirect.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toIndirect.buffer = cullIndirectBuffers_[frameIndex].buffer();
    toIndirect.offset = 0;
    toIndirect.size = VK_WHOLE_SIZE;

    VkDependencyInfo indirectDependency{};
    indirectDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    indirectDependency.bufferMemoryBarrierCount = 1;
    indirectDependency.pBufferMemoryBarriers = &toIndirect;
    vkCmdPipelineBarrier2(commandBuffer, &indirectDependency);
}

VkBuffer PunctualShadows::cullIndirectBuffer(uint32_t frameIndex) const
{
    if (frameIndex >= cullIndirectBuffers_.size()) {
        return VK_NULL_HANDLE;
    }

    return cullIndirectBuffers_[frameIndex].buffer();
}

void PunctualShadows::upload(uint32_t frameIndex)
{
    if (frameIndex >= slotBuffers_.size() || slots_.empty()) {
        return;
    }

    const size_t count = std::min<size_t>(slots_.size(), kMaxPunctualShadowSlots);
    slotBuffers_[frameIndex].upload(std::as_bytes(std::span<const GpuShadowSlot>(slots_.data(), count)));
}

VkDeviceAddress PunctualShadows::slotBufferAddress(uint32_t frameIndex) const
{
    if (frameIndex >= slotBuffers_.size()) {
        return 0;
    }

    return slotBuffers_[frameIndex].deviceAddress();
}

const glm::mat4& PunctualShadows::slotViewProjection(uint32_t slot) const
{
    if (slot >= slots_.size()) {
        return kIdentityViewProjection;
    }

    return slots_[slot].viewProjection;
}

const Frustum& PunctualShadows::slotFrustum(uint32_t slot) const
{
    if (slot >= slotFrustums_.size()) {
        return kEmptyFrustum;
    }

    return slotFrustums_[slot];
}

ShadowAtlasRect PunctualShadows::slotRect(uint32_t slot) const
{
    if (slot >= slotRects_.size()) {
        return {};
    }

    return slotRects_[slot];
}

} // namespace ve::renderer
