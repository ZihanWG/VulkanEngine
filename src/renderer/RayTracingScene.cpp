#include "renderer/RayTracingScene.h"

#include "core/Logger.h"
#include "renderer/Mesh.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

// Scratch is only alive while a build runs, but the BLAS builds all run at
// startup, and a scene with hundreds of primitives would otherwise ask for the
// sum of every primitive's scratch at once. Builds are batched under this
// budget and the batches share the buffer, separated by a barrier.
constexpr VkDeviceSize kBottomLevelScratchBudget = 64ull * 1024ull * 1024ull;

// Rays are traced, never refit or updated in P1: a full rebuild of the TLAS
// every frame, and BLAS built once. Fast trace is the right half of the trade
// for both.
constexpr VkBuildAccelerationStructureFlagsKHR kBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

VkAccelerationStructureGeometryKHR instanceGeometry(VkDeviceAddress instances)
{
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instances;
    return geometry;
}

VkAccelerationStructureBuildGeometryInfoKHR buildInfo(VkAccelerationStructureTypeKHR type,
                                                      const VkAccelerationStructureGeometryKHR* geometry)
{
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = type;
    info.flags = kBuildFlags;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = geometry;
    return info;
}

constexpr VkAccessFlags2 kAccelerationStructureReadWrite =
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

// A global memory barrier, deliberately unscoped: acceleration-structure reads
// reach every BLAS a TLAS references, which no single buffer barrier names.
void accelerationStructureBarrier(VkCommandBuffer commandBuffer,
                                  VkPipelineStageFlags2 sourceStages,
                                  VkAccessFlags2 sourceAccess,
                                  VkPipelineStageFlags2 destinationStages,
                                  VkAccessFlags2 destinationAccess)
{
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = sourceStages;
    barrier.srcAccessMask = sourceAccess;
    barrier.dstStageMask = destinationStages;
    barrier.dstAccessMask = destinationAccess;

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace

size_t RayTracingScene::BottomLevelKeyHash::operator()(const BottomLevelKey& key) const
{
    size_t hash = std::hash<const void*>{}(key.mesh);
    hash ^= std::hash<uint64_t>{}((static_cast<uint64_t>(key.firstIndex) << 32) | key.indexCount) +
            0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    return hash;
}

RayTracingScene::~RayTracingScene()
{
    destroy();
}

void RayTracingScene::createTopLevel(rhi::VulkanContext& context, uint32_t framesInFlight, uint32_t maxInstances)
{
    destroy();
    if (!context.rayQueryEnabled()) {
        throw std::logic_error("RayTracingScene::createTopLevel needs a device with ray query.");
    }
    if (framesInFlight == 0 || maxInstances == 0) {
        throw std::invalid_argument("RayTracingScene::createTopLevel needs at least one frame and one instance.");
    }

    context_ = &context;
    try {
        // The geometry table is indexed by frameDataIndex, so it is sized by the
        // draw-item cap regardless of how many instances the device allows.
        const uint32_t geometryRecords = maxInstances;
        maxInstances_ = static_cast<uint32_t>(
            std::min<uint64_t>(maxInstances, context.device().maxAccelerationStructureInstanceCount()));

        const VkAccelerationStructureGeometryKHR geometry = instanceGeometry(0);
        VkAccelerationStructureBuildGeometryInfoKHR info =
            buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry);
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(
            context.vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &maxInstances_, &sizes);

        topLevel_.create(context,
                         VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                         sizes.accelerationStructureSize,
                         "SceneTopLevelAccelerationStructure");
        topLevelScratchSize_ = sizes.buildScratchSize;

        const VkDeviceSize alignment = context.device().accelerationStructureScratchAlignment();
        rhi::VulkanBufferCreateInfo scratchInfo{};
        scratchInfo.size = topLevelScratchSize_ + alignment;
        scratchInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        scratchInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        scratchInfo.requestDeviceAddress = true;
        scratch_.create(context, scratchInfo);
        rhi::debug::setObjectName(
            context.vkDevice(), scratch_.buffer(), VK_OBJECT_TYPE_BUFFER, "AccelerationStructureScratch");

        frames_.resize(framesInFlight);
        for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
            FrameBuffers& frame = frames_[frameIndex];

            rhi::VulkanBufferCreateInfo instanceInfo{};
            instanceInfo.size = static_cast<VkDeviceSize>(maxInstances_) * sizeof(VkAccelerationStructureInstanceKHR);
            instanceInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
            instanceInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            instanceInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            instanceInfo.requestDeviceAddress = true;
            frame.instances.create(context, instanceInfo);
            rhi::debug::setObjectName(context.vkDevice(),
                                      frame.instances.buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "RayTracingInstances" + std::to_string(frameIndex));

            rhi::VulkanBufferCreateInfo geometryInfo{};
            geometryInfo.size = static_cast<VkDeviceSize>(geometryRecords) * sizeof(RayTracingGeometryRecord);
            geometryInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            geometryInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            geometryInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            geometryInfo.requestDeviceAddress = true;
            frame.geometry.create(context, geometryInfo);
            rhi::debug::setObjectName(context.vkDevice(),
                                      frame.geometry.buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "RayTracingGeometryRecords" + std::to_string(frameIndex));
        }

        stats_ = {};
        stats_.topLevelBytes = topLevel_.size();
        stats_.scratchBytes = scratch_.size();
    } catch (...) {
        destroy();
        throw;
    }
}

void RayTracingScene::buildBottomLevel(const rhi::VulkanCommandContext& commandContext,
                                       std::span<const Mesh* const> meshes)
{
    if (!hasTopLevel() || context_ == nullptr) {
        throw std::logic_error("RayTracingScene::buildBottomLevel called before createTopLevel.");
    }
    const auto buildStart = std::chrono::steady_clock::now();
    rhi::VulkanContext& context = *context_;
    const VkDevice device = context.vkDevice();
    const VkDeviceSize alignment = std::max<VkDeviceSize>(context.device().accelerationStructureScratchAlignment(), 1);

    // Every (mesh, level-0 primitive range) a draw item can name. Draw items
    // carry exactly these ranges -- a primitive's own, or the whole of a mesh
    // without a primitive table -- so the lookup in writeInstances is exact.
    std::vector<BottomLevelKey> keys;
    for (const Mesh* mesh : meshes) {
        if (mesh == nullptr || !mesh->valid()) {
            continue;
        }
        if (mesh->vertexBufferAddress() == 0 || mesh->indexBufferAddress() == 0) {
            Logger::warn("Ray-traced shadows: '" + mesh->debugName() +
                         "' has no buffer device addresses and casts no ray-traced shadow.");
            continue;
        }
        if (mesh->hasSubMeshes()) {
            for (const MeshPrimitive& primitive : mesh->primitives()) {
                if (primitive.indexCount >= 3) {
                    keys.push_back({mesh, primitive.firstIndex, primitive.indexCount});
                }
            }
        } else if (mesh->indexCount() >= 3) {
            keys.push_back({mesh, 0, mesh->indexCount()});
        }
    }

    struct PendingBuild {
        VkAccelerationStructureGeometryKHR geometry{};
        VkAccelerationStructureBuildRangeInfoKHR range{};
        VkDeviceSize scratchSize = 0;
    };
    std::vector<PendingBuild> pending;
    pending.reserve(keys.size());
    bottomLevels_.clear();
    bottomLevels_.reserve(keys.size());
    bottomLevelIndex_.clear();
    stats_.bottomLevelBytes = 0;
    stats_.bottomLevelTriangles = 0;

    for (const BottomLevelKey& key : keys) {
        if (bottomLevelIndex_.contains(key)) {
            continue;
        }
        PendingBuild build{};
        build.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        build.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // No OPAQUE flag on the geometry: the same mesh is drawn with opaque and
        // cutout materials, so opacity is decided per instance instead
        // (FORCE_OPAQUE / FORCE_NO_OPAQUE in writeInstances).
        build.geometry.flags = 0;
        VkAccelerationStructureGeometryTrianglesDataKHR& triangles = build.geometry.geometry.triangles;
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = key.mesh->vertexBufferAddress();
        triangles.vertexStride = kRtVertexStrideFloats * sizeof(float);
        triangles.maxVertex = key.mesh->vertexCount() == 0 ? 0 : key.mesh->vertexCount() - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress =
            key.mesh->indexBufferAddress() + static_cast<VkDeviceAddress>(key.firstIndex) * sizeof(uint32_t);
        build.range.primitiveCount = key.indexCount / 3;

        const VkAccelerationStructureBuildGeometryInfoKHR info =
            buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &build.geometry);
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(
            device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &build.range.primitiveCount, &sizes);
        build.scratchSize = alignUpPowerOfTwo(sizes.buildScratchSize, alignment);

        rhi::VulkanAccelerationStructure& bottomLevel = bottomLevels_.emplace_back();
        bottomLevel.create(context,
                           VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                           sizes.accelerationStructureSize,
                           "BottomLevel " + key.mesh->debugName() + " @" + std::to_string(key.firstIndex));
        bottomLevelIndex_.emplace(key, static_cast<uint32_t>(bottomLevels_.size() - 1));
        stats_.bottomLevelBytes += bottomLevel.size();
        stats_.bottomLevelTriangles += build.range.primitiveCount;
        pending.push_back(build);
    }

    // Batches whose scratch fits the budget; one oversized build is a batch of
    // its own. The scratch buffer is sized for the largest batch and the TLAS.
    std::vector<std::pair<size_t, size_t>> batches;
    VkDeviceSize largestBatch = 0;
    for (size_t begin = 0; begin < pending.size();) {
        VkDeviceSize batchScratch = 0;
        size_t end = begin;
        while (end < pending.size() &&
               (end == begin || batchScratch + pending[end].scratchSize <= kBottomLevelScratchBudget)) {
            batchScratch += pending[end].scratchSize;
            ++end;
        }
        batches.emplace_back(begin, end);
        largestBatch = std::max(largestBatch, batchScratch);
        begin = end;
    }
    ensureScratch(largestBatch);
    const VkDeviceAddress scratchBase = this->scratchBase();

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext.commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    rhi::debug::beginLabel(commandBuffer, "BuildBottomLevelAccelerationStructures");

    // The mesh buffers were filled by copies in earlier submissions on this
    // queue; a barrier's first scope reaches back over them.
    {
        VkMemoryBarrier2 uploadBarrier{};
        uploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        uploadBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &uploadBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> ranges;
    for (const auto& [begin, end] : batches) {
        infos.clear();
        ranges.clear();
        VkDeviceSize scratchOffset = 0;
        for (size_t index = begin; index < end; ++index) {
            VkAccelerationStructureBuildGeometryInfoKHR info =
                buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &pending[index].geometry);
            info.dstAccelerationStructure = bottomLevels_[index].handle();
            info.scratchData.deviceAddress = scratchBase + scratchOffset;
            scratchOffset += pending[index].scratchSize;
            infos.push_back(info);
            ranges.push_back(&pending[index].range);
        }
        vkCmdBuildAccelerationStructuresKHR(
            commandBuffer, static_cast<uint32_t>(infos.size()), infos.data(), ranges.data());
        // The next batch reuses the scratch, and the TLAS build below reads
        // every BLAS this one wrote.
        accelerationStructureBarrier(commandBuffer,
                                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     kAccelerationStructureReadWrite);
    }

    // An empty TLAS, so the descriptor names a built structure from the first
    // frame. Frames that trace rebuild it with their own instances.
    {
        const VkAccelerationStructureGeometryKHR geometry = instanceGeometry(frames_.front().instances.deviceAddress());
        VkAccelerationStructureBuildGeometryInfoKHR info =
            buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry);
        info.dstAccelerationStructure = topLevel_.handle();
        info.scratchData.deviceAddress = scratchBase;
        const VkAccelerationStructureBuildRangeInfoKHR emptyRange{};
        const VkAccelerationStructureBuildRangeInfoKHR* emptyRangePointer = &emptyRange;
        vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &info, &emptyRangePointer);
    }

    rhi::debug::endLabel(commandBuffer);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;
    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    const VkResult submitted = vkQueueSubmit2(context.graphicsQueue(), 1, &submitInfo, fence);
    const VkResult waited = submitted == VK_SUCCESS
                                ? vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max())
                                : submitted;
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandContext.commandPool(), 1, &commandBuffer);
    VK_CHECK(waited);

    stats_.bottomLevelCount = static_cast<uint32_t>(bottomLevels_.size());
    stats_.scratchBytes = scratch_.size();
    stats_.bottomLevelBuildMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    ready_ = true;
}

void RayTracingScene::createSkinned(const RayTracingSkinnedSource& source,
                                    const std::filesystem::path& preSkinShaderPath)
{
    if (!ready_ || context_ == nullptr) {
        throw std::logic_error("RayTracingScene::createSkinned called before buildBottomLevel.");
    }
    skinned_.create(*context_, source, preSkinShaderPath);
    try {
        // The first frame builds and every later one refits, sharing the scratch
        // with the TLAS build that follows them in the same pass.
        ensureScratch(std::max(skinned_.buildScratchSize(), skinned_.updateScratchSize()));
    } catch (...) {
        skinned_.destroy();
        throw;
    }
    stats_.scratchBytes = scratch_.size();
}

void RayTracingScene::ensureScratch(VkDeviceSize bytes)
{
    const VkDeviceSize alignment =
        std::max<VkDeviceSize>(context_->device().accelerationStructureScratchAlignment(), 1);
    // The TLAS build always shares it, so it never shrinks below that.
    const VkDeviceSize needed =
        std::max(alignUpPowerOfTwo(bytes, alignment), alignUpPowerOfTwo(topLevelScratchSize_, alignment)) + alignment;
    if (scratch_.size() >= needed) {
        return;
    }
    rhi::VulkanBufferCreateInfo scratchInfo{};
    scratchInfo.size = needed;
    scratchInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    scratchInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    scratchInfo.requestDeviceAddress = true;
    scratch_.create(*context_, scratchInfo);
    rhi::debug::setObjectName(
        context_->vkDevice(), scratch_.buffer(), VK_OBJECT_TYPE_BUFFER, "AccelerationStructureScratch");
}

VkDeviceAddress RayTracingScene::scratchBase() const
{
    const VkDeviceSize alignment =
        std::max<VkDeviceSize>(context_->device().accelerationStructureScratchAlignment(), 1);
    return alignUpPowerOfTwo(scratch_.deviceAddress(), alignment);
}

void RayTracingScene::destroy()
{
    skinned_.destroy();
    ready_ = false;
    frames_.clear();
    scratch_.reset();
    topLevel_.reset();
    bottomLevelIndex_.clear();
    bottomLevels_.clear();
    topLevelScratchSize_ = 0;
    maxInstances_ = 0;
    stats_ = {};
    context_ = nullptr;
}

uint32_t RayTracingScene::writeInstances(uint32_t frameIndex, std::span<const RayTracingInstanceInput> inputs)
{
    if (!ready_ || frameIndex >= frames_.size()) {
        return 0;
    }
    FrameBuffers& frame = frames_[frameIndex];

    const std::span<std::byte> instanceBytes = frame.instances.mapRange(0, frame.instances.size());
    auto* instances = reinterpret_cast<VkAccelerationStructureInstanceKHR*>(instanceBytes.data());
    const std::span<std::byte> geometryBytes = frame.geometry.mapRange(0, frame.geometry.size());
    auto* geometry = reinterpret_cast<RayTracingGeometryRecord*>(geometryBytes.data());
    const size_t geometryCapacity = geometryBytes.size() / sizeof(RayTracingGeometryRecord);

    uint32_t count = 0;
    uint32_t alphaTested = 0;
    uint32_t missing = 0;
    frame.skinnedPalette = 0;
    for (const RayTracingInstanceInput& input : inputs) {
        if (input.kind == RayTracedCasterKind::None) {
            continue;
        }
        VkDeviceAddress bottomLevel = 0;
        RayTracingGeometryRecord record{};
        if (input.skinned) {
            if (skinned_.valid() && input.jointPalette != 0) {
                bottomLevel = skinned_.bottomLevelAddress();
                // Positions, not renderer::Vertex: nothing reads this record,
                // because the skinned caster is opaque and never runs the
                // alpha test that would.
                record = {skinned_.positionsAddress(), skinned_.indexAddress()};
            }
        } else {
            const auto found = bottomLevelIndex_.find({input.mesh, input.firstIndex, input.indexCount});
            if (found != bottomLevelIndex_.end()) {
                bottomLevel = bottomLevels_[found->second].deviceAddress();
                record = {input.mesh->vertexBufferAddress(),
                          input.mesh->indexBufferAddress() +
                              static_cast<uint64_t>(input.firstIndex) * sizeof(uint32_t)};
            }
        }
        if (bottomLevel == 0 || count >= maxInstances_ || input.frameDataIndex >= geometryCapacity) {
            ++missing;
            continue;
        }
        if (input.skinned) {
            frame.skinnedPalette = input.jointPalette;
        }

        // Built whole on the stack and copied once: the destination is
        // write-combined memory, where field-by-field stores are the slow way.
        VkAccelerationStructureInstanceKHR instance{};
        const std::array<float, 12> transform = rayTracingInstanceTransform(input.model);
        std::memcpy(&instance.transform, transform.data(), sizeof(instance.transform));
        instance.instanceCustomIndex = input.frameDataIndex;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        // Facing culling off: a shadow is cast by whichever side of a caster
        // faces the sun, which is what the cascades' shadow pipelines draw too.
        VkGeometryInstanceFlagsKHR flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        if (input.kind == RayTracedCasterKind::AlphaTested) {
            flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
            ++alphaTested;
        } else {
            flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        }
        instance.flags = flags;
        instance.accelerationStructureReference = bottomLevel;
        std::memcpy(&instances[count], &instance, sizeof(instance));
        std::memcpy(&geometry[input.frameDataIndex], &record, sizeof(record));
        ++count;
    }

    frame.instances.flush(0, static_cast<VkDeviceSize>(count) * sizeof(VkAccelerationStructureInstanceKHR));
    frame.instances.unmap();
    frame.geometry.flush(0, frame.geometry.size());
    frame.geometry.unmap();

    frame.instanceCount = count;
    stats_.instances = count;
    stats_.alphaTestedInstances = alphaTested;
    stats_.instancesWithoutBottomLevel = missing;
    stats_.skinnedInstance = frame.skinnedPalette != 0;
    return count;
}

void RayTracingScene::recordTopLevelBuild(VkCommandBuffer commandBuffer, uint32_t frameIndex)
{
    if (!ready_ || frameIndex >= frames_.size()) {
        return;
    }
    const FrameBuffers& frame = frames_[frameIndex];

    // What the previous frame left in flight that this one is about to rewrite:
    // its rays read the TLAS and the skinned BLAS, its builds wrote both and the
    // scratch, and its refit read the skinned positions the pre-skin below
    // overwrites. The graph cannot see any of it -- it rebuilds buffer state
    // every frame. The fragment-shader destination is for the BLASes built at
    // startup, whose writes no later barrier makes visible to the rays.
    accelerationStructureBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 kAccelerationStructureReadWrite,
                                 VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 kAccelerationStructureReadWrite | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    if (frame.skinnedPalette != 0 && skinned_.valid()) {
        skinned_.record(commandBuffer, frame.skinnedPalette, scratchBase());
        // The refit wrote a BLAS this frame's TLAS build reads and its rays
        // traverse, and the scratch the TLAS build reuses. The graph's barrier
        // after the build names only the TLAS buffer, so the rays' half of this
        // has to be here.
        accelerationStructureBarrier(commandBuffer,
                                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                     VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     kAccelerationStructureReadWrite);
        stats_.skinnedBuilds = skinned_.builds();
        stats_.skinnedRefits = skinned_.refits();
    }

    const VkAccelerationStructureGeometryKHR geometry = instanceGeometry(frame.instances.deviceAddress());
    VkAccelerationStructureBuildGeometryInfoKHR info =
        buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry);
    info.dstAccelerationStructure = topLevel_.handle();
    info.scratchData.deviceAddress = scratchBase();
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = frame.instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &info, &rangePointer);
}

VkDeviceAddress RayTracingScene::geometryTableAddress(uint32_t frameIndex) const
{
    return frameIndex < frames_.size() ? frames_[frameIndex].geometry.deviceAddress() : 0;
}

} // namespace ve::renderer
