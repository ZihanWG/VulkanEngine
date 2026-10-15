#include "renderer/RayTracingSkinning.h"

#include "renderer/RayTracedShadows.h"
#include "renderer/SkinnedMesh.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <cstddef>
#include <span>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace ve::renderer {

// skin_positions.comp reads the skinning stream as a uvec4 and a vec4 per
// vertex; a member added to SkinningVertex moves every weight it reads.
static_assert(sizeof(SkinningVertex) == 32);
static_assert(offsetof(SkinningVertex, weights) == 16);

namespace {

// Refitting is what makes a per-frame update affordable; the BVH it keeps was
// built for the first pose, so quality drifts with how far the mesh bends from
// it. The demo rig bends by ~22 degrees per joint, well inside what a refit
// handles.
constexpr VkBuildAccelerationStructureFlagsKHR kSkinnedBuildFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

VkAccelerationStructureGeometryKHR skinnedGeometry(const RayTracingSkinnedSource& source, VkDeviceAddress positions)
{
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    // Opaque in the geometry as well as per instance: the skinned material has no
    // alpha test, the same as its cascade caster (docs/skeletal_animation.md).
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    VkAccelerationStructureGeometryTrianglesDataKHR& triangles = geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = positions;
    triangles.vertexStride = kRtSkinnedPositionStrideFloats * sizeof(float);
    triangles.maxVertex = source.vertexCount == 0 ? 0 : source.vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = source.indices;
    return geometry;
}

} // namespace

RayTracingSkinnedCaster::~RayTracingSkinnedCaster()
{
    destroy();
}

void RayTracingSkinnedCaster::create(rhi::VulkanContext& context,
                                     const RayTracingSkinnedSource& source,
                                     const std::filesystem::path& preSkinShaderPath)
{
    destroy();
    if (!context.rayQueryEnabled()) {
        throw std::logic_error("RayTracingSkinnedCaster::create needs a device with ray query.");
    }
    if (source.vertices == 0 || source.skinning == 0 || source.indices == 0 || source.vertexCount == 0 ||
        source.indexCount < 3) {
        throw std::invalid_argument("The skinned mesh has no addressable geometry to trace.");
    }

    context_ = &context;
    source_ = source;
    try {
        rhi::VulkanBufferCreateInfo positionsInfo{};
        positionsInfo.size =
            static_cast<VkDeviceSize>(source.vertexCount) * kRtSkinnedPositionStrideFloats * sizeof(float);
        positionsInfo.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        positionsInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        positionsInfo.requestDeviceAddress = true;
        positions_.create(context, positionsInfo);
        rhi::debug::setObjectName(
            context.vkDevice(), positions_.buffer(), VK_OBJECT_TYPE_BUFFER, "RayTracingSkinnedPositions");

        const VkAccelerationStructureGeometryKHR geometry = skinnedGeometry(source, positions_.deviceAddress());
        VkAccelerationStructureBuildGeometryInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        info.flags = kSkinnedBuildFlags;
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.geometryCount = 1;
        info.pGeometries = &geometry;
        const uint32_t primitiveCount = source.indexCount / 3;
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(
            context.vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primitiveCount, &sizes);
        bottomLevel_.create(context,
                            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                            sizes.accelerationStructureSize,
                            "SkinnedBottomLevelAccelerationStructure");
        buildScratchSize_ = sizes.buildScratchSize;
        updateScratchSize_ = sizes.updateScratchSize;

        const VkPushConstantRange pushRange{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(RayTracingPreSkinPushConstants))};
        rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.shaderPath = preSkinShaderPath;
        pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushRange, 1);
        pipelineInfo.pipelineCache = context.pipelineCache();
        preSkinPipeline_.create(context.vkDevice(), pipelineInfo);
        rhi::debug::setObjectName(
            context.vkDevice(), preSkinPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "SkinPositionsComputePipeline");
    } catch (...) {
        destroy();
        throw;
    }
}

void RayTracingSkinnedCaster::destroy()
{
    preSkinPipeline_.reset();
    bottomLevel_.reset();
    positions_.reset();
    source_ = {};
    buildScratchSize_ = 0;
    updateScratchSize_ = 0;
    built_ = false;
    builds_ = 0;
    refits_ = 0;
    context_ = nullptr;
}

void RayTracingSkinnedCaster::record(VkCommandBuffer commandBuffer,
                                     VkDeviceAddress jointPalette,
                                     VkDeviceAddress scratch)
{
    if (!valid() || jointPalette == 0) {
        return;
    }

    RayTracingPreSkinPushConstants push{};
    push.sourceVertices = source_.vertices;
    push.skinning = source_.skinning;
    push.jointPalette = jointPalette;
    push.skinnedPositions = positions_.deviceAddress();
    push.vertexCount = source_.vertexCount;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, preSkinPipeline_.pipeline());
    vkCmdPushConstants(commandBuffer,
                       preSkinPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(push)),
                       &push);
    vkCmdDispatch(commandBuffer, rayTracingPreSkinGroupCount(source_.vertexCount), 1, 1);

    // The dispatch writes the positions the build reads, through the build's
    // shader-read input access.
    VkMemoryBarrier2 positionsBarrier{};
    positionsBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    positionsBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    positionsBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    positionsBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    positionsBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &positionsBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    const VkAccelerationStructureGeometryKHR geometry = skinnedGeometry(source_, positions_.deviceAddress());
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = kSkinnedBuildFlags;
    // A refit in place once a build exists: same triangles, same counts, new
    // positions, which is exactly the case update mode is for.
    info.mode =
        built_ ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.srcAccelerationStructure = built_ ? bottomLevel_.handle() : VK_NULL_HANDLE;
    info.dstAccelerationStructure = bottomLevel_.handle();
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    info.scratchData.deviceAddress = scratch;
    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = source_.indexCount / 3;
    const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &info, &rangePointer);

    if (built_) {
        ++refits_;
    } else {
        ++builds_;
        built_ = true;
    }
}

} // namespace ve::renderer
