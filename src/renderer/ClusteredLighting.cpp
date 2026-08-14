#include "renderer/ClusteredLighting.h"

#include "core/Logger.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <glm/geometric.hpp>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

// Bindings 0..4 are lights, cluster AABBs, the cluster grid, the light index
// list and the params block, in that order -- see light_cull.comp. They are not
// named individually here because every loop below indexes them positionally.
constexpr uint32_t kBindingCount = 5;

VkBufferMemoryBarrier2 storageBufferBarrier(VkBuffer buffer,
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
    barrier.size = VK_WHOLE_SIZE;
    return barrier;
}

void submitBufferBarriers(VkCommandBuffer commandBuffer, std::span<const VkBufferMemoryBarrier2> barriers)
{
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
    dependency.pBufferMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace

void ClusteredLighting::create(rhi::VulkanContext& context,
                               uint32_t frameCount,
                               const std::filesystem::path& clusterBuildShaderPath,
                               const std::filesystem::path& lightCullShaderPath)
{
    context_ = &context;
    available_ = false;

    createBuffers(frameCount);

    // The brute-force fallback only needs the light buffers above, so a failure
    // to build the clustered compute path leaves punctual lighting working.
    try {
        createDescriptorResources(frameCount);
        createPipelines(clusterBuildShaderPath, lightCullShaderPath);
        available_ = true;
        Logger::info("Clustered (Forward+) lighting enabled with a " + std::to_string(kClusterGridX) + "x" +
                     std::to_string(kClusterGridY) + "x" + std::to_string(kClusterGridZ) + " froxel grid.");
    } catch (const std::exception& error) {
        available_ = false;
        Logger::warn(std::string("Clustered lighting unavailable; falling back to brute-force light evaluation: ") +
                     error.what());
    }
}

void ClusteredLighting::createBuffers(uint32_t frameCount)
{
    const VkDevice device = context_->vkDevice();

    // When the async compute queue lives in a different family, every clustered
    // buffer is created CONCURRENT across both families so the cluster passes can
    // write on the compute queue while the fragment shader reads on graphics —
    // no queue-family ownership transfers needed.
    std::array<uint32_t, 2> sharedFamilies{};
    size_t sharedFamilyCount = 0;
    if (context_->asyncComputeAvailable() &&
        context_->asyncComputeQueueFamily() != context_->queueFamilies().graphicsFamily.value()) {
        sharedFamilies = {context_->queueFamilies().graphicsFamily.value(), context_->asyncComputeQueueFamily()};
        sharedFamilyCount = 2;
    }

    auto makeBuffer = [&](std::vector<rhi::VulkanBuffer>& buffers,
                          VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          bool hostVisible,
                          bool deviceAddress,
                          const char* name) {
        buffers.clear();
        buffers.resize(frameCount);
        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.memoryUsage = hostVisible ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            bufferInfo.allocationFlags = hostVisible ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT : 0;
            bufferInfo.requestDeviceAddress = deviceAddress;
            bufferInfo.sharedQueueFamilies =
                std::span<const uint32_t>(sharedFamilies.data(), sharedFamilyCount);
            buffers[frameIndex].createBuffer(*context_, bufferInfo);
            rhi::debug::setObjectName(
                device, buffers[frameIndex].buffer(), VK_OBJECT_TYPE_BUFFER, name + std::to_string(frameIndex));
        }
    };

    constexpr VkBufferUsageFlags kStorageBda =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    makeBuffer(lightBuffers_,
               static_cast<VkDeviceSize>(kMaxLights) * sizeof(GpuLight),
               kStorageBda,
               /*hostVisible=*/true,
               /*deviceAddress=*/true,
               "ClusteredLightBuffer");
    makeBuffer(clusterAabbBuffers_,
               static_cast<VkDeviceSize>(kClusterCount) * sizeof(ClusterAabb),
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               /*hostVisible=*/false,
               /*deviceAddress=*/false,
               "ClusterAabbBuffer");
    makeBuffer(clusterGridBuffers_,
               static_cast<VkDeviceSize>(kClusterCount) * sizeof(ClusterCell),
               kStorageBda,
               /*hostVisible=*/false,
               /*deviceAddress=*/true,
               "ClusterGridBuffer");
    makeBuffer(lightIndexBuffers_,
               static_cast<VkDeviceSize>(kClusterCount) * kMaxLightsPerCluster * sizeof(uint32_t),
               kStorageBda,
               /*hostVisible=*/false,
               /*deviceAddress=*/true,
               "ClusterLightIndexBuffer");
    makeBuffer(paramsBuffers_,
               static_cast<VkDeviceSize>(sizeof(ClusterGridParams)),
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               /*hostVisible=*/true,
               /*deviceAddress=*/false,
               "ClusterParamsBuffer");
}

void ClusteredLighting::createDescriptorResources(uint32_t frameCount)
{
    const VkDevice device = context_->vkDevice();

    std::array<VkDescriptorSetLayoutBinding, kBindingCount> bindings{};
    for (uint32_t binding = 0; binding < kBindingCount; ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    descriptorSetLayout_.create(device,
                                std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(
        device, descriptorSetLayout_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "ClusterCullDescriptorSetLayout");

    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = frameCount * kBindingCount;
    descriptorPool_.create(
        device, std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), frameCount);
    rhi::debug::setObjectName(
        device, descriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "ClusterCullDescriptorPool");

    descriptorSets_.assign(frameCount, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> setLayouts(frameCount, descriptorSetLayout_.handle());
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool_.handle();
    allocateInfo.descriptorSetCount = frameCount;
    allocateInfo.pSetLayouts = setLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, descriptorSets_.data()));

    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        const std::array<const rhi::VulkanBuffer*, kBindingCount> boundBuffers{
            &lightBuffers_[frameIndex],
            &clusterAabbBuffers_[frameIndex],
            &clusterGridBuffers_[frameIndex],
            &lightIndexBuffers_[frameIndex],
            &paramsBuffers_[frameIndex],
        };

        std::array<VkDescriptorBufferInfo, kBindingCount> bufferInfos{};
        std::array<VkWriteDescriptorSet, kBindingCount> writes{};
        for (uint32_t binding = 0; binding < kBindingCount; ++binding) {
            bufferInfos[binding].buffer = boundBuffers[binding]->buffer();
            bufferInfos[binding].offset = 0;
            bufferInfos[binding].range = boundBuffers[binding]->size();

            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = descriptorSets_[frameIndex];
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &bufferInfos[binding];
        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        rhi::debug::setObjectName(device,
                                  descriptorSets_[frameIndex],
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                  "ClusterCullDescriptorSet" + std::to_string(frameIndex));
    }
}

void ClusteredLighting::createPipelines(const std::filesystem::path& clusterBuildShaderPath,
                                        const std::filesystem::path& lightCullShaderPath)
{
    const VkDevice device = context_->vkDevice();
    const VkDescriptorSetLayout layout = descriptorSetLayout_.handle();

    rhi::VulkanComputePipelineCreateInfo buildInfo{};
    buildInfo.shaderPath = clusterBuildShaderPath;
    buildInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&layout, 1);
    buildInfo.pipelineCache = context_->pipelineCache();
    clusterBuildPipeline_.create(device, buildInfo);
    rhi::debug::setObjectName(
        device, clusterBuildPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "ClusterBuildComputePipeline");

    rhi::VulkanComputePipelineCreateInfo cullInfo{};
    cullInfo.shaderPath = lightCullShaderPath;
    cullInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&layout, 1);
    cullInfo.pipelineCache = context_->pipelineCache();
    lightCullPipeline_.create(device, cullInfo);
    rhi::debug::setObjectName(
        device, lightCullPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "LightCullComputePipeline");
}

uint32_t ClusteredLighting::lightCount() const
{
    return static_cast<uint32_t>(std::min<size_t>(lights_.size(), kMaxLights));
}

void ClusteredLighting::addPointLight(const glm::vec3& position, const glm::vec3& color, float intensity, float range)
{
    GpuLight light{};
    light.positionRange = glm::vec4(position, range);
    light.colorIntensity = glm::vec4(color, intensity);
    light.directionType = glm::vec4(0.0f, -1.0f, 0.0f, static_cast<float>(LightType::Point));
    light.spotScaleOffset = glm::vec4(-1.0f, 1.0f, 0.0f, 0.0f);
    lights_.push_back(light);
}

void ClusteredLighting::addSpotLight(const glm::vec3& position,
                                     const glm::vec3& direction,
                                     const glm::vec3& color,
                                     float intensity,
                                     float range,
                                     float innerAngleRadians,
                                     float outerAngleRadians)
{
    const float cosInner = std::cos(innerAngleRadians);
    const float cosOuter = std::cos(outerAngleRadians);
    const float invDelta = 1.0f / std::max(cosInner - cosOuter, 1.0e-4f);

    GpuLight light{};
    light.positionRange = glm::vec4(position, range);
    light.colorIntensity = glm::vec4(color, intensity);
    light.directionType = glm::vec4(glm::normalize(direction), static_cast<float>(LightType::Spot));
    light.spotScaleOffset = glm::vec4(cosOuter, invDelta, 0.0f, 0.0f);
    lights_.push_back(light);
}

void ClusteredLighting::upload(uint32_t frameIndex)
{
    if (frameIndex >= lightBuffers_.size() || lights_.empty()) {
        return;
    }

    const size_t count = std::min<size_t>(lights_.size(), kMaxLights);
    lightBuffers_[frameIndex].upload(std::as_bytes(std::span<const GpuLight>(lights_.data(), count)));
}

void ClusteredLighting::updateParams(uint32_t frameIndex,
                                     const glm::mat4& viewMatrix,
                                     const glm::mat4& inverseProjection,
                                     float zNear,
                                     float zFar,
                                     float screenWidth,
                                     float screenHeight)
{
    if (frameIndex >= paramsBuffers_.size()) {
        return;
    }

    ClusterGridParams params{};
    params.viewMatrix = viewMatrix;
    params.inverseProjection = inverseProjection;
    params.gridDims = glm::uvec4(kClusterGridX, kClusterGridY, kClusterGridZ, kMaxLightsPerCluster);
    params.zNearFarScreen = glm::vec4(zNear, zFar, screenWidth, screenHeight);
    params.lightCountAndPad = glm::uvec4(lightCount(), 0, 0, 0);
    paramsBuffers_[frameIndex].upload(std::as_bytes(std::span<const ClusterGridParams>(&params, 1)));
}

void ClusteredLighting::recordClusterBuild(VkCommandBuffer commandBuffer, uint32_t frameIndex, bool asyncQueue)
{
    (void)asyncQueue; // the build->cull barrier is compute->compute on any queue

    if (!available_ || frameIndex >= descriptorSets_.size()) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, clusterBuildPipeline_.pipeline());
    const VkDescriptorSet descriptorSet = descriptorSets_[frameIndex];
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            clusterBuildPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);

    const uint32_t groupCount = (kClusterCount + kClusterCullLocalSize - 1) / kClusterCullLocalSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);

    // The froxel AABBs are consumed by light_cull.comp next.
    const VkBufferMemoryBarrier2 barrier = storageBufferBarrier(clusterAabbBuffers_[frameIndex].buffer(),
                                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    submitBufferBarriers(commandBuffer, std::span<const VkBufferMemoryBarrier2>(&barrier, 1));
}

void ClusteredLighting::recordLightCull(VkCommandBuffer commandBuffer, uint32_t frameIndex, bool asyncQueue)
{
    if (!available_ || frameIndex >= descriptorSets_.size()) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, lightCullPipeline_.pipeline());
    const VkDescriptorSet descriptorSet = descriptorSets_[frameIndex];
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, lightCullPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    const uint32_t groupCount = (kClusterCount + kClusterCullLocalSize - 1) / kClusterCullLocalSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);

    // On the async compute queue the fragment stage is not a valid barrier
    // destination; the graphics submit's semaphore wait (at FRAGMENT_SHADER)
    // provides the cross-queue execution + memory dependency instead.
    if (asyncQueue) {
        return;
    }

    // The grid + index list are read by the main HDR fragment shader (via BDA).
    const std::array<VkBufferMemoryBarrier2, 2> barriers{
        storageBufferBarrier(clusterGridBuffers_[frameIndex].buffer(),
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
        storageBufferBarrier(lightIndexBuffers_[frameIndex].buffer(),
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
    };
    submitBufferBarriers(commandBuffer, std::span<const VkBufferMemoryBarrier2>(barriers.data(), barriers.size()));
}

VkDeviceAddress ClusteredLighting::lightBufferAddress(uint32_t frameIndex) const
{
    return frameIndex < lightBuffers_.size() ? lightBuffers_[frameIndex].deviceAddress() : 0;
}

VkDeviceAddress ClusteredLighting::clusterGridAddress(uint32_t frameIndex) const
{
    return frameIndex < clusterGridBuffers_.size() ? clusterGridBuffers_[frameIndex].deviceAddress() : 0;
}

VkDeviceAddress ClusteredLighting::lightIndexListAddress(uint32_t frameIndex) const
{
    return frameIndex < lightIndexBuffers_.size() ? lightIndexBuffers_[frameIndex].deviceAddress() : 0;
}

} // namespace ve::renderer
