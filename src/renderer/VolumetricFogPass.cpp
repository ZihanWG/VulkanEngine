#include "renderer/VolumetricFogPass.h"

#include "core/Logger.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <array>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

constexpr uint32_t kInjectBindingParams = 0;
constexpr uint32_t kInjectBindingShadowMap = 1;
constexpr uint32_t kInjectBindingScatterVolume = 2;
constexpr uint32_t kInjectBindingCount = 3;

constexpr uint32_t kIntegrateBindingParams = 0;
constexpr uint32_t kIntegrateBindingScatterVolume = 1;
constexpr uint32_t kIntegrateBindingIntegratedVolume = 2;
constexpr uint32_t kIntegrateBindingCount = 3;

// Matches the local_size in both fog shaders.
constexpr uint32_t kFogLocalSize = 8;

[[nodiscard]] uint32_t dispatchCount(uint32_t total, uint32_t localSize)
{
    return (total + localSize - 1) / localSize;
}

VkImageMemoryBarrier2 volumeBarrier(VkImage image,
                                    VkImageLayout oldLayout,
                                    VkImageLayout newLayout,
                                    VkPipelineStageFlags2 srcStage,
                                    VkAccessFlags2 srcAccess,
                                    VkPipelineStageFlags2 dstStage,
                                    VkAccessFlags2 dstAccess)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return barrier;
}

void submitImageBarriers(VkCommandBuffer commandBuffer, std::span<const VkImageMemoryBarrier2> barriers)
{
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
    dependency.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace

VolumetricFogPass::~VolumetricFogPass()
{
    reset();
}

void VolumetricFogPass::create(rhi::VulkanContext& context,
                               uint32_t frameCount,
                               const std::filesystem::path& injectShaderPath,
                               const std::filesystem::path& integrateShaderPath,
                               VkImageView shadowMapView,
                               VkSampler shadowMapSampler)
{
    reset();
    context_ = &context;
    shadowMapView_ = shadowMapView;
    shadowMapSampler_ = shadowMapSampler;

    // Volumes and sampler are created outside the try: the material descriptor
    // set binds the integrated volume at a sampler3D binding whether or not fog
    // is enabled, and a descriptor cannot be left dangling. Only the compute
    // half below is optional.
    createVolumes();
    createSampler();

    try {
        if (shadowMapView_ == VK_NULL_HANDLE || shadowMapSampler_ == VK_NULL_HANDLE) {
            throw std::runtime_error("Volumetric fog requires the cascaded shadow map to sample.");
        }

        createDescriptorResources(frameCount);
        createPipelines(injectShaderPath, integrateShaderPath);
        writeDescriptorSets();

        available_ = true;
        Logger::info("Volumetric fog enabled with a " + std::to_string(kFogGridX) + "x" +
                     std::to_string(kFogGridY) + "x" + std::to_string(kFogGridZ) + " froxel volume.");
    } catch (const std::exception& error) {
        // Optional subsystem: a failure leaves the scene rendering without fog
        // rather than failing device creation, matching the clustered lighting
        // and punctual shadow policy. The volumes deliberately survive so the
        // material descriptors still have something valid to bind.
        injectPipeline_.reset();
        integratePipeline_.reset();
        injectSets_.clear();
        integrateSets_.clear();
        descriptorPool_.reset();
        injectSetLayout_.reset();
        integrateSetLayout_.reset();
        injectParamBuffers_.clear();
        integrateParamBuffers_.clear();
        available_ = false;
        Logger::warn(std::string("Volumetric fog unavailable: ") + error.what());
    }
}

void VolumetricFogPass::reset()
{
    if (context_ != nullptr && sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_->vkDevice(), sampler_, nullptr);
    }
    sampler_ = VK_NULL_HANDLE;

    injectPipeline_.reset();
    integratePipeline_.reset();
    injectSets_.clear();
    integrateSets_.clear();
    descriptorPool_.reset();
    injectSetLayout_.reset();
    integrateSetLayout_.reset();
    injectParamBuffers_.clear();
    integrateParamBuffers_.clear();
    scatterVolume_.reset();
    integratedVolume_.reset();
    scatterVolumeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    integratedVolumeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    shadowMapView_ = VK_NULL_HANDLE;
    shadowMapSampler_ = VK_NULL_HANDLE;
    volumeInitialized_ = false;
    available_ = false;
    context_ = nullptr;
}

void VolumetricFogPass::createVolumes()
{
    const auto makeVolume = [&](rhi::VulkanImage& image, const char* name) {
        rhi::VulkanImageCreateInfo info{};
        info.width = kFogGridX;
        info.height = kFogGridY;
        info.depth = kFogGridZ;
        // Half float: the volume is 160x90x64 and both scattering and
        // transmittance are low-frequency, so full float would double the
        // footprint for no visible gain.
        info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        // TRANSFER_DST is for the one-time neutral clear in
        // ensureVolumeInitialized.
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.viewType = VK_IMAGE_VIEW_TYPE_3D;
        info.debugName = name;
        image.create(*context_, info);
    };

    makeVolume(scatterVolume_, "FogScatterVolume");
    makeVolume(integratedVolume_, "FogIntegratedVolume");
}

void VolumetricFogPass::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // Linear in all three axes: the volume is far coarser than the framebuffer,
    // so trilinear filtering across froxels is what keeps the fog smooth instead
    // of showing the grid.
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Clamping matters at the volume's far face: a fragment past the fog range
    // should keep the fog accumulated at the edge, not wrap around to the near
    // slice.
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_->vkDevice(), &samplerInfo, nullptr, &sampler_));
    rhi::debug::setObjectName(context_->vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "FogVolumeSampler");
}

void VolumetricFogPass::createDescriptorResources(uint32_t frameCount)
{
    const VkDevice device = context_->vkDevice();

    std::array<VkDescriptorSetLayoutBinding, kInjectBindingCount> injectBindings{};
    injectBindings[kInjectBindingParams].binding = kInjectBindingParams;
    injectBindings[kInjectBindingParams].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    injectBindings[kInjectBindingParams].descriptorCount = 1;
    injectBindings[kInjectBindingParams].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    injectBindings[kInjectBindingShadowMap].binding = kInjectBindingShadowMap;
    injectBindings[kInjectBindingShadowMap].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    injectBindings[kInjectBindingShadowMap].descriptorCount = 1;
    injectBindings[kInjectBindingShadowMap].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    injectBindings[kInjectBindingScatterVolume].binding = kInjectBindingScatterVolume;
    injectBindings[kInjectBindingScatterVolume].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    injectBindings[kInjectBindingScatterVolume].descriptorCount = 1;
    injectBindings[kInjectBindingScatterVolume].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    injectSetLayout_.create(device,
                            std::span<const VkDescriptorSetLayoutBinding>(injectBindings.data(),
                                                                          injectBindings.size()));

    std::array<VkDescriptorSetLayoutBinding, kIntegrateBindingCount> integrateBindings{};
    integrateBindings[kIntegrateBindingParams].binding = kIntegrateBindingParams;
    integrateBindings[kIntegrateBindingParams].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    integrateBindings[kIntegrateBindingParams].descriptorCount = 1;
    integrateBindings[kIntegrateBindingParams].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    integrateBindings[kIntegrateBindingScatterVolume].binding = kIntegrateBindingScatterVolume;
    integrateBindings[kIntegrateBindingScatterVolume].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    integrateBindings[kIntegrateBindingScatterVolume].descriptorCount = 1;
    integrateBindings[kIntegrateBindingScatterVolume].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    integrateBindings[kIntegrateBindingIntegratedVolume].binding = kIntegrateBindingIntegratedVolume;
    integrateBindings[kIntegrateBindingIntegratedVolume].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    integrateBindings[kIntegrateBindingIntegratedVolume].descriptorCount = 1;
    integrateBindings[kIntegrateBindingIntegratedVolume].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    integrateSetLayout_.create(device,
                               std::span<const VkDescriptorSetLayoutBinding>(integrateBindings.data(),
                                                                             integrateBindings.size()));

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = frameCount * 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = frameCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[2].descriptorCount = frameCount * 3;
    descriptorPool_.create(device, std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()),
                           frameCount * 2);

    injectParamBuffers_.resize(frameCount);
    integrateParamBuffers_.resize(frameCount);
    injectSets_.resize(frameCount);
    integrateSets_.resize(frameCount);

    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        const auto makeUniformBuffer = [&](rhi::VulkanBuffer& buffer, VkDeviceSize size, const char* name) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = size;
            bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            buffer.createBuffer(*context_, bufferInfo);
            rhi::debug::setObjectName(
                device, buffer.buffer(), VK_OBJECT_TYPE_BUFFER, name + std::to_string(frameIndex));
        };

        makeUniformBuffer(injectParamBuffers_[frameIndex], sizeof(FogInjectParams), "FogInjectParams");
        makeUniformBuffer(integrateParamBuffers_[frameIndex], sizeof(FogIntegrateParams), "FogIntegrateParams");

        const std::array<VkDescriptorSetLayout, 2> layouts{injectSetLayout_.handle(),
                                                           integrateSetLayout_.handle()};
        std::array<VkDescriptorSet, 2> sets{};
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocateInfo.pSetLayouts = layouts.data();
        VK_CHECK(vkAllocateDescriptorSets(device, &allocateInfo, sets.data()));

        injectSets_[frameIndex] = sets[0];
        integrateSets_[frameIndex] = sets[1];
    }
}

void VolumetricFogPass::createPipelines(const std::filesystem::path& injectShaderPath,
                                        const std::filesystem::path& integrateShaderPath)
{
    const VkDevice device = context_->vkDevice();

    const VkDescriptorSetLayout injectLayout = injectSetLayout_.handle();
    rhi::VulkanComputePipelineCreateInfo injectInfo{};
    injectInfo.shaderPath = injectShaderPath;
    injectInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&injectLayout, 1);
    injectInfo.pipelineCache = context_->pipelineCache();
    injectPipeline_.create(device, injectInfo);
    rhi::debug::setObjectName(device, injectPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "FogInjectPipeline");

    const VkDescriptorSetLayout integrateLayout = integrateSetLayout_.handle();
    rhi::VulkanComputePipelineCreateInfo integrateInfo{};
    integrateInfo.shaderPath = integrateShaderPath;
    integrateInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&integrateLayout, 1);
    integrateInfo.pipelineCache = context_->pipelineCache();
    integratePipeline_.create(device, integrateInfo);
    rhi::debug::setObjectName(
        device, integratePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "FogIntegratePipeline");
}

void VolumetricFogPass::writeDescriptorSets()
{
    const VkDevice device = context_->vkDevice();

    for (size_t frameIndex = 0; frameIndex < injectSets_.size(); ++frameIndex) {
        VkDescriptorBufferInfo injectParamsInfo{};
        injectParamsInfo.buffer = injectParamBuffers_[frameIndex].buffer();
        injectParamsInfo.offset = 0;
        injectParamsInfo.range = sizeof(FogInjectParams);

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler = shadowMapSampler_;
        shadowInfo.imageView = shadowMapView_;
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo scatterInfo{};
        scatterInfo.imageView = scatterVolume_.imageView();
        scatterInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo integrateParamsInfo{};
        integrateParamsInfo.buffer = integrateParamBuffers_[frameIndex].buffer();
        integrateParamsInfo.offset = 0;
        integrateParamsInfo.range = sizeof(FogIntegrateParams);

        VkDescriptorImageInfo integratedInfo{};
        integratedInfo.imageView = integratedVolume_.imageView();
        integratedInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 6> writes{};
        const auto makeWrite =
            [](VkDescriptorSet set, uint32_t binding, VkDescriptorType type) {
                VkWriteDescriptorSet write{};
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = set;
                write.dstBinding = binding;
                write.dstArrayElement = 0;
                write.descriptorCount = 1;
                write.descriptorType = type;
                return write;
            };

        writes[0] = makeWrite(injectSets_[frameIndex], kInjectBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writes[0].pBufferInfo = &injectParamsInfo;
        writes[1] =
            makeWrite(injectSets_[frameIndex], kInjectBindingShadowMap, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writes[1].pImageInfo = &shadowInfo;
        writes[2] =
            makeWrite(injectSets_[frameIndex], kInjectBindingScatterVolume, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writes[2].pImageInfo = &scatterInfo;

        writes[3] =
            makeWrite(integrateSets_[frameIndex], kIntegrateBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        writes[3].pBufferInfo = &integrateParamsInfo;
        writes[4] = makeWrite(
            integrateSets_[frameIndex], kIntegrateBindingScatterVolume, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writes[4].pImageInfo = &scatterInfo;
        writes[5] = makeWrite(
            integrateSets_[frameIndex], kIntegrateBindingIntegratedVolume, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writes[5].pImageInfo = &integratedInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void VolumetricFogPass::updateShadowMap(VkImageView shadowMapView, VkSampler shadowMapSampler)
{
    if (!available_ || shadowMapView == VK_NULL_HANDLE || shadowMapSampler == VK_NULL_HANDLE) {
        return;
    }

    shadowMapView_ = shadowMapView;
    shadowMapSampler_ = shadowMapSampler;
    writeDescriptorSets();
}

void VolumetricFogPass::updateParams(uint32_t frameIndex, const FogInjectParams& injectParams)
{
    if (!available_ || frameIndex >= injectParamBuffers_.size()) {
        return;
    }

    injectParamBuffers_[frameIndex].upload(std::as_bytes(std::span<const FogInjectParams>(&injectParams, 1)));

    FogIntegrateParams integrateParams{};
    integrateParams.fogParams = glm::vec4(injectParams.fogParams.y, 0.0f, 0.0f, 0.0f);
    integrateParamBuffers_[frameIndex].upload(
        std::as_bytes(std::span<const FogIntegrateParams>(&integrateParams, 1)));
}

void VolumetricFogPass::ensureVolumeInitialized(VkCommandBuffer commandBuffer)
{
    if (volumeInitialized_ || !hasVolume()) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "FogVolumeInit");

    const VkImageMemoryBarrier2 toTransfer = volumeBarrier(integratedVolume_.image(),
                                                           VK_IMAGE_LAYOUT_UNDEFINED,
                                                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                                           VK_ACCESS_2_NONE,
                                                           VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                                           VK_ACCESS_2_TRANSFER_WRITE_BIT);
    submitImageBarriers(commandBuffer, std::span<const VkImageMemoryBarrier2>(&toTransfer, 1));

    // rgb = 0 scattered light, a = 1 transmittance: the identity for the apply
    // step, so sampling it changes nothing.
    VkClearColorValue clearValue{};
    clearValue.float32[0] = 0.0f;
    clearValue.float32[1] = 0.0f;
    clearValue.float32[2] = 0.0f;
    clearValue.float32[3] = 1.0f;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(
        commandBuffer, integratedVolume_.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &range);

    const VkImageMemoryBarrier2 toSampled = volumeBarrier(integratedVolume_.image(),
                                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                          VK_PIPELINE_STAGE_2_CLEAR_BIT,
                                                          VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    submitImageBarriers(commandBuffer, std::span<const VkImageMemoryBarrier2>(&toSampled, 1));

    integratedVolumeLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    volumeInitialized_ = true;

    rhi::debug::endLabel(commandBuffer);
}

void VolumetricFogPass::recordCommands(VkCommandBuffer commandBuffer, uint32_t frameIndex)
{
    if (!available_ || frameIndex >= injectSets_.size()) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "VolumetricFog");

    // Both volumes are storage images for the duration of these two passes. The
    // integrated volume also has to come back from SHADER_READ_ONLY, which is
    // where the previous frame's composite left it.
    std::array<VkImageMemoryBarrier2, 2> toGeneral{
        volumeBarrier(scatterVolume_.image(),
                      scatterVolumeLayout_,
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        volumeBarrier(integratedVolume_.image(),
                      integratedVolumeLayout_,
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)};
    submitImageBarriers(commandBuffer, std::span<const VkImageMemoryBarrier2>(toGeneral.data(), toGeneral.size()));
    scatterVolumeLayout_ = VK_IMAGE_LAYOUT_GENERAL;
    integratedVolumeLayout_ = VK_IMAGE_LAYOUT_GENERAL;

    rhi::debug::beginLabel(commandBuffer, "FogInject");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, injectPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            injectPipeline_.layout(),
                            0,
                            1,
                            &injectSets_[frameIndex],
                            0,
                            nullptr);
    vkCmdDispatch(commandBuffer,
                  dispatchCount(kFogGridX, kFogLocalSize),
                  dispatchCount(kFogGridY, kFogLocalSize),
                  kFogGridZ);
    rhi::debug::endLabel(commandBuffer);

    // Integration reads every froxel injection wrote, and each column reads
    // slices written by different workgroups, so the whole dispatch has to
    // complete first.
    const VkImageMemoryBarrier2 injectToIntegrate = volumeBarrier(scatterVolume_.image(),
                                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    submitImageBarriers(commandBuffer, std::span<const VkImageMemoryBarrier2>(&injectToIntegrate, 1));

    rhi::debug::beginLabel(commandBuffer, "FogIntegrate");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            integratePipeline_.layout(),
                            0,
                            1,
                            &integrateSets_[frameIndex],
                            0,
                            nullptr);
    // One invocation per column; the march over Z happens inside the shader.
    vkCmdDispatch(
        commandBuffer, dispatchCount(kFogGridX, kFogLocalSize), dispatchCount(kFogGridY, kFogLocalSize), 1);
    rhi::debug::endLabel(commandBuffer);

    // Hand the integrated volume to the composite pass as a sampled image.
    const VkImageMemoryBarrier2 toSampled = volumeBarrier(integratedVolume_.image(),
                                                          VK_IMAGE_LAYOUT_GENERAL,
                                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    submitImageBarriers(commandBuffer, std::span<const VkImageMemoryBarrier2>(&toSampled, 1));
    integratedVolumeLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    rhi::debug::endLabel(commandBuffer);
}

} // namespace ve::renderer
