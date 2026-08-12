#include "renderer/GroundTruthAmbientOcclusion.h"

#include "core/Logger.h"
#include "renderer/GpuProfiler.h"
#include "renderer/RenderGraph.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <array>
#include <exception>
#include <span>
#include <string>

#include <glm/gtc/matrix_inverse.hpp>

namespace ve::renderer {

namespace {

// std430 mirror of the GtaoParamsBuffer block in gtao.frag.
struct GtaoParams {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 inverseProjection{1.0f};
    // x = radius, y = falloff range, z = slice count, w = steps per slice.
    glm::vec4 params0{0.5f, 0.6f, 3.0f, 6.0f};
    // x = intensity, y = power, z = thickness, w = frame index.
    glm::vec4 params1{1.0f, 2.0f, 0.5f, 0.0f};
    // xy = written/allocated for the thin G-buffer, which is sub-rected. Depth
    // is still sized to what is written. zw unused.
    glm::vec4 subRect{1.0f, 1.0f, 0.0f, 0.0f};
};

static_assert(sizeof(GtaoParams) == 240);

} // namespace

GroundTruthAmbientOcclusion::GroundTruthAmbientOcclusion(rhi::VulkanContext& context,
                                                         rhi::VulkanSwapchain& swapchain,
                                                         const RenderResolution& renderResolution,
                                                         RenderGraph& renderGraph,
                                                         GpuProfiler& gpuProfiler,
                                                         SsaoSettings& settings)
    : context_(context), swapchain_(swapchain), renderResolution_(renderResolution), renderGraph_(renderGraph),
      gpuProfiler_(gpuProfiler), settings_(settings)
{
}

GroundTruthAmbientOcclusion::~GroundTruthAmbientOcclusion()
{
    destroyResources();
    blurPipeline_.reset();
    pipeline_.reset();
    descriptorSetLayout_.reset();
}

void GroundTruthAmbientOcclusion::createDescriptorSetLayout()
{
    // 0 = main depth, 1 = thin G-buffer normal/roughness, 2 = params SSBO.
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t bindingIndex = 0; bindingIndex < 2; ++bindingIndex) {
        bindings[bindingIndex].binding = bindingIndex;
        bindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[bindingIndex].descriptorCount = 1;
        bindings[bindingIndex].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    descriptorSetLayout_.create(context_.vkDevice(),
                                std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              descriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "GtaoDescriptorSetLayout");
}

void GroundTruthAmbientOcclusion::createPipeline(const std::filesystem::path& vertexShaderPath,
                                                 const std::filesystem::path& fragmentShaderPath,
                                                 const std::filesystem::path& blurFragmentShaderPath)
{
    const VkDescriptorSetLayout setLayout = descriptorSetLayout_.handle();

    // The horizon-search trace and the bilateral blur share the descriptor layout
    // (depth + one sampled image + params SSBO) and the single-channel target
    // format; only the fragment shader and the sampled image differ.
    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = vertexShaderPath;
    pipelineInfo.fragmentShaderPath = fragmentShaderPath;
    pipelineInfo.colorFormat = VK_FORMAT_R8_UNORM; // single-channel visibility target
    pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&setLayout, 1);
    pipelineInfo.pipelineCache = context_.pipelineCache();
    pipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(context_.vkDevice(), pipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "GtaoPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "GtaoPipelineLayout");

    pipelineInfo.fragmentShaderPath = blurFragmentShaderPath;
    blurPipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(context_.vkDevice(), blurPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "GtaoBlurPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), blurPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "GtaoBlurPipelineLayout");
}

void GroundTruthAmbientOcclusion::createResources(VkImageView normalRoughnessView, uint32_t frameCount)
{
    destroyResources();

    try {
        if (!swapchain_.depthSupportsSampling()) {
            throw std::runtime_error("GTAO requires a samplable main depth image.");
        }
        if (pipeline_.pipeline() == VK_NULL_HANDLE || blurPipeline_.pipeline() == VK_NULL_HANDLE ||
            descriptorSetLayout_.handle() == VK_NULL_HANDLE) {
            throw std::runtime_error("GTAO pipeline resources are missing.");
        }
        if (normalRoughnessView == VK_NULL_HANDLE) {
            throw std::runtime_error("GTAO requires the thin G-buffer image view.");
        }

        // Raw (pre-denoise) visibility target the trace writes and the upsample
        // reads. The horizon search runs at half resolution (4x fewer searches);
        // the joint-bilateral upsample restores full resolution from the full-res
        // depth buffer.
        // Half of the *allocation*: the trace still writes only half of what the
        // frame writes, and the two halves round independently, which is why the
        // blur gets its own uv scale rather than reusing the scene's.
        const VkExtent2D extent = renderResolution_.allocationExtent();
        const VkExtent2D halfExtent = RenderResolution::halved(extent);
        rhi::VulkanImageCreateInfo rawAoInfo{};
        rawAoInfo.width = halfExtent.width;
        rawAoInfo.height = halfExtent.height;
        rawAoInfo.format = VK_FORMAT_R8_UNORM;
        rawAoInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        rawAoInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rawAoInfo.debugName = "GtaoRawAmbientOcclusion";
        rawAo_.create(context_, rawAoInfo);
        rawAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &sampler_));
        rhi::debug::setObjectName(context_.vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "GtaoNearestClampSampler");

        frameParamsBuffers_.resize(frameCount);
        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = sizeof(GtaoParams);
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            frameParamsBuffers_[frameIndex].createBuffer(context_, bufferInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameParamsBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GtaoParamsBuffer" + std::to_string(frameIndex));
        }

        // Two descriptor sets per frame sharing one layout: the trace
        // (depth + normal + params) and the bilateral blur (depth + raw AO + params).
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = frameCount * 4;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = frameCount * 2;
        descriptorPool_.create(context_.vkDevice(),
                               std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()),
                               frameCount * 2);
        rhi::debug::setObjectName(
            context_.vkDevice(), descriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "GtaoDescriptorPool");

        descriptorSets_.resize(frameCount, VK_NULL_HANDLE);
        blurDescriptorSets_.resize(frameCount, VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> setLayouts(frameCount, descriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = frameCount;
        allocateInfo.pSetLayouts = setLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, descriptorSets_.data()));
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, blurDescriptorSets_.data()));

        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            VkDescriptorImageInfo depthInfo{};
            depthInfo.sampler = sampler_;
            depthInfo.imageView = swapchain_.depthImageView();
            depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo normalRoughnessInfo{};
            normalRoughnessInfo.sampler = sampler_;
            normalRoughnessInfo.imageView = normalRoughnessView;
            normalRoughnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo rawAoDescInfo{};
            rawAoDescInfo.sampler = sampler_;
            rawAoDescInfo.imageView = rawAo_.imageView();
            rawAoDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo paramsInfo{};
            paramsInfo.buffer = frameParamsBuffers_[frameIndex].buffer();
            paramsInfo.offset = 0;
            paramsInfo.range = sizeof(GtaoParams);

            const std::array<VkDescriptorSet, 2> targetSets{descriptorSets_[frameIndex], blurDescriptorSets_[frameIndex]};
            // Binding 1 differs per set: the trace reads the normal, the blur reads raw AO.
            const std::array<const VkDescriptorImageInfo*, 2> secondImageInfos{&normalRoughnessInfo, &rawAoDescInfo};

            std::array<VkWriteDescriptorSet, 6> writes{};
            for (uint32_t setIndex = 0; setIndex < 2; ++setIndex) {
                const uint32_t base = setIndex * 3u;
                writes[base + 0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[base + 0].dstSet = targetSets[setIndex];
                writes[base + 0].dstBinding = 0;
                writes[base + 0].descriptorCount = 1;
                writes[base + 0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[base + 0].pImageInfo = &depthInfo;

                writes[base + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[base + 1].dstSet = targetSets[setIndex];
                writes[base + 1].dstBinding = 1;
                writes[base + 1].descriptorCount = 1;
                writes[base + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[base + 1].pImageInfo = secondImageInfos[setIndex];

                writes[base + 2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[base + 2].dstSet = targetSets[setIndex];
                writes[base + 2].dstBinding = 2;
                writes[base + 2].descriptorCount = 1;
                writes[base + 2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[base + 2].pBufferInfo = &paramsInfo;
            }
            vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        available_ = true;
        Logger::info("Ground-truth ambient occlusion enabled (horizon-search GTAO).");
    } catch (const std::exception& error) {
        Logger::warn(std::string("Ground-truth ambient occlusion unavailable: ") + error.what());
        destroyResources();
    }
}

void GroundTruthAmbientOcclusion::destroyResources()
{
    available_ = false;
    descriptorSets_.clear();
    blurDescriptorSets_.clear();
    descriptorPool_.reset();
    frameParamsBuffers_.clear();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    rawAo_.reset();
    rawAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void GroundTruthAmbientOcclusion::uploadParams(uint32_t frameIndex,
                                               const glm::mat4& view,
                                               const glm::mat4& projection,
                                               uint32_t frameCounter)
{
    if (frameIndex >= frameParamsBuffers_.size() || !frameParamsBuffers_[frameIndex].valid()) {
        return;
    }

    GtaoParams params{};
    params.view = view;
    params.projection = projection;
    params.inverseProjection = glm::inverse(projection);
    params.params0 = {std::max(settings_.radius, 0.01f),
                      std::clamp(settings_.falloff, 0.01f, 1.0f),
                      static_cast<float>(std::max(settings_.sliceCount, 1)),
                      static_cast<float>(std::max(settings_.stepsPerSlice, 1))};
    // xy: the thin G-buffer, which shares the scene allocation. zw: the raw AO
    // target, whose used and allocated halves round independently.
    params.subRect = glm::vec4(renderResolution_.uvScale(),
                               RenderResolution::subRectUvScale(
                                   RenderResolution::halved(renderResolution_.extent()),
                                   RenderResolution::halved(renderResolution_.allocationExtent())));
    params.params1 = {std::max(settings_.intensity, 0.0f),
                      std::max(settings_.power, 0.0001f),
                      std::max(settings_.thickness, 0.01f),
                      static_cast<float>(frameCounter % 64u)};
    frameParamsBuffers_[frameIndex].upload(std::as_bytes(std::span<const GtaoParams>(&params, 1)));
}

void GroundTruthAmbientOcclusion::recordCommands(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent)
{
    if (!available_ || frameIndex >= descriptorSets_.size() || frameIndex >= blurDescriptorSets_.size()) {
        return;
    }

    const VkExtent2D halfExtent{std::max(1u, extent.width / 2u), std::max(1u, extent.height / 2u)};

    VkViewport viewport{};
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{};

    // Trace: horizon search into the half-resolution raw AO target.
    {
        const GpuProfileScope traceScope(gpuProfiler_, frameIndex, commandBuffer, "GTAO");
        rhi::debug::beginLabel(commandBuffer, "GTAOPass");
        renderGraph_.beginGtaoPass();

        viewport.width = static_cast<float>(halfExtent.width);
        viewport.height = static_cast<float>(halfExtent.height);
        scissor.extent = halfExtent;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_.layout(),
                                0,
                                1,
                                &descriptorSets_[frameIndex],
                                0,
                                nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        renderGraph_.endGtaoPass();
        rhi::debug::endLabel(commandBuffer);
    }

    // Joint-bilateral upsample + denoise: half-res raw AO -> full-res composite AO.
    {
        const GpuProfileScope blurScope(gpuProfiler_, frameIndex, commandBuffer, "GTAOBlur");
        rhi::debug::beginLabel(commandBuffer, "GTAOBlurPass");
        renderGraph_.beginGtaoBlurPass();

        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        scissor.extent = extent;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline_.pipeline());
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                blurPipeline_.layout(),
                                0,
                                1,
                                &blurDescriptorSets_[frameIndex],
                                0,
                                nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        renderGraph_.endGtaoBlurPass();
        rhi::debug::endLabel(commandBuffer);
    }
}

} // namespace ve::renderer
