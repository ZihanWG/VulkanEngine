#include "renderer/ScreenSpaceReflections.h"

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

constexpr VkFormat kSceneColorCopyFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

// A half-size copy is a filtered downscale, which vkCmdCopyImage cannot do --
// it requires matching extents. vkCmdBlitImage can, but only if the format
// advertises the blit and linear-filter features on both ends. Without them the
// subsystem keeps the full-size copy rather than silently producing a
// point-sampled, aliased reflection source.
[[nodiscard]] bool supportsFilteredBlit(VkPhysicalDevice physicalDevice, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);

    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                              VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}


// std430 mirror of the SsrParamsBuffer block in ssr_trace.frag.
struct SsrParams {
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 inverseProjection{1.0f};
    glm::vec4 marchParams{48.0f, 5.0f, 30.0f, 0.35f};
    glm::vec4 weightParams{1.0f, 0.6f, 0.1f, 0.0f};
    // xy = written/allocated for the thin G-buffer, which is sub-rected. Depth
    // and the scene-colour copy are still sized to what is written, so they are
    // sampled unscaled. zw unused.
    glm::vec4 subRect{1.0f, 1.0f, 0.0f, 0.0f};
};

static_assert(sizeof(SsrParams) == 240);

} // namespace

ScreenSpaceReflections::ScreenSpaceReflections(rhi::VulkanContext& context,
                                               rhi::VulkanSwapchain& swapchain,
                                               const RenderResolution& renderResolution,
                                               RenderGraph& renderGraph,
                                               GpuProfiler& gpuProfiler,
                                               SsrSettings& settings)
    : context_(context), swapchain_(swapchain), renderResolution_(renderResolution), renderGraph_(renderGraph),
      gpuProfiler_(gpuProfiler), settings_(settings)
{
}

ScreenSpaceReflections::~ScreenSpaceReflections()
{
    destroyResources();
    pipeline_.reset();
    descriptorSetLayout_.reset();
}

void ScreenSpaceReflections::createDescriptorSetLayout()
{
    // 0 = main depth, 1 = thin G-buffer, 2 = scene-color copy, 3 = params SSBO.
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    for (uint32_t bindingIndex = 0; bindingIndex < 3; ++bindingIndex) {
        bindings[bindingIndex].binding = bindingIndex;
        bindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[bindingIndex].descriptorCount = 1;
        bindings[bindingIndex].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // The prefiltered environment cube and the split-sum BRDF LUT: the same two
    // the main pass uses for specular IBL. This pass needs them to know what it
    // is replacing, not to add anything -- see the shader.
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    descriptorSetLayout_.create(context_.vkDevice(),
                                std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              descriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "SsrDescriptorSetLayout");
}

void ScreenSpaceReflections::createPipeline(const std::filesystem::path& vertexShaderPath,
                                            const std::filesystem::path& fragmentShaderPath)
{
    const VkDescriptorSetLayout setLayout = descriptorSetLayout_.handle();

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = vertexShaderPath;
    pipelineInfo.fragmentShaderPath = fragmentShaderPath;
    pipelineInfo.colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // matches SceneColorHDR
    pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&setLayout, 1);
    pipelineInfo.enableAdditiveBlend = true;
    pipelineInfo.pipelineCache = context_.pipelineCache();
    pipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(context_.vkDevice(), pipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "SsrTracePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "SsrTracePipelineLayout");
}

void ScreenSpaceReflections::createResources(VkImageView normalRoughnessView, uint32_t frameCount)
{
    destroyResources();

    try {
        if (!swapchain_.depthSupportsSampling()) {
            throw std::runtime_error("SSR requires a samplable main depth image.");
        }
        if (pipeline_.pipeline() == VK_NULL_HANDLE || descriptorSetLayout_.handle() == VK_NULL_HANDLE) {
            throw std::runtime_error("SSR pipeline resources are missing.");
        }
        if (normalRoughnessView == VK_NULL_HANDLE) {
            throw std::runtime_error("SSR requires the thin G-buffer image view.");
        }

        // Allocated at the maximum like the scene colour it mirrors; the copy
        // itself still transfers only the sub-rect the frame wrote.
        //
        // Half resolution when the format can be blitted with linear filtering:
        // the trace takes one point sample from this image, so the full-size
        // copy was buying sharpness nothing asked for, at four times the bytes
        // and four times the per-frame transfer. A driver without the blit
        // features falls back to the full-size vkCmdCopyImage path.
        const VkExtent2D sceneAllocation = renderResolution_.allocationExtent();
        halfResolutionSceneColorCopy_ = supportsFilteredBlit(context_.physicalDevice(), kSceneColorCopyFormat);
        const VkExtent2D extent =
            halfResolutionSceneColorCopy_ ? RenderResolution::halved(sceneAllocation) : sceneAllocation;
        sceneColorCopyAllocationExtent_ = extent;

        rhi::VulkanImageCreateInfo copyInfo{};
        copyInfo.width = extent.width;
        copyInfo.height = extent.height;
        copyInfo.format = kSceneColorCopyFormat;
        copyInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        copyInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyInfo.debugName = "SsrSceneColorCopy";
        sceneColorCopy_.create(context_, copyInfo);
        sceneColorCopyLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &sampler_));
        rhi::debug::setObjectName(context_.vkDevice(), sampler_, VK_OBJECT_TYPE_SAMPLER, "SsrLinearClampSampler");

        frameParamsBuffers_.resize(frameCount);
        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = sizeof(SsrParams);
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            frameParamsBuffers_[frameIndex].createBuffer(context_, bufferInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameParamsBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "SsrParamsBuffer" + std::to_string(frameIndex));
        }

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        // 5 sampled images per set: depth, thin G-buffer, scene-colour copy, and
        // the prefiltered environment + BRDF LUT written later.
        poolSizes[0].descriptorCount = frameCount * 5;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = frameCount;
        descriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), frameCount);
        rhi::debug::setObjectName(
            context_.vkDevice(), descriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "SsrDescriptorPool");

        descriptorSets_.resize(frameCount, VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> setLayouts(frameCount, descriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_.handle();
        allocateInfo.descriptorSetCount = frameCount;
        allocateInfo.pSetLayouts = setLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, descriptorSets_.data()));

        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            VkDescriptorImageInfo depthInfo{};
            depthInfo.sampler = sampler_;
            depthInfo.imageView = swapchain_.depthImageView();
            depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo normalRoughnessInfo{};
            normalRoughnessInfo.sampler = sampler_;
            normalRoughnessInfo.imageView = normalRoughnessView;
            normalRoughnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo colorCopyInfo{};
            colorCopyInfo.sampler = sampler_;
            colorCopyInfo.imageView = sceneColorCopy_.imageView();
            colorCopyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo paramsInfo{};
            paramsInfo.buffer = frameParamsBuffers_[frameIndex].buffer();
            paramsInfo.offset = 0;
            paramsInfo.range = sizeof(SsrParams);

            std::array<VkWriteDescriptorSet, 4> writes{};
            const std::array<const VkDescriptorImageInfo*, 3> imageInfos{&depthInfo, &normalRoughnessInfo,
                                                                         &colorCopyInfo};
            for (uint32_t bindingIndex = 0; bindingIndex < 3; ++bindingIndex) {
                writes[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[bindingIndex].dstSet = descriptorSets_[frameIndex];
                writes[bindingIndex].dstBinding = bindingIndex;
                writes[bindingIndex].descriptorCount = 1;
                writes[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[bindingIndex].pImageInfo = imageInfos[bindingIndex];
            }
            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = descriptorSets_[frameIndex];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3].pBufferInfo = &paramsInfo;
            vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        available_ = true;
        Logger::info("Screen-space reflections enabled (screen-space march + binary refinement).");
    } catch (const std::exception& error) {
        Logger::warn(std::string("Screen-space reflections unavailable: ") + error.what());
        destroyResources();
    }
}

void ScreenSpaceReflections::destroyResources()
{
    available_ = false;
    iblBound_ = false;
    descriptorSets_.clear();
    descriptorPool_.reset();
    frameParamsBuffers_.clear();
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }
    sceneColorCopy_.reset();
    sceneColorCopyLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void ScreenSpaceReflections::uploadParams(uint32_t frameIndex,
                                          const glm::mat4& view,
                                          const glm::mat4& projection,
                                          uint32_t frameCounter)
{
    if (frameIndex >= frameParamsBuffers_.size() || !frameParamsBuffers_[frameIndex].valid()) {
        return;
    }

    SsrParams params{};
    params.view = view;
    params.projection = projection;
    params.inverseProjection = glm::inverse(projection);
    params.marchParams = {static_cast<float>(settings_.maxSteps),
                          static_cast<float>(settings_.refinementSteps),
                          settings_.maxDistance,
                          settings_.thickness};
    // xy: the full-size sources (depth, thin G-buffer). zw: the scene-colour
    // copy, which halves and rounds independently and so carries its own --
    // exactly the drift RenderResolution::subRectUvScale exists to warn about.
    const glm::vec2 copyUvScale =
        halfResolutionSceneColorCopy_
            ? RenderResolution::subRectUvScale(RenderResolution::halved(renderResolution_.extent()),
                                               sceneColorCopyAllocationExtent_)
            : renderResolution_.uvScale();
    params.subRect = glm::vec4(renderResolution_.uvScale(), copyUvScale);
    params.weightParams = {settings_.intensity,
                           settings_.maxRoughness,
                           settings_.screenEdgeFade,
                           static_cast<float>(frameCounter % 64u)};
    frameParamsBuffers_[frameIndex].upload(std::as_bytes(std::span<const SsrParams>(&params, 1)));
}

void ScreenSpaceReflections::recordCommands(VkCommandBuffer commandBuffer,
                                            uint32_t frameIndex,
                                            VkImage sceneColorImage,
                                            VkExtent2D extent)
{
    if (!available_ || frameIndex >= descriptorSets_.size() || sceneColorImage == VK_NULL_HANDLE) {
        return;
    }

    {
        const GpuProfileScope copyScope(gpuProfiler_, frameIndex, commandBuffer, "SSRCopy");
        rhi::debug::beginLabel(commandBuffer, "SSRCopyPass");
        renderGraph_.beginSsrCopyPass();

        if (halfResolutionSceneColorCopy_) {
            // Source is the sub-rect the frame actually wrote; destination is
            // half of that, which is the sub-rect of the half-size copy. Both
            // sides use RenderResolution::halved so the used and allocated
            // halves cannot round apart.
            const VkExtent2D destination = RenderResolution::halved(extent);
            VkImageBlit blitRegion{};
            blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blitRegion.srcOffsets[0] = {0, 0, 0};
            blitRegion.srcOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
            blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blitRegion.dstOffsets[0] = {0, 0, 0};
            blitRegion.dstOffsets[1] = {
                static_cast<int32_t>(destination.width), static_cast<int32_t>(destination.height), 1};
            vkCmdBlitImage(commandBuffer,
                           sceneColorImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           sceneColorCopy_.image(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &blitRegion,
                           VK_FILTER_LINEAR);
        } else {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.extent = {extent.width, extent.height, 1};
            vkCmdCopyImage(commandBuffer,
                           sceneColorImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           sceneColorCopy_.image(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &copyRegion);
        }

        renderGraph_.endSsrCopyPass();
        rhi::debug::endLabel(commandBuffer);
    }

    {
        const GpuProfileScope traceScope(gpuProfiler_, frameIndex, commandBuffer, "SSRTrace");
        rhi::debug::beginLabel(commandBuffer, "SSRTracePass");
        renderGraph_.beginSsrTracePass();

        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{};
        scissor.extent = extent;
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

        renderGraph_.endSsrTracePass();
        rhi::debug::endLabel(commandBuffer);
    }
}

// Bindings 4 and 5 are written separately from the rest.
//
// The IBL resources are built in createSceneSharedResources, which runs after
// createResources at startup, so they do not exist when the set is first
// written. Rather than reorder initialisation -- an init-order bug in exactly
// this pass has bitten before -- the Renderer calls this once the environment
// exists, and again whenever it is reloaded. Until it lands, isIblBound() is
// false and the pass falls back to its old additive behaviour.
void ScreenSpaceReflections::updateIblDescriptors(VkImageView prefilteredEnvView,
                                                  VkSampler prefilteredEnvSampler,
                                                  VkImageView brdfLutView,
                                                  VkSampler brdfLutSampler)
{
    if (descriptorSets_.empty() || prefilteredEnvView == VK_NULL_HANDLE || brdfLutView == VK_NULL_HANDLE ||
        prefilteredEnvSampler == VK_NULL_HANDLE || brdfLutSampler == VK_NULL_HANDLE) {
        return;
    }

    VkDescriptorImageInfo envInfo{};
    envInfo.sampler = prefilteredEnvSampler;
    envInfo.imageView = prefilteredEnvView;
    envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo lutInfo{};
    lutInfo.sampler = brdfLutSampler;
    lutInfo.imageView = brdfLutView;
    lutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    for (VkDescriptorSet descriptorSet : descriptorSets_) {
        if (descriptorSet == VK_NULL_HANDLE) {
            continue;
        }
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSet;
        writes[0].dstBinding = 4;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &envInfo;

        writes[1] = writes[0];
        writes[1].dstBinding = 5;
        writes[1].pImageInfo = &lutInfo;

        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    iblBound_ = true;
}

} // namespace ve::renderer
