#include "renderer/Renderer.h"

#include "core/Logger.h"
#include "core/Window.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ve {

namespace {

struct ObjectFrameData {
    glm::mat4 mvp{1.0f};
    glm::mat4 model{1.0f};
    glm::mat4 lightMvp[4]{{1.0f}, {1.0f}, {1.0f}, {1.0f}};
    glm::vec4 lightDirection{0.35f, -0.65f, -0.55f, 0.0f};
    glm::vec4 lightColor{0.85f, 0.85f, 0.85f, 1.0f};
    glm::vec4 ambientColor{0.15f, 0.15f, 0.15f, 1.0f};
    glm::vec4 cascadeSplits{40.0f};
    glm::vec4 shadowSettings{0.002f, 0.005f, 1.0f, 1.0f};
    glm::vec4 baseColorFactor{1.0f};
    glm::vec4 materialParams{0.0f, 0.5f, 1.0f, 0.0f};
    glm::vec4 cameraPosition{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 cameraForward{0.0f, 0.0f, -1.0f, 4.0f};
    glm::uvec4 textureIndices{0, 0, 0, 0};
};

// Mirrors the shader's std430 buffer_reference block. std430 stores mat4 as
// four 16-byte columns and vec4/uvec4 as 16 bytes. The four lightMvp matrices
// intentionally duplicate cascade data per draw for this educational milestone;
// a later scene/light buffer can remove that per-object cost.
// materialParams.x = metallic, y = roughness, z = multiScatterStrength,
// and w is reserved for future scalar material data.
// cascadeSplits stores positive camera-view depths for cascades 0..3.
// cameraForward.xyz is the camera forward vector, and w stores cascade count.
// textureIndices.x = base color, y = normal, z = metallic-roughness, w = reserved.
static_assert(offsetof(ObjectFrameData, mvp) == 0);
static_assert(offsetof(ObjectFrameData, model) == 64);
static_assert(offsetof(ObjectFrameData, lightMvp) == 128);
static_assert(offsetof(ObjectFrameData, lightDirection) == 384);
static_assert(offsetof(ObjectFrameData, lightColor) == 400);
static_assert(offsetof(ObjectFrameData, ambientColor) == 416);
static_assert(offsetof(ObjectFrameData, cascadeSplits) == 432);
static_assert(offsetof(ObjectFrameData, shadowSettings) == 448);
static_assert(offsetof(ObjectFrameData, baseColorFactor) == 464);
static_assert(offsetof(ObjectFrameData, materialParams) == 480);
static_assert(offsetof(ObjectFrameData, cameraPosition) == 496);
static_assert(offsetof(ObjectFrameData, cameraForward) == 512);
static_assert(offsetof(ObjectFrameData, textureIndices) == 528);
static_assert(sizeof(ObjectFrameData) == 544);

constexpr uint32_t kMaxFrameObjects = 1024;
constexpr uint32_t kMaxDrawItems = 1024;
constexpr uint32_t kMaxMaterialDescriptorSets = 256;
constexpr uint32_t kGpuCullLocalSize = 64;
constexpr uint32_t kMaxMeshDrawBatches = kMaxDrawItems;
constexpr VkDeviceSize kBatchVisibleCountBufferSize = kMaxMeshDrawBatches * sizeof(uint32_t);
constexpr float kUnboundedCullExtent = 100000000.0f;

const glm::vec4 kDirectionalLightDirection{0.35f, -0.65f, -0.55f, 0.0f};
const glm::vec4 kDirectionalLightColor{0.85f, 0.85f, 0.85f, 1.0f};
const glm::vec4 kAmbientLightColor{0.15f, 0.15f, 0.15f, 1.0f};

struct PushConstants {
    VkDeviceAddress objectFrameDataAddress = 0;
    uint32_t cascadeIndex = 0;
    uint32_t padding = 0;
};

static_assert(offsetof(PushConstants, objectFrameDataAddress) == 0);
static_assert(offsetof(PushConstants, cascadeIndex) == 8);
static_assert(sizeof(PushConstants) == 16);

// Mirrors src/shaders/cull.comp. Main and shadow GPU culling both use this
// std430 input record: vec4 members are 16-byte aligned, then scalar draw and
// batch fields are packed into the next 32 bytes for a 64-byte runtime-array
// stride. objectFrameDataIndex becomes VkDrawIndexedIndirectCommand::firstInstance
// on the bindless main and material-independent shadow indirect paths.
struct GpuCullDrawItem {
    glm::vec4 boundsMin{0.0f};
    glm::vec4 boundsMax{0.0f};
    uint32_t indexCount = 0;
    uint32_t firstIndex = 0;
    int32_t vertexOffset = 0;
    uint32_t objectFrameDataIndex = 0;
    uint32_t batchIndex = 0;
    uint32_t batchOutputBase = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

static_assert(offsetof(GpuCullDrawItem, boundsMin) == 0);
static_assert(offsetof(GpuCullDrawItem, boundsMax) == 16);
static_assert(offsetof(GpuCullDrawItem, indexCount) == 32);
static_assert(offsetof(GpuCullDrawItem, firstIndex) == 36);
static_assert(offsetof(GpuCullDrawItem, vertexOffset) == 40);
static_assert(offsetof(GpuCullDrawItem, objectFrameDataIndex) == 44);
static_assert(offsetof(GpuCullDrawItem, batchIndex) == 48);
static_assert(offsetof(GpuCullDrawItem, batchOutputBase) == 52);
static_assert(sizeof(GpuCullDrawItem) == 64);

struct GpuCullPushConstants {
    std::array<glm::vec4, 6> frustumPlanes{};
    glm::uvec4 params{0, 0, 0, 0};
};

static_assert(offsetof(GpuCullPushConstants, frustumPlanes) == 0);
static_assert(offsetof(GpuCullPushConstants, params) == 96);
static_assert(sizeof(GpuCullPushConstants) == 112);
static_assert(sizeof(GpuCullPushConstants) <= 128);

struct SkyboxPushConstants {
    glm::mat4 inverseViewProjection{1.0f};
};

static_assert(sizeof(SkyboxPushConstants) == sizeof(glm::mat4));

std::filesystem::path shaderPath(const char* filename)
{
#if defined(VULKAN_ENGINE_SHADER_DIR)
    return std::filesystem::path(VULKAN_ENGINE_SHADER_DIR) / filename;
#else
    return std::filesystem::path("shaders") / filename;
#endif
}

std::filesystem::path assetPath(const char* relativePath)
{
#if defined(VULKAN_ENGINE_ASSET_DIR)
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR) / relativePath;
#else
    return std::filesystem::path("assets") / relativePath;
#endif
}

} // namespace

Renderer::Renderer(Window& window) : window_(window)
{
    context_.initialize(window_);

    frames_.resize(rhi::kMaxFramesInFlight);
    timestampQuery_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    swapchain_.initialize(context_, window_.framebufferExtent());
    createMaterialDescriptorSetLayout();
    createBindlessMaterialTextureHeap();
    createSkyboxDescriptorSetLayout();
    createShadowMap();
    createPipeline();
    commandContext_.initialize(context_, frames_);
    createScene();
    createObjectFrameDataBuffers();
    createIndirectDrawBuffers();
    createShadowIndirectDrawBuffers();
    createGpuCullingResources();
    sync_.initialize(context_, frames_, swapchain_.imageCount());
    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);

    initialized_ = true;
}

Renderer::~Renderer()
{
    if (initialized_) {
        waitIdle();
    }
}

void Renderer::drawFrame()
{
    if (window_.isMinimized()) {
        return;
    }

    if (window_.wasResized()) {
        recreateSwapchain();
        window_.clearResizedFlag();
    }

    renderer::FrameResources& frame = frames_[currentFrame_];
    VK_CHECK(vkWaitForFences(context_.vkDevice(), 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX));
    tryPrintGpuTimings(currentFrame_);

    uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        context_.vkDevice(), swapchain_.handle(), UINT64_MAX, frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error(std::string("vkAcquireNextImageKHR failed: ") + rhi::vkResultToString(acquireResult));
    }

    if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
        VK_CHECK(vkWaitForFences(context_.vkDevice(), 1, &imagesInFlight_[imageIndex], VK_TRUE, UINT64_MAX));
    }
    imagesInFlight_[imageIndex] = frame.inFlightFence;

    VK_CHECK(vkResetFences(context_.vkDevice(), 1, &frame.inFlightFence));
    VK_CHECK(vkResetCommandBuffer(frame.commandBuffer, 0));

    updateFrameData(currentFrame_);
    recordRenderCommands(frame.commandBuffer, imageIndex);
    const VkSemaphore renderFinished = sync_.renderFinishedSemaphore(imageIndex);

    VkSemaphoreSubmitInfo waitSemaphore{};
    waitSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemaphore.semaphore = frame.imageAvailable;
    waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = frame.commandBuffer;

    VkSemaphoreSubmitInfo signalSemaphore{};
    signalSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphore.semaphore = renderFinished;
    signalSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemaphore;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemaphore;

    VK_CHECK(vkQueueSubmit2(context_.graphicsQueue(), 1, &submitInfo, frame.inFlightFence));
    timestampQuery_.markFrameSubmitted(currentFrame_);

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    const VkSwapchainKHR swapchain = swapchain_.handle();
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(context_.presentQueue(), &presentInfo);
    const bool needsRecreate = presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR ||
                               acquireResult == VK_SUBOPTIMAL_KHR || window_.wasResized();

    if (presentResult != VK_SUCCESS && presentResult != VK_ERROR_OUT_OF_DATE_KHR &&
        presentResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error(std::string("vkQueuePresentKHR failed: ") + rhi::vkResultToString(presentResult));
    }

    if (needsRecreate) {
        recreateSwapchain();
        window_.clearResizedFlag();
    }

    currentFrame_ = (currentFrame_ + 1) % static_cast<uint32_t>(frames_.size());
}

void Renderer::waitIdle()
{
    context_.waitIdle();
}

void Renderer::createMaterialDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Set 0 binding 0 is the base color texture, binding 1 is the cascaded
    // shadow-map array, binding 2 is the tangent-space normal map, and binding 3 is the
    // metallic-roughness map. Binding 4 is diffuse irradiance, binding 5 is
    // prefiltered environment specular, and binding 6 is the split-sum BRDF LUT.
    // Object MVP/model/light/material data stays on the BDA + vertex push constant path.
    materialDescriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              materialDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "MaterialDescriptorSetLayout");
}

void Renderer::createBindlessMaterialTextureHeap()
{
    bindlessMaterialTexturesAvailable_ = false;
    bindlessTextureHeap_.reset();

    if (!useBindlessMaterialTextures_) {
        return;
    }

    if (!context_.device().descriptorIndexingEnabled()) {
        Logger::warn("Bindless material descriptors unavailable; falling back to per-material descriptor binding.");
        return;
    }

    try {
        bindlessTextureHeap_.create(context_, renderer::BindlessTextureHeap::kDefaultMaxTextures);
        bindlessMaterialTexturesAvailable_ = true;
        Logger::info("Bindless material texture heap enabled with " +
                     std::to_string(bindlessTextureHeap_.maxTextures()) +
                     " descriptors per material texture class.");
    } catch (const std::exception& error) {
        Logger::warn(std::string("Bindless material descriptors unavailable; falling back to per-material "
                                 "descriptor binding: ") +
                     error.what());
        bindlessTextureHeap_.reset();
    }
}

void Renderer::createSkyboxDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    skyboxDescriptorSetLayout_.create(context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(&binding, 1));
    rhi::debug::setObjectName(context_.vkDevice(),
                              skyboxDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "SkyboxDescriptorSetLayout");
}

void Renderer::createGpuCullingResources()
{
    destroyGpuCullingResources();

    if (!useGpuCulling_) {
        if (useGpuShadowCulling_) {
            Logger::warn("GPU shadow culling unavailable because the shared GPU culling pipeline is disabled.");
        }
        return;
    }

    try {
        if (frameIndirectDrawBuffers_.size() != frames_.size()) {
            throw std::runtime_error("GPU culling requires one indirect output buffer per frame.");
        }

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
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

        gpuCullDescriptorSetLayout_.create(
            context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  gpuCullDescriptorSetLayout_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                  "GpuCullDescriptorSetLayout");

        const VkDescriptorSetLayout cullDescriptorSetLayout = gpuCullDescriptorSetLayout_.handle();
        const VkPushConstantRange pushConstantRange{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(GpuCullPushConstants))};

        rhi::VulkanComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.shaderPath = shaderPath("cull.comp.spv");
        pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&cullDescriptorSetLayout, 1);
        pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
        gpuCullPipeline_.create(context_.vkDevice(), pipelineInfo);
        rhi::debug::setObjectName(
            context_.vkDevice(), gpuCullPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "GpuCullComputePipeline");
        rhi::debug::setObjectName(
            context_.vkDevice(), gpuCullPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "GpuCullPipelineLayout");

        frameCullInputBuffers_.resize(frames_.size());
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

        frameBatchVisibleCountBuffers_.resize(frames_.size());
        frameBatchVisibleCountReadbackBuffers_.resize(frames_.size());
        frameGpuCullTotalDrawItems_.assign(frames_.size(), 0);
        frameGpuCullBatchCounts_.assign(frames_.size(), 0);
        frameGpuCullReadbackReady_.assign(frames_.size(), 0);
        frameGpuCullIndirectCountPath_.assign(frames_.size(), 0);
        for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
            rhi::VulkanBufferCreateInfo gpuCountInfo{};
            gpuCountInfo.size = kBatchVisibleCountBufferSize;
            gpuCountInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            gpuCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            frameBatchVisibleCountBuffers_[frameIndex].createBuffer(context_, gpuCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameBatchVisibleCountBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuBatchVisibleCountBuffer" + std::to_string(frameIndex));

            rhi::VulkanBufferCreateInfo readbackCountInfo{};
            readbackCountInfo.size = kBatchVisibleCountBufferSize;
            readbackCountInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            readbackCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            readbackCountInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            frameBatchVisibleCountReadbackBuffers_[frameIndex].createBuffer(context_, readbackCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameBatchVisibleCountReadbackBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuBatchVisibleCountReadbackBuffer" + std::to_string(frameIndex));
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(frames_.size() * bindings.size());

        gpuCullDescriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(&poolSize, 1), static_cast<uint32_t>(frames_.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  gpuCullDescriptorPool_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                  "GpuCullDescriptorPool");

        gpuCullDescriptorSets_.resize(frames_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> setLayouts(frames_.size(), gpuCullDescriptorSetLayout_.handle());
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
            visibleCountBufferInfo.range = kBatchVisibleCountBufferSize;

            std::array<VkWriteDescriptorSet, 3> writes{};
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

            vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      gpuCullDescriptorSets_[frameIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "GpuCullDescriptorSet" + std::to_string(frameIndex));
        }

        gpuCullingAvailable_ = true;
        Logger::info("GPU frustum culling enabled for main-pass indirect command generation and per-batch visible "
                     "count readback.");
        if (context_.device().drawIndexedIndirectCountAvailable()) {
            Logger::info("vkCmdDrawIndexedIndirectCount support detected; compacted per-batch indirect-count drawing "
                         "will be used when the main pass can use bindless multi-draw indirect.");
        }

        createGpuShadowCullingResources();
    } catch (const std::exception& error) {
        Logger::warn(std::string("GPU frustum culling unavailable; falling back to CPU culling: ") + error.what());
        destroyGpuCullingResources();
    }
}

void Renderer::destroyGpuCullingResources()
{
    destroyGpuShadowCullingResources();

    gpuCullingAvailable_ = false;
    gpuCullDescriptorSets_.clear();
    gpuCullDescriptorPool_.reset();
    frameGpuCullIndirectCountPath_.clear();
    frameGpuCullReadbackReady_.clear();
    frameGpuCullBatchCounts_.clear();
    frameGpuCullTotalDrawItems_.clear();
    frameBatchVisibleCountReadbackBuffers_.clear();
    frameBatchVisibleCountBuffers_.clear();
    frameCullInputBuffers_.clear();
    gpuCullPipeline_.reset();
    gpuCullDescriptorSetLayout_.reset();
}

void Renderer::createGpuShadowCullingResources()
{
    destroyGpuShadowCullingResources();

    if (!useGpuShadowCulling_) {
        return;
    }

    try {
        if (gpuCullDescriptorSetLayout_.handle() == VK_NULL_HANDLE || gpuCullPipeline_.pipeline() == VK_NULL_HANDLE ||
            gpuCullPipeline_.layout() == VK_NULL_HANDLE) {
            throw std::runtime_error("GPU shadow culling requires the shared cull.comp pipeline.");
        }
        if (!isShadowIndirectActive()) {
            throw std::runtime_error("GPU shadow culling requires the shadow indirect draw path.");
        }

        frameShadowCullInputBuffers_.resize(frames_.size());
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

        frameShadowBatchVisibleCountBuffers_.resize(frames_.size());
        frameShadowBatchVisibleCountReadbackBuffers_.resize(frames_.size());
        frameGpuShadowCullTotalDrawItems_.assign(frames_.size(), 0);
        frameGpuShadowCullBatchCounts_.assign(frames_.size(), 0);
        frameGpuShadowCullReadbackReady_.assign(frames_.size(), 0);
        frameGpuShadowCullIndirectCountPath_.assign(frames_.size(), 0);
        for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
            rhi::VulkanBufferCreateInfo gpuCountInfo{};
            gpuCountInfo.size = kBatchVisibleCountBufferSize;
            gpuCountInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            gpuCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            frameShadowBatchVisibleCountBuffers_[frameIndex].createBuffer(context_, gpuCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameShadowBatchVisibleCountBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuShadowBatchVisibleCountBuffer" + std::to_string(frameIndex));

            rhi::VulkanBufferCreateInfo readbackCountInfo{};
            readbackCountInfo.size = kBatchVisibleCountBufferSize;
            readbackCountInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            readbackCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            readbackCountInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            frameShadowBatchVisibleCountReadbackBuffers_[frameIndex].createBuffer(context_, readbackCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameShadowBatchVisibleCountReadbackBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuShadowBatchVisibleCountReadbackBuffer" + std::to_string(frameIndex));
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(frames_.size() * 3);

        shadowCullDescriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(&poolSize, 1), static_cast<uint32_t>(frames_.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  shadowCullDescriptorPool_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                  "GpuShadowCullDescriptorPool");

        shadowCullDescriptorSets_.resize(frames_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> setLayouts(frames_.size(), gpuCullDescriptorSetLayout_.handle());
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
            visibleCountBufferInfo.range = kBatchVisibleCountBufferSize;

            std::array<VkWriteDescriptorSet, 3> writes{};
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

            vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      shadowCullDescriptorSets_[frameIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "GpuShadowCullDescriptorSet" + std::to_string(frameIndex));
        }

        gpuShadowCullingAvailable_ = true;
        Logger::info("GPU shadow culling preparation enabled with per-frame shadow cull input, compacted indirect "
                     "output, and per-batch visible count buffers.");
    } catch (const std::exception& error) {
        Logger::warn(std::string("GPU shadow culling unavailable; using CPU shadow culling fallback: ") +
                     error.what());
        destroyGpuShadowCullingResources();
    }
}

void Renderer::destroyGpuShadowCullingResources()
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

void Renderer::createShadowMap()
{
    // The CSM depth array is fixed-size for now and intentionally independent
    // of swapchain resize; only the main color/depth targets follow the window extent.
    shadowMap_.create(context_, shadowSettings_.resolution, shadowSettings_.resolution, activeCascadeCount());
}

void Renderer::createPipeline()
{
    const VkVertexInputBindingDescription binding = renderer::vertexBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 5> attributes = renderer::vertexAttributeDescriptions();
    const bool bindlessMaterialTexturesActive = isBindlessMaterialTextureActive();
    std::array<VkDescriptorSetLayout, 2> mainDescriptorSetLayouts{
        materialDescriptorSetLayout_.handle(),
        bindlessTextureHeap_.descriptorSetLayout(),
    };
    const VkDescriptorSetLayout skyboxDescriptorSetLayout = skyboxDescriptorSetLayout_.handle();
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};
    const VkPushConstantRange skyboxPushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(sizeof(SkyboxPushConstants))};

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple.vert.spv");
    pipelineInfo.fragmentShaderPath =
        bindlessMaterialTexturesActive ? shaderPath("simple_bindless.frag.spv") : shaderPath("simple.frag.spv");
    pipelineInfo.colorFormat = swapchain_.colorFormat();
    pipelineInfo.depthFormat = swapchain_.depthFormat();
    pipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    pipelineInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(attributes.data(), attributes.size());
    pipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(mainDescriptorSetLayouts.data(), bindlessMaterialTexturesActive ? 2 : 1);
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.enableDepth = true;

    pipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "MainGraphicsPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "MainPipelineLayout");
    pipelineColorFormat_ = pipelineInfo.colorFormat;
    pipelineDepthFormat_ = pipelineInfo.depthFormat;

    rhi::VulkanPipelineCreateInfo skyboxPipelineInfo{};
    skyboxPipelineInfo.vertexShaderPath = shaderPath("skybox.vert.spv");
    skyboxPipelineInfo.fragmentShaderPath = shaderPath("skybox.frag.spv");
    skyboxPipelineInfo.colorFormat = swapchain_.colorFormat();
    skyboxPipelineInfo.depthFormat = swapchain_.depthFormat();
    skyboxPipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&skyboxDescriptorSetLayout, 1);
    skyboxPipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&skyboxPushConstantRange, 1);
    skyboxPipelineInfo.enableDepth = true;
    skyboxPipelineInfo.depthWriteEnable = false;
    skyboxPipelineInfo.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    skyboxPipelineInfo.cullMode = VK_CULL_MODE_NONE;

    skyboxPipeline_.create(context_.vkDevice(), skyboxPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), skyboxPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "SkyboxPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), skyboxPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "SkyboxPipelineLayout");
    skyboxPipelineColorFormat_ = skyboxPipelineInfo.colorFormat;
    skyboxPipelineDepthFormat_ = skyboxPipelineInfo.depthFormat;

    rhi::VulkanPipelineCreateInfo shadowPipelineInfo{};
    shadowPipelineInfo.vertexShaderPath = shaderPath("shadow.vert.spv");
    shadowPipelineInfo.depthFormat = shadowMap_.format();
    shadowPipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    shadowPipelineInfo.vertexAttributes = std::span<const VkVertexInputAttributeDescription>(attributes.data(), 1);
    shadowPipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    shadowPipelineInfo.enableColorAttachment = false;
    shadowPipelineInfo.enableDepth = true;
    shadowPipelineInfo.depthWriteEnable = true;
    // Static raster depth bias offsets shadow caster depth to reduce shadow acne.
    // Bias is a tradeoff: too much separation causes peter panning.
    shadowPipelineInfo.enableDepthBias = true;
    shadowPipelineInfo.cullMode = VK_CULL_MODE_NONE;
    shadowPipelineInfo.depthBiasConstantFactor = shadowSettings_.rasterDepthBiasConstantFactor;
    shadowPipelineInfo.depthBiasSlopeFactor = shadowSettings_.rasterDepthBiasSlopeFactor;

    shadowPipeline_.create(context_.vkDevice(), shadowPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), shadowPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "ShadowPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), shadowPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "ShadowPipelineLayout");
    shadowPipelineDepthFormat_ = shadowPipelineInfo.depthFormat;
}

void Renderer::createScene()
{
    renderObjects_.clear();
    allDrawItems_.clear();
    visibleDrawItems_.clear();
    shadowDrawItems_.clear();
    shadowMeshDrawBatches_.clear();
    gpuShadowMeshDrawBatches_.clear();
    for (std::vector<DrawItem>& cascadeDrawItems : shadowCascadeDrawItems_) {
        cascadeDrawItems.clear();
    }
    for (std::vector<MeshDrawBatch>& cascadeBatches : shadowCascadeMeshDrawBatches_) {
        cascadeBatches.clear();
    }
    shadowVisibleDrawItemsPerCascade_.fill(0);
    shadowBatchCountPerCascade_.fill(0);
    cullingStats_ = {};
    shadowCullingStats_ = {};
    importedMeshes_.clear();
    importedMaterials_.clear();
    importedTextures_.clear();

    cubeMesh_ = renderer::Mesh::createCube(context_, commandContext_);
    createCheckerboardTexture();
    createNormalTexture();
    createMetallicRoughnessTexture();
    createEnvironmentMap();
    createMaterial();

    camera_.position = {0.0f, 0.35f, 5.5f};
    camera_.target = {0.0f, 0.1f, 0.0f};

    const auto addCube = [this](std::string debugName,
                                const renderer::Material* material,
                                const glm::vec3& position,
                                const glm::vec3& rotationRadians,
                                const glm::vec3& scale) {
        renderer::RenderObject cube{};
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = true;
        renderObjects_.push_back(std::move(cube));
    };

    const auto addCubeFallbackScene = [&addCube, this]() {
        renderObjects_.reserve(4);
        addCube(
            "Center Cube", &materialVariants_.at(0), {0.0f, -0.1f, 0.0f}, {0.2f, 0.0f, 0.0f}, {0.7f, 0.7f, 0.7f});
        addCube(
            "Left Cube", &materialVariants_.at(1), {-1.35f, -0.15f, -0.35f}, {0.0f, 0.35f, 0.2f}, {0.5f, 0.5f, 0.5f});
        addCube("Right Cube",
                &materialVariants_.at(2),
                {1.35f, -0.05f, -0.25f},
                {0.25f, -0.35f, 0.0f},
                {0.55f, 0.8f, 0.55f});
        addCube("Elevated Cube",
                &materialVariants_.at(3),
                {0.0f, 1.0f, -0.7f},
                {-0.3f, 0.2f, 0.45f},
                {0.45f, 0.45f, 0.45f});
    };

    const std::array<std::filesystem::path, 2> modelCandidates = {
        assetPath("models/test_mesh.gltf"),
        assetPath("models/test_mesh.glb"),
    };

    for (const std::filesystem::path& modelPath : modelCandidates) {
        if (!std::filesystem::exists(modelPath)) {
            continue;
        }

        try {
            renderer::LoadedGltfAsset loadedAsset =
                renderer::Mesh::createFromGltf(context_, commandContext_, modelPath);
            createImportedGltfTextures(loadedAsset.textures);
            createImportedGltfMaterials(loadedAsset.materials);
            importedMeshes_ = std::move(loadedAsset.meshes);

            renderObjects_.reserve(loadedAsset.nodeMeshInstances.size());
            for (const renderer::GltfNodeMeshInstance& instance : loadedAsset.nodeMeshInstances) {
                if (instance.meshIndex >= importedMeshes_.size() || !importedMeshes_[instance.meshIndex].valid()) {
                    Logger::warn("Skipping imported glTF RenderObject with invalid mesh index " +
                                 std::to_string(instance.meshIndex) + ".");
                    continue;
                }

                renderer::RenderObject importedObject{};
                importedObject.mesh = &importedMeshes_[instance.meshIndex];
                importedObject.material =
                    importedMaterials_.empty() ? &materialVariants_.at(0) : &importedMaterials_.front();
                if (!importedMaterials_.empty()) {
                    importedObject.materialTable = importedMaterials_.data();
                    importedObject.materialCount = importedMaterials_.size();
                }
                importedObject.debugName =
                    instance.debugName.empty() ? "Imported glTF Node" : instance.debugName;
                importedObject.transform = renderer::Transform::fromMatrix(instance.transform);
                renderObjects_.push_back(std::move(importedObject));
            }

            if (renderObjects_.empty()) {
                throw std::runtime_error("Loaded glTF asset did not produce any valid RenderObjects.");
            }

            Logger::info("Loaded glTF scene: " + modelPath.string() + " with " +
                         std::to_string(importedMeshes_.size()) + " mesh slot(s), " +
                         std::to_string(renderObjects_.size()) + " render object(s), and " +
                         std::to_string(importedMaterials_.size()) + " material(s).");
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load glTF mesh '" + modelPath.string() + "': " + error.what());
        }
    }

    Logger::warn("No supported glTF mesh asset loaded; using built-in cube fallback scene.");
    addCubeFallbackScene();
}

void Renderer::createCheckerboardTexture()
{
    const std::filesystem::path texturePath = assetPath("textures/checker.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            checkerboardTexture_.createFromFile(context_, commandContext_, texturePath, true);
            nameTextureResources(checkerboardTexture_, "BaseColorTexture");
            Logger::info("Loaded texture: " + texturePath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Texture asset missing, using procedural checkerboard fallback: " + texturePath.string());
    }

    checkerboardTexture_.createCheckerboard(context_, commandContext_, 256, 256);
    nameTextureResources(checkerboardTexture_, "BaseColorTexture");
}

void Renderer::createNormalTexture()
{
    normalMapAssetLoaded_ = false;
    bool loadedAsset = false;

    const std::filesystem::path texturePath = assetPath("textures/checker_normal.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            normalMapTexture_.createFromFile(context_, commandContext_, texturePath, true);
            normalMapAssetLoaded_ = true;
            nameTextureResources(normalMapTexture_, "NormalTexture");
            Logger::info("Loaded normal texture: " + texturePath.string());
            loadedAsset = true;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load normal texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Normal texture asset missing, using procedural flat normal fallback: " + texturePath.string());
    }

    if (!loadedAsset) {
        constexpr uint32_t width = 4;
        constexpr uint32_t height = 4;
        std::array<uint8_t, width * height * 4> pixels{};
        for (size_t offset = 0; offset < pixels.size(); offset += 4) {
            pixels[offset + 0] = 128;
            pixels[offset + 1] = 128;
            pixels[offset + 2] = 255;
            pixels[offset + 3] = 255;
        }

        normalMapTexture_.createFromRgba8(context_,
                                          commandContext_,
                                          width,
                                          height,
                                          std::span<const uint8_t>(pixels.data(), pixels.size()),
                                          VK_FORMAT_R8G8B8A8_UNORM,
                                          false);
        nameTextureResources(normalMapTexture_, "NormalTexture");
    }

    createFlatNormalTexture();
}

void Renderer::createFlatNormalTexture()
{
    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 128;
        pixels[offset + 1] = 128;
        pixels[offset + 2] = 255;
        pixels[offset + 3] = 255;
    }

    flatNormalTexture_.createFromRgba8(context_,
                                       commandContext_,
                                       width,
                                       height,
                                       std::span<const uint8_t>(pixels.data(), pixels.size()),
                                       VK_FORMAT_R8G8B8A8_UNORM,
                                       false);
    nameTextureResources(flatNormalTexture_, "FlatNormalTexture");
}

void Renderer::createMetallicRoughnessTexture()
{
    metallicRoughnessMapAssetLoaded_ = false;
    bool loadedAsset = false;

    const std::filesystem::path texturePath = assetPath("textures/checker_mr.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            metallicRoughnessTexture_.createFromFile(context_, commandContext_, texturePath, true);
            metallicRoughnessMapAssetLoaded_ = true;
            nameTextureResources(metallicRoughnessTexture_, "MetallicRoughnessTexture");
            Logger::info("Loaded metallic-roughness texture: " + texturePath.string());
            loadedAsset = true;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load metallic-roughness texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Metallic-roughness texture asset missing, using procedural neutral fallback: " +
                     texturePath.string());
    }

    if (!loadedAsset) {
        constexpr uint32_t width = 4;
        constexpr uint32_t height = 4;
        std::array<uint8_t, width * height * 4> pixels{};
        for (size_t offset = 0; offset < pixels.size(); offset += 4) {
            pixels[offset + 0] = 255;
            pixels[offset + 1] = 255;
            pixels[offset + 2] = 0;
            pixels[offset + 3] = 255;
        }

        metallicRoughnessTexture_.createFromRgba8(context_,
                                                  commandContext_,
                                                  width,
                                                  height,
                                                  std::span<const uint8_t>(pixels.data(), pixels.size()),
                                                  VK_FORMAT_R8G8B8A8_UNORM,
                                                  false);
        nameTextureResources(metallicRoughnessTexture_, "MetallicRoughnessTexture");
    }

    createNeutralMetallicRoughnessTexture();
}

void Renderer::createNeutralMetallicRoughnessTexture()
{
    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 255;
        pixels[offset + 1] = 255;
        pixels[offset + 2] = 0;
        pixels[offset + 3] = 255;
    }

    neutralMetallicRoughnessTexture_.createFromRgba8(context_,
                                                     commandContext_,
                                                     width,
                                                     height,
                                                     std::span<const uint8_t>(pixels.data(), pixels.size()),
                                                     VK_FORMAT_R8G8B8A8_UNORM,
                                                     false);
    nameTextureResources(neutralMetallicRoughnessTexture_, "NeutralMetallicRoughnessTexture");
}

void Renderer::createEnvironmentMap()
{
    // The visible environment cubemap stays dedicated to the skybox. A separate
    // low-frequency cubemap feeds diffuse IBL, while a mipmapped cubemap and 2D
    // BRDF LUT feed split-sum specular IBL.
    environmentMap_.createProcedural(context_, commandContext_, 32);
    nameEnvironmentMapResources(environmentMap_, "EnvironmentCubemap");
    createDiffuseIrradianceMap();
    createPrefilteredEnvironmentMap();
    createBrdfLutTexture();
    createSkyboxDescriptorSet();
    Logger::info("Created procedural environment cubemaps and BRDF LUT for skybox, diffuse IBL, "
                 "and specular IBL.");
}

void Renderer::createDiffuseIrradianceMap()
{
    try {
        diffuseIrradianceMap_.createProceduralDiffuseIrradiance(context_, commandContext_, 32);
        nameEnvironmentMapResources(diffuseIrradianceMap_, "DiffuseIrradianceCubemap");
        return;
    } catch (const std::exception& error) {
        Logger::warn(std::string("Failed to create procedural diffuse irradiance cubemap, using "
                                 "neutral fallback: ") +
                     error.what());
    }

    std::array<uint8_t, 6 * 4> neutralPixels{};
    for (size_t offset = 0; offset < neutralPixels.size(); offset += 4) {
        neutralPixels[offset + 0] = 80;
        neutralPixels[offset + 1] = 80;
        neutralPixels[offset + 2] = 80;
        neutralPixels[offset + 3] = 255;
    }

    diffuseIrradianceMap_.createFromRgba8Faces(context_,
                                               commandContext_,
                                               1,
                                               std::span<const uint8_t>(neutralPixels.data(), neutralPixels.size()),
                                               VK_FORMAT_R8G8B8A8_UNORM);
    nameEnvironmentMapResources(diffuseIrradianceMap_, "DiffuseIrradianceCubemap");
}

void Renderer::createPrefilteredEnvironmentMap()
{
    try {
        prefilteredEnvironmentMap_.createProceduralPrefilteredSpecular(context_, commandContext_, 64);
        nameEnvironmentMapResources(prefilteredEnvironmentMap_, "PrefilteredSpecularCubemap");
        return;
    } catch (const std::exception& error) {
        Logger::warn(std::string("Failed to create procedural prefiltered specular cubemap, using "
                                 "neutral fallback: ") +
                     error.what());
    }

    std::array<uint8_t, 6 * 4> neutralPixels{};
    for (size_t offset = 0; offset < neutralPixels.size(); offset += 4) {
        neutralPixels[offset + 0] = 96;
        neutralPixels[offset + 1] = 96;
        neutralPixels[offset + 2] = 96;
        neutralPixels[offset + 3] = 255;
    }

    prefilteredEnvironmentMap_.createFromRgba8Faces(
        context_,
        commandContext_,
        1,
        std::span<const uint8_t>(neutralPixels.data(), neutralPixels.size()),
        VK_FORMAT_R8G8B8A8_UNORM);
    nameEnvironmentMapResources(prefilteredEnvironmentMap_, "PrefilteredSpecularCubemap");
}

void Renderer::createBrdfLutTexture()
{
    brdfLutTexture_.create(context_, commandContext_, 256);
    nameBrdfLutResources(brdfLutTexture_, "BrdfLut");
}

void Renderer::createMaterial()
{
    materialVariants_.clear();
    materialVariants_.reserve(4);

    if (isBindlessMaterialTextureActive()) {
        bindlessBaseColorFallbackIndex_ = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::BaseColor, checkerboardTexture_);
        bindlessNormalFallbackIndex_ = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::Normal, flatNormalTexture_);
        bindlessMetallicRoughnessFallbackIndex_ = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::MetallicRoughness, neutralMetallicRoughnessTexture_);
    }

    const auto addMaterial = [this](std::string debugName,
                                    const glm::vec4& baseColorFactor,
                                    float metallic,
                                    float roughness,
                                    float multiScatterStrength) {
        renderer::Material material{};
        material.debugName = std::move(debugName);
        material.baseColorTexture = &checkerboardTexture_;
        material.normalTexture = &normalMapTexture_;
        material.metallicRoughnessTexture = &metallicRoughnessTexture_;
        material.baseColorFactor = baseColorFactor;
        material.metallic = metallic;
        material.roughness = roughness;
        material.multiScatterStrength = multiScatterStrength;
        material.hasNormalMap = normalMapAssetLoaded_;
        material.hasMetallicRoughnessMap = metallicRoughnessMapAssetLoaded_;
        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        materialVariants_.push_back(std::move(material));
    };

    addMaterial("Checkerboard Matte", {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.75f, 0.0f);
    addMaterial("Checkerboard Warm Semi-Metal", {1.0f, 0.82f, 0.65f, 1.0f}, 0.35f, 0.38f, 0.5f);
    addMaterial("Checkerboard Cool Rough Metal", {0.72f, 0.84f, 1.0f, 1.0f}, 0.85f, 0.62f, 1.0f);
    addMaterial("Checkerboard Glossy Dielectric", {0.9f, 1.0f, 0.78f, 1.0f}, 0.0f, 0.18f, 0.25f);

    checkerboardMaterial_ = materialVariants_.front();
}

void Renderer::assignBindlessTextureIndices(renderer::Material& material)
{
    if (!isBindlessMaterialTextureActive()) {
        return;
    }

    material.baseColorTextureIndex =
        material.baseColorTexture && material.baseColorTexture->valid()
            ? bindlessTextureHeap_.registerTexture(renderer::BindlessTextureHeap::TextureKind::BaseColor,
                                                   *material.baseColorTexture)
            : bindlessBaseColorFallbackIndex_;
    material.normalTextureIndex =
        material.normalTexture && material.normalTexture->valid()
            ? bindlessTextureHeap_.registerTexture(renderer::BindlessTextureHeap::TextureKind::Normal,
                                                   *material.normalTexture)
            : bindlessNormalFallbackIndex_;
    material.metallicRoughnessTextureIndex =
        material.metallicRoughnessTexture && material.metallicRoughnessTexture->valid()
            ? bindlessTextureHeap_.registerTexture(renderer::BindlessTextureHeap::TextureKind::MetallicRoughness,
                                                   *material.metallicRoughnessTexture)
            : bindlessMetallicRoughnessFallbackIndex_;
}

void Renderer::createMaterialDescriptorSet(renderer::Material& material)
{
    if (!material.baseColorTexture || !material.baseColorTexture->valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid base color texture.");
    }
    if (!material.normalTexture || !material.normalTexture->valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid normal texture.");
    }
    if (!material.metallicRoughnessTexture || !material.metallicRoughnessTexture->valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid metallic-roughness texture.");
    }
    if (!shadowMap_.valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid shadow map.");
    }
    if (!diffuseIrradianceMap_.valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid diffuse irradiance map.");
    }
    if (!prefilteredEnvironmentMap_.valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid prefiltered environment map.");
    }
    if (!brdfLutTexture_.valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid BRDF LUT texture.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = kMaxMaterialDescriptorSets * 7;

    if (materialDescriptorPool_.handle() == VK_NULL_HANDLE) {
        materialDescriptorPool_.create(
            context_.vkDevice(), std::span<const VkDescriptorPoolSize>(&poolSize, 1), kMaxMaterialDescriptorSets);
    }

    const VkDescriptorSetLayout descriptorSetLayout = materialDescriptorSetLayout_.handle();
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = materialDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, &material.descriptorSet));

    VkDescriptorImageInfo baseColorInfo{};
    baseColorInfo.sampler = material.baseColorTexture->sampler();
    baseColorInfo.imageView = material.baseColorTexture->imageView();
    baseColorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowInfo{};
    shadowInfo.sampler = shadowMap_.sampler();
    shadowInfo.imageView = shadowMap_.imageView();
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalInfo{};
    normalInfo.sampler = material.normalTexture->sampler();
    normalInfo.imageView = material.normalTexture->imageView();
    normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo metallicRoughnessInfo{};
    metallicRoughnessInfo.sampler = material.metallicRoughnessTexture->sampler();
    metallicRoughnessInfo.imageView = material.metallicRoughnessTexture->imageView();
    metallicRoughnessInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo diffuseIrradianceInfo{};
    diffuseIrradianceInfo.sampler = diffuseIrradianceMap_.sampler();
    diffuseIrradianceInfo.imageView = diffuseIrradianceMap_.imageView();
    diffuseIrradianceInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo prefilteredEnvironmentInfo{};
    prefilteredEnvironmentInfo.sampler = prefilteredEnvironmentMap_.sampler();
    prefilteredEnvironmentInfo.imageView = prefilteredEnvironmentMap_.imageView();
    prefilteredEnvironmentInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo brdfLutInfo{};
    brdfLutInfo.sampler = brdfLutTexture_.sampler();
    brdfLutInfo.imageView = brdfLutTexture_.imageView();
    brdfLutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 7> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = material.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &baseColorInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = material.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &shadowInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = material.descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].dstArrayElement = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &normalInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = material.descriptorSet;
    writes[3].dstBinding = 3;
    writes[3].dstArrayElement = 0;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &metallicRoughnessInfo;

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = material.descriptorSet;
    writes[4].dstBinding = 4;
    writes[4].dstArrayElement = 0;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &diffuseIrradianceInfo;

    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = material.descriptorSet;
    writes[5].dstBinding = 5;
    writes[5].dstArrayElement = 0;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &prefilteredEnvironmentInfo;

    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = material.descriptorSet;
    writes[6].dstBinding = 6;
    writes[6].dstArrayElement = 0;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = &brdfLutInfo;

    // The material descriptor stores sampled images only: base color at binding 0,
    // cascaded shadow-map array at binding 1, normal map at binding 2, and metallic-roughness
    // map at binding 3. Bindings 4-6 are diffuse irradiance, prefiltered specular
    // environment, and the BRDF LUT. Object data remains outside descriptors.
    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::createImportedGltfTextures(const std::vector<renderer::GltfTextureInfo>& textureInfos)
{
    importedTextures_.clear();
    importedTextures_.resize(textureInfos.size());

    for (size_t textureIndex = 0; textureIndex < textureInfos.size(); ++textureIndex) {
        const renderer::GltfTextureInfo& textureInfo = textureInfos[textureIndex];
        if (textureInfo.path.empty() && textureInfo.encodedData.empty()) {
            continue;
        }

        try {
            if (!textureInfo.path.empty()) {
                if (!std::filesystem::exists(textureInfo.path)) {
                    Logger::warn("glTF texture image is missing; material fallback will be used: " +
                                 textureInfo.path.string());
                    continue;
                }

                importedTextures_[textureIndex].createFromFile(context_, commandContext_, textureInfo.path, true);
                Logger::info("Loaded glTF texture: " + textureInfo.path.string());
            } else {
                importedTextures_[textureIndex].createFromEncodedBytes(
                    context_,
                    commandContext_,
                    std::span<const uint8_t>(textureInfo.encodedData.data(), textureInfo.encodedData.size()),
                    true);
                Logger::info("Loaded embedded glTF texture: " + textureInfo.debugName);
            }

            nameTextureResources(importedTextures_[textureIndex], "GltfTexture" + std::to_string(textureIndex));
        } catch (const std::exception& error) {
            const std::string textureName =
                !textureInfo.path.empty() ? textureInfo.path.string() : textureInfo.debugName;
            Logger::warn("Failed to load glTF texture '" + textureName +
                         "'; material fallback will be used: " + error.what());
        }
    }
}

void Renderer::createImportedGltfMaterials(const std::vector<renderer::GltfMaterialInfo>& materialInfos)
{
    std::vector<renderer::GltfMaterialInfo> defaultMaterialInfos;
    const std::vector<renderer::GltfMaterialInfo>* sourceMaterialInfos = &materialInfos;
    if (materialInfos.empty()) {
        renderer::GltfMaterialInfo defaultMaterial{};
        defaultMaterial.debugName = "Default glTF Material";
        defaultMaterialInfos.push_back(std::move(defaultMaterial));
        sourceMaterialInfos = &defaultMaterialInfos;
    }

    importedMaterials_.clear();
    importedMaterials_.reserve(sourceMaterialInfos->size());

    const auto textureOrFallback = [this](int textureIndex,
                                          const rhi::VulkanTexture& fallbackTexture) -> const rhi::VulkanTexture* {
        if (textureIndex >= 0 && static_cast<size_t>(textureIndex) < importedTextures_.size() &&
            importedTextures_[static_cast<size_t>(textureIndex)].valid()) {
            return &importedTextures_[static_cast<size_t>(textureIndex)];
        }
        return &fallbackTexture;
    };

    const auto textureLoaded = [this](int textureIndex) {
        return textureIndex >= 0 && static_cast<size_t>(textureIndex) < importedTextures_.size() &&
               importedTextures_[static_cast<size_t>(textureIndex)].valid();
    };

    for (const renderer::GltfMaterialInfo& materialInfo : *sourceMaterialInfos) {
        renderer::Material material{};
        material.debugName = materialInfo.debugName.empty() ? "glTF Material" : materialInfo.debugName;
        material.baseColorTexture = textureOrFallback(materialInfo.baseColorTextureIndex, checkerboardTexture_);
        material.normalTexture = textureOrFallback(materialInfo.normalTextureIndex, flatNormalTexture_);
        material.metallicRoughnessTexture =
            textureOrFallback(materialInfo.metallicRoughnessTextureIndex, neutralMetallicRoughnessTexture_);
        material.baseColorFactor = materialInfo.baseColorFactor;
        material.metallic = materialInfo.metallic;
        material.roughness = materialInfo.roughness;
        material.multiScatterStrength = 1.0f;
        material.hasNormalMap = textureLoaded(materialInfo.normalTextureIndex);
        material.hasMetallicRoughnessMap = textureLoaded(materialInfo.metallicRoughnessTextureIndex);

        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        importedMaterials_.push_back(std::move(material));
    }
}

void Renderer::createSkyboxDescriptorSet()
{
    if (!environmentMap_.valid()) {
        throw std::runtime_error("Cannot create a skybox descriptor set without a valid environment map.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    skyboxDescriptorSet_ = VK_NULL_HANDLE;
    skyboxDescriptorPool_.create(context_.vkDevice(), std::span<const VkDescriptorPoolSize>(&poolSize, 1), 1);

    const VkDescriptorSetLayout descriptorSetLayout = skyboxDescriptorSetLayout_.handle();
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = skyboxDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, &skyboxDescriptorSet_));

    VkDescriptorImageInfo environmentInfo{};
    environmentInfo.sampler = environmentMap_.sampler();
    environmentInfo.imageView = environmentMap_.imageView();
    environmentInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = skyboxDescriptorSet_;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &environmentInfo;

    vkUpdateDescriptorSets(context_.vkDevice(), 1, &write, 0, nullptr);
}

void Renderer::createObjectFrameDataBuffers()
{
    frameObjectDataBuffers_.resize(frames_.size());

    for (size_t frameIndex = 0; frameIndex < frameObjectDataBuffers_.size(); ++frameIndex) {
        rhi::VulkanBuffer& frameObjectDataBuffer = frameObjectDataBuffers_[frameIndex];
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxDrawItems * sizeof(ObjectFrameData));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        bufferInfo.requestDeviceAddress = true;
        frameObjectDataBuffer.createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameObjectDataBuffer.buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "ObjectFrameDataBuffer" + std::to_string(frameIndex));
    }
}

void Renderer::createIndirectDrawBuffers()
{
    frameIndirectDrawBuffers_.resize(frames_.size());

    for (size_t frameIndex = 0; frameIndex < frameIndirectDrawBuffers_.size(); ++frameIndex) {
        rhi::VulkanBuffer& indirectDrawBuffer = frameIndirectDrawBuffers_[frameIndex];
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxDrawItems * sizeof(VkDrawIndexedIndirectCommand));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        indirectDrawBuffer.createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  indirectDrawBuffer.buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "IndirectDrawCommandBuffer" + std::to_string(frameIndex));
    }
}

void Renderer::createShadowIndirectDrawBuffers()
{
    shadowIndirectAvailable_ = false;
    frameShadowIndirectDrawBuffers_.clear();

    try {
        frameShadowIndirectDrawBuffers_.resize(frames_.size());

        for (size_t frameIndex = 0; frameIndex < frameShadowIndirectDrawBuffers_.size(); ++frameIndex) {
            rhi::VulkanBuffer& indirectDrawBuffer = frameShadowIndirectDrawBuffers_[frameIndex];
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = static_cast<VkDeviceSize>(kMaxDrawItems * sizeof(VkDrawIndexedIndirectCommand));
            bufferInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            indirectDrawBuffer.createBuffer(context_, bufferInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      indirectDrawBuffer.buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "ShadowIndirectDrawCommandBuffer" + std::to_string(frameIndex));
        }

        shadowIndirectAvailable_ = true;
    } catch (const std::exception& error) {
        Logger::warn(std::string("Shadow indirect command buffers unavailable; falling back to direct shadow draws: ") +
                     error.what());
        frameShadowIndirectDrawBuffers_.clear();
    }
}

uint32_t Renderer::activeCascadeCount() const
{
    return std::clamp(csmSettings_.cascadeCount, 1U, kMaxShadowCascades);
}

void Renderer::updateCascades(float aspectRatio)
{
    const uint32_t cascadeCount = activeCascadeCount();
    const float nearPlane = std::max(0.001f, csmSettings_.nearPlane);
    const float cameraFarPlane = std::max(nearPlane + 0.001f, csmSettings_.farPlane);
    const float shadowFarPlane = std::clamp(csmSettings_.shadowDistance, nearPlane + 0.001f, cameraFarPlane);
    const float lambda = std::clamp(csmSettings_.lambda, 0.0f, 1.0f);

    const glm::vec3 cameraPosition = camera_.position;
    const glm::vec3 cameraForward = glm::normalize(camera_.target - camera_.position);
    const glm::vec3 cameraRight = glm::normalize(glm::cross(cameraForward, camera_.up));
    const glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, cameraForward));
    const float tanHalfFov = std::tan(camera_.verticalFovRadians * 0.5f);

    const glm::vec3 lightDirection = glm::normalize(
        glm::vec3{kDirectionalLightDirection.x, kDirectionalLightDirection.y, kDirectionalLightDirection.z});
    const glm::vec3 lightUp =
        std::abs(glm::dot(lightDirection, glm::vec3{0.0f, 1.0f, 0.0f})) > 0.95f
            ? glm::vec3{0.0f, 0.0f, 1.0f}
            : glm::vec3{0.0f, 1.0f, 0.0f};

    frameCascadeSplits_ = glm::vec4(shadowFarPlane);

    float cascadeNear = nearPlane;
    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        const float splitRatio = static_cast<float>(cascadeIndex + 1) / static_cast<float>(cascadeCount);
        const float uniformSplit = nearPlane + (shadowFarPlane - nearPlane) * splitRatio;
        const float logSplit = nearPlane * std::pow(shadowFarPlane / nearPlane, splitRatio);
        const float cascadeFar = glm::mix(uniformSplit, logSplit, lambda);
        frameCascadeSplits_[cascadeIndex] = cascadeFar;

        std::array<glm::vec3, 8> corners{};
        const auto writeDepthCorners = [&](float depth, uint32_t baseIndex) {
            const float halfHeight = tanHalfFov * depth;
            const float halfWidth = halfHeight * aspectRatio;
            const glm::vec3 center = cameraPosition + cameraForward * depth;

            corners[baseIndex + 0] = center - cameraRight * halfWidth - cameraUp * halfHeight;
            corners[baseIndex + 1] = center + cameraRight * halfWidth - cameraUp * halfHeight;
            corners[baseIndex + 2] = center - cameraRight * halfWidth + cameraUp * halfHeight;
            corners[baseIndex + 3] = center + cameraRight * halfWidth + cameraUp * halfHeight;
        };

        // Each cascade fits the camera-frustum slice between cascadeNear and
        // cascadeFar. This is intentionally readable; texel snapping and caster
        // expansion are left as follow-up stability work.
        writeDepthCorners(cascadeNear, 0);
        writeDepthCorners(cascadeFar, 4);

        glm::vec3 cascadeCenter{0.0f};
        for (const glm::vec3& corner : corners) {
            cascadeCenter += corner;
        }
        cascadeCenter /= static_cast<float>(corners.size());

        const glm::mat4 lightView = glm::lookAt(cascadeCenter - lightDirection, cascadeCenter, lightUp);
        glm::vec3 minBounds{std::numeric_limits<float>::infinity()};
        glm::vec3 maxBounds{-std::numeric_limits<float>::infinity()};
        for (const glm::vec3& corner : corners) {
            const glm::vec3 lightSpaceCorner = glm::vec3(lightView * glm::vec4(corner, 1.0f));
            minBounds = glm::min(minBounds, lightSpaceCorner);
            maxBounds = glm::max(maxBounds, lightSpaceCorner);
        }

        const float depthRange = std::max(cascadeFar - cascadeNear, 1.0f);
        const float zPadding = std::max(depthRange * 2.0f, 10.0f);
        const float orthoNear = std::max(0.001f, -maxBounds.z - zPadding);
        const float orthoFar = std::max(orthoNear + 0.001f, -minBounds.z + zPadding);

        glm::mat4 lightProjection =
            glm::ortho(minBounds.x, maxBounds.x, minBounds.y, maxBounds.y, orthoNear, orthoFar);
        lightProjection[1][1] *= -1.0f;

        CascadeFrameData& cascade = frameCascades_[cascadeIndex];
        cascade.lightViewProjection = lightProjection * lightView;
        cascade.lightFrustum = renderer::Frustum::fromViewProjection(cascade.lightViewProjection);
        cascade.splitDepth = cascadeFar;
        cascade.nearDepth = cascadeNear;
        cascade.farDepth = cascadeFar;

        for (size_t planeIndex = 0; planeIndex < frameShadowCascadeFrustumPlanes_[cascadeIndex].size(); ++planeIndex) {
            const renderer::FrustumPlane& lightPlane = cascade.lightFrustum.planes[planeIndex];
            frameShadowCascadeFrustumPlanes_[cascadeIndex][planeIndex] =
                glm::vec4(lightPlane.normal, lightPlane.distance);
        }

        cascadeNear = cascadeFar;
    }

    for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
        frameCascades_[cascadeIndex] = frameCascades_[cascadeCount - 1];
        frameCascadeSplits_[cascadeIndex] = shadowFarPlane;
        frameShadowCascadeFrustumPlanes_[cascadeIndex] = frameShadowCascadeFrustumPlanes_[cascadeCount - 1];
    }
}

const renderer::Material* Renderer::resolveMaterial(const renderer::RenderObject& object,
                                                    const renderer::MeshPrimitive* primitive) const
{
    if (primitive && object.materialTable && primitive->materialIndex < object.materialCount) {
        return &object.materialTable[primitive->materialIndex];
    }

    return object.material;
}

bool Renderer::appendDrawItemsForObject(uint32_t objectIndex, std::vector<DrawItem>& drawItems) const
{
    if (objectIndex >= renderObjects_.size()) {
        return true;
    }

    const renderer::RenderObject& object = renderObjects_[objectIndex];
    const renderer::Mesh* mesh = object.mesh;
    if (!mesh || !mesh->valid()) {
        return true;
    }

    if (mesh->hasSubMeshes()) {
        const std::span<const renderer::MeshPrimitive> primitives = mesh->primitives();
        for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
            if (drawItems.size() >= kMaxDrawItems) {
                return false;
            }

            const renderer::MeshPrimitive& primitive = primitives[primitiveIndex];
            if (primitive.indexCount == 0) {
                continue;
            }

            DrawItem drawItem{};
            drawItem.mesh = mesh;
            drawItem.material = resolveMaterial(object, &primitive);
            drawItem.objectIndex = objectIndex;
            drawItem.submeshIndex = static_cast<uint32_t>(primitiveIndex);
            drawItem.firstIndex = primitive.firstIndex;
            drawItem.indexCount = primitive.indexCount;
            drawItem.frameDataIndex = static_cast<uint32_t>(drawItems.size());
            drawItems.push_back(drawItem);
        }
        return true;
    }

    if (mesh->indexCount() == 0) {
        return true;
    }
    if (drawItems.size() >= kMaxDrawItems) {
        return false;
    }

    DrawItem drawItem{};
    drawItem.mesh = mesh;
    drawItem.material = object.material;
    drawItem.objectIndex = objectIndex;
    drawItem.indexCount = mesh->indexCount();
    drawItem.frameDataIndex = static_cast<uint32_t>(drawItems.size());
    drawItems.push_back(drawItem);
    return true;
}

void Renderer::buildDrawItems()
{
    allDrawItems_.clear();
    allDrawItems_.reserve(renderObjects_.size());

    const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        if (!appendDrawItemsForObject(static_cast<uint32_t>(objectIndex), allDrawItems_)) {
            break;
        }
    }

    std::stable_sort(allDrawItems_.begin(), allDrawItems_.end(), [](const DrawItem& lhs, const DrawItem& rhs) {
        return std::less<const renderer::Mesh*>{}(lhs.mesh, rhs.mesh);
    });

    for (size_t drawIndex = 0; drawIndex < allDrawItems_.size(); ++drawIndex) {
        allDrawItems_[drawIndex].frameDataIndex = static_cast<uint32_t>(drawIndex);
    }
}

void Renderer::buildVisibleDrawItems(const renderer::Frustum& frustum)
{
    visibleDrawItems_.clear();
    visibleDrawItems_.reserve(allDrawItems_.size());
    cullingStats_ = {};
    cullingStats_.totalDrawItems = allDrawItems_.size();

    const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
    std::vector<bool> objectVisible(objectCount, false);
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.mesh || !object.mesh->valid()) {
            continue;
        }

        ++cullingStats_.totalObjects;
        const renderer::Aabb worldBounds = object.worldBounds();
        if (worldBounds.valid() && !frustum.testAabb(worldBounds)) {
            ++cullingStats_.culledObjects;
            continue;
        }

        ++cullingStats_.visibleObjects;
        objectVisible[objectIndex] = true;
    }

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex]) {
            visibleDrawItems_.push_back(drawItem);
        }
    }
}

void Renderer::buildMeshDrawBatches()
{
    meshDrawBatches_.clear();
    cullingStats_.commandCount = visibleDrawItems_.size();
    buildMeshDrawBatchesForItems(visibleDrawItems_, meshDrawBatches_);
    cullingStats_.batchCount = meshDrawBatches_.size();
}

void Renderer::buildShadowDrawItems(uint32_t cascadeIndex, const renderer::Frustum& lightFrustum)
{
    if (cascadeIndex >= shadowCascadeDrawItems_.size()) {
        return;
    }

    std::vector<DrawItem>& cascadeDrawItems = shadowCascadeDrawItems_[cascadeIndex];
    cascadeDrawItems.clear();
    cascadeDrawItems.reserve(allDrawItems_.size());

    const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
    std::vector<bool> objectVisible(objectCount, false);
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.mesh || !object.mesh->valid()) {
            continue;
        }

        const renderer::Aabb worldBounds = object.worldBounds();
        if (worldBounds.valid() && !lightFrustum.testAabb(worldBounds)) {
            continue;
        }

        objectVisible[objectIndex] = true;
    }

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex]) {
            cascadeDrawItems.push_back(drawItem);
        }
    }
}

void Renderer::buildShadowMeshDrawBatches()
{
    shadowMeshDrawBatches_.clear();
    buildMeshDrawBatchesForItems(shadowDrawItems_, shadowMeshDrawBatches_);
    shadowCullingStats_.batchCount = shadowMeshDrawBatches_.size();
}

void Renderer::buildMeshDrawBatchesForItems(
    const std::vector<DrawItem>& drawItems,
    std::vector<MeshDrawBatch>& batches) const
{
    const renderer::Mesh* currentMesh = nullptr;
    MeshDrawBatch* currentBatch = nullptr;
    for (size_t drawItemIndex = 0; drawItemIndex < drawItems.size(); ++drawItemIndex) {
        const DrawItem& drawItem = drawItems[drawItemIndex];
        if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems) {
            currentMesh = nullptr;
            currentBatch = nullptr;
            continue;
        }

        if (!currentBatch || currentMesh != drawItem.mesh) {
            MeshDrawBatch batch{};
            batch.mesh = drawItem.mesh;
            batch.beginDrawItem = static_cast<uint32_t>(drawItemIndex);
            batch.compactedCommandOffset = static_cast<uint32_t>(drawItemIndex);
            batch.visibleCountOffset = static_cast<uint32_t>(batches.size() * sizeof(uint32_t));
            batches.push_back(batch);
            currentBatch = &batches.back();
            currentMesh = drawItem.mesh;
        }

        ++currentBatch->drawItemCount;
    }
}

void Renderer::updateGpuCullInputBuffer(uint32_t frameIndex)
{
    if (allDrawItems_.empty()) {
        return;
    }
    if (frameIndex >= frameCullInputBuffers_.size()) {
        throw std::runtime_error("GPU cull input buffer frame index is out of range.");
    }

    std::vector<GpuCullDrawItem> cullDrawItems(allDrawItems_.size());
    for (size_t drawIndex = 0; drawIndex < allDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = allDrawItems_[drawIndex];
        GpuCullDrawItem& gpuDrawItem = cullDrawItems[drawIndex];

        renderer::Aabb worldBounds{};
        if (drawItem.objectIndex < renderObjects_.size()) {
            worldBounds = renderObjects_[drawItem.objectIndex].worldBounds();
        }

        if (worldBounds.valid()) {
            gpuDrawItem.boundsMin = glm::vec4(worldBounds.min, 0.0f);
            gpuDrawItem.boundsMax = glm::vec4(worldBounds.max, 0.0f);
        } else {
            gpuDrawItem.boundsMin = glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
            gpuDrawItem.boundsMax = glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
        }

        gpuDrawItem.indexCount = drawItem.indexCount;
        gpuDrawItem.firstIndex = drawItem.firstIndex;
        gpuDrawItem.vertexOffset = drawItem.vertexOffset;
        gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;
    }

    for (size_t batchIndex = 0; batchIndex < meshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = meshDrawBatches_[batchIndex];
        const uint32_t endDrawItem = std::min<uint32_t>(
            batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
        for (uint32_t drawItemIndex = batch.beginDrawItem; drawItemIndex < endDrawItem; ++drawItemIndex) {
            cullDrawItems[drawItemIndex].batchIndex = static_cast<uint32_t>(batchIndex);
            cullDrawItems[drawItemIndex].batchOutputBase = batch.compactedCommandOffset;
        }
    }

    frameCullInputBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullDrawItem>(cullDrawItems.data(), cullDrawItems.size())));
}

void Renderer::updateGpuShadowCullInputBuffer(uint32_t frameIndex)
{
    if (allDrawItems_.empty()) {
        return;
    }
    if (frameIndex >= frameShadowCullInputBuffers_.size()) {
        throw std::runtime_error("GPU shadow cull input buffer frame index is out of range.");
    }

    std::vector<GpuCullDrawItem> cullDrawItems(allDrawItems_.size());
    for (size_t drawIndex = 0; drawIndex < allDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = allDrawItems_[drawIndex];
        GpuCullDrawItem& gpuDrawItem = cullDrawItems[drawIndex];

        renderer::Aabb worldBounds{};
        if (drawItem.objectIndex < renderObjects_.size()) {
            worldBounds = renderObjects_[drawItem.objectIndex].worldBounds();
        }

        if (worldBounds.valid()) {
            gpuDrawItem.boundsMin = glm::vec4(worldBounds.min, 0.0f);
            gpuDrawItem.boundsMax = glm::vec4(worldBounds.max, 0.0f);
        } else {
            gpuDrawItem.boundsMin = glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
            gpuDrawItem.boundsMax = glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
        }

        gpuDrawItem.indexCount = drawItem.indexCount;
        gpuDrawItem.firstIndex = drawItem.firstIndex;
        gpuDrawItem.vertexOffset = drawItem.vertexOffset;
        gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;
    }

    for (size_t batchIndex = 0; batchIndex < gpuShadowMeshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = gpuShadowMeshDrawBatches_[batchIndex];
        const uint32_t endDrawItem = std::min<uint32_t>(
            batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
        for (uint32_t drawItemIndex = batch.beginDrawItem; drawItemIndex < endDrawItem; ++drawItemIndex) {
            cullDrawItems[drawItemIndex].batchIndex = static_cast<uint32_t>(batchIndex);
            cullDrawItems[drawItemIndex].batchOutputBase = batch.compactedCommandOffset;
        }
    }

    frameShadowCullInputBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullDrawItem>(cullDrawItems.data(), cullDrawItems.size())));
}

void Renderer::updateIndirectDrawBuffer(uint32_t frameIndex)
{
    if (visibleDrawItems_.empty()) {
        return;
    }

    std::vector<VkDrawIndexedIndirectCommand> indirectCommands(visibleDrawItems_.size());
    const bool objectDataArrayIndexingActive = isMainPassMultiDrawIndirectActive();
    for (size_t drawIndex = 0; drawIndex < visibleDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = visibleDrawItems_[drawIndex];
        VkDrawIndexedIndirectCommand& command = indirectCommands[drawIndex];
        command.indexCount = drawItem.indexCount;
        command.instanceCount = 1;
        command.firstIndex = drawItem.firstIndex;
        command.vertexOffset = drawItem.vertexOffset;
        command.firstInstance = objectDataArrayIndexingActive ? drawItem.frameDataIndex : 0;
    }

    frameIndirectDrawBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const VkDrawIndexedIndirectCommand>(indirectCommands.data(),
                                                                            indirectCommands.size())));
}

void Renderer::updateShadowIndirectDrawBuffer(uint32_t frameIndex)
{
    if (!isShadowIndirectActive() || shadowDrawItems_.empty()) {
        return;
    }

    std::vector<VkDrawIndexedIndirectCommand> indirectCommands(shadowDrawItems_.size());
    for (size_t drawIndex = 0; drawIndex < shadowDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = shadowDrawItems_[drawIndex];
        VkDrawIndexedIndirectCommand& command = indirectCommands[drawIndex];
        command.indexCount = drawItem.indexCount;
        command.instanceCount = 1;
        command.firstIndex = drawItem.firstIndex;
        command.vertexOffset = drawItem.vertexOffset;
        command.firstInstance = drawItem.frameDataIndex;
    }

    frameShadowIndirectDrawBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const VkDrawIndexedIndirectCommand>(indirectCommands.data(),
                                                                            indirectCommands.size())));
}

void Renderer::nameTextureResources(const rhi::VulkanTexture& texture, std::string_view name) const
{
    if (!texture.valid()) {
        return;
    }

    const std::string prefix{name};
    rhi::debug::setObjectName(context_.vkDevice(), texture.image(), VK_OBJECT_TYPE_IMAGE, prefix + "Image");
    rhi::debug::setObjectName(context_.vkDevice(), texture.imageView(), VK_OBJECT_TYPE_IMAGE_VIEW, prefix + "View");
    rhi::debug::setObjectName(context_.vkDevice(), texture.sampler(), VK_OBJECT_TYPE_SAMPLER, prefix + "Sampler");
}

void Renderer::nameEnvironmentMapResources(const rhi::VulkanEnvironmentMap& environmentMap, std::string_view name) const
{
    if (!environmentMap.valid()) {
        return;
    }

    const std::string prefix{name};
    rhi::debug::setObjectName(context_.vkDevice(), environmentMap.image(), VK_OBJECT_TYPE_IMAGE, prefix + "Image");
    rhi::debug::setObjectName(
        context_.vkDevice(), environmentMap.imageView(), VK_OBJECT_TYPE_IMAGE_VIEW, prefix + "View");
    rhi::debug::setObjectName(
        context_.vkDevice(), environmentMap.sampler(), VK_OBJECT_TYPE_SAMPLER, prefix + "Sampler");
}

void Renderer::nameBrdfLutResources(const rhi::VulkanBrdfLut& brdfLut, std::string_view name) const
{
    if (!brdfLut.valid()) {
        return;
    }

    const std::string prefix{name};
    rhi::debug::setObjectName(context_.vkDevice(), brdfLut.image(), VK_OBJECT_TYPE_IMAGE, prefix + "Image");
    rhi::debug::setObjectName(context_.vkDevice(), brdfLut.imageView(), VK_OBJECT_TYPE_IMAGE_VIEW, prefix + "View");
    rhi::debug::setObjectName(context_.vkDevice(), brdfLut.sampler(), VK_OBJECT_TYPE_SAMPLER, prefix + "Sampler");
}

void Renderer::tryPrintGpuTimings(uint32_t frameIndex)
{
    rhi::VulkanTimestampQuery::Results results{};
    if (!timestampQuery_.readFrame(frameIndex, results) || !results.valid) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - lastGpuTimingPrint_ < std::chrono::seconds(1)) {
        return;
    }

    lastGpuTimingPrint_ = now;

    std::ostringstream message;
    message << std::fixed << std::setprecision(3) << "GPU timings:\n"
            << "  ShadowPass: " << results.shadowPassMs << " ms\n"
            << "  MainPass: " << results.mainPassMs << " ms\n"
            << "  Skybox: " << results.skyboxMs << " ms\n"
            << "  RenderObjects: " << results.renderObjectsMs << " ms\n";
    if (cullingStats_.gpuCulling) {
        const uint32_t totalDrawItems = frameIndex < frameGpuCullTotalDrawItems_.size()
                                            ? frameGpuCullTotalDrawItems_[frameIndex]
                                            : static_cast<uint32_t>(cullingStats_.totalDrawItems);
        const uint32_t batchCount = frameIndex < frameGpuCullBatchCounts_.size()
                                        ? frameGpuCullBatchCounts_[frameIndex]
                                        : static_cast<uint32_t>(cullingStats_.batchCount);
        uint32_t visibleDrawItems = 0;
        if (readGpuVisibleCount(frameIndex, visibleDrawItems)) {
            const uint32_t culledDrawItems =
                totalDrawItems > visibleDrawItems ? totalDrawItems - visibleDrawItems : 0;
            message << "GPU culling:\n"
                    << "  total draw items: " << totalDrawItems << "\n"
                    << "  visible draw items: " << visibleDrawItems << "\n"
                    << "  culled draw items: " << culledDrawItems << "\n"
                    << "  batches: " << batchCount << "\n"
                    << "  indirect count path: "
                    << (isFrameIndirectCountPathActive(frameIndex) ? "enabled" : "disabled");
        } else {
            message << "GPU culling:\n"
                    << "  total draw items: " << totalDrawItems << "\n"
                    << "  visible draw items: unavailable\n"
                    << "  batches: " << batchCount << "\n"
                    << "  indirect count path: "
                    << (isFrameIndirectCountPathActive(frameIndex) ? "enabled" : "disabled");
        }
    } else {
        message << "Culling: total=" << cullingStats_.totalObjects
                << " visible=" << cullingStats_.visibleObjects
                << " culled=" << cullingStats_.culledObjects
                << " drawItems=" << cullingStats_.totalDrawItems
                << " meshBatches=" << cullingStats_.batchCount
                << " commandCount=" << cullingStats_.commandCount;
    }
    message << "\nShadow culling:\n";
    message << "  cascade count: " << shadowCullingStats_.cascadeCount << "\n"
            << "  total shadow draw items across cascades: " << shadowCullingStats_.totalDrawItems << "\n"
            << "  visible shadow draw items across cascades: " << shadowCullingStats_.visibleDrawItems << "\n"
            << "  culled shadow draw items across cascades: " << shadowCullingStats_.culledDrawItems << "\n"
            << "  shadow batches across cascades: " << shadowCullingStats_.batchCount << "\n";
    for (size_t cascadeIndex = 0; cascadeIndex < shadowCullingStats_.cascadeCount &&
                                  cascadeIndex < shadowVisibleDrawItemsPerCascade_.size();
         ++cascadeIndex) {
        message << "  cascade " << cascadeIndex << ": visible draw items "
                << shadowVisibleDrawItemsPerCascade_[cascadeIndex]
                << ", batches " << shadowBatchCountPerCascade_[cascadeIndex]
                << ", split depth " << frameCascades_[cascadeIndex].splitDepth << "\n";
    }
    message << "  GPU shadow culling: " << (shadowCullingStats_.gpuCulling ? "enabled" : "disabled") << "\n"
            << "  shadow path: "
            << (shadowCullingStats_.gpuCulling
                    ? (isShadowIndirectCountPathActive(frameIndex) ? "per-cascade indirect count"
                                                                   : "per-cascade indirect fallback")
                    : "per-cascade direct fallback");
    Logger::info(message.str());
}

void Renderer::updateFrameData(uint32_t frameIndex)
{
    const auto now = std::chrono::steady_clock::now();
    const float elapsedSeconds = std::chrono::duration<float>(now - startTime_).count();

    if (renderObjects_.empty()) {
        allDrawItems_.clear();
        visibleDrawItems_.clear();
        shadowDrawItems_.clear();
        meshDrawBatches_.clear();
        shadowMeshDrawBatches_.clear();
        gpuShadowMeshDrawBatches_.clear();
        for (std::vector<DrawItem>& cascadeDrawItems : shadowCascadeDrawItems_) {
            cascadeDrawItems.clear();
        }
        for (std::vector<MeshDrawBatch>& cascadeBatches : shadowCascadeMeshDrawBatches_) {
            cascadeBatches.clear();
        }
        shadowVisibleDrawItemsPerCascade_.fill(0);
        shadowBatchCountPerCascade_.fill(0);
        cullingStats_ = {};
        shadowCullingStats_ = {};
        if (frameIndex < frameGpuCullTotalDrawItems_.size()) {
            frameGpuCullTotalDrawItems_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuCullBatchCounts_.size()) {
            frameGpuCullBatchCounts_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuCullReadbackReady_.size()) {
            frameGpuCullReadbackReady_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuCullIndirectCountPath_.size()) {
            frameGpuCullIndirectCountPath_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuShadowCullTotalDrawItems_.size()) {
            frameGpuShadowCullTotalDrawItems_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuShadowCullBatchCounts_.size()) {
            frameGpuShadowCullBatchCounts_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuShadowCullReadbackReady_.size()) {
            frameGpuShadowCullReadbackReady_[frameIndex] = 0;
        }
        if (frameIndex < frameGpuShadowCullIndirectCountPath_.size()) {
            frameGpuShadowCullIndirectCountPath_[frameIndex] = 0;
        }
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    const float aspect =
        extent.height == 0 ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 view = camera_.viewMatrix();
    const glm::mat4 projection = camera_.projectionMatrix(aspect);
    const glm::mat4 viewProjection = projection * view;
    updateCascades(aspect);

    for (size_t objectIndex = 0; objectIndex < renderObjects_.size(); ++objectIndex) {
        renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.animateTransform) {
            continue;
        }

        switch (objectIndex) {
        case 0:
            object.transform.rotationRadians = {0.2f, elapsedSeconds, 0.0f};
            break;
        case 1:
            object.transform.rotationRadians = {elapsedSeconds * 1.15f, 0.35f, 0.2f};
            break;
        case 2:
            object.transform.rotationRadians = {0.25f, -0.35f, elapsedSeconds * 0.9f};
            break;
        case 3:
            object.transform.rotationRadians = {elapsedSeconds * 0.35f, elapsedSeconds * 0.55f, 0.45f};
            break;
        default:
            object.transform.rotationRadians = {elapsedSeconds * (0.2f + 0.05f * static_cast<float>(objectIndex)),
                                                elapsedSeconds * 0.4f,
                                                elapsedSeconds * 0.3f};
            break;
        }
    }

    buildDrawItems();
    if (frameIndex < frameGpuCullTotalDrawItems_.size()) {
        frameGpuCullTotalDrawItems_[frameIndex] = static_cast<uint32_t>(std::min(allDrawItems_.size(),
                                                                                 static_cast<size_t>(kMaxDrawItems)));
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

    const renderer::Frustum cameraFrustum = renderer::Frustum::fromViewProjection(viewProjection);
    for (size_t planeIndex = 0; planeIndex < frameFrustumPlanes_.size(); ++planeIndex) {
        const renderer::FrustumPlane& cameraPlane = cameraFrustum.planes[planeIndex];
        frameFrustumPlanes_[planeIndex] = glm::vec4(cameraPlane.normal, cameraPlane.distance);
    }

    const uint32_t cascadeCount = activeCascadeCount();
    shadowDrawItems_.clear();
    shadowMeshDrawBatches_.clear();
    gpuShadowMeshDrawBatches_.clear();
    shadowCullingStats_ = {};
    shadowCullingStats_.cascadeCount = cascadeCount;
    shadowCullingStats_.totalDrawItems = allDrawItems_.size() * cascadeCount;
    shadowVisibleDrawItemsPerCascade_.fill(0);
    shadowBatchCountPerCascade_.fill(0);
    for (std::vector<DrawItem>& cascadeDrawItems : shadowCascadeDrawItems_) {
        cascadeDrawItems.clear();
    }
    for (std::vector<MeshDrawBatch>& cascadeBatches : shadowCascadeMeshDrawBatches_) {
        cascadeBatches.clear();
    }

    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        buildShadowDrawItems(cascadeIndex, frameCascades_[cascadeIndex].lightFrustum);
        buildMeshDrawBatchesForItems(shadowCascadeDrawItems_[cascadeIndex],
                                     shadowCascadeMeshDrawBatches_[cascadeIndex]);
        shadowVisibleDrawItemsPerCascade_[cascadeIndex] =
            static_cast<uint32_t>(shadowCascadeDrawItems_[cascadeIndex].size());
        shadowBatchCountPerCascade_[cascadeIndex] =
            static_cast<uint32_t>(shadowCascadeMeshDrawBatches_[cascadeIndex].size());
        shadowCullingStats_.visibleDrawItems += shadowCascadeDrawItems_[cascadeIndex].size();
        shadowCullingStats_.batchCount += shadowCascadeMeshDrawBatches_[cascadeIndex].size();
    }
    shadowCullingStats_.culledDrawItems =
        shadowCullingStats_.totalDrawItems > shadowCullingStats_.visibleDrawItems
            ? shadowCullingStats_.totalDrawItems - shadowCullingStats_.visibleDrawItems
            : 0;

    const bool gpuShadowCullingActive = isGpuShadowCullingActive();
    if (gpuShadowCullingActive) {
        buildMeshDrawBatchesForItems(allDrawItems_, gpuShadowMeshDrawBatches_);

        bool shadowIndirectCountPathActive = isShadowIndirectCountSupported();
        if (shadowIndirectCountPathActive) {
            const uint32_t maxDrawIndirectCount = context_.device().maxDrawIndirectCount();
            for (const MeshDrawBatch& batch : gpuShadowMeshDrawBatches_) {
                if (batch.drawItemCount > maxDrawIndirectCount) {
                    shadowIndirectCountPathActive = false;
                    break;
                }
            }
        }

        if (frameIndex < frameGpuShadowCullTotalDrawItems_.size()) {
            frameGpuShadowCullTotalDrawItems_[frameIndex] =
                static_cast<uint32_t>(std::min(allDrawItems_.size() * cascadeCount,
                                               static_cast<size_t>(kMaxDrawItems)));
        }
        if (frameIndex < frameGpuShadowCullBatchCounts_.size()) {
            frameGpuShadowCullBatchCounts_[frameIndex] =
                static_cast<uint32_t>(gpuShadowMeshDrawBatches_.size() * cascadeCount);
        }
        if (frameIndex < frameGpuShadowCullIndirectCountPath_.size()) {
            frameGpuShadowCullIndirectCountPath_[frameIndex] = shadowIndirectCountPathActive ? 1 : 0;
        }

        shadowCullingStats_.gpuCulling = true;
        shadowCullingStats_.indirectDrawing = true;
        updateGpuShadowCullInputBuffer(frameIndex);
    } else {
        shadowCullingStats_.indirectDrawing = false;
    }

    const bool gpuCullingActive = isGpuCullingActive();
    if (gpuCullingActive) {
        visibleDrawItems_ = allDrawItems_;
        cullingStats_ = {};
        cullingStats_.gpuCulling = true;
        cullingStats_.totalObjects = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
        cullingStats_.totalDrawItems = allDrawItems_.size();
    } else {
        buildVisibleDrawItems(cameraFrustum);
    }
    buildMeshDrawBatches();

    if (gpuCullingActive) {
        bool indirectCountPathActive = isMainPassIndirectCountSupported();
        if (indirectCountPathActive) {
            const uint32_t maxDrawIndirectCount = context_.device().maxDrawIndirectCount();
            for (const MeshDrawBatch& batch : meshDrawBatches_) {
                if (batch.drawItemCount > maxDrawIndirectCount) {
                    indirectCountPathActive = false;
                    break;
                }
            }
        }

        if (frameIndex < frameGpuCullBatchCounts_.size()) {
            frameGpuCullBatchCounts_[frameIndex] = static_cast<uint32_t>(meshDrawBatches_.size());
        }
        if (frameIndex < frameGpuCullIndirectCountPath_.size()) {
            frameGpuCullIndirectCountPath_[frameIndex] = indirectCountPathActive ? 1 : 0;
        }
        updateGpuCullInputBuffer(frameIndex);
    } else {
        updateIndirectDrawBuffer(frameIndex);
    }

    const size_t objectFrameCount = std::min(allDrawItems_.size(), static_cast<size_t>(kMaxDrawItems));
    std::vector<ObjectFrameData> objectFrameData(objectFrameCount);

    for (size_t drawIndex = 0; drawIndex < objectFrameCount; ++drawIndex) {
        const DrawItem& drawItem = allDrawItems_[drawIndex];
        if (drawItem.objectIndex >= renderObjects_.size()) {
            continue;
        }

        const renderer::RenderObject& object = renderObjects_[drawItem.objectIndex];
        if (!object.mesh) {
            continue;
        }

        const glm::mat4 model = object.transform.modelMatrix();
        ObjectFrameData& frameData = objectFrameData[drawIndex];
        frameData.mvp = viewProjection * model;
        frameData.model = model;
        for (uint32_t cascadeIndex = 0; cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
            frameData.lightMvp[cascadeIndex] = frameCascades_[cascadeIndex].lightViewProjection * model;
        }
        frameData.lightDirection = kDirectionalLightDirection;
        frameData.lightColor = kDirectionalLightColor;
        frameData.ambientColor = kAmbientLightColor;
        frameData.cascadeSplits = frameCascadeSplits_;
        frameData.shadowSettings = {csmSettings_.depthBiasConstant,
                                    csmSettings_.depthBiasSlope,
                                    shadowSettings_.enablePcf ? 1.0f : 0.0f,
                                    static_cast<float>(std::max(shadowSettings_.pcfRadius, 0))};
        const renderer::Material* material = drawItem.material ? drawItem.material : object.material;
        if (material) {
            frameData.baseColorFactor = material->baseColorFactor;
            frameData.materialParams = {material->metallic,
                                        material->roughness,
                                        material->multiScatterStrength,
                                        0.0f};
            frameData.textureIndices = {material->baseColorTextureIndex,
                                        material->normalTextureIndex,
                                        material->metallicRoughnessTextureIndex,
                                        0};
        }
        frameData.cameraPosition = glm::vec4(camera_.position, 1.0f);
        frameData.cameraForward =
            glm::vec4(glm::normalize(camera_.target - camera_.position), static_cast<float>(cascadeCount));
    }

    frameObjectDataBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const ObjectFrameData>(objectFrameData.data(), objectFrameData.size())));
}

void Renderer::recreateSwapchain()
{
    if (window_.isMinimized()) {
        return;
    }

    context_.waitIdle();
    swapchain_.recreate(context_, window_.framebufferExtent());
    sync_.recreateRenderFinishedSemaphores(swapchain_.imageCount());

    const bool pipelineNeedsRecreate =
        pipeline_.pipeline() == VK_NULL_HANDLE || pipelineColorFormat_ != swapchain_.colorFormat() ||
        pipelineDepthFormat_ != swapchain_.depthFormat() || skyboxPipeline_.pipeline() == VK_NULL_HANDLE ||
        skyboxPipelineColorFormat_ != swapchain_.colorFormat() ||
        skyboxPipelineDepthFormat_ != swapchain_.depthFormat() || shadowPipelineDepthFormat_ != shadowMap_.format();
    if (pipelineNeedsRecreate) {
        createPipeline();
    }

    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
}

bool Renderer::readGpuVisibleCount(uint32_t frameIndex, uint32_t& visibleCount)
{
    visibleCount = 0;
    if (!isGpuCullingActive() || frameIndex >= frameBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameGpuCullReadbackReady_.size() || frameGpuCullReadbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = frameBatchVisibleCountReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    const bool indirectCountPathActive = isFrameIndirectCountPathActive(frameIndex);
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

bool Renderer::readGpuShadowVisibleCount(uint32_t frameIndex, uint32_t& visibleCount)
{
    visibleCount = 0;
    if (!isGpuShadowCullingActive() || frameIndex >= frameShadowBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameGpuShadowCullReadbackReady_.size() ||
        frameGpuShadowCullReadbackReady_[frameIndex] == 0) {
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

bool Renderer::isGpuCullingActive() const
{
    return useGpuCulling_ && gpuCullingAvailable_ && gpuCullPipeline_.pipeline() != VK_NULL_HANDLE &&
           gpuCullPipeline_.layout() != VK_NULL_HANDLE && !gpuCullDescriptorSets_.empty() &&
           frameCullInputBuffers_.size() == frames_.size() &&
           frameBatchVisibleCountBuffers_.size() == frames_.size() &&
           frameBatchVisibleCountReadbackBuffers_.size() == frames_.size();
}

bool Renderer::isGpuShadowCullingActive() const
{
    return useGpuShadowCulling_ && gpuShadowCullingAvailable_ && isShadowIndirectActive() &&
           gpuCullPipeline_.pipeline() != VK_NULL_HANDLE && gpuCullPipeline_.layout() != VK_NULL_HANDLE &&
           !shadowCullDescriptorSets_.empty() && frameShadowCullInputBuffers_.size() == frames_.size() &&
           frameShadowBatchVisibleCountBuffers_.size() == frames_.size() &&
           frameShadowBatchVisibleCountReadbackBuffers_.size() == frames_.size();
}

bool Renderer::isBindlessMaterialTextureActive() const
{
    return useBindlessMaterialTextures_ && bindlessMaterialTexturesAvailable_ && bindlessTextureHeap_.valid();
}

bool Renderer::isMainPassMultiDrawIndirectActive() const
{
    return isBindlessMaterialTextureActive() && context_.device().multiDrawIndirectEnabled() &&
           context_.device().drawIndirectFirstInstanceEnabled();
}

bool Renderer::isMainPassIndirectCountSupported() const
{
    return isGpuCullingActive() && isMainPassMultiDrawIndirectActive() &&
           context_.device().drawIndexedIndirectCountAvailable();
}

bool Renderer::isFrameIndirectCountPathActive(uint32_t frameIndex) const
{
    return frameIndex < frameGpuCullIndirectCountPath_.size() && frameGpuCullIndirectCountPath_[frameIndex] != 0;
}

bool Renderer::isShadowIndirectCountSupported() const
{
    return isGpuShadowCullingActive() && context_.device().drawIndexedIndirectCountAvailable();
}

bool Renderer::isShadowIndirectCountPathActive(uint32_t frameIndex) const
{
    return frameIndex < frameGpuShadowCullIndirectCountPath_.size() &&
           frameGpuShadowCullIndirectCountPath_[frameIndex] != 0;
}

bool Renderer::isShadowIndirectActive() const
{
    return shadowIndirectAvailable_ && context_.device().multiDrawIndirectEnabled() &&
           context_.device().drawIndirectFirstInstanceEnabled() &&
           frameShadowIndirectDrawBuffers_.size() == frames_.size();
}

VkDescriptorSet Renderer::globalMaterialDescriptorSet() const
{
    if (!materialVariants_.empty()) {
        return materialVariants_.front().descriptorSet;
    }

    return checkerboardMaterial_.descriptorSet;
}

void Renderer::recordGpuCullingCommands(VkCommandBuffer commandBuffer)
{
    if (!isGpuCullingActive() || allDrawItems_.empty()) {
        return;
    }
    if (currentFrame_ >= gpuCullDescriptorSets_.size() || currentFrame_ >= frameBatchVisibleCountBuffers_.size() ||
        currentFrame_ >= frameBatchVisibleCountReadbackBuffers_.size() ||
        currentFrame_ >= frameGpuCullReadbackReady_.size()) {
        return;
    }

    VkBuffer visibleCountBuffer = frameBatchVisibleCountBuffers_.at(currentFrame_).buffer();
    VkBuffer visibleCountReadbackBuffer = frameBatchVisibleCountReadbackBuffers_.at(currentFrame_).buffer();
    if (visibleCountBuffer == VK_NULL_HANDLE || visibleCountReadbackBuffer == VK_NULL_HANDLE) {
        return;
    }

    const bool indirectCountPathActive = isFrameIndirectCountPathActive(currentFrame_);

    rhi::debug::beginLabel(commandBuffer, "GpuCulling");
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

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.pipeline());

    const VkDescriptorSet descriptorSet = gpuCullDescriptorSets_[currentFrame_];
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpuCullPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);

    GpuCullPushConstants pushConstants{};
    pushConstants.frustumPlanes = frameFrustumPlanes_;
    pushConstants.params = glm::uvec4(static_cast<uint32_t>(allDrawItems_.size()),
                                      isMainPassMultiDrawIndirectActive() ? 1U : 0U,
                                      indirectCountPathActive ? 1U : 0U,
                                      0);
    vkCmdPushConstants(commandBuffer,
                       gpuCullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(GpuCullPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "ComputeCullDispatch");
    const uint32_t groupCount =
        (static_cast<uint32_t>(allDrawItems_.size()) + kGpuCullLocalSize - 1) / kGpuCullLocalSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);
    rhi::debug::endLabel(commandBuffer);

    std::array<VkBufferMemoryBarrier2, 2> computeBarriers{};
    computeBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    computeBarriers[0].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    computeBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[0].buffer = frameIndirectDrawBuffers_.at(currentFrame_).buffer();
    computeBarriers[0].offset = 0;
    computeBarriers[0].size =
        static_cast<VkDeviceSize>(allDrawItems_.size() * sizeof(VkDrawIndexedIndirectCommand));

    computeBarriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    computeBarriers[1].dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    computeBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarriers[1].buffer = visibleCountBuffer;
    computeBarriers[1].offset = 0;
    computeBarriers[1].size = kBatchVisibleCountBufferSize;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(computeBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = computeBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    VkBufferCopy visibleCountCopy{};
    visibleCountCopy.size = kBatchVisibleCountBufferSize;
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
    readbackBarrier.size = kBatchVisibleCountBufferSize;

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameGpuCullReadbackReady_[currentFrame_] = 1;
    rhi::debug::endLabel(commandBuffer);
}

void Renderer::recordGpuShadowCullingCommands(VkCommandBuffer commandBuffer, uint32_t cascadeIndex)
{
    if (!isGpuShadowCullingActive() || allDrawItems_.empty()) {
        return;
    }
    if (cascadeIndex >= activeCascadeCount()) {
        return;
    }
    if (currentFrame_ >= shadowCullDescriptorSets_.size() ||
        currentFrame_ >= frameShadowBatchVisibleCountBuffers_.size() ||
        currentFrame_ >= frameShadowBatchVisibleCountReadbackBuffers_.size() ||
        currentFrame_ >= frameShadowIndirectDrawBuffers_.size() ||
        currentFrame_ >= frameGpuShadowCullReadbackReady_.size()) {
        return;
    }

    VkBuffer visibleCountBuffer = frameShadowBatchVisibleCountBuffers_.at(currentFrame_).buffer();
    VkBuffer visibleCountReadbackBuffer = frameShadowBatchVisibleCountReadbackBuffers_.at(currentFrame_).buffer();
    VkBuffer shadowIndirectDrawBuffer = frameShadowIndirectDrawBuffers_.at(currentFrame_).buffer();
    if (visibleCountBuffer == VK_NULL_HANDLE || visibleCountReadbackBuffer == VK_NULL_HANDLE ||
        shadowIndirectDrawBuffer == VK_NULL_HANDLE) {
        return;
    }

    const VkDeviceSize shadowIndirectBufferSize = std::min<VkDeviceSize>(
        frameShadowIndirectDrawBuffers_.at(currentFrame_).size(),
        static_cast<VkDeviceSize>(allDrawItems_.size() * sizeof(VkDrawIndexedIndirectCommand)));
    if (shadowIndirectBufferSize == 0) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "GpuShadowCullingCascade" + std::to_string(cascadeIndex));
    vkCmdFillBuffer(commandBuffer, visibleCountBuffer, 0, kBatchVisibleCountBufferSize, 0);
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
    resetBarriers[0].size = kBatchVisibleCountBufferSize;

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

    const VkDescriptorSet descriptorSet = shadowCullDescriptorSets_[currentFrame_];
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpuCullPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);

    GpuCullPushConstants pushConstants{};
    pushConstants.frustumPlanes = frameShadowCascadeFrustumPlanes_[cascadeIndex];
    pushConstants.params =
        glm::uvec4(static_cast<uint32_t>(allDrawItems_.size()), 1U, 1U, 0U);
    vkCmdPushConstants(commandBuffer,
                       gpuCullPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(GpuCullPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "ShadowCullDispatchCascade" + std::to_string(cascadeIndex));
    const uint32_t groupCount =
        (static_cast<uint32_t>(allDrawItems_.size()) + kGpuCullLocalSize - 1) / kGpuCullLocalSize;
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
    computeBarriers[1].size = kBatchVisibleCountBufferSize;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(computeBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = computeBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    VkBufferCopy visibleCountCopy{};
    visibleCountCopy.size = kBatchVisibleCountBufferSize;
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
    readbackBarrier.size = kBatchVisibleCountBufferSize;

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameGpuShadowCullReadbackReady_[currentFrame_] = 1;
    rhi::debug::endLabel(commandBuffer);
}

void Renderer::recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t mainDrawItemCount = visibleDrawItems_.size();

    renderGraph_.beginFrame(commandBuffer, swapchain_, shadowMap_, imageIndex);
    rhi::debug::beginLabel(commandBuffer, "Frame");
    timestampQuery_.resetFrame(commandBuffer, currentFrame_);

    const bool gpuShadowCullingActive = isGpuShadowCullingActive() && !allDrawItems_.empty();
    const uint32_t cascadeCount = activeCascadeCount();

    rhi::debug::beginLabel(commandBuffer, "CSMShadowPass");
    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::ShadowPass);

    const VkExtent2D shadowExtent = shadowMap_.extent();
    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = static_cast<float>(shadowExtent.width);
    shadowViewport.height = static_cast<float>(shadowExtent.height);
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = shadowExtent;

    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        if (gpuShadowCullingActive) {
            recordGpuShadowCullingCommands(commandBuffer, cascadeIndex);
        }

        const std::vector<DrawItem>& activeShadowDrawItems =
            gpuShadowCullingActive ? allDrawItems_ : shadowCascadeDrawItems_[cascadeIndex];
        const std::vector<MeshDrawBatch>& activeShadowMeshDrawBatches =
            gpuShadowCullingActive ? gpuShadowMeshDrawBatches_ : shadowCascadeMeshDrawBatches_[cascadeIndex];
        const size_t shadowDrawItemCount = activeShadowDrawItems.size();
        const bool shadowIndirectCountPathActive =
            gpuShadowCullingActive && isShadowIndirectCountPathActive(currentFrame_);

        rhi::debug::beginLabel(commandBuffer, "ShadowCascade" + std::to_string(cascadeIndex));
        renderGraph_.beginShadowPass(cascadeIndex);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_.pipeline());
        vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

        const std::string shadowCullingLabel =
            gpuShadowCullingActive
                ? "ShadowCasterCulling GPU cascade " + std::to_string(cascadeIndex) + " max " +
                      std::to_string(shadowDrawItemCount)
                : "ShadowCasterCulling cascade " + std::to_string(cascadeIndex) + " visible " +
                      std::to_string(shadowDrawItemCount) + "/" + std::to_string(allDrawItems_.size());
        rhi::debug::beginLabel(commandBuffer, shadowCullingLabel);
        rhi::debug::endLabel(commandBuffer);

        const std::string shadowDrawLabel =
            "ShadowCascade" + std::to_string(cascadeIndex) + " DrawItems " +
            std::to_string(shadowDrawItemCount) +
            (gpuShadowCullingActive ? (shadowIndirectCountPathActive ? " GPU culling indirect-count"
                                                                     : " GPU culling indirect fallback")
                                    : " CPU culling direct fallback");
        rhi::debug::beginLabel(commandBuffer, shadowDrawLabel);

        const renderer::Mesh* boundShadowMesh = nullptr;
        if (gpuShadowCullingActive && !activeShadowDrawItems.empty()) {
            const PushConstants pushConstants{objectFrameDataBaseAddress, cascadeIndex, 0};
            vkCmdPushConstants(commandBuffer,
                               shadowPipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &pushConstants);

            const VkBuffer shadowIndirectDrawBuffer = frameShadowIndirectDrawBuffers_.at(currentFrame_).buffer();
            const VkBuffer shadowBatchVisibleCountBuffer =
                shadowIndirectCountPathActive ? frameShadowBatchVisibleCountBuffers_.at(currentFrame_).buffer()
                                              : VK_NULL_HANDLE;
            const std::string shadowBatchesLabel =
                "ShadowIndirectDrawBatches " + std::to_string(activeShadowMeshDrawBatches.size()) +
                " max commands " + std::to_string(shadowDrawItemCount);
            rhi::debug::beginLabel(commandBuffer, shadowBatchesLabel);
            for (const MeshDrawBatch& batch : activeShadowMeshDrawBatches) {
                if (!batch.mesh || batch.drawItemCount == 0) {
                    continue;
                }

                if (boundShadowMesh != batch.mesh) {
                    const VkBuffer vertexBuffers[] = {batch.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, batch.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundShadowMesh = batch.mesh;
                }

                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(batch.compactedCommandOffset * sizeof(VkDrawIndexedIndirectCommand));
                if (shadowIndirectCountPathActive && shadowBatchVisibleCountBuffer != VK_NULL_HANDLE) {
                    vkCmdDrawIndexedIndirectCount(commandBuffer,
                                                 shadowIndirectDrawBuffer,
                                                 indirectOffset,
                                                 shadowBatchVisibleCountBuffer,
                                                 batch.visibleCountOffset,
                                                 batch.drawItemCount,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                } else {
                    vkCmdDrawIndexedIndirect(commandBuffer,
                                             shadowIndirectDrawBuffer,
                                             indirectOffset,
                                             batch.drawItemCount,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }
            }
            rhi::debug::endLabel(commandBuffer);
        } else {
            for (size_t drawIndex = 0; drawIndex < shadowDrawItemCount; ++drawIndex) {
                const DrawItem& drawItem = activeShadowDrawItems[drawIndex];
                if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems) {
                    continue;
                }

                const PushConstants pushConstants{
                    objectFrameDataBaseAddress +
                        static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData)),
                    cascadeIndex,
                    0};

                vkCmdPushConstants(commandBuffer,
                                   shadowPipeline_.layout(),
                                   VK_SHADER_STAGE_VERTEX_BIT,
                                   0,
                                   static_cast<uint32_t>(sizeof(PushConstants)),
                                   &pushConstants);

                const renderer::Mesh* mesh = drawItem.mesh;
                if (boundShadowMesh != mesh) {
                    const VkBuffer vertexBuffers[] = {mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundShadowMesh = mesh;
                }

                if (drawItem.indexCount > 0) {
                    vkCmdDrawIndexed(
                        commandBuffer, drawItem.indexCount, 1, drawItem.firstIndex, drawItem.vertexOffset, 0);
                }
            }
        }
        rhi::debug::endLabel(commandBuffer);

        renderGraph_.endShadowPass(cascadeIndex + 1 == cascadeCount);
        rhi::debug::endLabel(commandBuffer);
    }

    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::ShadowPass);
    rhi::debug::endLabel(commandBuffer);

    recordGpuCullingCommands(commandBuffer);

    rhi::debug::beginLabel(commandBuffer, "MainPass");
    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::MainPass);
    renderGraph_.beginMainPass();

    const VkExtent2D extent = swapchain_.extent();
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    // Viewport and scissor depend on the current swapchain extent, so they stay
    // dynamic instead of forcing a new pipeline for every resize.
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    rhi::debug::beginLabel(commandBuffer, "Skybox");
    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Skybox);
    if (skyboxDescriptorSet_ != VK_NULL_HANDLE) {
        const float aspect =
            extent.height == 0 ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 skyboxView = camera_.viewMatrix();
        skyboxView[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::mat4 projection = camera_.projectionMatrix(aspect);
        const SkyboxPushConstants skyboxPushConstants{glm::inverse(projection * skyboxView)};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline_.pipeline());
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skyboxPipeline_.layout(),
                                0,
                                1,
                                &skyboxDescriptorSet_,
                                0,
                                nullptr);
        vkCmdPushConstants(commandBuffer,
                           skyboxPipeline_.layout(),
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(SkyboxPushConstants)),
                           &skyboxPushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Skybox);
    rhi::debug::endLabel(commandBuffer);

    const std::string objectDrawLabel =
        cullingStats_.gpuCulling
            ? "MainPass IndirectDrawItems " + std::to_string(mainDrawItemCount) + " GPU culling batches " +
                  std::to_string(meshDrawBatches_.size())
            : "MainPass IndirectDrawItems " + std::to_string(mainDrawItemCount) + " visible objects " +
                  std::to_string(cullingStats_.visibleObjects) + "/" + std::to_string(cullingStats_.totalObjects) +
                  " batches " + std::to_string(meshDrawBatches_.size());
    rhi::debug::beginLabel(commandBuffer, objectDrawLabel);
    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::RenderObjects);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());

    const VkDescriptorSet globalDescriptorSet = globalMaterialDescriptorSet();
    const bool bindlessMaterialTexturesActive = isBindlessMaterialTextureActive();
    const bool multiDrawIndirectActive = isMainPassMultiDrawIndirectActive();
    bool bindlessDescriptorSetsBound = false;
    if (bindlessMaterialTexturesActive) {
        if (globalDescriptorSet != VK_NULL_HANDLE) {
            const std::array<VkDescriptorSet, 2> descriptorSets{
                globalDescriptorSet,
                bindlessTextureHeap_.descriptorSet(),
            };
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_.layout(),
                                    0,
                                    static_cast<uint32_t>(descriptorSets.size()),
                                    descriptorSets.data(),
                                    0,
                                    nullptr);
            bindlessDescriptorSetsBound = true;
        }
    }

    const renderer::Mesh* boundMesh = nullptr;
    const VkBuffer indirectDrawBuffer = frameIndirectDrawBuffers_.at(currentFrame_).buffer();
    const bool indirectCountPathActive = isFrameIndirectCountPathActive(currentFrame_);
    const VkBuffer batchVisibleCountBuffer =
        indirectCountPathActive ? frameBatchVisibleCountBuffers_.at(currentFrame_).buffer() : VK_NULL_HANDLE;
    if (multiDrawIndirectActive) {
        if (bindlessDescriptorSetsBound) {
            const PushConstants pushConstants{objectFrameDataBaseAddress};
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &pushConstants);

            const std::string batchesLabel =
                std::string(indirectCountPathActive ? "IndirectCountBatches " : "MultiDrawIndirectBatches ") +
                std::to_string(meshDrawBatches_.size()) + " max commands " + std::to_string(mainDrawItemCount);
            rhi::debug::beginLabel(commandBuffer, batchesLabel);
            for (const MeshDrawBatch& batch : meshDrawBatches_) {
                if (!batch.mesh || batch.drawItemCount == 0) {
                    continue;
                }

                if (boundMesh != batch.mesh) {
                    const VkBuffer vertexBuffers[] = {batch.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, batch.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundMesh = batch.mesh;
                }

                const std::string batchLabel =
                    std::string(indirectCountPathActive ? "MeshBatchIndirectCount max commands "
                                                        : "MeshBatchDraw commands ") +
                    std::to_string(batch.drawItemCount);
                rhi::debug::beginLabel(commandBuffer, batchLabel);
                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(batch.compactedCommandOffset * sizeof(VkDrawIndexedIndirectCommand));
                if (indirectCountPathActive && batchVisibleCountBuffer != VK_NULL_HANDLE) {
                    vkCmdDrawIndexedIndirectCount(commandBuffer,
                                                 indirectDrawBuffer,
                                                 indirectOffset,
                                                 batchVisibleCountBuffer,
                                                 batch.visibleCountOffset,
                                                 batch.drawItemCount,
                                                 sizeof(VkDrawIndexedIndirectCommand));
                } else {
                    vkCmdDrawIndexedIndirect(commandBuffer,
                                             indirectDrawBuffer,
                                             indirectOffset,
                                             batch.drawItemCount,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }
                rhi::debug::endLabel(commandBuffer);
            }
            rhi::debug::endLabel(commandBuffer);
        }
    } else {
        for (size_t drawIndex = 0; drawIndex < mainDrawItemCount; ++drawIndex) {
            const DrawItem& drawItem = visibleDrawItems_[drawIndex];
            if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems) {
                continue;
            }

            if (bindlessMaterialTexturesActive) {
                if (!bindlessDescriptorSetsBound) {
                    continue;
                }
            } else {
                if (!drawItem.material || drawItem.material->descriptorSet == VK_NULL_HANDLE) {
                    continue;
                }

                const VkDescriptorSet descriptorSet = drawItem.material->descriptorSet;
                vkCmdBindDescriptorSets(commandBuffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipeline_.layout(),
                                        0,
                                        1,
                                        &descriptorSet,
                                        0,
                                        nullptr);
            }

            const PushConstants pushConstants{
                objectFrameDataBaseAddress +
                static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData))};

            // Fallback recording pushes the address of this draw's object data; firstInstance stays zero.
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &pushConstants);

            const renderer::Mesh* mesh = drawItem.mesh;
            if (boundMesh != mesh) {
                const VkBuffer vertexBuffers[] = {mesh->vertexBuffer()};
                const VkDeviceSize vertexOffsets[] = {0};
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                boundMesh = mesh;
            }

            if (drawItem.indexCount > 0) {
                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(drawIndex * sizeof(VkDrawIndexedIndirectCommand));
                vkCmdDrawIndexedIndirect(
                    commandBuffer, indirectDrawBuffer, indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
            }
        }
    }
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::RenderObjects);
    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endMainPass();
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::MainPass);
    rhi::debug::endLabel(commandBuffer);
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endFrame();
}

} // namespace ve
