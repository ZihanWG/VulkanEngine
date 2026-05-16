#include "renderer/Renderer.h"

#include "core/Logger.h"
#include "core/Window.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"

#include <imgui.h>

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
#include <glm/vec2.hpp>
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
    glm::vec4 cameraPosition{0.0f, 0.0f, 0.0f, 0.0f};
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
// cameraPosition.xyz is the world-space camera position, and w stores the
// cascade debug-color toggle as 0.0 or 1.0.
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
constexpr VkFormat kSceneColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kBloomColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr uint32_t kLuminanceLocalSizeX = 16;
constexpr uint32_t kLuminanceLocalSizeY = 16;
constexpr uint32_t kHistogramBinCount = 256;
constexpr uint32_t kHistogramLocalSizeX = 16;
constexpr uint32_t kHistogramLocalSizeY = 16;
constexpr float kMinAverageLuminance = 0.0001f;
constexpr float kDefaultHistogramMinLogLuminance = -10.0f;
constexpr float kDefaultHistogramMaxLogLuminance = 4.0f;

const glm::vec4 kDirectionalLightDirection{0.35f, -0.65f, -0.55f, 0.0f};
const glm::vec4 kDirectionalLightColor{0.85f, 0.85f, 0.85f, 1.0f};
const glm::vec4 kAmbientLightColor{0.15f, 0.15f, 0.15f, 1.0f};

struct PushConstants {
    VkDeviceAddress objectFrameDataAddress = 0;
    uint32_t cascadeIndex = 0;
    uint32_t toneMappingOperator = 0;
    float exposure = 1.0f;
};

static_assert(offsetof(PushConstants, objectFrameDataAddress) == 0);
static_assert(offsetof(PushConstants, cascadeIndex) == 8);
static_assert(offsetof(PushConstants, toneMappingOperator) == 12);
static_assert(offsetof(PushConstants, exposure) == 16);
static_assert(sizeof(PushConstants) == 24);

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
    float exposure = 1.0f;
    uint32_t toneMappingOperator = 0;
};

static_assert(offsetof(SkyboxPushConstants, inverseViewProjection) == 0);
static_assert(offsetof(SkyboxPushConstants, exposure) == sizeof(glm::mat4));
static_assert(offsetof(SkyboxPushConstants, toneMappingOperator) == sizeof(glm::mat4) + sizeof(float));
static_assert(sizeof(SkyboxPushConstants) >= sizeof(glm::mat4) + sizeof(float) + sizeof(uint32_t));

struct BloomExtractPushConstants {
    float threshold = 1.0f;
};

static_assert(sizeof(BloomExtractPushConstants) == 4);

struct BloomBlurPushConstants {
    glm::vec2 texelSize{1.0f, 1.0f};
    uint32_t horizontal = 0;
    uint32_t padding = 0;
};

static_assert(offsetof(BloomBlurPushConstants, texelSize) == 0);
static_assert(offsetof(BloomBlurPushConstants, horizontal) == 8);
static_assert(sizeof(BloomBlurPushConstants) == 16);

struct CompositePushConstants {
    float exposure = 1.0f;
    float bloomIntensity = 0.1f;
    uint32_t toneMappingOperator = 0;
    uint32_t bloomEnabled = 1;
};

static_assert(offsetof(CompositePushConstants, exposure) == 0);
static_assert(offsetof(CompositePushConstants, bloomIntensity) == 4);
static_assert(offsetof(CompositePushConstants, toneMappingOperator) == 8);
static_assert(offsetof(CompositePushConstants, bloomEnabled) == 12);
static_assert(sizeof(CompositePushConstants) == 16);

struct LuminancePushConstants {
    glm::uvec4 params{0, 0, 0, 0};
};

static_assert(sizeof(LuminancePushConstants) == 16);

struct LuminancePartial {
    float sumLogLuminance = 0.0f;
    float sampleCount = 0.0f;
    float padding0 = 0.0f;
    float padding1 = 0.0f;
};

static_assert(sizeof(LuminancePartial) == 16);

struct HistogramPushConstants {
    glm::uvec4 params{0, 0, 0, 0};
    glm::vec4 logLuminanceRange{kDefaultHistogramMinLogLuminance, kDefaultHistogramMaxLogLuminance, 0.0001f, 0.0f};
};

static_assert(offsetof(HistogramPushConstants, params) == 0);
static_assert(offsetof(HistogramPushConstants, logLuminanceRange) == 16);
static_assert(sizeof(HistogramPushConstants) == 32);

enum class ExposureMode : int {
    Manual = 0,
    LogAverage = 1,
    Histogram = 2,
};

uint32_t toneMappingOperatorValue(int operatorType)
{
    return operatorType == 1 ? 1u : 0u;
}

float toneMappingExposureValue(float exposure)
{
    return std::max(exposure, 0.0f);
}

ExposureMode exposureModeValue(int exposureMode)
{
    if (exposureMode == static_cast<int>(ExposureMode::Manual)) {
        return ExposureMode::Manual;
    }
    if (exposureMode == static_cast<int>(ExposureMode::LogAverage)) {
        return ExposureMode::LogAverage;
    }

    return ExposureMode::Histogram;
}

std::string_view exposureModeName(ExposureMode exposureMode)
{
    switch (exposureMode) {
    case ExposureMode::Manual:
        return "manual";
    case ExposureMode::LogAverage:
        return "log-average";
    case ExposureMode::Histogram:
        return "histogram";
    }

    return "histogram";
}

std::pair<float, float> sanitizedHistogramLogRange(float minLogLuminance, float maxLogLuminance)
{
    if (!std::isfinite(minLogLuminance) || !std::isfinite(maxLogLuminance) ||
        maxLogLuminance <= minLogLuminance + 0.001f) {
        return {kDefaultHistogramMinLogLuminance, kDefaultHistogramMaxLogLuminance};
    }

    return {minLogLuminance, maxLogLuminance};
}

std::pair<float, float> sanitizedPercentileRange(float lowPercentile, float highPercentile)
{
    float low = std::clamp(lowPercentile, 0.0f, 1.0f);
    float high = std::clamp(highPercentile, 0.0f, 1.0f);
    if (high <= low) {
        low = 0.05f;
        high = 0.95f;
    }

    return {low, high};
}

void setViewportAndScissor(VkCommandBuffer commandBuffer, VkExtent2D extent)
{
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

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

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

std::filesystem::path defaultRuntimeSettingsPath()
{
#if defined(VULKAN_ENGINE_CONFIG_DIR)
    return std::filesystem::path(VULKAN_ENGINE_CONFIG_DIR) / "runtime_settings.json";
#else
    return std::filesystem::path("config") / "runtime_settings.json";
#endif
}

std::string_view colorSpaceName(rhi::TextureColorSpace colorSpace)
{
    switch (colorSpace) {
    case rhi::TextureColorSpace::Linear:
        return "linear";
    case rhi::TextureColorSpace::SRGB:
        return "sRGB";
    }

    return "unknown";
}

float historyValue(double value)
{
    if (!std::isfinite(value) || value < 0.0) {
        return 0.0f;
    }

    return static_cast<float>(value);
}

float knownGpuFrameTotalMs(const rhi::VulkanTimestampQuery::Results& results)
{
    return historyValue(results.shadowPassMs + results.mainPassMs + results.bloomMs + results.autoExposureMs +
                        results.histogramExposureMs + results.compositeMs);
}

std::string resourceUsageList(const renderer::RenderPassNode& pass, renderer::RenderResourceAccess access)
{
    std::ostringstream names;
    bool first = true;
    for (const renderer::RenderResourceUsage& usage : pass.resourceUsages) {
        if (usage.access != access) {
            continue;
        }

        if (!first) {
            names << ", ";
        }
        names << usage.resource.name;
        first = false;
    }

    return first ? "-" : names.str();
}

} // namespace

void Renderer::DebugHistory::push(float value)
{
    samples[cursor] = std::isfinite(value) && value >= 0.0f ? value : 0.0f;
    cursor = (cursor + 1) % samples.size();
    count = std::min(count + 1, samples.size());
}

float Renderer::DebugHistory::latest() const
{
    if (empty()) {
        return 0.0f;
    }

    const size_t index = (cursor + samples.size() - 1) % samples.size();
    return samples[index];
}

float Renderer::DebugHistory::average() const
{
    if (empty()) {
        return 0.0f;
    }

    float total = 0.0f;
    const size_t sampleCount = count == samples.size() ? samples.size() : count;
    for (size_t index = 0; index < sampleCount; ++index) {
        total += samples[index];
    }

    return total / static_cast<float>(sampleCount);
}

float Renderer::DebugHistory::max() const
{
    if (empty()) {
        return 0.0f;
    }

    float maximum = 0.0f;
    const size_t sampleCount = count == samples.size() ? samples.size() : count;
    for (size_t index = 0; index < sampleCount; ++index) {
        maximum = std::max(maximum, samples[index]);
    }

    return maximum;
}

size_t Renderer::DebugHistory::copyChronological(std::array<float, kDebugHistoryCapacity>& output) const
{
    const size_t sampleCount = count == samples.size() ? samples.size() : count;
    if (sampleCount == 0) {
        return 0;
    }

    const size_t start = count == samples.size() ? cursor : 0;
    for (size_t index = 0; index < sampleCount; ++index) {
        output[index] = samples[(start + index) % samples.size()];
    }

    return sampleCount;
}

Renderer::Renderer(Window& window) : window_(window)
{
    runtimeSettingsPath_ = defaultRuntimeSettingsPath();
    loadRuntimeSettingsAtStartup();

    context_.initialize(window_);

    frames_.resize(rhi::kMaxFramesInFlight);
    timestampQuery_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    swapchain_.initialize(context_, window_.framebufferExtent());
    imguiLayer_.initialize(window_, context_, swapchain_.colorFormat(), swapchain_.imageCount());
    createMaterialDescriptorSetLayout();
    createBindlessMaterialTextureHeap();
    createSkyboxDescriptorSetLayout();
    createPostProcessDescriptorSetLayouts();
    createPostProcessSampler();
    createShadowMap();
    createPostProcessResources();
    createPipeline();
    commandContext_.initialize(context_, frames_);
    createScene();
    createObjectFrameDataBuffers();
    createIndirectDrawBuffers();
    createShadowIndirectDrawBuffers();
    createGpuCullingResources();
    sync_.initialize(context_, frames_, swapchain_.imageCount());
    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
    currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    averageLuminance_ = toneMappingSettings_.targetLuminance;
    histogramClippedLuminance_ = toneMappingSettings_.targetLuminance;
    lastExposureLogPrint_ = std::chrono::steady_clock::now();
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();

    initialized_ = true;
}

Renderer::~Renderer()
{
    if (initialized_) {
        waitIdle();
        imguiLayer_.shutdown();
        postProcessDescriptorPool_.reset();
        destroyPostProcessSampler();
    }
}

void Renderer::drawFrame()
{
    if (window_.isMinimized()) {
        return;
    }

    updateCpuFrameTime();

    if (window_.wasResized()) {
        recreateSwapchain();
        window_.clearResizedFlag();
    }

    renderer::FrameResources& frame = frames_[currentFrame_];
    VK_CHECK(vkWaitForFences(context_.vkDevice(), 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX));
    updateAutoExposureFromReadback(currentFrame_);
    tryPrintExposureStats();
    tryPrintGpuTimings(currentFrame_);
    pushCullingHistorySample(currentFrame_);
    pushExposureHistorySample();

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

    imguiLayer_.beginFrame();
    buildDebugUi();
    imguiLayer_.endFrame();
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
    signalSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

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

void Renderer::handleEvent(const SDL_Event& event)
{
    imguiLayer_.handleEvent(event);
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
                     std::to_string(bindlessTextureHeap_.maxTextures()) + " descriptors per material texture class.");
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

void Renderer::createPostProcessDescriptorSetLayouts()
{
    VkDescriptorSetLayoutBinding singleImageBinding{};
    singleImageBinding.binding = 0;
    singleImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    singleImageBinding.descriptorCount = 1;
    singleImageBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    postProcessSingleImageDescriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(&singleImageBinding, 1));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessSingleImageDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessSingleImageDescriptorSetLayout");

    std::array<VkDescriptorSetLayoutBinding, 2> compositeBindings{};
    compositeBindings[0] = singleImageBinding;
    compositeBindings[1] = singleImageBinding;
    compositeBindings[1].binding = 1;

    postProcessCompositeDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(compositeBindings.data(), compositeBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessCompositeDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessCompositeDescriptorSetLayout");

    std::array<VkDescriptorSetLayoutBinding, 2> luminanceBindings{};
    luminanceBindings[0].binding = 0;
    luminanceBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    luminanceBindings[0].descriptorCount = 1;
    luminanceBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    luminanceBindings[1].binding = 1;
    luminanceBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    luminanceBindings[1].descriptorCount = 1;
    luminanceBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    try {
        postProcessLuminanceDescriptorSetLayout_.create(
            context_.vkDevice(),
            std::span<const VkDescriptorSetLayoutBinding>(luminanceBindings.data(), luminanceBindings.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  postProcessLuminanceDescriptorSetLayout_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                  "PostProcessLuminanceDescriptorSetLayout");
    } catch (const std::exception& error) {
        disableAutoExposureFallback(std::string("Auto exposure descriptor layout creation failed: ") + error.what());
    }
}

void Renderer::createPostProcessSampler()
{
    destroyPostProcessSampler();

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &postProcessSampler_));
    rhi::debug::setObjectName(
        context_.vkDevice(), postProcessSampler_, VK_OBJECT_TYPE_SAMPLER, "PostProcessLinearClampSampler");
}

void Renderer::destroyPostProcessSampler()
{
    if (postProcessSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), postProcessSampler_, nullptr);
        postProcessSampler_ = VK_NULL_HANDLE;
    }
}

void Renderer::createPostProcessResources()
{
    const VkExtent2D extent = swapchain_.extent();
    if (extent.width == 0 || extent.height == 0) {
        throw std::runtime_error("Cannot create post-process resources for a zero-sized swapchain extent.");
    }

    postProcessDescriptorPool_.reset();
    bloomExtractDescriptorSet_ = VK_NULL_HANDLE;
    bloomBlurHorizontalDescriptorSet_ = VK_NULL_HANDLE;
    bloomBlurVerticalDescriptorSet_ = VK_NULL_HANDLE;
    compositeDescriptorSet_ = VK_NULL_HANDLE;
    luminanceDescriptorSets_.clear();
    histogramDescriptorSets_.clear();
    destroyLuminanceResources();
    destroyHistogramResources();

    rhi::VulkanImageCreateInfo sceneColorInfo{};
    sceneColorInfo.width = extent.width;
    sceneColorInfo.height = extent.height;
    sceneColorInfo.format = kSceneColorFormat;
    sceneColorInfo.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sceneColorInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sceneColorInfo.debugName = "SceneColorHDR";
    sceneColor_.create(context_, sceneColorInfo);
    sceneColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    bloomExtent_.width = std::max(1u, extent.width / 2u);
    bloomExtent_.height = std::max(1u, extent.height / 2u);

    rhi::VulkanImageCreateInfo bloomInfo{};
    bloomInfo.width = bloomExtent_.width;
    bloomInfo.height = bloomExtent_.height;
    bloomInfo.format = kBloomColorFormat;
    bloomInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bloomInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    bloomInfo.debugName = "BloomExtract";
    bloomExtract_.create(context_, bloomInfo);
    bloomExtractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    bloomInfo.debugName = "BloomPing";
    bloomPing_.create(context_, bloomInfo);
    bloomPingLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    bloomInfo.debugName = "BloomPong";
    bloomPong_.create(context_, bloomInfo);
    bloomPongLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    try {
        createLuminanceResources();
    } catch (const std::exception& error) {
        disableLogAverageExposureFallback(
            std::string("Log-average exposure luminance resources unavailable: ") + error.what());
    }

    try {
        createHistogramResources();
    } catch (const std::exception& error) {
        disableHistogramExposureFallback(
            std::string("Histogram exposure resources unavailable: ") + error.what());
    }

    createPostProcessDescriptorSets();
}

void Renderer::createPostProcessDescriptorSets()
{
    if (postProcessSampler_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot create post-process descriptors without a sampler.");
    }

    const bool createLuminanceDescriptors =
        autoExposureAvailable_ && !frameLuminanceBuffers_.empty() &&
        frameLuminanceBuffers_.size() == frames_.size() &&
        postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    const bool createHistogramDescriptors =
        histogramExposureAvailable_ && !frameHistogramBuffers_.empty() &&
        frameHistogramBuffers_.size() == frames_.size() &&
        postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    const uint32_t exposureDescriptorSetCount =
        (createLuminanceDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (createHistogramDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 5 + exposureDescriptorSetCount;
    uint32_t poolSizeCount = 1;
    uint32_t maxSets = 4;
    if (exposureDescriptorSetCount > 0) {
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = exposureDescriptorSetCount;
        poolSizeCount = 2;
        maxSets += exposureDescriptorSetCount;
    }

    postProcessDescriptorPool_.create(
        context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizeCount), maxSets);
    rhi::debug::setObjectName(
        context_.vkDevice(), postProcessDescriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "PostProcessPool");

    std::array<VkDescriptorSetLayout, 4> descriptorSetLayouts{
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessCompositeDescriptorSetLayout_.handle(),
    };
    std::array<VkDescriptorSet, 4> descriptorSets{};

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    allocateInfo.pSetLayouts = descriptorSetLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, descriptorSets.data()));

    bloomExtractDescriptorSet_ = descriptorSets[0];
    bloomBlurHorizontalDescriptorSet_ = descriptorSets[1];
    bloomBlurVerticalDescriptorSet_ = descriptorSets[2];
    compositeDescriptorSet_ = descriptorSets[3];

    rhi::debug::setObjectName(
        context_.vkDevice(), bloomExtractDescriptorSet_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "BloomExtractDescriptorSet");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomBlurHorizontalDescriptorSet_,
                              VK_OBJECT_TYPE_DESCRIPTOR_SET,
                              "BloomBlurHorizontalDescriptorSet");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomBlurVerticalDescriptorSet_,
                              VK_OBJECT_TYPE_DESCRIPTOR_SET,
                              "BloomBlurVerticalDescriptorSet");
    rhi::debug::setObjectName(
        context_.vkDevice(), compositeDescriptorSet_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "CompositeDescriptorSet");

    const auto imageInfo = [this](VkImageView imageView) {
        VkDescriptorImageInfo info{};
        info.sampler = postProcessSampler_;
        info.imageView = imageView;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return info;
    };

    std::array<VkDescriptorImageInfo, 5> imageInfos{
        imageInfo(sceneColor_.imageView()),
        imageInfo(bloomExtract_.imageView()),
        imageInfo(bloomPing_.imageView()),
        imageInfo(sceneColor_.imageView()),
        imageInfo(bloomPong_.imageView()),
    };

    std::array<VkWriteDescriptorSet, 5> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = bloomExtractDescriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &imageInfos[0];

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = bloomBlurHorizontalDescriptorSet_;
    writes[1].dstBinding = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &imageInfos[1];

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = bloomBlurVerticalDescriptorSet_;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &imageInfos[2];

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = compositeDescriptorSet_;
    writes[3].dstBinding = 0;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &imageInfos[3];

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = compositeDescriptorSet_;
    writes[4].dstBinding = 1;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &imageInfos[4];

    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkDescriptorImageInfo sceneColorInfo{};
    sceneColorInfo.sampler = postProcessSampler_;
    sceneColorInfo.imageView = sceneColor_.imageView();
    sceneColorInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    if (createLuminanceDescriptors) {
        try {
            luminanceDescriptorSets_.assign(frames_.size(), VK_NULL_HANDLE);
            std::vector<VkDescriptorSetLayout> luminanceLayouts(frames_.size(),
                                                                postProcessLuminanceDescriptorSetLayout_.handle());
            VkDescriptorSetAllocateInfo luminanceAllocateInfo{};
            luminanceAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            luminanceAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            luminanceAllocateInfo.descriptorSetCount = static_cast<uint32_t>(luminanceDescriptorSets_.size());
            luminanceAllocateInfo.pSetLayouts = luminanceLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(
                context_.vkDevice(), &luminanceAllocateInfo, luminanceDescriptorSets_.data()));

            for (size_t frameIndex = 0; frameIndex < luminanceDescriptorSets_.size(); ++frameIndex) {
                VkDescriptorBufferInfo luminanceBufferInfo{};
                luminanceBufferInfo.buffer = frameLuminanceBuffers_[frameIndex].buffer();
                luminanceBufferInfo.offset = 0;
                luminanceBufferInfo.range = frameLuminanceBuffers_[frameIndex].size();

                std::array<VkWriteDescriptorSet, 2> luminanceWrites{};
                luminanceWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                luminanceWrites[0].dstSet = luminanceDescriptorSets_[frameIndex];
                luminanceWrites[0].dstBinding = 0;
                luminanceWrites[0].descriptorCount = 1;
                luminanceWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                luminanceWrites[0].pImageInfo = &sceneColorInfo;

                luminanceWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                luminanceWrites[1].dstSet = luminanceDescriptorSets_[frameIndex];
                luminanceWrites[1].dstBinding = 1;
                luminanceWrites[1].descriptorCount = 1;
                luminanceWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                luminanceWrites[1].pBufferInfo = &luminanceBufferInfo;

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(luminanceWrites.size()),
                                       luminanceWrites.data(),
                                       0,
                                       nullptr);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          luminanceDescriptorSets_[frameIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "LuminanceDescriptorSet" + std::to_string(frameIndex));
            }
        } catch (const std::exception& error) {
            luminanceDescriptorSets_.clear();
            disableLogAverageExposureFallback(
                std::string("Log-average exposure descriptor allocation failed: ") + error.what());
        }
    }

    if (createHistogramDescriptors) {
        try {
            histogramDescriptorSets_.assign(frames_.size(), VK_NULL_HANDLE);
            std::vector<VkDescriptorSetLayout> histogramLayouts(frames_.size(),
                                                                postProcessLuminanceDescriptorSetLayout_.handle());
            VkDescriptorSetAllocateInfo histogramAllocateInfo{};
            histogramAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            histogramAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            histogramAllocateInfo.descriptorSetCount = static_cast<uint32_t>(histogramDescriptorSets_.size());
            histogramAllocateInfo.pSetLayouts = histogramLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(
                context_.vkDevice(), &histogramAllocateInfo, histogramDescriptorSets_.data()));

            for (size_t frameIndex = 0; frameIndex < histogramDescriptorSets_.size(); ++frameIndex) {
                VkDescriptorBufferInfo histogramBufferInfo{};
                histogramBufferInfo.buffer = frameHistogramBuffers_[frameIndex].buffer();
                histogramBufferInfo.offset = 0;
                histogramBufferInfo.range = frameHistogramBuffers_[frameIndex].size();

                std::array<VkWriteDescriptorSet, 2> histogramWrites{};
                histogramWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                histogramWrites[0].dstSet = histogramDescriptorSets_[frameIndex];
                histogramWrites[0].dstBinding = 0;
                histogramWrites[0].descriptorCount = 1;
                histogramWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                histogramWrites[0].pImageInfo = &sceneColorInfo;

                histogramWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                histogramWrites[1].dstSet = histogramDescriptorSets_[frameIndex];
                histogramWrites[1].dstBinding = 1;
                histogramWrites[1].descriptorCount = 1;
                histogramWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                histogramWrites[1].pBufferInfo = &histogramBufferInfo;

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(histogramWrites.size()),
                                       histogramWrites.data(),
                                       0,
                                       nullptr);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          histogramDescriptorSets_[frameIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "HistogramDescriptorSet" + std::to_string(frameIndex));
            }
        } catch (const std::exception& error) {
            histogramDescriptorSets_.clear();
            disableHistogramExposureFallback(
                std::string("Histogram exposure descriptor allocation failed: ") + error.what());
        }
    }
}

void Renderer::createLuminanceResources()
{
    destroyLuminanceResources();

    if (!toneMappingSettings_.enableAutoExposure) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        return;
    }
    if (postProcessLuminanceDescriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("missing luminance descriptor set layout");
    }

    const VkExtent3D sceneExtent = sceneColor_.extent();
    if (sceneExtent.width == 0 || sceneExtent.height == 0) {
        throw std::runtime_error("scene color extent is zero");
    }

    luminanceGroupCountX_ = (sceneExtent.width + kLuminanceLocalSizeX - 1) / kLuminanceLocalSizeX;
    luminanceGroupCountY_ = (sceneExtent.height + kLuminanceLocalSizeY - 1) / kLuminanceLocalSizeY;
    luminancePartialCount_ = luminanceGroupCountX_ * luminanceGroupCountY_;
    if (luminancePartialCount_ == 0) {
        throw std::runtime_error("luminance reduction produced zero workgroups");
    }

    const VkDeviceSize luminanceBufferSize =
        static_cast<VkDeviceSize>(luminancePartialCount_) * sizeof(LuminancePartial);

    frameLuminanceBuffers_.resize(frames_.size());
    frameLuminanceReadbackBuffers_.resize(frames_.size());
    frameLuminanceReadbackReady_.assign(frames_.size(), 0);

    for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo luminanceInfo{};
        luminanceInfo.size = luminanceBufferSize;
        luminanceInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        luminanceInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameLuminanceBuffers_[frameIndex].createBuffer(context_, luminanceInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameLuminanceBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminancePartialBuffer" + std::to_string(frameIndex));

        rhi::VulkanBufferCreateInfo readbackInfo{};
        readbackInfo.size = luminanceBufferSize;
        readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        readbackInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        readbackInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        frameLuminanceReadbackBuffers_[frameIndex].createBuffer(context_, readbackInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameLuminanceReadbackBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminanceReadbackBuffer" + std::to_string(frameIndex));
    }

    autoExposureAvailable_ = true;
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
}

void Renderer::destroyLuminanceResources()
{
    autoExposureAvailable_ = false;
    luminanceDescriptorSets_.clear();
    frameLuminanceReadbackReady_.clear();
    frameLuminanceReadbackBuffers_.clear();
    frameLuminanceBuffers_.clear();
    luminancePartialCount_ = 0;
    luminanceGroupCountX_ = 0;
    luminanceGroupCountY_ = 0;
}

void Renderer::createHistogramResources()
{
    destroyHistogramResources();

    if (!toneMappingSettings_.enableAutoExposure) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        return;
    }
    if (postProcessLuminanceDescriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("missing exposure descriptor set layout");
    }

    const VkExtent3D sceneExtent = sceneColor_.extent();
    if (sceneExtent.width == 0 || sceneExtent.height == 0) {
        throw std::runtime_error("scene color extent is zero");
    }

    const VkDeviceSize histogramBufferSize = static_cast<VkDeviceSize>(kHistogramBinCount * sizeof(uint32_t));

    frameHistogramBuffers_.resize(frames_.size());
    frameHistogramReadbackBuffers_.resize(frames_.size());
    frameHistogramReadbackReady_.assign(frames_.size(), 0);

    for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo histogramInfo{};
        histogramInfo.size = histogramBufferSize;
        histogramInfo.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        histogramInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameHistogramBuffers_[frameIndex].createBuffer(context_, histogramInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameHistogramBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminanceHistogramBuffer" + std::to_string(frameIndex));

        rhi::VulkanBufferCreateInfo readbackInfo{};
        readbackInfo.size = histogramBufferSize;
        readbackInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        readbackInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        readbackInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        frameHistogramReadbackBuffers_[frameIndex].createBuffer(context_, readbackInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameHistogramReadbackBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminanceHistogramReadbackBuffer" + std::to_string(frameIndex));
    }

    histogramExposureAvailable_ = true;
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
}

void Renderer::destroyHistogramResources()
{
    histogramExposureAvailable_ = false;
    histogramDescriptorSets_.clear();
    frameHistogramReadbackReady_.clear();
    frameHistogramReadbackBuffers_.clear();
    frameHistogramBuffers_.clear();
}

void Renderer::disableAutoExposureFallback(std::string_view reason)
{
    if (!autoExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; disabling auto exposure and using manual exposure.");
        autoExposureWarningLogged_ = true;
    }

    toneMappingSettings_.enableAutoExposure = false;
    currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    destroyLuminanceResources();
    destroyHistogramResources();
    luminancePipeline_.reset();
    histogramPipeline_.reset();
}

void Renderer::disableLogAverageExposureFallback(std::string_view reason)
{
    if (!logAverageExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; log-average exposure fallback unavailable.");
        logAverageExposureWarningLogged_ = true;
    }

    destroyLuminanceResources();
    luminancePipeline_.reset();
    if (!isHistogramExposureActive()) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }
}

void Renderer::disableHistogramExposureFallback(std::string_view reason)
{
    if (!histogramExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; falling back to log-average exposure when available.");
        histogramExposureWarningLogged_ = true;
    }

    destroyHistogramResources();
    histogramPipeline_.reset();
    if (!isLogAverageExposureActive()) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }
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

        gpuCullDescriptorPool_.create(context_.vkDevice(),
                                      std::span<const VkDescriptorPoolSize>(&poolSize, 1),
                                      static_cast<uint32_t>(frames_.size()));
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

            vkUpdateDescriptorSets(
                context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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

        shadowCullDescriptorPool_.create(context_.vkDevice(),
                                         std::span<const VkDescriptorPoolSize>(&poolSize, 1),
                                         static_cast<uint32_t>(frames_.size()));
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

            vkUpdateDescriptorSets(
                context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      shadowCullDescriptorSets_[frameIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "GpuShadowCullDescriptorSet" + std::to_string(frameIndex));
        }

        gpuShadowCullingAvailable_ = true;
        Logger::info("GPU shadow culling preparation enabled with per-frame shadow cull input, compacted indirect "
                     "output, and per-batch visible count buffers.");
    } catch (const std::exception& error) {
        Logger::warn(std::string("GPU shadow culling unavailable; using CPU shadow culling fallback: ") + error.what());
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
    const VkDescriptorSetLayout postProcessSingleImageDescriptorSetLayout =
        postProcessSingleImageDescriptorSetLayout_.handle();
    const VkDescriptorSetLayout postProcessCompositeDescriptorSetLayout =
        postProcessCompositeDescriptorSetLayout_.handle();
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};
    const VkPushConstantRange skyboxPushConstantRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      0,
                                                      static_cast<uint32_t>(sizeof(SkyboxPushConstants))};
    const VkPushConstantRange bloomExtractPushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomExtractPushConstants))};
    const VkPushConstantRange bloomBlurPushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomBlurPushConstants))};
    const VkPushConstantRange compositePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(CompositePushConstants))};

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple.vert.spv");
    pipelineInfo.fragmentShaderPath =
        bindlessMaterialTexturesActive ? shaderPath("simple_bindless.frag.spv") : shaderPath("simple.frag.spv");
    pipelineInfo.colorFormat = kSceneColorFormat;
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
    skyboxPipelineInfo.colorFormat = kSceneColorFormat;
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
    const VkPushConstantRange shadowPushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};
    shadowPipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&shadowPushConstantRange, 1);
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

    rhi::VulkanPipelineCreateInfo bloomExtractPipelineInfo{};
    bloomExtractPipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomExtractPipelineInfo.fragmentShaderPath = shaderPath("bloom_extract.frag.spv");
    bloomExtractPipelineInfo.colorFormat = kBloomColorFormat;
    bloomExtractPipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessSingleImageDescriptorSetLayout, 1);
    bloomExtractPipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&bloomExtractPushConstantRange, 1);

    bloomExtractPipeline_.create(context_.vkDevice(), bloomExtractPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomExtractPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "BloomExtractPipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomExtractPipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "BloomExtractPipelineLayout");
    bloomExtractPipelineColorFormat_ = bloomExtractPipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo bloomBlurPipelineInfo{};
    bloomBlurPipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomBlurPipelineInfo.fragmentShaderPath = shaderPath("bloom_blur.frag.spv");
    bloomBlurPipelineInfo.colorFormat = kBloomColorFormat;
    bloomBlurPipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessSingleImageDescriptorSetLayout, 1);
    bloomBlurPipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&bloomBlurPushConstantRange, 1);

    bloomBlurPipeline_.create(context_.vkDevice(), bloomBlurPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomBlurPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "BloomBlurPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), bloomBlurPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "BloomBlurPipelineLayout");
    bloomBlurPipelineColorFormat_ = bloomBlurPipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo compositePipelineInfo{};
    compositePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    compositePipelineInfo.fragmentShaderPath = shaderPath("composite.frag.spv");
    compositePipelineInfo.colorFormat = swapchain_.colorFormat();
    compositePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessCompositeDescriptorSetLayout, 1);
    compositePipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&compositePushConstantRange, 1);

    compositePipeline_.create(context_.vkDevice(), compositePipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), compositePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "CompositePipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), compositePipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "CompositePipelineLayout");
    compositePipelineColorFormat_ = compositePipelineInfo.colorFormat;

    luminancePipeline_.reset();
    histogramPipeline_.reset();
    if (toneMappingSettings_.enableAutoExposure &&
        postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE) {
        const VkDescriptorSetLayout exposureDescriptorSetLayout = postProcessLuminanceDescriptorSetLayout_.handle();
        try {
            const VkPushConstantRange luminancePushConstantRange{
                VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(LuminancePushConstants))};

            rhi::VulkanComputePipelineCreateInfo luminancePipelineInfo{};
            luminancePipelineInfo.shaderPath = shaderPath("luminance.comp.spv");
            luminancePipelineInfo.descriptorSetLayouts =
                std::span<const VkDescriptorSetLayout>(&exposureDescriptorSetLayout, 1);
            luminancePipelineInfo.pushConstantRanges =
                std::span<const VkPushConstantRange>(&luminancePushConstantRange, 1);
            luminancePipeline_.create(context_.vkDevice(), luminancePipelineInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      luminancePipeline_.pipeline(),
                                      VK_OBJECT_TYPE_PIPELINE,
                                      "AutoExposureComputePipeline");
            rhi::debug::setObjectName(context_.vkDevice(),
                                      luminancePipeline_.layout(),
                                      VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                      "AutoExposurePipelineLayout");
        } catch (const std::exception& error) {
            disableLogAverageExposureFallback(
                std::string("Log-average exposure compute pipeline creation failed: ") + error.what());
        }

        try {
            const VkPushConstantRange histogramPushConstantRange{
                VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(HistogramPushConstants))};

            rhi::VulkanComputePipelineCreateInfo histogramPipelineInfo{};
            histogramPipelineInfo.shaderPath = shaderPath("luminance_histogram.comp.spv");
            histogramPipelineInfo.descriptorSetLayouts =
                std::span<const VkDescriptorSetLayout>(&exposureDescriptorSetLayout, 1);
            histogramPipelineInfo.pushConstantRanges =
                std::span<const VkPushConstantRange>(&histogramPushConstantRange, 1);
            histogramPipeline_.create(context_.vkDevice(), histogramPipelineInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      histogramPipeline_.pipeline(),
                                      VK_OBJECT_TYPE_PIPELINE,
                                      "HistogramExposureComputePipeline");
            rhi::debug::setObjectName(context_.vkDevice(),
                                      histogramPipeline_.layout(),
                                      VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                      "HistogramExposurePipelineLayout");
        } catch (const std::exception& error) {
            disableHistogramExposureFallback(
                std::string("Histogram exposure compute pipeline creation failed: ") + error.what());
        }
    }
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
    importedBaseColorTextures_.clear();
    importedNormalTextures_.clear();
    importedMetallicRoughnessTextures_.clear();

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
        addCube("Center Cube", &materialVariants_.at(0), {0.0f, -0.1f, 0.0f}, {0.2f, 0.0f, 0.0f}, {0.7f, 0.7f, 0.7f});
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
            createImportedGltfTextures(loadedAsset.textures, loadedAsset.materials);
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
                importedObject.debugName = instance.debugName.empty() ? "Imported glTF Node" : instance.debugName;
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
            checkerboardTexture_.createFromFile(
                context_, commandContext_, texturePath, rhi::TextureColorSpace::SRGB, true);
            nameTextureResources(checkerboardTexture_, "BaseColorTexture");
            Logger::info("Loaded base color texture as sRGB: " + texturePath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Texture asset missing, using procedural checkerboard fallback: " + texturePath.string());
    }

    checkerboardTexture_.createCheckerboard(context_, commandContext_, 256, 256, rhi::TextureColorSpace::SRGB);
    nameTextureResources(checkerboardTexture_, "BaseColorTexture");
    Logger::info("Created procedural checkerboard base color texture as sRGB.");
}

void Renderer::createNormalTexture()
{
    normalMapAssetLoaded_ = false;
    bool loadedAsset = false;

    const std::filesystem::path texturePath = assetPath("textures/checker_normal.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            normalMapTexture_.createFromFile(
                context_, commandContext_, texturePath, rhi::TextureColorSpace::Linear, true);
            normalMapAssetLoaded_ = true;
            nameTextureResources(normalMapTexture_, "NormalTexture");
            Logger::info("Loaded normal texture as linear UNORM: " + texturePath.string());
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
        Logger::info("Created procedural flat normal texture as linear UNORM.");
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
            metallicRoughnessTexture_.createFromFile(
                context_, commandContext_, texturePath, rhi::TextureColorSpace::Linear, true);
            metallicRoughnessMapAssetLoaded_ = true;
            nameTextureResources(metallicRoughnessTexture_, "MetallicRoughnessTexture");
            Logger::info("Loaded metallic-roughness texture as linear UNORM: " + texturePath.string());
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
        Logger::info("Created procedural neutral metallic-roughness texture as linear UNORM.");
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
    const std::filesystem::path hdrEnvironmentPath = assetPath("environments/studio.hdr");
    bool loadedHdrEnvironment = false;
    if (std::filesystem::exists(hdrEnvironmentPath)) {
        try {
            constexpr uint32_t kHdrEnvironmentFaceSize = 128;
            const rhi::HdrEnvironmentCubeData hdrCubeData =
                rhi::VulkanEnvironmentMap::loadHdrEquirectangularFaces(hdrEnvironmentPath, kHdrEnvironmentFaceSize);
            const std::span<const float> hdrCubePixels(hdrCubeData.rgba32fPixels.data(),
                                                       hdrCubeData.rgba32fPixels.size());

            environmentMap_.createFromRgba32fFaces(context_, commandContext_, hdrCubeData.faceSize, hdrCubePixels);
            nameEnvironmentMapResources(environmentMap_, "HdrEnvironmentCubemap");

            diffuseIrradianceMap_.createDiffuseIrradianceFromRgba32fFaces(
                context_, commandContext_, hdrCubeData.faceSize, hdrCubePixels, 32);
            nameEnvironmentMapResources(diffuseIrradianceMap_, "HdrDiffuseIrradianceCubemap");

            prefilteredEnvironmentMap_.createPrefilteredSpecularFromRgba32fFaces(
                context_, commandContext_, hdrCubeData.faceSize, hdrCubePixels, 64);
            nameEnvironmentMapResources(prefilteredEnvironmentMap_, "HdrPrefilteredSpecularCubemap");

            loadedHdrEnvironment = true;
            Logger::info(std::string("Loaded HDR environment from ") + hdrEnvironmentPath.string() +
                         " with approximate CPU equirectangular-to-cubemap conversion.");
        } catch (const std::exception& error) {
            Logger::warn(std::string("Failed to load HDR environment '") + hdrEnvironmentPath.string() +
                         "'; using procedural environment fallback: " + error.what());
            environmentMap_.reset();
            diffuseIrradianceMap_.reset();
            prefilteredEnvironmentMap_.reset();
        }
    } else {
        Logger::info(std::string("Optional HDR environment not found at ") + hdrEnvironmentPath.string() +
                     "; using procedural environment fallback.");
    }

    if (!loadedHdrEnvironment) {
        environmentMap_.createProcedural(context_, commandContext_, 32);
        nameEnvironmentMapResources(environmentMap_, "EnvironmentCubemap");
        createDiffuseIrradianceMap();
        createPrefilteredEnvironmentMap();
        Logger::info("Created procedural environment cubemaps for skybox, diffuse IBL, and specular IBL.");
    }
    hdrEnvironmentLoaded_ = loadedHdrEnvironment;

    createBrdfLutTexture();
    createSkyboxDescriptorSet();
    Logger::info("Created BRDF LUT for split-sum specular IBL.");
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
    material.normalTextureIndex = material.normalTexture && material.normalTexture->valid()
                                      ? bindlessTextureHeap_.registerTexture(
                                            renderer::BindlessTextureHeap::TextureKind::Normal, *material.normalTexture)
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

void Renderer::createImportedGltfTextures(const std::vector<renderer::GltfTextureInfo>& textureInfos,
                                          const std::vector<renderer::GltfMaterialInfo>& materialInfos)
{
    importedBaseColorTextures_.clear();
    importedNormalTextures_.clear();
    importedMetallicRoughnessTextures_.clear();
    importedBaseColorTextures_.resize(textureInfos.size());
    importedNormalTextures_.resize(textureInfos.size());
    importedMetallicRoughnessTextures_.resize(textureInfos.size());

    std::vector<uint8_t> baseColorNeeded(textureInfos.size(), 0);
    std::vector<uint8_t> normalNeeded(textureInfos.size(), 0);
    std::vector<uint8_t> metallicRoughnessNeeded(textureInfos.size(), 0);

    const auto markNeeded = [textureCount = textureInfos.size()](int textureIndex, std::vector<uint8_t>& needed) {
        if (textureIndex >= 0 && static_cast<size_t>(textureIndex) < textureCount) {
            needed[static_cast<size_t>(textureIndex)] = 1;
        }
    };

    for (const renderer::GltfMaterialInfo& materialInfo : materialInfos) {
        markNeeded(materialInfo.baseColorTextureIndex, baseColorNeeded);
        markNeeded(materialInfo.normalTextureIndex, normalNeeded);
        markNeeded(materialInfo.metallicRoughnessTextureIndex, metallicRoughnessNeeded);
    }

    const auto loadTexture = [this, &textureInfos](size_t textureIndex,
                                                   rhi::TextureColorSpace colorSpace,
                                                   std::string_view slotName,
                                                   std::string_view debugPrefix,
                                                   std::vector<rhi::VulkanTexture>& textures) {
        const renderer::GltfTextureInfo& textureInfo = textureInfos[textureIndex];
        if (textureInfo.path.empty() && textureInfo.encodedData.empty()) {
            return;
        }

        try {
            if (!textureInfo.path.empty()) {
                if (!std::filesystem::exists(textureInfo.path)) {
                    Logger::warn("glTF texture image is missing; material fallback will be used: " +
                                 textureInfo.path.string());
                    return;
                }

                textures[textureIndex].createFromFile(context_, commandContext_, textureInfo.path, colorSpace, true);
                Logger::info("Loaded glTF " + std::string(slotName) + " texture as " +
                             std::string(colorSpaceName(colorSpace)) + ": " + textureInfo.path.string());
            } else {
                textures[textureIndex].createFromEncodedBytes(
                    context_,
                    commandContext_,
                    std::span<const uint8_t>(textureInfo.encodedData.data(), textureInfo.encodedData.size()),
                    colorSpace,
                    true);
                Logger::info("Loaded embedded glTF " + std::string(slotName) + " texture as " +
                             std::string(colorSpaceName(colorSpace)) + ": " + textureInfo.debugName);
            }

            nameTextureResources(textures[textureIndex], std::string(debugPrefix) + std::to_string(textureIndex));
        } catch (const std::exception& error) {
            const std::string textureName =
                !textureInfo.path.empty() ? textureInfo.path.string() : textureInfo.debugName;
            Logger::warn("Failed to load glTF " + std::string(slotName) + " texture '" + textureName +
                         "'; material fallback will be used: " + error.what());
        }
    };

    for (size_t textureIndex = 0; textureIndex < textureInfos.size(); ++textureIndex) {
        if (baseColorNeeded[textureIndex] != 0) {
            loadTexture(textureIndex,
                        rhi::TextureColorSpace::SRGB,
                        "base color",
                        "GltfBaseColorTexture",
                        importedBaseColorTextures_);
        }
        if (normalNeeded[textureIndex] != 0) {
            loadTexture(
                textureIndex, rhi::TextureColorSpace::Linear, "normal", "GltfNormalTexture", importedNormalTextures_);
        }
        if (metallicRoughnessNeeded[textureIndex] != 0) {
            loadTexture(textureIndex,
                        rhi::TextureColorSpace::Linear,
                        "metallic-roughness",
                        "GltfMetallicRoughnessTexture",
                        importedMetallicRoughnessTextures_);
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

    const auto textureOrFallback = [](int textureIndex,
                                      const std::vector<rhi::VulkanTexture>& textures,
                                      const rhi::VulkanTexture& fallbackTexture) -> const rhi::VulkanTexture* {
        if (textureIndex >= 0 && static_cast<size_t>(textureIndex) < textures.size() &&
            textures[static_cast<size_t>(textureIndex)].valid()) {
            return &textures[static_cast<size_t>(textureIndex)];
        }
        return &fallbackTexture;
    };

    const auto textureLoaded = [](int textureIndex, const std::vector<rhi::VulkanTexture>& textures) {
        return textureIndex >= 0 && static_cast<size_t>(textureIndex) < textures.size() &&
               textures[static_cast<size_t>(textureIndex)].valid();
    };

    for (const renderer::GltfMaterialInfo& materialInfo : *sourceMaterialInfos) {
        renderer::Material material{};
        material.debugName = materialInfo.debugName.empty() ? "glTF Material" : materialInfo.debugName;
        material.baseColorTexture =
            textureOrFallback(materialInfo.baseColorTextureIndex, importedBaseColorTextures_, checkerboardTexture_);
        material.normalTexture =
            textureOrFallback(materialInfo.normalTextureIndex, importedNormalTextures_, flatNormalTexture_);
        material.metallicRoughnessTexture = textureOrFallback(materialInfo.metallicRoughnessTextureIndex,
                                                              importedMetallicRoughnessTextures_,
                                                              neutralMetallicRoughnessTexture_);
        material.baseColorFactor = materialInfo.baseColorFactor;
        material.metallic = materialInfo.metallic;
        material.roughness = materialInfo.roughness;
        material.multiScatterStrength = 1.0f;
        material.hasNormalMap = textureLoaded(materialInfo.normalTextureIndex, importedNormalTextures_);
        material.hasMetallicRoughnessMap =
            textureLoaded(materialInfo.metallicRoughnessTextureIndex, importedMetallicRoughnessTextures_);

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
    const glm::vec3 lightUp = std::abs(glm::dot(lightDirection, glm::vec3{0.0f, 1.0f, 0.0f})) > 0.95f
                                  ? glm::vec3{0.0f, 0.0f, 1.0f}
                                  : glm::vec3{0.0f, 1.0f, 0.0f};
    const glm::vec3 lightRight = glm::normalize(glm::cross(lightDirection, lightUp));
    const glm::vec3 lightBasisUp = glm::normalize(glm::cross(lightRight, lightDirection));

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

        // Each cascade still starts from the readable fitted bounds of the
        // camera-frustum slice between cascadeNear and cascadeFar.
        writeDepthCorners(cascadeNear, 0);
        writeDepthCorners(cascadeFar, 4);

        glm::vec3 cascadeCenter{0.0f};
        for (const glm::vec3& corner : corners) {
            cascadeCenter += corner;
        }
        cascadeCenter /= static_cast<float>(corners.size());

        const glm::mat4 fitLightView = glm::lookAt(cascadeCenter - lightDirection, cascadeCenter, lightUp);
        glm::vec3 minBounds{std::numeric_limits<float>::infinity()};
        glm::vec3 maxBounds{-std::numeric_limits<float>::infinity()};
        for (const glm::vec3& corner : corners) {
            const glm::vec3 lightSpaceCorner = glm::vec3(fitLightView * glm::vec4(corner, 1.0f));
            minBounds = glm::min(minBounds, lightSpaceCorner);
            maxBounds = glm::max(maxBounds, lightSpaceCorner);
        }

        const float orthoWidth = std::max(maxBounds.x - minBounds.x, 0.001f);
        const float orthoHeight = std::max(maxBounds.y - minBounds.y, 0.001f);
        const float orthoExtent = std::max(orthoWidth, orthoHeight);
        const float shadowResolution = static_cast<float>(std::max(shadowSettings_.resolution, 1U));
        const float worldUnitsPerTexel = orthoExtent / shadowResolution;

        glm::vec3 lightViewCenter = cascadeCenter;
        if (csmSettings_.enableTexelSnapping && worldUnitsPerTexel > std::numeric_limits<float>::epsilon()) {
            // CSM shimmering happens when the camera moves by a sub-texel amount
            // in the light projection: static receivers then sample a slightly
            // different part of the shadow map every frame. Snapping the
            // light-view center to worldUnitsPerTexel increments keeps the
            // shadow texel grid from sliding continuously with the camera. This
            // is a basic stabilization step, not a full production CSM solution
            // with stable crop matrices, cascade blending, or per-cascade tuning.
            const float centerX = glm::dot(lightViewCenter, lightRight);
            const float centerY = glm::dot(lightViewCenter, lightBasisUp);
            const float snappedCenterX = std::round(centerX / worldUnitsPerTexel) * worldUnitsPerTexel;
            const float snappedCenterY = std::round(centerY / worldUnitsPerTexel) * worldUnitsPerTexel;
            lightViewCenter += lightRight * (snappedCenterX - centerX) + lightBasisUp * (snappedCenterY - centerY);

            // The orthographic bounds were fitted before moving the light view
            // by less than one shadow texel, so add a one-texel guard band to
            // preserve coverage after the quantized center shift.
            minBounds.x -= worldUnitsPerTexel;
            minBounds.y -= worldUnitsPerTexel;
            maxBounds.x += worldUnitsPerTexel;
            maxBounds.y += worldUnitsPerTexel;
        }

        const glm::mat4 lightView = glm::lookAt(lightViewCenter - lightDirection, lightViewCenter, lightUp);
        const float depthRange = std::max(cascadeFar - cascadeNear, 1.0f);
        const float zPadding = std::max(depthRange * 2.0f, 10.0f);
        const float orthoNear = std::max(0.001f, -maxBounds.z - zPadding);
        const float orthoFar = std::max(orthoNear + 0.001f, -minBounds.z + zPadding);

        glm::mat4 lightProjection = glm::ortho(minBounds.x, maxBounds.x, minBounds.y, maxBounds.y, orthoNear, orthoFar);
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

void Renderer::buildMeshDrawBatchesForItems(const std::vector<DrawItem>& drawItems,
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
            gpuDrawItem.boundsMin =
                glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
            gpuDrawItem.boundsMax = glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
        }

        gpuDrawItem.indexCount = drawItem.indexCount;
        gpuDrawItem.firstIndex = drawItem.firstIndex;
        gpuDrawItem.vertexOffset = drawItem.vertexOffset;
        gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;
    }

    for (size_t batchIndex = 0; batchIndex < meshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = meshDrawBatches_[batchIndex];
        const uint32_t endDrawItem =
            std::min<uint32_t>(batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
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
            gpuDrawItem.boundsMin =
                glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
            gpuDrawItem.boundsMax = glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
        }

        gpuDrawItem.indexCount = drawItem.indexCount;
        gpuDrawItem.firstIndex = drawItem.firstIndex;
        gpuDrawItem.vertexOffset = drawItem.vertexOffset;
        gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;
    }

    for (size_t batchIndex = 0; batchIndex < gpuShadowMeshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = gpuShadowMeshDrawBatches_[batchIndex];
        const uint32_t endDrawItem =
            std::min<uint32_t>(batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
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
        .upload(std::as_bytes(
            std::span<const VkDrawIndexedIndirectCommand>(indirectCommands.data(), indirectCommands.size())));
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
        .upload(std::as_bytes(
            std::span<const VkDrawIndexedIndirectCommand>(indirectCommands.data(), indirectCommands.size())));
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

void Renderer::tryPrintExposureStats()
{
    const auto now = std::chrono::steady_clock::now();
    if (now - lastExposureLogPrint_ < std::chrono::seconds(1)) {
        return;
    }

    lastExposureLogPrint_ = now;
    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    const auto [lowPercentile, highPercentile] =
        sanitizedPercentileRange(toneMappingSettings_.lowPercentile, toneMappingSettings_.highPercentile);

    std::ostringstream message;
    message << std::fixed << std::setprecision(4) << "Exposure:\n"
            << "  mode: " << exposureModeName(mode) << "\n"
            << "  average luminance: " << averageLuminance_ << "\n"
            << "  histogram clipped luminance: " << histogramClippedLuminance_ << "\n"
            << "  exposure: " << currentToneMappingExposure() << "\n"
            << "  low percentile: " << lowPercentile << "\n"
            << "  high percentile: " << highPercentile;
    Logger::info(message.str());
}

void Renderer::tryPrintGpuTimings(uint32_t frameIndex)
{
    rhi::VulkanTimestampQuery::Results results{};
    if (!timestampQuery_.readFrame(frameIndex, results) || !results.valid) {
        return;
    }
    latestGpuTimings_ = results;
    pushGpuTimingSample(results);

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
            << "  RenderObjects: " << results.renderObjectsMs << " ms\n"
            << "  Bloom: " << results.bloomMs << " ms\n"
            << "  AutoExposure: " << results.autoExposureMs << " ms\n"
            << "  HistogramExposure: " << results.histogramExposureMs << " ms\n"
            << "  Composite: " << results.compositeMs << " ms\n";
    if (cullingStats_.gpuCulling) {
        const uint32_t totalDrawItems = frameIndex < frameGpuCullTotalDrawItems_.size()
                                            ? frameGpuCullTotalDrawItems_[frameIndex]
                                            : static_cast<uint32_t>(cullingStats_.totalDrawItems);
        const uint32_t batchCount = frameIndex < frameGpuCullBatchCounts_.size()
                                        ? frameGpuCullBatchCounts_[frameIndex]
                                        : static_cast<uint32_t>(cullingStats_.batchCount);
        uint32_t visibleDrawItems = 0;
        if (readGpuVisibleCount(frameIndex, visibleDrawItems)) {
            const uint32_t culledDrawItems = totalDrawItems > visibleDrawItems ? totalDrawItems - visibleDrawItems : 0;
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
        message << "Culling: total=" << cullingStats_.totalObjects << " visible=" << cullingStats_.visibleObjects
                << " culled=" << cullingStats_.culledObjects << " drawItems=" << cullingStats_.totalDrawItems
                << " meshBatches=" << cullingStats_.batchCount << " commandCount=" << cullingStats_.commandCount;
    }
    message << "\nCSM:\n"
            << "  cascades: " << activeCascadeCount() << "\n"
            << "  texel snapping: " << (csmSettings_.enableTexelSnapping ? "enabled" : "disabled") << "\n"
            << "  debug colors: " << (csmSettings_.enableCascadeDebugColors ? "enabled" : "disabled") << "\n";
    message << "Shadow culling:\n";
    message << "  cascade count: " << shadowCullingStats_.cascadeCount << "\n"
            << "  total shadow draw items across cascades: " << shadowCullingStats_.totalDrawItems << "\n"
            << "  visible shadow draw items across cascades: " << shadowCullingStats_.visibleDrawItems << "\n"
            << "  culled shadow draw items across cascades: " << shadowCullingStats_.culledDrawItems << "\n"
            << "  shadow batches across cascades: " << shadowCullingStats_.batchCount << "\n";
    for (size_t cascadeIndex = 0;
         cascadeIndex < shadowCullingStats_.cascadeCount && cascadeIndex < shadowVisibleDrawItemsPerCascade_.size();
         ++cascadeIndex) {
        message << "  cascade " << cascadeIndex << ": visible draw items "
                << shadowVisibleDrawItemsPerCascade_[cascadeIndex] << ", batches "
                << shadowBatchCountPerCascade_[cascadeIndex] << ", split depth "
                << frameCascades_[cascadeIndex].splitDepth << "\n";
    }
    message << "  GPU shadow culling: " << (shadowCullingStats_.gpuCulling ? "enabled" : "disabled") << "\n"
            << "  shadow path: "
            << (shadowCullingStats_.gpuCulling
                    ? (isShadowIndirectCountPathActive(frameIndex) ? "per-cascade indirect count"
                                                                   : "per-cascade indirect fallback")
                    : "per-cascade direct fallback");
    Logger::info(message.str());
}

void Renderer::updateCpuFrameTime()
{
    const auto now = std::chrono::steady_clock::now();
    cpuFrameDeltaMs_ = std::chrono::duration<float, std::milli>(now - lastFrameStartTime_).count();
    lastFrameStartTime_ = now;
    cpuFps_ = cpuFrameDeltaMs_ > 0.0f ? 1000.0f / cpuFrameDeltaMs_ : 0.0f;
}

void Renderer::pushGpuTimingSample(const rhi::VulkanTimestampQuery::Results& results)
{
    if (!results.valid) {
        return;
    }

    gpuTimingHistory_.shadowPass.push(historyValue(results.shadowPassMs));
    gpuTimingHistory_.mainPass.push(historyValue(results.mainPassMs));
    gpuTimingHistory_.bloom.push(historyValue(results.bloomMs));
    gpuTimingHistory_.composite.push(historyValue(results.compositeMs));
    gpuTimingHistory_.autoExposure.push(historyValue(results.autoExposureMs));
    gpuTimingHistory_.histogramExposure.push(historyValue(results.histogramExposureMs));
    gpuTimingHistory_.skybox.push(historyValue(results.skyboxMs));
    gpuTimingHistory_.renderObjects.push(historyValue(results.renderObjectsMs));
    gpuTimingHistory_.knownFrameTotal.push(knownGpuFrameTotalMs(results));
}

Renderer::CullingDebugSnapshot Renderer::cullingDebugSnapshot(uint32_t frameIndex)
{
    CullingDebugSnapshot snapshot{};
    snapshot.totalDrawItems = static_cast<uint32_t>(
        std::min<size_t>(cullingStats_.totalDrawItems, std::numeric_limits<uint32_t>::max()));
    if (cullingStats_.gpuCulling && frameIndex < frameGpuCullTotalDrawItems_.size()) {
        snapshot.totalDrawItems = frameGpuCullTotalDrawItems_[frameIndex];
    }

    snapshot.visibleDrawItems =
        static_cast<uint32_t>(std::min<size_t>(visibleDrawItems_.size(), snapshot.totalDrawItems));
    uint32_t gpuVisibleDrawItems = 0;
    if (readGpuVisibleCount(frameIndex, gpuVisibleDrawItems)) {
        snapshot.visibleDrawItems = std::min(gpuVisibleDrawItems, snapshot.totalDrawItems);
    }
    snapshot.culledDrawItems = snapshot.totalDrawItems > snapshot.visibleDrawItems
                                   ? snapshot.totalDrawItems - snapshot.visibleDrawItems
                                   : 0;

    snapshot.shadowDrawItems = static_cast<uint32_t>(
        std::min<size_t>(shadowCullingStats_.totalDrawItems, std::numeric_limits<uint32_t>::max()));
    if (shadowCullingStats_.gpuCulling && frameIndex < frameGpuShadowCullTotalDrawItems_.size()) {
        snapshot.shadowDrawItems = frameGpuShadowCullTotalDrawItems_[frameIndex];
    }

    snapshot.visibleShadowDrawItems = static_cast<uint32_t>(
        std::min<size_t>(shadowCullingStats_.visibleDrawItems, snapshot.shadowDrawItems));
    uint32_t gpuVisibleShadowDrawItems = 0;
    if (readGpuShadowVisibleCount(frameIndex, gpuVisibleShadowDrawItems)) {
        snapshot.visibleShadowDrawItems = std::min(gpuVisibleShadowDrawItems, snapshot.shadowDrawItems);
    }
    snapshot.culledShadowDrawItems = snapshot.shadowDrawItems > snapshot.visibleShadowDrawItems
                                         ? snapshot.shadowDrawItems - snapshot.visibleShadowDrawItems
                                         : 0;
    snapshot.shadowBatchCount = shadowCullingStats_.batchCount;
    snapshot.gpuCulling = isGpuCullingActive();
    snapshot.gpuShadowCulling = isGpuShadowCullingActive();
    return snapshot;
}

void Renderer::pushCullingHistorySample(uint32_t frameIndex)
{
    const CullingDebugSnapshot snapshot = cullingDebugSnapshot(frameIndex);
    visibleMainDrawItemsHistory_.push(static_cast<float>(snapshot.visibleDrawItems));
    culledMainDrawItemsHistory_.push(static_cast<float>(snapshot.culledDrawItems));
    visibleShadowDrawItemsHistory_.push(static_cast<float>(snapshot.visibleShadowDrawItems));
    culledShadowDrawItemsHistory_.push(static_cast<float>(snapshot.culledShadowDrawItems));
}

void Renderer::pushExposureHistorySample()
{
    exposureHistory_.push(currentToneMappingExposure());
    averageLuminanceHistory_.push(averageLuminance_);
    histogramClippedLuminanceHistory_.push(histogramClippedLuminance_);
}

void Renderer::loadRuntimeSettingsAtStartup()
{
    RuntimeSettings settings{};
    const RuntimeSettingsLoadResult result = loadRuntimeSettingsDetailed(runtimeSettingsPath_, settings);
    applyRuntimeSettings(settings, RuntimeSettingsApplyMode::Startup);

    lastRuntimeSettingsLoadStatus_ = result.message;
    runtimeSettingsWarning_ =
        result.status == RuntimeSettingsLoadStatus::Loaded ? std::string{} : result.message;
}

void Renderer::applyRuntimeSettings(const RuntimeSettings& settings, RuntimeSettingsApplyMode mode)
{
    toneMappingSettings_ = settings.toneMapping;
    bloomSettings_ = settings.bloom;
    debugUiSettings_ = settings.debugUi;

    csmSettings_.lambda = settings.csm.lambda;
    csmSettings_.shadowDistance = settings.csm.shadowDistance;
    csmSettings_.enableTexelSnapping = settings.csm.enableTexelSnapping;
    csmSettings_.enableCascadeDebugColors = settings.csm.enableCascadeDebugColors;

    if (mode == RuntimeSettingsApplyMode::Startup) {
        csmSettings_.cascadeCount = settings.csm.cascadeCount;
        useGpuCulling_ = settings.useGpuCulling;
        useGpuShadowCulling_ = settings.useGpuShadowCulling;
        useBindlessMaterialTextures_ = settings.enableBindlessMaterialTextures;
    } else {
        if (!settings.useGpuCulling || gpuCullingAvailable_) {
            useGpuCulling_ = settings.useGpuCulling;
        }
        if (!settings.useGpuShadowCulling || gpuShadowCullingAvailable_) {
            useGpuShadowCulling_ = settings.useGpuShadowCulling;
        }
    }

    clampRuntimeSettings();
    if (!toneMappingSettings_.enableAutoExposure ||
        exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
}

RuntimeSettings Renderer::captureRuntimeSettings() const
{
    RuntimeSettings settings{};
    settings.toneMapping = toneMappingSettings_;
    settings.bloom = bloomSettings_;
    settings.csm = csmSettings_;
    settings.debugUi = debugUiSettings_;
    settings.useGpuCulling = useGpuCulling_;
    settings.useGpuShadowCulling = useGpuShadowCulling_;
    settings.enableBindlessMaterialTextures = useBindlessMaterialTextures_;
    return settings;
}

void Renderer::saveRuntimeSettingsFromUi()
{
    const RuntimeSettingsSaveResult result = saveRuntimeSettingsDetailed(runtimeSettingsPath_, captureRuntimeSettings());
    lastRuntimeSettingsSaveStatus_ = result.message;
    if (result.saved) {
        runtimeSettingsWarning_.clear();
    } else {
        runtimeSettingsWarning_ = result.message;
    }
}

void Renderer::reloadRuntimeSettingsFromUi()
{
    RuntimeSettings settings{};
    const RuntimeSettingsLoadResult result = loadRuntimeSettingsDetailed(runtimeSettingsPath_, settings);
    applyRuntimeSettings(settings, RuntimeSettingsApplyMode::Runtime);

    lastRuntimeSettingsLoadStatus_ =
        result.message + " Runtime-safe values applied; startup-applied values require restart.";
    runtimeSettingsWarning_ =
        result.status == RuntimeSettingsLoadStatus::Loaded ? std::string{} : result.message;
}

void Renderer::resetRuntimeSettingsToDefaults()
{
    applyRuntimeSettings(RuntimeSettings{}, RuntimeSettingsApplyMode::Runtime);
    lastRuntimeSettingsLoadStatus_ = "Defaults restored in memory. Save Settings will write them to disk.";
    runtimeSettingsWarning_.clear();
}

void Renderer::buildDebugUi()
{
    if (!imguiLayer_.initialized()) {
        return;
    }

    ImGui::Begin("VulkanEngine Debug");

    drawRuntimeSettingsDebugUi();

    if (ImGui::CollapsingHeader("Debug Views", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show Render Graph panel", &debugUiSettings_.showRenderGraphPanel);
        ImGui::Checkbox("Show GPU Timing graphs", &debugUiSettings_.showGpuTimingGraphs);
        ImGui::Checkbox("Show Culling stats", &debugUiSettings_.showCullingStats);
        ImGui::Checkbox("Show Exposure graphs", &debugUiSettings_.showExposureGraphs);
    }

    if (ImGui::CollapsingHeader("Tone Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* toneMappers[] = {"Reinhard", "ACES"};
        ImGui::Combo("Tone mapper", &toneMappingSettings_.operatorType, toneMappers, IM_ARRAYSIZE(toneMappers));
        ImGui::DragFloat("Manual exposure", &toneMappingSettings_.manualExposure, 0.01f, 0.0f, 64.0f, "%.3f");

        const char* exposureModes[] = {"Manual", "Log-average", "Histogram"};
        int exposureMode =
            toneMappingSettings_.enableAutoExposure ? toneMappingSettings_.exposureMode : 0;
        exposureMode = static_cast<int>(exposureModeValue(exposureMode));
        if (ImGui::Combo("Exposure mode", &exposureMode, exposureModes, IM_ARRAYSIZE(exposureModes))) {
            toneMappingSettings_.exposureMode = exposureMode;
            toneMappingSettings_.enableAutoExposure = exposureMode != 0;
        }

        ImGui::DragFloat("Target luminance", &toneMappingSettings_.targetLuminance, 0.001f, 0.001f, 8.0f, "%.3f");
        ImGui::DragFloat("Min exposure", &toneMappingSettings_.minExposure, 0.01f, 0.0f, 64.0f, "%.3f");
        ImGui::DragFloat("Max exposure", &toneMappingSettings_.maxExposure, 0.01f, 0.0f, 64.0f, "%.3f");
        ImGui::DragFloat("Adaptation rate", &toneMappingSettings_.adaptationRate, 0.01f, 0.0f, 16.0f, "%.3f");
        ImGui::SliderFloat("Histogram low percentile", &toneMappingSettings_.lowPercentile, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Histogram high percentile", &toneMappingSettings_.highPercentile, 0.0f, 1.0f, "%.3f");
    }

    if (ImGui::CollapsingHeader("Bloom", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enabled", &bloomSettings_.enabled);
        ImGui::DragFloat("Threshold", &bloomSettings_.threshold, 0.01f, 0.0f, 32.0f, "%.3f");
        ImGui::DragFloat("Intensity", &bloomSettings_.intensity, 0.01f, 0.0f, 8.0f, "%.3f");
    }

    if (ImGui::CollapsingHeader("CSM", ImGuiTreeNodeFlags_DefaultOpen)) {
        int cascadeCount = static_cast<int>(activeCascadeCount());
        ImGui::BeginDisabled();
        ImGui::SliderInt("Cascade count (startup)", &cascadeCount, 1, static_cast<int>(kMaxShadowCascades));
        ImGui::EndDisabled();
        ImGui::SliderFloat("Lambda", &csmSettings_.lambda, 0.0f, 1.0f, "%.3f");
        ImGui::DragFloat("Shadow distance", &csmSettings_.shadowDistance, 0.1f, 1.0f, csmSettings_.farPlane, "%.2f");
        ImGui::Checkbox("Texel snapping enabled", &csmSettings_.enableTexelSnapping);
        ImGui::Checkbox("Cascade debug colors enabled", &csmSettings_.enableCascadeDebugColors);
    }

    if (ImGui::CollapsingHeader("GPU Culling", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!gpuCullingAvailable_) {
            ImGui::BeginDisabled();
        }
        ImGui::Checkbox("Main GPU culling enabled", &useGpuCulling_);
        if (!gpuCullingAvailable_) {
            ImGui::EndDisabled();
        }

        if (!gpuShadowCullingAvailable_) {
            ImGui::BeginDisabled();
        }
        ImGui::Checkbox("Shadow GPU culling enabled", &useGpuShadowCulling_);
        if (!gpuShadowCullingAvailable_) {
            ImGui::EndDisabled();
        }

        ImGui::Text("Bindless material textures: %s", isBindlessMaterialTextureActive() ? "active" : "fallback");
        bool bindlessEnabled = useBindlessMaterialTextures_;
        ImGui::BeginDisabled();
        ImGui::Checkbox("Bindless material textures enabled (startup)", &bindlessEnabled);
        ImGui::EndDisabled();
        ImGui::Text("Main indirect count path: %s",
                    isFrameIndirectCountPathActive(currentFrame_) ? "active" : "fallback");
        ImGui::Text("Shadow indirect count path: %s",
                    isShadowIndirectCountPathActive(currentFrame_) ? "active" : "fallback");
    }

    if (ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Environment source: %s", hdrEnvironmentLoaded_ ? "HDR environment loaded" : "procedural fallback");
        ImGui::Text("Tone mapping exposure: %.4f", currentToneMappingExposure());
    }

    if (debugUiSettings_.showRenderGraphPanel &&
        ImGui::CollapsingHeader("Render Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawRenderGraphDebugUi();
    }

    if (debugUiSettings_.showGpuTimingGraphs &&
        ImGui::CollapsingHeader("GPU Timings", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawGpuTimingDebugUi();
    }

    if (debugUiSettings_.showCullingStats &&
        ImGui::CollapsingHeader("Culling", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawCullingDebugUi();
    }

    if (debugUiSettings_.showExposureGraphs &&
        ImGui::CollapsingHeader("Exposure", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawExposureDebugUi();
    }

    ImGui::End();
    clampRuntimeSettings();
}

void Renderer::drawRuntimeSettingsDebugUi()
{
    if (!ImGui::CollapsingHeader("Runtime Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    const std::string settingsPath = runtimeSettingsPath_.string();
    ImGui::TextWrapped("File: %s", settingsPath.c_str());

    if (ImGui::Button("Save Settings")) {
        saveRuntimeSettingsFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reload Settings")) {
        reloadRuntimeSettingsFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset to Defaults")) {
        resetRuntimeSettingsToDefaults();
    }

    ImGui::TextWrapped("Last load: %s", lastRuntimeSettingsLoadStatus_.c_str());
    ImGui::TextWrapped("Last save: %s", lastRuntimeSettingsSaveStatus_.c_str());
    if (!runtimeSettingsWarning_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f), "Warning: %s", runtimeSettingsWarning_.c_str());
    }
    ImGui::TextDisabled("Runtime-safe: tone mapping, exposure, bloom, CSM lambda/distance/stability/debug, "
                        "available GPU culling toggles, and panel visibility.");
    ImGui::TextDisabled("Startup-applied: CSM cascade count, bindless material texture heap, and culling resources "
                        "that were disabled before initialization.");
}

void Renderer::drawRenderGraphDebugUi()
{
    const auto& passes = renderGraph_.debugPasses();
    ImGui::Text("Manual pass order: %zu passes", passes.size());

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("RenderGraphPassTable", 6, flags)) {
        return;
    }

    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Pass");
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupColumn("Reads");
    ImGui::TableSetupColumn("Writes");
    ImGui::TableSetupColumn("Notes");
    ImGui::TableHeadersRow();

    for (size_t index = 0; index < passes.size(); ++index) {
        const renderer::RenderPassNode& pass = passes[index];
        const std::string reads = resourceUsageList(pass, renderer::RenderResourceAccess::Read);
        const std::string writes = resourceUsageList(pass, renderer::RenderResourceAccess::Write);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%zu", index);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(pass.name);
        ImGui::TableNextColumn();
        ImGui::Text("%s / %s",
                    renderer::renderPassExecutionTypeName(pass.executionType),
                    renderer::renderPassTypeName(pass.type));
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", reads.c_str());
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", writes.c_str());
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", pass.transitionSummary);
    }

    ImGui::EndTable();
}

void Renderer::drawGpuTimingDebugUi()
{
    if (!latestGpuTimings_.valid) {
        ImGui::TextDisabled("GPU timings are waiting for the first completed frame.");
        ImGui::Text("CPU frame delta: %.3f ms (%.1f FPS)", cpuFrameDeltaMs_, cpuFps_);
        return;
    }

    ImGui::Text("Approx GPU total from known passes: %.3f ms", gpuTimingHistory_.knownFrameTotal.latest());
    ImGui::Text("CPU frame delta: %.3f ms (%.1f FPS)", cpuFrameDeltaMs_, cpuFps_);
    ImGui::TextDisabled("Skybox and RenderObjects are nested ranges inside Main; ImGui is not timestamped.");

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("GpuTimingHistoryTable", 5, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Range");
    ImGui::TableSetupColumn("Current ms", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Avg ms", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Max ms", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("History");
    ImGui::TableHeadersRow();

    drawTimingHistoryRow("Known GPU total", gpuTimingHistory_.knownFrameTotal);
    drawTimingHistoryRow("Shadow / CSM", gpuTimingHistory_.shadowPass);
    drawTimingHistoryRow("Main", gpuTimingHistory_.mainPass);
    drawTimingHistoryRow("Bloom", gpuTimingHistory_.bloom);
    drawTimingHistoryRow("Composite", gpuTimingHistory_.composite);
    drawTimingHistoryRow("AutoExposure / Luminance", gpuTimingHistory_.autoExposure);
    drawTimingHistoryRow("HistogramExposure", gpuTimingHistory_.histogramExposure);
    drawTimingHistoryRow("Skybox", gpuTimingHistory_.skybox);
    drawTimingHistoryRow("RenderObjects", gpuTimingHistory_.renderObjects);

    ImGui::EndTable();
}

void Renderer::drawCullingDebugUi()
{
    const CullingDebugSnapshot snapshot = cullingDebugSnapshot(currentFrame_);
    ImGui::Text("Total draw items: %u", snapshot.totalDrawItems);
    ImGui::Text("Visible draw items: %u", snapshot.visibleDrawItems);
    ImGui::Text("Culled draw items: %u", snapshot.culledDrawItems);
    ImGui::Text("Shadow draw items: %u", snapshot.shadowDrawItems);
    ImGui::Text("Visible shadow draw items: %u", snapshot.visibleShadowDrawItems);
    ImGui::Text("Culled shadow draw items: %u", snapshot.culledShadowDrawItems);
    ImGui::Text("Shadow batches: %zu", snapshot.shadowBatchCount);
    ImGui::Text("GPU culling: %s", snapshot.gpuCulling ? "enabled" : "disabled");
    ImGui::Text("GPU shadow culling: %s", snapshot.gpuShadowCulling ? "enabled" : "disabled");

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("CullingHistoryTable", 5, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Current", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("History");
    ImGui::TableHeadersRow();

    drawScalarHistoryRow("Visible main draw items", visibleMainDrawItemsHistory_, "%.0f");
    drawScalarHistoryRow("Culled main draw items", culledMainDrawItemsHistory_, "%.0f");
    drawScalarHistoryRow("Visible shadow draw items", visibleShadowDrawItemsHistory_, "%.0f");
    drawScalarHistoryRow("Culled shadow draw items", culledShadowDrawItemsHistory_, "%.0f");

    ImGui::EndTable();
}

void Renderer::drawExposureDebugUi()
{
    const ExposureMode mode = exposureModeValue(toneMappingSettings_.enableAutoExposure ? toneMappingSettings_.exposureMode
                                                                                        : 0);
    ImGui::Text("Current exposure: %.4f", currentToneMappingExposure());
    ImGui::Text("Log-average luminance: %.4f", averageLuminance_);
    ImGui::Text("Histogram clipped luminance: %.4f", histogramClippedLuminance_);
    ImGui::Text("Exposure mode: %s", exposureModeName(mode).data());

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("ExposureHistoryTable", 5, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Current", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("History");
    ImGui::TableHeadersRow();

    drawScalarHistoryRow("Exposure", exposureHistory_, "%.4f");
    drawScalarHistoryRow("Log-average luminance", averageLuminanceHistory_, "%.4f");
    drawScalarHistoryRow("Histogram clipped luminance", histogramClippedLuminanceHistory_, "%.4f");

    ImGui::EndTable();
}

void Renderer::drawTimingHistoryRow(const char* label, const DebugHistory& history) const
{
    ImGui::PushID(label);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", history.latest());
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", history.average());
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", history.max());
    ImGui::TableNextColumn();
    drawHistoryPlot(history, 42.0f);
    ImGui::PopID();
}

void Renderer::drawScalarHistoryRow(const char* label, const DebugHistory& history, const char* valueFormat) const
{
    ImGui::PushID(label);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text(valueFormat, history.latest());
    ImGui::TableNextColumn();
    ImGui::Text(valueFormat, history.average());
    ImGui::TableNextColumn();
    ImGui::Text(valueFormat, history.max());
    ImGui::TableNextColumn();
    drawHistoryPlot(history, 42.0f);
    ImGui::PopID();
}

void Renderer::drawHistoryPlot(const DebugHistory& history, float height) const
{
    if (history.empty()) {
        ImGui::TextDisabled("waiting");
        return;
    }

    std::array<float, kDebugHistoryCapacity> values{};
    const size_t sampleCount = history.copyChronological(values);
    const float scaleMax = std::max(history.max(), 0.001f);
    ImGui::PlotLines("##history",
                     values.data(),
                     static_cast<int>(sampleCount),
                     0,
                     nullptr,
                     0.0f,
                     scaleMax,
                     ImVec2(180.0f, height));
}

void Renderer::clampRuntimeSettings()
{
    toneMappingSettings_.operatorType = std::clamp(toneMappingSettings_.operatorType, 0, 1);
    if (!toneMappingSettings_.enableAutoExposure) {
        toneMappingSettings_.exposureMode = static_cast<int>(ExposureMode::Manual);
    } else {
        toneMappingSettings_.exposureMode = static_cast<int>(exposureModeValue(toneMappingSettings_.exposureMode));
    }
    toneMappingSettings_.manualExposure = std::max(toneMappingSettings_.manualExposure, 0.0f);
    toneMappingSettings_.targetLuminance = std::max(toneMappingSettings_.targetLuminance, kMinAverageLuminance);
    toneMappingSettings_.minExposure = std::max(toneMappingSettings_.minExposure, 0.0f);
    toneMappingSettings_.maxExposure = std::max(toneMappingSettings_.maxExposure, toneMappingSettings_.minExposure);
    toneMappingSettings_.adaptationRate = std::max(toneMappingSettings_.adaptationRate, 0.0f);

    toneMappingSettings_.lowPercentile = std::clamp(toneMappingSettings_.lowPercentile, 0.0f, 1.0f);
    toneMappingSettings_.highPercentile = std::clamp(toneMappingSettings_.highPercentile, 0.0f, 1.0f);
    if (toneMappingSettings_.highPercentile <= toneMappingSettings_.lowPercentile) {
        toneMappingSettings_.highPercentile = std::min(1.0f, toneMappingSettings_.lowPercentile + 0.01f);
        toneMappingSettings_.lowPercentile = std::min(toneMappingSettings_.lowPercentile,
                                                      toneMappingSettings_.highPercentile - 0.01f);
    }

    bloomSettings_.threshold = std::max(bloomSettings_.threshold, 0.0f);
    bloomSettings_.intensity = std::max(bloomSettings_.intensity, 0.0f);

    csmSettings_.cascadeCount = std::clamp(csmSettings_.cascadeCount, 1U, kMaxShadowCascades);
    csmSettings_.lambda = std::clamp(csmSettings_.lambda, 0.0f, 1.0f);
    csmSettings_.shadowDistance = std::clamp(csmSettings_.shadowDistance,
                                             csmSettings_.nearPlane + 0.001f,
                                             csmSettings_.farPlane);
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
        frameGpuCullTotalDrawItems_[frameIndex] =
            static_cast<uint32_t>(std::min(allDrawItems_.size(), static_cast<size_t>(kMaxDrawItems)));
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
            frameGpuShadowCullTotalDrawItems_[frameIndex] = static_cast<uint32_t>(
                std::min(allDrawItems_.size() * cascadeCount, static_cast<size_t>(kMaxDrawItems)));
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
            frameData.materialParams = {material->metallic, material->roughness, material->multiScatterStrength, 0.0f};
            frameData.textureIndices = {material->baseColorTextureIndex,
                                        material->normalTextureIndex,
                                        material->metallicRoughnessTextureIndex,
                                        0};
        }
        frameData.cameraPosition = glm::vec4(camera_.position, csmSettings_.enableCascadeDebugColors ? 1.0f : 0.0f);
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
    imguiLayer_.onSwapchainRecreated(swapchain_.colorFormat(), swapchain_.imageCount());
    createPostProcessResources();

    const bool exposurePipelineMissing =
        toneMappingSettings_.enableAutoExposure && postProcessLuminanceDescriptorSetLayout_.handle() != VK_NULL_HANDLE &&
        ((autoExposureAvailable_ && luminancePipeline_.pipeline() == VK_NULL_HANDLE) ||
         (histogramExposureAvailable_ && histogramPipeline_.pipeline() == VK_NULL_HANDLE));
    const bool pipelineNeedsRecreate =
        pipeline_.pipeline() == VK_NULL_HANDLE || pipelineColorFormat_ != kSceneColorFormat ||
        pipelineDepthFormat_ != swapchain_.depthFormat() || skyboxPipeline_.pipeline() == VK_NULL_HANDLE ||
        skyboxPipelineColorFormat_ != kSceneColorFormat || skyboxPipelineDepthFormat_ != swapchain_.depthFormat() ||
        shadowPipelineDepthFormat_ != shadowMap_.format() || bloomExtractPipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomExtractPipelineColorFormat_ != kBloomColorFormat || bloomBlurPipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomBlurPipelineColorFormat_ != kBloomColorFormat || compositePipeline_.pipeline() == VK_NULL_HANDLE ||
        compositePipelineColorFormat_ != swapchain_.colorFormat() || exposurePipelineMissing;
    if (pipelineNeedsRecreate) {
        createPipeline();
    }

    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
}

renderer::RenderGraphFrameResources Renderer::renderGraphFrameResources()
{
    const VkExtent3D sceneExtent = sceneColor_.extent();
    return renderer::RenderGraphFrameResources{
        renderer::RenderGraphImageResource{
            "SceneColor",
            sceneColor_.image(),
            sceneColor_.imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &sceneColorLayout_,
        },
        renderer::RenderGraphImageResource{
            "BloomExtract",
            bloomExtract_.image(),
            bloomExtract_.imageView(),
            bloomExtent_,
            &bloomExtractLayout_,
        },
        renderer::RenderGraphImageResource{
            "BloomPing",
            bloomPing_.image(),
            bloomPing_.imageView(),
            bloomExtent_,
            &bloomPingLayout_,
        },
        renderer::RenderGraphImageResource{
            "BloomPong",
            bloomPong_.image(),
            bloomPong_.imageView(),
            bloomExtent_,
            &bloomPongLayout_,
        },
    };
}

void Renderer::updateAutoExposureFromReadback(uint32_t frameIndex)
{
    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (!toneMappingSettings_.enableAutoExposure || mode == ExposureMode::Manual) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        return;
    }

    const auto readLogAverageLuminance = [this, frameIndex](float& luminance) {
        if (!isLogAverageExposureActive() || frameIndex >= frameLuminanceReadbackReady_.size() ||
            frameLuminanceReadbackReady_[frameIndex] == 0 || frameIndex >= frameLuminanceReadbackBuffers_.size() ||
            luminancePartialCount_ == 0) {
            return false;
        }

        rhi::VulkanBuffer& readbackBuffer = frameLuminanceReadbackBuffers_[frameIndex];
        if (!readbackBuffer.valid()) {
            return false;
        }

        std::vector<LuminancePartial> partials(luminancePartialCount_);
        readbackBuffer.download(
            std::as_writable_bytes(std::span<LuminancePartial>(partials.data(), partials.size())));

        double sumLogLuminance = 0.0;
        double sampleCount = 0.0;
        for (const LuminancePartial& partial : partials) {
            if (partial.sampleCount <= 0.0f || !std::isfinite(partial.sumLogLuminance) ||
                !std::isfinite(partial.sampleCount)) {
                continue;
            }
            sumLogLuminance += static_cast<double>(partial.sumLogLuminance);
            sampleCount += static_cast<double>(partial.sampleCount);
        }

        if (sampleCount <= 0.0) {
            return false;
        }

        const double averageLogLuminance = sumLogLuminance / sampleCount;
        const double averageLuminance = std::exp(averageLogLuminance);
        if (!std::isfinite(averageLuminance) || averageLuminance <= 0.0) {
            return false;
        }

        luminance = static_cast<float>(std::max(averageLuminance, static_cast<double>(kMinAverageLuminance)));
        return true;
    };

    const auto readHistogramLuminance = [this, frameIndex](float& luminance) {
        if (!isHistogramExposureActive() || frameIndex >= frameHistogramReadbackReady_.size() ||
            frameHistogramReadbackReady_[frameIndex] == 0 || frameIndex >= frameHistogramReadbackBuffers_.size()) {
            return false;
        }

        rhi::VulkanBuffer& readbackBuffer = frameHistogramReadbackBuffers_[frameIndex];
        if (!readbackBuffer.valid()) {
            return false;
        }

        std::array<uint32_t, kHistogramBinCount> bins{};
        readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(bins.data(), bins.size())));

        uint64_t totalSamples = 0;
        for (uint32_t count : bins) {
            totalSamples += count;
        }
        if (totalSamples == 0) {
            return false;
        }

        const auto [lowPercentile, highPercentile] =
            sanitizedPercentileRange(toneMappingSettings_.lowPercentile, toneMappingSettings_.highPercentile);
        const uint64_t lowCut =
            std::min<uint64_t>(totalSamples,
                               static_cast<uint64_t>(std::floor(static_cast<double>(totalSamples) *
                                                                static_cast<double>(lowPercentile))));
        uint64_t highCut =
            std::min<uint64_t>(totalSamples,
                               static_cast<uint64_t>(std::ceil(static_cast<double>(totalSamples) *
                                                               static_cast<double>(highPercentile))));
        if (highCut <= lowCut) {
            highCut = std::min<uint64_t>(totalSamples, lowCut + 1);
        }
        if (highCut <= lowCut) {
            return false;
        }

        const auto [minLogLuminance, maxLogLuminance] = sanitizedHistogramLogRange(
            toneMappingSettings_.histogramMinLogLuminance, toneMappingSettings_.histogramMaxLogLuminance);
        const double binWidth =
            static_cast<double>(maxLogLuminance - minLogLuminance) / static_cast<double>(kHistogramBinCount);

        uint64_t cumulative = 0;
        uint64_t weightedSampleCount = 0;
        double weightedLuminanceSum = 0.0;
        for (size_t binIndex = 0; binIndex < bins.size(); ++binIndex) {
            const uint64_t count = bins[binIndex];
            const uint64_t binStart = cumulative;
            const uint64_t binEnd = cumulative + count;
            cumulative = binEnd;

            const uint64_t clippedStart = std::max(binStart, lowCut);
            const uint64_t clippedEnd = std::min(binEnd, highCut);
            if (clippedEnd <= clippedStart) {
                continue;
            }

            const uint64_t includedCount = clippedEnd - clippedStart;
            const double logLuminance =
                static_cast<double>(minLogLuminance) + (static_cast<double>(binIndex) + 0.5) * binWidth;
            const double binLuminance = std::exp2(logLuminance);
            weightedLuminanceSum += binLuminance * static_cast<double>(includedCount);
            weightedSampleCount += includedCount;
        }

        if (weightedSampleCount == 0) {
            return false;
        }

        const double averageClippedLuminance =
            weightedLuminanceSum / static_cast<double>(weightedSampleCount);
        if (!std::isfinite(averageClippedLuminance) || averageClippedLuminance <= 0.0) {
            return false;
        }

        luminance =
            static_cast<float>(std::max(averageClippedLuminance, static_cast<double>(kMinAverageLuminance)));
        return true;
    };

    float logAverageLuminance = averageLuminance_;
    const bool logAverageRead = readLogAverageLuminance(logAverageLuminance);
    if (logAverageRead) {
        averageLuminance_ = logAverageLuminance;
    }

    float histogramLuminance = histogramClippedLuminance_;
    const bool histogramRead = readHistogramLuminance(histogramLuminance);
    if (histogramRead) {
        histogramClippedLuminance_ = histogramLuminance;
    }

    float exposureLuminance = 0.0f;
    bool hasExposureLuminance = false;
    if (mode == ExposureMode::LogAverage) {
        exposureLuminance = logAverageLuminance;
        hasExposureLuminance = logAverageRead;
    } else if (mode == ExposureMode::Histogram) {
        if (histogramRead) {
            exposureLuminance = histogramLuminance;
            hasExposureLuminance = true;
        } else if (logAverageRead) {
            exposureLuminance = logAverageLuminance;
            hasExposureLuminance = true;
        }
    }

    if (!hasExposureLuminance) {
        return;
    }

    const float minExposure = std::max(toneMappingSettings_.minExposure, 0.0f);
    const float maxExposure = std::max(toneMappingSettings_.maxExposure, minExposure);
    const float unclampedTarget =
        toneMappingSettings_.targetLuminance / std::max(exposureLuminance, kMinAverageLuminance);
    const float targetExposure = std::clamp(unclampedTarget, minExposure, maxExposure);

    const auto now = std::chrono::steady_clock::now();
    const float deltaTime = std::max(0.0f, std::chrono::duration<float>(now - lastAutoExposureUpdate_).count());
    lastAutoExposureUpdate_ = now;

    if (!std::isfinite(currentExposure_) || currentExposure_ <= 0.0f) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }

    const float adaptationRate = std::max(toneMappingSettings_.adaptationRate, 0.0f);
    const float adaptationAlpha = 1.0f - std::exp(-adaptationRate * deltaTime);
    currentExposure_ = std::clamp(currentExposure_ + (targetExposure - currentExposure_) * adaptationAlpha,
                                  minExposure,
                                  maxExposure);
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
        frameIndex >= frameGpuShadowCullReadbackReady_.size() || frameGpuShadowCullReadbackReady_[frameIndex] == 0) {
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
           frameCullInputBuffers_.size() == frames_.size() && frameBatchVisibleCountBuffers_.size() == frames_.size() &&
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

bool Renderer::isAutoExposureActive() const
{
    if (!toneMappingSettings_.enableAutoExposure) {
        return false;
    }

    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (mode == ExposureMode::Manual) {
        return false;
    }
    if (mode == ExposureMode::LogAverage) {
        return isLogAverageExposureActive();
    }

    return isHistogramExposureActive() || isLogAverageExposureActive();
}

bool Renderer::isLogAverageExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && autoExposureAvailable_ &&
           luminancePipeline_.pipeline() != VK_NULL_HANDLE && luminancePipeline_.layout() != VK_NULL_HANDLE &&
           luminanceDescriptorSets_.size() == frames_.size() && frameLuminanceBuffers_.size() == frames_.size() &&
           frameLuminanceReadbackBuffers_.size() == frames_.size() &&
           frameLuminanceReadbackReady_.size() == frames_.size() && luminancePartialCount_ > 0 &&
           luminanceGroupCountX_ > 0 && luminanceGroupCountY_ > 0;
}

bool Renderer::isHistogramExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && histogramExposureAvailable_ &&
           histogramPipeline_.pipeline() != VK_NULL_HANDLE && histogramPipeline_.layout() != VK_NULL_HANDLE &&
           histogramDescriptorSets_.size() == frames_.size() && frameHistogramBuffers_.size() == frames_.size() &&
           frameHistogramReadbackBuffers_.size() == frames_.size() &&
           frameHistogramReadbackReady_.size() == frames_.size();
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

float Renderer::currentToneMappingExposure() const
{
    if (!toneMappingSettings_.enableAutoExposure) {
        return toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }

    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (mode == ExposureMode::LogAverage && isLogAverageExposureActive()) {
        return toneMappingExposureValue(currentExposure_);
    }
    if (mode == ExposureMode::Histogram && (isHistogramExposureActive() || isLogAverageExposureActive())) {
        return toneMappingExposureValue(currentExposure_);
    }

    return toneMappingExposureValue(toneMappingSettings_.manualExposure);
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
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

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
    computeBarriers[0].size = static_cast<VkDeviceSize>(allDrawItems_.size() * sizeof(VkDrawIndexedIndirectCommand));

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

    const VkDeviceSize shadowIndirectBufferSize =
        std::min<VkDeviceSize>(frameShadowIndirectDrawBuffers_.at(currentFrame_).size(),
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
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuCullPipeline_.layout(), 0, 1, &descriptorSet, 0, nullptr);

    GpuCullPushConstants pushConstants{};
    pushConstants.frustumPlanes = frameShadowCascadeFrustumPlanes_[cascadeIndex];
    pushConstants.params = glm::uvec4(static_cast<uint32_t>(allDrawItems_.size()), 1U, 1U, 0U);
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

void Renderer::recordLuminanceCommands(VkCommandBuffer commandBuffer)
{
    if (!isLogAverageExposureActive() || exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual ||
        currentFrame_ >= luminanceDescriptorSets_.size() ||
        currentFrame_ >= frameLuminanceBuffers_.size() || currentFrame_ >= frameLuminanceReadbackBuffers_.size() ||
        currentFrame_ >= frameLuminanceReadbackReady_.size()) {
        return;
    }

    VkBuffer luminanceBuffer = frameLuminanceBuffers_[currentFrame_].buffer();
    VkBuffer readbackBuffer = frameLuminanceReadbackBuffers_[currentFrame_].buffer();
    if (luminanceBuffer == VK_NULL_HANDLE || readbackBuffer == VK_NULL_HANDLE) {
        return;
    }

    renderGraph_.beginLuminancePass();

    rhi::debug::beginLabel(commandBuffer, "LuminancePass");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminancePipeline_.pipeline());

    const VkDescriptorSet descriptorSet = luminanceDescriptorSets_[currentFrame_];
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            luminancePipeline_.layout(),
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);

    const VkExtent3D sceneExtent = sceneColor_.extent();
    const LuminancePushConstants pushConstants{
        glm::uvec4(sceneExtent.width, sceneExtent.height, luminanceGroupCountX_, 0)};
    vkCmdPushConstants(commandBuffer,
                       luminancePipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(LuminancePushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "AutoExposureCompute");
    vkCmdDispatch(commandBuffer, luminanceGroupCountX_, luminanceGroupCountY_, 1);
    rhi::debug::endLabel(commandBuffer);

    VkBufferMemoryBarrier2 computeBarrier{};
    computeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    computeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarrier.buffer = luminanceBuffer;
    computeBarrier.offset = 0;
    computeBarrier.size = frameLuminanceBuffers_[currentFrame_].size();

    VkDependencyInfo computeDependencyInfo{};
    computeDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    computeDependencyInfo.bufferMemoryBarrierCount = 1;
    computeDependencyInfo.pBufferMemoryBarriers = &computeBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &computeDependencyInfo);

    VkBufferCopy luminanceCopy{};
    luminanceCopy.size = frameLuminanceBuffers_[currentFrame_].size();
    vkCmdCopyBuffer(commandBuffer, luminanceBuffer, readbackBuffer, 1, &luminanceCopy);

    VkBufferMemoryBarrier2 readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    readbackBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    readbackBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    readbackBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.buffer = readbackBuffer;
    readbackBarrier.offset = 0;
    readbackBarrier.size = frameLuminanceReadbackBuffers_[currentFrame_].size();

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameLuminanceReadbackReady_[currentFrame_] = 1;
    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endLuminancePass();
}

void Renderer::recordHistogramCommands(VkCommandBuffer commandBuffer)
{
    if (!isHistogramExposureActive() || exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual ||
        currentFrame_ >= histogramDescriptorSets_.size() || currentFrame_ >= frameHistogramBuffers_.size() ||
        currentFrame_ >= frameHistogramReadbackBuffers_.size() || currentFrame_ >= frameHistogramReadbackReady_.size()) {
        return;
    }

    VkBuffer histogramBuffer = frameHistogramBuffers_[currentFrame_].buffer();
    VkBuffer readbackBuffer = frameHistogramReadbackBuffers_[currentFrame_].buffer();
    if (histogramBuffer == VK_NULL_HANDLE || readbackBuffer == VK_NULL_HANDLE) {
        return;
    }

    const VkExtent3D sceneExtent = sceneColor_.extent();
    const uint32_t groupCountX = (sceneExtent.width + kHistogramLocalSizeX - 1) / kHistogramLocalSizeX;
    const uint32_t groupCountY = (sceneExtent.height + kHistogramLocalSizeY - 1) / kHistogramLocalSizeY;
    if (groupCountX == 0 || groupCountY == 0) {
        return;
    }

    renderGraph_.beginHistogramExposurePass();

    rhi::debug::beginLabel(commandBuffer, "HistogramExposurePass");
    const VkDeviceSize histogramBufferSize = frameHistogramBuffers_[currentFrame_].size();
    vkCmdFillBuffer(commandBuffer, histogramBuffer, 0, histogramBufferSize, 0);

    VkBufferMemoryBarrier2 resetBarrier{};
    resetBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    resetBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    resetBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    resetBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resetBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    resetBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resetBarrier.buffer = histogramBuffer;
    resetBarrier.offset = 0;
    resetBarrier.size = histogramBufferSize;

    VkDependencyInfo resetDependencyInfo{};
    resetDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    resetDependencyInfo.bufferMemoryBarrierCount = 1;
    resetDependencyInfo.pBufferMemoryBarriers = &resetBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &resetDependencyInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline_.pipeline());

    const VkDescriptorSet descriptorSet = histogramDescriptorSets_[currentFrame_];
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            histogramPipeline_.layout(),
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);

    const auto [minLogLuminance, maxLogLuminance] = sanitizedHistogramLogRange(
        toneMappingSettings_.histogramMinLogLuminance, toneMappingSettings_.histogramMaxLogLuminance);
    const HistogramPushConstants pushConstants{
        glm::uvec4(sceneExtent.width, sceneExtent.height, kHistogramBinCount, 0),
        glm::vec4(minLogLuminance, maxLogLuminance, kMinAverageLuminance, 0.0f)};
    vkCmdPushConstants(commandBuffer,
                       histogramPipeline_.layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(HistogramPushConstants)),
                       &pushConstants);

    rhi::debug::beginLabel(commandBuffer, "HistogramCompute");
    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
    rhi::debug::endLabel(commandBuffer);

    VkBufferMemoryBarrier2 computeBarrier{};
    computeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    computeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    computeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    computeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeBarrier.buffer = histogramBuffer;
    computeBarrier.offset = 0;
    computeBarrier.size = histogramBufferSize;

    VkDependencyInfo computeDependencyInfo{};
    computeDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    computeDependencyInfo.bufferMemoryBarrierCount = 1;
    computeDependencyInfo.pBufferMemoryBarriers = &computeBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &computeDependencyInfo);

    VkBufferCopy histogramCopy{};
    histogramCopy.size = histogramBufferSize;
    vkCmdCopyBuffer(commandBuffer, histogramBuffer, readbackBuffer, 1, &histogramCopy);

    VkBufferMemoryBarrier2 readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    readbackBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    readbackBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    readbackBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.buffer = readbackBuffer;
    readbackBarrier.offset = 0;
    readbackBarrier.size = frameHistogramReadbackBuffers_[currentFrame_].size();

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameHistogramReadbackReady_[currentFrame_] = 1;
    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endHistogramExposurePass();
}

void Renderer::recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t mainDrawItemCount = visibleDrawItems_.size();

    renderGraph_.beginFrame(commandBuffer, swapchain_, shadowMap_, imageIndex, renderGraphFrameResources());
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

        const std::string shadowCullingLabel = gpuShadowCullingActive
                                                   ? "ShadowCasterCulling GPU cascade " + std::to_string(cascadeIndex) +
                                                         " max " + std::to_string(shadowDrawItemCount)
                                                   : "ShadowCasterCulling cascade " + std::to_string(cascadeIndex) +
                                                         " visible " + std::to_string(shadowDrawItemCount) + "/" +
                                                         std::to_string(allDrawItems_.size());
        rhi::debug::beginLabel(commandBuffer, shadowCullingLabel);
        rhi::debug::endLabel(commandBuffer);

        const std::string shadowDrawLabel =
            "ShadowCascade" + std::to_string(cascadeIndex) + " DrawItems " + std::to_string(shadowDrawItemCount) +
            (gpuShadowCullingActive
                 ? (shadowIndirectCountPathActive ? " GPU culling indirect-count" : " GPU culling indirect fallback")
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
            const std::string shadowBatchesLabel = "ShadowIndirectDrawBatches " +
                                                   std::to_string(activeShadowMeshDrawBatches.size()) +
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

    rhi::debug::beginLabel(commandBuffer, "MainHDRPass");
    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::MainPass);
    renderGraph_.beginMainHdrPass();

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
        const SkyboxPushConstants skyboxPushConstants{glm::inverse(projection * skyboxView),
                                                      currentToneMappingExposure(),
                                                      toneMappingOperatorValue(toneMappingSettings_.operatorType)};

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
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(SkyboxPushConstants)),
                           &skyboxPushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Skybox);
    rhi::debug::endLabel(commandBuffer);

    const std::string objectDrawLabel = cullingStats_.gpuCulling
                                            ? "MainHDRPass IndirectDrawItems " + std::to_string(mainDrawItemCount) +
                                                  " GPU culling batches " + std::to_string(meshDrawBatches_.size())
                                            : "MainHDRPass IndirectDrawItems " + std::to_string(mainDrawItemCount) +
                                                  " visible objects " + std::to_string(cullingStats_.visibleObjects) +
                                                  "/" + std::to_string(cullingStats_.totalObjects) + " batches " +
                                                  std::to_string(meshDrawBatches_.size());
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
    const uint32_t toneMappingOperator = toneMappingOperatorValue(toneMappingSettings_.operatorType);
    const float exposure = currentToneMappingExposure();
    if (multiDrawIndirectActive) {
        if (bindlessDescriptorSetsBound) {
            const PushConstants pushConstants{objectFrameDataBaseAddress, 0, toneMappingOperator, exposure};
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
                    static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData)),
                0,
                toneMappingOperator,
                exposure};

            // Fallback recording pushes the address of this draw's object data; firstInstance stays zero.
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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

    renderGraph_.endMainHdrPass();
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::MainPass);
    rhi::debug::endLabel(commandBuffer);

    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Bloom);

    rhi::debug::beginLabel(commandBuffer, "BloomExtractPass");
    renderGraph_.beginBloomExtractPass();
    setViewportAndScissor(commandBuffer, bloomExtent_);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomExtractPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomExtractPipeline_.layout(),
                            0,
                            1,
                            &bloomExtractDescriptorSet_,
                            0,
                            nullptr);
    const BloomExtractPushConstants bloomExtractPushConstants{bloomSettings_.threshold};
    vkCmdPushConstants(commandBuffer,
                       bloomExtractPipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(BloomExtractPushConstants)),
                       &bloomExtractPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endBloomExtractPass();
    rhi::debug::endLabel(commandBuffer);

    const BloomBlurPushConstants horizontalBlurPushConstants{
        glm::vec2{1.0f / static_cast<float>(bloomExtent_.width), 1.0f / static_cast<float>(bloomExtent_.height)},
        1u,
        0u};
    rhi::debug::beginLabel(commandBuffer, "BloomBlurHorizontal");
    renderGraph_.beginBloomBlurPass(true);
    setViewportAndScissor(commandBuffer, bloomExtent_);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomBlurPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomBlurPipeline_.layout(),
                            0,
                            1,
                            &bloomBlurHorizontalDescriptorSet_,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       bloomBlurPipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(BloomBlurPushConstants)),
                       &horizontalBlurPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endBloomBlurPass();
    rhi::debug::endLabel(commandBuffer);

    const BloomBlurPushConstants verticalBlurPushConstants{
        glm::vec2{1.0f / static_cast<float>(bloomExtent_.width), 1.0f / static_cast<float>(bloomExtent_.height)},
        0u,
        0u};
    rhi::debug::beginLabel(commandBuffer, "BloomBlurVertical");
    renderGraph_.beginBloomBlurPass(false);
    setViewportAndScissor(commandBuffer, bloomExtent_);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomBlurPipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomBlurPipeline_.layout(),
                            0,
                            1,
                            &bloomBlurVerticalDescriptorSet_,
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       bloomBlurPipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(BloomBlurPushConstants)),
                       &verticalBlurPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endBloomBlurPass();
    rhi::debug::endLabel(commandBuffer);

    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Bloom);

    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::AutoExposure);
    recordLuminanceCommands(commandBuffer);
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::AutoExposure);

    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::HistogramExposure);
    recordHistogramCommands(commandBuffer);
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::HistogramExposure);

    rhi::debug::beginLabel(commandBuffer, "CompositePass");
    timestampQuery_.writeBegin(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Composite);
    renderGraph_.beginCompositePass();
    setViewportAndScissor(commandBuffer, swapchain_.extent());
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            compositePipeline_.layout(),
                            0,
                            1,
                            &compositeDescriptorSet_,
                            0,
                            nullptr);
    const CompositePushConstants compositePushConstants{
        currentToneMappingExposure(),
        bloomSettings_.enabled ? std::max(bloomSettings_.intensity, 0.0f) : 0.0f,
        toneMappingOperatorValue(toneMappingSettings_.operatorType),
        bloomSettings_.enabled ? 1u : 0u};
    vkCmdPushConstants(commandBuffer,
                       compositePipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(CompositePushConstants)),
                       &compositePushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endCompositePass();
    timestampQuery_.writeEnd(commandBuffer, currentFrame_, rhi::VulkanTimestampQuery::Timer::Composite);
    rhi::debug::endLabel(commandBuffer);

    rhi::debug::beginLabel(commandBuffer, "ImGuiPass");
    renderGraph_.beginImGuiPass();
    imguiLayer_.render(commandBuffer);
    renderGraph_.endImGuiPass();
    rhi::debug::endLabel(commandBuffer);

    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endFrame();
}

} // namespace ve
