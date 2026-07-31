#pragma once

// Internal renderer helpers shared between Renderer.cpp and RendererDebugUi.cpp.
//
// These were previously a file-local anonymous namespace at the top of
// Renderer.cpp. They are relocated here so the debug-UI translation unit can
// reuse them after the debug rendering code was split out of Renderer.cpp.
//
// The helpers live in an anonymous namespace: every translation unit that
// includes this header gets its own internal-linkage copy. Each unit only uses
// a subset of the helpers, so the unused-helper diagnostics are intentionally
// suppressed for this block.

#include "renderer/Renderer.h"

#include "core/Logger.h"
#include "core/PngWriter.h"
#include "core/Window.h"
#include "renderer/Bounds.h"
#include "renderer/ExposureTypes.h"
#include "rhi/VulkanDebugUtils.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-const-variable"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

namespace ve {

namespace {

using Json = nlohmann::json;

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
    glm::vec4 emissiveFactor{0.0f, 0.0f, 0.0f, 0.0f};
    glm::mat4 currMvpNoJitter{1.0f};
    glm::mat4 prevMvpNoJitter{1.0f};
};

// Mirrors the shader's std430 buffer_reference block. std430 stores mat4 as
// four 16-byte columns and vec4/uvec4 as 16 bytes. The four lightMvp matrices
// intentionally duplicate cascade data per draw for this educational milestone;
// a later scene/light buffer can remove that per-object cost.
// materialParams.x = metallic, y = roughness, z = multiScatterStrength,
// and w is the alpha-test cutoff: >= 0 clips fragments whose base-color alpha
// falls below it (glTF MASK), < 0 disables the test (OPAQUE and BLEND). See
// renderer::kNoAlphaTestCutoff.
// cascadeSplits stores positive camera-view depths for cascades 0..3.
// cameraPosition.xyz is the world-space camera position, and w stores the
// cascade debug-color toggle as 0.0 or 1.0.
// cameraForward.xyz is the camera forward vector, and w stores cascade count.
// textureIndices.x = base color, y = normal, z = metallic-roughness, w = emissive.
// emissiveFactor.rgb is the emissive color factor; emissiveFactor.w is 1.0 when an
// emissive texture (sampled from the base-color array at textureIndices.w) is present.
// currMvpNoJitter/prevMvpNoJitter are the unjittered current/previous-frame MVP
// matrices used to derive per-pixel motion vectors; rasterization keeps using the
// jittered mvp so TAA jitter never leaks into the velocity buffer.
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
static_assert(offsetof(ObjectFrameData, emissiveFactor) == 544);
static_assert(offsetof(ObjectFrameData, currMvpNoJitter) == 560);
static_assert(offsetof(ObjectFrameData, prevMvpNoJitter) == 624);
static_assert(sizeof(ObjectFrameData) == 688);

constexpr uint32_t kMaxFrameObjects = 1024;
constexpr uint32_t kMaxDrawItems = 1024;
// The skinned demo mesh borrows the last object-data slot so it can reuse the
// per-frame ObjectFrameData buffer without colliding with the scene draw items.
constexpr uint32_t kSkinnedObjectFrameSlot = kMaxDrawItems - 1;
constexpr uint32_t kMaxMaterialDescriptorSets = 256;
constexpr uint32_t kGpuCullLocalSize = 64;
constexpr uint32_t kMaxMeshDrawBatches = kMaxDrawItems;
// [total, visible, frustum-culled, occlusion-culled, phase-2 rescued]
constexpr uint32_t kGpuCullStatsCounterCount = 5;
constexpr uint32_t kGpuCullStatsCounterOffset = kMaxMeshDrawBatches;
constexpr VkDeviceSize kBatchVisibleCountBufferSize = kMaxMeshDrawBatches * sizeof(uint32_t);
constexpr VkDeviceSize kGpuCullCountBufferSize =
    (kMaxMeshDrawBatches + kGpuCullStatsCounterCount) * sizeof(uint32_t);
constexpr float kUnboundedCullExtent = 100000000.0f;
constexpr VkFormat kSceneColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kBloomColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// UV-space motion vectors written by the main HDR pass as its second color
// attachment and consumed by the TAA resolve for history reprojection.
constexpr VkFormat kVelocityFormat = VK_FORMAT_R16G16_SFLOAT;
// Thin G-buffer written as the main pass's third attachment: octahedral-encoded
// world-space shading normal (RG), roughness (B), metallic (A). Consumed by the
// SSR trace pass.
constexpr VkFormat kNormalRoughnessFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// Single-channel GTAO visibility term (1 = fully lit) written by the GTAO pass
// and multiplied into scene color by the composite.
constexpr VkFormat kAmbientOcclusionFormat = VK_FORMAT_R8_UNORM;
constexpr VkFormat kDepthPyramidFormat = VK_FORMAT_R32_SFLOAT;
constexpr uint32_t kDepthPyramidLocalSizeX = 8;
constexpr uint32_t kDepthPyramidLocalSizeY = 8;
constexpr uint32_t kMaxBloomMipChainLevels = 4;
constexpr uint32_t kTaaHistoryCount = 2;
constexpr uint32_t kTaaJitterSampleCount = 8;
constexpr std::string_view kNoSavedSceneFoundMessage = "No saved scene found. Use Save Scene first.";

const glm::vec4 kDirectionalLightDirection{0.35f, -0.65f, -0.55f, 0.0f};
const glm::vec4 kDirectionalLightColor{0.85f, 0.85f, 0.85f, 1.0f};
const glm::vec4 kAmbientLightColor{0.15f, 0.15f, 0.15f, 1.0f};
const glm::vec4 kPortfolioLightDirection{-0.38f, -0.78f, -0.48f, 0.0f};
const glm::vec4 kPortfolioLightColor{1.08f, 1.03f, 0.94f, 1.0f};
const glm::vec4 kPortfolioAmbientLightColor{0.20f, 0.22f, 0.24f, 1.0f};
// Portfolio material-slot indices and occlusion-test scene constants now live in
// renderer/SceneBuilder.h alongside the scene-layout logic that consumes them.

struct PushConstants {
    VkDeviceAddress objectFrameDataAddress = 0;
    uint32_t cascadeIndex = 0;
    uint32_t toneMappingOperator = 0;
    float exposure = 1.0f;
    // Punctual lighting: the main HDR fragment shader reads lightCount lights
    // from lightBufferAddress via buffer_reference. When useClustered is set it
    // instead walks the per-froxel light list (clusterGridAddress +
    // lightIndexListAddress) produced by the cluster culling compute passes.
    // Shadow and skybox passes leave these fields zero (they never read lights).
    uint32_t lightCount = 0;
    VkDeviceAddress lightBufferAddress = 0;
    VkDeviceAddress clusterGridAddress = 0;
    VkDeviceAddress lightIndexListAddress = 0;
    float clusterZNear = 0.1f;
    float clusterZFar = 100.0f;
    float screenWidth = 0.0f;
    float screenHeight = 0.0f;
    uint32_t useClustered = 0;
    // Debug: when set, the main pass outputs a per-cluster light-count heatmap
    // instead of shaded color (requires the clustered path to be active).
    uint32_t debugClusterHeatmap = 0;
    // Skinning: the skinned vertex shader reads the joint-matrix palette from
    // this address via buffer_reference. Zero on all non-skinned draws.
    VkDeviceAddress jointMatricesAddress = 0;
};

static_assert(offsetof(PushConstants, objectFrameDataAddress) == 0);
static_assert(offsetof(PushConstants, cascadeIndex) == 8);
static_assert(offsetof(PushConstants, toneMappingOperator) == 12);
static_assert(offsetof(PushConstants, exposure) == 16);
static_assert(offsetof(PushConstants, lightCount) == 20);
static_assert(offsetof(PushConstants, lightBufferAddress) == 24);
static_assert(offsetof(PushConstants, clusterGridAddress) == 32);
static_assert(offsetof(PushConstants, lightIndexListAddress) == 40);
static_assert(offsetof(PushConstants, clusterZNear) == 48);
static_assert(offsetof(PushConstants, clusterZFar) == 52);
static_assert(offsetof(PushConstants, screenWidth) == 56);
static_assert(offsetof(PushConstants, screenHeight) == 60);
static_assert(offsetof(PushConstants, useClustered) == 64);
static_assert(offsetof(PushConstants, debugClusterHeatmap) == 68);
static_assert(offsetof(PushConstants, jointMatricesAddress) == 72);
static_assert(sizeof(PushConstants) <= 128);

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

// occlusionViewProjection projects against the previous frame's pyramid in
// phase 1; occlusionViewProjectionPhase2 is the current frame's unjittered VP
// used when phase 2 re-tests candidates against the mid-frame pyramid.
struct GpuCullFrameParams {
    glm::mat4 occlusionViewProjection{1.0f};
    glm::mat4 occlusionViewProjectionPhase2{1.0f};
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 viewportAndMipCount{0.0f};
    glm::vec4 occlusionSettings{0.0f};
    glm::uvec4 counterAndFlags{0, 0, 0, 0};
};

static_assert(offsetof(GpuCullFrameParams, occlusionViewProjection) == 0);
static_assert(offsetof(GpuCullFrameParams, occlusionViewProjectionPhase2) == 64);
static_assert(offsetof(GpuCullFrameParams, cameraPosition) == 128);
static_assert(offsetof(GpuCullFrameParams, viewportAndMipCount) == 144);
static_assert(offsetof(GpuCullFrameParams, occlusionSettings) == 160);
static_assert(offsetof(GpuCullFrameParams, counterAndFlags) == 176);
static_assert(sizeof(GpuCullFrameParams) == 192);

struct DepthPyramidPushConstants {
    glm::uvec4 sizes{0, 0, 0, 0};
};

static_assert(sizeof(DepthPyramidPushConstants) == 16);

// prevViewProjection is the previous frame's unjittered view-projection; the
// fragment shader projects the sky direction with w=0 (translation drops out)
// to derive rotation-only sky motion vectors. Exactly 128 bytes, the guaranteed
// push-constant minimum, so no room for further fields.
struct SkyboxPushConstants {
    glm::mat4 inverseViewProjection{1.0f};
    glm::mat4 prevViewProjection{1.0f};
};

static_assert(offsetof(SkyboxPushConstants, inverseViewProjection) == 0);
static_assert(offsetof(SkyboxPushConstants, prevViewProjection) == sizeof(glm::mat4));
static_assert(sizeof(SkyboxPushConstants) == 128);

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

struct BloomDownsamplePushConstants {
    glm::vec2 texelSize{1.0f, 1.0f};
    float threshold = 1.0f;
    uint32_t applyThreshold = 0;
};

static_assert(offsetof(BloomDownsamplePushConstants, texelSize) == 0);
static_assert(offsetof(BloomDownsamplePushConstants, threshold) == 8);
static_assert(offsetof(BloomDownsamplePushConstants, applyThreshold) == 12);
static_assert(sizeof(BloomDownsamplePushConstants) == 16);

struct BloomUpsamplePushConstants {
    glm::vec2 texelSize{1.0f, 1.0f};
    float radius = 1.0f;
    float padding = 0.0f;
};

static_assert(offsetof(BloomUpsamplePushConstants, texelSize) == 0);
static_assert(offsetof(BloomUpsamplePushConstants, radius) == 8);
static_assert(sizeof(BloomUpsamplePushConstants) == 16);

struct CompositePushConstants {
    float exposure = 1.0f;
    float bloomIntensity = 0.1f;
    uint32_t toneMappingOperator = 0;
    uint32_t bloomEnabled = 1;
    uint32_t bloomMethod = 1;
    uint32_t useGpuExposure = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    glm::mat4 invProjection{1.0f};
    glm::vec4 ssaoParams0{0.5f, 0.025f, 1.0f, 2.0f}; // radius, bias, intensity, power
    glm::vec4 ssaoParams1{0.0f, 16.0f, 0.0f, 0.0f};  // enabled, sampleCount
};

static_assert(offsetof(CompositePushConstants, exposure) == 0);
static_assert(offsetof(CompositePushConstants, bloomIntensity) == 4);
static_assert(offsetof(CompositePushConstants, toneMappingOperator) == 8);
static_assert(offsetof(CompositePushConstants, bloomEnabled) == 12);
static_assert(offsetof(CompositePushConstants, bloomMethod) == 16);
static_assert(offsetof(CompositePushConstants, useGpuExposure) == 20);
static_assert(offsetof(CompositePushConstants, invProjection) == 32);
static_assert(offsetof(CompositePushConstants, ssaoParams0) == 96);
static_assert(offsetof(CompositePushConstants, ssaoParams1) == 112);
static_assert(sizeof(CompositePushConstants) == 128);

struct TaaResolvePushConstants {
    glm::vec2 texelSize{1.0f, 1.0f};
    float feedback = 0.88f;
    uint32_t historyValid = 0;
    uint32_t neighborhoodClampEnabled = 1;
    uint32_t reprojectionEnabled = 1;
    uint32_t depthDilationEnabled = 1;
    uint32_t padding = 0;
};

static_assert(offsetof(TaaResolvePushConstants, texelSize) == 0);
static_assert(offsetof(TaaResolvePushConstants, feedback) == 8);
static_assert(offsetof(TaaResolvePushConstants, historyValid) == 12);
static_assert(offsetof(TaaResolvePushConstants, neighborhoodClampEnabled) == 16);
static_assert(offsetof(TaaResolvePushConstants, reprojectionEnabled) == 20);
static_assert(offsetof(TaaResolvePushConstants, depthDilationEnabled) == 24);
static_assert(sizeof(TaaResolvePushConstants) == 32);


struct RenderTargetDebugMetadata {
    const char* debugName = "";
    std::string dimensions;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    std::string usage;
    bool previewable = false;
    std::string sampledAs;
};

uint32_t toneMappingOperatorValue(int operatorType)
{
    return operatorType == 1 ? 1u : 0u;
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

const char* renderObjectSourceTypeName(renderer::RenderObjectSourceType sourceType)
{
    switch (sourceType) {
    case renderer::RenderObjectSourceType::BuiltInFallbackCube:
        return "built-in fallback cube";
    case renderer::RenderObjectSourceType::ImportedGltf:
        return "imported glTF";
    case renderer::RenderObjectSourceType::PortfolioShowcase:
        return "portfolio showcase";
    case renderer::RenderObjectSourceType::OcclusionTest:
        return "occlusion test";
    }

    return "unknown";
}

std::string formatVec3(const glm::vec3& value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << "(" << value.x << ", " << value.y << ", " << value.z << ")";
    return stream.str();
}

std::string formatAabb(const renderer::Aabb& bounds)
{
    if (!bounds.valid()) {
        return "invalid";
    }

    return "min " + formatVec3(bounds.min) + ", max " + formatVec3(bounds.max);
}

std::string formatMatrixRow(const glm::mat4& matrix, int row)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << "[" << matrix[0][row] << ", " << matrix[1][row] << ", "
           << matrix[2][row] << ", " << matrix[3][row] << "]";
    return stream.str();
}

std::string transformDebugSummary(const renderer::Transform& transform)
{
    if (transform.useMatrixOverride) {
        return "matrix override, translation " + formatVec3(glm::vec3(transform.matrixOverride[3]));
    }

    return "T " + formatVec3(transform.position) + ", R " + formatVec3(transform.rotationRadians) + ", S " +
           formatVec3(transform.scale);
}

uint32_t meshSubmeshCount(const renderer::Mesh* mesh)
{
    if (!mesh || !mesh->valid()) {
        return 0;
    }
    if (mesh->hasSubMeshes()) {
        return static_cast<uint32_t>(mesh->primitives().size());
    }
    return mesh->indexCount() > 0 ? 1U : 0U;
}

uint32_t calculateDepthPyramidMipLevels(VkExtent2D extent)
{
    uint32_t maxDimension = std::max(extent.width, extent.height);
    uint32_t mipLevels = 1;
    while (maxDimension > 1) {
        maxDimension /= 2;
        ++mipLevels;
    }
    return mipLevels;
}

uint32_t calculateBloomMipChainLevels(VkExtent2D extent)
{
    VkExtent2D mipExtent{std::max(1u, extent.width / 2u), std::max(1u, extent.height / 2u)};
    uint32_t mipLevels = 1;
    while (mipLevels < kMaxBloomMipChainLevels && (mipExtent.width > 1u || mipExtent.height > 1u)) {
        mipExtent.width = std::max(1u, mipExtent.width / 2u);
        mipExtent.height = std::max(1u, mipExtent.height / 2u);
        ++mipLevels;
    }
    return mipLevels;
}

VkExtent2D mipExtent(VkExtent2D baseExtent, uint32_t mipLevel)
{
    VkExtent2D extent = baseExtent;
    for (uint32_t level = 0; level < mipLevel; ++level) {
        extent.width = std::max(1u, extent.width / 2u);
        extent.height = std::max(1u, extent.height / 2u);
    }
    return extent;
}

VkExtent2D bloomMipExtent(VkExtent2D swapchainExtent, uint32_t mipLevel)
{
    const VkExtent2D baseExtent{std::max(1u, swapchainExtent.width / 2u),
                                std::max(1u, swapchainExtent.height / 2u)};
    return mipExtent(baseExtent, mipLevel);
}

bool depthPyramidFormatSupports(VkPhysicalDevice physicalDevice, VkFormatFeatureFlags requiredFeatures)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, kDepthPyramidFormat, &properties);

    return (properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures;
}

VkImageMemoryBarrier2 imageBarrier(VkImage image,
                                   VkImageAspectFlags aspectMask,
                                   VkImageLayout oldLayout,
                                   VkImageLayout newLayout,
                                   VkPipelineStageFlags2 srcStageMask,
                                   VkAccessFlags2 srcAccessMask,
                                   VkPipelineStageFlags2 dstStageMask,
                                   VkAccessFlags2 dstAccessMask,
                                   uint32_t baseMipLevel,
                                   uint32_t levelCount)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseMipLevel = baseMipLevel;
    barrier.subresourceRange.levelCount = levelCount;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

void recordImageBarrier(VkCommandBuffer commandBuffer, const VkImageMemoryBarrier2& barrier)
{
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

float maxMatrixDifference(const glm::mat4& lhs, const glm::mat4& rhs)
{
    float maxDifference = 0.0f;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            maxDifference = std::max(maxDifference, std::abs(lhs[column][row] - rhs[column][row]));
        }
    }
    return maxDifference;
}

const char* materialNameOrFallback(const renderer::Material* material)
{
    if (!material) {
        return "(none)";
    }
    if (material->debugName.empty()) {
        return "(unnamed material)";
    }
    return material->debugName.c_str();
}

std::string meshDebugLabel(const renderer::Mesh* mesh)
{
    if (!mesh) {
        return "(none)";
    }
    if (!mesh->debugName().empty()) {
        return mesh->debugName();
    }

    std::ostringstream stream;
    stream << "unnamed mesh " << static_cast<const void*>(mesh);
    return stream.str();
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

#if defined(__APPLE__)
const std::filesystem::path& macosBundleResourcesPath()
{
    static const std::filesystem::path resourcesPath = [] {
        uint32_t pathSize = 0;
        _NSGetExecutablePath(nullptr, &pathSize);
        if (pathSize == 0) {
            return std::filesystem::path{};
        }

        std::string executablePath(pathSize, '\0');
        if (_NSGetExecutablePath(executablePath.data(), &pathSize) != 0) {
            return std::filesystem::path{};
        }
        executablePath.resize(std::char_traits<char>::length(executablePath.c_str()));

        std::filesystem::path executable(executablePath);
        std::error_code error;
        if (!executable.is_absolute()) {
            executable = std::filesystem::absolute(executable, error);
            if (error) {
                return std::filesystem::path{};
            }
        }

        const std::filesystem::path canonicalExecutable = std::filesystem::weakly_canonical(executable, error);
        if (!error) {
            executable = canonicalExecutable;
        }

        const std::filesystem::path macosDirectory = executable.parent_path();
        const std::filesystem::path contentsDirectory = macosDirectory.parent_path();
        if (macosDirectory.filename() != "MacOS" || contentsDirectory.filename() != "Contents") {
            return std::filesystem::path{};
        }

        const std::filesystem::path resources = contentsDirectory / "Resources";
        return std::filesystem::is_directory(resources) ? resources : std::filesystem::path{};
    }();

    return resourcesPath;
}

std::filesystem::path bundledResourceDirectory(const char* directoryName)
{
    const std::filesystem::path& resources = macosBundleResourcesPath();
    if (resources.empty()) {
        return {};
    }

    const std::filesystem::path directory = resources / directoryName;
    return std::filesystem::is_directory(directory) ? directory : std::filesystem::path{};
}
#endif

std::filesystem::path shaderDirectory()
{
#if defined(__APPLE__)
    const std::filesystem::path bundledDirectory = bundledResourceDirectory("shaders");
    if (!bundledDirectory.empty()) {
        return bundledDirectory;
    }
#endif

#if defined(VULKAN_ENGINE_SHADER_DIR)
    return std::filesystem::path(VULKAN_ENGINE_SHADER_DIR);
#else
    return std::filesystem::path("shaders");
#endif
}

std::filesystem::path assetDirectory()
{
#if defined(__APPLE__)
    const std::filesystem::path bundledDirectory = bundledResourceDirectory("assets");
    if (!bundledDirectory.empty()) {
        return bundledDirectory;
    }
#endif

#if defined(VULKAN_ENGINE_ASSET_DIR)
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR);
#else
    return std::filesystem::path("assets");
#endif
}

std::filesystem::path configDirectory()
{
#if defined(__APPLE__)
    const std::filesystem::path bundledDirectory = bundledResourceDirectory("config");
    if (!bundledDirectory.empty()) {
        return bundledDirectory;
    }
#endif

#if defined(VULKAN_ENGINE_CONFIG_DIR)
    return std::filesystem::path(VULKAN_ENGINE_CONFIG_DIR);
#else
    return std::filesystem::path("config");
#endif
}

std::filesystem::path shaderPath(const char* filename)
{
    return shaderDirectory() / filename;
}

std::filesystem::path assetPath(const char* relativePath)
{
    return assetDirectory() / relativePath;
}

std::filesystem::path materialAssetPath(std::string_view filename)
{
    return assetPath("materials") / std::string(filename);
}

std::filesystem::path defaultRuntimeSettingsPath()
{
    return configDirectory() / "runtime_settings.json";
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

const char* materialSourceName(renderer::MaterialSource source)
{
    switch (source) {
    case renderer::MaterialSource::BuiltIn:
        return "built-in/procedural material";
    case renderer::MaterialSource::Gltf:
        return "glTF material";
    case renderer::MaterialSource::MaterialAsset:
        return "JSON material asset";
    case renderer::MaterialSource::Fallback:
        return "fallback material";
    }

    return "unknown";
}

const char* textureDebugSourceName(rhi::TextureDebugSource source)
{
    switch (source) {
    case rhi::TextureDebugSource::Unknown:
        return "unknown";
    case rhi::TextureDebugSource::LoadedFromDisk:
        return "loaded from disk";
    case rhi::TextureDebugSource::GltfExternalFile:
        return "imported from glTF external file";
    case rhi::TextureDebugSource::GltfEmbeddedData:
        return "imported from glTF embedded/data URI";
    case rhi::TextureDebugSource::ProceduralFallback:
        return "procedural fallback";
    }

    return "unknown";
}

const char* vkFormatName(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_UNDEFINED:
        return "VK_FORMAT_UNDEFINED";
    case VK_FORMAT_R8G8_UNORM:
        return "VK_FORMAT_R8G8_UNORM";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "VK_FORMAT_R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "VK_FORMAT_R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "VK_FORMAT_B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "VK_FORMAT_B8G8R8A8_SRGB";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "VK_FORMAT_R16G16B16A16_SFLOAT";
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return "VK_FORMAT_R32G32B32A32_SFLOAT";
    case VK_FORMAT_D16_UNORM:
        return "VK_FORMAT_D16_UNORM";
    case VK_FORMAT_D32_SFLOAT:
        return "VK_FORMAT_D32_SFLOAT";
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return "VK_FORMAT_D32_SFLOAT_S8_UINT";
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return "VK_FORMAT_D24_UNORM_S8_UINT";
    default:
        return "VkFormat(other)";
    }
}

const char* imageLayoutName(VkImageLayout layout)
{
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        return "VK_IMAGE_LAYOUT_UNDEFINED";
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
        return "VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
        return "VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL";
    case VK_IMAGE_LAYOUT_GENERAL:
        return "VK_IMAGE_LAYOUT_GENERAL";
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return "VK_IMAGE_LAYOUT_PRESENT_SRC_KHR";
    default:
        return "VkImageLayout(other)";
    }
}

std::filesystem::path projectRootPath()
{
#if defined(__APPLE__)
    const std::filesystem::path& resources = macosBundleResourcesPath();
    if (!resources.empty()) {
        return resources;
    }
#endif

#if defined(VULKAN_ENGINE_ASSET_DIR)
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

std::filesystem::path resolveProjectPath(const std::string& pathString)
{
    if (pathString.empty()) {
        return {};
    }

    std::filesystem::path path(pathString);
    if (path.is_absolute()) {
        return path.lexically_normal();
    }
    return (projectRootPath() / path).lexically_normal();
}

std::string stableProjectPathString(const std::filesystem::path& path)
{
    if (path.empty()) {
        return {};
    }

    std::error_code absolutePathError;
    const std::filesystem::path absolutePath = std::filesystem::absolute(path, absolutePathError);
    std::error_code absoluteRootError;
    const std::filesystem::path absoluteRoot = std::filesystem::absolute(projectRootPath(), absoluteRootError);
    if (!absolutePathError && !absoluteRootError) {
        std::error_code relativeError;
        const std::filesystem::path relativePath = std::filesystem::relative(absolutePath, absoluteRoot, relativeError);
        const std::string relativeString = relativePath.generic_string();
        if (!relativeError && !relativeString.empty() && relativeString.rfind("../", 0) != 0 &&
            relativeString != "..") {
            return relativeString;
        }
    }

    return path.lexically_normal().generic_string();
}

std::filesystem::path resolveMaterialTexturePath(const std::filesystem::path& materialPath,
                                                 const std::filesystem::path& texturePath)
{
    if (texturePath.empty()) {
        return {};
    }
    if (texturePath.is_absolute()) {
        return texturePath.lexically_normal();
    }

    const std::filesystem::path materialRelativePath =
        materialPath.empty() ? std::filesystem::path{} : (materialPath.parent_path() / texturePath).lexically_normal();
    if (!materialRelativePath.empty() && std::filesystem::exists(materialRelativePath)) {
        return materialRelativePath;
    }

    const std::filesystem::path projectRelativePath = (projectRootPath() / texturePath).lexically_normal();
    if (std::filesystem::exists(projectRelativePath)) {
        return projectRelativePath;
    }

    const std::filesystem::path assetRelativePath = (assetPath("") / texturePath).lexically_normal();
    if (std::filesystem::exists(assetRelativePath)) {
        return assetRelativePath;
    }

    return materialRelativePath.empty() ? projectRelativePath : materialRelativePath;
}

std::filesystem::path portfolioScreenshotDirectory()
{
    return projectRootPath() / "screenshots";
}

std::filesystem::path defaultSceneDocumentPath()
{
    return assetPath("scenes/default.scene.json");
}

renderer::Camera defaultCameraPreset()
{
    renderer::Camera camera{};
    camera.position = {0.0f, 0.35f, 5.5f};
    camera.target = {0.0f, 0.1f, 0.0f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.verticalFovRadians = glm::radians(60.0f);
    camera.nearPlane = 0.1f;
    camera.farPlane = 100.0f;
    return camera;
}

renderer::Camera portfolioCameraPreset()
{
    renderer::Camera camera{};
    camera.position = {1.82f, 0.78f, 2.92f};
    camera.target = {0.00f, -0.08f, 0.18f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.verticalFovRadians = glm::radians(45.0f);
    camera.nearPlane = 0.1f;
    camera.farPlane = 100.0f;
    return camera;
}

renderer::Camera occlusionTestCameraPreset()
{
    renderer::Camera camera{};
    camera.position = {0.0f, 1.45f, 7.2f};
    camera.target = {0.0f, 1.05f, -6.5f};
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.verticalFovRadians = glm::radians(55.0f);
    camera.nearPlane = 0.1f;
    camera.farPlane = 70.0f;
    return camera;
}

uint32_t renderObjectEditorId(const renderer::RenderObject& object)
{
    return object.sceneObjectId != 0 ? object.sceneObjectId : object.debugId;
}

bool isFiniteVec3(const glm::vec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

glm::vec3 normalizedOrFallback(const glm::vec3& value, const glm::vec3& fallback)
{
    if (!isFiniteVec3(value)) {
        return fallback;
    }

    const float length = glm::length(value);
    if (length <= 0.0001f) {
        return fallback;
    }

    return value / length;
}

Json vec3ToJson(const glm::vec3& value)
{
    return Json::array({value.x, value.y, value.z});
}

Json mat4ToJson(const glm::mat4& matrix)
{
    Json values = Json::array();
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            values.push_back(matrix[column][row]);
        }
    }
    return values;
}

const Json* jsonObjectMember(const Json& object, const char* name)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return nullptr;
    }
    if (!it->is_object()) {
        throw std::runtime_error(std::string("Expected object member '") + name + "'.");
    }
    return &(*it);
}

bool readJsonBool(const Json& object, const char* name, bool& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_boolean()) {
        throw std::runtime_error(std::string("Expected boolean member '") + name + "'.");
    }
    value = it->get<bool>();
    return true;
}

bool readJsonFloat(const Json& object, const char* name, float& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_number()) {
        throw std::runtime_error(std::string("Expected numeric member '") + name + "'.");
    }
    value = it->get<float>();
    return true;
}

bool readJsonString(const Json& object, const char* name, std::string& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string("Expected string member '") + name + "'.");
    }
    value = it->get<std::string>();
    return true;
}

bool readJsonUint32(const Json& object, const char* name, uint32_t& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_number_integer() && !it->is_number_unsigned()) {
        throw std::runtime_error(std::string("Expected integer member '") + name + "'.");
    }

    uint64_t parsed = 0;
    if (it->is_number_unsigned()) {
        parsed = it->get<uint64_t>();
    } else {
        const int64_t signedParsed = it->get<int64_t>();
        if (signedParsed < 0) {
            throw std::runtime_error(std::string("Expected non-negative integer member '") + name + "'.");
        }
        parsed = static_cast<uint64_t>(signedParsed);
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("Integer member '") + name + "' exceeds uint32 range.");
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool readJsonVec3Value(const Json& value, const char* name, glm::vec3& out)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string("Expected vec3 array member '") + name + "'.");
    }
    for (const Json& component : value) {
        if (!component.is_number()) {
            throw std::runtime_error(std::string("Expected numeric vec3 member '") + name + "'.");
        }
    }

    out = {value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
    return true;
}

bool readJsonVec3(const Json& object, const char* name, glm::vec3& value)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return false;
    }
    return readJsonVec3Value(*it, name, value);
}

bool readJsonMat4(const Json& object, const char* name, glm::mat4& matrix)
{
    const auto it = object.find(name);
    if (it == object.end()) {
        return false;
    }
    if (!it->is_array() || it->size() != 16) {
        throw std::runtime_error(std::string("Expected mat4 array member '") + name + "'.");
    }

    glm::mat4 parsed{1.0f};
    size_t valueIndex = 0;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            const Json& component = (*it)[valueIndex++];
            if (!component.is_number()) {
                throw std::runtime_error(std::string("Expected numeric mat4 member '") + name + "'.");
            }
            parsed[column][row] = component.get<float>();
        }
    }

    matrix = parsed;
    return true;
}

struct TransformComponents {
    glm::vec3 position{0.0f};
    glm::vec3 rotationRadians{0.0f};
    glm::vec3 scale{1.0f};
    bool valid = false;
};

TransformComponents editableTransformComponents(const renderer::Transform& transform)
{
    if (!transform.useMatrixOverride) {
        return {transform.position, transform.rotationRadians, transform.scale, true};
    }

    glm::vec3 scale{1.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 translation{0.0f};
    glm::vec3 skew{0.0f};
    glm::vec4 perspective{0.0f};
    const bool decomposed = glm::decompose(transform.matrixOverride, scale, orientation, translation, skew, perspective);
    if (!decomposed || !isFiniteVec3(translation) || !isFiniteVec3(scale)) {
        return {};
    }

    orientation = glm::normalize(orientation);
    // Extract Euler angles in the same intrinsic X*Y*Z order that
    // Transform::modelMatrix() composes them in, so gizmo/matrix edits round-trip
    // faithfully. glm::eulerAngles uses a different convention, which corrupted the
    // stored rotation (and the inspector readout) on the way back to TRS.
    glm::vec3 rotationRadians{0.0f};
    glm::extractEulerAngleXYZ(glm::mat4_cast(orientation), rotationRadians.x, rotationRadians.y, rotationRadians.z);
    if (!isFiniteVec3(rotationRadians)) {
        return {};
    }

    return {translation, rotationRadians, scale, true};
}

bool convertMatrixOverrideToEditableTrs(renderer::Transform& transform)
{
    if (!transform.useMatrixOverride) {
        return true;
    }

    const TransformComponents components = editableTransformComponents(transform);
    if (!components.valid) {
        return false;
    }

    transform.position = components.position;
    transform.rotationRadians = components.rotationRadians;
    transform.scale = components.scale;
    transform.matrixOverride = glm::mat4{1.0f};
    transform.useMatrixOverride = false;
    return true;
}

Json transformToJson(const renderer::Transform& transform)
{
    const TransformComponents components = editableTransformComponents(transform);
    Json json = Json{{"mode", transform.useMatrixOverride ? "matrix" : "trs"}};

    if (components.valid) {
        json["position"] = vec3ToJson(components.position);
        json["rotationDegrees"] = vec3ToJson(glm::degrees(components.rotationRadians));
        json["scale"] = vec3ToJson(components.scale);
    }
    if (transform.useMatrixOverride) {
        json["matrix"] = mat4ToJson(transform.matrixOverride);
    }

    return json;
}

std::string pointerString(const void* pointer)
{
    std::ostringstream stream;
    stream << pointer;
    return stream.str();
}

bool supportedScreenshotFormat(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return true;
    default:
        return false;
    }
}

std::string formatVec4(const glm::vec4& value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << "(" << value.x << ", " << value.y << ", " << value.z << ", "
           << value.w << ")";
    return stream.str();
}

template <typename Handle>
uint64_t vulkanHandleValue(Handle handle)
{
    if constexpr (std::is_pointer_v<Handle>) {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    } else {
        return static_cast<uint64_t>(handle);
    }
}

std::string descriptorSetDebugString(VkDescriptorSet descriptorSet)
{
    if (descriptorSet == VK_NULL_HANDLE) {
        return "(null)";
    }

    std::ostringstream stream;
    stream << "0x" << std::hex << vulkanHandleValue(descriptorSet);
    return stream.str();
}

float historyValue(double value)
{
    if (!std::isfinite(value) || value < 0.0) {
        return 0.0f;
    }

    return static_cast<float>(value);
}

float halton(uint32_t index, uint32_t base)
{
    float result = 0.0f;
    float fraction = 1.0f / static_cast<float>(base);
    while (index > 0) {
        result += fraction * static_cast<float>(index % base);
        index /= base;
        fraction /= static_cast<float>(base);
    }
    return result;
}

std::string resourceUsageList(const renderer::RenderPassNode& pass, renderer::RenderResourceAccess access)
{
    std::ostringstream names;
    bool first = true;
    for (const renderer::RenderResourceUsage& usage : pass.resourceUsages) {
        const bool matchesRead = access == renderer::RenderResourceAccess::Read &&
                                 (usage.access == renderer::RenderResourceAccess::Read ||
                                  usage.access == renderer::RenderResourceAccess::ReadWrite);
        const bool matchesWrite = access == renderer::RenderResourceAccess::Write &&
                                  (usage.access == renderer::RenderResourceAccess::Write ||
                                   usage.access == renderer::RenderResourceAccess::ReadWrite);
        if (!matchesRead && !matchesWrite) {
            continue;
        }

        if (!first) {
            names << ", ";
        }
        names << usage.resource.name << " [" << renderer::rgAccessName(usage.declaredAccess) << "]";
        first = false;
    }

    return first ? "-" : names.str();
}

} // namespace

} // namespace ve

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
