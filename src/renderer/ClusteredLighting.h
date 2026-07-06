#pragma once

#include "renderer/ClusterGrid.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace ve::rhi {
class VulkanContext;
}

namespace ve::renderer {

// Punctual light record shared with the lighting shaders. std430 stores each
// vec4 on a 16-byte boundary, so the four members give a 64-byte runtime-array
// stride that must match the GpuLight struct declared in the GLSL.
struct GpuLight {
    glm::vec4 positionRange{0.0f, 0.0f, 0.0f, 10.0f};   // xyz = world position, w = range
    glm::vec4 colorIntensity{1.0f, 1.0f, 1.0f, 1.0f};   // rgb = color, a = intensity
    glm::vec4 directionType{0.0f, -1.0f, 0.0f, 0.0f};   // xyz = spot direction, w = type
    glm::vec4 spotScaleOffset{-1.0f, 1.0f, 0.0f, 0.0f}; // x = cos(outer), y = 1/(cos(inner)-cos(outer))
};

static_assert(sizeof(GpuLight) == 64);
static_assert(offsetof(GpuLight, positionRange) == 0);
static_assert(offsetof(GpuLight, colorIntensity) == 16);
static_assert(offsetof(GpuLight, directionType) == 32);
static_assert(offsetof(GpuLight, spotScaleOffset) == 48);

// View-space froxel AABB written by cluster_build.comp.
struct ClusterAabb {
    glm::vec4 minView{0.0f};
    glm::vec4 maxView{0.0f};
};
static_assert(sizeof(ClusterAabb) == 32);

// Per-cluster light list header written by light_cull.comp.
struct ClusterCell {
    uint32_t offset = 0;
    uint32_t count = 0;
};
static_assert(sizeof(ClusterCell) == 8);

// Per-frame parameters consumed by both cluster compute shaders. Mirrors the
// ClusterParamsBuffer block in cluster_build.comp / light_cull.comp.
struct ClusterGridParams {
    glm::mat4 viewMatrix{1.0f};
    glm::mat4 inverseProjection{1.0f};
    glm::uvec4 gridDims{0, 0, 0, 0};         // x,y,z = grid dims, w = max lights per cluster
    glm::vec4 zNearFarScreen{0.0f};          // x = zNear, y = zFar, z = screenW, w = screenH
    glm::uvec4 lightCountAndPad{0, 0, 0, 0}; // x = light count
};

static_assert(offsetof(ClusterGridParams, viewMatrix) == 0);
static_assert(offsetof(ClusterGridParams, inverseProjection) == 64);
static_assert(offsetof(ClusterGridParams, gridDims) == 128);
static_assert(offsetof(ClusterGridParams, zNearFarScreen) == 144);
static_assert(offsetof(ClusterGridParams, lightCountAndPad) == 160);
static_assert(sizeof(ClusterGridParams) == 176);

enum class LightType : uint32_t {
    Point = 0,
    Spot = 1,
};

// Clustered (Forward+) punctual lighting. Owns the CPU light list, a per-frame
// buffer-device-address light buffer, the froxel grid, and the two compute
// passes that build the grid and assign lights to froxels. The main HDR
// fragment shader reads the light list + per-cluster index list through
// push-constant addresses, so per-frame updates stay free of descriptor-set
// hazards. Grid dims must match the kCluster* constants in the GLSL.
class ClusteredLighting final {
public:
    static constexpr uint32_t kMaxLights = 1024;
    // Grid dimensions, cluster count, and kMaxLightsPerCluster come from
    // ClusterGrid.h so the CPU reference math and this subsystem cannot drift.
    static constexpr uint32_t kClusterCullLocalSize = 64;

    void create(rhi::VulkanContext& context,
                uint32_t frameCount,
                const std::filesystem::path& clusterBuildShaderPath,
                const std::filesystem::path& lightCullShaderPath);

    [[nodiscard]] std::vector<GpuLight>& lights()
    {
        return lights_;
    }
    [[nodiscard]] const std::vector<GpuLight>& lights() const
    {
        return lights_;
    }
    [[nodiscard]] uint32_t lightCount() const;
    void clear()
    {
        lights_.clear();
    }

    void addPointLight(const glm::vec3& position, const glm::vec3& color, float intensity, float range);
    void addSpotLight(const glm::vec3& position,
                      const glm::vec3& direction,
                      const glm::vec3& color,
                      float intensity,
                      float range,
                      float innerAngleRadians,
                      float outerAngleRadians);

    void upload(uint32_t frameIndex);
    void updateParams(uint32_t frameIndex,
                      const glm::mat4& viewMatrix,
                      const glm::mat4& inverseProjection,
                      float zNear,
                      float zFar,
                      float screenWidth,
                      float screenHeight);

    // Records the froxel build + light assignment compute passes for the frame.
    // Each method emits its own Synchronization2 barriers; the assignment pass
    // ends with a barrier that makes its output visible to the fragment shader.
    // asyncQueue: recording happens on the async compute queue. The trailing
    // compute->fragment barriers are skipped there (FRAGMENT is not a valid
    // stage on a compute-only queue; the submit semaphore provides the
    // cross-queue memory dependency instead).
    void recordClusterBuild(VkCommandBuffer commandBuffer, uint32_t frameIndex, bool asyncQueue = false);
    void recordLightCull(VkCommandBuffer commandBuffer, uint32_t frameIndex, bool asyncQueue = false);

    [[nodiscard]] bool available() const
    {
        return available_;
    }
    [[nodiscard]] VkDeviceAddress lightBufferAddress(uint32_t frameIndex) const;
    [[nodiscard]] VkDeviceAddress clusterGridAddress(uint32_t frameIndex) const;
    [[nodiscard]] VkDeviceAddress lightIndexListAddress(uint32_t frameIndex) const;

private:
    void createBuffers(uint32_t frameCount);
    void createDescriptorResources(uint32_t frameCount);
    void createPipelines(const std::filesystem::path& clusterBuildShaderPath,
                         const std::filesystem::path& lightCullShaderPath);

    rhi::VulkanContext* context_ = nullptr;
    bool available_ = false;

    std::vector<GpuLight> lights_;
    std::vector<rhi::VulkanBuffer> lightBuffers_;       // per-frame, host-visible, BDA
    std::vector<rhi::VulkanBuffer> clusterAabbBuffers_; // per-frame, device-local
    std::vector<rhi::VulkanBuffer> clusterGridBuffers_; // per-frame, device-local, BDA
    std::vector<rhi::VulkanBuffer> lightIndexBuffers_;  // per-frame, device-local, BDA
    std::vector<rhi::VulkanBuffer> paramsBuffers_;      // per-frame, host-visible

    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    std::vector<VkDescriptorSet> descriptorSets_;
    rhi::VulkanComputePipeline clusterBuildPipeline_;
    rhi::VulkanComputePipeline lightCullPipeline_;
};

} // namespace ve::renderer
