#pragma once

// Vulkan side of froxel volumetric fog: owns the two 3D volumes, the two
// compute pipelines, and the per-frame parameter buffers.
//
// Ownership boundary mirrors ClusteredLighting and PunctualShadows: resources
// and recording live here, while the GPU-free froxel and integration math lives
// in VolumetricFog.h so it can be unit tested without a device. The renderer
// supplies the camera/light state and the cascaded shadow map to sample.

#include "renderer/VolumetricFog.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanImage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace ve::rhi {
class VulkanContext;
}

namespace ve::renderer {

inline constexpr uint32_t kMaxFogCascades = 4;

// Mirrors the FogParamsBuffer block in fog_inject.comp. std140, so every member
// sits on a 16-byte boundary already.
struct FogInjectParams {
    glm::mat4 inverseView{1.0f};
    glm::mat4 inverseProjection{1.0f};
    glm::mat4 cascadeViewProjection[kMaxFogCascades]{};
    glm::vec4 cascadeSplits{0.0f};
    glm::vec4 cameraPosition{0.0f};
    // xyz = direction the light travels along, w = cascade count
    glm::vec4 lightDirection{0.0f, -1.0f, 0.0f, 1.0f};
    glm::vec4 lightColor{1.0f};
    // x = density, y = max distance, z = anisotropy, w = height falloff
    glm::vec4 fogParams{0.02f, kDefaultFogMaxDistance, 0.3f, 0.0f};
    // x = base height, y = ambient scale
    glm::vec4 fogParams2{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 scatteringColor{1.0f};
    glm::vec4 ambientColor{0.0f};
    // Reprojection of the previous frame's volume. Needed as two matrices, not
    // one: the view-projection places a froxel in the previous frame's screen
    // tile, and the view alone recovers its previous view depth for the slice.
    glm::mat4 previousViewProjection{1.0f};
    glm::mat4 previousView{1.0f};
    // xyz = sub-froxel jitter in [0,1), w = how much history to keep.
    glm::vec4 jitterAndBlend{0.5f, 0.5f, 0.5f, 0.0f};
};

// Mirrors the push constant block in fog_inject.comp. The buffer addresses come
// from ClusteredLighting and PunctualShadows, so fog reuses the light lists and
// shadow tiles those already produce rather than building its own.
struct FogInjectPushConstants {
    VkDeviceAddress lightBufferAddress = 0;
    VkDeviceAddress clusterGridAddress = 0;
    VkDeviceAddress lightIndexListAddress = 0;
    VkDeviceAddress punctualShadowSlotAddress = 0;
    uint32_t lightCount = 0;
    uint32_t useClustered = 0;
    float clusterZNear = 0.1f;
    float clusterZFar = 100.0f;
};

static_assert(sizeof(FogInjectPushConstants) == 48);
static_assert(sizeof(FogInjectPushConstants) <= 128);

// Mirrors FogIntegrateParams in fog_integrate.comp.
struct FogIntegrateParams {
    glm::vec4 fogParams{kDefaultFogMaxDistance, 0.0f, 0.0f, 0.0f};
};

// Runtime-tunable fog settings, surfaced in the debug panel.
// FogSettings moved to renderer/RuntimeSettings.h so it could be serialized and
// clamped without dragging Vulkan into the settings layer, the same move
// SsaoSettings made. It is a `ve::` type; this header's users find it by
// enclosing-namespace lookup.

class VolumetricFogPass final {
public:
    VolumetricFogPass() = default;
    // The sampler is a raw VkSampler rather than an RAII wrapper, so unlike the
    // images and pipelines it needs an explicit destructor to avoid leaking at
    // device teardown. Renderer declares its VulkanContext before this member,
    // so the device outlives it.
    ~VolumetricFogPass();

    VolumetricFogPass(const VolumetricFogPass&) = delete;
    VolumetricFogPass& operator=(const VolumetricFogPass&) = delete;
    VolumetricFogPass(VolumetricFogPass&&) = delete;
    VolumetricFogPass& operator=(VolumetricFogPass&&) = delete;

    void create(rhi::VulkanContext& context,
                uint32_t frameCount,
                const std::filesystem::path& injectShaderPath,
                const std::filesystem::path& integrateShaderPath,
                VkImageView shadowMapView,
                VkSampler shadowMapSampler,
                VkImageView punctualShadowAtlasView,
                VkSampler punctualShadowAtlasSampler);
    void reset();

    // Rebinds the cascaded shadow map. Called when the shadow map is recreated,
    // since the descriptor sets cache its view.
    void updateShadowMap(VkImageView shadowMapView,
                         VkSampler shadowMapSampler,
                         VkImageView punctualShadowAtlasView,
                         VkSampler punctualShadowAtlasSampler);

    void updateParams(uint32_t frameIndex, const FogInjectParams& injectParams);

    // Records injection then integration, with a barrier between them: the
    // second pass reads every froxel the first wrote.
    void recordCommands(VkCommandBuffer commandBuffer,
                        uint32_t frameIndex,
                        const FogInjectPushConstants& pushConstants);

    // Clears the integrated volume to "no fog" and puts it in the layout the
    // material descriptors record. Must be called even on frames where fog is
    // disabled: binding 8 claims SHADER_READ_ONLY_OPTIMAL unconditionally, so
    // an untouched volume would sit in UNDEFINED and trip layout validation the
    // first time anything sampled the set. A no-op after the first call.
    void ensureVolumeInitialized(VkCommandBuffer commandBuffer);

    // Requests one re-clear of the volumes to their neutral state. Called when
    // fog is switched off: the skybox samples the volume unconditionally (it has
    // no push-constant room for an enable flag), so a disabled fog has to leave
    // behind a neutral volume rather than its last computed frame.
    void markVolumeNeedsClear()
    {
        volumeInitialized_ = false;
    }

    // The compute half works. False still leaves the volumes allocated.
    [[nodiscard]] bool available() const
    {
        return available_;
    }

    // The volumes exist and can be bound. True even when the compute half
    // failed, because the material descriptor set binds the integrated volume
    // at a sampler3D binding regardless of whether fog is on.
    [[nodiscard]] bool hasVolume() const
    {
        return integratedVolume_.imageView() != VK_NULL_HANDLE;
    }

    // Integrated volume, sampled by the composite pass as
    // scene * a + rgb.
    [[nodiscard]] const rhi::VulkanImage& integratedVolume() const
    {
        return integratedVolume_;
    }
    [[nodiscard]] VkSampler sampler() const
    {
        return sampler_;
    }
    [[nodiscard]] VkImageLayout* integratedVolumeLayoutPtr()
    {
        return &integratedVolumeLayout_;
    }

private:
    void createVolumes();
    void createSampler();
    void createDescriptorResources(uint32_t frameCount);
    void createPipelines(const std::filesystem::path& injectShaderPath,
                         const std::filesystem::path& integrateShaderPath);
    void writeDescriptorSets();

    rhi::VulkanContext* context_ = nullptr;
    bool available_ = false;

    // rgb = in-scattering per unit length, a = extinction. Written by injection,
    // consumed by integration. Two of them, ping-ponged: injection blends the
    // previous frame's volume, reprojected, into the one it is writing.
    std::array<rhi::VulkanImage, 2> scatterVolumes_;
    std::array<VkImageLayout, 2> scatterVolumeLayouts_{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t historyParity_ = 0;
    // rgb = accumulated in-scattering, a = transmittance. Read by composite.
    rhi::VulkanImage integratedVolume_;
    VkImageLayout integratedVolumeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampler sampler_ = VK_NULL_HANDLE;

    VkImageView shadowMapView_ = VK_NULL_HANDLE;
    VkSampler shadowMapSampler_ = VK_NULL_HANDLE;
    VkImageView punctualShadowAtlasView_ = VK_NULL_HANDLE;
    VkSampler punctualShadowAtlasSampler_ = VK_NULL_HANDLE;
    bool volumeInitialized_ = false;

    std::vector<rhi::VulkanBuffer> injectParamBuffers_;
    std::vector<rhi::VulkanBuffer> integrateParamBuffers_;

    rhi::VulkanDescriptorSetLayout injectSetLayout_;
    rhi::VulkanDescriptorSetLayout integrateSetLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    std::vector<VkDescriptorSet> injectSets_;
    std::vector<VkDescriptorSet> integrateSets_;

    rhi::VulkanComputePipeline injectPipeline_;
    rhi::VulkanComputePipeline integratePipeline_;
};

} // namespace ve::renderer
