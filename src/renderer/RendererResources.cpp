// Creation and teardown of the renderer's device-lifetime GPU objects:
// descriptor set layouts and sets, graphics and compute pipelines, the shadow
// map, the IBL chain, and the per-frame buffers.
//
// Split out of Renderer.cpp, which had grown past 6000 lines. These are all
// Renderer member functions still; only their definitions moved, so the class
// contract and every call site are unchanged. Shared file-local helpers live in
// RendererInternal.h, which is why no anonymous namespace travelled with them.
#include "renderer/Renderer.h"
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"


#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/norm.hpp>
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
#include <unordered_map>
#include <utility>
#include <vector>


namespace ve {

void Renderer::createMaterialDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
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

    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Set 0 binding 0 is the base color texture, binding 1 is the cascaded
    // shadow-map array, binding 2 is the tangent-space normal map, and binding 3 is the
    // metallic-roughness map. Binding 4 is diffuse irradiance, binding 5 is
    // prefiltered environment specular, and binding 6 is the split-sum BRDF LUT,
    // and binding 7 is the punctual (spot/point) shadow atlas. The per-slot
    // projections that go with binding 7 ride the BDA path, not a descriptor.
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
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // The integrated fog volume. The sky is infinitely far, so it samples the
    // volume's far slice and needs no depth of its own.
    //
    // There is no enable flag to go with it, unlike the main pass: the skybox
    // push constants are already two mat4s, exactly the 128-byte guaranteed
    // minimum, so there is nowhere to put one. Instead the volume is kept
    // neutral whenever fog is off (see markVolumeNeedsClear), which makes an
    // unconditional sample a no-op.
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    skyboxDescriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              skyboxDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "SkyboxDescriptorSetLayout");
}

void Renderer::createDepthPyramidDescriptorSetLayout()
{
    depthPyramid_.createDescriptorSetLayout();
}

void Renderer::destroyDepthPyramidResources()
{
    depthPyramid_.destroyResources();
}

void Renderer::invalidateDepthPyramid()
{
    depthPyramid_.invalidate();
}

void Renderer::createDepthPyramidResources()
{
    depthPyramid_.createResources();
    // The pyramid disables GPU occlusion culling when the device cannot build it;
    // afterwards the cull descriptors are (re)bound to the new pyramid image (a
    // no-op when no pyramid image was created).
    if (!depthPyramid_.buildAvailable()) {
        useGpuOcclusionCulling_ = false;
    }
    updateGpuCullingDepthPyramidDescriptors();
}

void Renderer::updateGpuCullingDepthPyramidDescriptors()
{
    gpuCulling_.updateDepthPyramidDescriptors();
}

void Renderer::createGpuCullingResources()
{
    gpuCulling_.createResources(
        static_cast<uint32_t>(frames_.size()), useGpuCulling_, useGpuShadowCulling_, isShadowIndirectActive());
}

void Renderer::destroyGpuCullingResources()
{
    gpuCulling_.destroyResources();
}

void Renderer::createShadowMap()
{
    // The CSM depth array is fixed-size for now and intentionally independent
    // of swapchain resize; only the main color/depth targets follow the window extent.
    imguiLayer_.clearRenderTargetPreviewDescriptors();
    shadowMap_.create(
        context_, shadowSettings_.resolution, shadowSettings_.resolution, activeCascadeCount(), "CascadedShadowMap");

    // The punctual atlas is fixed-size and independent of the CSM resolution
    // setting: its tiles are sized by the atlas grid, not by shadowSettings_.
    // Recreated alongside the CSM so a cascade-count change cannot leave the
    // material descriptor sets pointing at a destroyed image.
    punctualShadows_.create(context_, static_cast<uint32_t>(frames_.size()));
    // A fresh atlas image has undefined contents, which no input hash can
    // express: the key could match the previous atlas exactly while the memory
    // behind it no longer holds that render.
    invalidatePunctualShadowCache();

    // The fog injection pass samples the cascaded shadow map, so it is created
    // here alongside it -- a cascade-count change recreates the shadow map, and
    // the fog descriptors cache its view.
    // Fog samples both shadow sources: the cascades for the directional light
    // and the punctual atlas for the spot/point shafts, so it is created after
    // punctualShadows_ above. When the atlas failed to allocate, cascade 0's
    // single-layer 2D view stands in -- the binding is a sampler2D, so the CSM
    // array view would not do, and nothing samples it because every light then
    // carries the unshadowed sentinel.
    const VkImageView punctualAtlasView =
        punctualShadows_.valid() ? punctualShadows_.atlas().imageView() : shadowMap_.layerImageView(0);
    const VkSampler punctualAtlasSampler =
        punctualShadows_.valid() ? punctualShadows_.atlas().sampler() : shadowMap_.sampler();

    if (volumetricFog_.hasVolume()) {
        volumetricFog_.updateShadowMap(
            shadowMap_.imageView(), shadowMap_.sampler(), punctualAtlasView, punctualAtlasSampler);
    } else {
        volumetricFog_.create(context_,
                              static_cast<uint32_t>(frames_.size()),
                              shaderPath("fog_inject.comp.spv"),
                              shaderPath("fog_integrate.comp.spv"),
                              shadowMap_.imageView(),
                              shadowMap_.sampler(),
                              punctualAtlasView,
                              punctualAtlasSampler);
    }
}

void Renderer::createPipeline()
{
    createMainGraphicsPipeline();
    createSkinnedPipeline();
    createTransparentPipeline();
    createSkyboxPipeline();
    createShadowPipeline();
    postProcess_.createBloomPipelines();
    postProcess_.createTaaResolvePipeline();
    postProcess_.createCompositePipeline();
    ssr_.createPipeline(shaderPath("fullscreen.vert.spv"), shaderPath("ssr_trace.frag.spv"));
    gtao_.createPipeline(
        shaderPath("fullscreen.vert.spv"), shaderPath("gtao.frag.spv"), shaderPath("gtao_blur.frag.spv"));
    createComputePipelines();
}

void Renderer::createMainGraphicsPipeline()
{
    const VkVertexInputBindingDescription binding = renderer::vertexBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 5> attributes = renderer::vertexAttributeDescriptions();
    const bool bindlessMaterialTexturesActive = isBindlessMaterialTextureActive();
    std::array<VkDescriptorSetLayout, 2> mainDescriptorSetLayouts{
        materialDescriptorSetLayout_.handle(),
        bindlessTextureHeap_.descriptorSetLayout(),
    };
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};

    // The main pass renders HDR color, UV-space motion vectors, and the thin
    // G-buffer (normal/roughness/metallic) for SSR (MRT).
    const std::array<VkFormat, 3> mainPassColorFormats{kSceneColorFormat, kVelocityFormat, kNormalRoughnessFormat};

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple.vert.spv");
    pipelineInfo.fragmentShaderPath =
        bindlessMaterialTexturesActive ? shaderPath("simple_bindless.frag.spv") : shaderPath("simple.frag.spv");
    pipelineInfo.colorFormat = kSceneColorFormat;
    pipelineInfo.colorFormats = std::span<const VkFormat>(mainPassColorFormats.data(), mainPassColorFormats.size());
    pipelineInfo.depthFormat = swapchain_.depthFormat();
    pipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    pipelineInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(attributes.data(), attributes.size());
    pipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(mainDescriptorSetLayouts.data(), bindlessMaterialTexturesActive ? 2 : 1);
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.enableDepth = true;

    pipelineInfo.pipelineCache = context_.pipelineCache();
    pipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "MainGraphicsPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), pipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "MainPipelineLayout");
    pipelineColorFormat_ = pipelineInfo.colorFormat;
    pipelineDepthFormat_ = pipelineInfo.depthFormat;
}

void Renderer::createSkinnedPipeline()
{
    // The skinned demo reuses the bindless fragment shader, so it only exists when
    // bindless material textures are active. It shares the main pipeline layout
    // (descriptor sets + push constant range) so the bound sets stay compatible.
    if (!isBindlessMaterialTextureActive()) {
        return;
    }

    const std::array<VkVertexInputBindingDescription, 2> bindings = renderer::skinnedVertexBindingDescriptions();
    const std::array<VkVertexInputAttributeDescription, 7> attributes = renderer::skinnedVertexAttributeDescriptions();
    std::array<VkDescriptorSetLayout, 2> skinnedDescriptorSetLayouts{
        materialDescriptorSetLayout_.handle(),
        bindlessTextureHeap_.descriptorSetLayout(),
    };
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};

    const std::array<VkFormat, 3> mainPassColorFormats{kSceneColorFormat, kVelocityFormat, kNormalRoughnessFormat};

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple_skinned.vert.spv");
    pipelineInfo.fragmentShaderPath = shaderPath("simple_bindless.frag.spv");
    pipelineInfo.colorFormat = kSceneColorFormat;
    pipelineInfo.colorFormats = std::span<const VkFormat>(mainPassColorFormats.data(), mainPassColorFormats.size());
    pipelineInfo.depthFormat = swapchain_.depthFormat();
    pipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(bindings.data(), bindings.size());
    pipelineInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(attributes.data(), attributes.size());
    pipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(skinnedDescriptorSetLayouts.data(), skinnedDescriptorSetLayouts.size());
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.enableDepth = true;
    pipelineInfo.cullMode = VK_CULL_MODE_NONE;

    pipelineInfo.pipelineCache = context_.pipelineCache();
    skinnedPipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), skinnedPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "SkinnedGraphicsPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), skinnedPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "SkinnedPipelineLayout");
}

void Renderer::createTransparentPipeline()
{
    // Same shaders and descriptor contract as the main pass; only the blend,
    // depth-write, and attachment state differ. Reusing simple_bindless.frag also
    // means BLEND materials get the identical PBR/IBL/clustered lighting path
    // rather than a separate, drifting shader.
    const VkVertexInputBindingDescription binding = renderer::vertexBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 5> attributes = renderer::vertexAttributeDescriptions();
    const bool bindlessMaterialTexturesActive = isBindlessMaterialTextureActive();
    if (!bindlessMaterialTexturesActive) {
        // The transparent pass only has a bindless recording path.
        transparentPipeline_.reset();
        return;
    }

    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{
        materialDescriptorSetLayout_.handle(),
        bindlessTextureHeap_.descriptorSetLayout(),
    };
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};

    // Same MRT set as the main pass. The shared fragment shader declares all three
    // outputs, so binding fewer would leave declared outputs without attachments;
    // more importantly, letting transparents write velocity is what gives them
    // correct TAA reprojection instead of inheriting the background's motion.
    // Only attachment 0 blends (see enableAlphaBlend).
    const std::array<VkFormat, 3> transparentColorFormats{kSceneColorFormat, kVelocityFormat, kNormalRoughnessFormat};

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple.vert.spv");
    pipelineInfo.fragmentShaderPath = shaderPath("simple_bindless.frag.spv");
    pipelineInfo.colorFormat = kSceneColorFormat;
    pipelineInfo.colorFormats =
        std::span<const VkFormat>(transparentColorFormats.data(), transparentColorFormats.size());
    pipelineInfo.depthFormat = swapchain_.depthFormat();
    pipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    pipelineInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(attributes.data(), attributes.size());
    pipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(descriptorSetLayouts.data(), descriptorSetLayouts.size());
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.enableAlphaBlend = true;
    pipelineInfo.independentBlendAvailable = context_.device().independentBlendEnabled();
    pipelineInfo.enableDepth = true;
    // Test against opaque depth, but do not write: transparents must not occlude
    // each other, that is what the back-to-front sort is for.
    pipelineInfo.depthWriteEnable = false;
    // Blended surfaces are routinely seen from both sides (glass, foliage cards).
    pipelineInfo.cullMode = VK_CULL_MODE_NONE;

    pipelineInfo.pipelineCache = context_.pipelineCache();
    transparentPipeline_.create(context_.vkDevice(), pipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), transparentPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "TransparentPipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              transparentPipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "TransparentPipelineLayout");
}

void Renderer::createSkyboxPipeline()
{
    const VkDescriptorSetLayout skyboxDescriptorSetLayout = skyboxDescriptorSetLayout_.handle();
    const VkPushConstantRange skyboxPushConstantRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      0,
                                                      static_cast<uint32_t>(sizeof(SkyboxPushConstants))};

    const std::array<VkFormat, 3> mainPassColorFormats{kSceneColorFormat, kVelocityFormat, kNormalRoughnessFormat};

    rhi::VulkanPipelineCreateInfo skyboxPipelineInfo{};
    skyboxPipelineInfo.vertexShaderPath = shaderPath("skybox.vert.spv");
    skyboxPipelineInfo.fragmentShaderPath = shaderPath("skybox.frag.spv");
    skyboxPipelineInfo.colorFormat = kSceneColorFormat;
    skyboxPipelineInfo.colorFormats = std::span<const VkFormat>(mainPassColorFormats.data(), mainPassColorFormats.size());
    skyboxPipelineInfo.depthFormat = swapchain_.depthFormat();
    skyboxPipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&skyboxDescriptorSetLayout, 1);
    skyboxPipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&skyboxPushConstantRange, 1);
    skyboxPipelineInfo.enableDepth = true;
    skyboxPipelineInfo.depthWriteEnable = false;
    skyboxPipelineInfo.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    skyboxPipelineInfo.cullMode = VK_CULL_MODE_NONE;

    skyboxPipelineInfo.pipelineCache = context_.pipelineCache();
    skyboxPipeline_.create(context_.vkDevice(), skyboxPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), skyboxPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "SkyboxPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), skyboxPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "SkyboxPipelineLayout");
    skyboxPipelineColorFormat_ = skyboxPipelineInfo.colorFormat;
    skyboxPipelineDepthFormat_ = skyboxPipelineInfo.depthFormat;
}

void Renderer::createShadowPipeline()
{
    const VkVertexInputBindingDescription binding = renderer::vertexBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 5> attributes = renderer::vertexAttributeDescriptions();

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

    shadowPipelineInfo.pipelineCache = context_.pipelineCache();
    shadowPipeline_.create(context_.vkDevice(), shadowPipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), shadowPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "ShadowPipeline");
    rhi::debug::setObjectName(
        context_.vkDevice(), shadowPipeline_.layout(), VK_OBJECT_TYPE_PIPELINE_LAYOUT, "ShadowPipelineLayout");
    shadowPipelineDepthFormat_ = shadowPipelineInfo.depthFormat;

    createMaskedShadowPipeline(binding, attributes);
    createPunctualShadowPipeline(binding, attributes);
}

void Renderer::createPunctualShadowPipeline(const VkVertexInputBindingDescription& binding,
                                            const std::array<VkVertexInputAttributeDescription, 5>& attributes)
{
    if (!punctualShadows_.valid()) {
        punctualShadowPipeline_.reset();
        return;
    }

    rhi::VulkanPipelineCreateInfo info{};
    info.vertexShaderPath = shaderPath("shadow_punctual.vert.spv");
    info.depthFormat = punctualShadows_.atlas().format();
    info.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    // Depth-only, so position (location 0) is the only attribute consumed.
    info.vertexAttributes = std::span<const VkVertexInputAttributeDescription>(attributes.data(), 1);
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(sizeof(PunctualShadowPushConstants))};
    info.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    info.enableColorAttachment = false;
    info.enableDepth = true;
    info.depthWriteEnable = true;
    // Same acne/peter-panning tradeoff as the CSM pipeline, and the same
    // settings drive it so tuning one shadow type does not desync the other.
    info.enableDepthBias = true;
    info.cullMode = VK_CULL_MODE_NONE;
    info.depthBiasConstantFactor = shadowSettings_.rasterDepthBiasConstantFactor;
    info.depthBiasSlopeFactor = shadowSettings_.rasterDepthBiasSlopeFactor;
    info.pipelineCache = context_.pipelineCache();

    punctualShadowPipeline_.create(context_.vkDevice(), info);
    rhi::debug::setObjectName(context_.vkDevice(),
                              punctualShadowPipeline_.pipeline(),
                              VK_OBJECT_TYPE_PIPELINE,
                              "PunctualShadowPipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              punctualShadowPipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "PunctualShadowPipelineLayout");
    punctualShadowPipelineDepthFormat_ = info.depthFormat;
}

void Renderer::createMaskedShadowPipeline(const VkVertexInputBindingDescription& binding,
                                          const std::array<VkVertexInputAttributeDescription, 5>& attributes)
{
    // Cutout shadow casters need the bindless base-color array to run the alpha
    // test, so this variant only exists when the bindless heap does. Without it
    // MASK casters fall back to shadowPipeline_ and throw a solid silhouette.
    if (!isBindlessMaterialTextureActive() || bindlessTextureHeap_.descriptorSetLayout() == VK_NULL_HANDLE) {
        maskedShadowPipeline_.reset();
        return;
    }

    // Position (location 0) drives the depth write; UV (location 2) feeds the
    // cutout sample. The intermediate attributes are skipped, so this cannot be a
    // prefix subspan of the shared attribute list.
    const std::array<VkVertexInputAttributeDescription, 2> maskedAttributes{attributes[0], attributes[2]};
    const VkDescriptorSetLayout bindlessLayout = bindlessTextureHeap_.descriptorSetLayout();
    const VkPushConstantRange maskedPushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};

    rhi::VulkanPipelineCreateInfo maskedInfo{};
    maskedInfo.vertexShaderPath = shaderPath("shadow_masked.vert.spv");
    maskedInfo.fragmentShaderPath = shaderPath("shadow_masked.frag.spv");
    maskedInfo.depthFormat = shadowMap_.format();
    maskedInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    maskedInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(maskedAttributes.data(), maskedAttributes.size());
    maskedInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&bindlessLayout, 1);
    maskedInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&maskedPushConstantRange, 1);
    maskedInfo.enableColorAttachment = false;
    maskedInfo.enableDepth = true;
    maskedInfo.depthWriteEnable = true;
    maskedInfo.enableDepthBias = true;
    maskedInfo.cullMode = VK_CULL_MODE_NONE;
    maskedInfo.depthBiasConstantFactor = shadowSettings_.rasterDepthBiasConstantFactor;
    maskedInfo.depthBiasSlopeFactor = shadowSettings_.rasterDepthBiasSlopeFactor;
    maskedInfo.pipelineCache = context_.pipelineCache();

    maskedShadowPipeline_.create(context_.vkDevice(), maskedInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), maskedShadowPipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "MaskedShadowPipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              maskedShadowPipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "MaskedShadowPipelineLayout");
}

void Renderer::createComputePipelines()
{
    depthPyramid_.createPipeline();

    // Exposure compute pipelines (luminance/histogram/reduce) now live in PostProcessStack.
    postProcess_.createExposureComputePipelines();
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
    poolSize.descriptorCount = kMaxMaterialDescriptorSets * 9;

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

    // When the punctual atlas failed to allocate, bind cascade 0's single-layer
    // view in its place so the descriptor stays complete and type-correct -- the
    // shader declares binding 7 as sampler2D, so the CSM's 2D_ARRAY view would
    // not do. Nothing ever samples it: without an atlas no light is handed a
    // slot, so every light decodes to the unshadowed sentinel and the shader
    // skips the fetch entirely.
    // Without fog the descriptor still has to be complete and type-correct
    // (binding 8 is a sampler3D), so it binds the fog volume whenever the
    // subsystem allocated one. Nothing samples it: the shader gates on a
    // non-zero fog max distance, which stays zero while fog is off.
    if (!volumetricFog_.hasVolume()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a fog volume for binding 8.");
    }
    VkDescriptorImageInfo fogVolumeInfo{};
    fogVolumeInfo.sampler = volumetricFog_.sampler();
    fogVolumeInfo.imageView = volumetricFog_.integratedVolume().imageView();
    fogVolumeInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const bool punctualAtlasAvailable = punctualShadows_.valid();
    VkDescriptorImageInfo punctualShadowInfo{};
    punctualShadowInfo.sampler =
        punctualAtlasAvailable ? punctualShadows_.atlas().sampler() : shadowMap_.sampler();
    punctualShadowInfo.imageView =
        punctualAtlasAvailable ? punctualShadows_.atlas().imageView() : shadowMap_.layerImageView(0);
    punctualShadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 9> writes{};
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

    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = material.descriptorSet;
    writes[7].dstBinding = 7;
    writes[7].dstArrayElement = 0;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &punctualShadowInfo;

    writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[8].dstSet = material.descriptorSet;
    writes[8].dstBinding = 8;
    writes[8].dstArrayElement = 0;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &fogVolumeInfo;

    // The material descriptor stores sampled images only: base color at binding 0,
    // cascaded shadow-map array at binding 1, normal map at binding 2, and metallic-roughness
    // map at binding 3. Bindings 4-6 are diffuse irradiance, prefiltered specular
    // environment, and the BRDF LUT; binding 7 is the punctual shadow atlas.
    // Object data remains outside descriptors.
    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::createSkyboxDescriptorSet()
{
    if (!environmentMap_.valid()) {
        throw std::runtime_error("Cannot create a skybox descriptor set without a valid environment map.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 2;

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

    VkDescriptorImageInfo fogVolumeInfo{};
    fogVolumeInfo.sampler = volumetricFog_.sampler();
    fogVolumeInfo.imageView = volumetricFog_.integratedVolume().imageView();
    fogVolumeInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = skyboxDescriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &environmentInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = skyboxDescriptorSet_;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &fogVolumeInfo;

    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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

} // namespace ve
