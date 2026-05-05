#include "renderer/Renderer.h"

#include "core/Logger.h"
#include "core/Window.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ve {

namespace {

struct ObjectFrameData {
    glm::mat4 mvp{1.0f};
    glm::mat4 model{1.0f};
    glm::mat4 lightMvp{1.0f};
    glm::vec4 lightDirection{0.35f, -0.65f, -0.55f, 0.0f};
    glm::vec4 lightColor{0.85f, 0.85f, 0.85f, 1.0f};
    glm::vec4 ambientColor{0.15f, 0.15f, 0.15f, 1.0f};
    glm::vec4 shadowSettings{0.002f, 0.005f, 1.0f, 1.0f};
    glm::vec4 baseColorFactor{1.0f};
    glm::vec4 materialParams{0.0f, 0.5f, 1.0f, 0.0f};
    glm::vec4 cameraPosition{0.0f, 0.0f, 0.0f, 1.0f};
};

// Mirrors the shader's std430 buffer_reference block. std430 stores mat4 as
// four 16-byte columns and vec4 as 16 bytes, so this 304-byte stride keeps each
// field and each per-object BDA entry on a 16-byte boundary.
// materialParams.x = metallic, y = roughness, z = multiScatterStrength,
// and w is reserved for future scalar material data.
static_assert(offsetof(ObjectFrameData, mvp) == 0);
static_assert(offsetof(ObjectFrameData, model) == 64);
static_assert(offsetof(ObjectFrameData, lightMvp) == 128);
static_assert(offsetof(ObjectFrameData, lightDirection) == 192);
static_assert(offsetof(ObjectFrameData, lightColor) == 208);
static_assert(offsetof(ObjectFrameData, ambientColor) == 224);
static_assert(offsetof(ObjectFrameData, shadowSettings) == 240);
static_assert(offsetof(ObjectFrameData, baseColorFactor) == 256);
static_assert(offsetof(ObjectFrameData, materialParams) == 272);
static_assert(offsetof(ObjectFrameData, cameraPosition) == 288);
static_assert(sizeof(ObjectFrameData) == 304);

constexpr uint32_t kMaxFrameObjects = 64;
constexpr uint32_t kMaxMaterialDescriptorSets = 8;

const glm::vec4 kDirectionalLightDirection{0.35f, -0.65f, -0.55f, 0.0f};
const glm::vec4 kDirectionalLightColor{0.85f, 0.85f, 0.85f, 1.0f};
const glm::vec4 kAmbientLightColor{0.15f, 0.15f, 0.15f, 1.0f};

struct ShadowSceneBounds {
    glm::vec3 center{0.0f};
    float radius = 1.0f;
    float lightDistance = 8.0f;
    float nearPlane = 0.1f;
    float farPlane = 12.0f;
};

struct PushConstants {
    VkDeviceAddress objectFrameDataAddress = 0;
};

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

ShadowSceneBounds fixedCubeSceneShadowBounds()
{
    // This fixed sphere covers the current four-cube demo through their rotations.
    // It is stable because it does not chase the camera, but it is not a substitute
    // for cascaded shadow maps or texel snapping once the scene grows.
    const glm::vec3 sceneCenter{0.0f, 0.25f, -0.3f};
    constexpr float sceneRadius = 3.4f;
    constexpr float lightDistance = 8.4f;
    constexpr float farPadding = 1.0f;

    return {sceneCenter, sceneRadius, lightDistance, 0.1f,
            lightDistance + sceneRadius + farPadding};
}

glm::mat4 directionalLightViewProjection()
{
    const glm::vec3 lightDirection = glm::normalize(glm::vec3{
        kDirectionalLightDirection.x, kDirectionalLightDirection.y, kDirectionalLightDirection.z});
    const ShadowSceneBounds sceneBounds = fixedCubeSceneShadowBounds();
    const glm::vec3 lightPosition = sceneBounds.center - lightDirection * sceneBounds.lightDistance;
    const glm::vec3 up = std::abs(glm::dot(lightDirection, glm::vec3{0.0f, 1.0f, 0.0f})) > 0.95f
                             ? glm::vec3{0.0f, 0.0f, 1.0f}
                             : glm::vec3{0.0f, 1.0f, 0.0f};

    // The orthographic extent and near/far planes come from a fixed demo-scene
    // bound. A tighter bound gives the 2048 shadow map more useful texel density.
    glm::mat4 lightProjection =
        glm::ortho(-sceneBounds.radius, sceneBounds.radius, -sceneBounds.radius, sceneBounds.radius,
                   sceneBounds.nearPlane, sceneBounds.farPlane);
    lightProjection[1][1] *= -1.0f;
    return lightProjection * glm::lookAt(lightPosition, sceneBounds.center, up);
}

} // namespace

Renderer::Renderer(Window& window) : window_(window)
{
    context_.initialize(window_);

    frames_.resize(rhi::kMaxFramesInFlight);
    swapchain_.initialize(context_, window_.framebufferExtent());
    createMaterialDescriptorSetLayout();
    createSkyboxDescriptorSetLayout();
    createShadowMap();
    createPipeline();
    commandContext_.initialize(context_, frames_);
    createScene();
    createObjectFrameDataBuffers();
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

    uint32_t imageIndex = 0;
    const VkResult acquireResult =
        vkAcquireNextImageKHR(context_.vkDevice(), swapchain_.handle(), UINT64_MAX,
                              frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error(std::string("vkAcquireNextImageKHR failed: ") +
                                 rhi::vkResultToString(acquireResult));
    }

    if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
        VK_CHECK(vkWaitForFences(context_.vkDevice(), 1, &imagesInFlight_[imageIndex], VK_TRUE,
                                 UINT64_MAX));
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

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    const VkSwapchainKHR swapchain = swapchain_.handle();
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(context_.presentQueue(), &presentInfo);
    const bool needsRecreate = presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
                               presentResult == VK_SUBOPTIMAL_KHR ||
                               acquireResult == VK_SUBOPTIMAL_KHR || window_.wasResized();

    if (presentResult != VK_SUCCESS && presentResult != VK_ERROR_OUT_OF_DATE_KHR &&
        presentResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error(std::string("vkQueuePresentKHR failed: ") +
                                 rhi::vkResultToString(presentResult));
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

    // Set 0 binding 0 is the base color texture, binding 1 is the shadow map,
    // binding 2 is the tangent-space normal map, and binding 3 is the
    // metallic-roughness map. Binding 4 is diffuse irradiance, binding 5 is
    // prefiltered environment specular, and binding 6 is the split-sum BRDF LUT.
    // Object MVP/model/light/material data stays on the BDA + vertex push constant path.
    materialDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
}

void Renderer::createSkyboxDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    skyboxDescriptorSetLayout_.create(context_.vkDevice(),
                                      std::span<const VkDescriptorSetLayoutBinding>(&binding, 1));
}

void Renderer::createShadowMap()
{
    // The directional shadow map is fixed-size for now and intentionally independent
    // of swapchain resize; only the main color/depth targets follow the window extent.
    shadowMap_.create(context_, shadowSettings_.resolution, shadowSettings_.resolution);
}

void Renderer::createPipeline()
{
    const VkVertexInputBindingDescription binding = renderer::vertexBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 5> attributes =
        renderer::vertexAttributeDescriptions();
    const VkDescriptorSetLayout descriptorSetLayout = materialDescriptorSetLayout_.handle();
    const VkDescriptorSetLayout skyboxDescriptorSetLayout = skyboxDescriptorSetLayout_.handle();
    const VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_VERTEX_BIT, 0,
                                                static_cast<uint32_t>(sizeof(PushConstants))};
    const VkPushConstantRange skyboxPushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(sizeof(SkyboxPushConstants))};

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple.vert.spv");
    pipelineInfo.fragmentShaderPath = shaderPath("simple.frag.spv");
    pipelineInfo.colorFormat = swapchain_.colorFormat();
    pipelineInfo.depthFormat = swapchain_.depthFormat();
    pipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    pipelineInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(attributes.data(), attributes.size());
    pipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&descriptorSetLayout, 1);
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.enableDepth = true;

    pipeline_.create(context_.vkDevice(), pipelineInfo);
    pipelineColorFormat_ = pipelineInfo.colorFormat;
    pipelineDepthFormat_ = pipelineInfo.depthFormat;

    rhi::VulkanPipelineCreateInfo skyboxPipelineInfo{};
    skyboxPipelineInfo.vertexShaderPath = shaderPath("skybox.vert.spv");
    skyboxPipelineInfo.fragmentShaderPath = shaderPath("skybox.frag.spv");
    skyboxPipelineInfo.colorFormat = swapchain_.colorFormat();
    skyboxPipelineInfo.depthFormat = swapchain_.depthFormat();
    skyboxPipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&skyboxDescriptorSetLayout, 1);
    skyboxPipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&skyboxPushConstantRange, 1);
    skyboxPipelineInfo.enableDepth = true;
    skyboxPipelineInfo.depthWriteEnable = false;
    skyboxPipelineInfo.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    skyboxPipelineInfo.cullMode = VK_CULL_MODE_NONE;

    skyboxPipeline_.create(context_.vkDevice(), skyboxPipelineInfo);
    skyboxPipelineColorFormat_ = skyboxPipelineInfo.colorFormat;
    skyboxPipelineDepthFormat_ = skyboxPipelineInfo.depthFormat;

    rhi::VulkanPipelineCreateInfo shadowPipelineInfo{};
    shadowPipelineInfo.vertexShaderPath = shaderPath("shadow.vert.spv");
    shadowPipelineInfo.depthFormat = shadowMap_.format();
    shadowPipelineInfo.vertexBindings =
        std::span<const VkVertexInputBindingDescription>(&binding, 1);
    shadowPipelineInfo.vertexAttributes =
        std::span<const VkVertexInputAttributeDescription>(attributes.data(), 1);
    shadowPipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&pushConstantRange, 1);
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
    shadowPipelineDepthFormat_ = shadowPipelineInfo.depthFormat;
}

void Renderer::createScene()
{
    renderObjects_.clear();

    cubeMesh_ = renderer::Mesh::createCube(context_, commandContext_);
    createCheckerboardTexture();
    createNormalTexture();
    createMetallicRoughnessTexture();
    createEnvironmentMap();
    createMaterial();

    camera_.position = {0.0f, 0.35f, 5.5f};
    camera_.target = {0.0f, 0.1f, 0.0f};

    renderObjects_.reserve(4);
    const auto addCube = [this](std::string debugName, const renderer::Material* material,
                                const glm::vec3& position, const glm::vec3& rotationRadians,
                                const glm::vec3& scale) {
        renderer::RenderObject cube{};
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        renderObjects_.push_back(std::move(cube));
    };

    addCube("Center Cube", &materialVariants_.at(0), {0.0f, -0.1f, 0.0f}, {0.2f, 0.0f, 0.0f},
            {0.7f, 0.7f, 0.7f});
    addCube("Left Cube", &materialVariants_.at(1), {-1.35f, -0.15f, -0.35f}, {0.0f, 0.35f, 0.2f},
            {0.5f, 0.5f, 0.5f});
    addCube("Right Cube", &materialVariants_.at(2), {1.35f, -0.05f, -0.25f}, {0.25f, -0.35f, 0.0f},
            {0.55f, 0.8f, 0.55f});
    addCube("Elevated Cube", &materialVariants_.at(3), {0.0f, 1.0f, -0.7f}, {-0.3f, 0.2f, 0.45f},
            {0.45f, 0.45f, 0.45f});
}

void Renderer::createCheckerboardTexture()
{
    const std::filesystem::path texturePath = assetPath("textures/checker.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            checkerboardTexture_.createFromFile(context_, commandContext_, texturePath, true);
            Logger::info("Loaded texture: " + texturePath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Texture asset missing, using procedural checkerboard fallback: " +
                     texturePath.string());
    }

    checkerboardTexture_.createCheckerboard(context_, commandContext_, 256, 256);
}

void Renderer::createNormalTexture()
{
    normalMapAssetLoaded_ = false;

    const std::filesystem::path texturePath = assetPath("textures/checker_normal.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            normalMapTexture_.createFromFile(context_, commandContext_, texturePath, true);
            normalMapAssetLoaded_ = true;
            Logger::info("Loaded normal texture: " + texturePath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load normal texture '" + texturePath.string() +
                         "': " + error.what());
        }
    } else {
        Logger::warn("Normal texture asset missing, using procedural flat normal fallback: " +
                     texturePath.string());
    }

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 128;
        pixels[offset + 1] = 128;
        pixels[offset + 2] = 255;
        pixels[offset + 3] = 255;
    }

    normalMapTexture_.createFromRgba8(context_, commandContext_, width, height,
                                      std::span<const uint8_t>(pixels.data(), pixels.size()),
                                      VK_FORMAT_R8G8B8A8_UNORM, false);
}

void Renderer::createMetallicRoughnessTexture()
{
    metallicRoughnessMapAssetLoaded_ = false;

    const std::filesystem::path texturePath = assetPath("textures/checker_mr.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            metallicRoughnessTexture_.createFromFile(context_, commandContext_, texturePath, true);
            metallicRoughnessMapAssetLoaded_ = true;
            Logger::info("Loaded metallic-roughness texture: " + texturePath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load metallic-roughness texture '" + texturePath.string() +
                         "': " + error.what());
        }
    } else {
        Logger::warn(
            "Metallic-roughness texture asset missing, using procedural neutral fallback: " +
            texturePath.string());
    }

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 255;
        pixels[offset + 1] = 255;
        pixels[offset + 2] = 0;
        pixels[offset + 3] = 255;
    }

    metallicRoughnessTexture_.createFromRgba8(
        context_, commandContext_, width, height,
        std::span<const uint8_t>(pixels.data(), pixels.size()), VK_FORMAT_R8G8B8A8_UNORM, false);
}

void Renderer::createEnvironmentMap()
{
    // The visible environment cubemap stays dedicated to the skybox. A separate
    // low-frequency cubemap feeds diffuse IBL, while a mipmapped cubemap and 2D
    // BRDF LUT feed split-sum specular IBL.
    environmentMap_.createProcedural(context_, commandContext_, 32);
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

    diffuseIrradianceMap_.createFromRgba8Faces(
        context_, commandContext_, 1,
        std::span<const uint8_t>(neutralPixels.data(), neutralPixels.size()),
        VK_FORMAT_R8G8B8A8_UNORM);
}

void Renderer::createPrefilteredEnvironmentMap()
{
    try {
        prefilteredEnvironmentMap_.createProceduralPrefilteredSpecular(context_, commandContext_,
                                                                       64);
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
        context_, commandContext_, 1,
        std::span<const uint8_t>(neutralPixels.data(), neutralPixels.size()),
        VK_FORMAT_R8G8B8A8_UNORM);
}

void Renderer::createBrdfLutTexture()
{
    brdfLutTexture_.create(context_, commandContext_, 256);
}

void Renderer::createMaterial()
{
    materialVariants_.clear();
    materialVariants_.reserve(4);

    const auto addMaterial = [this](std::string debugName, const glm::vec4& baseColorFactor,
                                    float metallic, float roughness, float multiScatterStrength) {
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
        createMaterialDescriptorSet(material);
        materialVariants_.push_back(std::move(material));
    };

    addMaterial("Checkerboard Matte", {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.75f, 0.0f);
    addMaterial("Checkerboard Warm Semi-Metal", {1.0f, 0.82f, 0.65f, 1.0f}, 0.35f, 0.38f, 0.5f);
    addMaterial("Checkerboard Cool Rough Metal", {0.72f, 0.84f, 1.0f, 1.0f}, 0.85f, 0.62f, 1.0f);
    addMaterial("Checkerboard Glossy Dielectric", {0.9f, 1.0f, 0.78f, 1.0f}, 0.0f, 0.18f, 0.25f);

    checkerboardMaterial_ = materialVariants_.front();
}

void Renderer::createMaterialDescriptorSet(renderer::Material& material)
{
    if (!material.baseColorTexture || !material.baseColorTexture->valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid base color texture.");
    }
    if (!material.normalTexture || !material.normalTexture->valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid normal texture.");
    }
    if (!material.metallicRoughnessTexture || !material.metallicRoughnessTexture->valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid metallic-roughness texture.");
    }
    if (!shadowMap_.valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid shadow map.");
    }
    if (!diffuseIrradianceMap_.valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid diffuse irradiance map.");
    }
    if (!prefilteredEnvironmentMap_.valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid prefiltered environment map.");
    }
    if (!brdfLutTexture_.valid()) {
        throw std::runtime_error(
            "Cannot create a material descriptor set without a valid BRDF LUT texture.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = kMaxMaterialDescriptorSets * 7;

    if (materialDescriptorPool_.handle() == VK_NULL_HANDLE) {
        materialDescriptorPool_.create(context_.vkDevice(),
                                       std::span<const VkDescriptorPoolSize>(&poolSize, 1),
                                       kMaxMaterialDescriptorSets);
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
    // shadow map at binding 1, normal map at binding 2, and metallic-roughness
    // map at binding 3. Bindings 4-6 are diffuse irradiance, prefiltered specular
    // environment, and the BRDF LUT. Object data remains outside descriptors.
    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(),
                           0, nullptr);
}

void Renderer::createSkyboxDescriptorSet()
{
    if (!environmentMap_.valid()) {
        throw std::runtime_error(
            "Cannot create a skybox descriptor set without a valid environment map.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    skyboxDescriptorSet_ = VK_NULL_HANDLE;
    skyboxDescriptorPool_.create(context_.vkDevice(),
                                 std::span<const VkDescriptorPoolSize>(&poolSize, 1), 1);

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

    for (rhi::VulkanBuffer& frameObjectDataBuffer : frameObjectDataBuffers_) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxFrameObjects * sizeof(ObjectFrameData));
        bufferInfo.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        bufferInfo.requestDeviceAddress = true;
        frameObjectDataBuffer.createBuffer(context_, bufferInfo);
    }
}

void Renderer::updateFrameData(uint32_t frameIndex)
{
    const auto now = std::chrono::steady_clock::now();
    const float elapsedSeconds = std::chrono::duration<float>(now - startTime_).count();

    if (renderObjects_.empty()) {
        return;
    }

    const size_t objectCount =
        std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
    std::vector<ObjectFrameData> objectFrameData(objectCount);

    const VkExtent2D extent = swapchain_.extent();
    const float aspect = extent.height == 0
                             ? 1.0f
                             : static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 view = camera_.viewMatrix();
    const glm::mat4 projection = camera_.projectionMatrix(aspect);
    const glm::mat4 lightViewProjection = directionalLightViewProjection();

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        renderer::RenderObject& object = renderObjects_[objectIndex];

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
            object.transform.rotationRadians = {elapsedSeconds * 0.35f, elapsedSeconds * 0.55f,
                                                0.45f};
            break;
        default:
            object.transform.rotationRadians = {
                elapsedSeconds * (0.2f + 0.05f * static_cast<float>(objectIndex)),
                elapsedSeconds * 0.4f, elapsedSeconds * 0.3f};
            break;
        }

        const glm::mat4 model = object.transform.modelMatrix();
        ObjectFrameData& frameData = objectFrameData[objectIndex];
        frameData.mvp = projection * view * model;
        frameData.model = model;
        frameData.lightMvp = lightViewProjection * model;
        frameData.lightDirection = kDirectionalLightDirection;
        frameData.lightColor = kDirectionalLightColor;
        frameData.ambientColor = kAmbientLightColor;
        frameData.shadowSettings = {shadowSettings_.constantBias, shadowSettings_.slopeBias,
                                    shadowSettings_.enablePcf ? 1.0f : 0.0f,
                                    static_cast<float>(std::max(shadowSettings_.pcfRadius, 0))};
        if (object.material) {
            frameData.baseColorFactor = object.material->baseColorFactor;
            frameData.materialParams = {object.material->metallic, object.material->roughness,
                                        object.material->multiScatterStrength, 0.0f};
        }
        frameData.cameraPosition = glm::vec4(camera_.position, 1.0f);
    }

    frameObjectDataBuffers_.at(frameIndex)
        .upload(std::as_bytes(
            std::span<const ObjectFrameData>(objectFrameData.data(), objectFrameData.size())));
}

void Renderer::recreateSwapchain()
{
    if (window_.isMinimized()) {
        return;
    }

    context_.waitIdle();
    swapchain_.recreate(context_, window_.framebufferExtent());
    sync_.recreateRenderFinishedSemaphores(swapchain_.imageCount());

    const bool pipelineNeedsRecreate = pipeline_.pipeline() == VK_NULL_HANDLE ||
                                       pipelineColorFormat_ != swapchain_.colorFormat() ||
                                       pipelineDepthFormat_ != swapchain_.depthFormat() ||
                                       skyboxPipeline_.pipeline() == VK_NULL_HANDLE ||
                                       skyboxPipelineColorFormat_ != swapchain_.colorFormat() ||
                                       skyboxPipelineDepthFormat_ != swapchain_.depthFormat() ||
                                       shadowPipelineDepthFormat_ != shadowMap_.format();
    if (pipelineNeedsRecreate) {
        createPipeline();
    }

    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
}

void Renderer::recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    const VkImage swapchainImage = swapchain_.image(imageIndex);
    const VkDeviceAddress objectFrameDataBaseAddress =
        frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t objectCount =
        std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));

    // Shadow map layout transition: previous shader reads must finish before this
    // frame clears and writes the depth attachment.
    transitionShadowMapImage(commandBuffer, shadowMap_.layout(),
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    shadowMap_.setLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkClearValue shadowDepthClear{};
    shadowDepthClear.depthStencil.depth = 1.0f;
    shadowDepthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo shadowDepthAttachment{};
    shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    shadowDepthAttachment.imageView = shadowMap_.imageView();
    shadowDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    shadowDepthAttachment.clearValue = shadowDepthClear;

    // Depth-only Dynamic Rendering writes the scene from the directional light.
    VkRenderingInfo shadowRenderingInfo{};
    shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    shadowRenderingInfo.renderArea.offset = {0, 0};
    shadowRenderingInfo.renderArea.extent = shadowMap_.extent();
    shadowRenderingInfo.layerCount = 1;
    shadowRenderingInfo.colorAttachmentCount = 0;
    shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

    vkCmdBeginRendering(commandBuffer, &shadowRenderingInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_.pipeline());

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

    vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.mesh) {
            continue;
        }

        const PushConstants pushConstants{
            objectFrameDataBaseAddress +
            static_cast<VkDeviceAddress>(objectIndex * sizeof(ObjectFrameData))};

        vkCmdPushConstants(commandBuffer, shadowPipeline_.layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                           static_cast<uint32_t>(sizeof(PushConstants)), &pushConstants);

        const VkBuffer vertexBuffers[] = {object.mesh->vertexBuffer()};
        const VkDeviceSize vertexOffsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, object.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT16);

        vkCmdDrawIndexed(commandBuffer, object.mesh->indexCount(), 1, 0, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);

    // The main pass samples this depth image in the fragment shader at set 0 binding 1.
    transitionShadowMapImage(commandBuffer, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    shadowMap_.setLayout(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

    // Synchronization2 barrier: make the acquired present image writable as a color attachment.
    transitionSwapchainImage(commandBuffer, swapchainImage, swapchain_.imageLayout(imageIndex),
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    swapchain_.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    transitionDepthImage(commandBuffer);

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.03f;
    clearColor.color.float32[1] = 0.04f;
    clearColor.color.float32[2] = 0.07f;
    clearColor.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchain_.imageView(imageIndex);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearColor;

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = swapchain_.depthImageView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = swapchain_.extent();
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    // Main Dynamic Rendering draws the skybox first, then consumes the shadow map
    // through the mesh material descriptor set.
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

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

    if (skyboxDescriptorSet_ != VK_NULL_HANDLE) {
        const float aspect = extent.height == 0 ? 1.0f
                                                : static_cast<float>(extent.width) /
                                                      static_cast<float>(extent.height);
        glm::mat4 skyboxView = camera_.viewMatrix();
        skyboxView[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::mat4 projection = camera_.projectionMatrix(aspect);
        const SkyboxPushConstants skyboxPushConstants{glm::inverse(projection * skyboxView)};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          skyboxPipeline_.pipeline());
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skyboxPipeline_.layout(), 0, 1, &skyboxDescriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, skyboxPipeline_.layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                           static_cast<uint32_t>(sizeof(SkyboxPushConstants)),
                           &skyboxPushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.mesh || !object.material || object.material->descriptorSet == VK_NULL_HANDLE) {
            continue;
        }

        const VkDescriptorSet descriptorSet = object.material->descriptorSet;
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.layout(),
                                0, 1, &descriptorSet, 0, nullptr);

        const PushConstants pushConstants{
            objectFrameDataBaseAddress +
            static_cast<VkDeviceAddress>(objectIndex * sizeof(ObjectFrameData))};

        // The material descriptor binds the texture/sampler for the fragment shader.
        // The pushed address points at this object's BDA frame data for the vertex shader.
        vkCmdPushConstants(commandBuffer, pipeline_.layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                           static_cast<uint32_t>(sizeof(PushConstants)), &pushConstants);

        const VkBuffer vertexBuffers[] = {object.mesh->vertexBuffer()};
        const VkDeviceSize vertexOffsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, object.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT16);

        vkCmdDrawIndexed(commandBuffer, object.mesh->indexCount(), 1, 0, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);

    // Synchronization2 barrier: presentation reads from images in PRESENT_SRC_KHR layout.
    transitionSwapchainImage(commandBuffer, swapchainImage,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    swapchain_.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void Renderer::transitionShadowMapImage(VkCommandBuffer commandBuffer, VkImageLayout oldLayout,
                                        VkImageLayout newLayout)
{
    // Synchronization2 orders shadow depth writes against later fragment shader sampling,
    // and orders previous frame sampling before this frame overwrites the map.
    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        srcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        srcAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        dstAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dstAccess = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

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
    barrier.image = shadowMap_.image();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void Renderer::transitionSwapchainImage(VkCommandBuffer commandBuffer, VkImage image,
                                        VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dstAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        srcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }

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
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void Renderer::transitionDepthImage(VkCommandBuffer commandBuffer)
{
    const VkImageLayout oldLayout = swapchain_.depthImageLayout();
    constexpr VkImageLayout newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 srcAccess = VK_ACCESS_2_NONE;

    if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        srcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchain_.depthImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    // Depth is cleared and written every frame. This barrier initializes the
    // image after swapchain creation and orders later depth writes on the queue.
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    swapchain_.setDepthImageLayout(newLayout);
}

} // namespace ve
