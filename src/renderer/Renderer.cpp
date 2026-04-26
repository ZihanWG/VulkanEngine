#include "renderer/Renderer.h"

#include "core/Window.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <glm/mat4x4.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ve {

namespace {

struct ObjectFrameData {
    glm::mat4 mvp{1.0f};
};

// The shader buffer_reference block contains one std430 mat4. Each entry is
// 64 bytes, a multiple of the 16-byte matrix alignment, so base + index * size
// remains correctly aligned for this simple per-object MVP array.
static_assert(sizeof(ObjectFrameData) == 64);

constexpr uint32_t kMaxFrameObjects = 64;

struct PushConstants {
    VkDeviceAddress objectFrameDataAddress = 0;
};

std::filesystem::path shaderPath(const char* filename)
{
#if defined(VULKAN_ENGINE_SHADER_DIR)
    return std::filesystem::path(VULKAN_ENGINE_SHADER_DIR) / filename;
#else
    return std::filesystem::path("shaders") / filename;
#endif
}

} // namespace

Renderer::Renderer(Window& window)
    : window_(window)
{
    context_.initialize(window_);

    frames_.resize(rhi::kMaxFramesInFlight);
    swapchain_.initialize(context_, window_.framebufferExtent());
    createMaterialDescriptorSetLayout();
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
    const VkResult acquireResult = vkAcquireNextImageKHR(
        context_.vkDevice(),
        swapchain_.handle(),
        UINT64_MAX,
        frame.imageAvailable,
        VK_NULL_HANDLE,
        &imageIndex);

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

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    const VkSwapchainKHR swapchain = swapchain_.handle();
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(context_.presentQueue(), &presentInfo);
    const bool needsRecreate = presentResult == VK_ERROR_OUT_OF_DATE_KHR
        || presentResult == VK_SUBOPTIMAL_KHR
        || acquireResult == VK_SUBOPTIMAL_KHR
        || window_.wasResized();

    if (presentResult != VK_SUCCESS && presentResult != VK_ERROR_OUT_OF_DATE_KHR && presentResult != VK_SUBOPTIMAL_KHR) {
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
    VkDescriptorSetLayoutBinding textureBinding{};
    textureBinding.binding = 0;
    textureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureBinding.descriptorCount = 1;
    textureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Set 0 binding 0 is intentionally only for the sampled texture. MVP data
    // stays on the Buffer Device Address + vertex push constant path.
    materialDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(&textureBinding, 1));
}

void Renderer::createPipeline()
{
    const VkVertexInputBindingDescription binding = renderer::vertexBindingDescription();
    const std::array<VkVertexInputAttributeDescription, 3> attributes = renderer::vertexAttributeDescriptions();
    const VkDescriptorSetLayout descriptorSetLayout = materialDescriptorSetLayout_.handle();
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        static_cast<uint32_t>(sizeof(PushConstants))
    };

    rhi::VulkanPipelineCreateInfo pipelineInfo{};
    pipelineInfo.vertexShaderPath = shaderPath("simple.vert.spv");
    pipelineInfo.fragmentShaderPath = shaderPath("simple.frag.spv");
    pipelineInfo.colorFormat = swapchain_.colorFormat();
    pipelineInfo.depthFormat = swapchain_.depthFormat();
    pipelineInfo.vertexBindings = std::span<const VkVertexInputBindingDescription>(&binding, 1);
    pipelineInfo.vertexAttributes = std::span<const VkVertexInputAttributeDescription>(attributes.data(), attributes.size());
    pipelineInfo.descriptorSetLayouts = std::span<const VkDescriptorSetLayout>(&descriptorSetLayout, 1);
    pipelineInfo.pushConstantRanges = std::span<const VkPushConstantRange>(&pushConstantRange, 1);
    pipelineInfo.enableDepth = true;

    pipeline_.create(context_.vkDevice(), pipelineInfo);
    pipelineColorFormat_ = pipelineInfo.colorFormat;
    pipelineDepthFormat_ = pipelineInfo.depthFormat;
}

void Renderer::createScene()
{
    renderObjects_.clear();

    cubeMesh_ = renderer::Mesh::createCube(context_, commandContext_);
    createCheckerboardTexture();
    createMaterial();

    camera_.position = {0.0f, 0.35f, 5.5f};
    camera_.target = {0.0f, 0.1f, 0.0f};

    renderObjects_.reserve(4);
    const auto addCube = [this](
                             std::string debugName,
                             const glm::vec3& position,
                             const glm::vec3& rotationRadians,
                             const glm::vec3& scale) {
        renderer::RenderObject cube{};
        cube.mesh = &cubeMesh_;
        cube.material = &checkerboardMaterial_;
        cube.debugName = std::move(debugName);
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        renderObjects_.push_back(std::move(cube));
    };

    addCube("Center Cube", {0.0f, -0.1f, 0.0f}, {0.2f, 0.0f, 0.0f}, {0.7f, 0.7f, 0.7f});
    addCube("Left Cube", {-1.35f, -0.15f, -0.35f}, {0.0f, 0.35f, 0.2f}, {0.5f, 0.5f, 0.5f});
    addCube("Right Cube", {1.35f, -0.05f, -0.25f}, {0.25f, -0.35f, 0.0f}, {0.55f, 0.8f, 0.55f});
    addCube("Elevated Cube", {0.0f, 1.0f, -0.7f}, {-0.3f, 0.2f, 0.45f}, {0.45f, 0.45f, 0.45f});
}

void Renderer::createCheckerboardTexture()
{
    checkerboardTexture_.createCheckerboard(context_, commandContext_, 256, 256);
}

void Renderer::createMaterial()
{
    checkerboardMaterial_ = renderer::Material{};
    checkerboardMaterial_.debugName = "Checkerboard";
    checkerboardMaterial_.baseColorTexture = &checkerboardTexture_;
    createMaterialDescriptorSet(checkerboardMaterial_);
}

void Renderer::createMaterialDescriptorSet(renderer::Material& material)
{
    if (!material.baseColorTexture || !material.baseColorTexture->valid()) {
        throw std::runtime_error("Cannot create a material descriptor set without a valid base color texture.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    if (materialDescriptorPool_.handle() == VK_NULL_HANDLE) {
        materialDescriptorPool_.create(
            context_.vkDevice(),
            std::span<const VkDescriptorPoolSize>(&poolSize, 1),
            1);
    }

    const VkDescriptorSetLayout descriptorSetLayout = materialDescriptorSetLayout_.handle();
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = materialDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, &material.descriptorSet));

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = material.baseColorTexture->sampler();
    imageInfo.imageView = material.baseColorTexture->imageView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = material.descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;

    // The material descriptor stores the sampler, image view, and final shader-read layout.
    // The texture contents are already uploaded before this update happens.
    vkUpdateDescriptorSets(context_.vkDevice(), 1, &write, 0, nullptr);
}

void Renderer::createObjectFrameDataBuffers()
{
    frameObjectDataBuffers_.resize(frames_.size());

    for (rhi::VulkanBuffer& frameObjectDataBuffer : frameObjectDataBuffers_) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxFrameObjects * sizeof(ObjectFrameData));
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
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

    const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
    std::vector<ObjectFrameData> objectFrameData(objectCount);

    const VkExtent2D extent = swapchain_.extent();
    const float aspect = extent.height == 0
        ? 1.0f
        : static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 view = camera_.viewMatrix();
    const glm::mat4 projection = camera_.projectionMatrix(aspect);

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
            object.transform.rotationRadians = {elapsedSeconds * 0.35f, elapsedSeconds * 0.55f, 0.45f};
            break;
        default:
            object.transform.rotationRadians = {
                elapsedSeconds * (0.2f + 0.05f * static_cast<float>(objectIndex)),
                elapsedSeconds * 0.4f,
                elapsedSeconds * 0.3f
            };
            break;
        }

        const glm::mat4 model = object.transform.modelMatrix();
        objectFrameData[objectIndex].mvp = projection * view * model;
    }

    frameObjectDataBuffers_.at(frameIndex).upload(
        std::as_bytes(std::span<const ObjectFrameData>(objectFrameData.data(), objectFrameData.size())));
}

void Renderer::recreateSwapchain()
{
    if (window_.isMinimized()) {
        return;
    }

    context_.waitIdle();
    swapchain_.recreate(context_, window_.framebufferExtent());
    sync_.recreateRenderFinishedSemaphores(swapchain_.imageCount());

    const bool pipelineNeedsRecreate = pipeline_.pipeline() == VK_NULL_HANDLE
        || pipelineColorFormat_ != swapchain_.colorFormat()
        || pipelineDepthFormat_ != swapchain_.depthFormat();
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

    // Synchronization2 barrier: make the acquired present image writable as a color attachment.
    transitionSwapchainImage(
        commandBuffer,
        swapchainImage,
        swapchain_.imageLayout(imageIndex),
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

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());

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

    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.mesh || !object.material || object.material->descriptorSet == VK_NULL_HANDLE) {
            continue;
        }

        const VkDescriptorSet descriptorSet = object.material->descriptorSet;
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_.layout(),
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        const PushConstants pushConstants{
            objectFrameDataBaseAddress + static_cast<VkDeviceAddress>(objectIndex * sizeof(ObjectFrameData))
        };

        // The material descriptor binds the texture/sampler for the fragment shader.
        // The pushed address points at this object's MVP data for the vertex shader.
        vkCmdPushConstants(
            commandBuffer,
            pipeline_.layout(),
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            static_cast<uint32_t>(sizeof(PushConstants)),
            &pushConstants);

        const VkBuffer vertexBuffers[] = {object.mesh->vertexBuffer()};
        const VkDeviceSize vertexOffsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, object.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT16);

        vkCmdDrawIndexed(commandBuffer, object.mesh->indexCount(), 1, 0, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);

    // Synchronization2 barrier: presentation reads from images in PRESENT_SRC_KHR layout.
    transitionSwapchainImage(
        commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    swapchain_.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void Renderer::transitionSwapchainImage(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout)
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
        srcStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        srcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
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
