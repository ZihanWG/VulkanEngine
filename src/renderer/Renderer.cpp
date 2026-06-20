#include "renderer/Renderer.h"
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "core/PngWriter.h"
#include "core/Window.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <ImGuizmo.h> // must follow imgui.h (relies on its types)
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
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
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

namespace ve {


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
    sceneDocumentPath_ = defaultSceneDocumentPath();
    loadRuntimeSettingsAtStartup();

    context_.initialize(window_);

    frames_.resize(rhi::kMaxFramesInFlight);
    portfolioScreenshotReadbacks_.resize(frames_.size());
    gpuProfiler_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    swapchain_.initialize(context_, window_.framebufferExtent());
    imguiLayer_.initialize(window_, context_, swapchain_.colorFormat(), swapchain_.imageCount());
    createMaterialDescriptorSetLayout();
    createBindlessMaterialTextureHeap();
    createSkyboxDescriptorSetLayout();
    createPostProcessDescriptorSetLayouts();
    createDepthPyramidDescriptorSetLayout();
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
        destroyDepthPyramidResources();
        postProcessDescriptorPool_.reset();
        destroyTaaResources();
        destroyPostProcessSampler();
    }
}

void Renderer::drawFrame()
{
    if (window_.isMinimized()) {
        return;
    }

    updateCpuFrameTime();
    updateEditorCamera(cpuFrameDeltaMs_ * 0.001f);

    if (window_.wasResized()) {
        recreateSwapchain();
        window_.clearResizedFlag();
    }

    renderer::FrameResources& frame = frames_[currentFrame_];
    VK_CHECK(vkWaitForFences(context_.vkDevice(), 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX));
    processPortfolioScreenshotReadback(currentFrame_);
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
    drawViewportGizmo();
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
    gpuProfiler_.markFrameSubmitted(currentFrame_);

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
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F11) {
        setPortfolioCaptureMode(!portfolioCaptureMode_);
    }
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F12) {
        requestPortfolioScreenshot();
    }

    const bool uiWantsMouse = imguiLayer_.wantsMouseCapture();
    const bool uiWantsKeyboard = imguiLayer_.wantsKeyboardCapture();

    switch (event.type) {
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        if (event.button.button == SDL_BUTTON_RIGHT && !uiWantsMouse) {
            cameraFlying_ = true;
            editorCamera_.syncFromCamera(camera_);
            pendingLookDelta_ = glm::vec2(0.0f);
            if (SDL_Window* sdlWindow = window_.nativeHandle()) {
                SDL_SetWindowRelativeMouseMode(sdlWindow, true);
            }
        } else if (event.button.button == SDL_BUTTON_MIDDLE && !uiWantsMouse) {
            cameraPanning_ = true;
        } else if (event.button.button == SDL_BUTTON_LEFT && !uiWantsMouse && !ImGuizmo::IsOver() &&
                   !ImGuizmo::IsUsing()) {
            if ((SDL_GetModState() & SDL_KMOD_ALT) != 0) {
                cameraOrbiting_ = true;
            } else {
                leftMouseDown_ = true;
                leftMouseDragged_ = false;
                leftMouseDownPosition_ = glm::vec2(event.button.x, event.button.y);
            }
        }
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (event.button.button == SDL_BUTTON_RIGHT) {
            cameraFlying_ = false;
            if (SDL_Window* sdlWindow = window_.nativeHandle()) {
                SDL_SetWindowRelativeMouseMode(sdlWindow, false);
            }
        } else if (event.button.button == SDL_BUTTON_MIDDLE) {
            cameraPanning_ = false;
        } else if (event.button.button == SDL_BUTTON_LEFT) {
            if (cameraOrbiting_) {
                cameraOrbiting_ = false;
            } else if (leftMouseDown_ && !leftMouseDragged_ && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing()) {
                pickObjectAtCursor(event.button.x, event.button.y);
            }
            leftMouseDown_ = false;
        }
        break;
    }
    case SDL_EVENT_MOUSE_MOTION: {
        const float xrel = event.motion.xrel;
        const float yrel = event.motion.yrel;
        if (cameraFlying_) {
            pendingLookDelta_ += glm::vec2(xrel, yrel);
        } else if (cameraOrbiting_) {
            constexpr float orbitSensitivity = 0.01f;
            editorCamera_.orbit(camera_, -xrel * orbitSensitivity, -yrel * orbitSensitivity);
        } else if (cameraPanning_) {
            const float panScale = 0.01f * std::max(glm::length(camera_.target - camera_.position), 1.0f);
            editorCamera_.pan(camera_, -xrel * panScale, yrel * panScale);
        }
        if (leftMouseDown_) {
            const glm::vec2 current(event.motion.x, event.motion.y);
            if (glm::length(current - leftMouseDownPosition_) > 4.0f) {
                leftMouseDragged_ = true;
            }
        }
        break;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
        if (!uiWantsMouse) {
            pendingScroll_ += event.wheel.y;
        }
        break;
    }
    case SDL_EVENT_KEY_DOWN: {
        if (!event.key.repeat && !cameraFlying_ && !uiWantsKeyboard) {
            if (event.key.key == SDLK_W) {
                gizmoOperation_ = GizmoOperation::Translate;
            } else if (event.key.key == SDLK_E) {
                gizmoOperation_ = GizmoOperation::Rotate;
            } else if (event.key.key == SDLK_R) {
                gizmoOperation_ = GizmoOperation::Scale;
            } else if (event.key.key == SDLK_X) {
                gizmoWorldSpace_ = !gizmoWorldSpace_;
            }
        }
        break;
    }
    default:
        break;
    }
}

void Renderer::updateEditorCamera(float deltaSeconds)
{
    if (cameraFlying_) {
        EditorCameraInput input;
        const bool* keys = SDL_GetKeyboardState(nullptr);
        if (keys) {
            if (keys[SDL_SCANCODE_W]) {
                input.moveAxis.z += 1.0f;
            }
            if (keys[SDL_SCANCODE_S]) {
                input.moveAxis.z -= 1.0f;
            }
            if (keys[SDL_SCANCODE_D]) {
                input.moveAxis.x += 1.0f;
            }
            if (keys[SDL_SCANCODE_A]) {
                input.moveAxis.x -= 1.0f;
            }
            if (keys[SDL_SCANCODE_E]) {
                input.moveAxis.y += 1.0f;
            }
            if (keys[SDL_SCANCODE_Q]) {
                input.moveAxis.y -= 1.0f;
            }
            if (keys[SDL_SCANCODE_LSHIFT] || keys[SDL_SCANCODE_RSHIFT]) {
                input.speedScale = 3.0f;
            }
        }
        constexpr float lookSensitivity = 0.0025f;
        input.lookDelta = glm::vec2(pendingLookDelta_.x * lookSensitivity, -pendingLookDelta_.y * lookSensitivity);
        editorCamera_.updateFly(camera_, input, deltaSeconds);
    }

    if (pendingScroll_ != 0.0f) {
        if (cameraFlying_) {
            const float scaled = editorCamera_.moveSpeed() * std::pow(1.15f, pendingScroll_);
            editorCamera_.setMoveSpeed(std::clamp(scaled, 0.1f, 1000.0f));
        } else {
            const float distance = glm::length(camera_.target - camera_.position);
            editorCamera_.dolly(camera_, pendingScroll_ * 0.1f * std::max(distance, 0.5f));
        }
    }

    pendingLookDelta_ = glm::vec2(0.0f);
    pendingScroll_ = 0.0f;
}

renderer::Ray Renderer::screenPointToRay(float pixelX, float pixelY) const
{
    const ImGuiIO& io = ImGui::GetIO();
    const float width = io.DisplaySize.x > 0.0f ? io.DisplaySize.x : 1.0f;
    const float height = io.DisplaySize.y > 0.0f ? io.DisplaySize.y : 1.0f;

    // NDC in [-1, 1]. The Vulkan Y-flip is baked into frameViewProjection_, so the
    // same screen->NDC mapping inverts consistently. (If picking ends up vertically
    // mirrored on a given driver, flip the sign of ndcY.)
    const float ndcX = (2.0f * pixelX / width) - 1.0f;
    const float ndcY = (2.0f * pixelY / height) - 1.0f;

    const glm::mat4 invViewProjection = glm::inverse(frameViewProjection_);
    glm::vec4 nearPoint = invViewProjection * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    glm::vec4 farPoint = invViewProjection * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    renderer::Ray ray;
    ray.origin = glm::vec3(nearPoint);
    ray.direction = glm::normalize(glm::vec3(farPoint - nearPoint));
    return ray;
}

void Renderer::pickObjectAtCursor(float pixelX, float pixelY)
{
    const renderer::Ray ray = screenPointToRay(pixelX, pixelY);
    float closestDistance = std::numeric_limits<float>::infinity();
    size_t hitIndex = kInvalidRenderObjectIndex;

    for (size_t index = 0; index < renderObjects_.size(); ++index) {
        const renderer::RenderObject& object = renderObjects_[index];
        if (!isRenderObjectActive(object)) {
            continue;
        }

        const renderer::Aabb bounds = object.worldBounds();
        float distance = 0.0f;
        if (bounds.intersectRay(ray, distance) && distance < closestDistance) {
            closestDistance = distance;
            hitIndex = index;
        }
    }

    // Clicking empty space clears the selection.
    selectedRenderObjectIndex_ = hitIndex;
    invalidateDepthPyramid();
}

void Renderer::drawViewportGizmo()
{
    if (cameraFlying_ || selectedRenderObjectIndex_ >= renderObjects_.size()) {
        return;
    }

    renderer::RenderObject& object = renderObjects_[selectedRenderObjectIndex_];
    const ImGuiIO& io = ImGui::GetIO();

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
    ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);

    const float aspect = io.DisplaySize.y > 0.0f ? io.DisplaySize.x / io.DisplaySize.y : 1.0f;
    glm::mat4 view = camera_.viewMatrix();
    glm::mat4 projection = camera_.projectionMatrix(aspect);
    projection[1][1] *= -1.0f; // undo the Vulkan Y-flip; ImGuizmo expects GL-style projection

    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    if (gizmoOperation_ == GizmoOperation::Rotate) {
        operation = ImGuizmo::ROTATE;
    } else if (gizmoOperation_ == GizmoOperation::Scale) {
        operation = ImGuizmo::SCALE;
    }
    const ImGuizmo::MODE mode = gizmoWorldSpace_ ? ImGuizmo::WORLD : ImGuizmo::LOCAL;

    glm::mat4 model = object.transform.modelMatrix();
    if (ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection), operation, mode, glm::value_ptr(model))) {
        object.transform = renderer::Transform::fromMatrix(model);
        convertMatrixOverrideToEditableTrs(object.transform);
        object.animateTransform = false;
        invalidateDepthPyramid();
    }
}

void Renderer::waitIdle()
{
    context_.waitIdle();
}

void Renderer::requestPortfolioScreenshot()
{
    if (portfolioScreenshotRequested_ || hasPendingPortfolioScreenshotReadback()) {
        portfolioScreenshotStatus_ = "Screenshot capture is already queued.";
        return;
    }
    if (!swapchain_.supportsTransferSrc()) {
        portfolioScreenshotStatus_ = "Screenshot unavailable: swapchain transfer-source usage is unsupported.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }
    if (!supportedScreenshotFormat(swapchain_.colorFormat())) {
        portfolioScreenshotStatus_ =
            std::string("Screenshot unavailable: unsupported swapchain format ") + vkFormatName(swapchain_.colorFormat()) + ".";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }

    if (!ensurePortfolioShowcaseSceneReady()) {
        return;
    }

    if (!portfolioCaptureMode_) {
        setPortfolioCaptureMode(true);
    } else {
        applyPortfolioCaptureSettings();
    }

    if (!portfolioCaptureMode_) {
        portfolioScreenshotStatus_ = "Screenshot unavailable: portfolio showcase scene could not be activated.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }

    portfolioScreenshotRequested_ = true;
    portfolioScreenshotStatus_ = "Portfolio showcase screenshot queued for the next rendered frame.";
}

bool Renderer::hasPendingPortfolioScreenshotReadback() const
{
    for (const PortfolioScreenshotReadback& readback : portfolioScreenshotReadbacks_) {
        if (readback.pending) {
            return true;
        }
    }

    return false;
}

void Renderer::recordPortfolioScreenshotCopy(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    if (!portfolioScreenshotRequested_) {
        return;
    }

    if (!ensurePortfolioShowcaseSceneReady()) {
        portfolioScreenshotRequested_ = false;
        return;
    }
    if (!portfolioCaptureMode_) {
        setPortfolioCaptureMode(true);
        portfolioScreenshotStatus_ =
            "Screenshot deferred: portfolio capture mode was not active for the frame being copied.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }
    if (!currentFrameHasPortfolioShowcaseDrawItems()) {
        applyPortfolioCaptureSettings();
        portfolioScreenshotStatus_ =
            "Screenshot deferred: portfolio showcase draw items were not active for the frame being copied.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }

    portfolioScreenshotRequested_ = false;
    if (currentFrame_ >= portfolioScreenshotReadbacks_.size()) {
        portfolioScreenshotStatus_ = "Screenshot failed: frame readback slots are not initialized.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }
    if (!swapchain_.supportsTransferSrc()) {
        portfolioScreenshotStatus_ = "Screenshot failed: swapchain transfer-source usage is unsupported.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    const VkFormat format = swapchain_.colorFormat();
    if (extent.width == 0 || extent.height == 0 || !supportedScreenshotFormat(format)) {
        portfolioScreenshotStatus_ =
            std::string("Screenshot failed: unsupported extent or format ") + vkFormatName(format) + ".";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }

    PortfolioScreenshotReadback& readback = portfolioScreenshotReadbacks_[currentFrame_];
    if (readback.pending) {
        portfolioScreenshotStatus_ = "Screenshot capture skipped: previous readback is still pending.";
        Logger::warn(portfolioScreenshotStatus_);
        return;
    }

    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 4U;
    if (!readback.buffer.valid() || readback.buffer.size() != byteSize) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = byteSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        readback.buffer.createBuffer(context_, bufferInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  readback.buffer.buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "PortfolioScreenshotReadbackBuffer" + std::to_string(currentFrame_));
    }

    // Screenshot capture sits between CompositePass and ImGuiPass. The swapchain
    // image is copied as a transfer source, then returned to color attachment
    // layout so the normal overlay/present path can continue unchanged.
    const VkImageLayout oldLayout = swapchain_.imageLayout(imageIndex);
    VkImageMemoryBarrier2 toTransferBarrier{};
    toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toTransferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toTransferBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toTransferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toTransferBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toTransferBarrier.oldLayout = oldLayout;
    toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.image = swapchain_.image(imageIndex);
    toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferBarrier.subresourceRange.baseMipLevel = 0;
    toTransferBarrier.subresourceRange.levelCount = 1;
    toTransferBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo toTransferDependency{};
    toTransferDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    toTransferDependency.imageMemoryBarrierCount = 1;
    toTransferDependency.pImageMemoryBarriers = &toTransferBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &toTransferDependency);
    swapchain_.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(commandBuffer,
                           swapchain_.image(imageIndex),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer.buffer(),
                           1,
                           &copyRegion);

    VkBufferMemoryBarrier2 hostReadBarrier{};
    hostReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    hostReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    hostReadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    hostReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostReadBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    hostReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hostReadBarrier.buffer = readback.buffer.buffer();
    hostReadBarrier.offset = 0;
    hostReadBarrier.size = byteSize;

    VkImageMemoryBarrier2 toColorBarrier{};
    toColorBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toColorBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    toColorBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toColorBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toColorBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toColorBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toColorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColorBarrier.image = swapchain_.image(imageIndex);
    toColorBarrier.subresourceRange = toTransferBarrier.subresourceRange;

    std::array<VkImageMemoryBarrier2, 1> imageBarriers{toColorBarrier};
    VkDependencyInfo afterCopyDependency{};
    afterCopyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    afterCopyDependency.bufferMemoryBarrierCount = 1;
    afterCopyDependency.pBufferMemoryBarriers = &hostReadBarrier;
    afterCopyDependency.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
    afterCopyDependency.pImageMemoryBarriers = imageBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &afterCopyDependency);
    swapchain_.setImageLayout(imageIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    const std::filesystem::path outputDirectory = portfolioScreenshotDirectory();
    readback.extent = extent;
    readback.format = format;
    readback.timestampedPath =
        outputDirectory / ("vulkan_engine_portfolio_" + portfolioTimestamp() + ".png");
    readback.latestPath = outputDirectory / "vulkan_engine_portfolio_latest.png";
    readback.pending = true;
    portfolioScreenshotStatus_ =
        "Screenshot readback queued from final composite before ImGui overlay.";
}

void Renderer::processPortfolioScreenshotReadback(uint32_t frameIndex)
{
    if (frameIndex >= portfolioScreenshotReadbacks_.size()) {
        return;
    }

    PortfolioScreenshotReadback& readback = portfolioScreenshotReadbacks_[frameIndex];
    if (!readback.pending || !readback.buffer.valid()) {
        return;
    }

    try {
        std::vector<std::byte> pixels(static_cast<size_t>(readback.buffer.size()));
        readback.buffer.download(std::span<std::byte>(pixels.data(), pixels.size()));

        const std::vector<uint8_t> rgba = convertScreenshotToRgba8(pixels, readback.extent, readback.format);
        const uint32_t rowStride = readback.extent.width * 4U;
        writePngRgba8(readback.timestampedPath, readback.extent.width, readback.extent.height, rgba, rowStride);
        writePngRgba8(readback.latestPath, readback.extent.width, readback.extent.height, rgba, rowStride);

        lastPortfolioScreenshotPath_ = readback.timestampedPath;
        portfolioScreenshotStatus_ = "Saved portfolio screenshot: " + readback.timestampedPath.string();
        Logger::info(portfolioScreenshotStatus_);
    } catch (const std::exception& error) {
        portfolioScreenshotStatus_ = std::string("Screenshot save failed: ") + error.what();
        Logger::warn(portfolioScreenshotStatus_);
    }

    readback.pending = false;
}

void Renderer::setPortfolioCaptureMode(bool enabled)
{
    if (portfolioCaptureMode_ == enabled) {
        if (enabled && ensurePortfolioShowcaseSceneReady()) {
            applyPortfolioCaptureSettings();
        }
        return;
    }

    if (enabled) {
        if (!ensurePortfolioShowcaseSceneReady()) {
            portfolioCaptureMode_ = false;
            return;
        }
        portfolioCaptureSavedState_.camera = camera_;
        portfolioCaptureSavedState_.toneMapping = toneMappingSettings_;
        portfolioCaptureSavedState_.bloom = bloomSettings_;
        portfolioCaptureSavedState_.csm = csmSettings_;
        portfolioCaptureSavedState_.currentExposure = currentExposure_;
        portfolioCaptureSavedState_.valid = true;
        portfolioCaptureMode_ = true;
        applyPortfolioCaptureSettings();
    } else {
        portfolioCaptureMode_ = false;
        restorePortfolioCaptureSettings();
    }
}

void Renderer::applyPortfolioCaptureSettings()
{
    resetPortfolioShowcaseObjectsToPreset();

    camera_ = portfolioCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;

    toneMappingSettings_.operatorType = 1;
    toneMappingSettings_.enableAutoExposure = false;
    toneMappingSettings_.exposureMode = static_cast<int>(ExposureMode::Manual);
    toneMappingSettings_.manualExposure = 1.0f;
    currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);

    bloomSettings_.enabled = true;
    bloomSettings_.useMipChain = true;
    bloomSettings_.threshold = 1.05f;
    bloomSettings_.intensity = 0.12f;
    bloomSettings_.radius = 0.9f;

    csmSettings_.enableTexelSnapping = true;
    csmSettings_.enableCascadeDebugColors = false;
    csmSettings_.lambda = 0.58f;
    csmSettings_.shadowDistance = std::min(csmSettings_.farPlane, 40.0f);
    clampRuntimeSettings();
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

void Renderer::restorePortfolioCaptureSettings()
{
    if (!portfolioCaptureSavedState_.valid) {
        return;
    }

    camera_ = portfolioCaptureSavedState_.camera;
    toneMappingSettings_ = portfolioCaptureSavedState_.toneMapping;
    bloomSettings_ = portfolioCaptureSavedState_.bloom;
    csmSettings_ = portfolioCaptureSavedState_.csm;
    currentExposure_ = portfolioCaptureSavedState_.currentExposure;
    portfolioCaptureSavedState_ = {};
    clampRuntimeSettings();
    invalidateDepthPyramid();
    invalidateTaaHistory();
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

    std::array<VkDescriptorSetLayoutBinding, 2> dualImageBindings{};
    dualImageBindings[0] = singleImageBinding;
    dualImageBindings[1] = singleImageBinding;
    dualImageBindings[1].binding = 1;

    postProcessDualImageDescriptorSetLayout_.create(
        context_.vkDevice(),
        std::span<const VkDescriptorSetLayoutBinding>(dualImageBindings.data(), dualImageBindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              postProcessDualImageDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "PostProcessDualImageDescriptorSetLayout");

    std::array<VkDescriptorSetLayoutBinding, 5> compositeBindings{};
    compositeBindings[0] = singleImageBinding;
    compositeBindings[1] = singleImageBinding;
    compositeBindings[1].binding = 1;
    compositeBindings[2] = singleImageBinding;
    compositeBindings[2].binding = 2;
    compositeBindings[3].binding = 3;
    compositeBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compositeBindings[3].descriptorCount = 1;
    compositeBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    compositeBindings[4] = singleImageBinding; // main depth, sampled for SSAO
    compositeBindings[4].binding = 4;

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

    std::array<VkDescriptorSetLayoutBinding, 3> exposureReduceBindings{};
    for (uint32_t bindingIndex = 0; bindingIndex < exposureReduceBindings.size(); ++bindingIndex) {
        exposureReduceBindings[bindingIndex].binding = bindingIndex;
        exposureReduceBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureReduceBindings[bindingIndex].descriptorCount = 1;
        exposureReduceBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    try {
        postProcessExposureReduceDescriptorSetLayout_.create(
            context_.vkDevice(),
            std::span<const VkDescriptorSetLayoutBinding>(exposureReduceBindings.data(),
                                                          exposureReduceBindings.size()));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  postProcessExposureReduceDescriptorSetLayout_.handle(),
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                  "PostProcessExposureReduceDescriptorSetLayout");
    } catch (const std::exception& error) {
        disableAutoExposureFallback(std::string("Exposure reduce descriptor layout creation failed: ") + error.what());
    }
}

void Renderer::createDepthPyramidDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    depthPyramidDescriptorSetLayout_.create(
        context_.vkDevice(), std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()));
    rhi::debug::setObjectName(context_.vkDevice(),
                              depthPyramidDescriptorSetLayout_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                              "DepthPyramidDescriptorSetLayout");
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

void Renderer::createTaaResources()
{
    destroyTaaResources();

    const VkExtent2D extent = swapchain_.extent();
    if (extent.width == 0 || extent.height == 0) {
        throw std::runtime_error("Cannot create TAA history resources for a zero-sized swapchain extent.");
    }

    rhi::VulkanImageCreateInfo historyInfo{};
    historyInfo.width = extent.width;
    historyInfo.height = extent.height;
    historyInfo.format = kSceneColorFormat;
    historyInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    historyInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    for (uint32_t index = 0; index < kTaaHistoryCount; ++index) {
        historyInfo.debugName = "TAAHistory" + std::to_string(index);
        taaHistoryImages_[index].create(context_, historyInfo);
        taaHistoryLayouts_[index] = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    invalidateTaaHistory();
}

void Renderer::destroyTaaResources()
{
    taaResolveDescriptorSets_.fill(VK_NULL_HANDLE);
    taaBloomExtractDescriptorSets_.fill(VK_NULL_HANDLE);
    taaBloomMipDownsampleDescriptorSets_.fill(VK_NULL_HANDLE);
    for (auto& descriptorSets : taaCompositeDescriptorSets_) {
        descriptorSets.clear();
    }
    for (auto& descriptorSets : taaLuminanceDescriptorSets_) {
        descriptorSets.clear();
    }
    for (auto& descriptorSets : taaHistogramDescriptorSets_) {
        descriptorSets.clear();
    }

    for (rhi::VulkanImage& image : taaHistoryImages_) {
        image.reset();
    }
    taaHistoryLayouts_.fill(VK_IMAGE_LAYOUT_UNDEFINED);
    invalidateTaaHistory();
}

void Renderer::invalidateTaaHistory()
{
    taaHistoryValid_ = false;
    taaHistoryWriteIndex_ = 0;
    taaPostProcessHistoryIndex_ = 0;
    taaJitterIndex_ = 0;
    taaCurrentJitterPixels_ = {0.0f, 0.0f};
    taaPreviousJitterPixels_ = {0.0f, 0.0f};
    taaCurrentJitterNdc_ = {0.0f, 0.0f};
    taaPreviousJitterNdc_ = {0.0f, 0.0f};
    frameJitteredProjection_ = glm::mat4{1.0f};
    frameJitteredViewProjection_ = glm::mat4{1.0f};
    previousFrameViewProjection_ = glm::mat4{1.0f};
}

void Renderer::destroyDepthPyramidResources()
{
    depthPyramidValid_ = false;
    depthPyramidBuildAvailable_ = false;
    depthPyramidDescriptorSets_.clear();
    depthPyramidDescriptorPool_.reset();

    for (VkImageView imageView : depthPyramidMipImageViews_) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(context_.vkDevice(), imageView, nullptr);
        }
    }
    depthPyramidMipImageViews_.clear();

    if (depthPyramidSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(context_.vkDevice(), depthPyramidSampler_, nullptr);
        depthPyramidSampler_ = VK_NULL_HANDLE;
    }

    depthPyramid_.reset();
    depthPyramidLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    depthPyramidMipLevels_ = 0;
    selectedDepthPyramidDebugMip_ = 0;
    depthPyramidViewProjection_ = glm::mat4{1.0f};
    depthPyramidCameraPosition_ = glm::vec3{0.0f};
}

void Renderer::invalidateDepthPyramid()
{
    depthPyramidValid_ = false;
    depthPyramidViewProjection_ = glm::mat4{1.0f};
    depthPyramidCameraPosition_ = glm::vec3{0.0f};
}

void Renderer::createDepthPyramidResources()
{
    destroyDepthPyramidResources();

    if (depthPyramidDescriptorSetLayout_.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot create depth pyramid resources without a descriptor set layout.");
    }

    const bool sampledFormatSupported =
        depthPyramidFormatSupports(context_.physicalDevice(), VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
    const bool storageFormatSupported =
        depthPyramidFormatSupports(context_.physicalDevice(), VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
    if (!sampledFormatSupported) {
        Logger::warn("Depth pyramid disabled: VK_FORMAT_R32_SFLOAT is not sampleable on this device.");
        useGpuOcclusionCulling_ = false;
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    depthPyramidBuildAvailable_ = swapchain_.depthSupportsSampling() && storageFormatSupported;
    depthPyramidMipLevels_ = depthPyramidBuildAvailable_ ? calculateDepthPyramidMipLevels(extent) : 1U;

    rhi::VulkanImageCreateInfo imageInfo{};
    imageInfo.width = extent.width;
    imageInfo.height = extent.height;
    imageInfo.format = kDepthPyramidFormat;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | (depthPyramidBuildAvailable_ ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    imageInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageInfo.mipLevels = depthPyramidMipLevels_;
    imageInfo.debugName = "DepthPyramidHiZ";
    depthPyramid_.create(context_, imageInfo);
    depthPyramidLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    depthPyramidMipImageViews_.resize(depthPyramidMipLevels_, VK_NULL_HANDLE);
    for (uint32_t mipLevel = 0; mipLevel < depthPyramidMipLevels_; ++mipLevel) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = depthPyramid_.image();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kDepthPyramidFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = mipLevel;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(context_.vkDevice(), &viewInfo, nullptr, &depthPyramidMipImageViews_[mipLevel]));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  depthPyramidMipImageViews_[mipLevel],
                                  VK_OBJECT_TYPE_IMAGE_VIEW,
                                  "DepthPyramidHiZMip" + std::to_string(mipLevel) + "View");
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(std::max(depthPyramidMipLevels_, 1u) - 1u);
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    VK_CHECK(vkCreateSampler(context_.vkDevice(), &samplerInfo, nullptr, &depthPyramidSampler_));
    rhi::debug::setObjectName(
        context_.vkDevice(), depthPyramidSampler_, VK_OBJECT_TYPE_SAMPLER, "DepthPyramidNearestClampSampler");

    if (!depthPyramidBuildAvailable_) {
        useGpuOcclusionCulling_ = false;
        if (!swapchain_.depthSupportsSampling()) {
            Logger::warn("Depth pyramid generation disabled: selected main depth format is not sampleable.");
        }
        if (!storageFormatSupported) {
            Logger::warn("Depth pyramid generation disabled: VK_FORMAT_R32_SFLOAT storage images are unsupported.");
        }
        updateGpuCullingDepthPyramidDescriptors();
        return;
    }

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = depthPyramidMipLevels_;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = depthPyramidMipLevels_;
    depthPyramidDescriptorPool_.create(
        context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()), depthPyramidMipLevels_);
    rhi::debug::setObjectName(context_.vkDevice(),
                              depthPyramidDescriptorPool_.handle(),
                              VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                              "DepthPyramidDescriptorPool");

    depthPyramidDescriptorSets_.resize(depthPyramidMipLevels_, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(depthPyramidMipLevels_, depthPyramidDescriptorSetLayout_.handle());
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = depthPyramidDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(depthPyramidDescriptorSets_.size());
    allocateInfo.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, depthPyramidDescriptorSets_.data()));

    for (uint32_t mipLevel = 0; mipLevel < depthPyramidMipLevels_; ++mipLevel) {
        VkDescriptorImageInfo sourceInfo{};
        sourceInfo.sampler = depthPyramidSampler_;
        sourceInfo.imageView = mipLevel == 0 ? swapchain_.depthImageView() : depthPyramidMipImageViews_[mipLevel - 1];
        sourceInfo.imageLayout =
            mipLevel == 0 ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView = depthPyramidMipImageViews_[mipLevel];
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = depthPyramidDescriptorSets_[mipLevel];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &sourceInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = depthPyramidDescriptorSets_[mipLevel];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outputInfo;

        vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  depthPyramidDescriptorSets_[mipLevel],
                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                  "DepthPyramidDescriptorSetMip" + std::to_string(mipLevel));
    }

    updateGpuCullingDepthPyramidDescriptors();
}

void Renderer::updateGpuCullingDepthPyramidDescriptors()
{
    if (depthPyramid_.imageView() == VK_NULL_HANDLE || depthPyramidSampler_ == VK_NULL_HANDLE) {
        return;
    }

    VkDescriptorImageInfo depthPyramidInfo{};
    depthPyramidInfo.sampler = depthPyramidSampler_;
    depthPyramidInfo.imageView = depthPyramid_.imageView();
    depthPyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const auto updateSets = [this, &depthPyramidInfo](const std::vector<VkDescriptorSet>& descriptorSets) {
        for (VkDescriptorSet descriptorSet : descriptorSets) {
            if (descriptorSet == VK_NULL_HANDLE) {
                continue;
            }

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSet;
            write.dstBinding = 3;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &depthPyramidInfo;
            vkUpdateDescriptorSets(context_.vkDevice(), 1, &write, 0, nullptr);
        }
    };

    updateSets(gpuCullDescriptorSets_);
    updateSets(shadowCullDescriptorSets_);
}

void Renderer::createPostProcessResources()
{
    const VkExtent2D extent = swapchain_.extent();
    if (extent.width == 0 || extent.height == 0) {
        throw std::runtime_error("Cannot create post-process resources for a zero-sized swapchain extent.");
    }

    imguiLayer_.clearRenderTargetPreviewDescriptors();
    postProcessDescriptorPool_.reset();
    bloomExtractDescriptorSet_ = VK_NULL_HANDLE;
    bloomBlurHorizontalDescriptorSet_ = VK_NULL_HANDLE;
    bloomBlurVerticalDescriptorSet_ = VK_NULL_HANDLE;
    compositeDescriptorSet_ = VK_NULL_HANDLE;
    bloomMipDownsampleDescriptorSets_.clear();
    bloomMipUpsampleDescriptorSets_.clear();
    compositeDescriptorSets_.clear();
    luminanceDescriptorSets_.clear();
    histogramDescriptorSets_.clear();
    exposureReduceDescriptorSets_.clear();
    destroyTaaResources();
    bloomMipDownsampleImages_.clear();
    bloomMipUpsampleImages_.clear();
    bloomMipDownsampleLayouts_.clear();
    bloomMipUpsampleLayouts_.clear();
    selectedBloomMipDebugLevel_ = 0;
    destroyDepthPyramidResources();
    destroyLuminanceResources();
    destroyHistogramResources();
    destroyExposureResources();

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

    createTaaResources();

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

    const uint32_t bloomMipCount = calculateBloomMipChainLevels(extent);
    bloomMipDownsampleImages_.resize(bloomMipCount);
    bloomMipDownsampleLayouts_.assign(bloomMipCount, VK_IMAGE_LAYOUT_UNDEFINED);
    for (uint32_t level = 0; level < bloomMipCount; ++level) {
        const VkExtent2D mipSize = bloomMipExtent(extent, level);
        bloomInfo.width = mipSize.width;
        bloomInfo.height = mipSize.height;
        bloomInfo.debugName = "BloomMipDownsample" + std::to_string(level);
        bloomMipDownsampleImages_[level].create(context_, bloomInfo);
    }

    const uint32_t bloomUpsampleCount = bloomMipCount > 1u ? bloomMipCount - 1u : 0u;
    bloomMipUpsampleImages_.resize(bloomUpsampleCount);
    bloomMipUpsampleLayouts_.assign(bloomUpsampleCount, VK_IMAGE_LAYOUT_UNDEFINED);
    for (uint32_t level = 0; level < bloomUpsampleCount; ++level) {
        const VkExtent2D mipSize = bloomMipExtent(extent, level);
        bloomInfo.width = mipSize.width;
        bloomInfo.height = mipSize.height;
        bloomInfo.debugName = "BloomMipUpsample" + std::to_string(level);
        bloomMipUpsampleImages_[level].create(context_, bloomInfo);
    }

    createDepthPyramidResources();

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

    createExposureResources();
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
    const bool createExposureReduceDescriptors =
        createLuminanceDescriptors && createHistogramDescriptors && exposureReduceAvailable_ &&
        frameExposureBuffers_.size() == frames_.size() &&
        postProcessExposureReduceDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    const bool createCompositeDescriptors =
        !frameExposureBuffers_.empty() && frameExposureBuffers_.size() == frames_.size() &&
        postProcessCompositeDescriptorSetLayout_.handle() != VK_NULL_HANDLE;
    const uint32_t exposureDescriptorSetCount =
        (createLuminanceDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (createHistogramDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (createExposureReduceDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u);
    const uint32_t compositeDescriptorSetCount =
        createCompositeDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u;
    const uint32_t legacyBloomSetCount = 3u;
    const uint32_t bloomDownsampleSetCount = static_cast<uint32_t>(bloomMipDownsampleImages_.size());
    const uint32_t bloomUpsampleSetCount = static_cast<uint32_t>(bloomMipUpsampleImages_.size());
    const uint32_t taaResolveSetCount = kTaaHistoryCount;
    const uint32_t taaBloomExtractSetCount = kTaaHistoryCount;
    const uint32_t taaBloomDownsampleSetCount = bloomMipDownsampleImages_.empty() ? 0u : kTaaHistoryCount;
    const uint32_t taaCompositeDescriptorSetCount = createCompositeDescriptors
                                                       ? kTaaHistoryCount * static_cast<uint32_t>(frames_.size())
                                                       : 0u;
    const uint32_t taaLuminanceDescriptorSetCount = createLuminanceDescriptors
                                                       ? kTaaHistoryCount * static_cast<uint32_t>(frames_.size())
                                                       : 0u;
    const uint32_t taaHistogramDescriptorSetCount = createHistogramDescriptors
                                                       ? kTaaHistoryCount * static_cast<uint32_t>(frames_.size())
                                                       : 0u;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount =
        legacyBloomSetCount + bloomDownsampleSetCount + (2u * bloomUpsampleSetCount) +
        (4u * compositeDescriptorSetCount) +
        (createLuminanceDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (createHistogramDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (2u * taaResolveSetCount) + taaBloomExtractSetCount + taaBloomDownsampleSetCount +
        (4u * taaCompositeDescriptorSetCount) + taaLuminanceDescriptorSetCount +
        taaHistogramDescriptorSetCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount =
        (createLuminanceDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (createHistogramDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u) +
        (3u * (createExposureReduceDescriptors ? static_cast<uint32_t>(frames_.size()) : 0u)) +
        compositeDescriptorSetCount + taaCompositeDescriptorSetCount + taaLuminanceDescriptorSetCount +
        taaHistogramDescriptorSetCount;
    const uint32_t poolSizeCount = poolSizes[1].descriptorCount > 0 ? 2u : 1u;
    const uint32_t maxSets = legacyBloomSetCount + bloomDownsampleSetCount + bloomUpsampleSetCount +
                             compositeDescriptorSetCount + exposureDescriptorSetCount + taaResolveSetCount +
                             taaBloomExtractSetCount + taaBloomDownsampleSetCount +
                             taaCompositeDescriptorSetCount + taaLuminanceDescriptorSetCount +
                             taaHistogramDescriptorSetCount;

    postProcessDescriptorPool_.create(
        context_.vkDevice(), std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizeCount), maxSets);
    rhi::debug::setObjectName(
        context_.vkDevice(), postProcessDescriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "PostProcessPool");

    std::array<VkDescriptorSetLayout, 3> legacyDescriptorSetLayouts{
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessSingleImageDescriptorSetLayout_.handle(),
        postProcessSingleImageDescriptorSetLayout_.handle(),
    };
    std::array<VkDescriptorSet, 3> legacyDescriptorSets{};

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(legacyDescriptorSetLayouts.size());
    allocateInfo.pSetLayouts = legacyDescriptorSetLayouts.data();
    VK_CHECK(vkAllocateDescriptorSets(context_.vkDevice(), &allocateInfo, legacyDescriptorSets.data()));

    bloomExtractDescriptorSet_ = legacyDescriptorSets[0];
    bloomBlurHorizontalDescriptorSet_ = legacyDescriptorSets[1];
    bloomBlurVerticalDescriptorSet_ = legacyDescriptorSets[2];

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

    const auto imageInfo = [this](VkImageView imageView) {
        VkDescriptorImageInfo info{};
        info.sampler = postProcessSampler_;
        info.imageView = imageView;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return info;
    };

    // Depth image info for the composite SSAO sampler. Falls back to a sampleable
    // texture when the depth format itself cannot be sampled.
    const auto depthInfo = [this]() {
        VkDescriptorImageInfo info{};
        info.sampler = postProcessSampler_;
        if (swapchain_.depthSupportsSampling()) {
            info.imageView = swapchain_.depthImageView();
            info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        } else {
            info.imageView = checkerboardTexture_.imageView();
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        return info;
    };

    std::array<VkDescriptorImageInfo, 3> legacyImageInfos{
        imageInfo(sceneColor_.imageView()),
        imageInfo(bloomExtract_.imageView()),
        imageInfo(bloomPing_.imageView()),
    };

    std::array<VkWriteDescriptorSet, 3> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = bloomExtractDescriptorSet_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &legacyImageInfos[0];

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = bloomBlurHorizontalDescriptorSet_;
    writes[1].dstBinding = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &legacyImageInfos[1];

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = bloomBlurVerticalDescriptorSet_;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &legacyImageInfos[2];

    vkUpdateDescriptorSets(context_.vkDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE && taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
        std::array<VkDescriptorSetLayout, kTaaHistoryCount> taaResolveLayouts{
            postProcessDualImageDescriptorSetLayout_.handle(),
            postProcessDualImageDescriptorSetLayout_.handle(),
        };
        VkDescriptorSetAllocateInfo taaResolveAllocateInfo{};
        taaResolveAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        taaResolveAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        taaResolveAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaResolveLayouts.size());
        taaResolveAllocateInfo.pSetLayouts = taaResolveLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &taaResolveAllocateInfo, taaResolveDescriptorSets_.data()));

        std::array<VkDescriptorSetLayout, kTaaHistoryCount> taaSingleImageLayouts{
            postProcessSingleImageDescriptorSetLayout_.handle(),
            postProcessSingleImageDescriptorSetLayout_.handle(),
        };
        VkDescriptorSetAllocateInfo taaBloomAllocateInfo{};
        taaBloomAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        taaBloomAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        taaBloomAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaSingleImageLayouts.size());
        taaBloomAllocateInfo.pSetLayouts = taaSingleImageLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &taaBloomAllocateInfo, taaBloomExtractDescriptorSets_.data()));

        std::array<std::array<VkDescriptorImageInfo, 2>, kTaaHistoryCount> taaResolveImageInfos{};
        std::array<VkDescriptorImageInfo, kTaaHistoryCount> taaBloomImageInfos{};
        std::array<VkWriteDescriptorSet, kTaaHistoryCount * 3u> taaWrites{};
        for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
            taaResolveImageInfos[historyIndex][0] = imageInfo(sceneColor_.imageView());
            taaResolveImageInfos[historyIndex][1] = imageInfo(taaHistoryImages_[historyIndex].imageView());
            taaBloomImageInfos[historyIndex] = imageInfo(taaHistoryImages_[historyIndex].imageView());

            const uint32_t writeBase = historyIndex * 3u;
            taaWrites[writeBase].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            taaWrites[writeBase].dstSet = taaResolveDescriptorSets_[historyIndex];
            taaWrites[writeBase].dstBinding = 0;
            taaWrites[writeBase].descriptorCount = 1;
            taaWrites[writeBase].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            taaWrites[writeBase].pImageInfo = &taaResolveImageInfos[historyIndex][0];

            taaWrites[writeBase + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            taaWrites[writeBase + 1].dstSet = taaResolveDescriptorSets_[historyIndex];
            taaWrites[writeBase + 1].dstBinding = 1;
            taaWrites[writeBase + 1].descriptorCount = 1;
            taaWrites[writeBase + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            taaWrites[writeBase + 1].pImageInfo = &taaResolveImageInfos[historyIndex][1];

            taaWrites[writeBase + 2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            taaWrites[writeBase + 2].dstSet = taaBloomExtractDescriptorSets_[historyIndex];
            taaWrites[writeBase + 2].dstBinding = 0;
            taaWrites[writeBase + 2].descriptorCount = 1;
            taaWrites[writeBase + 2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            taaWrites[writeBase + 2].pImageInfo = &taaBloomImageInfos[historyIndex];

            rhi::debug::setObjectName(context_.vkDevice(),
                                      taaResolveDescriptorSets_[historyIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "TAAResolveDescriptorSet" + std::to_string(historyIndex));
            rhi::debug::setObjectName(context_.vkDevice(),
                                      taaBloomExtractDescriptorSets_[historyIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "TAABloomExtractDescriptorSet" + std::to_string(historyIndex));
        }
        vkUpdateDescriptorSets(
            context_.vkDevice(), static_cast<uint32_t>(taaWrites.size()), taaWrites.data(), 0, nullptr);

        if (!bloomMipDownsampleImages_.empty()) {
            VkDescriptorSetAllocateInfo taaDownsampleAllocateInfo{};
            taaDownsampleAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            taaDownsampleAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            taaDownsampleAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaSingleImageLayouts.size());
            taaDownsampleAllocateInfo.pSetLayouts = taaSingleImageLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(
                context_.vkDevice(), &taaDownsampleAllocateInfo, taaBloomMipDownsampleDescriptorSets_.data()));

            std::array<VkDescriptorImageInfo, kTaaHistoryCount> taaDownsampleImageInfos{};
            std::array<VkWriteDescriptorSet, kTaaHistoryCount> taaDownsampleWrites{};
            for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                taaDownsampleImageInfos[historyIndex] = imageInfo(taaHistoryImages_[historyIndex].imageView());
                taaDownsampleWrites[historyIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                taaDownsampleWrites[historyIndex].dstSet = taaBloomMipDownsampleDescriptorSets_[historyIndex];
                taaDownsampleWrites[historyIndex].dstBinding = 0;
                taaDownsampleWrites[historyIndex].descriptorCount = 1;
                taaDownsampleWrites[historyIndex].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                taaDownsampleWrites[historyIndex].pImageInfo = &taaDownsampleImageInfos[historyIndex];
                rhi::debug::setObjectName(context_.vkDevice(),
                                          taaBloomMipDownsampleDescriptorSets_[historyIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "TAABloomMipDownsampleDescriptorSet" + std::to_string(historyIndex));
            }
            vkUpdateDescriptorSets(context_.vkDevice(),
                                   static_cast<uint32_t>(taaDownsampleWrites.size()),
                                   taaDownsampleWrites.data(),
                                   0,
                                   nullptr);
        }
    }

    if (!bloomMipDownsampleImages_.empty()) {
        bloomMipDownsampleDescriptorSets_.assign(bloomMipDownsampleImages_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> downsampleLayouts(bloomMipDownsampleImages_.size(),
                                                             postProcessSingleImageDescriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo downsampleAllocateInfo{};
        downsampleAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        downsampleAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        downsampleAllocateInfo.descriptorSetCount = static_cast<uint32_t>(downsampleLayouts.size());
        downsampleAllocateInfo.pSetLayouts = downsampleLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &downsampleAllocateInfo, bloomMipDownsampleDescriptorSets_.data()));

        std::vector<VkDescriptorImageInfo> downsampleImageInfos(bloomMipDownsampleImages_.size());
        std::vector<VkWriteDescriptorSet> downsampleWrites(bloomMipDownsampleImages_.size());
        for (size_t level = 0; level < bloomMipDownsampleImages_.size(); ++level) {
            const VkImageView sourceView =
                level == 0 ? sceneColor_.imageView() : bloomMipDownsampleImages_[level - 1].imageView();
            downsampleImageInfos[level] = imageInfo(sourceView);
            downsampleWrites[level].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            downsampleWrites[level].dstSet = bloomMipDownsampleDescriptorSets_[level];
            downsampleWrites[level].dstBinding = 0;
            downsampleWrites[level].descriptorCount = 1;
            downsampleWrites[level].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            downsampleWrites[level].pImageInfo = &downsampleImageInfos[level];
            rhi::debug::setObjectName(context_.vkDevice(),
                                      bloomMipDownsampleDescriptorSets_[level],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "BloomMipDownsampleDescriptorSet" + std::to_string(level));
        }
        vkUpdateDescriptorSets(context_.vkDevice(),
                               static_cast<uint32_t>(downsampleWrites.size()),
                               downsampleWrites.data(),
                               0,
                               nullptr);
    }

    if (!bloomMipUpsampleImages_.empty()) {
        bloomMipUpsampleDescriptorSets_.assign(bloomMipUpsampleImages_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> upsampleLayouts(bloomMipUpsampleImages_.size(),
                                                           postProcessDualImageDescriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo upsampleAllocateInfo{};
        upsampleAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        upsampleAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        upsampleAllocateInfo.descriptorSetCount = static_cast<uint32_t>(upsampleLayouts.size());
        upsampleAllocateInfo.pSetLayouts = upsampleLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &upsampleAllocateInfo, bloomMipUpsampleDescriptorSets_.data()));

        std::vector<std::array<VkDescriptorImageInfo, 2>> upsampleImageInfos(bloomMipUpsampleImages_.size());
        std::vector<VkWriteDescriptorSet> upsampleWrites(bloomMipUpsampleImages_.size() * 2u);
        for (size_t level = 0; level < bloomMipUpsampleImages_.size(); ++level) {
            const VkImageView lowerView =
                level + 1 == bloomMipDownsampleImages_.size() - 1
                    ? bloomMipDownsampleImages_[level + 1].imageView()
                    : bloomMipUpsampleImages_[level + 1].imageView();
            upsampleImageInfos[level][0] = imageInfo(bloomMipDownsampleImages_[level].imageView());
            upsampleImageInfos[level][1] = imageInfo(lowerView);

            const size_t writeBase = level * 2u;
            upsampleWrites[writeBase].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            upsampleWrites[writeBase].dstSet = bloomMipUpsampleDescriptorSets_[level];
            upsampleWrites[writeBase].dstBinding = 0;
            upsampleWrites[writeBase].descriptorCount = 1;
            upsampleWrites[writeBase].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            upsampleWrites[writeBase].pImageInfo = &upsampleImageInfos[level][0];

            upsampleWrites[writeBase + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            upsampleWrites[writeBase + 1].dstSet = bloomMipUpsampleDescriptorSets_[level];
            upsampleWrites[writeBase + 1].dstBinding = 1;
            upsampleWrites[writeBase + 1].descriptorCount = 1;
            upsampleWrites[writeBase + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            upsampleWrites[writeBase + 1].pImageInfo = &upsampleImageInfos[level][1];

            rhi::debug::setObjectName(context_.vkDevice(),
                                      bloomMipUpsampleDescriptorSets_[level],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "BloomMipUpsampleDescriptorSet" + std::to_string(level));
        }
        vkUpdateDescriptorSets(context_.vkDevice(),
                               static_cast<uint32_t>(upsampleWrites.size()),
                               upsampleWrites.data(),
                               0,
                               nullptr);
    }

    if (createCompositeDescriptors) {
        compositeDescriptorSets_.assign(frames_.size(), VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayout> compositeLayouts(frames_.size(),
                                                            postProcessCompositeDescriptorSetLayout_.handle());
        VkDescriptorSetAllocateInfo compositeAllocateInfo{};
        compositeAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        compositeAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
        compositeAllocateInfo.descriptorSetCount = static_cast<uint32_t>(compositeDescriptorSets_.size());
        compositeAllocateInfo.pSetLayouts = compositeLayouts.data();
        VK_CHECK(vkAllocateDescriptorSets(
            context_.vkDevice(), &compositeAllocateInfo, compositeDescriptorSets_.data()));
        compositeDescriptorSet_ = compositeDescriptorSets_.empty() ? VK_NULL_HANDLE : compositeDescriptorSets_.front();

        std::vector<std::array<VkDescriptorImageInfo, 4>> compositeImageInfos(frames_.size());
        std::vector<VkDescriptorBufferInfo> compositeExposureInfos(frames_.size());
        std::vector<VkWriteDescriptorSet> compositeWrites(frames_.size() * 5u);
        VkImageView mipBloomView = bloomPong_.imageView();
        if (!bloomMipUpsampleImages_.empty()) {
            mipBloomView = bloomMipUpsampleImages_.front().imageView();
        } else if (!bloomMipDownsampleImages_.empty()) {
            mipBloomView = bloomMipDownsampleImages_.front().imageView();
        }

        // The main depth buffer is sampled by the composite pass for SSAO. When the
        // depth format cannot be sampled, bind a harmless fallback and disable SSAO.
        ssaoAvailable_ = swapchain_.depthSupportsSampling();
        const VkDescriptorImageInfo compositeDepthInfo = depthInfo();
        for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
            compositeImageInfos[frameIndex][0] = imageInfo(sceneColor_.imageView());
            compositeImageInfos[frameIndex][1] = imageInfo(bloomPong_.imageView());
            compositeImageInfos[frameIndex][2] = imageInfo(mipBloomView);
            compositeImageInfos[frameIndex][3] = compositeDepthInfo;

            compositeExposureInfos[frameIndex].buffer = frameExposureBuffers_[frameIndex].buffer();
            compositeExposureInfos[frameIndex].offset = 0;
            compositeExposureInfos[frameIndex].range = sizeof(ExposureState);

            const size_t writeBase = frameIndex * 5u;
            for (uint32_t binding = 0; binding < 3; ++binding) {
                compositeWrites[writeBase + binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                compositeWrites[writeBase + binding].dstSet = compositeDescriptorSets_[frameIndex];
                compositeWrites[writeBase + binding].dstBinding = binding;
                compositeWrites[writeBase + binding].descriptorCount = 1;
                compositeWrites[writeBase + binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                compositeWrites[writeBase + binding].pImageInfo = &compositeImageInfos[frameIndex][binding];
            }

            compositeWrites[writeBase + 3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compositeWrites[writeBase + 3].dstSet = compositeDescriptorSets_[frameIndex];
            compositeWrites[writeBase + 3].dstBinding = 3;
            compositeWrites[writeBase + 3].descriptorCount = 1;
            compositeWrites[writeBase + 3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            compositeWrites[writeBase + 3].pBufferInfo = &compositeExposureInfos[frameIndex];

            compositeWrites[writeBase + 4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            compositeWrites[writeBase + 4].dstSet = compositeDescriptorSets_[frameIndex];
            compositeWrites[writeBase + 4].dstBinding = 4;
            compositeWrites[writeBase + 4].descriptorCount = 1;
            compositeWrites[writeBase + 4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            compositeWrites[writeBase + 4].pImageInfo = &compositeImageInfos[frameIndex][3];

            rhi::debug::setObjectName(context_.vkDevice(),
                                      compositeDescriptorSets_[frameIndex],
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                      "CompositeDescriptorSet" + std::to_string(frameIndex));
        }

        vkUpdateDescriptorSets(context_.vkDevice(),
                               static_cast<uint32_t>(compositeWrites.size()),
                               compositeWrites.data(),
                               0,
                               nullptr);

        if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE && taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
            for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                auto& taaDescriptorSets = taaCompositeDescriptorSets_[historyIndex];
                taaDescriptorSets.assign(frames_.size(), VK_NULL_HANDLE);
                std::vector<VkDescriptorSetLayout> taaCompositeLayouts(
                    frames_.size(), postProcessCompositeDescriptorSetLayout_.handle());
                VkDescriptorSetAllocateInfo taaCompositeAllocateInfo{};
                taaCompositeAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                taaCompositeAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
                taaCompositeAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaDescriptorSets.size());
                taaCompositeAllocateInfo.pSetLayouts = taaCompositeLayouts.data();
                VK_CHECK(vkAllocateDescriptorSets(
                    context_.vkDevice(), &taaCompositeAllocateInfo, taaDescriptorSets.data()));

                std::vector<std::array<VkDescriptorImageInfo, 4>> taaCompositeImageInfos(frames_.size());
                std::vector<VkDescriptorBufferInfo> taaCompositeExposureInfos(frames_.size());
                std::vector<VkWriteDescriptorSet> taaCompositeWrites(frames_.size() * 5u);
                const VkDescriptorImageInfo taaCompositeDepthInfo = depthInfo();
                for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
                    taaCompositeImageInfos[frameIndex][0] =
                        imageInfo(taaHistoryImages_[historyIndex].imageView());
                    taaCompositeImageInfos[frameIndex][1] = imageInfo(bloomPong_.imageView());
                    taaCompositeImageInfos[frameIndex][2] = imageInfo(mipBloomView);
                    taaCompositeImageInfos[frameIndex][3] = taaCompositeDepthInfo;

                    taaCompositeExposureInfos[frameIndex].buffer = frameExposureBuffers_[frameIndex].buffer();
                    taaCompositeExposureInfos[frameIndex].offset = 0;
                    taaCompositeExposureInfos[frameIndex].range = sizeof(ExposureState);

                    const size_t writeBase = frameIndex * 5u;
                    for (uint32_t binding = 0; binding < 3; ++binding) {
                        taaCompositeWrites[writeBase + binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaCompositeWrites[writeBase + binding].dstSet = taaDescriptorSets[frameIndex];
                        taaCompositeWrites[writeBase + binding].dstBinding = binding;
                        taaCompositeWrites[writeBase + binding].descriptorCount = 1;
                        taaCompositeWrites[writeBase + binding].descriptorType =
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        taaCompositeWrites[writeBase + binding].pImageInfo =
                            &taaCompositeImageInfos[frameIndex][binding];
                    }

                    taaCompositeWrites[writeBase + 3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    taaCompositeWrites[writeBase + 3].dstSet = taaDescriptorSets[frameIndex];
                    taaCompositeWrites[writeBase + 3].dstBinding = 3;
                    taaCompositeWrites[writeBase + 3].descriptorCount = 1;
                    taaCompositeWrites[writeBase + 3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    taaCompositeWrites[writeBase + 3].pBufferInfo = &taaCompositeExposureInfos[frameIndex];

                    taaCompositeWrites[writeBase + 4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    taaCompositeWrites[writeBase + 4].dstSet = taaDescriptorSets[frameIndex];
                    taaCompositeWrites[writeBase + 4].dstBinding = 4;
                    taaCompositeWrites[writeBase + 4].descriptorCount = 1;
                    taaCompositeWrites[writeBase + 4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    taaCompositeWrites[writeBase + 4].pImageInfo = &taaCompositeImageInfos[frameIndex][3];

                    rhi::debug::setObjectName(context_.vkDevice(),
                                              taaDescriptorSets[frameIndex],
                                              VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                              "TAACompositeDescriptorSet" + std::to_string(historyIndex) + "_" +
                                                  std::to_string(frameIndex));
                }

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(taaCompositeWrites.size()),
                                       taaCompositeWrites.data(),
                                       0,
                                       nullptr);
            }
        }
    }

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

            if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE &&
                taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
                for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                    auto& taaDescriptorSets = taaLuminanceDescriptorSets_[historyIndex];
                    taaDescriptorSets.assign(frames_.size(), VK_NULL_HANDLE);
                    std::vector<VkDescriptorSetLayout> taaLuminanceLayouts(
                        frames_.size(), postProcessLuminanceDescriptorSetLayout_.handle());
                    VkDescriptorSetAllocateInfo taaLuminanceAllocateInfo{};
                    taaLuminanceAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    taaLuminanceAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
                    taaLuminanceAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaDescriptorSets.size());
                    taaLuminanceAllocateInfo.pSetLayouts = taaLuminanceLayouts.data();
                    VK_CHECK(vkAllocateDescriptorSets(
                        context_.vkDevice(), &taaLuminanceAllocateInfo, taaDescriptorSets.data()));

                    for (size_t frameIndex = 0; frameIndex < taaDescriptorSets.size(); ++frameIndex) {
                        VkDescriptorImageInfo taaSceneColorInfo =
                            imageInfo(taaHistoryImages_[historyIndex].imageView());
                        VkDescriptorBufferInfo luminanceBufferInfo{};
                        luminanceBufferInfo.buffer = frameLuminanceBuffers_[frameIndex].buffer();
                        luminanceBufferInfo.offset = 0;
                        luminanceBufferInfo.range = frameLuminanceBuffers_[frameIndex].size();

                        std::array<VkWriteDescriptorSet, 2> taaLuminanceWrites{};
                        taaLuminanceWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaLuminanceWrites[0].dstSet = taaDescriptorSets[frameIndex];
                        taaLuminanceWrites[0].dstBinding = 0;
                        taaLuminanceWrites[0].descriptorCount = 1;
                        taaLuminanceWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        taaLuminanceWrites[0].pImageInfo = &taaSceneColorInfo;

                        taaLuminanceWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaLuminanceWrites[1].dstSet = taaDescriptorSets[frameIndex];
                        taaLuminanceWrites[1].dstBinding = 1;
                        taaLuminanceWrites[1].descriptorCount = 1;
                        taaLuminanceWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        taaLuminanceWrites[1].pBufferInfo = &luminanceBufferInfo;

                        vkUpdateDescriptorSets(context_.vkDevice(),
                                               static_cast<uint32_t>(taaLuminanceWrites.size()),
                                               taaLuminanceWrites.data(),
                                               0,
                                               nullptr);
                        rhi::debug::setObjectName(context_.vkDevice(),
                                                  taaDescriptorSets[frameIndex],
                                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                                  "TAALuminanceDescriptorSet" + std::to_string(historyIndex) + "_" +
                                                      std::to_string(frameIndex));
                    }
                }
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

            if (taaHistoryImages_[0].imageView() != VK_NULL_HANDLE &&
                taaHistoryImages_[1].imageView() != VK_NULL_HANDLE) {
                for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                    auto& taaDescriptorSets = taaHistogramDescriptorSets_[historyIndex];
                    taaDescriptorSets.assign(frames_.size(), VK_NULL_HANDLE);
                    std::vector<VkDescriptorSetLayout> taaHistogramLayouts(
                        frames_.size(), postProcessLuminanceDescriptorSetLayout_.handle());
                    VkDescriptorSetAllocateInfo taaHistogramAllocateInfo{};
                    taaHistogramAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    taaHistogramAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
                    taaHistogramAllocateInfo.descriptorSetCount = static_cast<uint32_t>(taaDescriptorSets.size());
                    taaHistogramAllocateInfo.pSetLayouts = taaHistogramLayouts.data();
                    VK_CHECK(vkAllocateDescriptorSets(
                        context_.vkDevice(), &taaHistogramAllocateInfo, taaDescriptorSets.data()));

                    for (size_t frameIndex = 0; frameIndex < taaDescriptorSets.size(); ++frameIndex) {
                        VkDescriptorImageInfo taaSceneColorInfo =
                            imageInfo(taaHistoryImages_[historyIndex].imageView());
                        VkDescriptorBufferInfo histogramBufferInfo{};
                        histogramBufferInfo.buffer = frameHistogramBuffers_[frameIndex].buffer();
                        histogramBufferInfo.offset = 0;
                        histogramBufferInfo.range = frameHistogramBuffers_[frameIndex].size();

                        std::array<VkWriteDescriptorSet, 2> taaHistogramWrites{};
                        taaHistogramWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaHistogramWrites[0].dstSet = taaDescriptorSets[frameIndex];
                        taaHistogramWrites[0].dstBinding = 0;
                        taaHistogramWrites[0].descriptorCount = 1;
                        taaHistogramWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        taaHistogramWrites[0].pImageInfo = &taaSceneColorInfo;

                        taaHistogramWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        taaHistogramWrites[1].dstSet = taaDescriptorSets[frameIndex];
                        taaHistogramWrites[1].dstBinding = 1;
                        taaHistogramWrites[1].descriptorCount = 1;
                        taaHistogramWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        taaHistogramWrites[1].pBufferInfo = &histogramBufferInfo;

                        vkUpdateDescriptorSets(context_.vkDevice(),
                                               static_cast<uint32_t>(taaHistogramWrites.size()),
                                               taaHistogramWrites.data(),
                                               0,
                                               nullptr);
                        rhi::debug::setObjectName(context_.vkDevice(),
                                                  taaDescriptorSets[frameIndex],
                                                  VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                                  "TAAHistogramDescriptorSet" + std::to_string(historyIndex) + "_" +
                                                      std::to_string(frameIndex));
                    }
                }
            }
        } catch (const std::exception& error) {
            histogramDescriptorSets_.clear();
            disableHistogramExposureFallback(
                std::string("Histogram exposure descriptor allocation failed: ") + error.what());
        }
    }

    if (createExposureReduceDescriptors) {
        try {
            exposureReduceDescriptorSets_.assign(frames_.size(), VK_NULL_HANDLE);
            std::vector<VkDescriptorSetLayout> exposureLayouts(
                frames_.size(), postProcessExposureReduceDescriptorSetLayout_.handle());
            VkDescriptorSetAllocateInfo exposureAllocateInfo{};
            exposureAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            exposureAllocateInfo.descriptorPool = postProcessDescriptorPool_.handle();
            exposureAllocateInfo.descriptorSetCount = static_cast<uint32_t>(exposureReduceDescriptorSets_.size());
            exposureAllocateInfo.pSetLayouts = exposureLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(
                context_.vkDevice(), &exposureAllocateInfo, exposureReduceDescriptorSets_.data()));

            for (size_t frameIndex = 0; frameIndex < exposureReduceDescriptorSets_.size(); ++frameIndex) {
                std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
                bufferInfos[0].buffer = frameLuminanceBuffers_[frameIndex].buffer();
                bufferInfos[0].range = frameLuminanceBuffers_[frameIndex].size();
                bufferInfos[1].buffer = frameHistogramBuffers_[frameIndex].buffer();
                bufferInfos[1].range = frameHistogramBuffers_[frameIndex].size();
                bufferInfos[2].buffer = frameExposureBuffers_[frameIndex].buffer();
                bufferInfos[2].range = sizeof(ExposureState);

                std::array<VkWriteDescriptorSet, 3> exposureWrites{};
                for (uint32_t binding = 0; binding < exposureWrites.size(); ++binding) {
                    exposureWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    exposureWrites[binding].dstSet = exposureReduceDescriptorSets_[frameIndex];
                    exposureWrites[binding].dstBinding = binding;
                    exposureWrites[binding].descriptorCount = 1;
                    exposureWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    exposureWrites[binding].pBufferInfo = &bufferInfos[binding];
                }

                vkUpdateDescriptorSets(context_.vkDevice(),
                                       static_cast<uint32_t>(exposureWrites.size()),
                                       exposureWrites.data(),
                                       0,
                                       nullptr);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          exposureReduceDescriptorSets_[frameIndex],
                                          VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                          "ExposureReduceDescriptorSet" + std::to_string(frameIndex));
            }
        } catch (const std::exception& error) {
            exposureReduceDescriptorSets_.clear();
            disableAutoExposureFallback(std::string("GPU exposure reduce descriptor allocation failed: ") +
                                        error.what());
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
    frameLuminanceReadbackBuffers_.clear();
    frameLuminanceReadbackReady_.clear();

    for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo luminanceInfo{};
        luminanceInfo.size = luminanceBufferSize;
        luminanceInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        luminanceInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameLuminanceBuffers_[frameIndex].createBuffer(context_, luminanceInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameLuminanceBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminancePartialBuffer" + std::to_string(frameIndex));
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
    frameHistogramReadbackBuffers_.clear();
    frameHistogramReadbackReady_.clear();

    for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo histogramInfo{};
        histogramInfo.size = histogramBufferSize;
        histogramInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        histogramInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        frameHistogramBuffers_[frameIndex].createBuffer(context_, histogramInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameHistogramBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "LuminanceHistogramBuffer" + std::to_string(frameIndex));
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

void Renderer::createExposureResources()
{
    destroyExposureResources();

    if (frames_.empty()) {
        return;
    }

    frameExposureBuffers_.resize(frames_.size());
    frameExposureReadbackReady_.assign(frames_.size(), 0);

    const ExposureMode mode =
        exposureModeValue(toneMappingSettings_.enableAutoExposure ? toneMappingSettings_.exposureMode
                                                                  : static_cast<int>(ExposureMode::Manual));
    const ExposureState initialState{
        toneMappingExposureValue(toneMappingSettings_.enableAutoExposure ? currentExposure_
                                                                         : toneMappingSettings_.manualExposure),
        averageLuminance_,
        histogramClippedLuminance_,
        static_cast<uint32_t>(mode),
    };

    for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
        rhi::VulkanBufferCreateInfo exposureInfo{};
        exposureInfo.size = sizeof(ExposureState);
        exposureInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        exposureInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        exposureInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        frameExposureBuffers_[frameIndex].createBuffer(context_, exposureInfo);
        frameExposureBuffers_[frameIndex].upload(
            std::as_bytes(std::span<const ExposureState>(&initialState, 1)));
        rhi::debug::setObjectName(context_.vkDevice(),
                                  frameExposureBuffers_[frameIndex].buffer(),
                                  VK_OBJECT_TYPE_BUFFER,
                                  "ExposureStateBuffer" + std::to_string(frameIndex));
    }

    exposureReduceAvailable_ = true;
}

void Renderer::destroyExposureResources()
{
    exposureReduceAvailable_ = false;
    exposureReduceDescriptorSets_.clear();
    frameExposureReadbackReady_.clear();
    frameExposureBuffers_.clear();
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
    exposureReducePipeline_.reset();
    exposureReduceAvailable_ = false;
}

void Renderer::disableLogAverageExposureFallback(std::string_view reason)
{
    if (!logAverageExposureWarningLogged_) {
        Logger::warn(std::string(reason) + "; log-average exposure fallback unavailable.");
        logAverageExposureWarningLogged_ = true;
    }

    destroyLuminanceResources();
    luminancePipeline_.reset();
    exposureReducePipeline_.reset();
    exposureReduceAvailable_ = false;
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
    exposureReducePipeline_.reset();
    exposureReduceAvailable_ = false;
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

        if (depthPyramid_.image() == VK_NULL_HANDLE || depthPyramidSampler_ == VK_NULL_HANDLE) {
            throw std::runtime_error("GPU culling requires a valid depth pyramid descriptor resource.");
        }

        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
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

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

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

        frameGpuCullParamBuffers_.resize(frames_.size());
        for (size_t frameIndex = 0; frameIndex < frameGpuCullParamBuffers_.size(); ++frameIndex) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = static_cast<VkDeviceSize>(sizeof(GpuCullFrameParams));
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            frameGpuCullParamBuffers_[frameIndex].createBuffer(context_, bufferInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameGpuCullParamBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuCullFrameParamsBuffer" + std::to_string(frameIndex));
        }

        frameBatchVisibleCountBuffers_.resize(frames_.size());
        frameBatchVisibleCountReadbackBuffers_.resize(frames_.size());
        frameGpuCullTotalDrawItems_.assign(frames_.size(), 0);
        frameGpuCullBatchCounts_.assign(frames_.size(), 0);
        frameGpuCullReadbackReady_.assign(frames_.size(), 0);
        frameGpuCullIndirectCountPath_.assign(frames_.size(), 0);
        for (size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
            rhi::VulkanBufferCreateInfo gpuCountInfo{};
            gpuCountInfo.size = kGpuCullCountBufferSize;
            gpuCountInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            gpuCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            frameBatchVisibleCountBuffers_[frameIndex].createBuffer(context_, gpuCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameBatchVisibleCountBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuBatchVisibleCountBuffer" + std::to_string(frameIndex));

            rhi::VulkanBufferCreateInfo readbackCountInfo{};
            readbackCountInfo.size = kGpuCullCountBufferSize;
            readbackCountInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            readbackCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            readbackCountInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            frameBatchVisibleCountReadbackBuffers_[frameIndex].createBuffer(context_, readbackCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameBatchVisibleCountReadbackBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuBatchVisibleCountReadbackBuffer" + std::to_string(frameIndex));
        }

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(frames_.size() * 4);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(frames_.size());

        gpuCullDescriptorPool_.create(context_.vkDevice(),
                                      std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()),
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
            visibleCountBufferInfo.range = kGpuCullCountBufferSize;

            VkDescriptorImageInfo depthPyramidInfo{};
            depthPyramidInfo.sampler = depthPyramidSampler_;
            depthPyramidInfo.imageView = depthPyramid_.imageView();
            depthPyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo frameParamsBufferInfo{};
            frameParamsBufferInfo.buffer = frameGpuCullParamBuffers_[frameIndex].buffer();
            frameParamsBufferInfo.offset = 0;
            frameParamsBufferInfo.range = frameGpuCullParamBuffers_[frameIndex].size();

            std::array<VkWriteDescriptorSet, 5> writes{};
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

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = gpuCullDescriptorSets_[frameIndex];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].pImageInfo = &depthPyramidInfo;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = gpuCullDescriptorSets_[frameIndex];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[4].pBufferInfo = &frameParamsBufferInfo;

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
    frameGpuCullParamBuffers_.clear();
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
            gpuCountInfo.size = kGpuCullCountBufferSize;
            gpuCountInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            gpuCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            frameShadowBatchVisibleCountBuffers_[frameIndex].createBuffer(context_, gpuCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameShadowBatchVisibleCountBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuShadowBatchVisibleCountBuffer" + std::to_string(frameIndex));

            rhi::VulkanBufferCreateInfo readbackCountInfo{};
            readbackCountInfo.size = kGpuCullCountBufferSize;
            readbackCountInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            readbackCountInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            readbackCountInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            frameShadowBatchVisibleCountReadbackBuffers_[frameIndex].createBuffer(context_, readbackCountInfo);
            rhi::debug::setObjectName(context_.vkDevice(),
                                      frameShadowBatchVisibleCountReadbackBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "GpuShadowBatchVisibleCountReadbackBuffer" + std::to_string(frameIndex));
        }

        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(frames_.size() * 4);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(frames_.size());

        shadowCullDescriptorPool_.create(context_.vkDevice(),
                                         std::span<const VkDescriptorPoolSize>(poolSizes.data(), poolSizes.size()),
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
            visibleCountBufferInfo.range = kGpuCullCountBufferSize;

            VkDescriptorImageInfo depthPyramidInfo{};
            depthPyramidInfo.sampler = depthPyramidSampler_;
            depthPyramidInfo.imageView = depthPyramid_.imageView();
            depthPyramidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo frameParamsBufferInfo{};
            frameParamsBufferInfo.buffer = frameGpuCullParamBuffers_[frameIndex].buffer();
            frameParamsBufferInfo.offset = 0;
            frameParamsBufferInfo.range = frameGpuCullParamBuffers_[frameIndex].size();

            std::array<VkWriteDescriptorSet, 5> writes{};
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

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = shadowCullDescriptorSets_[frameIndex];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].pImageInfo = &depthPyramidInfo;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = shadowCullDescriptorSets_[frameIndex];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[4].pBufferInfo = &frameParamsBufferInfo;

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
    imguiLayer_.clearRenderTargetPreviewDescriptors();
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
    const VkDescriptorSetLayout postProcessDualImageDescriptorSetLayout =
        postProcessDualImageDescriptorSetLayout_.handle();
    const VkDescriptorSetLayout postProcessCompositeDescriptorSetLayout =
        postProcessCompositeDescriptorSetLayout_.handle();
    const VkDescriptorSetLayout postProcessExposureReduceDescriptorSetLayout =
        postProcessExposureReduceDescriptorSetLayout_.handle();
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(PushConstants))};
    const VkPushConstantRange skyboxPushConstantRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      0,
                                                      static_cast<uint32_t>(sizeof(SkyboxPushConstants))};
    const VkPushConstantRange bloomExtractPushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomExtractPushConstants))};
    const VkPushConstantRange bloomBlurPushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomBlurPushConstants))};
    const VkPushConstantRange bloomDownsamplePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomDownsamplePushConstants))};
    const VkPushConstantRange bloomUpsamplePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(BloomUpsamplePushConstants))};
    const VkPushConstantRange taaResolvePushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, static_cast<uint32_t>(sizeof(TaaResolvePushConstants))};
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

    rhi::VulkanPipelineCreateInfo bloomDownsamplePipelineInfo{};
    bloomDownsamplePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomDownsamplePipelineInfo.fragmentShaderPath = shaderPath("bloom_downsample.frag.spv");
    bloomDownsamplePipelineInfo.colorFormat = kBloomColorFormat;
    bloomDownsamplePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessSingleImageDescriptorSetLayout, 1);
    bloomDownsamplePipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&bloomDownsamplePushConstantRange, 1);

    bloomDownsamplePipeline_.create(context_.vkDevice(), bloomDownsamplePipelineInfo);
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomDownsamplePipeline_.pipeline(),
                              VK_OBJECT_TYPE_PIPELINE,
                              "BloomDownsamplePipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomDownsamplePipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "BloomDownsamplePipelineLayout");
    bloomDownsamplePipelineColorFormat_ = bloomDownsamplePipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo bloomUpsamplePipelineInfo{};
    bloomUpsamplePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    bloomUpsamplePipelineInfo.fragmentShaderPath = shaderPath("bloom_upsample.frag.spv");
    bloomUpsamplePipelineInfo.colorFormat = kBloomColorFormat;
    bloomUpsamplePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessDualImageDescriptorSetLayout, 1);
    bloomUpsamplePipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&bloomUpsamplePushConstantRange, 1);

    bloomUpsamplePipeline_.create(context_.vkDevice(), bloomUpsamplePipelineInfo);
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomUpsamplePipeline_.pipeline(),
                              VK_OBJECT_TYPE_PIPELINE,
                              "BloomUpsamplePipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              bloomUpsamplePipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "BloomUpsamplePipelineLayout");
    bloomUpsamplePipelineColorFormat_ = bloomUpsamplePipelineInfo.colorFormat;

    rhi::VulkanPipelineCreateInfo taaResolvePipelineInfo{};
    taaResolvePipelineInfo.vertexShaderPath = shaderPath("fullscreen.vert.spv");
    taaResolvePipelineInfo.fragmentShaderPath = shaderPath("taa_resolve.frag.spv");
    taaResolvePipelineInfo.colorFormat = kSceneColorFormat;
    taaResolvePipelineInfo.descriptorSetLayouts =
        std::span<const VkDescriptorSetLayout>(&postProcessDualImageDescriptorSetLayout, 1);
    taaResolvePipelineInfo.pushConstantRanges =
        std::span<const VkPushConstantRange>(&taaResolvePushConstantRange, 1);

    taaResolvePipeline_.create(context_.vkDevice(), taaResolvePipelineInfo);
    rhi::debug::setObjectName(
        context_.vkDevice(), taaResolvePipeline_.pipeline(), VK_OBJECT_TYPE_PIPELINE, "TAAResolvePipeline");
    rhi::debug::setObjectName(context_.vkDevice(),
                              taaResolvePipeline_.layout(),
                              VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "TAAResolvePipelineLayout");
    taaResolvePipelineColorFormat_ = taaResolvePipelineInfo.colorFormat;

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
    exposureReducePipeline_.reset();
    depthPyramidPipeline_.reset();
    if (depthPyramidDescriptorSetLayout_.handle() != VK_NULL_HANDLE) {
        const VkDescriptorSetLayout depthPyramidDescriptorSetLayout = depthPyramidDescriptorSetLayout_.handle();
        const VkPushConstantRange depthPyramidPushConstantRange{
            VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(DepthPyramidPushConstants))};

        rhi::VulkanComputePipelineCreateInfo depthPyramidPipelineInfo{};
        depthPyramidPipelineInfo.shaderPath = shaderPath("depth_pyramid.comp.spv");
        depthPyramidPipelineInfo.descriptorSetLayouts =
            std::span<const VkDescriptorSetLayout>(&depthPyramidDescriptorSetLayout, 1);
        depthPyramidPipelineInfo.pushConstantRanges =
            std::span<const VkPushConstantRange>(&depthPyramidPushConstantRange, 1);
        depthPyramidPipeline_.create(context_.vkDevice(), depthPyramidPipelineInfo);
        rhi::debug::setObjectName(context_.vkDevice(),
                                  depthPyramidPipeline_.pipeline(),
                                  VK_OBJECT_TYPE_PIPELINE,
                                  "DepthPyramidComputePipeline");
        rhi::debug::setObjectName(context_.vkDevice(),
                                  depthPyramidPipeline_.layout(),
                                  VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                  "DepthPyramidPipelineLayout");
    }
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

        if (postProcessExposureReduceDescriptorSetLayout != VK_NULL_HANDLE) {
            try {
                const VkPushConstantRange exposureReducePushConstantRange{
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(sizeof(ExposureReducePushConstants))};

                rhi::VulkanComputePipelineCreateInfo exposureReducePipelineInfo{};
                exposureReducePipelineInfo.shaderPath = shaderPath("exposure_reduce.comp.spv");
                exposureReducePipelineInfo.descriptorSetLayouts =
                    std::span<const VkDescriptorSetLayout>(&postProcessExposureReduceDescriptorSetLayout, 1);
                exposureReducePipelineInfo.pushConstantRanges =
                    std::span<const VkPushConstantRange>(&exposureReducePushConstantRange, 1);
                exposureReducePipeline_.create(context_.vkDevice(), exposureReducePipelineInfo);
                rhi::debug::setObjectName(context_.vkDevice(),
                                          exposureReducePipeline_.pipeline(),
                                          VK_OBJECT_TYPE_PIPELINE,
                                          "ExposureReduceComputePipeline");
                rhi::debug::setObjectName(context_.vkDevice(),
                                          exposureReducePipeline_.layout(),
                                          VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                          "ExposureReducePipelineLayout");
            } catch (const std::exception& error) {
                disableAutoExposureFallback(std::string("GPU exposure reduce compute pipeline creation failed: ") +
                                            error.what());
            }
        }
    }
}

uint32_t Renderer::allocateRenderObjectDebugId()
{
    const uint32_t debugId = nextRenderObjectDebugId_;
    ++nextRenderObjectDebugId_;
    if (nextRenderObjectDebugId_ == 0) {
        nextRenderObjectDebugId_ = 1;
    }
    return debugId;
}

void Renderer::createScene()
{
    imguiLayer_.clearTexturePreviewDescriptors();
    renderObjects_.clear();
    selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
    nextRenderObjectDebugId_ = 1;
    occlusionTestSceneActive_ = false;
    occlusionTestSceneStatus_ = "Occlusion test scene not loaded.";
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
    materialAssetTextures_.clear();

    cubeMesh_ = renderer::Mesh::createCube(context_, commandContext_);
    portfolioSphereMesh_ = renderer::Mesh::createUvSphere(context_, commandContext_);
    createCheckerboardTexture();
    createPortfolioBaseColorTexture();
    createPortfolioBackdropTexture();
    createNormalTexture();
    createMetallicRoughnessTexture();
    createEnvironmentMap();
    createMaterial();

    camera_ = defaultCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;

    const auto addCube = [this](std::string debugName,
                                const renderer::Material* material,
                                const glm::vec3& position,
                                const glm::vec3& rotationRadians,
                                const glm::vec3& scale,
                                bool animateTransform = true,
                                bool portfolioOnly = false,
                                bool hideInPortfolio = false,
                                renderer::RenderObjectSourceType sourceType =
                                    renderer::RenderObjectSourceType::BuiltInFallbackCube) {
        renderer::RenderObject cube{};
        cube.debugId = allocateRenderObjectDebugId();
        cube.sceneObjectId = cube.debugId;
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.sourceType = sourceType;
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = animateTransform;
        cube.portfolioOnly = portfolioOnly;
        cube.hideInPortfolio = hideInPortfolio;
        renderObjects_.push_back(std::move(cube));
    };

    const auto addCubeFallbackScene = [&addCube, this]() {
        renderObjects_.reserve(4);
        addCube("Center Cube",
                &materialVariants_.at(0),
                {0.0f, -0.1f, 0.0f},
                {0.2f, 0.0f, 0.0f},
                {0.7f, 0.7f, 0.7f},
                true,
                false,
                true);
        addCube(
            "Left Cube",
            &materialVariants_.at(1),
            {-1.35f, -0.15f, -0.35f},
            {0.0f, 0.35f, 0.2f},
            {0.5f, 0.5f, 0.5f},
            true,
            false,
            true);
        addCube("Right Cube",
                &materialVariants_.at(2),
                {1.35f, -0.05f, -0.25f},
                {0.25f, -0.35f, 0.0f},
                {0.55f, 0.8f, 0.55f},
                true,
                false,
                true);
        addCube("Elevated Cube",
                &materialVariants_.at(3),
                {0.0f, 1.0f, -0.7f},
                {-0.3f, 0.2f, 0.45f},
                {0.45f, 0.45f, 0.45f},
                true,
                false,
                true);
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

            renderObjects_.reserve(loadedAsset.nodeMeshInstances.size() + 8);
            for (const renderer::GltfNodeMeshInstance& instance : loadedAsset.nodeMeshInstances) {
                if (instance.meshIndex >= importedMeshes_.size() || !importedMeshes_[instance.meshIndex].valid()) {
                    Logger::warn("Skipping imported glTF RenderObject with invalid mesh index " +
                                 std::to_string(instance.meshIndex) + ".");
                    continue;
                }

                renderer::RenderObject importedObject{};
                importedObject.debugId = allocateRenderObjectDebugId();
                importedObject.sceneObjectId = importedObject.debugId;
                importedObject.mesh = &importedMeshes_[instance.meshIndex];
                importedObject.material =
                    importedMaterials_.empty() ? &materialVariants_.at(0) : &importedMaterials_.front();
                if (!importedMaterials_.empty()) {
                    importedObject.materialTable = importedMaterials_.data();
                    importedObject.materialCount = importedMaterials_.size();
                }
                importedObject.debugName = instance.debugName.empty() ? "Imported glTF Node" : instance.debugName;
                importedObject.sourceType = renderer::RenderObjectSourceType::ImportedGltf;
                importedObject.transform = renderer::Transform::fromMatrix(instance.transform);
                importedObject.hideInPortfolio = true;
                renderObjects_.push_back(std::move(importedObject));
            }

            if (renderObjects_.empty()) {
                throw std::runtime_error("Loaded glTF asset did not produce any valid RenderObjects.");
            }

            Logger::info("Loaded glTF scene: " + modelPath.string() + " with " +
                         std::to_string(importedMeshes_.size()) + " mesh slot(s), " +
                         std::to_string(renderObjects_.size()) + " render object(s), and " +
                         std::to_string(importedMaterials_.size()) + " material(s).");
            addPortfolioShowcaseObjects();
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load glTF mesh '" + modelPath.string() + "': " + error.what());
        }
    }

    Logger::warn("No supported glTF mesh asset loaded; using built-in cube fallback scene.");
    addCubeFallbackScene();
    addPortfolioShowcaseObjects();
}

void Renderer::addPortfolioShowcaseObjects()
{
    if (hasPortfolioShowcaseScene()) {
        return;
    }
    if (materialVariants_.size() <= kPortfolioBackdropMaterialIndex) {
        Logger::warn("Portfolio showcase scene was not added because portfolio materials are unavailable.");
        return;
    }

    const auto addPortfolioCube = [this](std::string debugName,
                                         const renderer::Material* material,
                                         const glm::vec3& position,
                                         const glm::vec3& rotationRadians,
                                         const glm::vec3& scale) {
        renderer::RenderObject cube{};
        cube.debugId = allocateRenderObjectDebugId();
        cube.sceneObjectId = cube.debugId;
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.sourceType = renderer::RenderObjectSourceType::PortfolioShowcase;
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = false;
        cube.portfolioOnly = true;
        renderObjects_.push_back(std::move(cube));
    };

    const auto addPortfolioSphere = [this](std::string debugName,
                                           const renderer::Material* material,
                                           const glm::vec3& position,
                                           const glm::vec3& scale) {
        renderer::RenderObject sphere{};
        sphere.debugId = allocateRenderObjectDebugId();
        sphere.sceneObjectId = sphere.debugId;
        sphere.mesh = &portfolioSphereMesh_;
        sphere.material = material;
        sphere.debugName = std::move(debugName);
        sphere.sourceType = renderer::RenderObjectSourceType::PortfolioShowcase;
        sphere.transform.position = position;
        sphere.transform.scale = scale;
        sphere.animateTransform = false;
        sphere.portfolioOnly = true;
        renderObjects_.push_back(std::move(sphere));
    };

    renderObjects_.reserve(renderObjects_.size() + 8);
    addPortfolioCube("Portfolio Studio Floor",
                     &materialVariants_.at(kPortfolioGroundMaterialIndex),
                     {0.0f, -0.56f, 0.24f},
                     {0.0f, 0.0f, 0.0f},
                     {11.0f, 0.08f, 6.4f});
    addPortfolioCube("Portfolio Studio Backdrop",
                     &materialVariants_.at(kPortfolioBackdropMaterialIndex),
                     {0.0f, 2.08f, -2.82f},
                     {0.0f, 0.0f, 0.0f},
                     {60.0f, 9.0f, 0.08f});
    addPortfolioCube("Portfolio Side Plinth",
                     &materialVariants_.at(kPortfolioGroundMaterialIndex),
                     {1.02f, -0.42f, -0.18f},
                     {0.0f, -0.16f, 0.0f},
                     {0.96f, 0.28f, 0.70f});
    addPortfolioSphere("Portfolio Hero Ceramic",
                       &materialVariants_.at(kPortfolioHeroCeramicMaterialIndex),
                       {0.0f, -0.11f, 0.08f},
                       {0.82f, 0.82f, 0.82f});
    addPortfolioSphere("Portfolio Matte Gray",
                       &materialVariants_.at(kPortfolioMatteGrayMaterialIndex),
                       {-0.92f, -0.24f, 0.06f},
                       {0.56f, 0.56f, 0.56f});
    addPortfolioSphere("Portfolio Glossy Blue",
                       &materialVariants_.at(kPortfolioGlossyBlueMaterialIndex),
                       {-0.62f, -0.33f, 0.66f},
                       {0.38f, 0.38f, 0.38f});
    addPortfolioSphere("Portfolio Rough Metal",
                       &materialVariants_.at(kPortfolioRoughMetalMaterialIndex),
                       {0.96f, -0.29f, 0.42f},
                       {0.46f, 0.46f, 0.46f});
    addPortfolioSphere("Portfolio Polished Metal Small",
                       &materialVariants_.at(kPortfolioPolishedMetalSmallMaterialIndex),
                       {1.04f, -0.09f, -0.18f},
                       {0.38f, 0.38f, 0.38f});
}

void Renderer::resetPortfolioShowcaseObjectsToPreset()
{
    const auto resetObject = [this](std::string_view debugName,
                                    const glm::vec3& position,
                                    const glm::vec3& rotationRadians,
                                    const glm::vec3& scale) {
        for (renderer::RenderObject& object : renderObjects_) {
            if (object.sourceType != renderer::RenderObjectSourceType::PortfolioShowcase ||
                object.debugName != debugName) {
                continue;
            }

            object.transform.position = position;
            object.transform.rotationRadians = rotationRadians;
            object.transform.scale = scale;
            object.transform.matrixOverride = glm::mat4{1.0f};
            object.transform.useMatrixOverride = false;
            object.visible = true;
            object.animateTransform = false;
            object.portfolioOnly = true;
            object.hideInPortfolio = false;
            return;
        }
    };

    resetObject("Portfolio Studio Floor", {0.0f, -0.56f, 0.24f}, {0.0f, 0.0f, 0.0f}, {11.0f, 0.08f, 6.4f});
    resetObject("Portfolio Studio Backdrop", {0.0f, 2.08f, -2.82f}, {0.0f, 0.0f, 0.0f}, {60.0f, 9.0f, 0.08f});
    resetObject("Portfolio Side Plinth", {1.02f, -0.42f, -0.18f}, {0.0f, -0.16f, 0.0f}, {0.96f, 0.28f, 0.70f});
    resetObject("Portfolio Hero Ceramic", {0.0f, -0.11f, 0.08f}, {0.0f, 0.0f, 0.0f}, {0.82f, 0.82f, 0.82f});
    resetObject("Portfolio Matte Gray", {-0.92f, -0.24f, 0.06f}, {0.0f, 0.0f, 0.0f}, {0.56f, 0.56f, 0.56f});
    resetObject("Portfolio Glossy Blue", {-0.62f, -0.33f, 0.66f}, {0.0f, 0.0f, 0.0f}, {0.38f, 0.38f, 0.38f});
    resetObject("Portfolio Rough Metal", {0.96f, -0.29f, 0.42f}, {0.0f, 0.0f, 0.0f}, {0.46f, 0.46f, 0.46f});
    resetObject("Portfolio Polished Metal Small", {1.04f, -0.09f, -0.18f}, {0.0f, 0.0f, 0.0f}, {0.38f, 0.38f, 0.38f});
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

bool Renderer::hasPortfolioShowcaseScene() const
{
    bool hasFloor = false;
    bool hasBackdrop = false;
    bool hasHero = false;
    size_t materialSampleCount = 0;

    for (const renderer::RenderObject& object : renderObjects_) {
        if (object.sourceType != renderer::RenderObjectSourceType::PortfolioShowcase || !object.mesh ||
            !object.mesh->valid() || !object.material) {
            continue;
        }

        if (object.debugName == "Portfolio Studio Floor") {
            hasFloor = true;
            continue;
        }
        if (object.debugName == "Portfolio Studio Backdrop") {
            hasBackdrop = true;
            continue;
        }
        if (object.material->debugName == "Portfolio_HeroCeramic") {
            hasHero = true;
            continue;
        }
        if (object.material->debugName == "Portfolio_MatteGray" ||
            object.material->debugName == "Portfolio_GlossyBlue" ||
            object.material->debugName == "Portfolio_RoughMetal" ||
            object.material->debugName == "Portfolio_PolishedMetalSmall") {
            ++materialSampleCount;
        }
    }

    return hasFloor && hasBackdrop && hasHero && materialSampleCount >= 4;
}

bool Renderer::currentFrameHasPortfolioShowcaseDrawItems() const
{
    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex >= renderObjects_.size()) {
            continue;
        }

        const renderer::RenderObject& object = renderObjects_[drawItem.objectIndex];
        if (object.sourceType == renderer::RenderObjectSourceType::PortfolioShowcase && object.portfolioOnly &&
            !object.hideInPortfolio) {
            return true;
        }
    }

    return false;
}

bool Renderer::ensurePortfolioShowcaseSceneReady()
{
    if (hasPortfolioShowcaseScene()) {
        return true;
    }

    if (!cubeMesh_.valid() || !portfolioSphereMesh_.valid() ||
        materialVariants_.size() <= kPortfolioBackdropMaterialIndex) {
        portfolioScreenshotStatus_ =
            "Portfolio showcase scene is unavailable: required meshes or materials are not initialized.";
        Logger::warn(portfolioScreenshotStatus_);
        return false;
    }

    Logger::warn("Portfolio showcase scene was missing; rebuilding portfolio-only showcase objects.");
    addPortfolioShowcaseObjects();
    if (hasPortfolioShowcaseScene()) {
        return true;
    }

    portfolioScreenshotStatus_ = "Portfolio showcase scene is unavailable after rebuild.";
    Logger::warn(portfolioScreenshotStatus_);
    return false;
}

void Renderer::addOcclusionTestSceneObjects()
{
    if (!cubeMesh_.valid() || materialVariants_.empty()) {
        occlusionTestSceneStatus_ =
            "Occlusion test scene is unavailable: cube mesh or runtime materials are not initialized.";
        Logger::warn(occlusionTestSceneStatus_);
        return;
    }

    const auto materialAt = [this](size_t materialIndex) -> const renderer::Material* {
        return &materialVariants_.at(materialIndex % materialVariants_.size());
    };

    const auto addCube = [this](std::string debugName,
                                const renderer::Material* material,
                                const glm::vec3& position,
                                const glm::vec3& rotationRadians,
                                const glm::vec3& scale) {
        renderer::RenderObject cube{};
        cube.debugId = allocateRenderObjectDebugId();
        cube.sceneObjectId = cube.debugId;
        cube.mesh = &cubeMesh_;
        cube.material = material;
        cube.debugName = std::move(debugName);
        cube.sourceType = renderer::RenderObjectSourceType::OcclusionTest;
        cube.transform.position = position;
        cube.transform.rotationRadians = rotationRadians;
        cube.transform.scale = scale;
        cube.animateTransform = false;
        cube.portfolioOnly = false;
        cube.hideInPortfolio = true;
        renderObjects_.push_back(std::move(cube));
    };

    renderObjects_.reserve(renderObjects_.size() + static_cast<size_t>(kOcclusionTestObjectCount));

    const size_t groundMaterial = materialVariants_.size() > kPortfolioGroundMaterialIndex
                                      ? kPortfolioGroundMaterialIndex
                                      : 0;
    addCube("Occlusion Test Ground",
            materialAt(groundMaterial),
            {0.0f, -0.10f, -4.0f},
            {0.0f, 0.0f, 0.0f},
            {13.0f, 0.12f, 22.0f});

    const std::array<float, kOcclusionTestOccluderCount> occluderX = {-2.6f, -1.3f, 0.0f, 1.3f, 2.6f};
    for (size_t occluderIndex = 0; occluderIndex < occluderX.size(); ++occluderIndex) {
        std::ostringstream name;
        name << "Occlusion Test Wall " << (occluderIndex + 1);
        const float zOffset = (occluderIndex % 2 == 0) ? 0.15f : -0.08f;
        addCube(name.str(),
                materialAt(occluderIndex),
                {occluderX[occluderIndex], 1.35f, 0.35f + zOffset},
                {0.0f, 0.0f, 0.0f},
                {0.98f, 3.10f, 0.55f});
    }

    for (int row = 0; row < kOcclusionTestGridRows; ++row) {
        for (int column = 0; column < kOcclusionTestGridColumns; ++column) {
            const float x = -5.5f + static_cast<float>(column) * 1.0f;
            const float z = -2.4f - static_cast<float>(row) * 1.05f;
            const bool topWitnessRow = row == kOcclusionTestGridRows - 1;
            const bool sideWitnessColumn = column == 0 || column == kOcclusionTestGridColumns - 1;
            const float y = topWitnessRow ? 3.05f : 0.22f + 0.34f * static_cast<float>((row + column) % 3);
            const float uniformScale = sideWitnessColumn ? 0.40f : 0.32f + 0.035f * static_cast<float>((row + column) % 4);

            std::ostringstream name;
            name << "Occlusion Test Hidden Cube r" << row << " c" << column;
            if (topWitnessRow || sideWitnessColumn) {
                name << " visible-edge";
            }

            addCube(name.str(),
                    materialAt(static_cast<size_t>((row + column) % 4)),
                    {x, y, z},
                    {0.0f, 0.15f * static_cast<float>((row + column) % 5), 0.0f},
                    {uniformScale, uniformScale, uniformScale});
        }
    }
}

void Renderer::resetOcclusionTestSceneToPreset()
{
    const auto firstRemoved = std::remove_if(renderObjects_.begin(), renderObjects_.end(), [](const auto& object) {
        return object.sourceType == renderer::RenderObjectSourceType::OcclusionTest;
    });

    if (firstRemoved != renderObjects_.end()) {
        const size_t firstRemovedIndex = static_cast<size_t>(firstRemoved - renderObjects_.begin());
        if (selectedRenderObjectIndex_ >= firstRemovedIndex) {
            selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
        }
        renderObjects_.erase(firstRemoved, renderObjects_.end());
    }

    addOcclusionTestSceneObjects();
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

bool Renderer::hasOcclusionTestScene() const
{
    size_t occlusionObjectCount = 0;
    bool hasGround = false;
    bool hasWall = false;
    bool hasHiddenObject = false;

    for (const renderer::RenderObject& object : renderObjects_) {
        if (object.sourceType != renderer::RenderObjectSourceType::OcclusionTest || !object.mesh ||
            !object.mesh->valid() || !object.material) {
            continue;
        }

        ++occlusionObjectCount;
        if (object.debugName == "Occlusion Test Ground") {
            hasGround = true;
        } else if (object.debugName.find("Occlusion Test Wall") == 0) {
            hasWall = true;
        } else if (object.debugName.find("Occlusion Test Hidden Cube") == 0) {
            hasHiddenObject = true;
        }
    }

    return hasGround && hasWall && hasHiddenObject &&
           occlusionObjectCount >= static_cast<size_t>(kOcclusionTestObjectCount);
}

void Renderer::createCheckerboardTexture()
{
    const std::filesystem::path texturePath = assetPath("textures/checker.png");
    if (std::filesystem::exists(texturePath)) {
        try {
            checkerboardTexture_.createFromFile(
                context_, commandContext_, texturePath, rhi::TextureColorSpace::SRGB, true);
            checkerboardTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
                "Checkerboard base color",
                texturePath.string(),
                rhi::TextureColorSpace::SRGB,
                rhi::TextureDebugSource::LoadedFromDisk,
                false,
            });
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
    checkerboardTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
        "Procedural checkerboard base color",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        true,
    });
    nameTextureResources(checkerboardTexture_, "BaseColorTexture");
    Logger::info("Created procedural checkerboard base color texture as sRGB.");
}

void Renderer::createPortfolioBaseColorTexture()
{
    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 255;
        pixels[offset + 1] = 255;
        pixels[offset + 2] = 255;
        pixels[offset + 3] = 255;
    }

    portfolioBaseColorTexture_.createFromRgba8(context_,
                                               commandContext_,
                                               width,
                                               height,
                                               std::span<const uint8_t>(pixels.data(), pixels.size()),
                                               VK_FORMAT_R8G8B8A8_SRGB,
                                               false);
    portfolioBaseColorTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
        "Portfolio solid base color",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        false,
    });
    nameTextureResources(portfolioBaseColorTexture_, "PortfolioBaseColorTexture");
}

void Renderer::createPortfolioBackdropTexture()
{
    constexpr uint32_t width = 16;
    constexpr uint32_t height = 64;
    std::array<uint8_t, width * height * 4> pixels{};
    const glm::vec3 bottom{132.0f, 144.0f, 154.0f};
    const glm::vec3 top{78.0f, 94.0f, 112.0f};

    for (uint32_t y = 0; y < height; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(height - 1);
        const glm::vec3 color = glm::mix(bottom, top, t);
        for (uint32_t x = 0; x < width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * 4U;
            pixels[offset + 0] = static_cast<uint8_t>(std::clamp(color.r, 0.0f, 255.0f));
            pixels[offset + 1] = static_cast<uint8_t>(std::clamp(color.g, 0.0f, 255.0f));
            pixels[offset + 2] = static_cast<uint8_t>(std::clamp(color.b, 0.0f, 255.0f));
            pixels[offset + 3] = 255;
        }
    }

    portfolioBackdropTexture_.createFromRgba8(context_,
                                              commandContext_,
                                              width,
                                              height,
                                              std::span<const uint8_t>(pixels.data(), pixels.size()),
                                              VK_FORMAT_R8G8B8A8_SRGB,
                                              false);
    portfolioBackdropTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
        "Portfolio studio backdrop gradient",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        false,
    });
    nameTextureResources(portfolioBackdropTexture_, "PortfolioBackdropTexture");
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
            normalMapTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
                "Checker normal map",
                texturePath.string(),
                rhi::TextureColorSpace::Linear,
                rhi::TextureDebugSource::LoadedFromDisk,
                false,
            });
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
        normalMapTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
            "Procedural flat normal map",
            {},
            rhi::TextureColorSpace::Linear,
            rhi::TextureDebugSource::ProceduralFallback,
            true,
        });
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
    flatNormalTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
        "Flat normal fallback",
        {},
        rhi::TextureColorSpace::Linear,
        rhi::TextureDebugSource::ProceduralFallback,
        true,
    });
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
            metallicRoughnessTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
                "Checker metallic-roughness map",
                texturePath.string(),
                rhi::TextureColorSpace::Linear,
                rhi::TextureDebugSource::LoadedFromDisk,
                false,
            });
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
        metallicRoughnessTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
            "Procedural neutral metallic-roughness map",
            {},
            rhi::TextureColorSpace::Linear,
            rhi::TextureDebugSource::ProceduralFallback,
            true,
        });
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
    neutralMetallicRoughnessTexture_.setDebugMetadata(rhi::TextureDebugMetadata{
        "Neutral metallic-roughness fallback",
        {},
        rhi::TextureColorSpace::Linear,
        rhi::TextureDebugSource::ProceduralFallback,
        true,
    });
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

const rhi::VulkanTexture* Renderer::loadMaterialAssetTextureOrFallback(
    const std::filesystem::path& materialPath,
    const std::filesystem::path& texturePath,
    rhi::TextureColorSpace colorSpace,
    std::string_view slotName,
    const rhi::VulkanTexture& fallbackTexture,
    bool& fallbackUsed)
{
    fallbackUsed = false;
    if (texturePath.empty()) {
        return &fallbackTexture;
    }

    const std::filesystem::path resolvedTexturePath = resolveMaterialTexturePath(materialPath, texturePath);
    if (!std::filesystem::exists(resolvedTexturePath)) {
        fallbackUsed = true;
        Logger::warn("Material asset texture is missing for " + std::string(slotName) +
                     "; using fallback texture: " + resolvedTexturePath.string());
        return &fallbackTexture;
    }

    auto texture = std::make_unique<rhi::VulkanTexture>();
    try {
        texture->createFromFile(context_, commandContext_, resolvedTexturePath, colorSpace, true);
        texture->setDebugMetadata(rhi::TextureDebugMetadata{
            "Material asset " + std::string(slotName) + " texture",
            resolvedTexturePath.string(),
            colorSpace,
            rhi::TextureDebugSource::LoadedFromDisk,
            false,
        });
        nameTextureResources(*texture, "MaterialAssetTexture_" + std::string(slotName));
        (void)assetManager_.registerTextureAsset(resolvedTexturePath, std::string(slotName));
        const rhi::VulkanTexture* texturePointer = texture.get();
        materialAssetTextures_.push_back(std::move(texture));
        Logger::info("Loaded material asset " + std::string(slotName) + " texture as " +
                     std::string(colorSpaceName(colorSpace)) + ": " + resolvedTexturePath.string());
        return texturePointer;
    } catch (const std::exception& error) {
        fallbackUsed = true;
        Logger::warn("Failed to load material asset " + std::string(slotName) + " texture '" +
                     resolvedTexturePath.string() + "'; using fallback texture: " + error.what());
        return &fallbackTexture;
    }
}

renderer::Material Renderer::createMaterialFromAsset(const assets::MaterialAsset& materialAsset,
                                                     const rhi::VulkanTexture& baseColorFallback,
                                                     const rhi::VulkanTexture& normalFallback,
                                                     const rhi::VulkanTexture& metallicRoughnessFallback,
                                                     float multiScatterStrength,
                                                     renderer::MaterialSource fallbackSource)
{
    bool baseColorLoadFallback = false;
    bool normalLoadFallback = false;
    bool metallicRoughnessLoadFallback = false;

    renderer::Material material{};
    material.debugName = materialAsset.name.empty() ? "Material Asset" : materialAsset.name;
    material.assetName = materialAsset.name;
    material.sourceAssetPath = materialAsset.sourcePath;
    material.shader = materialAsset.shader.empty() ? "pbr_opaque" : materialAsset.shader;
    material.baseColorTexturePath = materialAsset.textures.baseColor;
    material.normalTexturePath = materialAsset.textures.normal;
    material.metallicRoughnessTexturePath = materialAsset.textures.metallicRoughness;
    material.alphaMode = materialAsset.alphaMode.empty() ? "OPAQUE" : materialAsset.alphaMode;
    material.baseColorTexture = loadMaterialAssetTextureOrFallback(materialAsset.sourcePath,
                                                                   materialAsset.textures.baseColor,
                                                                   rhi::TextureColorSpace::SRGB,
                                                                   "base color",
                                                                   baseColorFallback,
                                                                   baseColorLoadFallback);
    material.normalTexture = loadMaterialAssetTextureOrFallback(materialAsset.sourcePath,
                                                                materialAsset.textures.normal,
                                                                rhi::TextureColorSpace::Linear,
                                                                "normal",
                                                                normalFallback,
                                                                normalLoadFallback);
    material.metallicRoughnessTexture =
        loadMaterialAssetTextureOrFallback(materialAsset.sourcePath,
                                           materialAsset.textures.metallicRoughness,
                                           rhi::TextureColorSpace::Linear,
                                           "metallic-roughness",
                                           metallicRoughnessFallback,
                                           metallicRoughnessLoadFallback);
    material.baseColorFactor = materialAsset.baseColorFactor;
    material.metallic = std::clamp(materialAsset.metallicFactor, 0.0f, 1.0f);
    material.roughness = std::clamp(materialAsset.roughnessFactor, 0.0f, 1.0f);
    material.multiScatterStrength = multiScatterStrength;
    material.alphaCutoff = std::max(materialAsset.alphaCutoff, 0.0f);
    material.doubleSided = materialAsset.doubleSided;
    material.source = materialAsset.fallback ? fallbackSource : renderer::MaterialSource::MaterialAsset;
    material.baseColorTextureFallback =
        baseColorLoadFallback || (material.baseColorTexture && material.baseColorTexture->debugMetadata().fallback);
    material.normalTextureFallback =
        normalLoadFallback || (material.normalTexture && material.normalTexture->debugMetadata().fallback);
    material.metallicRoughnessTextureFallback =
        metallicRoughnessLoadFallback ||
        (material.metallicRoughnessTexture && material.metallicRoughnessTexture->debugMetadata().fallback);
    material.hasNormalMap = !material.normalTextureFallback;
    material.hasMetallicRoughnessMap = !material.metallicRoughnessTextureFallback;

    assignBindlessTextureIndices(material);
    createMaterialDescriptorSet(material);
    return material;
}

assets::MaterialAsset Renderer::runtimeMaterialToAsset(const renderer::Material& material) const
{
    assets::MaterialAsset materialAsset{};
    materialAsset.sourcePath = material.sourceAssetPath;
    materialAsset.name = !material.assetName.empty() ? material.assetName : material.debugName;
    if (materialAsset.name.empty()) {
        materialAsset.name = "Material";
    }
    materialAsset.shader = material.shader.empty() ? "pbr_opaque" : material.shader;
    materialAsset.baseColorFactor = material.baseColorFactor;
    materialAsset.metallicFactor = material.metallic;
    materialAsset.roughnessFactor = material.roughness;
    materialAsset.textures.baseColor = material.baseColorTexturePath;
    materialAsset.textures.normal = material.normalTexturePath;
    materialAsset.textures.metallicRoughness = material.metallicRoughnessTexturePath;
    materialAsset.alphaMode = material.alphaMode.empty() ? "OPAQUE" : material.alphaMode;
    materialAsset.alphaCutoff = material.alphaCutoff;
    materialAsset.doubleSided = material.doubleSided;
    materialAsset.fallback = false;
    return materialAsset;
}

bool Renderer::saveMaterialAssetFromUi(renderer::Material& material)
{
    if (material.sourceAssetPath.empty()) {
        lastMaterialAssetStatus_ = "Save Material skipped: selected material has no source asset path. Save As is TODO.";
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    std::string errorMessage;
    assets::MaterialAsset materialAsset = runtimeMaterialToAsset(material);
    if (!assetManager_.saveMaterialAsset(material.sourceAssetPath, materialAsset, &errorMessage)) {
        lastMaterialAssetStatus_ = errorMessage;
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    material.source = renderer::MaterialSource::MaterialAsset;
    material.assetName = materialAsset.name;
    lastMaterialAssetStatus_ = "Saved material asset: " + material.sourceAssetPath.string();
    Logger::info(lastMaterialAssetStatus_);
    invalidateTaaHistory();
    return true;
}

bool Renderer::reloadMaterialAssetFromUi(renderer::Material& material)
{
    if (material.sourceAssetPath.empty()) {
        lastMaterialAssetStatus_ = "Reload Material skipped: selected material has no source asset path.";
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    std::string errorMessage;
    const assets::MaterialAssetHandle handle = assetManager_.loadMaterialAsset(material.sourceAssetPath, &errorMessage);
    if (!handle) {
        lastMaterialAssetStatus_ = errorMessage;
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    const assets::MaterialAsset* materialAsset = assetManager_.materialAsset(handle);
    if (!materialAsset) {
        lastMaterialAssetStatus_ = "Reload Material failed: loaded material asset was not registered.";
        Logger::warn(lastMaterialAssetStatus_);
        return false;
    }

    material.debugName = materialAsset->name.empty() ? material.debugName : materialAsset->name;
    material.assetName = materialAsset->name;
    material.shader = materialAsset->shader.empty() ? "pbr_opaque" : materialAsset->shader;
    material.baseColorFactor = materialAsset->baseColorFactor;
    material.metallic = std::clamp(materialAsset->metallicFactor, 0.0f, 1.0f);
    material.roughness = std::clamp(materialAsset->roughnessFactor, 0.0f, 1.0f);
    material.baseColorTexturePath = materialAsset->textures.baseColor;
    material.normalTexturePath = materialAsset->textures.normal;
    material.metallicRoughnessTexturePath = materialAsset->textures.metallicRoughness;
    material.alphaMode = materialAsset->alphaMode.empty() ? "OPAQUE" : materialAsset->alphaMode;
    material.alphaCutoff = std::max(materialAsset->alphaCutoff, 0.0f);
    material.doubleSided = materialAsset->doubleSided;
    material.source = renderer::MaterialSource::MaterialAsset;

    lastMaterialAssetStatus_ =
        "Reloaded material scalar/metadata fields from " + material.sourceAssetPath.string() +
        ". Texture rebinding is not hot-reloaded in Phase 3.";
    Logger::info(lastMaterialAssetStatus_);
    invalidateTaaHistory();
    return true;
}

void Renderer::createMaterial()
{
    materialVariants_.clear();
    materialVariants_.reserve(11);

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
        material.source = renderer::MaterialSource::BuiltIn;
        material.hasNormalMap = normalMapAssetLoaded_;
        material.hasMetallicRoughnessMap = metallicRoughnessMapAssetLoaded_;
        material.baseColorTextureFallback = checkerboardTexture_.debugMetadata().fallback;
        material.normalTextureFallback = !normalMapAssetLoaded_;
        material.metallicRoughnessTextureFallback = !metallicRoughnessMapAssetLoaded_;
        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        materialVariants_.push_back(std::move(material));
    };

    addMaterial("Checkerboard Matte", {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.75f, 0.0f);
    addMaterial("Checkerboard Warm Semi-Metal", {1.0f, 0.82f, 0.65f, 1.0f}, 0.35f, 0.38f, 0.5f);
    addMaterial("Checkerboard Cool Rough Metal", {0.72f, 0.84f, 1.0f, 1.0f}, 0.85f, 0.62f, 1.0f);
    addMaterial("Checkerboard Glossy Dielectric", {0.9f, 1.0f, 0.78f, 1.0f}, 0.0f, 0.18f, 0.25f);

    const auto addPortfolioMaterial = [this](std::string debugName,
                                             const rhi::VulkanTexture* baseColorTexture,
                                             const glm::vec4& baseColorFactor,
                                             float metallic,
                                             float roughness,
                                             float multiScatterStrength) {
        renderer::Material material{};
        material.debugName = std::move(debugName);
        material.assetName = material.debugName;
        material.shader = "pbr_opaque";
        material.alphaMode = "OPAQUE";
        material.baseColorTexture = baseColorTexture;
        material.normalTexture = &flatNormalTexture_;
        material.metallicRoughnessTexture = &neutralMetallicRoughnessTexture_;
        material.baseColorFactor = baseColorFactor;
        material.metallic = metallic;
        material.roughness = roughness;
        material.multiScatterStrength = multiScatterStrength;
        material.alphaCutoff = 0.5f;
        material.source = renderer::MaterialSource::BuiltIn;
        material.hasNormalMap = false;
        material.hasMetallicRoughnessMap = false;
        material.doubleSided = false;
        material.baseColorTextureFallback = false;
        material.normalTextureFallback = true;
        material.metallicRoughnessTextureFallback = true;
        assignBindlessTextureIndices(material);
        createMaterialDescriptorSet(material);
        materialVariants_.push_back(std::move(material));
    };

    const auto portfolioMaterialAssetOrFallback =
        [this](std::string_view filename,
               std::string debugName,
               const glm::vec4& baseColorFactor,
               float metallic,
               float roughness) {
            const std::filesystem::path path = materialAssetPath(filename);
            std::string errorMessage;
            const assets::MaterialAssetHandle handle = assetManager_.loadMaterialAsset(path, &errorMessage);
            if (handle) {
                const assets::MaterialAsset* materialAsset = assetManager_.materialAsset(handle);
                if (materialAsset) {
                    Logger::info("Loaded portfolio material asset: " + path.string());
                    return *materialAsset;
                }
            }

            Logger::warn(errorMessage.empty() ? "Portfolio material asset failed to load; using fallback values: " +
                                                    path.string()
                                              : errorMessage + "; using fallback values.");
            assets::MaterialAsset fallback = assets::AssetManager::fallbackMaterialAsset(std::move(debugName));
            fallback.sourcePath = path;
            fallback.baseColorFactor = baseColorFactor;
            fallback.metallicFactor = metallic;
            fallback.roughnessFactor = roughness;
            return fallback;
        };

    const auto addPortfolioMaterialAsset =
        [this, &portfolioMaterialAssetOrFallback](std::string_view filename,
                                                  std::string debugName,
                                                  const glm::vec4& baseColorFactor,
                                                  float metallic,
                                                  float roughness,
                                                  float multiScatterStrength) {
            const assets::MaterialAsset materialAsset =
                portfolioMaterialAssetOrFallback(filename, debugName, baseColorFactor, metallic, roughness);
            renderer::Material material = createMaterialFromAsset(materialAsset,
                                                                  portfolioBaseColorTexture_,
                                                                  flatNormalTexture_,
                                                                  neutralMetallicRoughnessTexture_,
                                                                  multiScatterStrength,
                                                                  renderer::MaterialSource::Fallback);
            materialVariants_.push_back(std::move(material));
        };

    addPortfolioMaterial(
        "Portfolio_Ground", &portfolioBaseColorTexture_, {0.30f, 0.32f, 0.32f, 1.0f}, 0.0f, 0.86f, 0.0f);
    addPortfolioMaterialAsset("portfolio_matte_gray.material.json",
                              "Portfolio_MatteGray",
                              {0.66f, 0.66f, 0.62f, 1.0f},
                              0.0f,
                              0.85f,
                              0.0f);
    addPortfolioMaterialAsset("portfolio_glossy_blue.material.json",
                              "Portfolio_GlossyBlue",
                              {0.18f, 0.43f, 0.88f, 1.0f},
                              0.0f,
                              0.30f,
                              0.2f);
    addPortfolioMaterialAsset("portfolio_rough_metal.material.json",
                              "Portfolio_RoughMetal",
                              {0.76f, 0.74f, 0.70f, 1.0f},
                              1.0f,
                              0.60f,
                              0.70f);
    addPortfolioMaterialAsset("portfolio_polished_metal_small.material.json",
                              "Portfolio_PolishedMetalSmall",
                              {0.82f, 0.85f, 0.88f, 1.0f},
                              1.0f,
                              0.23f,
                              0.40f);
    addPortfolioMaterialAsset("portfolio_hero_ceramic.material.json",
                              "Portfolio_HeroCeramic",
                              {0.66f, 0.72f, 0.76f, 1.0f},
                              0.0f,
                              0.55f,
                              0.05f);
    addPortfolioMaterial(
        "Portfolio_Backdrop", &portfolioBackdropTexture_, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.94f, 0.0f);

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

    // Decode the needed textures on worker threads, then upload them here on the
    // device-owning thread. PNG/JPEG decoding is the expensive part and is safe to
    // parallelize; the Vulkan uploads stay serial on this thread.
    struct PendingTextureUpload {
        size_t textureIndex = 0;
        rhi::TextureColorSpace colorSpace = rhi::TextureColorSpace::Linear;
        std::string_view slotName;
        std::string debugPrefix;
        std::vector<rhi::VulkanTexture>* textures = nullptr;
        const renderer::GltfTextureInfo* info = nullptr;
        std::future<rhi::DecodedImage> decode;
    };

    std::vector<PendingTextureUpload> pendingUploads;

    const auto enqueueDecode = [this, &textureInfos, &pendingUploads](size_t textureIndex,
                                                                      rhi::TextureColorSpace colorSpace,
                                                                      std::string_view slotName,
                                                                      std::string_view debugPrefix,
                                                                      std::vector<rhi::VulkanTexture>& textures) {
        const renderer::GltfTextureInfo& textureInfo = textureInfos[textureIndex];
        if (textureInfo.path.empty() && textureInfo.encodedData.empty()) {
            return;
        }
        if (!textureInfo.path.empty() && !std::filesystem::exists(textureInfo.path)) {
            Logger::warn("glTF texture image is missing; material fallback will be used: " +
                         textureInfo.path.string());
            return;
        }

        const renderer::GltfTextureInfo* infoPtr = &textureInfo;
        std::future<rhi::DecodedImage> decode = jobSystem_.enqueue([infoPtr]() -> rhi::DecodedImage {
            if (!infoPtr->path.empty()) {
                return rhi::VulkanTexture::decodeImageFile(infoPtr->path);
            }
            return rhi::VulkanTexture::decodeImageBytes(
                std::span<const uint8_t>(infoPtr->encodedData.data(), infoPtr->encodedData.size()));
        });

        pendingUploads.push_back(PendingTextureUpload{
            textureIndex, colorSpace, slotName, std::string(debugPrefix), &textures, infoPtr, std::move(decode)});
    };

    for (size_t textureIndex = 0; textureIndex < textureInfos.size(); ++textureIndex) {
        if (baseColorNeeded[textureIndex] != 0) {
            enqueueDecode(textureIndex,
                          rhi::TextureColorSpace::SRGB,
                          "base color",
                          "GltfBaseColorTexture",
                          importedBaseColorTextures_);
        }
        if (normalNeeded[textureIndex] != 0) {
            enqueueDecode(
                textureIndex, rhi::TextureColorSpace::Linear, "normal", "GltfNormalTexture", importedNormalTextures_);
        }
        if (metallicRoughnessNeeded[textureIndex] != 0) {
            enqueueDecode(textureIndex,
                          rhi::TextureColorSpace::Linear,
                          "metallic-roughness",
                          "GltfMetallicRoughnessTexture",
                          importedMetallicRoughnessTextures_);
        }
    }

    for (PendingTextureUpload& pending : pendingUploads) {
        std::vector<rhi::VulkanTexture>& textures = *pending.textures;
        const renderer::GltfTextureInfo& textureInfo = *pending.info;
        try {
            const rhi::DecodedImage decoded = pending.decode.get();
            textures[pending.textureIndex].createFromRgba8(context_,
                                                           commandContext_,
                                                           decoded.width,
                                                           decoded.height,
                                                           decoded.pixels,
                                                           rhi::rgba8FormatForColorSpace(pending.colorSpace),
                                                           true);
            if (!textureInfo.path.empty()) {
                Logger::info("Loaded glTF " + std::string(pending.slotName) + " texture as " +
                             std::string(colorSpaceName(pending.colorSpace)) + ": " + textureInfo.path.string());
            } else {
                Logger::info("Loaded embedded glTF " + std::string(pending.slotName) + " texture as " +
                             std::string(colorSpaceName(pending.colorSpace)) + ": " + textureInfo.debugName);
            }

            textures[pending.textureIndex].setDebugMetadata(rhi::TextureDebugMetadata{
                textureInfo.debugName.empty() ? pending.debugPrefix + std::to_string(pending.textureIndex)
                                              : textureInfo.debugName,
                textureInfo.path.empty() ? std::string{} : textureInfo.path.string(),
                pending.colorSpace,
                textureInfo.embedded ? rhi::TextureDebugSource::GltfEmbeddedData
                                     : rhi::TextureDebugSource::GltfExternalFile,
                false,
            });
            nameTextureResources(textures[pending.textureIndex],
                                 pending.debugPrefix + std::to_string(pending.textureIndex));
        } catch (const std::exception& error) {
            const std::string textureName =
                !textureInfo.path.empty() ? textureInfo.path.string() : textureInfo.debugName;
            Logger::warn("Failed to load glTF " + std::string(pending.slotName) + " texture '" + textureName +
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
        defaultMaterial.fallback = true;
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
        material.source =
            materialInfo.fallback ? renderer::MaterialSource::Fallback : renderer::MaterialSource::Gltf;
        material.hasNormalMap = textureLoaded(materialInfo.normalTextureIndex, importedNormalTextures_);
        material.hasMetallicRoughnessMap =
            textureLoaded(materialInfo.metallicRoughnessTextureIndex, importedMetallicRoughnessTextures_);
        material.baseColorTextureFallback = !textureLoaded(materialInfo.baseColorTextureIndex, importedBaseColorTextures_);
        material.normalTextureFallback = !material.hasNormalMap;
        material.metallicRoughnessTextureFallback = !material.hasMetallicRoughnessMap;

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

glm::vec4 Renderer::activeDirectionalLightDirection() const
{
    if (portfolioCaptureMode_) {
        return kPortfolioLightDirection;
    }

    const glm::vec3 direction =
        normalizedOrFallback(directionalLightSettings_.direction,
                             glm::normalize(glm::vec3{kDirectionalLightDirection.x,
                                                      kDirectionalLightDirection.y,
                                                      kDirectionalLightDirection.z}));
    return glm::vec4(direction, 0.0f);
}

glm::vec4 Renderer::activeDirectionalLightColor() const
{
    if (portfolioCaptureMode_) {
        return kPortfolioLightColor;
    }

    const float intensity = std::max(directionalLightSettings_.intensity, 0.0f);
    return glm::vec4(glm::max(directionalLightSettings_.color, glm::vec3{0.0f}) * intensity, 1.0f);
}

void Renderer::updateCascades(float aspectRatio)
{
    // The cascade fitting math is GPU-independent and lives in CascadeMath.h so
    // it can be unit-tested. Gather the inputs from renderer state, run the pure
    // solver, then cache the results plus their packed frustum planes.
    const glm::vec4 activeLightDirection = activeDirectionalLightDirection();

    renderer::CascadeBuildInput cascadeInput{};
    cascadeInput.requestedCascadeCount = csmSettings_.cascadeCount;
    cascadeInput.nearPlane = csmSettings_.nearPlane;
    cascadeInput.farPlane = csmSettings_.farPlane;
    cascadeInput.shadowDistance = csmSettings_.shadowDistance;
    cascadeInput.lambda = csmSettings_.lambda;
    cascadeInput.enableTexelSnapping = csmSettings_.enableTexelSnapping;
    cascadeInput.shadowResolution = shadowSettings_.resolution;
    cascadeInput.cameraPosition = camera_.position;
    cascadeInput.cameraTarget = camera_.target;
    cascadeInput.cameraUp = camera_.up;
    cascadeInput.cameraVerticalFovRadians = camera_.verticalFovRadians;
    cascadeInput.lightDirection =
        glm::vec3{activeLightDirection.x, activeLightDirection.y, activeLightDirection.z};
    cascadeInput.aspectRatio = aspectRatio;

    const renderer::CascadeBuildOutput cascadeOutput = renderer::computeShadowCascades(cascadeInput);

    frameCascades_ = cascadeOutput.cascades;
    frameCascadeSplits_ = cascadeOutput.splitDistances;

    for (uint32_t cascadeIndex = 0; cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
        const renderer::Frustum& lightFrustum = frameCascades_[cascadeIndex].lightFrustum;
        for (size_t planeIndex = 0; planeIndex < frameShadowCascadeFrustumPlanes_[cascadeIndex].size(); ++planeIndex) {
            const renderer::FrustumPlane& lightPlane = lightFrustum.planes[planeIndex];
            frameShadowCascadeFrustumPlanes_[cascadeIndex][planeIndex] =
                glm::vec4(lightPlane.normal, lightPlane.distance);
        }
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

bool Renderer::isRenderObjectActive(const renderer::RenderObject& object) const
{
    if (!object.visible) {
        return false;
    }
    if (object.sourceType == renderer::RenderObjectSourceType::OcclusionTest) {
        return occlusionTestSceneActive_ && !portfolioCaptureMode_;
    }
    if (occlusionTestSceneActive_ && !portfolioCaptureMode_) {
        return false;
    }
    if (portfolioCaptureMode_ && object.hideInPortfolio) {
        return false;
    }
    return !object.portfolioOnly || portfolioCaptureMode_;
}

bool Renderer::appendDrawItemsForObject(uint32_t objectIndex, std::vector<DrawItem>& drawItems) const
{
    if (objectIndex >= renderObjects_.size()) {
        return true;
    }

    const renderer::RenderObject& object = renderObjects_[objectIndex];
    if (!isRenderObjectActive(object)) {
        return true;
    }

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
        if (!isRenderObjectActive(object)) {
            continue;
        }
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
        if (!isRenderObjectActive(object)) {
            continue;
        }
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
    if (frameIndex >= frameCullInputBuffers_.size() || frameIndex >= frameGpuCullParamBuffers_.size()) {
        throw std::runtime_error("GPU cull input buffer frame index is out of range.");
    }

    const bool occlusionEnabledThisFrame = isGpuOcclusionCullingActive() && previousFrameDepthValidForOcclusion();
    const VkExtent2D extent = swapchain_.extent();

    GpuCullFrameParams frameParams{};
    frameParams.occlusionViewProjection = depthPyramidViewProjection_;
    frameParams.cameraPosition = glm::vec4(frameCameraPosition_, 0.0f);
    frameParams.viewportAndMipCount = glm::vec4(static_cast<float>(extent.width),
                                                static_cast<float>(extent.height),
                                                static_cast<float>(depthPyramidMipLevels_),
                                                0.0f);
    frameParams.occlusionSettings = glm::vec4(gpuOcclusionDepthBias_,
                                              gpuOcclusionNearDisableDistance_,
                                              gpuOcclusionMaxScreenCoverage_,
                                              gpuOcclusionMinScreenPixels_);
    frameParams.counterAndFlags = glm::uvec4(kGpuCullStatsCounterOffset, occlusionEnabledThisFrame ? 1u : 0u, 0u, 0u);
    frameGpuCullParamBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullFrameParams>(&frameParams, 1)));

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
    if (frameIndex >= frameShadowCullInputBuffers_.size() || frameIndex >= frameGpuCullParamBuffers_.size()) {
        throw std::runtime_error("GPU shadow cull input buffer frame index is out of range.");
    }

    GpuCullFrameParams frameParams{};
    frameParams.counterAndFlags = glm::uvec4(kGpuCullStatsCounterOffset, 0u, 0u, 0u);
    frameGpuCullParamBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullFrameParams>(&frameParams, 1)));

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
    renderer::GpuProfiler::FrameResults results{};
    if (!gpuProfiler_.readFrame(frameIndex, results) || !results.valid) {
        return;
    }
    latestGpuProfilerResults_ = results;
    pushGpuTimingSample(results);

    const auto now = std::chrono::steady_clock::now();
    if (now - lastGpuTimingPrint_ < std::chrono::seconds(1)) {
        return;
    }

    lastGpuTimingPrint_ = now;

    std::ostringstream message;
    message << std::fixed << std::setprecision(3) << "GPU timings:\n"
            << "  Frame total: " << results.totalGpuTimeMs << " ms\n"
            << "  timestamp queries: " << results.queryCount << "/" << results.maxQueryCount << "\n";
    if (results.queryLimitExceeded) {
        message << "  warning: timestamp query capacity was exceeded\n";
    }
    for (const renderer::GpuProfiler::ScopeResult& scope : results.scopes) {
        message << "  " << scope.name << ": " << scope.elapsedMs << " ms\n";
    }
    if (cullingStats_.gpuCulling) {
        const uint32_t totalDrawItems = frameIndex < frameGpuCullTotalDrawItems_.size()
                                            ? frameGpuCullTotalDrawItems_[frameIndex]
                                            : static_cast<uint32_t>(cullingStats_.totalDrawItems);
        const uint32_t batchCount = frameIndex < frameGpuCullBatchCounts_.size()
                                        ? frameGpuCullBatchCounts_[frameIndex]
                                        : static_cast<uint32_t>(cullingStats_.batchCount);
        uint32_t visibleDrawItems = 0;
        GpuCullCounters counters{};
        if (readGpuCullCounters(frameIndex, counters)) {
            const uint32_t culledDrawItems =
                counters.totalDrawItems > counters.visibleDrawItems
                    ? counters.totalDrawItems - counters.visibleDrawItems
                    : 0;
            message << "GPU culling:\n"
                    << "  total draw items: " << counters.totalDrawItems << "\n"
                    << "  visible draw items: " << counters.visibleDrawItems << "\n"
                    << "  frustum culled draw items: " << counters.frustumCulledDrawItems << "\n"
                    << "  occlusion culled draw items: " << counters.occlusionCulledDrawItems << "\n"
                    << "  total culled draw items: " << culledDrawItems << "\n"
                    << "  depth pyramid mips: " << depthPyramidMipLevels_ << "\n"
                    << "  occlusion culling: " << (isGpuOcclusionCullingActive() ? "enabled" : "disabled") << "\n"
                    << "  batches: " << batchCount << "\n"
                    << "  indirect count path: "
                    << (isFrameIndirectCountPathActive(frameIndex) ? "enabled" : "disabled");
        } else if (readGpuVisibleCount(frameIndex, visibleDrawItems)) {
            const uint32_t culledDrawItems = totalDrawItems > visibleDrawItems ? totalDrawItems - visibleDrawItems : 0;
            message << "GPU culling:\n"
                    << "  total draw items: " << totalDrawItems << "\n"
                    << "  visible draw items: " << visibleDrawItems << "\n"
                    << "  culled draw items: " << culledDrawItems << "\n"
                    << "  occlusion culling: " << (isGpuOcclusionCullingActive() ? "enabled" : "disabled") << "\n"
                    << "  batches: " << batchCount << "\n"
                    << "  indirect count path: "
                    << (isFrameIndirectCountPathActive(frameIndex) ? "enabled" : "disabled");
        } else {
            message << "GPU culling:\n"
                    << "  total draw items: " << totalDrawItems << "\n"
                    << "  visible draw items: unavailable\n"
                    << "  occlusion culling: " << (isGpuOcclusionCullingActive() ? "enabled" : "disabled") << "\n"
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

void Renderer::pushGpuTimingSample(const renderer::GpuProfiler::FrameResults& results)
{
    if (!results.valid) {
        return;
    }

    gpuFrameTimeHistory_.push(historyValue(results.totalGpuTimeMs));
    for (const renderer::GpuProfiler::ScopeResult& scope : results.scopes) {
        if (DebugHistory* history = gpuTimingHistoryForPass(scope.name)) {
            history->push(historyValue(scope.elapsedMs));
        }
    }
}

void Renderer::resetGpuProfilerHistory()
{
    gpuFrameTimeHistory_ = {};
    gpuTimingHistories_.clear();
}

Renderer::DebugHistory* Renderer::gpuTimingHistoryForPass(std::string_view name)
{
    for (GpuTimingHistory& entry : gpuTimingHistories_) {
        if (entry.name == name) {
            return &entry.history;
        }
    }

    GpuTimingHistory entry{};
    entry.name = std::string(name);
    gpuTimingHistories_.push_back(std::move(entry));
    return &gpuTimingHistories_.back().history;
}

const Renderer::DebugHistory* Renderer::gpuTimingHistoryForPass(std::string_view name) const
{
    for (const GpuTimingHistory& entry : gpuTimingHistories_) {
        if (entry.name == name) {
            return &entry.history;
        }
    }

    return nullptr;
}

Renderer::CullingDebugSnapshot Renderer::cullingDebugSnapshot(uint32_t frameIndex)
{
    CullingDebugSnapshot snapshot{};
    snapshot.totalObjects =
        static_cast<uint32_t>(std::min<size_t>(cullingStats_.totalObjects, std::numeric_limits<uint32_t>::max()));
    snapshot.gpuCulling = isGpuCullingActive();
    snapshot.gpuOcclusionCulling = isGpuOcclusionCullingActive();
    snapshot.gpuShadowCulling = isGpuShadowCullingActive();
    snapshot.occlusionTestSceneActive = occlusionTestSceneActive_ && !portfolioCaptureMode_;
    snapshot.depthPyramidBuildAvailable = depthPyramidBuildAvailable_;
    snapshot.depthPyramidValid = depthPyramidValid_;
    snapshot.previousFrameDepthValid = previousFrameDepthValidForOcclusion();
    snapshot.depthPyramidMipCount = depthPyramidMipLevels_;
    snapshot.totalDrawItems = static_cast<uint32_t>(
        std::min<size_t>(cullingStats_.totalDrawItems, std::numeric_limits<uint32_t>::max()));
    if (cullingStats_.gpuCulling && frameIndex < frameGpuCullTotalDrawItems_.size()) {
        snapshot.totalDrawItems = frameGpuCullTotalDrawItems_[frameIndex];
    }

    snapshot.visibleDrawItems =
        static_cast<uint32_t>(std::min<size_t>(visibleDrawItems_.size(), snapshot.totalDrawItems));
    snapshot.frustumCulledDrawItems = snapshot.totalDrawItems > snapshot.visibleDrawItems
                                          ? snapshot.totalDrawItems - snapshot.visibleDrawItems
                                          : 0;
    uint32_t gpuVisibleDrawItems = 0;
    GpuCullCounters gpuCounters{};
    if (readGpuCullCounters(frameIndex, gpuCounters)) {
        snapshot.totalDrawItems = std::min(gpuCounters.totalDrawItems, snapshot.totalDrawItems);
        snapshot.visibleDrawItems = std::min(gpuCounters.visibleDrawItems, snapshot.totalDrawItems);
        snapshot.frustumCulledDrawItems = std::min(gpuCounters.frustumCulledDrawItems, snapshot.totalDrawItems);
        snapshot.occlusionCulledDrawItems = std::min(gpuCounters.occlusionCulledDrawItems, snapshot.totalDrawItems);
    } else if (readGpuVisibleCount(frameIndex, gpuVisibleDrawItems)) {
        snapshot.visibleDrawItems = std::min(gpuVisibleDrawItems, snapshot.totalDrawItems);
        snapshot.frustumCulledDrawItems = snapshot.totalDrawItems > snapshot.visibleDrawItems
                                              ? snapshot.totalDrawItems - snapshot.visibleDrawItems
                                              : 0;
    }
    snapshot.culledDrawItems = snapshot.totalDrawItems > snapshot.visibleDrawItems
                                   ? snapshot.totalDrawItems - snapshot.visibleDrawItems
                                   : 0;
    if (!snapshot.gpuCulling) {
        snapshot.frustumCulledDrawItems = snapshot.culledDrawItems;
        snapshot.occlusionCulledDrawItems = 0;
    }

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
    return snapshot;
}

Renderer::ObjectDrawDebugInfo Renderer::objectDrawDebugInfo(uint32_t objectIndex) const
{
    ObjectDrawDebugInfo debugInfo{};

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex != objectIndex) {
            continue;
        }

        ++debugInfo.drawItemCount;
        if (!debugInfo.hasObjectDataIndex) {
            debugInfo.firstObjectDataIndex = drawItem.frameDataIndex;
            debugInfo.hasObjectDataIndex = true;
        }
    }

    for (const DrawItem& drawItem : visibleDrawItems_) {
        if (drawItem.objectIndex == objectIndex) {
            ++debugInfo.visibleMainDrawItemCount;
        }
    }

    const uint32_t cascadeCount = activeCascadeCount();
    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount && cascadeIndex < shadowCascadeDrawItems_.size();
         ++cascadeIndex) {
        bool visibleInCascade = false;
        for (const DrawItem& drawItem : shadowCascadeDrawItems_[cascadeIndex]) {
            if (drawItem.objectIndex == objectIndex) {
                ++debugInfo.visibleShadowDrawItemCount;
                visibleInCascade = true;
            }
        }
        if (visibleInCascade) {
            ++debugInfo.shadowVisibleCascadeCount;
        }
    }

    return debugInfo;
}

std::vector<const renderer::Material*> Renderer::materialsForObject(const renderer::RenderObject& object) const
{
    std::vector<const renderer::Material*> materials;
    if (object.mesh && object.mesh->hasSubMeshes()) {
        const std::span<const renderer::MeshPrimitive> primitives = object.mesh->primitives();
        materials.reserve(primitives.size());
        for (const renderer::MeshPrimitive& primitive : primitives) {
            const renderer::Material* material = resolveMaterial(object, &primitive);
            if (!material) {
                continue;
            }
            if (std::find(materials.begin(), materials.end(), material) == materials.end()) {
                materials.push_back(material);
            }
        }
    }

    if (materials.empty() && object.material) {
        materials.push_back(object.material);
    }

    return materials;
}

const renderer::Material* Renderer::primaryMaterialForObject(const renderer::RenderObject& object) const
{
    const std::vector<const renderer::Material*> materials = materialsForObject(object);
    return materials.empty() ? nullptr : materials.front();
}

renderer::Material* Renderer::mutableMaterialFromPointer(const renderer::Material* material)
{
    if (!material) {
        return nullptr;
    }

    for (renderer::Material& candidate : materialVariants_) {
        if (&candidate == material) {
            return &candidate;
        }
    }
    for (renderer::Material& candidate : importedMaterials_) {
        if (&candidate == material) {
            return &candidate;
        }
    }
    if (&checkerboardMaterial_ == material) {
        return &checkerboardMaterial_;
    }

    return nullptr;
}

renderer::Material* Renderer::primaryMutableMaterialForObject(renderer::RenderObject& object)
{
    return mutableMaterialFromPointer(primaryMaterialForObject(object));
}

renderer::Material* Renderer::findRuntimeMaterialByAssetPath(const std::filesystem::path& path)
{
    if (path.empty()) {
        return nullptr;
    }

    const std::string key = path.lexically_normal().generic_string();
    const auto matchesPath = [&key](const renderer::Material& material) {
        return !material.sourceAssetPath.empty() && material.sourceAssetPath.lexically_normal().generic_string() == key;
    };

    for (renderer::Material& material : materialVariants_) {
        if (matchesPath(material)) {
            return &material;
        }
    }
    for (renderer::Material& material : importedMaterials_) {
        if (matchesPath(material)) {
            return &material;
        }
    }
    if (matchesPath(checkerboardMaterial_)) {
        return &checkerboardMaterial_;
    }

    return nullptr;
}

std::string Renderer::materialDebugLabel(const renderer::RenderObject& object) const
{
    if (!object.mesh) {
        return "(none)";
    }

    const std::span<const renderer::MeshPrimitive> primitives = object.mesh->primitives();
    if (!primitives.empty()) {
        std::vector<const renderer::Material*> uniqueMaterials;
        uniqueMaterials.reserve(primitives.size());
        for (const renderer::MeshPrimitive& primitive : primitives) {
            const renderer::Material* material = resolveMaterial(object, &primitive);
            if (!material) {
                continue;
            }
            if (std::find(uniqueMaterials.begin(), uniqueMaterials.end(), material) == uniqueMaterials.end()) {
                uniqueMaterials.push_back(material);
            }
        }

        if (uniqueMaterials.empty()) {
            return "(none)";
        }
        if (uniqueMaterials.size() == 1) {
            return materialNameOrFallback(uniqueMaterials.front());
        }

        return std::string(materialNameOrFallback(uniqueMaterials.front())) + " +" +
               std::to_string(uniqueMaterials.size() - 1) + " more";
    }

    return materialNameOrFallback(object.material);
}

std::string Renderer::mainCullingDebugLabel(const ObjectDrawDebugInfo& debugInfo) const
{
    if (allDrawItems_.empty()) {
        return "draw data pending";
    }
    if (debugInfo.drawItemCount == 0) {
        return "no draw items";
    }
    if (isGpuCullingActive()) {
        return "GPU culling active; per-object readback unavailable";
    }
    return debugInfo.visibleMainDrawItemCount > 0 ? "visible" : "culled";
}

std::string Renderer::shadowCullingDebugLabel(const ObjectDrawDebugInfo& debugInfo) const
{
    if (allDrawItems_.empty() || shadowCullingStats_.cascadeCount == 0) {
        return "draw data pending";
    }
    if (debugInfo.drawItemCount == 0) {
        return "no draw items";
    }
    if (isGpuShadowCullingActive()) {
        return "GPU shadow culling active; per-object readback unavailable";
    }
    if (debugInfo.shadowVisibleCascadeCount == 0) {
        return "culled from shadow cascades";
    }

    return "visible in " + std::to_string(debugInfo.shadowVisibleCascadeCount) + "/" +
           std::to_string(shadowCullingStats_.cascadeCount) + " cascades";
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
    const TaaSettings previousTaaSettings = taaSettings_;
    toneMappingSettings_ = settings.toneMapping;
    bloomSettings_ = settings.bloom;
    taaSettings_ = settings.taa;
    debugUiSettings_ = settings.debugUi;

    csmSettings_.lambda = settings.csm.lambda;
    csmSettings_.shadowDistance = settings.csm.shadowDistance;
    csmSettings_.enableTexelSnapping = settings.csm.enableTexelSnapping;
    csmSettings_.enableCascadeDebugColors = settings.csm.enableCascadeDebugColors;

    if (mode == RuntimeSettingsApplyMode::Startup) {
        csmSettings_.cascadeCount = settings.csm.cascadeCount;
        useGpuCulling_ = settings.useGpuCulling;
        useGpuShadowCulling_ = settings.useGpuShadowCulling;
        useGpuOcclusionCulling_ = settings.enableGpuOcclusionCulling && settings.useGpuCulling;
        useBindlessMaterialTextures_ = settings.enableBindlessMaterialTextures;
    } else {
        if (!settings.useGpuCulling || gpuCullingAvailable_) {
            useGpuCulling_ = settings.useGpuCulling;
        }
        if (!settings.useGpuShadowCulling || gpuShadowCullingAvailable_) {
            useGpuShadowCulling_ = settings.useGpuShadowCulling;
        }
        useGpuOcclusionCulling_ = settings.enableGpuOcclusionCulling;
    }

    clampRuntimeSettings();
    if (mode == RuntimeSettingsApplyMode::Startup) {
        useGpuOcclusionCulling_ = settings.enableGpuOcclusionCulling && settings.useGpuCulling;
    }
    if (!toneMappingSettings_.enableAutoExposure ||
        exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    }
    if (mode == RuntimeSettingsApplyMode::Runtime &&
        (previousTaaSettings.enabled != taaSettings_.enabled ||
         previousTaaSettings.jitterEnabled != taaSettings_.jitterEnabled ||
         previousTaaSettings.neighborhoodClampEnabled != taaSettings_.neighborhoodClampEnabled ||
         previousTaaSettings.feedback != taaSettings_.feedback)) {
        invalidateTaaHistory();
    }
    lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
}

RuntimeSettings Renderer::captureRuntimeSettings() const
{
    RuntimeSettings settings{};
    settings.toneMapping = toneMappingSettings_;
    settings.bloom = bloomSettings_;
    settings.taa = taaSettings_;
    settings.csm = csmSettings_;
    settings.debugUi = debugUiSettings_;
    settings.useGpuCulling = useGpuCulling_;
    settings.useGpuShadowCulling = useGpuShadowCulling_;
    settings.enableGpuOcclusionCulling = useGpuOcclusionCulling_;
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

void Renderer::resetCameraToDefault()
{
    camera_ = defaultCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;
    clampRuntimeSettings();
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

void Renderer::resetCameraToPortfolioPreset()
{
    camera_ = portfolioCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;
    clampRuntimeSettings();
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

void Renderer::resetCameraToOcclusionTestPreset()
{
    camera_ = occlusionTestCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;
    clampRuntimeSettings();
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

void Renderer::resetDirectionalLightToDefault()
{
    directionalLightSettings_.direction = {kDirectionalLightDirection.x,
                                           kDirectionalLightDirection.y,
                                           kDirectionalLightDirection.z};
    directionalLightSettings_.color = {kDirectionalLightColor.x, kDirectionalLightColor.y, kDirectionalLightColor.z};
    directionalLightSettings_.intensity = 1.0f;
}

void Renderer::loadOcclusionTestScene()
{
    if (portfolioCaptureMode_) {
        setPortfolioCaptureMode(false);
    }

    resetOcclusionTestSceneToPreset();
    if (!hasOcclusionTestScene()) {
        occlusionTestSceneActive_ = false;
        return;
    }

    occlusionTestSceneActive_ = true;
    resetCameraToOcclusionTestPreset();
    resetDirectionalLightToDefault();
    debugUiSettings_.showCullingStats = true;
    debugUiSettings_.showGpuTimingGraphs = true;
    debugUiSettings_.showRenderGraphPanel = true;
    occlusionTestSceneStatus_ = "Occlusion test scene active: " +
                                std::to_string(kOcclusionTestObjectCount) +
                                " procedural cube objects, including 5 occluder walls and 120 hidden/edge cubes.";
    Logger::info(occlusionTestSceneStatus_);
}

void Renderer::enableOcclusionTestSettings()
{
    useGpuCulling_ = true;
    useGpuOcclusionCulling_ = true;
    gpuOcclusionDepthBias_ = 0.003f;
    gpuOcclusionNearDisableDistance_ = 1.25f;
    gpuOcclusionMaxScreenCoverage_ = 0.60f;
    gpuOcclusionMinScreenPixels_ = 4.0f;
    debugUiSettings_.showCullingStats = true;
    debugUiSettings_.showGpuTimingGraphs = true;
    debugUiSettings_.showRenderGraphPanel = true;
    clampRuntimeSettings();
    occlusionTestSceneStatus_ =
        useGpuOcclusionCulling_
            ? "Occlusion test settings enabled. Let one or two frames pass for previous-frame depth."
            : "Occlusion test settings requested, but GPU culling or depth pyramid resources are unavailable.";
}

bool Renderer::previousFrameDepthValidForOcclusion() const
{
    return depthPyramidValid_ &&
           maxMatrixDifference(frameViewProjection_, depthPyramidViewProjection_) <= 0.0005f &&
           glm::distance(frameCameraPosition_, depthPyramidCameraPosition_) <= 0.01f;
}

void Renderer::saveSceneFromUi()
{
    try {
        const std::filesystem::path parentPath = sceneDocumentPath_.parent_path();
        if (!parentPath.empty()) {
            std::error_code createError;
            std::filesystem::create_directories(parentPath, createError);
            if (createError) {
                throw std::runtime_error("could not create scene directory '" + parentPath.string() +
                                         "': " + createError.message());
            }
        }

        Json cameraJson = Json{{"position", vec3ToJson(camera_.position)},
                               {"target", vec3ToJson(camera_.target)},
                               {"up", vec3ToJson(camera_.up)},
                               {"verticalFovDegrees", glm::degrees(camera_.verticalFovRadians)},
                               {"nearPlane", camera_.nearPlane},
                               {"farPlane", camera_.farPlane}};

        Json lightJson = Json{{"direction", vec3ToJson(directionalLightSettings_.direction)},
                              {"color", vec3ToJson(directionalLightSettings_.color)},
                              {"intensity", directionalLightSettings_.intensity},
                              {"portfolioPresetActive", portfolioCaptureMode_}};

        Json objectsJson = Json::array();
        for (size_t objectIndex = 0; objectIndex < renderObjects_.size(); ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            const uint32_t objectId = renderObjectEditorId(object);
            const renderer::Material* material = primaryMaterialForObject(object);
            const ObjectDrawDebugInfo debugInfo = objectDrawDebugInfo(static_cast<uint32_t>(objectIndex));

            Json objectJson = Json{{"id", objectId},
                                   {"debugId", object.debugId},
                                   {"name", object.debugName},
                                   {"visible", object.visible},
                                   {"source", renderObjectSourceTypeName(object.sourceType)},
                                   {"portfolioOnly", object.portfolioOnly},
                                   {"hideInPortfolio", object.hideInPortfolio},
                                   {"transform", transformToJson(object.transform)},
                                   {"drawItemCount", debugInfo.drawItemCount}};

            objectJson["mesh"] = Json{{"name", object.mesh ? object.mesh->debugName() : std::string{}},
                                      {"pointer", pointerString(object.mesh)},
                                      {"submeshCount", meshSubmeshCount(object.mesh)}};
            objectJson["material"] =
                Json{{"name", material ? material->debugName : std::string{}},
                     {"assetName", material ? material->assetName : std::string{}},
                     {"assetPath", material ? stableProjectPathString(material->sourceAssetPath) : std::string{}},
                     {"shader", material ? material->shader : std::string{}},
                     {"primaryLabel", materialDebugLabel(object)},
                     {"pointer", pointerString(material)},
                     {"slotCount", object.materialCount},
                     {"source", material ? std::string(materialSourceName(material->source)) : std::string{"none"}},
                     {"materialAssetRebinding", object.materialTable ? "metadata-only for material tables"
                                                                      : "restored by assetPath when available"}};

            objectsJson.push_back(std::move(objectJson));
        }

        const Json sceneJson = Json{{"schemaVersion", 1},
                                    {"sceneName", portfolioCaptureMode_ ? "Portfolio Runtime Scene"
                                                  : (occlusionTestSceneActive_ ? "Occlusion Test Runtime Scene"
                                                                               : "Default Runtime Scene")},
                                    {"camera", std::move(cameraJson)},
                                    {"directionalLight", std::move(lightJson)},
                                    {"objects", std::move(objectsJson)},
                                     {"limitations",
                                     Json::array({"Mesh references and glTF material-table references are saved as "
                                                  "debug metadata only.",
                                                  "Simple object material asset paths are restored when they match a "
                                                  "loaded runtime material.",
                                                  "glTF material-table assignments remain runtime data and are not "
                                                  "rebuilt from scene JSON.",
                                                  "Load preserves current runtime mesh/material pointers and restores "
                                                  "matching object transforms, names, visibility, camera, and light."})}};

        std::ofstream output(sceneDocumentPath_);
        if (!output) {
            throw std::runtime_error("could not open scene file for writing");
        }
        output << sceneJson.dump(4) << '\n';
        if (!output) {
            throw std::runtime_error("failed while writing scene file");
        }

        lastSceneSaveStatus_ = "Saved scene to " + sceneDocumentPath_.string() + ".";
        Logger::info(lastSceneSaveStatus_);
    } catch (const std::exception& error) {
        lastSceneSaveStatus_ = "Scene save failed: " + std::string(error.what());
        Logger::warn(lastSceneSaveStatus_);
    }
}

void Renderer::loadSceneFromUi()
{
    try {
        std::error_code existsError;
        if (!std::filesystem::exists(sceneDocumentPath_, existsError)) {
            if (existsError) {
                throw std::runtime_error("could not check scene file: " + existsError.message());
            }
            lastSceneLoadStatus_ =
                std::string(kNoSavedSceneFoundMessage) + " Scene path: " + sceneDocumentPath_.string() + ".";
            Logger::warn(lastSceneLoadStatus_);
            return;
        }

        std::ifstream input(sceneDocumentPath_);
        if (!input) {
            throw std::runtime_error("could not open scene file for reading");
        }

        const Json sceneJson = Json::parse(input);
        if (!sceneJson.is_object()) {
            throw std::runtime_error("scene root must be a JSON object");
        }

        if (const Json* cameraJson = jsonObjectMember(sceneJson, "camera")) {
            readJsonVec3(*cameraJson, "position", camera_.position);
            readJsonVec3(*cameraJson, "target", camera_.target);
            readJsonVec3(*cameraJson, "up", camera_.up);

            float fovDegrees = glm::degrees(camera_.verticalFovRadians);
            if (readJsonFloat(*cameraJson, "verticalFovDegrees", fovDegrees)) {
                camera_.verticalFovRadians = glm::radians(std::clamp(fovDegrees, 1.0f, 160.0f));
            } else {
                readJsonFloat(*cameraJson, "verticalFovRadians", camera_.verticalFovRadians);
                camera_.verticalFovRadians = std::clamp(camera_.verticalFovRadians,
                                                        glm::radians(1.0f),
                                                        glm::radians(160.0f));
            }

            readJsonFloat(*cameraJson, "nearPlane", camera_.nearPlane);
            readJsonFloat(*cameraJson, "farPlane", camera_.farPlane);
            camera_.nearPlane = std::max(camera_.nearPlane, 0.001f);
            camera_.farPlane = std::max(camera_.farPlane, camera_.nearPlane + 0.001f);
            camera_.up = normalizedOrFallback(camera_.up, {0.0f, 1.0f, 0.0f});
            if (glm::length(camera_.target - camera_.position) <= 0.001f) {
                camera_.target = camera_.position + glm::vec3{0.0f, 0.0f, -1.0f};
            }
            csmSettings_.nearPlane = camera_.nearPlane;
            csmSettings_.farPlane = camera_.farPlane;
        }

        if (const Json* lightJson = jsonObjectMember(sceneJson, "directionalLight")) {
            readJsonVec3(*lightJson, "direction", directionalLightSettings_.direction);
            readJsonVec3(*lightJson, "color", directionalLightSettings_.color);
            readJsonFloat(*lightJson, "intensity", directionalLightSettings_.intensity);
            directionalLightSettings_.direction =
                normalizedOrFallback(directionalLightSettings_.direction,
                                     {kDirectionalLightDirection.x,
                                      kDirectionalLightDirection.y,
                                      kDirectionalLightDirection.z});
            directionalLightSettings_.color = glm::max(directionalLightSettings_.color, glm::vec3{0.0f});
            directionalLightSettings_.intensity = std::max(directionalLightSettings_.intensity, 0.0f);
        }

        size_t matchedObjects = 0;
        size_t skippedObjects = 0;
        size_t restoredMaterialAssets = 0;
        size_t skippedMaterialAssets = 0;
        std::vector<uint8_t> objectUsed(renderObjects_.size(), 0);
        if (const auto objectsIt = sceneJson.find("objects"); objectsIt != sceneJson.end()) {
            if (!objectsIt->is_array()) {
                throw std::runtime_error("Expected array member 'objects'.");
            }

            for (const Json& objectJson : *objectsIt) {
                if (!objectJson.is_object()) {
                    ++skippedObjects;
                    continue;
                }

                uint32_t objectId = 0;
                readJsonUint32(objectJson, "id", objectId);
                if (objectId == 0) {
                    readJsonUint32(objectJson, "sceneObjectId", objectId);
                }

                std::string objectName;
                readJsonString(objectJson, "name", objectName);

                size_t objectIndex = kInvalidRenderObjectIndex;
                if (objectId != 0) {
                    for (size_t candidateIndex = 0; candidateIndex < renderObjects_.size(); ++candidateIndex) {
                        if (objectUsed[candidateIndex]) {
                            continue;
                        }
                        const renderer::RenderObject& candidate = renderObjects_[candidateIndex];
                        if (renderObjectEditorId(candidate) == objectId || candidate.debugId == objectId) {
                            objectIndex = candidateIndex;
                            break;
                        }
                    }
                }

                if (objectIndex == kInvalidRenderObjectIndex && !objectName.empty()) {
                    for (size_t candidateIndex = 0; candidateIndex < renderObjects_.size(); ++candidateIndex) {
                        if (!objectUsed[candidateIndex] && renderObjects_[candidateIndex].debugName == objectName) {
                            objectIndex = candidateIndex;
                            break;
                        }
                    }
                }

                if (objectIndex == kInvalidRenderObjectIndex) {
                    ++skippedObjects;
                    continue;
                }

                objectUsed[objectIndex] = 1;
                renderer::RenderObject& object = renderObjects_[objectIndex];
                if (objectId != 0) {
                    object.sceneObjectId = objectId;
                    object.debugId = objectId;
                } else if (object.sceneObjectId == 0) {
                    object.sceneObjectId = object.debugId;
                }
                if (!objectName.empty()) {
                    object.debugName = objectName;
                }
                readJsonBool(objectJson, "visible", object.visible);

                if (const Json* materialJson = jsonObjectMember(objectJson, "material")) {
                    std::string materialAssetPathString;
                    readJsonString(*materialJson, "assetPath", materialAssetPathString);
                    if (!materialAssetPathString.empty()) {
                        const std::filesystem::path materialPath = resolveProjectPath(materialAssetPathString);
                        if (renderer::Material* material = findRuntimeMaterialByAssetPath(materialPath)) {
                            if (!object.materialTable) {
                                object.material = material;
                                ++restoredMaterialAssets;
                            } else {
                                ++skippedMaterialAssets;
                            }
                        } else {
                            ++skippedMaterialAssets;
                            Logger::warn("Scene material asset path did not match a runtime material: " +
                                         materialAssetPathString);
                        }
                    }
                }

                if (const Json* transformJson = jsonObjectMember(objectJson, "transform")) {
                    std::string mode = "trs";
                    readJsonString(*transformJson, "mode", mode);

                    glm::mat4 matrix{1.0f};
                    const bool hasMatrix = readJsonMat4(*transformJson, "matrix", matrix);
                    if (mode == "matrix" && hasMatrix) {
                        object.transform = renderer::Transform::fromMatrix(matrix);
                    } else {
                        renderer::Transform editableTransform = object.transform;
                        convertMatrixOverrideToEditableTrs(editableTransform);
                        editableTransform.useMatrixOverride = false;
                        editableTransform.matrixOverride = glm::mat4{1.0f};

                        readJsonVec3(*transformJson, "position", editableTransform.position);

                        glm::vec3 rotationDegrees = glm::degrees(editableTransform.rotationRadians);
                        if (readJsonVec3(*transformJson, "rotationDegrees", rotationDegrees)) {
                            editableTransform.rotationRadians = glm::radians(rotationDegrees);
                        } else {
                            readJsonVec3(*transformJson, "rotationRadians", editableTransform.rotationRadians);
                        }

                        readJsonVec3(*transformJson, "scale", editableTransform.scale);
                        object.transform = editableTransform;
                    }
                }

                object.animateTransform = false;
                ++matchedObjects;
            }
        }

        uint32_t maxObjectId = 0;
        for (renderer::RenderObject& object : renderObjects_) {
            if (object.sceneObjectId == 0) {
                object.sceneObjectId = object.debugId;
            }
            maxObjectId = std::max(maxObjectId, renderObjectEditorId(object));
        }
        if (maxObjectId < std::numeric_limits<uint32_t>::max()) {
            nextRenderObjectDebugId_ = std::max(nextRenderObjectDebugId_, maxObjectId + 1);
        }

        clampRuntimeSettings();
        invalidateDepthPyramid();
        invalidateTaaHistory();
        lastSceneLoadStatus_ = "Loaded scene from " + sceneDocumentPath_.string() + ". Matched " +
                               std::to_string(matchedObjects) + " object(s), skipped " +
                               std::to_string(skippedObjects) + ", restored " +
                               std::to_string(restoredMaterialAssets) + " material asset assignment(s), skipped " +
                               std::to_string(skippedMaterialAssets) + ".";
        Logger::info(lastSceneLoadStatus_);
    } catch (const std::exception& error) {
        lastSceneLoadStatus_ = "Scene load failed: " + std::string(error.what());
        Logger::warn(lastSceneLoadStatus_);
    }
}


void Renderer::clampRuntimeSettings()
{
    // The settings-struct clamping is GPU-independent and lives in
    // RuntimeSettings.cpp (compiled into VulkanEngineCore) so it can be tested.
    ve::clampRuntimeSettings(
        toneMappingSettings_, bloomSettings_, taaSettings_, csmSettings_, debugUiSettings_);

    // GPU occlusion tuning is renderer state, not part of the settings structs,
    // so it stays here.
    gpuOcclusionDepthBias_ = std::clamp(gpuOcclusionDepthBias_, 0.0f, 0.05f);
    gpuOcclusionNearDisableDistance_ = std::clamp(gpuOcclusionNearDisableDistance_, 0.0f, 10.0f);
    gpuOcclusionMaxScreenCoverage_ = std::clamp(gpuOcclusionMaxScreenCoverage_, 0.01f, 1.0f);
    gpuOcclusionMinScreenPixels_ = std::clamp(gpuOcclusionMinScreenPixels_, 1.0f, 64.0f);
    if (!isGpuCullingActive() || !depthPyramidBuildAvailable_) {
        useGpuOcclusionCulling_ = false;
    }
}

void Renderer::updateFrameData(uint32_t frameIndex)
{
    const auto now = std::chrono::steady_clock::now();
    const float elapsedSeconds = std::chrono::duration<float>(now - startTime_).count();

    const VkExtent2D extent = swapchain_.extent();
    const float aspect =
        extent.height == 0 ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 view = camera_.viewMatrix();
    const glm::mat4 projection = camera_.projectionMatrix(aspect);
    const glm::mat4 viewProjection = projection * view;
    glm::mat4 jitteredProjection = projection;
    taaPreviousJitterPixels_ = taaCurrentJitterPixels_;
    taaPreviousJitterNdc_ = taaCurrentJitterNdc_;
    taaCurrentJitterPixels_ = {0.0f, 0.0f};
    taaCurrentJitterNdc_ = {0.0f, 0.0f};
    if (isTaaJitterActive() && extent.width > 0 && extent.height > 0) {
        const uint32_t sampleIndex = (taaJitterIndex_ % kTaaJitterSampleCount) + 1u;
        taaCurrentJitterPixels_ = {
            halton(sampleIndex, 2u) - 0.5f,
            halton(sampleIndex, 3u) - 0.5f,
        };
        taaCurrentJitterNdc_ = {
            2.0f * taaCurrentJitterPixels_.x / static_cast<float>(extent.width),
            2.0f * taaCurrentJitterPixels_.y / static_cast<float>(extent.height),
        };
        jitteredProjection[2][0] += taaCurrentJitterNdc_.x;
        jitteredProjection[2][1] += taaCurrentJitterNdc_.y;
        ++taaJitterIndex_;
    }
    previousFrameViewProjection_ = frameJitteredViewProjection_;
    frameJitteredProjection_ = jitteredProjection;
    frameJitteredViewProjection_ = jitteredProjection * view;
    frameViewProjection_ = viewProjection;
    frameCameraPosition_ = camera_.position;

    if (renderObjects_.empty()) {
        resetFrameStateForEmptyScene(frameIndex);
        return;
    }

    updateCascades(aspect);

    if (updateAnimatedTransforms(elapsedSeconds)) {
        invalidateDepthPyramid();
    }

    buildDrawItems();
    resetGpuCullFrameCounters(frameIndex);

    const renderer::Frustum cameraFrustum = renderer::Frustum::fromViewProjection(viewProjection);
    for (size_t planeIndex = 0; planeIndex < frameFrustumPlanes_.size(); ++planeIndex) {
        const renderer::FrustumPlane& cameraPlane = cameraFrustum.planes[planeIndex];
        frameFrustumPlanes_[planeIndex] = glm::vec4(cameraPlane.normal, cameraPlane.distance);
    }

    buildShadowFrameData(frameIndex);
    buildMainCullingFrameData(frameIndex, cameraFrustum);
    uploadObjectFrameData(frameIndex);
}

void Renderer::resetFrameStateForEmptyScene(uint32_t frameIndex)
{
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
}

bool Renderer::updateAnimatedTransforms(float elapsedSeconds)
{
    bool animatedTransformUpdated = false;
    for (size_t objectIndex = 0; objectIndex < renderObjects_.size(); ++objectIndex) {
        renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.animateTransform) {
            continue;
        }

        animatedTransformUpdated = true;
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
    return animatedTransformUpdated;
}

void Renderer::resetGpuCullFrameCounters(uint32_t frameIndex)
{
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
}

void Renderer::buildShadowFrameData(uint32_t frameIndex)
{
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
}

void Renderer::buildMainCullingFrameData(uint32_t frameIndex, const renderer::Frustum& cameraFrustum)
{
    const bool gpuCullingActive = isGpuCullingActive();
    if (gpuCullingActive) {
        visibleDrawItems_ = allDrawItems_;
        cullingStats_ = {};
        cullingStats_.gpuCulling = true;
        const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
        for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            if (isRenderObjectActive(object) && object.mesh && object.mesh->valid()) {
                ++cullingStats_.totalObjects;
            }
        }
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
}

void Renderer::uploadObjectFrameData(uint32_t frameIndex)
{
    const uint32_t cascadeCount = activeCascadeCount();
    const size_t objectFrameCount = std::min(allDrawItems_.size(), static_cast<size_t>(kMaxDrawItems));
    std::vector<ObjectFrameData> objectFrameData(objectFrameCount);
    const glm::vec4 activeLightDirection = activeDirectionalLightDirection();
    const glm::vec4 activeLightColor = activeDirectionalLightColor();
    const glm::vec4 activeAmbientLightColor =
        portfolioCaptureMode_ ? kPortfolioAmbientLightColor : kAmbientLightColor;

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
        frameData.mvp = frameJitteredViewProjection_ * model;
        frameData.model = model;
        for (uint32_t cascadeIndex = 0; cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
            frameData.lightMvp[cascadeIndex] = frameCascades_[cascadeIndex].lightViewProjection * model;
        }
        frameData.lightDirection = activeLightDirection;
        frameData.lightColor = activeLightColor;
        frameData.ambientColor = activeAmbientLightColor;
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
         (histogramExposureAvailable_ && histogramPipeline_.pipeline() == VK_NULL_HANDLE) ||
         (exposureReduceAvailable_ && exposureReducePipeline_.pipeline() == VK_NULL_HANDLE));
    const bool pipelineNeedsRecreate =
        pipeline_.pipeline() == VK_NULL_HANDLE || pipelineColorFormat_ != kSceneColorFormat ||
        pipelineDepthFormat_ != swapchain_.depthFormat() || skyboxPipeline_.pipeline() == VK_NULL_HANDLE ||
        skyboxPipelineColorFormat_ != kSceneColorFormat || skyboxPipelineDepthFormat_ != swapchain_.depthFormat() ||
        shadowPipelineDepthFormat_ != shadowMap_.format() || bloomExtractPipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomExtractPipelineColorFormat_ != kBloomColorFormat || bloomBlurPipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomBlurPipelineColorFormat_ != kBloomColorFormat ||
        bloomDownsamplePipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomDownsamplePipelineColorFormat_ != kBloomColorFormat ||
        bloomUpsamplePipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomUpsamplePipelineColorFormat_ != kBloomColorFormat || taaResolvePipeline_.pipeline() == VK_NULL_HANDLE ||
        taaResolvePipelineColorFormat_ != kSceneColorFormat || compositePipeline_.pipeline() == VK_NULL_HANDLE ||
        compositePipelineColorFormat_ != swapchain_.colorFormat() || exposurePipelineMissing;
    if (pipelineNeedsRecreate) {
        createPipeline();
    }

    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
}

renderer::RenderGraphFrameResources Renderer::renderGraphFrameResources()
{
    const VkExtent3D sceneExtent = sceneColor_.extent();
    const VkExtent3D depthPyramidExtent = depthPyramid_.extent();
    VkClearValue sceneClear{};
    sceneClear.color.float32[0] = 0.03f;
    sceneClear.color.float32[1] = 0.04f;
    sceneClear.color.float32[2] = 0.07f;
    sceneClear.color.float32[3] = 1.0f;

    VkClearValue bloomClear{};
    bloomClear.color.float32[0] = 0.0f;
    bloomClear.color.float32[1] = 0.0f;
    bloomClear.color.float32[2] = 0.0f;
    bloomClear.color.float32[3] = 1.0f;

    const auto bufferResource = [](const char* name,
                                   const std::vector<rhi::VulkanBuffer>& buffers,
                                   uint32_t frameIndex,
                                   VkBufferUsageFlags usage) {
        if (frameIndex >= buffers.size() || !buffers[frameIndex].valid()) {
            return renderer::RenderGraphBufferResource{};
        }

        return renderer::RenderGraphBufferResource{
            name,
            buffers[frameIndex].buffer(),
            buffers[frameIndex].size(),
            usage,
            true,
        };
    };

    const auto bloomResource = [&bloomClear](const char* name,
                                             rhi::VulkanImage& image,
                                             VkImageLayout& layout) {
        const VkExtent3D extent = image.extent();
        return renderer::RenderGraphImageResource{
            name,
            image.image(),
            image.imageView(),
            VkExtent2D{extent.width, extent.height},
            &layout,
            image.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        };
    };

    const auto taaHistoryResource = [&bloomClear](const char* name,
                                                  rhi::VulkanImage& image,
                                                  VkImageLayout& layout) {
        const VkExtent3D extent = image.extent();
        return renderer::RenderGraphImageResource{
            name,
            image.image(),
            image.imageView(),
            VkExtent2D{extent.width, extent.height},
            &layout,
            image.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            true,
        };
    };

    std::vector<renderer::RenderGraphImageResource> bloomDownsampleResources;
    bloomDownsampleResources.reserve(bloomMipDownsampleImages_.size());
    for (size_t level = 0; level < bloomMipDownsampleImages_.size() && level < bloomMipDownsampleLayouts_.size();
         ++level) {
        const std::string name = "BloomMipDownsample" + std::to_string(level);
        bloomDownsampleResources.push_back(
            bloomResource(name.c_str(), bloomMipDownsampleImages_[level], bloomMipDownsampleLayouts_[level]));
    }

    std::vector<renderer::RenderGraphImageResource> bloomUpsampleResources;
    bloomUpsampleResources.reserve(bloomMipUpsampleImages_.size());
    for (size_t level = 0; level < bloomMipUpsampleImages_.size() && level < bloomMipUpsampleLayouts_.size(); ++level) {
        const std::string name = "BloomMipUpsample" + std::to_string(level);
        bloomUpsampleResources.push_back(
            bloomResource(name.c_str(), bloomMipUpsampleImages_[level], bloomMipUpsampleLayouts_[level]));
    }

    return renderer::RenderGraphFrameResources{
        renderer::RenderGraphImageResource{
            "SceneColor",
            sceneColor_.image(),
            sceneColor_.imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &sceneColorLayout_,
            sceneColor_.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            sceneClear,
            true,
            false,
        },
        taaHistoryResource("TAAHistoryRead",
                           taaHistoryImages_[taaHistoryReadIndex()],
                           taaHistoryLayouts_[taaHistoryReadIndex()]),
        taaHistoryResource("TAAHistoryWrite",
                           taaHistoryImages_[taaHistoryWriteIndex()],
                           taaHistoryLayouts_[taaHistoryWriteIndex()]),
        renderer::RenderGraphImageResource{
            "BloomExtract",
            bloomExtract_.image(),
            bloomExtract_.imageView(),
            bloomExtent_,
            &bloomExtractLayout_,
            bloomExtract_.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "BloomPing",
            bloomPing_.image(),
            bloomPing_.imageView(),
            bloomExtent_,
            &bloomPingLayout_,
            bloomPing_.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "BloomPong",
            bloomPong_.image(),
            bloomPong_.imageView(),
            bloomExtent_,
            &bloomPongLayout_,
            bloomPong_.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        },
        std::move(bloomDownsampleResources),
        std::move(bloomUpsampleResources),
        renderer::RenderGraphImageResource{
            "DepthPyramidHiZ",
            depthPyramid_.image(),
            depthPyramid_.imageView(),
            VkExtent2D{depthPyramidExtent.width, depthPyramidExtent.height},
            &depthPyramidLayout_,
            depthPyramid_.format(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            depthPyramidMipLevels_,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            {},
            false,
            true,
        },
        bufferResource("MainCullInput",
                       frameCullInputBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        bufferResource("MainCullIndirectOutput",
                       frameIndirectDrawBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT),
        bufferResource("MainCullVisibleCounts",
                       frameBatchVisibleCountBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        bufferResource("MainCullReadback",
                       frameBatchVisibleCountReadbackBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("LuminancePartials",
                       frameLuminanceBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        bufferResource("LuminanceReadback",
                       frameLuminanceReadbackBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("LuminanceHistogram",
                       frameHistogramBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("HistogramReadback",
                       frameHistogramReadbackBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("ExposureState",
                       frameExposureBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        isTaaActive(),
    };
}

void Renderer::updateAutoExposureFromReadback(uint32_t frameIndex)
{
    const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
    if (!toneMappingSettings_.enableAutoExposure || mode == ExposureMode::Manual) {
        currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
        if (frameIndex < frameExposureBuffers_.size() && frameExposureBuffers_[frameIndex].valid()) {
            const ExposureState manualState{
                currentExposure_,
                averageLuminance_,
                histogramClippedLuminance_,
                static_cast<uint32_t>(ExposureMode::Manual),
            };
            frameExposureBuffers_[frameIndex].upload(std::as_bytes(std::span<const ExposureState>(&manualState, 1)));
            if (frameIndex < frameExposureReadbackReady_.size()) {
                frameExposureReadbackReady_[frameIndex] = 1;
            }
        }
        return;
    }

    if (frameIndex >= frameExposureReadbackReady_.size() || frameExposureReadbackReady_[frameIndex] == 0 ||
        frameIndex >= frameExposureBuffers_.size() || !frameExposureBuffers_[frameIndex].valid()) {
        return;
    }

    ExposureState state{};
    frameExposureBuffers_[frameIndex].download(std::as_writable_bytes(std::span<ExposureState>(&state, 1)));
    if (std::isfinite(state.exposure) && state.exposure > 0.0f) {
        currentExposure_ = toneMappingExposureValue(state.exposure);
    }
    if (std::isfinite(state.averageLuminance) && state.averageLuminance > 0.0f) {
        averageLuminance_ = std::max(state.averageLuminance, kMinAverageLuminance);
    }
    if (std::isfinite(state.histogramLuminance) && state.histogramLuminance > 0.0f) {
        histogramClippedLuminance_ = std::max(state.histogramLuminance, kMinAverageLuminance);
    }
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

bool Renderer::readGpuCullCounters(uint32_t frameIndex, GpuCullCounters& counters)
{
    counters = {};
    if (!isGpuCullingActive() || frameIndex >= frameBatchVisibleCountReadbackBuffers_.size() ||
        frameIndex >= frameGpuCullReadbackReady_.size() || frameGpuCullReadbackReady_[frameIndex] == 0) {
        return false;
    }

    rhi::VulkanBuffer& readbackBuffer = frameBatchVisibleCountReadbackBuffers_[frameIndex];
    if (!readbackBuffer.valid()) {
        return false;
    }

    std::array<uint32_t, kGpuCullStatsCounterCount> values{};
    readbackBuffer.download(std::as_writable_bytes(std::span<uint32_t>(values.data(), values.size())),
                            static_cast<VkDeviceSize>(kGpuCullStatsCounterOffset * sizeof(uint32_t)));

    counters.totalDrawItems = values[0];
    counters.visibleDrawItems = values[1];
    counters.frustumCulledDrawItems = values[2];
    counters.occlusionCulledDrawItems = values[3];
    if (frameIndex < frameGpuCullTotalDrawItems_.size()) {
        counters.totalDrawItems = std::min(counters.totalDrawItems, frameGpuCullTotalDrawItems_[frameIndex]);
        counters.visibleDrawItems = std::min(counters.visibleDrawItems, counters.totalDrawItems);
        counters.frustumCulledDrawItems = std::min(counters.frustumCulledDrawItems, counters.totalDrawItems);
        counters.occlusionCulledDrawItems = std::min(counters.occlusionCulledDrawItems, counters.totalDrawItems);
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
           frameBatchVisibleCountReadbackBuffers_.size() == frames_.size() &&
           frameGpuCullParamBuffers_.size() == frames_.size();
}

bool Renderer::isGpuOcclusionCullingActive() const
{
    return useGpuOcclusionCulling_ && isGpuCullingActive() && depthPyramidBuildAvailable_ && depthPyramidValid_ &&
           depthPyramid_.image() != VK_NULL_HANDLE && depthPyramidSampler_ != VK_NULL_HANDLE &&
           depthPyramidMipLevels_ > 0;
}

bool Renderer::isGpuShadowCullingActive() const
{
    return useGpuShadowCulling_ && gpuShadowCullingAvailable_ && isShadowIndirectActive() &&
           gpuCullPipeline_.pipeline() != VK_NULL_HANDLE && gpuCullPipeline_.layout() != VK_NULL_HANDLE &&
           !shadowCullDescriptorSets_.empty() && frameShadowCullInputBuffers_.size() == frames_.size() &&
           frameShadowBatchVisibleCountBuffers_.size() == frames_.size() &&
           frameShadowBatchVisibleCountReadbackBuffers_.size() == frames_.size() &&
           frameGpuCullParamBuffers_.size() == frames_.size();
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
           luminancePartialCount_ > 0 && luminanceGroupCountX_ > 0 && luminanceGroupCountY_ > 0;
}

bool Renderer::isHistogramExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && histogramExposureAvailable_ &&
           histogramPipeline_.pipeline() != VK_NULL_HANDLE && histogramPipeline_.layout() != VK_NULL_HANDLE &&
           histogramDescriptorSets_.size() == frames_.size() && frameHistogramBuffers_.size() == frames_.size() &&
           isLogAverageExposureActive();
}

bool Renderer::isGpuExposureActive() const
{
    return toneMappingSettings_.enableAutoExposure && exposureReduceAvailable_ &&
           exposureReducePipeline_.pipeline() != VK_NULL_HANDLE && exposureReducePipeline_.layout() != VK_NULL_HANDLE &&
           exposureReduceDescriptorSets_.size() == frames_.size() && frameExposureBuffers_.size() == frames_.size();
}

bool Renderer::isTaaActive() const
{
    return taaSettings_.enabled && taaResolvePipeline_.pipeline() != VK_NULL_HANDLE &&
           taaResolvePipeline_.layout() != VK_NULL_HANDLE &&
           taaHistoryImages_[0].imageView() != VK_NULL_HANDLE && taaHistoryImages_[1].imageView() != VK_NULL_HANDLE &&
           taaResolveDescriptorSets_[0] != VK_NULL_HANDLE && taaResolveDescriptorSets_[1] != VK_NULL_HANDLE;
}

bool Renderer::isTaaJitterActive() const
{
    return isTaaActive() && taaSettings_.jitterEnabled;
}

uint32_t Renderer::taaHistoryReadIndex() const
{
    return (taaHistoryWriteIndex_ + 1u) % kTaaHistoryCount;
}

uint32_t Renderer::taaHistoryWriteIndex() const
{
    return taaHistoryWriteIndex_ % kTaaHistoryCount;
}

VkDescriptorSet Renderer::activeBloomExtractDescriptorSet() const
{
    if (isTaaActive()) {
        const VkDescriptorSet descriptorSet =
            taaBloomExtractDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (descriptorSet != VK_NULL_HANDLE) {
            return descriptorSet;
        }
    }
    return bloomExtractDescriptorSet_;
}

VkDescriptorSet Renderer::activeBloomMipDownsampleDescriptorSet(uint32_t level) const
{
    if (level == 0 && isTaaActive()) {
        const VkDescriptorSet descriptorSet =
            taaBloomMipDownsampleDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (descriptorSet != VK_NULL_HANDLE) {
            return descriptorSet;
        }
    }
    return level < bloomMipDownsampleDescriptorSets_.size() ? bloomMipDownsampleDescriptorSets_[level]
                                                           : VK_NULL_HANDLE;
}

VkDescriptorSet Renderer::activeCompositeDescriptorSet() const
{
    if (isTaaActive()) {
        const auto& descriptorSets = taaCompositeDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (currentFrame_ < descriptorSets.size() && descriptorSets[currentFrame_] != VK_NULL_HANDLE) {
            return descriptorSets[currentFrame_];
        }
    }
    return currentFrame_ < compositeDescriptorSets_.size() ? compositeDescriptorSets_[currentFrame_]
                                                          : compositeDescriptorSet_;
}

VkDescriptorSet Renderer::activeLuminanceDescriptorSet() const
{
    if (isTaaActive()) {
        const auto& descriptorSets = taaLuminanceDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (currentFrame_ < descriptorSets.size() && descriptorSets[currentFrame_] != VK_NULL_HANDLE) {
            return descriptorSets[currentFrame_];
        }
    }
    return currentFrame_ < luminanceDescriptorSets_.size() ? luminanceDescriptorSets_[currentFrame_]
                                                          : VK_NULL_HANDLE;
}

VkDescriptorSet Renderer::activeHistogramDescriptorSet() const
{
    if (isTaaActive()) {
        const auto& descriptorSets = taaHistogramDescriptorSets_[taaPostProcessHistoryIndex_ % kTaaHistoryCount];
        if (currentFrame_ < descriptorSets.size() && descriptorSets[currentFrame_] != VK_NULL_HANDLE) {
            return descriptorSets[currentFrame_];
        }
    }
    return currentFrame_ < histogramDescriptorSets_.size() ? histogramDescriptorSets_[currentFrame_]
                                                          : VK_NULL_HANDLE;
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

    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, "MainGpuCullingPass");
    renderGraph_.beginMainGpuCullingPass();
    rhi::debug::beginLabel(commandBuffer, "GpuCulling");
    ensureDepthPyramidShaderReadLayout(commandBuffer);
    vkCmdFillBuffer(commandBuffer, visibleCountBuffer, 0, kGpuCullCountBufferSize, 0);

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
    resetCountBarrier.size = kGpuCullCountBufferSize;

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
                                      1U);
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

    VkBufferMemoryBarrier2 visibleCountCopyBarrier{};
    visibleCountCopyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    visibleCountCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    visibleCountCopyBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    visibleCountCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    visibleCountCopyBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    visibleCountCopyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    visibleCountCopyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    visibleCountCopyBarrier.buffer = visibleCountBuffer;
    visibleCountCopyBarrier.offset = 0;
    visibleCountCopyBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo visibleCountCopyDependency{};
    visibleCountCopyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    visibleCountCopyDependency.bufferMemoryBarrierCount = 1;
    visibleCountCopyDependency.pBufferMemoryBarriers = &visibleCountCopyBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &visibleCountCopyDependency);

    VkBufferCopy visibleCountCopy{};
    visibleCountCopy.size = kGpuCullCountBufferSize;
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
    readbackBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameGpuCullReadbackReady_[currentFrame_] = 1;
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endMainGpuCullingPass();
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

    const std::string profileName = "ShadowGpuCullingCascade" + std::to_string(cascadeIndex);
    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, profileName);
    rhi::debug::beginLabel(commandBuffer, "GpuShadowCullingCascade" + std::to_string(cascadeIndex));
    ensureDepthPyramidShaderReadLayout(commandBuffer);
    vkCmdFillBuffer(commandBuffer, visibleCountBuffer, 0, kGpuCullCountBufferSize, 0);
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
    resetBarriers[0].size = kGpuCullCountBufferSize;

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
    computeBarriers[1].size = kGpuCullCountBufferSize;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(computeBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = computeBarriers.data();
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    VkBufferCopy visibleCountCopy{};
    visibleCountCopy.size = kGpuCullCountBufferSize;
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
    readbackBarrier.size = kGpuCullCountBufferSize;

    VkDependencyInfo readbackDependencyInfo{};
    readbackDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    readbackDependencyInfo.bufferMemoryBarrierCount = 1;
    readbackDependencyInfo.pBufferMemoryBarriers = &readbackBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &readbackDependencyInfo);

    frameGpuShadowCullReadbackReady_[currentFrame_] = 1;
    rhi::debug::endLabel(commandBuffer);
}

void Renderer::ensureDepthPyramidShaderReadLayout(VkCommandBuffer commandBuffer)
{
    if (depthPyramid_.image() == VK_NULL_HANDLE || depthPyramidMipLevels_ == 0 ||
        depthPyramidLayout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        return;
    }

    const bool undefined = depthPyramidLayout_ == VK_IMAGE_LAYOUT_UNDEFINED;
    const VkImageMemoryBarrier2 shaderReadBarrier =
        imageBarrier(depthPyramid_.image(),
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     depthPyramidLayout_,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     undefined ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     undefined ? VK_ACCESS_2_NONE
                               : (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     0,
                     depthPyramidMipLevels_);
    recordImageBarrier(commandBuffer, shaderReadBarrier);
    depthPyramidLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void Renderer::recordDepthPyramidCommands(VkCommandBuffer commandBuffer)
{
    if (!depthPyramidBuildAvailable_ || depthPyramid_.image() == VK_NULL_HANDLE ||
        depthPyramidPipeline_.pipeline() == VK_NULL_HANDLE ||
        depthPyramidPipeline_.layout() == VK_NULL_HANDLE || depthPyramidDescriptorSets_.empty() ||
        depthPyramidMipImageViews_.empty() || depthPyramidMipLevels_ == 0) {
        depthPyramidValid_ = false;
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, "DepthPyramid");
    renderGraph_.beginDepthPyramidPass();
    rhi::debug::beginLabel(commandBuffer, "DepthPyramid");

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, depthPyramidPipeline_.pipeline());

    const VkExtent2D baseExtent = swapchain_.extent();
    for (uint32_t mipLevel = 0; mipLevel < depthPyramidMipLevels_; ++mipLevel) {
        const VkExtent2D sourceExtent = mipLevel == 0 ? baseExtent : mipExtent(baseExtent, mipLevel - 1);
        const VkExtent2D destinationExtent = mipExtent(baseExtent, mipLevel);
        const VkDescriptorSet descriptorSet = depthPyramidDescriptorSets_[mipLevel];
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                depthPyramidPipeline_.layout(),
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);

        const DepthPyramidPushConstants pushConstants{
            glm::uvec4(sourceExtent.width, sourceExtent.height, destinationExtent.width, destinationExtent.height)};
        vkCmdPushConstants(commandBuffer,
                           depthPyramidPipeline_.layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(DepthPyramidPushConstants)),
                           &pushConstants);

        const uint32_t groupCountX = (destinationExtent.width + kDepthPyramidLocalSizeX - 1) / kDepthPyramidLocalSizeX;
        const uint32_t groupCountY =
            (destinationExtent.height + kDepthPyramidLocalSizeY - 1) / kDepthPyramidLocalSizeY;
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

        const VkImageMemoryBarrier2 mipWriteToRead = imageBarrier(depthPyramid_.image(),
                                                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                                                  mipLevel,
                                                                  1);
        recordImageBarrier(commandBuffer, mipWriteToRead);
    }

    const VkImageMemoryBarrier2 pyramidToShaderRead =
        imageBarrier(depthPyramid_.image(),
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     0,
                     depthPyramidMipLevels_);
    recordImageBarrier(commandBuffer, pyramidToShaderRead);
    depthPyramidLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthPyramidValid_ = true;
    depthPyramidViewProjection_ = frameViewProjection_;
    depthPyramidCameraPosition_ = frameCameraPosition_;

    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endDepthPyramidPass();
}

void Renderer::recordLuminanceCommands(VkCommandBuffer commandBuffer)
{
    if (!isLogAverageExposureActive() || exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual ||
        currentFrame_ >= frameLuminanceBuffers_.size()) {
        return;
    }

    VkBuffer luminanceBuffer = frameLuminanceBuffers_[currentFrame_].buffer();
    if (luminanceBuffer == VK_NULL_HANDLE) {
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, "LuminancePass");
    renderGraph_.beginLuminancePass();

    rhi::debug::beginLabel(commandBuffer, "LuminancePass");
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminancePipeline_.pipeline());

    const VkDescriptorSet descriptorSet = activeLuminanceDescriptorSet();
    if (descriptorSet == VK_NULL_HANDLE) {
        renderGraph_.endLuminancePass();
        rhi::debug::endLabel(commandBuffer);
        return;
    }
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

    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endLuminancePass();
}

void Renderer::recordHistogramCommands(VkCommandBuffer commandBuffer)
{
    if (!isHistogramExposureActive() || exposureModeValue(toneMappingSettings_.exposureMode) == ExposureMode::Manual ||
        currentFrame_ >= frameHistogramBuffers_.size() ||
        currentFrame_ >= frameExposureReadbackReady_.size()) {
        return;
    }

    VkBuffer histogramBuffer = frameHistogramBuffers_[currentFrame_].buffer();
    if (histogramBuffer == VK_NULL_HANDLE) {
        return;
    }

    const VkExtent3D sceneExtent = sceneColor_.extent();
    const uint32_t groupCountX = (sceneExtent.width + kHistogramLocalSizeX - 1) / kHistogramLocalSizeX;
    const uint32_t groupCountY = (sceneExtent.height + kHistogramLocalSizeY - 1) / kHistogramLocalSizeY;
    if (groupCountX == 0 || groupCountY == 0) {
        return;
    }

    const renderer::GpuProfileScope profileScope(gpuProfiler_, currentFrame_, commandBuffer, "Histogram Exposure");
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

    const VkDescriptorSet descriptorSet = activeHistogramDescriptorSet();
    if (descriptorSet == VK_NULL_HANDLE) {
        renderGraph_.endHistogramExposurePass();
        rhi::debug::endLabel(commandBuffer);
        return;
    }
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
    computeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
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

    if (isGpuExposureActive() && currentFrame_ < exposureReduceDescriptorSets_.size() &&
        currentFrame_ < frameExposureBuffers_.size()) {
        VkBuffer exposureBuffer = frameExposureBuffers_[currentFrame_].buffer();
        if (exposureBuffer != VK_NULL_HANDLE) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposureReducePipeline_.pipeline());

            const VkDescriptorSet exposureDescriptorSet = exposureReduceDescriptorSets_[currentFrame_];
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    exposureReducePipeline_.layout(),
                                    0,
                                    1,
                                    &exposureDescriptorSet,
                                    0,
                                    nullptr);

            const auto now = std::chrono::steady_clock::now();
            const float deltaTime =
                std::max(0.0f, std::chrono::duration<float>(now - lastAutoExposureUpdate_).count());
            lastAutoExposureUpdate_ = now;
            const auto [lowPercentile, highPercentile] =
                sanitizedPercentileRange(toneMappingSettings_.lowPercentile, toneMappingSettings_.highPercentile);
            const auto [reduceMinLogLuminance, reduceMaxLogLuminance] = sanitizedHistogramLogRange(
                toneMappingSettings_.histogramMinLogLuminance, toneMappingSettings_.histogramMaxLogLuminance);
            const float minExposure = std::max(toneMappingSettings_.minExposure, 0.0f);
            const float maxExposure = std::max(toneMappingSettings_.maxExposure, minExposure);
            const ExposureMode mode = exposureModeValue(toneMappingSettings_.exposureMode);
            const ExposureReducePushConstants exposurePushConstants{
                glm::uvec4(static_cast<uint32_t>(mode), luminancePartialCount_, kHistogramBinCount, 0),
                glm::vec4(toneMappingExposureValue(toneMappingSettings_.manualExposure),
                          std::max(toneMappingSettings_.targetLuminance, kMinAverageLuminance),
                          minExposure,
                          maxExposure),
                glm::vec4(deltaTime, std::max(toneMappingSettings_.adaptationRate, 0.0f), lowPercentile, highPercentile),
                glm::vec4(reduceMinLogLuminance, reduceMaxLogLuminance, kMinAverageLuminance, 0.0f)};
            vkCmdPushConstants(commandBuffer,
                               exposureReducePipeline_.layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(ExposureReducePushConstants)),
                               &exposurePushConstants);

            rhi::debug::beginLabel(commandBuffer, "ExposureReduce");
            vkCmdDispatch(commandBuffer, 1, 1, 1);
            rhi::debug::endLabel(commandBuffer);

            VkBufferMemoryBarrier2 exposureBarrier{};
            exposureBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            exposureBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            exposureBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            exposureBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            exposureBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
            exposureBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            exposureBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            exposureBarrier.buffer = exposureBuffer;
            exposureBarrier.offset = 0;
            exposureBarrier.size = sizeof(ExposureState);

            VkDependencyInfo exposureDependencyInfo{};
            exposureDependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            exposureDependencyInfo.bufferMemoryBarrierCount = 1;
            exposureDependencyInfo.pBufferMemoryBarriers = &exposureBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &exposureDependencyInfo);

            frameExposureReadbackReady_[currentFrame_] = 1;
        }
    }
    rhi::debug::endLabel(commandBuffer);

    renderGraph_.endHistogramExposurePass();
}

void Renderer::recordTaaResolveCommands(VkCommandBuffer commandBuffer)
{
    if (!isTaaActive()) {
        return;
    }

    const uint32_t readIndex = taaHistoryReadIndex();
    const VkDescriptorSet descriptorSet = taaResolveDescriptorSets_[readIndex];
    if (descriptorSet == VK_NULL_HANDLE) {
        return;
    }

    const VkExtent3D sceneExtent = sceneColor_.extent();
    if (sceneExtent.width == 0 || sceneExtent.height == 0) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "TAAResolvePass");
    const bool taaProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "TAAResolvePass");
    renderGraph_.beginTaaResolvePass();
    setViewportAndScissor(commandBuffer, VkExtent2D{sceneExtent.width, sceneExtent.height});
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, taaResolvePipeline_.pipeline());
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            taaResolvePipeline_.layout(),
                            0,
                            1,
                            &descriptorSet,
                            0,
                            nullptr);

    const TaaResolvePushConstants pushConstants{
        glm::vec2{1.0f / static_cast<float>(sceneExtent.width), 1.0f / static_cast<float>(sceneExtent.height)},
        taaSettings_.feedback,
        taaHistoryValid_ ? 1u : 0u,
        taaSettings_.neighborhoodClampEnabled ? 1u : 0u,
        {0u, 0u, 0u}};
    vkCmdPushConstants(commandBuffer,
                       taaResolvePipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(TaaResolvePushConstants)),
                       &pushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endTaaResolvePass();
    if (taaProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
}

void Renderer::recordLegacyBloomCommands(VkCommandBuffer commandBuffer)
{
    rhi::debug::beginLabel(commandBuffer, "BloomExtractPass");
    const bool bloomExtractProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "BloomExtractPass");
    renderGraph_.beginBloomExtractPass();
    setViewportAndScissor(commandBuffer, bloomExtent_);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomExtractPipeline_.pipeline());
    const VkDescriptorSet bloomExtractDescriptorSet = activeBloomExtractDescriptorSet();
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            bloomExtractPipeline_.layout(),
                            0,
                            1,
                            &bloomExtractDescriptorSet,
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
    if (bloomExtractProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    const BloomBlurPushConstants horizontalBlurPushConstants{
        glm::vec2{1.0f / static_cast<float>(bloomExtent_.width), 1.0f / static_cast<float>(bloomExtent_.height)},
        1u,
        0u};
    rhi::debug::beginLabel(commandBuffer, "BloomBlurHorizontal");
    const bool bloomBlurHorizontalProfileScope =
        gpuProfiler_.beginScope(currentFrame_, commandBuffer, "BloomBlurHorizontal");
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
    if (bloomBlurHorizontalProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    const BloomBlurPushConstants verticalBlurPushConstants{
        glm::vec2{1.0f / static_cast<float>(bloomExtent_.width), 1.0f / static_cast<float>(bloomExtent_.height)},
        0u,
        0u};
    rhi::debug::beginLabel(commandBuffer, "BloomBlurVertical");
    const bool bloomBlurVerticalProfileScope =
        gpuProfiler_.beginScope(currentFrame_, commandBuffer, "BloomBlurVertical");
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
    if (bloomBlurVerticalProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
}

void Renderer::recordMipChainBloomCommands(VkCommandBuffer commandBuffer)
{
    if (bloomMipDownsampleImages_.empty() ||
        bloomMipDownsampleDescriptorSets_.size() != bloomMipDownsampleImages_.size() ||
        bloomDownsamplePipeline_.pipeline() == VK_NULL_HANDLE ||
        bloomDownsamplePipeline_.layout() == VK_NULL_HANDLE) {
        return;
    }

    rhi::debug::beginLabel(commandBuffer, "BloomMipChain");

    const bool downsampleProfileScope =
        gpuProfiler_.beginScope(currentFrame_, commandBuffer, "Bloom Downsample Chain");
    rhi::debug::beginLabel(commandBuffer, "Bloom Downsample Chain");
    for (uint32_t level = 0; level < bloomMipDownsampleImages_.size(); ++level) {
        const VkExtent2D sourceExtent =
            level == 0 ? swapchain_.extent() : VkExtent2D{bloomMipDownsampleImages_[level - 1].extent().width,
                                                          bloomMipDownsampleImages_[level - 1].extent().height};
        const VkExtent3D outputExtent = bloomMipDownsampleImages_[level].extent();
        const VkExtent2D outputSize{outputExtent.width, outputExtent.height};

        rhi::debug::beginLabel(commandBuffer, "BloomDownsampleMip" + std::to_string(level));
        renderGraph_.beginBloomDownsamplePass(level);
        setViewportAndScissor(commandBuffer, outputSize);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomDownsamplePipeline_.pipeline());
        const VkDescriptorSet descriptorSet = activeBloomMipDownsampleDescriptorSet(level);
        if (descriptorSet == VK_NULL_HANDLE) {
            renderGraph_.endBloomDownsamplePass();
            rhi::debug::endLabel(commandBuffer);
            continue;
        }
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                bloomDownsamplePipeline_.layout(),
                                0,
                                1,
                                &descriptorSet,
                                0,
                                nullptr);
        const BloomDownsamplePushConstants pushConstants{
            glm::vec2{1.0f / static_cast<float>(sourceExtent.width),
                      1.0f / static_cast<float>(sourceExtent.height)},
            bloomSettings_.threshold,
            level == 0 ? 1u : 0u};
        vkCmdPushConstants(commandBuffer,
                           bloomDownsamplePipeline_.layout(),
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(BloomDownsamplePushConstants)),
                           &pushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        renderGraph_.endBloomDownsamplePass();
        rhi::debug::endLabel(commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);
    if (downsampleProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    if (!bloomMipUpsampleImages_.empty() &&
        bloomMipUpsampleDescriptorSets_.size() == bloomMipUpsampleImages_.size() &&
        bloomUpsamplePipeline_.pipeline() != VK_NULL_HANDLE && bloomUpsamplePipeline_.layout() != VK_NULL_HANDLE) {
        const bool upsampleProfileScope =
            gpuProfiler_.beginScope(currentFrame_, commandBuffer, "Bloom Upsample Chain");
        rhi::debug::beginLabel(commandBuffer, "Bloom Upsample Chain");
        for (uint32_t reverseIndex = 0; reverseIndex < bloomMipUpsampleImages_.size(); ++reverseIndex) {
            const uint32_t level =
                static_cast<uint32_t>(bloomMipUpsampleImages_.size() - 1u - reverseIndex);
            const VkExtent3D outputExtent = bloomMipUpsampleImages_[level].extent();
            const VkExtent2D outputSize{outputExtent.width, outputExtent.height};
            const VkExtent3D lowerExtent =
                level + 1u == bloomMipDownsampleImages_.size() - 1u
                    ? bloomMipDownsampleImages_[level + 1u].extent()
                    : bloomMipUpsampleImages_[level + 1u].extent();

            rhi::debug::beginLabel(commandBuffer, "BloomUpsampleMip" + std::to_string(level));
            renderGraph_.beginBloomUpsamplePass(level);
            setViewportAndScissor(commandBuffer, outputSize);
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bloomUpsamplePipeline_.pipeline());
            const VkDescriptorSet descriptorSet = bloomMipUpsampleDescriptorSets_[level];
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    bloomUpsamplePipeline_.layout(),
                                    0,
                                    1,
                                    &descriptorSet,
                                    0,
                                    nullptr);
            const BloomUpsamplePushConstants pushConstants{
                glm::vec2{1.0f / static_cast<float>(lowerExtent.width),
                          1.0f / static_cast<float>(lowerExtent.height)},
                bloomSettings_.radius,
                0.0f};
            vkCmdPushConstants(commandBuffer,
                               bloomUpsamplePipeline_.layout(),
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(BloomUpsamplePushConstants)),
                               &pushConstants);
            vkCmdDraw(commandBuffer, 3, 1, 0, 0);
            renderGraph_.endBloomUpsamplePass();
            rhi::debug::endLabel(commandBuffer);
        }
        rhi::debug::endLabel(commandBuffer);
        if (upsampleProfileScope) {
            gpuProfiler_.endScope(currentFrame_, commandBuffer);
        }
    }

    rhi::debug::endLabel(commandBuffer);
}

void Renderer::recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t mainDrawItemCount = visibleDrawItems_.size();
    const bool taaActiveThisFrame = isTaaActive();
    taaPostProcessHistoryIndex_ = taaActiveThisFrame ? taaHistoryWriteIndex() : 0u;

    renderGraph_.beginFrame(commandBuffer, swapchain_, shadowMap_, imageIndex, renderGraphFrameResources());
    rhi::debug::beginLabel(commandBuffer, "Frame");
    gpuProfiler_.beginFrame(currentFrame_, commandBuffer);

    const bool gpuShadowCullingActive = isGpuShadowCullingActive() && !allDrawItems_.empty();
    const uint32_t cascadeCount = activeCascadeCount();

    const bool csmProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "CSMShadowPass");
    rhi::debug::beginLabel(commandBuffer, "CSMShadowPass");

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

    rhi::debug::endLabel(commandBuffer);
    if (csmProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    recordGpuCullingCommands(commandBuffer);

    const bool mainHdrProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "MainHDRPass");
    rhi::debug::beginLabel(commandBuffer, "MainHDRPass");
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

    if (skyboxDescriptorSet_ != VK_NULL_HANDLE) {
        const renderer::GpuProfileScope skyboxScope(gpuProfiler_, currentFrame_, commandBuffer, "Skybox");
        rhi::debug::beginLabel(commandBuffer, "Skybox");
        glm::mat4 skyboxView = camera_.viewMatrix();
        skyboxView[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::mat4 projection = frameJitteredProjection_;
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
        rhi::debug::endLabel(commandBuffer);
    }

    const std::string objectDrawLabel = cullingStats_.gpuCulling
                                            ? "MainHDRPass IndirectDrawItems " + std::to_string(mainDrawItemCount) +
                                                  " GPU culling batches " + std::to_string(meshDrawBatches_.size())
                                            : "MainHDRPass IndirectDrawItems " + std::to_string(mainDrawItemCount) +
                                                  " visible objects " + std::to_string(cullingStats_.visibleObjects) +
                                                  "/" + std::to_string(cullingStats_.totalObjects) + " batches " +
                                                  std::to_string(meshDrawBatches_.size());
    const bool renderObjectsProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "RenderObjects");
    rhi::debug::beginLabel(commandBuffer, objectDrawLabel);
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
    rhi::debug::endLabel(commandBuffer);
    if (renderObjectsProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    renderGraph_.endMainHdrPass();
    rhi::debug::endLabel(commandBuffer);
    if (mainHdrProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    recordDepthPyramidCommands(commandBuffer);

    if (taaActiveThisFrame) {
        recordTaaResolveCommands(commandBuffer);
    }

    recordLegacyBloomCommands(commandBuffer);
    recordMipChainBloomCommands(commandBuffer);

    recordLuminanceCommands(commandBuffer);
    recordHistogramCommands(commandBuffer);

    rhi::debug::beginLabel(commandBuffer, "CompositePass");
    const bool compositeProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "CompositePass");
    renderGraph_.beginCompositePass();
    setViewportAndScissor(commandBuffer, swapchain_.extent());
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_.pipeline());
    const VkDescriptorSet compositeDescriptorSet = activeCompositeDescriptorSet();
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            compositePipeline_.layout(),
                            0,
                            1,
                            &compositeDescriptorSet,
                            0,
                            nullptr);
    CompositePushConstants compositePushConstants{
        currentToneMappingExposure(),
        bloomSettings_.enabled ? std::max(bloomSettings_.intensity, 0.0f) : 0.0f,
        toneMappingOperatorValue(toneMappingSettings_.operatorType),
        bloomSettings_.enabled ? 1u : 0u,
        bloomSettings_.useMipChain && (!bloomMipUpsampleImages_.empty() || !bloomMipDownsampleImages_.empty()) ? 1u
                                                                                                               : 0u,
        isGpuExposureActive() ? 1u : 0u};
    const bool ssaoActive = ssaoSettings_.enabled && ssaoAvailable_;
    compositePushConstants.invProjection = glm::inverse(frameJitteredProjection_);
    compositePushConstants.ssaoParams0 = glm::vec4(std::max(ssaoSettings_.radius, 0.0f),
                                                   ssaoSettings_.bias,
                                                   std::max(ssaoSettings_.intensity, 0.0f),
                                                   std::max(ssaoSettings_.power, 0.0001f));
    compositePushConstants.ssaoParams1 =
        glm::vec4(ssaoActive ? 1.0f : 0.0f, static_cast<float>(std::max(ssaoSettings_.sampleCount, 1)), 0.0f, 0.0f);
    vkCmdPushConstants(commandBuffer,
                       compositePipeline_.layout(),
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       static_cast<uint32_t>(sizeof(CompositePushConstants)),
                       &compositePushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    renderGraph_.endCompositePass();
    if (compositeProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    recordPortfolioScreenshotCopy(commandBuffer, imageIndex);

    rhi::debug::beginLabel(commandBuffer, "ImGuiPass");
    const bool imguiProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "ImGuiPass");
    renderGraph_.beginImGuiPass();
    imguiLayer_.render(commandBuffer);
    renderGraph_.endImGuiPass();
    if (imguiProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    if (taaActiveThisFrame) {
        taaHistoryValid_ = true;
        taaHistoryWriteIndex_ = (taaHistoryWriteIndex_ + 1u) % kTaaHistoryCount;
    }

    gpuProfiler_.endFrame(currentFrame_, commandBuffer);
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endFrame();
}

} // namespace ve
