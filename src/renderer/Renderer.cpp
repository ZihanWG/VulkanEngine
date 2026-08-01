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
#include <cctype>
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

    context_.initialize(window_, shaderDirectory());

    frames_.resize(rhi::kMaxFramesInFlight);
    screenshotCapture_.initialize(
        context_, static_cast<uint32_t>(frames_.size()), portfolioScreenshotDirectory());
    gpuProfiler_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    swapchain_.initialize(context_, window_.framebufferExtent());
    imguiLayer_.initialize(window_, context_, swapchain_.colorFormat(), swapchain_.imageCount());
    createMaterialDescriptorSetLayout();
    createBindlessMaterialTextureHeap();
    createSkyboxDescriptorSetLayout();
    postProcess_.createPostProcessDescriptorSetLayouts();
    ssr_.createDescriptorSetLayout();
    gtao_.createDescriptorSetLayout();
    createDepthPyramidDescriptorSetLayout();
    postProcess_.createPostProcessSampler();
    createShadowMap();
    recreatePostProcessResources();
    createPipeline();
    commandContext_.initialize(context_, frames_);
    asyncCompute_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    createScene();
    createObjectFrameDataBuffers();
    clusteredLighting_.create(context_,
                              static_cast<uint32_t>(frames_.size()),
                              shaderPath("cluster_build.comp.spv"),
                              shaderPath("light_cull.comp.spv"));
    updateDemoLights(0.0f);
    // Prefer a rigged glTF if one is present; otherwise fall back to the
    // self-contained procedural bone chain.
    if (!skinnedMesh_.createFromGltf(context_,
                                     commandContext_,
                                     static_cast<uint32_t>(frames_.size()),
                                     assetPath("models/skinned_rig.gltf"))) {
        skinnedMesh_.create(context_, commandContext_, static_cast<uint32_t>(frames_.size()));
    }
    createIndirectDrawBuffers();
    createShadowIndirectDrawBuffers();
    createGpuCullingResources();
    sync_.initialize(context_, frames_, swapchain_.imageCount());
    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
    currentExposure_ = toneMappingExposureValue(toneMappingSettings_.manualExposure);
    averageLuminance_ = toneMappingSettings_.targetLuminance;
    histogramClippedLuminance_ = toneMappingSettings_.targetLuminance;
    lastExposureLogPrint_ = std::chrono::steady_clock::now();

    initialized_ = true;
}

Renderer::~Renderer()
{
    if (initialized_) {
        waitIdle();
        imguiLayer_.shutdown();
        screenshotCapture_.shutdown();
        destroyDepthPyramidResources();
        // PostProcessStack owns the post-process sampler/pool/resources and frees
        // them in its destructor (RAII), so nothing post-process to free here.
    }
}

// Wrapper around PostProcessStack::createPostProcessResources that keeps the
// non-post-process work that was interleaved in the former monolithic method:
// invalidating ImGui render-target previews, resetting the bloom-mip debug
// selection, and recreating the depth pyramid (a GPU-culling resource).
void Renderer::recreatePostProcessResources()
{
    imguiLayer_.clearRenderTargetPreviewDescriptors();
    selectedBloomMipDebugLevel_ = 0;
    destroyDepthPyramidResources();
    postProcess_.createPostProcessResources(checkerboardTexture_.imageView(), static_cast<uint32_t>(frames_.size()));
    ssr_.createResources(postProcess_.normalRoughness().imageView(), static_cast<uint32_t>(frames_.size()));
    gtao_.createResources(postProcess_.normalRoughness().imageView(), static_cast<uint32_t>(frames_.size()));
    createDepthPyramidResources();
}

void Renderer::invalidateTaaHistory()
{
    // The jittered / previous view-projection matrices are Renderer frame state
    // consumed by the main pass, so reset them here; the TAA history and jitter
    // sequence are owned by PostProcessStack.
    frameJitteredProjection_ = glm::mat4{1.0f};
    frameJitteredViewProjection_ = glm::mat4{1.0f};
    previousFrameViewProjection_ = glm::mat4{1.0f};
    previousFrameViewProjectionValid_ = false;
    postProcess_.invalidateTaaHistory();
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
    postProcess_.updateAutoExposureFromReadback(currentFrame_);
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
    {
        const auto framePrepStart = std::chrono::steady_clock::now();
        updateFrameData(currentFrame_);
        framePrepCpuHistory_.push(
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - framePrepStart).count());
    }

    // Async compute: cluster build + light cull go to the compute queue before
    // the graphics command buffer is even recorded, so the GPU overlaps them
    // with the shadow passes (and with this CPU recording). The graphics submit
    // below waits on the semaphore at FRAGMENT_SHADER — the first stage that
    // reads the cluster buffers — so shadow/culling work is never blocked.
    if (frameAsyncComputeActive_) {
        const VkCommandBuffer asyncCommandBuffer = asyncCompute_.commandBuffer(currentFrame_);
        VK_CHECK(vkResetCommandBuffer(asyncCommandBuffer, 0));

        VkCommandBufferBeginInfo asyncBeginInfo{};
        asyncBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        asyncBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(asyncCommandBuffer, &asyncBeginInfo));
        rhi::debug::beginLabel(asyncCommandBuffer, "AsyncClusteredLighting");
        clusteredLighting_.recordClusterBuild(asyncCommandBuffer, currentFrame_, /*asyncQueue=*/true);
        clusteredLighting_.recordLightCull(asyncCommandBuffer, currentFrame_, /*asyncQueue=*/true);
        rhi::debug::endLabel(asyncCommandBuffer);
        VK_CHECK(vkEndCommandBuffer(asyncCommandBuffer));

        VkCommandBufferSubmitInfo asyncCommandBufferInfo{};
        asyncCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        asyncCommandBufferInfo.commandBuffer = asyncCommandBuffer;

        VkSemaphoreSubmitInfo asyncSignalSemaphore{};
        asyncSignalSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        asyncSignalSemaphore.semaphore = asyncCompute_.semaphore(currentFrame_);
        asyncSignalSemaphore.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

        VkSubmitInfo2 asyncSubmitInfo{};
        asyncSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        asyncSubmitInfo.commandBufferInfoCount = 1;
        asyncSubmitInfo.pCommandBufferInfos = &asyncCommandBufferInfo;
        asyncSubmitInfo.signalSemaphoreInfoCount = 1;
        asyncSubmitInfo.pSignalSemaphoreInfos = &asyncSignalSemaphore;
        VK_CHECK(vkQueueSubmit2(asyncCompute_.queue(), 1, &asyncSubmitInfo, VK_NULL_HANDLE));
    }

    recordRenderCommands(frame.commandBuffer, imageIndex);
    const VkSemaphore renderFinished = sync_.renderFinishedSemaphore(imageIndex);

    std::array<VkSemaphoreSubmitInfo, 2> waitSemaphores{};
    waitSemaphores[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemaphores[0].semaphore = frame.imageAvailable;
    waitSemaphores[0].stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    uint32_t waitSemaphoreCount = 1;
    if (frameAsyncComputeActive_) {
        waitSemaphores[waitSemaphoreCount].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waitSemaphores[waitSemaphoreCount].semaphore = asyncCompute_.semaphore(currentFrame_);
        // First stage that reads the cluster grid / light index buffers; shadow
        // depth-only rasterization and the culling compute run unblocked.
        waitSemaphores[waitSemaphoreCount].stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        ++waitSemaphoreCount;
    }

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = frame.commandBuffer;

    VkSemaphoreSubmitInfo signalSemaphore{};
    signalSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphore.semaphore = renderFinished;
    signalSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = waitSemaphoreCount;
    submitInfo.pWaitSemaphoreInfos = waitSemaphores.data();
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalSemaphore;

    VK_CHECK(vkQueueSubmit2(context_.graphicsQueue(), 1, &submitInfo, frame.inFlightFence));
    gpuProfiler_.markFrameSubmitted(currentFrame_);
    capturePreviousFrameMatrices();

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
        screenshotCapture_.setStatus("Screenshot capture is already queued.");
        return;
    }
    if (!swapchain_.supportsTransferSrc()) {
        screenshotCapture_.setStatus("Screenshot unavailable: swapchain transfer-source usage is unsupported.");
        Logger::warn(screenshotCapture_.status());
        return;
    }
    if (!supportedScreenshotFormat(swapchain_.colorFormat())) {
        screenshotCapture_.setStatus(
            std::string("Screenshot unavailable: unsupported swapchain format ") + vkFormatName(swapchain_.colorFormat()) + ".");
        Logger::warn(screenshotCapture_.status());
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
        screenshotCapture_.setStatus("Screenshot unavailable: portfolio showcase scene could not be activated.");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    portfolioScreenshotRequested_ = true;
    screenshotCapture_.setStatus("Portfolio showcase screenshot queued for the next rendered frame.");
}

bool Renderer::hasPendingPortfolioScreenshotReadback() const
{
    return screenshotCapture_.hasPending();
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
        screenshotCapture_.setStatus(
            "Screenshot deferred: portfolio capture mode was not active for the frame being copied.");
        Logger::warn(screenshotCapture_.status());
        return;
    }
    if (!currentFrameHasPortfolioShowcaseDrawItems()) {
        applyPortfolioCaptureSettings();
        screenshotCapture_.setStatus(
            "Screenshot deferred: portfolio showcase draw items were not active for the frame being copied.");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    portfolioScreenshotRequested_ = false;
    if (!swapchain_.supportsTransferSrc()) {
        screenshotCapture_.setStatus("Screenshot failed: swapchain transfer-source usage is unsupported.");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    const VkFormat format = swapchain_.colorFormat();
    if (extent.width == 0 || extent.height == 0 || !supportedScreenshotFormat(format)) {
        screenshotCapture_.setStatus(
            std::string("Screenshot failed: unsupported extent or format ") + vkFormatName(format) + ".");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    // Policy gate passed; hand the validated frame to the capture subsystem, which
    // owns the readback buffers and records the copy + barriers.
    screenshotCapture_.recordCopy(commandBuffer, currentFrame_, swapchain_, imageIndex);
}

void Renderer::processPortfolioScreenshotReadback(uint32_t frameIndex)
{
    screenshotCapture_.processReadback(frameIndex);
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
    shadowMap_.create(context_, shadowSettings_.resolution, shadowSettings_.resolution, activeCascadeCount());
}

void Renderer::createPipeline()
{
    createMainGraphicsPipeline();
    createSkinnedPipeline();
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
}

void Renderer::createComputePipelines()
{
    depthPyramid_.createPipeline();

    // Exposure compute pipelines (luminance/histogram/reduce) now live in PostProcessStack.
    postProcess_.createExposureComputePipelines();
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

renderer::SceneBuilder Renderer::makeSceneBuilder()
{
    return renderer::SceneBuilder(
        cubeMesh_, portfolioSphereMesh_, materialVariants_, [this] { return allocateRenderObjectDebugId(); });
}

void Renderer::createScene()
{
    resetSceneState();
    createSceneSharedResources();

    // The portfolio sphere showcase is the default editor scene. The glTF import
    // (tryLoadGltfScene) and the cube fallback remain available through the
    // scene-loading UI; they are just no longer the startup default.
    makeSceneBuilder().appendPortfolioShowcase(renderObjects_);
    if (renderObjects_.empty()) {
        Logger::warn("Portfolio showcase scene unavailable; using built-in cube fallback scene.");
        makeSceneBuilder().appendCubeFallback(renderObjects_);
    }
}

void Renderer::resetSceneState()
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
}

void Renderer::createSceneSharedResources()
{
    cubeMesh_ = renderer::Mesh::createCube(context_, commandContext_);
    portfolioSphereMesh_ = renderer::Mesh::createUvSphere(context_, commandContext_);
    const std::filesystem::path builtinAssetDir = assetDirectory();
    builtinTextureFactory_.createCheckerboardBaseColor(context_, commandContext_, builtinAssetDir, checkerboardTexture_);
    builtinTextureFactory_.createPortfolioBaseColor(context_, commandContext_, portfolioBaseColorTexture_);
    builtinTextureFactory_.createPortfolioBackdrop(context_, commandContext_, portfolioBackdropTexture_);
    builtinTextureFactory_.createNormal(
        context_, commandContext_, builtinAssetDir, normalMapTexture_, flatNormalTexture_, normalMapAssetLoaded_);
    builtinTextureFactory_.createMetallicRoughness(context_,
                                                   commandContext_,
                                                   builtinAssetDir,
                                                   metallicRoughnessTexture_,
                                                   neutralMetallicRoughnessTexture_,
                                                   metallicRoughnessMapAssetLoaded_);
    createEnvironmentMap();
    createMaterial();

    // Frame the portfolio sphere showcase, which is now the default editor scene.
    camera_ = portfolioCameraPreset();
    csmSettings_.nearPlane = camera_.nearPlane;
    csmSettings_.farPlane = camera_.farPlane;
}

bool Renderer::tryLoadGltfScene()
{
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
            makeSceneBuilder().appendPortfolioShowcase(renderObjects_);
            return true;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load glTF mesh '" + modelPath.string() + "': " + error.what());
        }
    }

    return false;
}

void Renderer::resetPortfolioShowcaseObjectsToPreset()
{
    renderer::SceneBuilder::resetPortfolioShowcaseToPreset(renderObjects_);
    invalidateDepthPyramid();
    invalidateTaaHistory();
}

bool Renderer::currentFrameHasPortfolioShowcaseDrawItems() const
{
    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex >= renderObjects_.size()) {
            continue;
        }

        const renderer::RenderObject& object = renderObjects_[drawItem.objectIndex];
        if (object.sourceType == renderer::RenderObjectSourceType::PortfolioShowcase && !object.hideInPortfolio) {
            return true;
        }
    }

    return false;
}

bool Renderer::ensurePortfolioShowcaseSceneReady()
{
    if (renderer::SceneBuilder::hasPortfolioShowcase(renderObjects_)) {
        return true;
    }

    if (!cubeMesh_.valid() || !portfolioSphereMesh_.valid() ||
        materialVariants_.size() <= renderer::kPortfolioBackdropMaterialIndex) {
        screenshotCapture_.setStatus(
            "Portfolio showcase scene is unavailable: required meshes or materials are not initialized.");
        Logger::warn(screenshotCapture_.status());
        return false;
    }

    Logger::warn("Portfolio showcase scene was missing; rebuilding portfolio-only showcase objects.");
    makeSceneBuilder().appendPortfolioShowcase(renderObjects_);
    if (renderer::SceneBuilder::hasPortfolioShowcase(renderObjects_)) {
        return true;
    }

    screenshotCapture_.setStatus("Portfolio showcase scene is unavailable after rebuild.");
    Logger::warn(screenshotCapture_.status());
    return false;
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

    makeSceneBuilder().appendOcclusionTest(renderObjects_, occlusionTestSceneStatus_);
    invalidateDepthPyramid();
    invalidateTaaHistory();
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
    material.emissiveFactor = glm::max(materialAsset.emissiveFactor, glm::vec3(0.0f));
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
    materialAsset.emissiveFactor = material.emissiveFactor;
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

namespace {

// Turns a free-form material name into a filesystem-safe slug for a new asset
// file (lowercase, alphanumerics preserved, runs of other characters collapsed
// to single underscores).
std::string slugifyMaterialName(std::string_view name)
{
    std::string slug;
    slug.reserve(name.size());
    bool pendingSeparator = false;
    for (const char ch : name) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch)) {
            if (pendingSeparator && !slug.empty()) {
                slug.push_back('_');
            }
            pendingSeparator = false;
            slug.push_back(static_cast<char>(std::tolower(uch)));
        } else {
            pendingSeparator = true;
        }
    }
    if (slug.empty()) {
        slug = "material";
    }
    return slug;
}

} // namespace

std::filesystem::path Renderer::makeNewMaterialAssetPath(const renderer::Material& material) const
{
    std::string_view name = !material.assetName.empty() ? std::string_view(material.assetName)
                                                        : std::string_view(material.debugName);
    const std::string slug = slugifyMaterialName(name);

    // Avoid clobbering an existing file on disk by appending a numeric suffix.
    std::filesystem::path candidate = materialAssetPath(slug + ".material.json");
    for (int index = 2; std::filesystem::exists(candidate); ++index) {
        candidate = materialAssetPath(slug + "_" + std::to_string(index) + ".material.json");
    }
    return candidate;
}

bool Renderer::saveMaterialAssetFromUi(renderer::Material& material)
{
    if (material.sourceAssetPath.empty()) {
        // "Save As": this material was created/edited in the UI without a backing
        // file (e.g. a glTF or procedural material). Synthesize a new asset path
        // under assets/materials/ so it can be persisted and reloaded later.
        material.sourceAssetPath = makeNewMaterialAssetPath(material);
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
    material.emissiveFactor = glm::max(materialAsset->emissiveFactor, glm::vec3(0.0f));
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

    createBuiltInMaterialVariants();
    createPortfolioMaterialVariants();

    checkerboardMaterial_ = materialVariants_.front();
}

void Renderer::createBuiltInMaterialVariants()
{
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
}

void Renderer::createPortfolioMaterialVariants()
{
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

    // Give the hero ceramic a soft warm emissive so factor-only emissive is
    // visible in the default scene and reads through bloom. Editable per material
    // from the Material Inspector.
    if (renderer::kPortfolioHeroCeramicMaterialIndex < materialVariants_.size()) {
        materialVariants_[renderer::kPortfolioHeroCeramicMaterialIndex].emissiveFactor = glm::vec3(0.9f, 0.45f, 0.15f);
    }
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
    // Emissive maps are sRGB color, so they share the base-color bindless array.
    // Without one, the index falls back and the shader skips sampling it
    // (hasEmissiveTexture stays false, so emissive uses the factor only).
    if (material.hasEmissiveTexture && material.emissiveTexture && material.emissiveTexture->valid()) {
        material.emissiveTextureIndex = bindlessTextureHeap_.registerTexture(
            renderer::BindlessTextureHeap::TextureKind::BaseColor, *material.emissiveTexture);
    } else {
        material.emissiveTextureIndex = bindlessBaseColorFallbackIndex_;
        material.hasEmissiveTexture = false;
    }
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
        // Emissive maps decode as sRGB into the base-color texture array.
        markNeeded(materialInfo.emissiveTextureIndex, baseColorNeeded);
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
        material.emissiveFactor = materialInfo.emissiveFactor;
        material.hasEmissiveTexture = textureLoaded(materialInfo.emissiveTextureIndex, importedBaseColorTextures_);
        material.emissiveTexture = material.hasEmissiveTexture
                                       ? &importedBaseColorTextures_[static_cast<size_t>(materialInfo.emissiveTextureIndex)]
                                       : nullptr;
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

void Renderer::updateDemoLights(float elapsedSeconds)
{
    constexpr float kDegToRad = 3.1415926535897932385f / 180.0f;
    constexpr float kGoldenAngle = 2.39996322972865332f;

    // Deterministic fractional hash so each light gets a stable radius/height/
    // speed/hue without storing per-light state; the count slider just adds or
    // removes entries from the end of the swarm.
    const auto hash01 = [](int index, float seed) {
        const float value = std::sin(static_cast<float>(index) * 12.9898f + seed) * 43758.5453f;
        return value - std::floor(value);
    };

    // Minimal HSV->RGB for evenly spread, saturated light colors.
    const auto hueColor = [](float hue) {
        const glm::vec3 k{1.0f, 2.0f / 3.0f, 1.0f / 3.0f};
        const glm::vec3 p =
            glm::abs(glm::fract(glm::vec3(hue) + k) * 6.0f - glm::vec3(3.0f));
        return glm::clamp(p - glm::vec3(1.0f), glm::vec3(0.0f), glm::vec3(1.0f));
    };

    clusteredLighting_.clear();

    const int lightCount = std::clamp(demoLightCount_, 0, 512);
    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        const float radius = 2.0f + 4.0f * hash01(lightIndex, 0.0f);
        const float height = 0.6f + 4.0f * hash01(lightIndex, 7.0f);
        const float speed = 0.15f + 0.55f * hash01(lightIndex, 13.0f);
        const float baseAngle = static_cast<float>(lightIndex) * kGoldenAngle;
        const float angle = baseAngle + (animateLights_ ? elapsedSeconds * speed : 0.0f);
        const glm::vec3 position{radius * std::cos(angle), height, radius * std::sin(angle)};
        const glm::vec3 color = hueColor(hash01(lightIndex, 21.0f));
        clusteredLighting_.addPointLight(position, color, demoLightIntensity_, demoLightRange_);
    }

    // A white overhead spot anchors the scene regardless of the swarm size.
    const float spotAngle = animateLights_ ? elapsedSeconds * 0.25f : 0.0f;
    clusteredLighting_.addSpotLight(glm::vec3{2.5f * std::cos(spotAngle), 6.0f, 2.5f * std::sin(spotAngle)},
                                    glm::vec3{0.0f, -1.0f, 0.0f},
                                    glm::vec3{1.0f, 1.0f, 1.0f},
                                    40.0f,
                                    16.0f,
                                    18.0f * kDegToRad,
                                    30.0f * kDegToRad);
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

void Renderer::framePrepParallelFor(size_t count, const std::function<void(size_t, size_t)>& body)
{
    // Chunks below this size cost more to dispatch than to run inline.
    constexpr size_t kMinChunkSize = 64;
    if (parallelFramePrepEnabled_) {
        jobSystem_.parallelFor(count, kMinChunkSize, body);
    } else if (count > 0) {
        body(0, count);
    }
}

void Renderer::updateFrameWorldBounds()
{
    const size_t objectCount = renderObjects_.size();
    frameWorldBounds_.resize(objectCount);
    framePrepParallelFor(objectCount, [this](size_t begin, size_t end) {
        for (size_t objectIndex = begin; objectIndex < end; ++objectIndex) {
            frameWorldBounds_[objectIndex] = renderObjects_[objectIndex].worldBounds();
        }
    });
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
    // uint8_t instead of vector<bool>: parallel chunks write disjoint indices,
    // which vector<bool>'s packed bits would turn into data races.
    std::vector<uint8_t> objectVisible(objectCount, 0);
    std::atomic<size_t> totalObjects{0};
    std::atomic<size_t> culledObjects{0};
    std::atomic<size_t> visibleObjects{0};
    framePrepParallelFor(objectCount, [&](size_t begin, size_t end) {
        size_t chunkTotal = 0;
        size_t chunkCulled = 0;
        size_t chunkVisible = 0;
        for (size_t objectIndex = begin; objectIndex < end; ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            if (!isRenderObjectActive(object)) {
                continue;
            }
            if (!object.mesh || !object.mesh->valid()) {
                continue;
            }

            ++chunkTotal;
            const renderer::Aabb& worldBounds = frameWorldBounds_[objectIndex];
            if (worldBounds.valid() && !frustum.testAabb(worldBounds)) {
                ++chunkCulled;
                continue;
            }

            ++chunkVisible;
            objectVisible[objectIndex] = 1;
        }
        totalObjects.fetch_add(chunkTotal, std::memory_order_relaxed);
        culledObjects.fetch_add(chunkCulled, std::memory_order_relaxed);
        visibleObjects.fetch_add(chunkVisible, std::memory_order_relaxed);
    });
    cullingStats_.totalObjects = totalObjects.load();
    cullingStats_.culledObjects = culledObjects.load();
    cullingStats_.visibleObjects = visibleObjects.load();

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex] != 0) {
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

    // Serial on purpose: this runs inside the per-cascade parallel loop in
    // buildShadowFrameData, and framePrepParallelFor must not nest.
    const size_t objectCount = std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
    std::vector<uint8_t> objectVisible(objectCount, 0);
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!isRenderObjectActive(object)) {
            continue;
        }
        if (!object.mesh || !object.mesh->valid()) {
            continue;
        }

        const renderer::Aabb& worldBounds = frameWorldBounds_[objectIndex];
        if (worldBounds.valid() && !lightFrustum.testAabb(worldBounds)) {
            continue;
        }

        objectVisible[objectIndex] = 1;
    }

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex] != 0) {
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
    if (!gpuCulling_.available()) {
        return;
    }

    // Two-phase mode drops the camera-still requirement: phase 1 projects with
    // the pyramid's stored (previous-frame) view-projection, and phase 2 corrects
    // any resulting false negatives against the mid-frame rebuild. Single-phase
    // keeps the conservative previous-frame validity gate.
    const bool occlusionEnabledThisFrame =
        isGpuOcclusionCullingActive() &&
        (frameTwoPhaseOcclusionActive_ || previousFrameDepthValidForOcclusion());
    const VkExtent2D extent = swapchain_.extent();

    GpuCullFrameParams frameParams{};
    frameParams.occlusionViewProjection = depthPyramid_.viewProjection();
    frameParams.occlusionViewProjectionPhase2 = frameViewProjection_;
    frameParams.cameraPosition = glm::vec4(frameCameraPosition_, 0.0f);
    frameParams.viewportAndMipCount = glm::vec4(static_cast<float>(extent.width),
                                                static_cast<float>(extent.height),
                                                static_cast<float>(depthPyramid_.mipLevels()),
                                                0.0f);
    frameParams.occlusionSettings = glm::vec4(gpuOcclusionDepthBias_,
                                              gpuOcclusionNearDisableDistance_,
                                              gpuOcclusionMaxScreenCoverage_,
                                              gpuOcclusionMinScreenPixels_);
    frameParams.counterAndFlags = glm::uvec4(kGpuCullStatsCounterOffset, occlusionEnabledThisFrame ? 1u : 0u, 0u, 0u);
    gpuCulling_.paramBuffer(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullFrameParams>(&frameParams, 1)));

    std::vector<GpuCullDrawItem> cullDrawItems(allDrawItems_.size());
    framePrepParallelFor(allDrawItems_.size(), [&](size_t begin, size_t end) {
        for (size_t drawIndex = begin; drawIndex < end; ++drawIndex) {
            const DrawItem& drawItem = allDrawItems_[drawIndex];
            GpuCullDrawItem& gpuDrawItem = cullDrawItems[drawIndex];

            renderer::Aabb worldBounds{};
            if (drawItem.objectIndex < frameWorldBounds_.size()) {
                worldBounds = frameWorldBounds_[drawItem.objectIndex];
            }

            if (worldBounds.valid()) {
                gpuDrawItem.boundsMin = glm::vec4(worldBounds.min, 0.0f);
                gpuDrawItem.boundsMax = glm::vec4(worldBounds.max, 0.0f);
            } else {
                gpuDrawItem.boundsMin =
                    glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
                gpuDrawItem.boundsMax =
                    glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
            }

            gpuDrawItem.indexCount = drawItem.indexCount;
            gpuDrawItem.firstIndex = drawItem.firstIndex;
            gpuDrawItem.vertexOffset = drawItem.vertexOffset;
            gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;
        }
    });

    for (size_t batchIndex = 0; batchIndex < meshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = meshDrawBatches_[batchIndex];
        const uint32_t endDrawItem =
            std::min<uint32_t>(batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
        for (uint32_t drawItemIndex = batch.beginDrawItem; drawItemIndex < endDrawItem; ++drawItemIndex) {
            cullDrawItems[drawItemIndex].batchIndex = static_cast<uint32_t>(batchIndex);
            cullDrawItems[drawItemIndex].batchOutputBase = batch.compactedCommandOffset;
        }
    }

    gpuCulling_.cullInputBuffer(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullDrawItem>(cullDrawItems.data(), cullDrawItems.size())));
}

void Renderer::updateGpuShadowCullInputBuffer(uint32_t frameIndex)
{
    if (allDrawItems_.empty()) {
        return;
    }
    if (!gpuCulling_.shadowAvailable()) {
        return;
    }

    GpuCullFrameParams frameParams{};
    frameParams.counterAndFlags = glm::uvec4(kGpuCullStatsCounterOffset, 0u, 0u, 0u);
    gpuCulling_.paramBuffer(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullFrameParams>(&frameParams, 1)));

    std::vector<GpuCullDrawItem> cullDrawItems(allDrawItems_.size());
    framePrepParallelFor(allDrawItems_.size(), [&](size_t begin, size_t end) {
        for (size_t drawIndex = begin; drawIndex < end; ++drawIndex) {
            const DrawItem& drawItem = allDrawItems_[drawIndex];
            GpuCullDrawItem& gpuDrawItem = cullDrawItems[drawIndex];

            renderer::Aabb worldBounds{};
            if (drawItem.objectIndex < frameWorldBounds_.size()) {
                worldBounds = frameWorldBounds_[drawItem.objectIndex];
            }

            if (worldBounds.valid()) {
                gpuDrawItem.boundsMin = glm::vec4(worldBounds.min, 0.0f);
                gpuDrawItem.boundsMax = glm::vec4(worldBounds.max, 0.0f);
            } else {
                gpuDrawItem.boundsMin =
                    glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
                gpuDrawItem.boundsMax =
                    glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
            }

            gpuDrawItem.indexCount = drawItem.indexCount;
            gpuDrawItem.firstIndex = drawItem.firstIndex;
            gpuDrawItem.vertexOffset = drawItem.vertexOffset;
            gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;
        }
    });

    for (size_t batchIndex = 0; batchIndex < gpuShadowMeshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = gpuShadowMeshDrawBatches_[batchIndex];
        const uint32_t endDrawItem =
            std::min<uint32_t>(batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
        for (uint32_t drawItemIndex = batch.beginDrawItem; drawItemIndex < endDrawItem; ++drawItemIndex) {
            cullDrawItems[drawItemIndex].batchIndex = static_cast<uint32_t>(batchIndex);
            cullDrawItems[drawItemIndex].batchOutputBase = batch.compactedCommandOffset;
        }
    }

    gpuCulling_.shadowCullInputBuffer(frameIndex)
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
            << "  exposure: " << postProcess_.currentToneMappingExposure() << "\n"
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
        const uint32_t totalDrawItems = gpuCulling_.available()
                                            ? gpuCulling_.mainTotalDrawItems(frameIndex)
                                            : static_cast<uint32_t>(cullingStats_.totalDrawItems);
        const uint32_t batchCount = gpuCulling_.available()
                                        ? gpuCulling_.mainBatchCount(frameIndex)
                                        : static_cast<uint32_t>(cullingStats_.batchCount);
        uint32_t visibleDrawItems = 0;
        renderer::GpuCullCounters counters{};
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
                    << "  depth pyramid mips: " << depthPyramid_.mipLevels() << "\n"
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
    snapshot.depthPyramidBuildAvailable = depthPyramid_.buildAvailable();
    snapshot.depthPyramidValid = depthPyramid_.valid();
    snapshot.previousFrameDepthValid = previousFrameDepthValidForOcclusion();
    snapshot.depthPyramidMipCount = depthPyramid_.mipLevels();
    snapshot.totalDrawItems = static_cast<uint32_t>(
        std::min<size_t>(cullingStats_.totalDrawItems, std::numeric_limits<uint32_t>::max()));
    if (cullingStats_.gpuCulling && gpuCulling_.available()) {
        snapshot.totalDrawItems = gpuCulling_.mainTotalDrawItems(frameIndex);
    }

    snapshot.visibleDrawItems =
        static_cast<uint32_t>(std::min<size_t>(visibleDrawItems_.size(), snapshot.totalDrawItems));
    snapshot.frustumCulledDrawItems = snapshot.totalDrawItems > snapshot.visibleDrawItems
                                          ? snapshot.totalDrawItems - snapshot.visibleDrawItems
                                          : 0;
    uint32_t gpuVisibleDrawItems = 0;
    renderer::GpuCullCounters gpuCounters{};
    if (readGpuCullCounters(frameIndex, gpuCounters)) {
        snapshot.totalDrawItems = std::min(gpuCounters.totalDrawItems, snapshot.totalDrawItems);
        snapshot.phase2RescuedDrawItems = gpuCounters.phase2RescuedDrawItems;
        // Phase 2 rescues count as visible; the effective occlusion-culled count
        // excludes them.
        snapshot.visibleDrawItems =
            std::min(gpuCounters.visibleDrawItems + gpuCounters.phase2RescuedDrawItems, snapshot.totalDrawItems);
        snapshot.frustumCulledDrawItems = std::min(gpuCounters.frustumCulledDrawItems, snapshot.totalDrawItems);
        snapshot.occlusionCulledDrawItems =
            std::min(gpuCounters.occlusionCulledDrawItems - gpuCounters.phase2RescuedDrawItems,
                     snapshot.totalDrawItems);
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
        snapshot.phase2RescuedDrawItems = 0;
    }
    snapshot.twoPhaseOcclusion = frameTwoPhaseOcclusionActive_;

    snapshot.shadowDrawItems = static_cast<uint32_t>(
        std::min<size_t>(shadowCullingStats_.totalDrawItems, std::numeric_limits<uint32_t>::max()));
    if (shadowCullingStats_.gpuCulling && gpuCulling_.shadowAvailable()) {
        snapshot.shadowDrawItems = gpuCulling_.shadowTotalDrawItems(frameIndex);
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
    exposureHistory_.push(postProcess_.currentToneMappingExposure());
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
    ssrSettings_ = settings.ssr;
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
        useTwoPhaseOcclusion_ = settings.enableTwoPhaseOcclusion;
        useAsyncCompute_ = settings.enableAsyncCompute;
        useBindlessMaterialTextures_ = settings.enableBindlessMaterialTextures;
    } else {
        if (!settings.useGpuCulling || gpuCulling_.available()) {
            useGpuCulling_ = settings.useGpuCulling;
        }
        if (!settings.useGpuShadowCulling || gpuCulling_.shadowAvailable()) {
            useGpuShadowCulling_ = settings.useGpuShadowCulling;
        }
        useGpuOcclusionCulling_ = settings.enableGpuOcclusionCulling;
        useTwoPhaseOcclusion_ = settings.enableTwoPhaseOcclusion;
        useAsyncCompute_ = settings.enableAsyncCompute;
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
    postProcess_.resetAutoExposureTimer();
}

RuntimeSettings Renderer::captureRuntimeSettings() const
{
    RuntimeSettings settings{};
    settings.toneMapping = toneMappingSettings_;
    settings.bloom = bloomSettings_;
    settings.taa = taaSettings_;
    settings.ssr = ssrSettings_;
    settings.csm = csmSettings_;
    settings.debugUi = debugUiSettings_;
    settings.useGpuCulling = useGpuCulling_;
    settings.useGpuShadowCulling = useGpuShadowCulling_;
    settings.enableGpuOcclusionCulling = useGpuOcclusionCulling_;
    settings.enableTwoPhaseOcclusion = useTwoPhaseOcclusion_;
    settings.enableAsyncCompute = useAsyncCompute_;
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
    if (!renderer::SceneBuilder::hasOcclusionTest(renderObjects_)) {
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
                                std::to_string(renderer::kOcclusionTestObjectCount) +
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
    return depthPyramid_.valid() &&
           maxMatrixDifference(frameViewProjection_, depthPyramid_.viewProjection()) <= 0.0005f &&
           glm::distance(frameCameraPosition_, depthPyramid_.cameraPosition()) <= 0.01f;
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
        toneMappingSettings_, bloomSettings_, taaSettings_, ssrSettings_, csmSettings_, debugUiSettings_);

    // GPU occlusion tuning is renderer state, not part of the settings structs,
    // so it stays here.
    gpuOcclusionDepthBias_ = std::clamp(gpuOcclusionDepthBias_, 0.0f, 0.05f);
    gpuOcclusionNearDisableDistance_ = std::clamp(gpuOcclusionNearDisableDistance_, 0.0f, 10.0f);
    gpuOcclusionMaxScreenCoverage_ = std::clamp(gpuOcclusionMaxScreenCoverage_, 0.01f, 1.0f);
    gpuOcclusionMinScreenPixels_ = std::clamp(gpuOcclusionMinScreenPixels_, 1.0f, 64.0f);
    if (!isGpuCullingActive() || !depthPyramid_.buildAvailable()) {
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
    // PostProcessStack owns the TAA jitter sequence/state; it returns the NDC
    // offset to fold into the main projection (zero when jitter is inactive).
    glm::mat4 jitteredProjection = projection;
    const glm::vec2 jitterNdc = postProcess_.advanceJitter(extent);
    jitteredProjection[2][0] += jitterNdc.x;
    jitteredProjection[2][1] += jitterNdc.y;
    frameJitteredProjection_ = jitteredProjection;
    frameJitteredViewProjection_ = jitteredProjection * view;
    frameViewProjection_ = viewProjection;
    frameCameraPosition_ = camera_.position;
    // First frame after a history reset: reproject against the current matrices
    // so the velocity buffer reads as zero motion instead of garbage.
    if (!previousFrameViewProjectionValid_) {
        previousFrameViewProjection_ = viewProjection;
        previousFrameViewProjectionValid_ = true;
    }

    // Regenerate the animated demo light swarm, then hand the froxel grid + light
    // culling the current view/inverse-projection and camera planes (view-space
    // work, so it runs regardless of scene contents).
    updateDemoLights(elapsedSeconds);
    // Advance the skinned animation by a speed-scaled frame delta so play/pause
    // holds the current pose and the speed slider changes playback continuously.
    const float skinnedDelta = elapsedSeconds - previousElapsedSeconds_;
    previousElapsedSeconds_ = elapsedSeconds;
    if (animateSkinnedMesh_) {
        skinnedAnimationTime_ += skinnedDelta * skinnedAnimationSpeed_;
    }
    if (skinnedMesh_.valid()) {
        skinnedMesh_.update(frameIndex, skinnedAnimationTime_);
    }
    clusteredLighting_.updateParams(frameIndex,
                                    view,
                                    glm::inverse(projection),
                                    camera_.nearPlane,
                                    camera_.farPlane,
                                    static_cast<float>(extent.width),
                                    static_cast<float>(extent.height));

    if (renderObjects_.empty()) {
        resetFrameStateForEmptyScene(frameIndex);
        return;
    }

    updateCascades(aspect);

    if (updateAnimatedTransforms(elapsedSeconds)) {
        invalidateDepthPyramid();
    }

    // Transforms are final for this frame; cache every object's world AABB once
    // for the visibility, shadow-cascade, and GPU-cull-input passes below.
    updateFrameWorldBounds();

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
    clusteredLighting_.upload(frameIndex);

    // Resolved here (not in recordRenderCommands) so drawFrame can submit the
    // async compute work before the graphics command buffer is even recorded.
    const bool clusteredLightingActiveThisFrame = clusteredLighting_.available() && useClusteredLighting_ &&
                                                  clusteredLighting_.lightCount() > 0 && !allDrawItems_.empty();
    frameAsyncComputeActive_ = clusteredLightingActiveThisFrame && useAsyncCompute_ && asyncCompute_.available();

    frameSsrActive_ = ssrSettings_.enabled && ssr_.available() && !allDrawItems_.empty();
    if (frameSsrActive_) {
        // The trace reconstructs positions from the jitter-rendered depth, so it
        // uses the same jittered projection the rasterizer used.
        ssr_.uploadParams(frameIndex, view, frameJitteredProjection_, ssrFrameCounter_++);
    }

    frameGtaoActive_ = ssaoSettings_.enabled && gtao_.available() && !allDrawItems_.empty();
    if (frameGtaoActive_) {
        // The horizon search reconstructs positions from the same jitter-rendered
        // depth the composite consumes, so it uses the jittered projection too.
        gtao_.uploadParams(frameIndex, view, frameJitteredProjection_, gtaoFrameCounter_++);
    }
}

// Called from drawFrame after command recording: both the object-data upload and
// the skybox push constants must still see the previous frame's matrices, so the
// capture happens only once the frame has been fully recorded.
void Renderer::capturePreviousFrameMatrices()
{
    previousFrameViewProjection_ = frameViewProjection_;
    previousFrameViewProjectionValid_ = true;
    for (renderer::RenderObject& object : renderObjects_) {
        object.previousModelMatrix = object.transform.modelMatrix();
        object.previousModelValid = true;
    }
    if (skinnedMesh_.valid()) {
        previousSkinnedModelMatrix_ = skinnedMesh_.modelMatrix();
        previousSkinnedModelValid_ = true;
    }
}

void Renderer::resetFrameStateForEmptyScene(uint32_t frameIndex)
{
    frameTwoPhaseOcclusionActive_ = false;
    frameAsyncComputeActive_ = false;
    frameSsrActive_ = false;
    frameGtaoActive_ = false;
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
    gpuCulling_.resetFrameCounters(frameIndex, 0);
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
    gpuCulling_.resetFrameCounters(
        frameIndex, static_cast<uint32_t>(std::min(allDrawItems_.size(), static_cast<size_t>(kMaxDrawItems))));
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

    // Cascades are independent (each writes only its own draw-item/batch slot),
    // so they run as parallel jobs; the stats reduction happens after the join.
    if (parallelFramePrepEnabled_) {
        jobSystem_.parallelFor(cascadeCount, 1, [this](size_t begin, size_t end) {
            for (size_t cascadeIndex = begin; cascadeIndex < end; ++cascadeIndex) {
                buildShadowDrawItems(static_cast<uint32_t>(cascadeIndex),
                                     frameCascades_[cascadeIndex].lightFrustum);
                buildMeshDrawBatchesForItems(shadowCascadeDrawItems_[cascadeIndex],
                                             shadowCascadeMeshDrawBatches_[cascadeIndex]);
            }
        });
    } else {
        for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            buildShadowDrawItems(cascadeIndex, frameCascades_[cascadeIndex].lightFrustum);
            buildMeshDrawBatchesForItems(shadowCascadeDrawItems_[cascadeIndex],
                                         shadowCascadeMeshDrawBatches_[cascadeIndex]);
        }
    }
    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
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

        gpuCulling_.setShadowCullFrameInfo(
            frameIndex,
            static_cast<uint32_t>(std::min(allDrawItems_.size() * cascadeCount, static_cast<size_t>(kMaxDrawItems))),
            static_cast<uint32_t>(gpuShadowMeshDrawBatches_.size() * cascadeCount),
            shadowIndirectCountPathActive);

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

        gpuCulling_.setMainCullFrameInfo(
            frameIndex, static_cast<uint32_t>(meshDrawBatches_.size()), indirectCountPathActive);
        // Two-phase occlusion needs the bindless multi-draw-indirect path (the
        // phase-2 pass replays the batch draws) and a valid previous-frame
        // pyramid. With the indirect-count path phase 2 re-compacts into the
        // batch regions; without it, phase 2 rewrites the fixed per-item slots.
        frameTwoPhaseOcclusionActive_ =
            useTwoPhaseOcclusion_ && isMainPassMultiDrawIndirectActive() && isGpuOcclusionCullingActive();
        updateGpuCullInputBuffer(frameIndex);
    } else {
        frameTwoPhaseOcclusionActive_ = false;
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

    // Per-item fill is the heaviest CPU loop of the frame (six mat4 multiplies
    // per draw item); every iteration writes only objectFrameData[drawIndex] and
    // reads shared frame state, so it chunks cleanly across the JobSystem.
    framePrepParallelFor(objectFrameCount, [&](size_t begin, size_t end) {
        for (size_t drawIndex = begin; drawIndex < end; ++drawIndex) {
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
            frameData.currMvpNoJitter = frameViewProjection_ * model;
            frameData.prevMvpNoJitter =
                previousFrameViewProjection_ * (object.previousModelValid ? object.previousModelMatrix : model);
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
                frameData.materialParams = {
                    material->metallic, material->roughness, material->multiScatterStrength, 0.0f};
                frameData.textureIndices = {material->baseColorTextureIndex,
                                            material->normalTextureIndex,
                                            material->metallicRoughnessTextureIndex,
                                            material->emissiveTextureIndex};
                frameData.emissiveFactor =
                    glm::vec4(material->emissiveFactor, material->hasEmissiveTexture ? 1.0f : 0.0f);
            }
            frameData.cameraPosition =
                glm::vec4(camera_.position, csmSettings_.enableCascadeDebugColors ? 1.0f : 0.0f);
            frameData.cameraForward =
                glm::vec4(glm::normalize(camera_.target - camera_.position), static_cast<float>(cascadeCount));
        }
    });

    frameObjectDataBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const ObjectFrameData>(objectFrameData.data(), objectFrameData.size())));

    // The skinned demo mesh isn't a RenderObject; give it its own ObjectFrameData
    // in the reserved last slot (same lighting/camera state as the scene draws).
    if (skinnedMesh_.valid()) {
        const glm::mat4 model = skinnedMesh_.modelMatrix();
        ObjectFrameData skinnedData{};
        skinnedData.mvp = frameJitteredViewProjection_ * model;
        skinnedData.model = model;
        skinnedData.currMvpNoJitter = frameViewProjection_ * model;
        skinnedData.prevMvpNoJitter =
            previousFrameViewProjection_ * (previousSkinnedModelValid_ ? previousSkinnedModelMatrix_ : model);
        for (uint32_t cascade = 0; cascade < kMaxShadowCascades; ++cascade) {
            skinnedData.lightMvp[cascade] = frameCascades_[cascade].lightViewProjection * model;
        }
        skinnedData.lightDirection = activeLightDirection;
        skinnedData.lightColor = activeLightColor;
        skinnedData.ambientColor = activeAmbientLightColor;
        skinnedData.cascadeSplits = frameCascadeSplits_;
        skinnedData.shadowSettings = {csmSettings_.depthBiasConstant,
                                      csmSettings_.depthBiasSlope,
                                      shadowSettings_.enablePcf ? 1.0f : 0.0f,
                                      static_cast<float>(std::max(shadowSettings_.pcfRadius, 0))};
        skinnedData.baseColorFactor = glm::vec4(0.85f, 0.45f, 0.32f, 1.0f);
        skinnedData.materialParams = {0.1f, 0.55f, 1.0f, 0.0f};
        skinnedData.cameraPosition = glm::vec4(camera_.position, csmSettings_.enableCascadeDebugColors ? 1.0f : 0.0f);
        skinnedData.cameraForward =
            glm::vec4(glm::normalize(camera_.target - camera_.position), static_cast<float>(cascadeCount));
        skinnedData.textureIndices = {bindlessBaseColorFallbackIndex_,
                                      bindlessNormalFallbackIndex_,
                                      bindlessMetallicRoughnessFallbackIndex_,
                                      0};
        frameObjectDataBuffers_.at(frameIndex)
            .upload(std::as_bytes(std::span<const ObjectFrameData>(&skinnedData, 1)),
                    static_cast<VkDeviceSize>(kSkinnedObjectFrameSlot) * sizeof(ObjectFrameData));
    }
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
    recreatePostProcessResources();

    const bool exposurePipelineMissing =
        toneMappingSettings_.enableAutoExposure &&
        postProcess_.luminanceDescriptorSetLayoutHandle() != VK_NULL_HANDLE &&
        ((postProcess_.autoExposureAvailable() && postProcess_.luminancePipeline().pipeline() == VK_NULL_HANDLE) ||
         (postProcess_.histogramExposureAvailable() && postProcess_.histogramPipeline().pipeline() == VK_NULL_HANDLE) ||
         (postProcess_.exposureReduceAvailable() &&
          postProcess_.exposureReducePipeline().pipeline() == VK_NULL_HANDLE));
    const bool pipelineNeedsRecreate =
        pipeline_.pipeline() == VK_NULL_HANDLE || pipelineColorFormat_ != kSceneColorFormat ||
        pipelineDepthFormat_ != swapchain_.depthFormat() || skyboxPipeline_.pipeline() == VK_NULL_HANDLE ||
        skyboxPipelineColorFormat_ != kSceneColorFormat || skyboxPipelineDepthFormat_ != swapchain_.depthFormat() ||
        shadowPipelineDepthFormat_ != shadowMap_.format() ||
        postProcess_.bloomExtractPipeline().pipeline() == VK_NULL_HANDLE ||
        postProcess_.bloomExtractPipelineColorFormat() != kBloomColorFormat ||
        postProcess_.bloomBlurPipeline().pipeline() == VK_NULL_HANDLE ||
        postProcess_.bloomBlurPipelineColorFormat() != kBloomColorFormat ||
        postProcess_.bloomDownsamplePipeline().pipeline() == VK_NULL_HANDLE ||
        postProcess_.bloomDownsamplePipelineColorFormat() != kBloomColorFormat ||
        postProcess_.bloomUpsamplePipeline().pipeline() == VK_NULL_HANDLE ||
        postProcess_.bloomUpsamplePipelineColorFormat() != kBloomColorFormat ||
        postProcess_.taaResolvePipeline().pipeline() == VK_NULL_HANDLE ||
        postProcess_.taaResolvePipelineColorFormat() != kSceneColorFormat ||
        postProcess_.compositePipeline().pipeline() == VK_NULL_HANDLE ||
        postProcess_.compositePipelineColorFormat() != swapchain_.colorFormat() || exposurePipelineMissing;
    if (pipelineNeedsRecreate) {
        createPipeline();
    }

    imagesInFlight_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
}

renderer::RenderGraphFrameResources Renderer::renderGraphFrameResources()
{
    const VkExtent3D sceneExtent = postProcess_.sceneColor().extent();
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

    const auto bloomResource = [&bloomClear](const char* name, const rhi::VulkanImage& image, VkImageLayout& layout) {
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

    const auto taaHistoryResource = [&bloomClear](
                                        const char* name, const rhi::VulkanImage& image, VkImageLayout& layout) {
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
    bloomDownsampleResources.reserve(postProcess_.bloomMipDownsampleImages().size());
    for (size_t level = 0; level < postProcess_.bloomMipDownsampleImages().size() &&
                           level < postProcess_.bloomMipDownsampleLayouts().size();
         ++level) {
        const std::string name = "BloomMipDownsample" + std::to_string(level);
        bloomDownsampleResources.push_back(bloomResource(name.c_str(),
                                                         postProcess_.bloomMipDownsampleImages()[level],
                                                         postProcess_.bloomMipDownsampleLayouts()[level]));
    }

    std::vector<renderer::RenderGraphImageResource> bloomUpsampleResources;
    bloomUpsampleResources.reserve(postProcess_.bloomMipUpsampleImages().size());
    for (size_t level = 0;
         level < postProcess_.bloomMipUpsampleImages().size() && level < postProcess_.bloomMipUpsampleLayouts().size();
         ++level) {
        const std::string name = "BloomMipUpsample" + std::to_string(level);
        bloomUpsampleResources.push_back(bloomResource(
            name.c_str(), postProcess_.bloomMipUpsampleImages()[level], postProcess_.bloomMipUpsampleLayouts()[level]));
    }

    renderer::RenderGraphImageResource gtaoRawResource{};
    if (gtao_.available()) {
        // Raw AO is half resolution; import its actual extent, not the scene's.
        const VkExtent3D gtaoRawExtent = gtao_.rawAmbientOcclusion().extent();
        gtaoRawResource = renderer::RenderGraphImageResource{
            "GtaoRawAmbientOcclusion",
            gtao_.rawAmbientOcclusion().image(),
            gtao_.rawAmbientOcclusion().imageView(),
            VkExtent2D{gtaoRawExtent.width, gtaoRawExtent.height},
            gtao_.rawAmbientOcclusionLayoutPtr(),
            gtao_.rawAmbientOcclusion().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            false,
            false,
        };
    }

    renderer::RenderGraphImageResource ssrSceneColorCopyResource{};
    if (ssr_.available()) {
        ssrSceneColorCopyResource = renderer::RenderGraphImageResource{
            "SsrSceneColorCopy",
            ssr_.sceneColorCopy().image(),
            ssr_.sceneColorCopy().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            ssr_.sceneColorCopyLayoutPtr(),
            ssr_.sceneColorCopy().format(),
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            false,
            false,
        };
    }

    return renderer::RenderGraphFrameResources{
        renderer::RenderGraphImageResource{
            "SceneColor",
            postProcess_.sceneColor().image(),
            postProcess_.sceneColor().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.sceneColorLayout(),
            postProcess_.sceneColor().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            sceneClear,
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "VelocityBuffer",
            postProcess_.velocity().image(),
            postProcess_.velocity().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.velocityLayout(),
            postProcess_.velocity().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "NormalRoughnessGBuffer",
            postProcess_.normalRoughness().image(),
            postProcess_.normalRoughness().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.normalRoughnessLayout(),
            postProcess_.normalRoughness().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "AmbientOcclusion",
            postProcess_.ambientOcclusion().image(),
            postProcess_.ambientOcclusion().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.ambientOcclusionLayout(),
            postProcess_.ambientOcclusion().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            true,
            false,
        },
        gtaoRawResource,
        ssrSceneColorCopyResource,
        taaHistoryResource("TAAHistoryRead",
                           postProcess_.taaHistoryImages()[postProcess_.taaHistoryReadIndex()],
                           postProcess_.taaHistoryLayouts()[postProcess_.taaHistoryReadIndex()]),
        taaHistoryResource("TAAHistoryWrite",
                           postProcess_.taaHistoryImages()[postProcess_.taaHistoryWriteIndex()],
                           postProcess_.taaHistoryLayouts()[postProcess_.taaHistoryWriteIndex()]),
        renderer::RenderGraphImageResource{
            "BloomExtract",
            postProcess_.bloomExtract().image(),
            postProcess_.bloomExtract().imageView(),
            postProcess_.bloomExtent(),
            &postProcess_.bloomExtractLayout(),
            postProcess_.bloomExtract().format(),
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
            postProcess_.bloomPing().image(),
            postProcess_.bloomPing().imageView(),
            postProcess_.bloomExtent(),
            &postProcess_.bloomPingLayout(),
            postProcess_.bloomPing().format(),
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
            postProcess_.bloomPong().image(),
            postProcess_.bloomPong().imageView(),
            postProcess_.bloomExtent(),
            &postProcess_.bloomPongLayout(),
            postProcess_.bloomPong().format(),
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
            depthPyramid_.layoutPtr(),
            depthPyramid_.format(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            depthPyramid_.mipLevels(),
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            {},
            false,
            true,
        },
        bufferResource(
            "MainCullInput", gpuCulling_.cullInputBuffers(), currentFrame_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        bufferResource("MainCullIndirectOutput",
                       frameIndirectDrawBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT),
        bufferResource("MainCullVisibleCounts",
                       gpuCulling_.visibleCountBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        bufferResource("MainCullReadback",
                       gpuCulling_.visibleCountReadbackBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource(
            "LuminancePartials", postProcess_.luminanceBuffers(), currentFrame_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        bufferResource("LuminanceReadback",
                       postProcess_.luminanceReadbackBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("LuminanceHistogram",
                       postProcess_.histogramBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("HistogramReadback",
                       postProcess_.histogramReadbackBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource(
            "ExposureState", postProcess_.exposureBuffers(), currentFrame_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        postProcess_.isTaaActive(),
        frameTwoPhaseOcclusionActive_,
        frameSsrActive_,
        frameGtaoActive_,
    };
}

bool Renderer::readGpuVisibleCount(uint32_t frameIndex, uint32_t& visibleCount)
{
    return gpuCulling_.readMainVisibleCount(isGpuCullingActive(), frameIndex, visibleCount);
}

bool Renderer::readGpuCullCounters(uint32_t frameIndex, renderer::GpuCullCounters& counters)
{
    return gpuCulling_.readMainCounters(isGpuCullingActive(), frameIndex, counters);
}

bool Renderer::readGpuShadowVisibleCount(uint32_t frameIndex, uint32_t& visibleCount)
{
    return gpuCulling_.readShadowVisibleCount(isGpuShadowCullingActive(), frameIndex, visibleCount);
}

bool Renderer::isGpuCullingActive() const
{
    return useGpuCulling_ && gpuCulling_.mainResourcesReady(static_cast<uint32_t>(frames_.size()));
}

bool Renderer::isGpuOcclusionCullingActive() const
{
    return useGpuOcclusionCulling_ && isGpuCullingActive() && depthPyramid_.buildAvailable() && depthPyramid_.valid() &&
           depthPyramid_.image() != VK_NULL_HANDLE && depthPyramid_.sampler() != VK_NULL_HANDLE &&
           depthPyramid_.mipLevels() > 0;
}

bool Renderer::isGpuShadowCullingActive() const
{
    return useGpuShadowCulling_ && isShadowIndirectActive() &&
           gpuCulling_.shadowResourcesReady(static_cast<uint32_t>(frames_.size()));
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
    return gpuCulling_.frameIndirectCountPathActive(frameIndex);
}

bool Renderer::isShadowIndirectCountSupported() const
{
    return isGpuShadowCullingActive() && context_.device().drawIndexedIndirectCountAvailable();
}

bool Renderer::isShadowIndirectCountPathActive(uint32_t frameIndex) const
{
    return gpuCulling_.frameShadowIndirectCountPathActive(frameIndex);
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
    gpuCulling_.recordMainCull(commandBuffer,
                               currentFrame_,
                               isGpuCullingActive(),
                               static_cast<uint32_t>(allDrawItems_.size()),
                               frameFrustumPlanes_,
                               isMainPassMultiDrawIndirectActive(),
                               /*copyReadback=*/!frameTwoPhaseOcclusionActive_);
}

void Renderer::recordGpuShadowCullingCommands(VkCommandBuffer commandBuffer, uint32_t cascadeIndex)
{
    gpuCulling_.recordShadowCull(commandBuffer,
                                 currentFrame_,
                                 isGpuShadowCullingActive(),
                                 cascadeIndex,
                                 activeCascadeCount(),
                                 static_cast<uint32_t>(allDrawItems_.size()),
                                 frameShadowCascadeFrustumPlanes_[cascadeIndex]);
}

void Renderer::ensureDepthPyramidShaderReadLayout(VkCommandBuffer commandBuffer)
{
    depthPyramid_.ensureShaderReadLayout(commandBuffer);
}

void Renderer::recordDepthPyramidCommands(VkCommandBuffer commandBuffer, bool midFrame)
{
    depthPyramid_.recordCommands(commandBuffer, currentFrame_, frameViewProjection_, frameCameraPosition_, midFrame);
}

void Renderer::recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t mainDrawItemCount = visibleDrawItems_.size();
    const bool clusteredLightingActive = clusteredLighting_.available() && useClusteredLighting_ &&
                                         clusteredLighting_.lightCount() > 0 && !allDrawItems_.empty();
    const bool taaActiveThisFrame = postProcess_.isTaaActive();
    postProcess_.beginFrame(currentFrame_, taaActiveThisFrame);

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
                shadowIndirectCountPathActive ? gpuCulling_.shadowVisibleCountBuffer(currentFrame_).buffer()
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

    // Clustered (Forward+) light assignment: rebuild the froxel AABBs, then cull
    // every light into its froxels. Both write buffers the main HDR fragment
    // shader reads, so the assignment pass barriers into the fragment stage.
    if (clusteredLightingActive && !frameAsyncComputeActive_) {
        {
            const renderer::GpuProfileScope buildScope(gpuProfiler_, currentFrame_, commandBuffer, "ClusterBuild");
            rhi::debug::beginLabel(commandBuffer, "ClusterBuild");
            clusteredLighting_.recordClusterBuild(commandBuffer, currentFrame_);
            rhi::debug::endLabel(commandBuffer);
        }
        {
            const renderer::GpuProfileScope cullScope(gpuProfiler_, currentFrame_, commandBuffer, "LightCull");
            rhi::debug::beginLabel(commandBuffer, "LightCull");
            clusteredLighting_.recordLightCull(commandBuffer, currentFrame_);
            rhi::debug::endLabel(commandBuffer);
        }
    }

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
        // The fragment shader projects the sky direction with w = 0, which drops
        // the translation column, so the full previous view-projection doubles as
        // the rotation-only sky reprojection matrix.
        const SkyboxPushConstants skyboxPushConstants{glm::inverse(projection * skyboxView),
                                                      previousFrameViewProjection_};

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
        indirectCountPathActive ? gpuCulling_.visibleCountBuffer(currentFrame_).buffer() : VK_NULL_HANDLE;
    const uint32_t toneMappingOperator = toneMappingOperatorValue(toneMappingSettings_.operatorType);
    const float exposure = postProcess_.currentToneMappingExposure();
    // Shared lighting push constant for the main pass. The clustered fields are
    // zeroed when the per-froxel path is inactive, which makes the fragment
    // shader fall back to brute-force light evaluation.
    PushConstants basePushConstants{};
    basePushConstants.objectFrameDataAddress = objectFrameDataBaseAddress;
    basePushConstants.cascadeIndex = 0;
    basePushConstants.toneMappingOperator = toneMappingOperator;
    basePushConstants.exposure = exposure;
    basePushConstants.lightCount = clusteredLighting_.lightCount();
    basePushConstants.lightBufferAddress = clusteredLighting_.lightBufferAddress(currentFrame_);
    basePushConstants.clusterGridAddress =
        clusteredLightingActive ? clusteredLighting_.clusterGridAddress(currentFrame_) : 0;
    basePushConstants.lightIndexListAddress =
        clusteredLightingActive ? clusteredLighting_.lightIndexListAddress(currentFrame_) : 0;
    basePushConstants.clusterZNear = camera_.nearPlane;
    basePushConstants.clusterZFar = camera_.farPlane;
    basePushConstants.screenWidth = static_cast<float>(extent.width);
    basePushConstants.screenHeight = static_cast<float>(extent.height);
    basePushConstants.useClustered = clusteredLightingActive ? 1u : 0u;
    basePushConstants.debugClusterHeatmap = showClusterHeatmap_ ? 1u : 0u;
    if (multiDrawIndirectActive) {
        if (bindlessDescriptorSetsBound) {
            const PushConstants pushConstants = basePushConstants;
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

            PushConstants pushConstants = basePushConstants;
            pushConstants.objectFrameDataAddress =
                objectFrameDataBaseAddress +
                static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData));

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

    // Skinned demo mesh: GPU vertex skinning via the joint-matrix palette. Shares
    // the main descriptor sets + push constant layout, adds the joint-palette
    // address, and binds a second vertex stream for joint indices/weights.
    if (showSkinnedMesh_ && skinnedMesh_.valid() && skinnedPipeline_.pipeline() != VK_NULL_HANDLE &&
        bindlessMaterialTexturesActive && globalDescriptorSet != VK_NULL_HANDLE) {
        const renderer::GpuProfileScope skinnedScope(gpuProfiler_, currentFrame_, commandBuffer, "SkinnedMesh");
        rhi::debug::beginLabel(commandBuffer, "SkinnedMesh");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skinnedPipeline_.pipeline());

        const std::array<VkDescriptorSet, 2> skinnedSets{globalDescriptorSet, bindlessTextureHeap_.descriptorSet()};
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skinnedPipeline_.layout(),
                                0,
                                static_cast<uint32_t>(skinnedSets.size()),
                                skinnedSets.data(),
                                0,
                                nullptr);

        PushConstants skinnedPush = basePushConstants;
        skinnedPush.objectFrameDataAddress = objectFrameDataBaseAddress +
                                             static_cast<VkDeviceAddress>(kSkinnedObjectFrameSlot) *
                                                 sizeof(ObjectFrameData);
        skinnedPush.jointMatricesAddress = skinnedMesh_.jointPaletteAddress(currentFrame_);
        vkCmdPushConstants(commandBuffer,
                           skinnedPipeline_.layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(PushConstants)),
                           &skinnedPush);

        const std::array<VkBuffer, 2> skinnedVertexBuffers{skinnedMesh_.geometryBuffer(), skinnedMesh_.skinningBuffer()};
        const std::array<VkDeviceSize, 2> skinnedOffsets{0, 0};
        vkCmdBindVertexBuffers(commandBuffer,
                               0,
                               static_cast<uint32_t>(skinnedVertexBuffers.size()),
                               skinnedVertexBuffers.data(),
                               skinnedOffsets.data());
        vkCmdBindIndexBuffer(commandBuffer, skinnedMesh_.indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, skinnedMesh_.indexCount(), 1, 0, 0, 0);
        rhi::debug::endLabel(commandBuffer);
    }

    renderGraph_.endMainHdrPass();
    rhi::debug::endLabel(commandBuffer);
    if (mainHdrProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    // Two-phase Hi-Z occlusion: rebuild the pyramid from phase-1 depth, re-test
    // the phase-1 occlusion candidates against it, then draw the rescued
    // (disoccluded) objects into the existing attachments with LOAD ops. The
    // final end-of-frame pyramid rebuild below then includes the rescued draws.
    if (frameTwoPhaseOcclusionActive_) {
        recordDepthPyramidCommands(commandBuffer, /*midFrame=*/true);
        gpuCulling_.recordMainCullPhase2(commandBuffer,
                                         currentFrame_,
                                         isGpuCullingActive(),
                                         static_cast<uint32_t>(allDrawItems_.size()),
                                         frameFrustumPlanes_,
                                         isMainPassMultiDrawIndirectActive());

        const bool phase2ProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "MainHDRPhase2");
        rhi::debug::beginLabel(commandBuffer, "MainHDRPhase2");
        renderGraph_.beginMainHdrPhase2Pass();

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());
        if (bindlessDescriptorSetsBound) {
            const std::array<VkDescriptorSet, 2> phase2Sets{
                globalDescriptorSet,
                bindlessTextureHeap_.descriptorSet(),
            };
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_.layout(),
                                    0,
                                    static_cast<uint32_t>(phase2Sets.size()),
                                    phase2Sets.data(),
                                    0,
                                    nullptr);
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &basePushConstants);

            const renderer::Mesh* phase2BoundMesh = nullptr;
            for (const MeshDrawBatch& batch : meshDrawBatches_) {
                if (!batch.mesh || batch.drawItemCount == 0) {
                    continue;
                }

                if (phase2BoundMesh != batch.mesh) {
                    const VkBuffer vertexBuffers[] = {batch.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, batch.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    phase2BoundMesh = batch.mesh;
                }

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
            }
        }

        renderGraph_.endMainHdrPhase2Pass();
        rhi::debug::endLabel(commandBuffer);
        if (phase2ProfileScope) {
            gpuProfiler_.endScope(currentFrame_, commandBuffer);
        }
    }

    if (frameSsrActive_) {
        ssr_.recordCommands(commandBuffer, currentFrame_, postProcess_.sceneColor().image(), extent);
    }

    if (frameGtaoActive_) {
        gtao_.recordCommands(commandBuffer, currentFrame_, extent);
    }

    recordDepthPyramidCommands(commandBuffer);

    if (taaActiveThisFrame) {
        postProcess_.recordTaaResolveCommands(commandBuffer);
    }

    postProcess_.recordLegacyBloomCommands(commandBuffer);
    postProcess_.recordMipChainBloomCommands(commandBuffer);

    postProcess_.recordLuminanceCommands(commandBuffer);
    postProcess_.recordHistogramCommands(commandBuffer);

    postProcess_.recordCompositeCommands(commandBuffer, frameJitteredProjection_);

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
        postProcess_.advanceTaaHistory();
    }

    gpuProfiler_.endFrame(currentFrame_, commandBuffer);
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endFrame();
}

} // namespace ve
