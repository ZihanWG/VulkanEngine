// Renderer core: construction and teardown, the frame loop entry point, input
// and viewport interaction, swapchain recreation, runtime settings, and the
// small state accessors the rest of the class is built on.
//
// The class is one type spread across several translation units, which is the
// same arrangement RendererDebugUi.cpp already used. Only definitions were
// split; the class contract in Renderer.h is unchanged, and shared file-local
// helpers live in RendererInternal.h:
//
//   Renderer.cpp           lifecycle, frame loop, input, settings, accessors
//   RendererResources.cpp  pipelines, descriptors, buffers, shadow map, IBL
//   RendererScene.cpp      scene building, materials, asset + scene JSON
//   RendererFrame.cpp      per-frame CPU prep: lights, cascades, draw items
//   RendererRecord.cpp     command recording and the render-graph resources
//   RendererDebugUi.cpp    ImGui panels
//
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
    frameOcclusionTested_.assign(frames_.size(), 0u);
    screenshotCapture_.initialize(
        context_, static_cast<uint32_t>(frames_.size()), portfolioScreenshotDirectory());
    gpuProfiler_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    swapchain_.initialize(context_, window_.framebufferExtent());
    // Before anything screen-sized is created below: every one of those targets
    // reads its size from renderResolution_.
    updateRenderResolution();
    imguiLayer_.initialize(
        window_, context_, swapchain_.colorFormat(), swapchain_.imageCount(), defaultDebugUiLayoutPath());
    createMaterialDescriptorSetLayout();
    createBindlessMaterialTextureHeap();
    createSkyboxDescriptorSetLayout();
    postProcess_.createPostProcessDescriptorSetLayouts();
    ssr_.createDescriptorSetLayout();
    gtao_.createDescriptorSetLayout();
    createDepthPyramidDescriptorSetLayout();
    postProcess_.createPostProcessSampler();
    createShadowMap();
    // Pipelines first: SSR/GTAO resource creation binds descriptor sets against
    // pipelines that must already exist, and bails out ("pipeline resources are
    // missing") otherwise. Pipeline creation only needs the descriptor set
    // layouts and swapchain formats built above, so this direction is the one
    // that satisfies both. Resize takes the reverse order because the pipeline
    // recreate decision there reads availability flags that resource creation
    // sets, and by then the pipelines already exist.
    createPipeline();
    recreatePostProcessResources();
    commandContext_.initialize(context_, frames_);
    asyncCompute_.initialize(context_, static_cast<uint32_t>(frames_.size()));
    // Before createScene, which builds the material descriptor sets: those sets
    // bind the probe atlases and the probe parameter buffer, so the resources
    // have to exist by then or every material would record null handles.
    //
    // Created once and kept across swapchain recreation -- the atlases are sized
    // by the probe grid, not the window, and their contents persist across
    // frames by design.
    irradianceProbes_.create(context_,
                             shaderPath("probe_debug_fill.comp.spv"),
                             shaderPath("probe_border.comp.spv"),
                             shaderPath("probe_convolve.comp.spv"));
    irradianceProbes_.setBounds(giGridBounds());
    createScene();
    createObjectFrameDataBuffers();
    clusteredLighting_.create(context_,
                              static_cast<uint32_t>(frames_.size()),
                              shaderPath("cluster_build.comp.spv"),
                              shaderPath("light_cull.comp.spv"));
    // After createScene: the capture pipeline needs the bindless heap populated
    // with the scene's textures, and it shares the material set layout.
    createProbeCapturePipeline();
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
    // Outside the initialized_ guard on purpose: this sampler is created while
    // building the descriptor set layout, before initialization completes, so a
    // failure during init would otherwise leak it -- which is exactly how it was
    // found.
    destroyShadowCompareSampler();
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
    // The material sets hold a view of the ambient-occlusion target, which was
    // just recreated at the new size. No-op on the first call, before any
    // material exists; createMaterialDescriptorSet writes the binding then.
    refreshMaterialAmbientOcclusionDescriptors();
}

void Renderer::invalidateTaaHistory()
{
    // The jittered / previous view-projection matrices are Renderer frame state
    // consumed by the main pass, so reset them here; the TAA history and jitter
    // sequence are owned by PostProcessStack.
    frameJitteredProjection_ = glm::mat4{1.0f};
    frameJitteredViewProjection_ = glm::mat4{1.0f};
    previousFrameViewProjection_ = glm::mat4{1.0f};
    previousFrameView_ = glm::mat4{1.0f};
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
    // A render-scale change resizes the same targets a window resize does, so it
    // is applied here at the top of the frame rather than wherever it was
    // requested -- the UI slider, or the dynamic-resolution controller below.
    if (renderResolution_.scale() != renderScaleSettings_.scale) {
        applyRenderScaleChange();
    }

    renderer::FrameResources& frame = frames_[currentFrame_];
    VK_CHECK(vkWaitForFences(context_.vkDevice(), 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX));
    processPortfolioScreenshotReadback(currentFrame_);
    postProcess_.updateAutoExposureFromReadback(currentFrame_);
    tryPrintExposureStats();
    tryPrintGpuTimings(currentFrame_);
    // Straight after the readback that refreshes gpuFrameTimeHistory_, so the
    // controller always sees the freshest GPU frame total available.
    updateDynamicResolution();
    // Before resetGpuCullFrameCounters clears this slot's readback-ready flag.
    updateOcclusionYield(currentFrame_);
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
                    << "  LOD distribution:";
            for (size_t level = 0; level < counters.lodDrawItems.size(); ++level) {
                message << " L" << level << "=" << counters.lodDrawItems[level];
            }
            message << "\n"
                    << "  depth pyramid mips: " << depthPyramid_.mipLevels() << "\n"
                    << "  occlusion culling: " << (isGpuOcclusionCullingActive() ? "enabled" : "disabled") << "\n"
                    << "  occlusion yield: " << occlusionYieldStateName()
                    << " zero=" << occlusionYield_.zeroYieldFrames()
                    << " sinceProbe=" << occlusionYield_.framesSinceProbe()
                    << " pyramid=" << (isDepthPyramidBuildRequired() ? "building" : "skipped") << "\n"
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
    message << "Punctual shadows:\n"
            << "  atlas: " << (punctualShadows_.valid() ? "available" : "unavailable") << "\n"
            << "  casting: " << (usePunctualShadows_ ? "enabled" : "disabled") << "\n"
            << "  slots used: " << punctualShadowSlotsUsed_ << "\n"
            << "  atlas occupancy: " << static_cast<int>(punctualShadows_.occupancy() * 100.0f) << "%\n"
            << "  caster draws recorded: " << punctualShadowDrawsRecorded_ << "\n"
            << "  atlas this frame: " << (punctualShadowCacheHit_ ? "fully cached" : "partial") << "\n"
            << "  tiles redrawn: " << punctualShadowSlotsRedrawn_ << "/" << punctualShadowSlotsUsed_ << "\n"
            << "  cull+record CPU: " << punctualShadowCpuMicros_ << " us\n"
            << "  frames served from cache: " << punctualShadowCachedFrames_ << "\n"
            << "  assignment churn this frame: " << punctualShadowAssignmentChurn_
            << ", cumulative: " << punctualShadowAssignmentChurnTotal_ << "\n";
    if (irradianceProbes_.available()) {
        message << "Irradiance probes:\n"
                << "  enabled: " << (giSettings_.enabled ? "yes" : "no") << "\n"
                << "  probes per frame: " << giSettings_.probesPerFrame << "\n"
                << "  capture draws recorded: " << probeCaptureDrawsRecorded_ << "\n"
                << "  capture cull+record CPU: " << probeCaptureCpuMicroseconds_ << " us\n"
                << "  cursor: " << irradianceProbes_.updateCursor() << "/" << renderer::kProbeCount
                << ", captured: " << irradianceProbes_.capturedProbeCount() << "\n"
                << "  accumulating: " << (irradianceProbes_.gridConverged() ? "yes" : "first cycle") << "\n";
    }
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
    // Flags this as a *new* reading for the dynamic-resolution controller, which
    // must not see the same frame time twice: repeats would fill its median
    // window with duplicates of one sample and defeat the outlier rejection.
    // gpuFrameTimeHistory_.latest() cannot distinguish the two on its own.
    freshGpuFrameMs_ = results.totalGpuTimeMs;
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
    // Assigned like any other setting; drawFrame notices renderResolution_ has
    // gone stale against it and routes the resize through recreateSwapchain,
    // which is the only point in the frame where destroying targets is safe.
    renderScaleSettings_ = settings.renderScale;
    dynamicResolutionSettings_ = settings.dynamicResolution;
    toneMappingSettings_ = settings.toneMapping;
    bloomSettings_ = settings.bloom;
    taaSettings_ = settings.taa;
    ssrSettings_ = settings.ssr;
    ssaoSettings_ = settings.ssao;
    fogSettings_ = settings.fog;
    lodSettings_ = settings.lod;
    giSettings_ = settings.gi;
    debugUiSettings_ = settings.debugUi;

    // Punctual shadows follow the culling toggles: honoured unconditionally at
    // startup, but at runtime the GPU caster cull can only be turned on when the
    // subsystem actually came up.
    showPunctualShadowDebug_ = settings.punctualShadows.debugView;
    usePunctualShadows_ = settings.punctualShadows.enabled;
    // Unconditional, unlike the culling toggles below: every use site already
    // ANDs this with ClusteredLighting::available(), so there is nothing to
    // guard against here.
    useClusteredLighting_ = settings.useClusteredLighting;

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
        useLayeredCascades_ = settings.enableLayeredCascades;
        useAdaptiveOcclusion_ = settings.enableAdaptiveOcclusion;
        useAsyncCompute_ = settings.enableAsyncCompute;
        useBindlessMaterialTextures_ = settings.enableBindlessMaterialTextures;
        // Assigned unguarded here: this runs before the punctual shadow
        // subsystem is created, so cullAvailable() has nothing to report yet.
        // The record path re-checks it every frame regardless.
        useGpuPunctualShadowCulling_ = settings.punctualShadows.gpuCasterCulling;
    } else {
        if (!settings.useGpuCulling || gpuCulling_.available()) {
            useGpuCulling_ = settings.useGpuCulling;
        }
        if (!settings.useGpuShadowCulling || gpuCulling_.shadowAvailable()) {
            useGpuShadowCulling_ = settings.useGpuShadowCulling;
        }
        useGpuOcclusionCulling_ = settings.enableGpuOcclusionCulling;
        useTwoPhaseOcclusion_ = settings.enableTwoPhaseOcclusion;
        useAdaptiveOcclusion_ = settings.enableAdaptiveOcclusion;
        useAsyncCompute_ = settings.enableAsyncCompute;
        if (!settings.punctualShadows.gpuCasterCulling || punctualShadows_.cullAvailable()) {
            useGpuPunctualShadowCulling_ = settings.punctualShadows.gpuCasterCulling;
        }
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
    settings.renderScale = renderScaleSettings_;
    settings.dynamicResolution = dynamicResolutionSettings_;
    settings.toneMapping = toneMappingSettings_;
    settings.bloom = bloomSettings_;
    settings.taa = taaSettings_;
    settings.ssr = ssrSettings_;
    settings.ssao = ssaoSettings_;
    settings.fog = fogSettings_;
    settings.punctualShadows.enabled = usePunctualShadows_;
    settings.punctualShadows.gpuCasterCulling = useGpuPunctualShadowCulling_;
    settings.punctualShadows.debugView = showPunctualShadowDebug_;
    settings.lod = lodSettings_;
    settings.gi = giSettings_;
    settings.csm = csmSettings_;
    settings.debugUi = debugUiSettings_;
    settings.useGpuCulling = useGpuCulling_;
    settings.useGpuShadowCulling = useGpuShadowCulling_;
    settings.enableGpuOcclusionCulling = useGpuOcclusionCulling_;
    settings.enableTwoPhaseOcclusion = useTwoPhaseOcclusion_;
    settings.enableLayeredCascades = useLayeredCascades_;
    settings.enableAdaptiveOcclusion = useAdaptiveOcclusion_;
    settings.useClusteredLighting = useClusteredLighting_;
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

renderer::ProbeGridBounds Renderer::giGridBounds() const
{
    renderer::ProbeGridBounds bounds{};
    bounds.origin = glm::vec3{giSettings_.gridOrigin[0], giSettings_.gridOrigin[1], giSettings_.gridOrigin[2]};
    bounds.spacing = glm::vec3{giSettings_.gridSpacing[0], giSettings_.gridSpacing[1], giSettings_.gridSpacing[2]};
    return bounds;
}

void Renderer::clampRuntimeSettings()
{
    // The settings-struct clamping is GPU-independent and lives in
    // RuntimeSettings.cpp (compiled into VulkanEngineCore) so it can be tested.
    ve::clampRuntimeSettings(
        renderScaleSettings_, dynamicResolutionSettings_, toneMappingSettings_, bloomSettings_, taaSettings_,
        ssrSettings_, ssaoSettings_, fogSettings_, csmSettings_, lodSettings_, giSettings_, debugUiSettings_);

    // Pushed here rather than at each edit site: clampRuntimeSettings runs after
    // every settings change (load, UI edit, reset), so the volume's copy of the
    // grid placement cannot drift from the settings it came from.
    irradianceProbes_.setBounds(giGridBounds());
    irradianceProbes_.setHysteresis(giSettings_.hysteresis);

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

void Renderer::updateRenderResolution()
{
    renderResolution_.update(swapchain_.extent(), renderScaleSettings_.scale);
    // The main depth buffer belongs to the swapchain but is sized with the other
    // internal targets, so it is resized here rather than inside the swapchain's
    // own (re)creation, which does not know the scale. A no-op at scale 1.0.
    // The ALLOCATION, like every other screen-space target. Sizing depth to what
    // the frame writes was the last thing that reallocated on a scale change --
    // and since that path no longer waits for the GPU, it was also destroying an
    // image in flight.
    const VkExtent2D depthExtent = renderResolution_.allocationExtent();
    if (depthExtent.width > 0 && depthExtent.height > 0) {
        swapchain_.resizeDepthImage(depthExtent);
    }
    if (!renderResolution_.isNative()) {
        Logger::info("Render scale " + std::to_string(renderResolution_.scale()) + ": rendering at " +
                     std::to_string(renderResolution_.extent().width) + "x" +
                     std::to_string(renderResolution_.extent().height) + ", presenting at " +
                     std::to_string(renderResolution_.outputExtent().width) + "x" +
                     std::to_string(renderResolution_.outputExtent().height));
    }
}

void Renderer::applyRenderScaleChange()
{
    if (window_.isMinimized()) {
        return;
    }

    const auto begin = std::chrono::steady_clock::now();

    // Every screen-space target is allocated at the maximum render resolution
    // and only its sub-rect is written, so a scale change moves viewports and uv
    // scales and touches no resource at all. The rebuild below is for the case
    // where the *allocation* moved, which only a window resize does.
    const VkExtent2D previousAllocation = renderResolution_.allocationExtent();
    const bool allocationUnchanged = previousAllocation.width == swapchain_.extent().width &&
                                     previousAllocation.height == swapchain_.extent().height;

    if (!allocationUnchanged) {
        // Destroys images earlier frames may still be reading. Measured at 10 ms
        // of the ~15 ms this used to cost every time, and irreducible -- it is
        // real in-flight GPU work, not driver overhead. Not paying it at all is
        // the entire point of the sub-rect design.
        context_.waitIdle();
    }
    const auto afterIdle = std::chrono::steady_clock::now();
    updateRenderResolution();
    if (!allocationUnchanged) {
        recreatePostProcessResources();
    }
    const auto afterRebuild = std::chrono::steady_clock::now();
    // Necessary either way: these three hold a frame of data in the previous
    // sub-rect, and staleness has nothing to do with whether the storage moved.
    // The pyramid in particular would reject visible geometry if it were
    // trusted at the wrong scale.
    invalidateTaaHistory();
    invalidateDepthPyramid();
    postProcess_.invalidateAmbientOcclusionHistory();

    lastRenderScaleApplyMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
    const float idleMs = std::chrono::duration<float, std::milli>(afterIdle - begin).count();
    const float rebuildMs = std::chrono::duration<float, std::milli>(afterRebuild - afterIdle).count();
    Logger::info("Render scale applied: " + std::to_string(renderResolution_.extent().width) + "x" +
                 std::to_string(renderResolution_.extent().height) + " in " +
                 std::to_string(lastRenderScaleApplyMs_) + " ms (waitIdle " + std::to_string(idleMs) +
                 " ms, rebuild " + std::to_string(rebuildMs) + " ms" +
                 (allocationUnchanged ? ", sub-rect only" : ", allocation moved") + ")");
}

void Renderer::updateDynamicResolution()
{
    // Zero means "no new measurement", which is what the controller expects on
    // the frames where no timestamp readback landed. Consumed here so the next
    // frame does not see it again.
    const float gpuFrameMs = freshGpuFrameMs_;
    freshGpuFrameMs_ = 0.0f;

    const float scale =
        dynamicResolution_.update(gpuFrameMs, renderScaleSettings_.scale, dynamicResolutionSettings_);
    if (scale == renderScaleSettings_.scale) {
        return;
    }

    // Written into the setting, not applied here: this runs mid-frame, and the
    // top-of-frame check is the only place a rebuild is safe. That also keeps the
    // controller's output visible on the slider and in the saved settings.
    renderScaleSettings_.scale = scale;
    pendingRenderScale_ = scale;
}

void Renderer::recreateSwapchain()
{
    if (window_.isMinimized()) {
        return;
    }

    context_.waitIdle();
    swapchain_.recreate(context_, window_.framebufferExtent());
    // The render extent can only be derived once the swapchain has picked its
    // actual size (surface capabilities may not grant the requested one), and
    // everything below is sized off it -- so this is the first thing after.
    updateRenderResolution();
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

const char* Renderer::occlusionYieldStateName() const
{
    switch (occlusionYield_.state()) {
    case renderer::OcclusionYieldController::State::Active:
        return "active";
    case renderer::OcclusionYieldController::State::Suspended:
        return "suspended";
    case renderer::OcclusionYieldController::State::Probing:
        return "probing";
    }
    return "unknown";
}

bool Renderer::isDepthPyramidBuildRequired() const
{
    // GPU occlusion culling is the pyramid's only consumer -- nothing else reads
    // it -- so with occlusion off the build is pure cost. Measured at 0.68 ms on
    // the default scene, which it was paying every frame regardless.
    //
    // Deliberately does NOT test depthPyramid_.valid() the way
    // isGpuOcclusionCullingActive does: validity is an *output* of the build, so
    // gating the build on it would latch the pyramid off forever after one skip.
    return useGpuOcclusionCulling_ && isGpuCullingActive() && depthPyramid_.buildAvailable() &&
           depthPyramid_.image() != VK_NULL_HANDLE && depthPyramid_.mipLevels() > 0 &&
           occlusionYield_.shouldBuildPyramid();
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

bool Renderer::isLayeredCascadeRenderingActive() const
{
    // Startup-fixed: the shadow pipelines bake the view mask, and the cascade
    // count is already a restart-only setting, so this cannot change per frame.
    return useLayeredCascades_ && context_.device().multiviewEnabled();
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

} // namespace ve
