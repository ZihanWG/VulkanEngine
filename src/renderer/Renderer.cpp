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

#include "renderer/TransientMemoryPlan.h"
#include "rhi/VulkanAliasingProbe.h"

#include <cstdio>
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

Renderer::Renderer(Window& window, const RendererStartupOverrides& overrides) : window_(window)
{
    runtimeSettingsPath_ = defaultRuntimeSettingsPath();
    sceneDocumentPath_ = defaultSceneDocumentPath();
    loadRuntimeSettingsAtStartup();

    // After the file, before anything reads the settings: an override exists to
    // beat what was persisted. Only the three stage toggles are overridden --
    // the clipmap numerics keep whatever the file configured, so an A/B changes
    // one thing.
    if (overrides.vsmStages.has_value()) {
        vsmSettings_.enableMarking = overrides.vsmStages->marking;
        vsmSettings_.enablePageRendering = overrides.vsmStages->pageRendering;
        vsmSettings_.enableShadows = overrides.vsmStages->shadows;
        Logger::info(std::string("VSM stages overridden from the command line: marking=") +
                     (vsmSettings_.enableMarking ? "on" : "off") +
                     " pageRendering=" + (vsmSettings_.enablePageRendering ? "on" : "off") +
                     " shadows=" + (vsmSettings_.enableShadows ? "on" : "off"));
    }

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
    // The VSM page pool goes here, next to the cascaded shadow map and for the
    // same reason: both are fixed-size light-space resources that do not follow
    // the window. It has to be before createPipeline(), because the page
    // pipeline bakes the pool's depth format -- created after, the pipeline is
    // silently skipped and page rendering never turns on.
    //
    // Gated on the startup setting because it is not cheap: the pool alone is
    // 4096x4096 D32 = 64 MiB, plus ~2 MiB of per-frame cull buffers. Paying that
    // for a subsystem that is off by default would be a worse trade than the
    // 17.48 MiB of bloom aliasing this project already measured and rejected as
    // a default (docs/design_decisions.md). Startup-only for the same reason
    // cascadeCount and enableLayeredCascades are: allocating it on a runtime
    // toggle means recreating the page pipeline, and there is no device-idle
    // wait available in the steady-state frame path.
    if (vsmSettings_.enableMarking) {
        virtualShadowMap_.createPagePoolResources(static_cast<uint32_t>(frames_.size()));
        virtualShadowMap_.createCullResources(static_cast<uint32_t>(frames_.size()), kMaxDrawItems);
    }
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
    const auto sceneCreateStart = std::chrono::steady_clock::now();
    createScene();
    assetLoadStats_.timings.sceneCreateMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sceneCreateStart).count();
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

    initialized_ = true;
}

void Renderer::finalizeAssetLoadStats(double rendererInitMs, double firstFrameMs)
{
    assetLoadStats_.timings.rendererInitMs = rendererInitMs;
    assetLoadStats_.timings.firstFrameMs = firstFrameMs;
    assetLoadStats_.textures = renderer::AssetLoadStatsRecorder::snapshotTextures();

    const VmaAllocator allocator = context_.allocator();
    if (allocator == VK_NULL_HANDLE) {
        return;
    }

    // vmaGetHeapBudgets reports per-heap; only device-local heaps are the VRAM
    // budget this baseline is about. On a unified-memory device every heap is
    // device-local, which is the honest answer there rather than a special case.
    const VkPhysicalDeviceMemoryProperties* memoryProperties = nullptr;
    vmaGetMemoryProperties(allocator, &memoryProperties);
    if (memoryProperties == nullptr) {
        return;
    }

    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());

    renderer::DeviceMemoryUsage usage{};
    for (uint32_t heap = 0; heap < memoryProperties->memoryHeapCount; ++heap) {
        if ((memoryProperties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
            continue;
        }
        usage.deviceLocalUsedBytes += budgets[heap].usage;
        usage.deviceLocalBudgetBytes += budgets[heap].budget;
        usage.deviceLocalAllocatedBytes += budgets[heap].statistics.allocationBytes;
        usage.allocationCount += budgets[heap].statistics.allocationCount;
    }
    usage.valid = true;
    assetLoadStats_.memory = usage;
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
    // createResources reallocates the descriptor sets, so the IBL bindings have to
    // be re-applied every time -- this runs more than once during startup alone,
    // and the first version bound them only from the environment path and lost
    // them to the second recreate.
    ssr_.updateIblDescriptors(prefilteredEnvironmentMap_.imageView(),
                              prefilteredEnvironmentMap_.sampler(),
                              brdfLutTexture_.imageView(),
                              brdfLutTexture_.sampler());
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

    // Needs a recorded frame to know resource lifetimes, so it lands after frame
    // one and then never again unless resources are rebuilt.
    applyTransientAliasingPlan();

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
    // Same reason as the line above: this slot's page-request buffer is about to
    // be cleared and rewritten during recording, so the only chance to read what
    // it produced last time round is here, after its fence proved the copy
    // retired.
    updateVsmPageRequestStats(currentFrame_);
    // Before residency: it decides which pages to invalidate from the skinned
    // caster's bounds and pose, and the page pass later draws whatever pose the
    // palette holds. Advancing the pose after that decision would let the two
    // disagree by a frame, in the direction that loses a shadow.
    advanceSkinnedAnimation(currentFrame_);
    // After the readback above, because it consumes the same request set, and
    // before recording, because the page pass draws what it decides.
    updateVsmResidency(currentFrame_);
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
    const bool wasPending = screenshotCapture_.hasPending();
    screenshotCapture_.processReadback(frameIndex);

    // The readback lags the recorded frame by the in-flight frame count, so
    // completion is detected here rather than assumed at the recorded frame.
    if (frameCaptureRecorded_ && wasPending && !screenshotCapture_.hasPending()) {
        frameCaptureComplete_ = true;
        Logger::info("Frame capture written: " + frameCaptureOutputPath_.string());
    }
}

void Renderer::requestFrameCaptureAt(uint64_t frameNumber, std::filesystem::path outputPath, bool includeUi)
{
    if (frameNumber == 0 || outputPath.empty()) {
        Logger::error("Frame capture needs a frame number of at least 1 and a non-empty output path.");
        return;
    }

    frameCaptureTargetFrame_ = frameNumber;
    frameCaptureOutputPath_ = std::move(outputPath);
    frameCaptureIncludesUi_ = includeUi;
    Logger::info("Frame capture requested at frame " + std::to_string(frameNumber) + " -> " +
                 frameCaptureOutputPath_.string() + (includeUi ? " (including the ImGui overlay)" : ""));
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
    // Frame clock, not steady_clock: with a wall-clock cadence two deterministic
    // runs sample *different frame numbers*, which makes the log look
    // nondeterministic even when every frame is identical.
    const double now = frameClock_.elapsedSeconds();
    if (now - lastExposureLogPrintSeconds_ < 1.0) {
        return;
    }

    lastExposureLogPrintSeconds_ = now;
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

    const double now = frameClock_.elapsedSeconds();
    if (now - lastGpuTimingPrintSeconds_ < 1.0) {
        return;
    }

    lastGpuTimingPrintSeconds_ = now;

    std::ostringstream message;
    message << std::fixed << std::setprecision(3) << "GPU timings:\n"
            << "  Frame total: " << results.totalGpuTimeMs << " ms\n";
    // CPU frame preparation next to the GPU frame total, because the pair is the
    // question -- prep that is comfortably inside the GPU frame is hidden, and
    // removing it would buy no frame time. It was previously visible only in the
    // ImGui panel, which made it unmeasurable in a headless or scripted run.
    if (!framePrepCpuHistory_.empty()) {
        message << "  Frame prep CPU: " << framePrepCpuHistory_.latest()
                << " ms (avg " << framePrepCpuHistory_.average() << ", max " << framePrepCpuHistory_.max()
                << ")\n";
    }
    message << "  draw items: " << allDrawItems_.size() << ", objects: " << renderObjects_.size() << "\n"
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
    // Page-request measurement. Printed rather than left in the debug panel
    // because --capture-frame excludes ImGui, so a GPU-derived number that only
    // exists on screen cannot be checked from a headless or scripted run.
    if (isVsmPageMarkingActive()) {
        const renderer::VsmClipmapSettings clipmap = vsmClipmapSettings();
        message << "VSM page marking:\n"
                << "  levels: " << clipmap.levelCount << ", level0 extent: " << clipmap.level0Extent
                << " m, texel0: " << renderer::vsmTexelWorldSize(clipmap, 0) << " m\n"
                << "  mark threads: " << virtualShadowMap_.lastMarkThreadCount()
                << " (stride " << vsmSettings_.markBlockStride << ")\n";
        if (vsmPageRequestStatsValid_) {
            message << "  requested pages: " << vsmPageRequestStats_.requestedPages << "/"
                    << renderer::kVsmMaxVirtualPages << " (peak " << vsmPeakRequestedPages_ << ", pool holds "
                    << renderer::kVsmPagePoolPageCount << ")\n"
                    << "  levels touched: " << vsmPageRequestStats_.lowestRequestedLevel << ".."
                    << vsmPageRequestStats_.highestRequestedLevel << "\n"
                    << "  per level:";
            for (uint32_t level = 0; level < clipmap.levelCount; ++level) {
                message << " L" << level << "=" << vsmPageRequestStats_.requestedPerLevel[level];
            }
            message << "\n";
        } else {
            message << "  requested pages: pending first readback\n";
        }
        if (isVsmPageRenderingActive()) {
            message << "  addressable now: " << vsmResidencyStats_.addressablePages << "/"
                    << vsmResidencyStats_.requestedPages << "\n"
                    << "  resident: " << vsmResidencyStats_.residentPages << "/"
                    << renderer::kVsmPagePoolPageCount << ", cached " << vsmResidencyStats_.cachedPages << "\n"
                    << "  drawn this frame: " << vsmPageDrawsRecorded_ << "/" << renderer::kMaxVsmPagesPerFrame
                    << " (" << vsmPageDrawsTotal_ << " since start)" 
                    << ", over budget " << vsmResidencyStats_.overBudgetPages << ", refused "
                    << vsmResidencyStats_.refusedPages << ", evicted " << vsmResidencyStats_.evictions << "\n"
                    << "  casters over the per-page cap: " << vsmPageCullOverflow_ << "\n"
                    << "  casters changed: " << vsmCastersChangedThisFrame_ << ", pages they invalidated: "
                    << vsmResidencyStats_.casterInvalidatedPages << "\n"
                    << "  skinned caster: pages drawn into " << vsmSkinnedPageDrawsRecorded_ << "\n";
            message << "  directional shadows: "
                    << (isVsmDirectionalShadowActive() ? "sampled from the page pool" : "cascades")
                    << "\n";
        } else {
            message << "  page rendering: disabled\n";
        }
    }
    message << "Punctual shadows:\n"
            << "  atlas: " << (punctualShadows_.valid() ? "available" : "unavailable") << "\n"
            << "  casting: " << (usePunctualShadows_ ? "enabled" : "disabled") << "\n"
            << "  slots used: " << punctualShadowSlotsUsed_ << "\n"
            << "  atlas occupancy: " << static_cast<int>(punctualShadows_.occupancy() * 100.0f) << "%\n"
            << "  caster draws recorded: " << punctualShadowDrawsRecorded_
            << " (skinned caster tiles: " << punctualShadowSkinnedDrawsRecorded_ << ")\n"
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

    // Cascade cache state, which until now lived only in the debug UI. A
    // scripted run could not see whether the cache was holding -- and "the
    // cascades stopped caching" is exactly the failure mode a caster keyed on
    // the wrong thing produces.
    message << "\n  cascade cache: " << (csmSettings_.enableCascadeCache ? "enabled" : "disabled")
            << ", redrawn this frame " << cascadeShadowCascadesRedrawn_ << "/" << activeCascadeCount()
            << ", consecutive cached frames " << cascadeShadowCachedFrames_;

    // The skinned caster is drawn directly and is in none of the counts above,
    // so without this line there is no way to tell from a scripted run whether
    // it cast anything -- and --capture-frame excludes the debug UI that would
    // otherwise show it.
    message << "\n  skinned caster: ";
    if (!skinnedCasterActive()) {
        message << "inactive";
    } else {
        message << "cascades";
        for (uint32_t cascadeIndex = 0; cascadeIndex < activeCascadeCount(); ++cascadeIndex) {
            if (skinnedCasterCastsIntoCascade(cascadeIndex)) {
                message << ' ' << cascadeIndex;
            }
        }
        const renderer::Aabb& bounds = skinnedMesh_.worldBounds();
        message << ", pose " << skinnedMesh_.poseHash() << ", bounds (" << bounds.min.x << ", " << bounds.min.y
                << ", " << bounds.min.z << ") to (" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z
                << ")";
        if (isLayeredCascadeRenderingActive()) {
            message << " [SKIPPED: layered cascades]";
        }
    }
    Logger::info(message.str());
}

void Renderer::updateCpuFrameTime()
{
    // The one place the frame clock is advanced. Everything downstream in this
    // frame -- animation, exposure adaptation, the editor camera -- reads the
    // clock rather than taking its own reading, so all of them stay consistent
    // with each other and all of them become reproducible together.
    const auto now = std::chrono::steady_clock::now();
    frameClock_.advance(std::chrono::duration<double>(now - startTime_).count());

    // Armed here because the clock has just advanced, so frameCount() is this
    // frame's number -- the same number the caller asked for.
    if (frameCaptureTargetFrame_ != 0 && !frameCaptureRecorded_ &&
        frameClock_.frameCount() == frameCaptureTargetFrame_) {
        frameCapturePending_ = true;
    }

    cpuFrameDeltaMs_ = static_cast<float>(frameClock_.deltaSeconds() * 1000.0);
    lastFrameStartTime_ = now;
    cpuFps_ = cpuFrameDeltaMs_ > 0.0f ? 1000.0f / cpuFrameDeltaMs_ : 0.0f;

    // Exposure adaptation used to take its own steady_clock reading inside the
    // exposure-reduce recording. Pushing the frame time in instead removes that
    // second, independent time source.
    postProcess_.setFrameTimeSeconds(static_cast<float>(frameClock_.elapsedSeconds()));
}

void Renderer::applyTransientAliasingPlan()
{
    // Replan when the allocation extent moves. A resize recreates the bloom
    // images at a new size, and a plan computed for the old extent would place
    // them at offsets that overlap or run past the pool. PostProcessStack drops
    // the stale plan on its side; this is what makes a new one get computed
    // rather than the chain staying private forever after the first resize.
    const VkExtent2D allocationExtent = renderResolution_.allocationExtent();
    if (transientAliasingApplied_ && (allocationExtent.width != transientAliasingPlanExtent_.width ||
                                      allocationExtent.height != transientAliasingPlanExtent_.height)) {
        transientAliasingApplied_ = false;
    }

    if (!useTransientAliasing_ || transientAliasingApplied_) {
        return;
    }

    // Only the bloom chain for now. Its resources are created and described in
    // one place, and its descriptors are rewritten by the same path that creates
    // the images, so nothing outside PostProcessStack has to change.
    const std::vector<renderer::RenderGraph::TransientTextureRecord> transients = renderGraph_.transientTextures();
    if (transients.empty()) {
        // No frame recorded yet; try again after the next one.
        return;
    }

    const std::vector<renderer::RenderGraphResourceLifetime> lifetimes = renderer::computeTextureLifetimes(
        renderGraph_.debugPasses(), renderGraph_.executionOrder(), renderGraph_.textureCount());

    std::vector<renderer::TransientAllocationRequest> requests;
    uint32_t commonMemoryTypeBits = ~0u;
    VkDeviceSize alignment = 1;
    for (const renderer::RenderGraph::TransientTextureRecord& record : transients) {
        if (record.name.rfind("Bloom", 0) != 0) {
            continue;
        }
        if (record.resourceIndex >= lifetimes.size() || !lifetimes[record.resourceIndex].used) {
            continue;
        }

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(context_.vkDevice(), record.image, &requirements);

        renderer::TransientAllocationRequest request{};
        request.name = record.name;
        request.size = requirements.size;
        request.alignment = requirements.alignment;
        request.firstPass = lifetimes[record.resourceIndex].firstPass;
        request.lastPass = lifetimes[record.resourceIndex].lastPass;
        requests.push_back(std::move(request));

        commonMemoryTypeBits &= requirements.memoryTypeBits;
        alignment = std::max(alignment, requirements.alignment);
    }

    if (requests.empty()) {
        transientAliasingApplied_ = true;
        return;
    }

    const renderer::TransientMemoryPlan plan = renderer::planTransientMemory(requests);

    std::unordered_map<std::string, VkDeviceSize> offsets;
    offsets.reserve(plan.allocations.size());
    for (const renderer::TransientAllocation& allocation : plan.allocations) {
        if (allocation.placed) {
            offsets.emplace(allocation.name, allocation.offset);
        }
    }

    // Images bound into the old pool must be destroyed before its memory is
    // freed, and recreatePostProcessResources destroys them. A one-time idle at
    // startup, on the same path a resize already takes -- never in the steady
    // frame loop.
    waitIdle();
    postProcess_.setBloomAliasPlan(std::move(offsets), plan.poolBytes, commonMemoryTypeBits, alignment);
    recreatePostProcessResources();
    transientAliasingApplied_ = true;
    transientAliasingPlanExtent_ = allocationExtent;

    const auto mib = [](VkDeviceSize bytes) {
        char buffer[32] = {};
        std::snprintf(buffer, sizeof(buffer), "%.2f", static_cast<double>(bytes) / (1024.0 * 1024.0));
        return std::string(buffer);
    };
    if (postProcess_.bloomImagesAreAliased()) {
        Logger::info("Bloom transient aliasing active: " + mib(plan.unaliasedBytes) + " MiB of images in a " +
                     mib(postProcess_.bloomPoolBytes()) + " MiB pool (" + mib(plan.savedBytes()) + " MiB saved).");
    } else {
        Logger::info("Bloom transient aliasing requested but not active; images keep private allocations.");
    }
}

void Renderer::logTransientPoolReport()
{
    const std::vector<renderer::RenderGraph::TransientTextureRecord> transients = renderGraph_.transientTextures();
    if (transients.empty()) {
        Logger::warn("Transient pool report: the graph recorded no transient textures.");
        return;
    }

    struct Entry {
        std::string name;
        VkDeviceSize bytes = 0;
        VkDeviceSize alignment = 0;
        uint32_t memoryTypeBits = 0;
    };

    std::vector<Entry> entries;
    entries.reserve(transients.size());
    VkDeviceSize totalBytes = 0;
    uint32_t commonMemoryTypeBits = ~0u;

    for (const renderer::RenderGraph::TransientTextureRecord& record : transients) {
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(context_.vkDevice(), record.image, &requirements);
        entries.push_back(Entry{record.name, requirements.size, requirements.alignment, requirements.memoryTypeBits});
        totalBytes += requirements.size;
        commonMemoryTypeBits &= requirements.memoryTypeBits;
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        if (lhs.bytes != rhs.bytes) {
            return lhs.bytes > rhs.bytes;
        }
        return lhs.name < rhs.name;
    });

    const auto mib = [](VkDeviceSize bytes) {
        char buffer[32] = {};
        std::snprintf(buffer, sizeof(buffer), "%8.2f", static_cast<double>(bytes) / (1024.0 * 1024.0));
        return std::string(buffer);
    };

    std::string message = "\n=== Transient pool (no aliasing) ===\n";
    for (const Entry& entry : entries) {
        message += "  " + mib(entry.bytes) + " MiB  align " + std::to_string(entry.alignment) + "  " + entry.name + "\n";
    }
    message += "  ----\n";
    message += "  " + mib(totalBytes) + " MiB  total across " + std::to_string(entries.size()) + " transient textures\n";

    char bitsBuffer[16] = {};
    std::snprintf(bitsBuffer, sizeof(bitsBuffer), "0x%x", commonMemoryTypeBits);
    message += std::string("  common memoryTypeBits: ") + bitsBuffer;
    if (commonMemoryTypeBits == 0) {
        // The one case the pool must fall back from rather than treat as a bug.
        message += "  (EMPTY -- a single shared pool is impossible here)";
    }
    message += "\n  render extent: " + std::to_string(renderResolution_.extent().width) + "x" +
               std::to_string(renderResolution_.extent().height);

    // What the packer would achieve on this frame. Reported before anything is
    // wired up, so the decision to build the pool rests on a number rather than
    // on the hope that aliasing helps.
    const std::vector<renderer::RenderPassNode>& passes = renderGraph_.debugPasses();
    const std::vector<renderer::RenderGraphResourceLifetime> lifetimes =
        renderer::computeTextureLifetimes(passes, renderGraph_.executionOrder(), renderGraph_.textureCount());

    std::vector<renderer::TransientAllocationRequest> requests;
    requests.reserve(transients.size());
    for (const renderer::RenderGraph::TransientTextureRecord& record : transients) {
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(context_.vkDevice(), record.image, &requirements);

        renderer::TransientAllocationRequest request{};
        request.name = record.name;
        request.size = requirements.size;
        request.alignment = requirements.alignment;
        if (record.resourceIndex < lifetimes.size() && lifetimes[record.resourceIndex].used) {
            request.firstPass = lifetimes[record.resourceIndex].firstPass;
            request.lastPass = lifetimes[record.resourceIndex].lastPass;
        } else {
            // Empty-lifetime convention: dropped rather than packed.
            request.firstPass = 1;
            request.lastPass = 0;
        }
        requests.push_back(std::move(request));
    }

    const renderer::TransientMemoryPlan plan = renderer::planTransientMemory(requests);

    message += "\n--- with aliasing (planned, not yet applied) ---";
    for (const renderer::TransientAllocation& allocation : plan.allocations) {
        if (!allocation.placed) {
            message += "\n  " + std::string(11, ' ') + "(dropped: no surviving pass)  " + allocation.name;
            continue;
        }
        message += "\n  offset " + mib(allocation.offset) + " MiB  passes " + std::to_string(allocation.firstPass) +
                   "-" + std::to_string(allocation.lastPass) + "  " + allocation.name;
    }
    message += "\n  ----";
    message += "\n  " + mib(plan.unaliasedBytes) + " MiB  without aliasing";
    message += "\n  " + mib(plan.poolBytes) + " MiB  pool with aliasing";
    message += "\n  " + mib(plan.savedBytes()) + " MiB  saved";
    {
        char ratioBuffer[32] = {};
        std::snprintf(ratioBuffer, sizeof(ratioBuffer), "%.1f%%", plan.reuseRatio() * 100.0);
        message += std::string("  (") + ratioBuffer + " reuse)";
    }
    message += "\n  passes recorded: " + std::to_string(passes.size());
    message += "\n=== end transient pool ===";

    Logger::info(message);
}

void Renderer::logImageMemoryAliasingProbe()
{
    const rhi::AliasingProbeResult probe = rhi::probeImageMemoryAliasing(context_, commandContext_);

    const auto mib = [](VkDeviceSize bytes) {
        return std::to_string(static_cast<double>(bytes) / (1024.0 * 1024.0));
    };

    std::string message = "Image memory aliasing probe: ";
    message += probe.supported ? "SUPPORTED" : "UNSUPPORTED";
    message += " (" + probe.detail + ")";
    message += "\n  first image (RGBA16F 512x512):  " + mib(probe.firstImageBytes) + " MiB";
    message += "\n  second image (R8 256x256):      " + mib(probe.secondImageBytes) + " MiB";
    message += "\n  shared allocation:              " + mib(probe.sharedAllocationBytes) + " MiB";
    message += "\n  second image offset:            " + std::to_string(probe.secondImageOffset);
    message += std::string("\n  aliased write observed:         ") +
               (probe.aliasedWriteObserved ? "yes (the two images provably share bytes)"
                                           : "no (inconclusive -- see AliasingProbeResult)");
    message += "\n  common memoryTypeBits:          0x";
    {
        char buffer[16] = {};
        std::snprintf(buffer, sizeof(buffer), "%x", probe.commonMemoryTypeBits);
        message += buffer;
    }

    if (probe.supported) {
        Logger::info(message);
    } else {
        Logger::warn(message);
    }
}

void Renderer::useDeterministicFrameClock(double stepSeconds)
{
    frameClock_.useFixedStep(stepSeconds);

    // Dynamic resolution feeds measured GPU frame time back into the render
    // extent, so leaving it on would let machine speed change the image even
    // with a fixed timestep. It defaults off, but a persisted
    // config/runtime_settings.json can have turned it on.
    dynamicResolutionSettings_.enabled = false;

    Logger::info("Deterministic frame clock enabled: fixed " +
                 std::to_string(frameClock_.fixedStepSeconds() * 1000.0) +
                 " ms timestep, dynamic resolution pinned off.");
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
    // Straight from the CPU-side budget, not from GPU counters: these count
    // geometry that never reached the GPU at all, so no readback can see them.
    snapshot.droppedObjects = frameCapacityBudget_.droppedObjects();
    snapshot.droppedDrawItems = frameCapacityBudget_.droppedDrawItems();
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

    // Whole-struct, like every other settings group above, and in RuntimeSettings.h
    // so it is testable -- see applyCsmSettings for what the field-by-field copy
    // that used to live here silently dropped.
    applyCsmSettings(settings.csm, csmSettings_, mode == RuntimeSettingsApplyMode::Startup);

    // Whole-struct too, and safe to follow at runtime: every field feeds the
    // marking dispatch's per-frame parameter upload, none of them size a
    // resource. Changing them invalidates the peak, which describes a clipmap
    // that no longer exists.
    // enableMarking is the one field that cannot follow at runtime: it decides
    // whether the 64 MiB page pool and its buffers exist at all, and only
    // startup can answer that. Everything else follows immediately.
    const bool markingWasEnabled = vsmSettings_.enableMarking;
    if (vsmSettings_ != settings.vsm) {
        vsmSettings_ = settings.vsm;
        if (mode != RuntimeSettingsApplyMode::Startup) {
            vsmSettings_.enableMarking = markingWasEnabled;
        }
        vsmPeakRequestedPages_ = 0;
        vsmPageRequestStatsValid_ = false;
        // A clipmap settings change moves every page's world rect, so nothing in
        // the pool describes what its entry claims any more.
        virtualShadowMap_.invalidateResidency();
    }

    if (mode == RuntimeSettingsApplyMode::Startup) {
        useTransientAliasing_ = settings.enableTransientAliasing;
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
    settings.vsm = vsmSettings_;
    settings.debugUi = debugUiSettings_;
    settings.enableTransientAliasing = useTransientAliasing_;
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

void Renderer::updateVsmPageRequestStats(uint32_t frameIndex)
{
    if (!isVsmPageMarkingActive()) {
        // Left as it was rather than zeroed: turning marking off should freeze
        // the last measurement on screen, not replace it with a zero that reads
        // like "no pages needed".
        return;
    }

    renderer::VsmPageRequestStats stats{};
    if (!virtualShadowMap_.readRequestStats(frameIndex, vsmSettings_.clipmapLevels, stats)) {
        return;
    }

    vsmPageRequestStats_ = stats;
    vsmPageRequestStatsValid_ = true;
    vsmPeakRequestedPages_ = std::max(vsmPeakRequestedPages_, stats.requestedPages);
}

void Renderer::updateVsmCasterInvalidation()
{
    vsmCastersChangedThisFrame_ = 0;
    if (!isVsmPageRenderingActive()) {
        // Residency is not being maintained, so there is nothing to keep in step
        // with. The states are dropped so that re-enabling starts from a clean
        // comparison rather than against a scene that has moved since.
        vsmCasterStates_.clear();
        skinnedVsmCasterState_ = VsmCasterState{};
        return;
    }

    const size_t objectCount = frameWorldBounds_.size();

    // An object count change means indices no longer name the same objects, so
    // every remembered key is meaningless. Rare (scene edits, spawns) and cheap
    // to handle bluntly.
    if (vsmCasterStates_.size() != objectCount) {
        vsmCasterStates_.assign(objectCount, VsmCasterState{});
        skinnedVsmCasterState_ = VsmCasterState{};
        virtualShadowMap_.invalidateResidency();
    }

    // One key per object, accumulated over the draw items that belong to it.
    // Hashing the model matrix rather than the world bounds is deliberate:
    // rotating a symmetric object leaves its AABB identical while changing every
    // shadow it casts.
    //
    // What this key deliberately omits, and why -- the list matters because each
    // omission becomes a stale-shadow bug the moment the feature it depends on
    // lands, and a stale shadow shows up far from its cause:
    //
    //   * alphaCutoff. Covered only indirectly: promoting a material to or from
    //     MASK changes the draw item's bucket, which IS hashed. A cutoff edited
    //     within MASK leaves a stale cutout shadow. CascadeShadowCaster hashes
    //     the value outright; do the same if that slider is ever used in anger.
    //   * The cull-selected LOD level. The page cull emits no level, so pages
    //     always draw the authored geometry -- deliberately, see
    //     docs/virtual_shadow_maps.md. ADD IT if page LOD ever lands.
    //   * The raster depth bias. The cascades hash it; here it is a compile-time
    //     constant in ShadowSettings with no UI, so it cannot move. ADD IT if it
    //     ever becomes a setting, because it is baked into the page pipeline and
    //     a change would leave every cached page rendered with the old value.
    //
    // Covered elsewhere rather than here: the light direction and the clipmap
    // settings (hashed by updateResidency, which drops residency wholesale), and
    // a scene switch (resetSceneState clears these keys, because mesh and
    // material are hashed by pointer and only unique within one scene).
    vsmCasterKeys_.assign(objectCount, renderer::ShadowCacheKey{});
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        vsmCasterKeys_[objectIndex].reset();
        vsmCasterKeys_[objectIndex].add(renderObjects_[objectIndex].transform.modelMatrix());
    }
    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex >= objectCount) {
            continue;
        }
        // Blended geometry is not a caster, so a change to it cannot change any
        // page -- the page cull filters it out before anything is drawn.
        if (drawItem.bucket == RenderBucket::Blend) {
            continue;
        }
        renderer::ShadowCacheKey& key = vsmCasterKeys_[drawItem.objectIndex];
        key.add(drawItem.mesh);
        key.add(drawItem.material);
        key.add(drawItem.firstIndex);
        key.add(drawItem.indexCount);
        key.add(static_cast<uint32_t>(drawItem.bucket));
    }

    const renderer::VsmClipmapSettings clipmap = vsmClipmapSettings();
    const glm::mat4 lightView = renderer::vsmLightView(directionalLightSettings_.direction);
    const glm::vec2 cameraLightSpaceXy = glm::vec2(lightView * glm::vec4(frameCameraPosition_, 1.0f));

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        VsmCasterState& state = vsmCasterStates_[objectIndex];
        const uint64_t key = vsmCasterKeys_[objectIndex].value();
        const renderer::Aabb& bounds = frameWorldBounds_[objectIndex];

        if (state.valid && state.key == key) {
            continue;
        }
        ++vsmCastersChangedThisFrame_;

        // Where it was, then where it is. Both matter: the old pages still hold
        // its depth, and the new ones do not hold it yet.
        if (state.valid && state.bounds.valid()) {
            virtualShadowMap_.invalidatePagesForBounds(
                clipmap, lightView, cameraLightSpaceXy, state.bounds.min, state.bounds.max);
        }
        if (bounds.valid()) {
            virtualShadowMap_.invalidatePagesForBounds(
                clipmap, lightView, cameraLightSpaceXy, bounds.min, bounds.max);
        }

        state.key = key;
        state.bounds = bounds;
        state.valid = true;
    }

    // The skinned caster, which the loop above cannot reach: it is keyed by
    // objectIndex over renderObjects_, and the skinned mesh is not one. Its key
    // is the joint palette rather than a model matrix, for the reason this whole
    // mechanism exists -- a skinned mesh deforms without its transform moving,
    // so the page keeps both its coordinates and its physical page while its
    // depth silently describes a pose that is gone.
    //
    // Its bounds change every frame it animates, so this dirties the pages it
    // covers every frame. That is not a tuning failure: animated geometry has no
    // cacheable shadow, and the honest cost is a redraw of the few pages it
    // touches. The count is reported next to the others.
    const bool skinnedCasts = skinnedCasterActive();
    const uint64_t skinnedKey = skinnedCasts ? skinnedMesh_.poseHash() : 0;
    if (skinnedVsmCasterState_.valid && skinnedVsmCasterState_.key != skinnedKey) {
        ++vsmCastersChangedThisFrame_;
        // Where it was, then where it is -- the same pair the loop above uses,
        // and for the same reason.
        if (skinnedVsmCasterState_.bounds.valid()) {
            virtualShadowMap_.invalidatePagesForBounds(clipmap,
                                                       lightView,
                                                       cameraLightSpaceXy,
                                                       skinnedVsmCasterState_.bounds.min,
                                                       skinnedVsmCasterState_.bounds.max);
        }
        if (skinnedCasts && skinnedMesh_.worldBounds().valid()) {
            virtualShadowMap_.invalidatePagesForBounds(clipmap,
                                                       lightView,
                                                       cameraLightSpaceXy,
                                                       skinnedMesh_.worldBounds().min,
                                                       skinnedMesh_.worldBounds().max);
        }
    }

    skinnedVsmCasterState_.key = skinnedKey;
    skinnedVsmCasterState_.bounds = skinnedCasts ? skinnedMesh_.worldBounds() : renderer::Aabb{};
    skinnedVsmCasterState_.valid = true;
}

void Renderer::updateVsmResidency(uint32_t frameIndex)
{
    if (!isVsmPageRenderingActive()) {
        vsmResidencyStats_ = {};
        vsmPageCullOverflow_ = 0;
        // Residency is left standing rather than dropped: the pages in the pool
        // are still correct for their absolute coordinates, and turning the
        // toggle back on should not have to redraw all of them.
        return;
    }

    ++vsmFrameCounter_;

    // Over-cap casters from the frame that last used this slot, read before the
    // dispatch below overwrites the counters.
    uint32_t overflow = 0;
    if (virtualShadowMap_.readPageCullOverflow(frameIndex, overflow)) {
        vsmPageCullOverflow_ = overflow;
    }

    // Before residency, so the pages a moved caster dirtied are queued by this
    // same update rather than a frame later.
    updateVsmCasterInvalidation();

    const glm::mat4 lightView = renderer::vsmLightView(directionalLightSettings_.direction);
    const glm::vec2 cameraLightSpaceXy = glm::vec2(lightView * glm::vec4(frameCameraPosition_, 1.0f));
    vsmResidencyStats_ = virtualShadowMap_.updateResidency(frameIndex,
                                                           vsmClipmapSettings(),
                                                           cameraLightSpaceXy,
                                                           directionalLightSettings_.direction,
                                                           vsmFrameCounter_);
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
        ssrSettings_, ssaoSettings_, fogSettings_, csmSettings_, vsmSettings_, lodSettings_, giSettings_,
        debugUiSettings_);

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
    // Deliberately does NOT test depthPyramid_.valid() the way
    // isGpuOcclusionCullingActive does: validity is an *output* of the build, so
    // gating the build on it would latch the pyramid off forever after one skip.
    const bool pyramidUsable = depthPyramid_.buildAvailable() && depthPyramid_.image() != VK_NULL_HANDLE &&
                               depthPyramid_.mipLevels() > 0;
    if (!pyramidUsable) {
        return false;
    }

    // GPU occlusion culling used to be the pyramid's only consumer, so with
    // occlusion off the build was pure cost (0.68 ms on the default scene) and
    // the yield controller was free to suspend it.
    //
    // VSM page marking is the second consumer, and it is not covered by the
    // yield decision at all: that controller suspends the build when occlusion
    // culls nothing, which says nothing about whether the marking pass still
    // needs depth. Left out, the skip calls depthPyramid_.invalidate() and page
    // marking measures one frame and then reports zero forever -- which is
    // exactly what it did before this branch existed.
    if (isVsmPageMarkingActive()) {
        return true;
    }

    return useGpuOcclusionCulling_ && isGpuCullingActive() && occlusionYield_.shouldBuildPyramid();
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
