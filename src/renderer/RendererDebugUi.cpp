// Debug ImGui panels for the renderer.
//
// These methods are still members of ve::Renderer; their definitions were split
// out of Renderer.cpp to keep that translation unit focused on frame rendering
// and to let the debug UI recompile independently. Shared file-local helpers
// live in RendererInternal.h, which is included by both translation units.
#include "renderer/Renderer.h"
#include "renderer/RendererInternal.h"

#include <imgui.h>

namespace ve {

void Renderer::buildDebugUi()
{
    if (!imguiLayer_.initialized()) {
        return;
    }

    // First run only: the layout file takes over from then on, so moving or
    // resizing the panel sticks across restarts.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 16.0f, viewport->WorkPos.y + 16.0f),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(460.0f, std::min(viewport->WorkSize.y - 32.0f, 780.0f)), ImGuiCond_FirstUseEver);
    ImGui::Begin("VulkanEngine Debug");

    drawStatusStrip();

    ImGui::Checkbox("Advanced mode", &debugUiSettings_.advancedMode);
    ImGui::SetItemTooltip("Off: only the common look and render-scale knobs.\n"
                          "On: reveals the Scene and Diagnostics tabs, the profiler and culling\n"
                          "readouts, and the side panels.");

    // Tabs rather than one column of sections. There are twenty-seven of them and
    // every one used to be default-open, so the window was taller than the display
    // and anything you wanted was a long scroll away.
    //
    // Grouped by *task* rather than by subsystem, which is the part that matters:
    // a render-scale experiment wants the scale, the profiler and the culling
    // counts at once, and those were previously at the top, the bottom and the
    // other bottom of the same scroll.
    if (ImGui::BeginTabBar("VulkanEngineDebugTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Performance")) {
            drawRenderScaleDebugUi();
            if (debugUiSettings_.advancedMode) {
                if (debugUiSettings_.showGpuTimingGraphs &&
                    ImGui::CollapsingHeader("GPU Profiler", ImGuiTreeNodeFlags_DefaultOpen)) {
                    drawGpuTimingDebugUi();
                }
                if (debugUiSettings_.showCullingStats &&
                    ImGui::CollapsingHeader("Culling", ImGuiTreeNodeFlags_DefaultOpen)) {
                    drawCullingDebugUi();
                }
                drawGpuCullingDebugUi();
                drawMeshLodDebugUi();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Image")) {
            drawToneMappingDebugUi();
            if (debugUiSettings_.showExposureGraphs &&
                ImGui::CollapsingHeader("Exposure", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawExposureDebugUi();
            }
            drawBloomDebugUi();
            drawTaaDebugUi();
            drawSsrDebugUi();
            drawSsaoDebugUi();
            drawVolumetricFogDebugUi();
            ImGui::EndTabItem();
        }

        // Scene and Diagnostics hold nothing outside advanced mode, so they are
        // not created at all rather than opening onto an empty page.
        if (debugUiSettings_.advancedMode && ImGui::BeginTabItem("Scene")) {
            drawScenePresetDebugUi();
            drawLightsDebugUi();
            drawShadowsDebugUi();
            drawEnvironmentDebugUi();
            drawSkeletalAnimationDebugUi();
            drawPortfolioCaptureDebugUi();
            ImGui::EndTabItem();
        }

        if (debugUiSettings_.advancedMode && ImGui::BeginTabItem("Diagnostics")) {
            drawDebugViewToggles();
            if (debugUiSettings_.showRenderGraphPanel &&
                ImGui::CollapsingHeader("Render Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawRenderGraphDebugUi();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Settings")) {
            drawControlsDebugUi();
            if (debugUiSettings_.advancedMode) {
                drawRuntimeSettingsDebugUi();
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();

    // Side panels are controlled by the toggles under Advanced, so only surface them
    // in advanced mode (their show* flags default to true).
    // Its own window rather than a section of the main panel, and not behind the
    // advanced-mode gate.
    //
    // The main panel's sections are all default-open, so it is already taller
    // than the display; anything appended to it lands below the fold, where ImGui
    // clips it rather than drawing it. That is not merely inconvenient here --
    // the atlas previews are how this subsystem is checked at all, and a preview
    // that is never drawn cannot be looked at or reasoned about.
    if (debugUiSettings_.showIrradianceProbePanel) {
        if (ImGui::Begin("Irradiance Probes", &debugUiSettings_.showIrradianceProbePanel)) {
            drawIrradianceProbesDebugUi();
        }
        ImGui::End();
    }

    if (debugUiSettings_.advancedMode) {
        if (debugUiSettings_.showSceneHierarchyPanel) {
            if (ImGui::Begin("Scene Hierarchy", &debugUiSettings_.showSceneHierarchyPanel)) {
                drawSceneHierarchyDebugUi();
            }
            ImGui::End();
        }

        if (debugUiSettings_.showMaterialInspectorPanel) {
            if (ImGui::Begin("Material Inspector", &debugUiSettings_.showMaterialInspectorPanel)) {
                drawMaterialInspectorDebugUi();
            }
            ImGui::End();
        }

        if (debugUiSettings_.showTextureDebugPanel) {
            if (ImGui::Begin("Texture Debug Views", &debugUiSettings_.showTextureDebugPanel)) {
                drawTextureDebugUi();
            }
            ImGui::End();
        }

        if (debugUiSettings_.showRenderTargetDebugPanel) {
            if (ImGui::Begin("Render Target Debug Views", &debugUiSettings_.showRenderTargetDebugPanel)) {
                drawRenderTargetDebugUi();
            }
            ImGui::End();
        }
    }

    clampRuntimeSettings();
}

void Renderer::drawControlsDebugUi()
{
    if (!ImGui::CollapsingHeader("Controls")) {
        return;
    }

    ImGui::SeparatorText("Selection & Gizmo");
    ImGui::BulletText("Left-click: select object");
    ImGui::BulletText("W / E / R: gizmo Move / Rotate / Scale");
    ImGui::BulletText("X: toggle gizmo World / Local space");

    ImGui::SeparatorText("Camera");
    ImGui::BulletText("Hold Right Mouse: fly  (WASD move, Q/E down/up, Shift faster, move mouse to look)");
    ImGui::BulletText("Alt + Left-drag: orbit");
    ImGui::BulletText("Middle-drag: pan");
    ImGui::BulletText("Scroll wheel: zoom / dolly");

    ImGui::SeparatorText("Capture");
    ImGui::BulletText("F11: toggle portfolio capture mode");
    ImGui::BulletText("F12: save portfolio screenshot");
}

void Renderer::drawDebugViewToggles()
{
    if (!ImGui::CollapsingHeader("Debug Views", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::Checkbox("Show Render Graph panel", &debugUiSettings_.showRenderGraphPanel);
    ImGui::Checkbox("Show Scene Hierarchy panel", &debugUiSettings_.showSceneHierarchyPanel);
    ImGui::Checkbox("Show Material Inspector", &debugUiSettings_.showMaterialInspectorPanel);
    ImGui::Checkbox("Show Texture Debug Views", &debugUiSettings_.showTextureDebugPanel);
    ImGui::Checkbox("Show Render Target Debug Views", &debugUiSettings_.showRenderTargetDebugPanel);
    ImGui::Checkbox("Show GPU Profiler panel", &debugUiSettings_.showGpuTimingGraphs);
    ImGui::Checkbox("Show Culling stats", &debugUiSettings_.showCullingStats);
    ImGui::Checkbox("Show Exposure graphs", &debugUiSettings_.showExposureGraphs);
}

// Always visible, above the tabs. Changing a parameter and then having to go
// somewhere else to find out whether it did anything was its own kind of
// friction, and these four readings are what almost every experiment here is
// actually watching.
void Renderer::drawStatusStrip()
{
    const float gpuFrameMs = gpuFrameTimeHistory_.empty() ? 0.0f : gpuFrameTimeHistory_.latest();
    if (gpuFrameMs > 0.0f) {
        ImGui::Text("GPU %.2f ms (%.0f fps)", static_cast<double>(gpuFrameMs), 1000.0 / gpuFrameMs);
    } else {
        ImGui::TextDisabled("GPU -- ms");
    }
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("CPU %.2f ms", static_cast<double>(cpuFrameDeltaMs_));

    const VkExtent2D renderExtent = renderResolution_.extent();
    const VkExtent2D outputExtent = renderResolution_.outputExtent();
    if (renderResolution_.isNative()) {
        ImGui::Text("%u x %u native", renderExtent.width, renderExtent.height);
    } else {
        ImGui::Text("%u x %u -> %u x %u (%.0f%%)",
                    renderExtent.width,
                    renderExtent.height,
                    outputExtent.width,
                    outputExtent.height,
                    static_cast<double>(renderResolution_.scale()) * 100.0);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("%zu/%zu draws", cullingStats_.visibleObjects, cullingStats_.totalObjects);

    ImGui::Separator();
}

void Renderer::drawRenderScaleDebugUi()
{
    if (!ImGui::CollapsingHeader("Render Scale", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    const VkExtent2D renderExtent = renderResolution_.extent();
    const VkExtent2D outputExtent = renderResolution_.outputExtent();
    ImGui::Text("Rendering %u x %u -> presenting %u x %u",
                renderExtent.width,
                renderExtent.height,
                outputExtent.width,
                outputExtent.height);
    const float pixelRatio = renderResolution_.scale() * renderResolution_.scale();
    ImGui::Text("Shaded pixels: %.0f%% of native", static_cast<double>(pixelRatio) * 100.0);

    // The slider edits a pending value and commits only when the drag ends.
    // Committing waits for the device to go idle and rebuilds every screen-sized
    // target, which is not something to do once per dragged frame.
    ImGui::BeginDisabled(dynamicResolutionSettings_.enabled);
    ImGui::SliderFloat(
        "Scale", &pendingRenderScale_, renderer::kMinRenderScale, renderer::kMaxRenderScale, "%.2f");
    const bool sliderActive = ImGui::IsItemActive();
    const bool sliderCommitted = ImGui::IsItemDeactivatedAfterEdit();
    if (sliderCommitted) {
        renderScaleSettings_.scale = pendingRenderScale_;
        clampRuntimeSettings();
    } else if (!sliderActive) {
        pendingRenderScale_ = renderScaleSettings_.scale;
    }

    const std::array<std::pair<const char*, float>, 4> presets{
        std::pair{"100%", 1.0f},
        std::pair{"75%", 0.75f},
        std::pair{"50%", 0.5f},
        std::pair{"33%", 1.0f / 3.0f},
    };
    for (size_t index = 0; index < presets.size(); ++index) {
        if (index > 0) {
            ImGui::SameLine();
        }
        if (ImGui::Button(presets[index].first)) {
            renderScaleSettings_.scale = presets[index].second;
            pendingRenderScale_ = renderScaleSettings_.scale;
            clampRuntimeSettings();
        }
    }

    ImGui::EndDisabled();

    if (ImGui::SliderFloat("Sharpness", &renderScaleSettings_.sharpness, 0.0f, 1.0f, "%.2f")) {
        clampRuntimeSettings();
    }
    ImGui::SetItemTooltip("Contrast-adaptive sharpening in the composite, applied to the tone-mapped\n"
                          "image with an anti-ringing clamp. Only runs when the frame is upscaled --\n"
                          "at scale 1.00 it is skipped entirely, whatever this says.");
    if (renderResolution_.isNative() && renderScaleSettings_.sharpness > 0.0f) {
        ImGui::SameLine();
        ImGui::TextDisabled("(inactive at 1.00)");
    }

    ImGui::Checkbox("Show sharpen delta", &showSharpenDelta_);
    ImGui::SetItemTooltip("Replaces the image with |sharpened - original|, amplified.\n"
                          "Black means the filter changed nothing there. The correction is a rim\n"
                          "one source texel wide, so a close-up of a smooth surface is exactly\n"
                          "where it has least to do -- judge it at normal viewing distance,\n"
                          "on a lit region with detail near the render-resolution scale.");
    if (showSharpenDelta_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        ImGui::SliderFloat("Gain##sharpenDelta", &sharpenDeltaGain_, 1.0f, 64.0f, "%.0fx");
    }

    ImGui::TextDisabled("Shades the scene at a fraction of the window and upscales in the composite.");
    ImGui::TextDisabled("The frame is fragment-bound, so cost tracks the shaded-pixel count almost");
    ImGui::TextDisabled("linearly. The ImGui overlay stays native. Pairs with TAA, which recovers");
    ImGui::TextDisabled("some of the lost detail across frames.");

    ImGui::SeparatorText("Dynamic resolution");

    ImGui::Checkbox("Enabled##dynres", &dynamicResolutionSettings_.enabled);
    ImGui::SetItemTooltip("Drives the scale above from measured GPU frame time.\n"
                          "Off by default: a resolution that moves under you invalidates\n"
                          "any measurement you are trying to take.");

    // Presented as an FPS target because that is how anyone thinks about it,
    // while the stored value is the millisecond budget the controller compares
    // against.
    float targetFps = 1000.0f / std::max(dynamicResolutionSettings_.targetFrameMs, 0.001f);
    if (ImGui::DragFloat("Target FPS", &targetFps, 1.0f, 10.0f, 240.0f, "%.0f")) {
        dynamicResolutionSettings_.targetFrameMs = 1000.0f / std::max(targetFps, 1.0f);
        clampRuntimeSettings();
    }
    ImGui::Text("Target GPU frame budget: %.2f ms", dynamicResolutionSettings_.targetFrameMs);

    if (ImGui::SliderFloat("Min scale",
                           &dynamicResolutionSettings_.minScale,
                           renderer::kMinRenderScale,
                           renderer::kMaxRenderScale,
                           "%.2f")) {
        clampRuntimeSettings();
    }
    if (ImGui::SliderFloat("Max scale",
                           &dynamicResolutionSettings_.maxScale,
                           renderer::kMinRenderScale,
                           renderer::kMaxRenderScale,
                           "%.2f")) {
        clampRuntimeSettings();
    }

    if (dynamicResolution_.hasMeasurement()) {
        ImGui::Text("Median GPU frame: %.2f ms (of %u samples)",
                    static_cast<double>(dynamicResolution_.medianGpuFrameMs()),
                    renderer::kDynamicResolutionSampleWindow);
        ImGui::SetItemTooltip("Median, not average: one hitch must not drop the resolution.");
    } else {
        ImGui::TextDisabled("Median GPU frame: waiting for a timestamp readback");
    }
    ImGui::Text("Scale changes: %u", dynamicResolution_.changeCount());
    if (dynamicResolutionSettings_.enabled &&
        dynamicResolution_.measurementsSinceChange() < renderer::kDynamicResolutionSettleFrames) {
        // Says "settling" rather than looking stuck: the controller ignores this
        // window because the GPU timestamps still describe the old resolution.
        ImGui::Text("Settling: %u / %u measurements",
                    dynamicResolution_.measurementsSinceChange(),
                    renderer::kDynamicResolutionSettleFrames);
    }
    ImGui::Text("Last apply cost: %.2f ms (CPU, one frame)", static_cast<double>(lastRenderScaleApplyMs_));
    ImGui::SetItemTooltip("Applying a scale idles the device and rebuilds every screen-sized\n"
                          "target. That is why the controller quantises to 0.05 steps, keeps a\n"
                          "deadband, and ignores several frames after each change.");
}

void Renderer::drawToneMappingDebugUi()
{
    if (!ImGui::CollapsingHeader("Tone Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    const char* toneMappers[] = {"Reinhard", "ACES"};
    ImGui::Combo("Tone mapper", &toneMappingSettings_.operatorType, toneMappers, IM_ARRAYSIZE(toneMappers));
    ImGui::DragFloat("Manual exposure", &toneMappingSettings_.manualExposure, 0.01f, 0.0f, 64.0f, "%.3f");

    const char* exposureModes[] = {"Manual", "Log-average", "Histogram"};
    int exposureMode = toneMappingSettings_.enableAutoExposure ? toneMappingSettings_.exposureMode : 0;
    exposureMode = static_cast<int>(exposureModeValue(exposureMode));
    if (ImGui::Combo("Exposure mode", &exposureMode, exposureModes, IM_ARRAYSIZE(exposureModes))) {
        toneMappingSettings_.exposureMode = exposureMode;
        toneMappingSettings_.enableAutoExposure = exposureMode != 0;
    }

    ImGui::DragFloat("Target luminance", &toneMappingSettings_.targetLuminance, 0.001f, 0.001f, 8.0f, "%.3f");
    ImGui::DragFloat("Min exposure", &toneMappingSettings_.minExposure, 0.01f, 0.0f, 64.0f, "%.3f");
    ImGui::DragFloat("Max exposure", &toneMappingSettings_.maxExposure, 0.01f, 0.0f, 64.0f, "%.3f");
    ImGui::DragFloat("Adaptation rate", &toneMappingSettings_.adaptationRate, 0.01f, 0.0f, 16.0f, "%.3f");
    ImGui::SliderFloat("Histogram low percentile", &toneMappingSettings_.lowPercentile, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Histogram high percentile", &toneMappingSettings_.highPercentile, 0.0f, 1.0f, "%.3f");
}

void Renderer::drawBloomDebugUi()
{
    if (!ImGui::CollapsingHeader("Bloom", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::Checkbox("Enabled##bloom", &bloomSettings_.enabled);
    ImGui::Checkbox("Use mip-chain bloom", &bloomSettings_.useMipChain);
    ImGui::Text("Method: %s", bloomSettings_.useMipChain ? "Mip-chain" : "Legacy separable blur");
    ImGui::Text("Mip count: %zu", postProcess_.bloomMipDownsampleImages().size());
    ImGui::DragFloat("Threshold", &bloomSettings_.threshold, 0.01f, 0.0f, 32.0f, "%.3f");
    ImGui::DragFloat("Intensity##bloom", &bloomSettings_.intensity, 0.01f, 0.0f, 8.0f, "%.3f");
    ImGui::DragFloat("Radius", &bloomSettings_.radius, 0.01f, 0.25f, 4.0f, "%.2f");
}

void Renderer::drawSsaoDebugUi()
{
    if (!ImGui::CollapsingHeader("Ambient Occlusion (GTAO)", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    if (!ssaoAvailable_) {
        ImGui::TextDisabled("Unavailable: the depth format cannot be sampled on this device.");
    }
    ImGui::BeginDisabled(!ssaoAvailable_);
    ImGui::Checkbox("Enabled##ssao", &ssaoSettings_.enabled);
    ImGui::DragFloat("Radius (view units)", &ssaoSettings_.radius, 0.01f, 0.05f, 5.0f, "%.3f");
    ImGui::DragFloat("Intensity##ssao", &ssaoSettings_.intensity, 0.05f, 0.0f, 4.0f, "%.2f");
    ImGui::DragFloat("Power", &ssaoSettings_.power, 0.05f, 0.1f, 8.0f, "%.2f");
    ImGui::SliderInt("Slices", &ssaoSettings_.sliceCount, 1, 8);
    ImGui::SliderInt("Steps / slice", &ssaoSettings_.stepsPerSlice, 2, 16);
    ImGui::DragFloat("Falloff", &ssaoSettings_.falloff, 0.01f, 0.05f, 1.0f, "%.2f");
    ImGui::Checkbox("Ambient only", &ssaoSettings_.ambientOnly);
    ImGui::SetItemTooltip("On (default): the main pass applies occlusion to the ambient/indirect term only,\n"
                          "sampling the previous frame's AO reprojected along the motion vector.\n"
                          "Off: the composite multiplies the whole scene colour by it, which also darkens\n"
                          "direct lighting. Kept for A/B comparison -- a crease in full sunlight going dark\n"
                          "is the artefact the default avoids.");
    ImGui::TextWrapped(
        "Ground-truth ambient occlusion: a horizon-search pass reads the main depth buffer and the thin "
        "G-buffer normal, and the main pass multiplies the visibility term into the ambient term.");
    ImGui::EndDisabled();
}

void Renderer::drawVolumetricFogDebugUi()
{
    if (!ImGui::CollapsingHeader("Volumetric Fog", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (!volumetricFog_.available()) {
        ImGui::TextDisabled("Volumetric fog unavailable; see the startup log.");
        return;
    }

    ImGui::Checkbox("Enable volumetric fog", &fogSettings_.enabled);
    ImGui::SetItemTooltip("Off by default: fog changes the look of every shot, so it is opt-in.");
    ImGui::BeginDisabled(!fogSettings_.enabled);
    ImGui::SliderFloat("Density", &fogSettings_.density, 0.0f, 0.3f, "%.3f");
    ImGui::SliderFloat("Max distance", &fogSettings_.maxDistance, 8.0f, 200.0f, "%.0f");
    ImGui::SetItemTooltip("Fog volume depth range. Shorter puts the 64 slices where the fog is visible.");
    ImGui::SliderFloat("Anisotropy", &fogSettings_.anisotropy, -0.9f, 0.9f, "%.2f");
    ImGui::SetItemTooltip("Positive scatters light forward, which is what makes a shaft\n"
                          "bloom when you look toward the light through it.");

    ImGui::SliderFloat("Light cull threshold", &fogSettings_.lightCullThreshold, 0.0f, 1.0f, "%.3f");
    ImGui::SetItemTooltip("Skips a light for a froxel when a conservative upper bound on what it\n"
                          "could contribute falls below this. 0 disables the cull and is the\n"
                          "reference to compare against. Raise it while watching the fog near\n"
                          "the edge of a light's range -- that is where it starts to show.");

    ImGui::Checkbox("Temporal filtering", &fogSettings_.temporalEnabled);
    ImGui::SetItemTooltip("Jitters the froxel sample each frame and blends against the\n"
                          "reprojected previous volume. Off shows the raw single-sample\n"
                          "result, which is what the filtering is judged against.");
    ImGui::BeginDisabled(!fogSettings_.temporalEnabled);
    ImGui::SliderFloat("History blend", &fogSettings_.temporalBlend, 0.0f, 0.98f, "%.2f");
    ImGui::SetItemTooltip("Fraction of the previous frame kept. Higher is smoother but\n"
                          "smears more when lights or the camera move.");
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Forward scattering. Positive gives the halo when looking toward a light.");
    ImGui::SliderFloat("Height falloff", &fogSettings_.heightFalloff, 0.0f, 0.5f, "%.3f");
    ImGui::SetItemTooltip("Zero is uniform density; larger values pull the fog into a ground layer.");
    ImGui::SliderFloat("Base height", &fogSettings_.baseHeight, -10.0f, 20.0f, "%.1f");
    ImGui::SliderFloat("Ambient scale", &fogSettings_.ambientScale, 0.0f, 3.0f, "%.2f");
    ImGui::ColorEdit3("Scattering color", fogSettings_.scatteringColor);
    ImGui::EndDisabled();

    ImGui::TextDisabled("Volume: %ux%ux%u froxels, exponential Z from %.1f.",
                        renderer::kFogGridX,
                        renderer::kFogGridY,
                        renderer::kFogGridZ,
                        renderer::kFogNearPlane);
    ImGui::TextDisabled("Directional light only; punctual light shafts are not wired up yet.");
}



void Renderer::drawIrradianceProbesDebugUi()
{
    if (!ImGui::CollapsingHeader("Global Illumination (Irradiance Probes)", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (!irradianceProbes_.available()) {
        ImGui::TextDisabled("Irradiance probes unavailable; see the startup log.");
        return;
    }

    ImGui::Checkbox("Enable irradiance probes", &giSettings_.enabled);
    ImGui::SetItemTooltip("Off by default. The atlases are still seeded once at startup so nothing\n"
                          "ever samples them out of an undefined layout; this controls whether\n"
                          "probes are captured and convolved each frame.");

    ImGui::BeginDisabled(!giSettings_.enabled);
    ImGui::SliderInt("Probes per frame",
                     &giSettings_.probesPerFrame,
                     0,
                     static_cast<int>(renderer::kMaxProbesPerFrame));
    ImGui::SetItemTooltip("Round robin over the grid. This is the whole cost control: raising it\n"
                          "makes the grid catch up with a lighting change sooner and makes every\n"
                          "frame more expensive. Zero pauses capture without losing the cursor.");
    ImGui::EndDisabled();

    ImGui::SliderFloat("Intensity##gi", &giSettings_.intensity, 0.0f, 4.0f, "%.2f");
    ImGui::SetItemTooltip("How strongly probe irradiance replaces the constant environment term.\n"
                          "Zero disables the lookup entirely rather than scaling it to nothing.");
    ImGui::SliderFloat("Surface bias", &giSettings_.surfaceBias, 0.0f, 4.0f, "%.2f");
    ImGui::SetItemTooltip("How far off a surface the grid is sampled from, in world units.\n"
                          "Too small and flat surfaces self-occlude into darkness; too large\n"
                          "and light leaks through thin geometry.");
    ImGui::SliderFloat("Hysteresis", &giSettings_.hysteresis, 0.0f, 0.99f, "%.2f");
    ImGui::SetItemTooltip("How much of a probe's previous value survives a re-capture.\n"
                          "Zero overwrites, which is the reference to compare against;\n"
                          "higher smooths the step when lighting moves and averages the\n"
                          "sub-texel capture jitter into extra angular detail, at the cost\n"
                          "of taking longer to catch up.");

    ImGui::SliderFloat("Bounce weight", &giSettings_.bounceWeight, 0.0f, 0.95f, "%.2f");
    ImGui::SetItemTooltip("Multi-bounce. How much of a captured surface's indirect light comes\n"
                          "from the grid rather than the constant ambient, so the next capture\n"
                          "sees light that has bounced once more. Zero is single bounce, and is\n"
                          "the reference to compare against.\n"
                          "This is a feedback loop: each round multiplies by albedo * weight.");
    // The number that says whether a chosen weight is safe, rather than leaving
    // the reader to work out that a feedback loop is what they just enabled.
    ImGui::TextDisabled("  Steady-state gain on an albedo-0.8 surface: %.2fx.",
                        static_cast<double>(renderer::probeBounceAmplification(0.8f, giSettings_.bounceWeight)));

    ImGui::Checkbox("Debug: probe irradiance only", &giSettings_.debugIrradianceOnly);
    ImGui::SetItemTooltip("Outputs the gathered indirect term on its own, bypassing exposure\n"
                          "and tone mapping -- auto-exposure would otherwise cancel exactly the\n"
                          "brightness change this view exists to show.\n"
                          "Shown before Intensity is applied, so that slider keeps its real\n"
                          "meaning instead of doubling as a brightness knob; use Debug gain.\n"
                          "The background is the skybox, not a probe value.");

    if (ImGui::Checkbox("Debug pattern", &giSettings_.debugPattern)) {
        // Updates are off by default, so without this the atlases would keep
        // whatever the last update wrote and the toggle would look inert.
        irradianceProbes_.markDirty();
    }
    ImGui::SetItemTooltip("On: each tile holds the direction its texels stand for, which is what\n"
                          "makes the tile addressing and the octahedral border visible.\n"
                          "Off: the neutral state -- no irradiance, maximum distance -- that an\n"
                          "atlas which has captured nothing should hold.");

    ImGui::DragFloat3("Grid origin", giSettings_.gridOrigin, 0.1f);
    ImGui::SetItemTooltip("World position of probe (0, 0, 0).");
    ImGui::DragFloat3("Grid spacing", giSettings_.gridSpacing, 0.05f, 0.05f, renderer::kProbeMaxDistance);
    ImGui::SetItemTooltip("Distance between adjacent probes on each axis.");

    const renderer::ProbeGridBounds bounds = giGridBounds();
    const glm::vec3 extent{bounds.spacing.x * static_cast<float>(renderer::kProbeGridX - 1),
                           bounds.spacing.y * static_cast<float>(renderer::kProbeGridY - 1),
                           bounds.spacing.z * static_cast<float>(renderer::kProbeGridZ - 1)};
    ImGui::TextDisabled("Grid: %ux%ux%u probes covering %.1f x %.1f x %.1f world units.",
                        renderer::kProbeGridX,
                        renderer::kProbeGridY,
                        renderer::kProbeGridZ,
                        extent.x,
                        extent.y,
                        extent.z);
    ImGui::TextDisabled("Tiles: %u core + %u border texels (irradiance), %u + %u (depth).",
                        renderer::kProbeIrradianceResolution,
                        renderer::kProbeBorderTexels,
                        renderer::kProbeDepthResolution,
                        renderer::kProbeBorderTexels);
    if (!irradianceProbes_.convolveAvailable() || probeCapturePipeline_.pipeline() == VK_NULL_HANDLE) {
        ImGui::TextDisabled("Capture unavailable; probes hold the debug pattern. See the startup log.");
    } else {
        const int perFrame = std::max(giSettings_.probesPerFrame, 1);
        ImGui::TextDisabled("Capture: %u faces per probe at %ux%u, %d probes per frame -> full grid every "
                            "%d frames.",
                            renderer::kProbeCaptureFaceCount,
                            renderer::kProbeCaptureFaceResolution,
                            renderer::kProbeCaptureFaceResolution,
                            giSettings_.probesPerFrame,
                            (static_cast<int>(renderer::kProbeCount) + perFrame - 1) / perFrame);
        // Zero draws with a non-empty batch is the useful diagnostic: it
        // separates "probes captured nothing" from "probes were never
        // captured", which look identical in the atlas.
        ImGui::TextDisabled("Capture CPU: %.0f us to cull and record.", probeCaptureCpuMicroseconds_);
        ImGui::SetItemTooltip("The probe GPU passes are timestamp-queried over shaders that are\n"
                              "byte-identical in every build, so this is the one probe cost a Debug\n"
                              "build actually misrepresents.");
        ImGui::TextDisabled("Capture draws last frame: %u. Cursor at probe %u of %u; %llu captured so far.",
                            probeCaptureDrawsRecorded_,
                            irradianceProbes_.updateCursor(),
                            renderer::kProbeCount,
                            static_cast<unsigned long long>(irradianceProbes_.capturedProbeCount()));
        // Hysteresis is forced to zero until every probe has been captured once,
        // because a probe blending against its neutral seed would stay
        // permanently dark. Worth showing rather than leaving the slider looking
        // inert for the first cycle.
        const glm::vec2 jitter = irradianceProbes_.captureJitter();
        ImGui::TextDisabled("Accumulating: %s. Capture jitter (%.3f, %.3f) texels.",
                            irradianceProbes_.gridConverged() ? "yes" : "no, first cycle still filling",
                            jitter.x,
                            jitter.y);
    }


    ImGui::SeparatorText("Atlas previews");

    // Read the tile edges, not the overall colour. With the debug pattern the
    // octahedral border is correct exactly when neighbouring tiles meet without
    // a visible seam of their own -- a wrong border draws a one-texel frame
    // around every tile, which is the single most legible check available before
    // anything captures real radiance.
    const float previewSize = 320.0f * std::clamp(debugUiSettings_.renderTargetPreviewScale, 0.25f, 2.0f);

    // Gathered irradiance in this scene measures roughly 0.05 to 0.25, so shown
    // at 1:1 the whole atlas lands in the bottom quarter of the display range
    // and reads as black however correct it is. Captured radiance is not a
    // display value and there is no exposure applied to it here, so the preview
    // needs its own gain the way the HDR render targets do.
    ImGui::SliderFloat("Debug gain", &giSettings_.previewGain, 1.0f, 16.0f, "%.1fx");
    ImGui::SetItemTooltip("Scales the atlas previews and the probe-only view. Probe values are\n"
                          "linear radiance, not display colour, and sit far below 1.0 in an\n"
                          "ordinary scene, so at 1:1 a correct atlas reads as black.");

    ImGui::TextDisabled("Irradiance %ux%u (%u x %u tiles), one tile per probe.",
                        renderer::kProbeIrradianceAtlasWidth,
                        renderer::kProbeIrradianceAtlasHeight,
                        renderer::kProbeAtlasTilesX,
                        renderer::kProbeAtlasTilesY);
    ImGui::TextDisabled("Columns are (x, y) with x fastest, rows are z: the leftmost 8 columns are the "
                        "lowest layer.");
    drawRenderTargetPreview(irradianceProbes_.irradianceAtlas().imageView(),
                            irradianceProbes_.sampler(),
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            renderer::kProbeIrradianceAtlasWidth,
                            renderer::kProbeIrradianceAtlasHeight,
                            previewSize,
                            giSettings_.previewGain);

    // Red is mean distance over kProbeMaxDistance. Green is the *squared* mean
    // over the same scale, so it saturates for anything past 8 units and is not
    // worth reading -- the tint is one multiplier for every channel, and the two
    // moments are orders of magnitude apart by construction. Saying so beats
    // leaving a channel that always looks blown out.
    ImGui::TextDisabled("Depth %ux%u. Red = mean distance / %.0f; green is its square and saturates.",
                        renderer::kProbeDepthAtlasWidth,
                        renderer::kProbeDepthAtlasHeight,
                        renderer::kProbeMaxDistance);
    ImGui::TextDisabled("Saturated yellow means nothing was hit in that direction, which is expected for "
                        "probes sitting in open air.");
    drawRenderTargetPreview(irradianceProbes_.depthAtlas().imageView(),
                            irradianceProbes_.sampler(),
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            renderer::kProbeDepthAtlasWidth,
                            renderer::kProbeDepthAtlasHeight,
                            previewSize,
                            1.0f / renderer::kProbeMaxDistance);
}

void Renderer::drawShadowsDebugUi()
{
    // Both shadow sources live here rather than splitting the directional ones
    // out under CSM and the punctual ones under Lights: they are tuned against
    // each other, and the atlas preview is the main diagnostic for either.
    if (!ImGui::CollapsingHeader("Shadows", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    ImGui::SeparatorText("Cascaded (directional)");
    int cascadeCount = static_cast<int>(activeCascadeCount());
    ImGui::BeginDisabled();
    ImGui::SliderInt("Cascade count (startup)", &cascadeCount, 1, static_cast<int>(kMaxShadowCascades));
    ImGui::EndDisabled();
    ImGui::SliderFloat("Lambda", &csmSettings_.lambda, 0.0f, 1.0f, "%.3f");
    ImGui::DragFloat("Shadow distance", &csmSettings_.shadowDistance, 0.1f, 1.0f, csmSettings_.farPlane, "%.2f");
    ImGui::Checkbox("Texel snapping enabled", &csmSettings_.enableTexelSnapping);
    ImGui::Checkbox("Cascade debug colors enabled", &csmSettings_.enableCascadeDebugColors);
    ImGui::SliderFloat("Normal-offset bias", &csmSettings_.normalBias, 0.0f, 0.02f, "%.4f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Offsets the shadow lookup along the surface normal, as a fraction of\n"
                          "each cascade's own extent. Lets the depth bias come down, which is\n"
                          "what causes peter-panning. 0 = depth bias only.");
    }
    ImGui::SliderFloat("Cascade blend band", &csmSettings_.cascadeBlend, 0.0f, 0.5f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cross-fades each cascade into the next over this fraction of its\n"
                          "depth range. 0 = hard split. Only fragments inside the band pay\n"
                          "for the second shadow lookup.");
    }

    ImGui::Checkbox("Stable (sphere) cascade fit", &csmSettings_.enableStableCascadeFit);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fits each cascade to its slice's bounding sphere rather than the\n"
                          "slice's light-space AABB, so the ortho extent stops breathing as\n"
                          "the camera turns. Removes rotation shimmer; costs sharpness, since\n"
                          "a sphere around the slice is wider than the slice.\n\n"
                          "It does NOT help the cascade cache below: the slice centre orbits\n"
                          "the camera, so any visible rotation still moves it many texels.");
    }

    ImGui::Checkbox("Cascade caching enabled", &csmSettings_.enableCascadeCache);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Redraws a cascade only when a hash of what it draws moves: the\n"
                          "cascade's fitted matrix, the raster depth bias, and every caster's\n"
                          "geometry, LOD level, transform and alpha cutoff. Off redraws all\n"
                          "cascades unconditionally, which is the A/B reference the cached\n"
                          "path must match pixel for pixel.");
    }

    // The hit rate, not just the on/off state. The cascades are fitted to the
    // camera, so this number is the thing to read before assuming the cache
    // ever fires: a fit that moves with the camera dirties every cascade on any
    // camera motion, and only a measurement says how often that is.
    {
        const uint32_t activeCascades = activeCascadeCount();
        ImGui::Text("Cascades: %s, %u/%u redrawn (%u frames fully cached)",
                    cascadeShadowCacheHit_ ? "reused" : (cascadeShadowCascadesRedrawn_ >= activeCascades ? "all redrawn" : "partial"),
                    cascadeShadowCascadesRedrawn_,
                    activeCascades,
                    cascadeShadowCachedFrames_);
        ImGui::SetItemTooltip("A fully cached frame skips the pass outright: no clear, no draws,\n"
                              "and no caster cull dispatch. Per cascade rather than per pass,\n"
                              "because a caster moving inside cascade 0 says nothing about the rest.");

        std::string cascadeStates;
        for (uint32_t cascadeIndex = 0; cascadeIndex < activeCascades; ++cascadeIndex) {
            if (cascadeIndex > 0) {
                cascadeStates += "  ";
            }
            cascadeStates += std::to_string(cascadeIndex);
            cascadeStates += ": ";
            cascadeStates += cascadeShadowDirty_[cascadeIndex] ? "redraw" : "cached";
        }
        ImGui::TextUnformatted(cascadeStates.c_str());
    }

    ImGui::SeparatorText("Punctual (spot/point)");
    if (!punctualShadows_.valid()) {
        ImGui::TextDisabled("Shadow atlas unavailable; point/spot lights do not cast.");
    } else {
        ImGui::Checkbox("Cast punctual shadows", &usePunctualShadows_);
        ImGui::SetItemTooltip("Spot lights render into a shared depth atlas and shadow the clustered pass.");
        ImGui::Text("Atlas: %ux%u, tiles %u-%upx",
                    renderer::kPunctualShadowAtlasSize,
                    renderer::kPunctualShadowAtlasSize,
                    renderer::kPunctualShadowMinTileSize,
                    renderer::kPunctualShadowMaxTileSize);
        // With mixed tile sizes a slot count says nothing about how full the
        // atlas is, so the occupancy fraction is the number that matters.
        ImGui::Text("Slots used: %u  (atlas %.0f%% full)",
                    punctualShadowSlotsUsed_,
                    punctualShadows_.occupancy() * 100.0f);
        ImGui::Text("Caster draws recorded: %u", punctualShadowDrawsRecorded_);
        // Assignment churn is what popping actually looks like, so it is
        // measured rather than inferred from the image.
        ImGui::Text("Assignment churn: %u this frame", punctualShadowAssignmentChurn_);
        ImGui::Text("Cull + record CPU: %lld us", punctualShadowCpuMicros_);
        ImGui::SetItemTooltip("What GPU caster culling would remove. Compare against the\n"
                              "PunctualShadowGpuCull and PunctualShadowAtlas rows in the\n"
                              "profiler before deciding the trade is worth it.");
        ImGui::Text("Atlas: %s, %u/%u tiles redrawn (%u frames fully cached)",
                    punctualShadowCacheHit_ ? "reused" : "partial",
                    punctualShadowSlotsRedrawn_,
                    punctualShadowSlotsUsed_,
                    punctualShadowCachedFrames_);
        ImGui::SetItemTooltip("The atlas is re-rendered only when a hash of its inputs moves:\n"
                              "light projections, tile rects, caster transforms and geometry,\n"
                              "and the raster depth bias. Static light over static geometry is free.");
        ImGui::SetItemTooltip("Lights that gained or lost their shadow since last frame.\n"
                              "Persistently non-zero is what reads as shadows popping in and out.");
        ImGui::SetItemTooltip("Zero here with slots > 0 means the atlas pass culled or skipped everything.");
        ImGui::BeginDisabled(!punctualShadows_.cullAvailable());
        ImGui::Checkbox("GPU caster culling", &useGpuPunctualShadowCulling_);
        ImGui::SetItemTooltip("Culls casters per slot in one compute dispatch instead of on the CPU.\n"
                              "Off by default: on this scene it is a net loss, trading ~40us of CPU\n"
                              "frustum tests for a dispatch and its barriers. It is here because the\n"
                              "CPU cost is O(slots x draw items) and this scene has 11 of the latter.");
        ImGui::EndDisabled();
        if (!punctualShadows_.cullAvailable()) {
            ImGui::TextDisabled("GPU caster culling unavailable; CPU frustum tests in use.");
        }

        ImGui::Checkbox("Debug: shadow term only", &showPunctualShadowDebug_);
        ImGui::SetItemTooltip("Greyscale punctual visibility, min across the lights that actually reach\n"
                              "each fragment. Flat white = the atlas is never sampled; any structure =\n"
                              "the lookup works. Several casters overlapping darkens a lot of the frame\n"
                              "here without the shaded image changing nearly as much.");
        // Six tiles per point light against 64 total, so the budget is explicit
        // rather than an implicit cap the user cannot see.
        ImGui::SliderInt("Max shadowed point lights",
                         &maxShadowCastingPointLights_,
                         0,
                         16);
        ImGui::SetItemTooltip("Each point light costs 6 tiles (one per cube face).\n"
                              "Nearest to the camera are served first.");

        // Looking at the atlas directly is the only reliable way to tell a
        // wrong projection from a wrong sample: in the beauty shot an overhead
        // spot is easily washed out by the directional key light.
        if (ImGui::TreeNodeEx("Atlas depth preview", ImGuiTreeNodeFlags_DefaultOpen)) {
            const VkExtent2D atlasExtent = punctualShadows_.atlas().extent();
            ImGui::TextDisabled("Tiles are %u-%upx, sized by each light's projected radius.",
                                renderer::kPunctualShadowMinTileSize,
                                renderer::kPunctualShadowMaxTileSize);
            // The preview is a plain ImGui::Image with a multiplicative tint, so
            // it cannot expand contrast near 1.0. Perspective depth sits up
            // there almost everywhere, which makes even a correctly rendered
            // tile read as solid red. The CSM cascade previews look fine only
            // because an orthographic projection gives linear depth. Use the
            // caster-draw count above to tell "empty" from "just compressed".
            ImGui::TextDisabled("Perspective depth reads near-white/red even when correct;");
            ImGui::TextDisabled("trust \"caster draws recorded\" above, not this image.");
            drawRenderTargetPreview(punctualShadows_.atlas().imageView(),
                                    punctualShadows_.atlas().sampler(),
                                    VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                                    atlasExtent.width,
                                    atlasExtent.height,
                                    160.0f * std::clamp(debugUiSettings_.renderTargetPreviewScale, 0.25f, 2.0f),
                                    1.0f);
            ImGui::TreePop();
        }

        float constantBias = punctualShadows_.constantBias();
        float normalBias = punctualShadows_.normalBias();
        bool biasChanged = ImGui::SliderFloat("Depth bias", &constantBias, 0.0f, 0.01f, "%.4f");
        biasChanged |= ImGui::SliderFloat("Normal bias", &normalBias, 0.0f, 0.2f, "%.3f");
        if (biasChanged) {
            punctualShadows_.setDepthBias(constantBias, normalBias);
        }
        ImGui::SetItemTooltip("Raise until acne disappears; too much detaches shadows from their casters.");
    }
}

void Renderer::drawLightsDebugUi()
{
    if (!ImGui::CollapsingHeader("Lights (Clustered)", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    const bool clusteredAvailable = clusteredLighting_.available();
    ImGui::Text("Clustered path: %s", clusteredAvailable ? "available" : "unavailable (brute force)");
    ImGui::Text("Froxel grid: %ux%ux%u (%u clusters)",
                renderer::kClusterGridX,
                renderer::kClusterGridY,
                renderer::kClusterGridZ,
                renderer::kClusterCount);
    ImGui::Text("Active lights: %u", clusteredLighting_.lightCount());

    ImGui::BeginDisabled(!clusteredAvailable);
    ImGui::Checkbox("Use clustered culling", &useClusteredLighting_);
    ImGui::SameLine();
    ImGui::Checkbox("Cluster heatmap", &showClusterHeatmap_);
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!asyncCompute_.available());
    ImGui::Checkbox("Async compute (cluster build + light cull)", &useAsyncCompute_);
    ImGui::EndDisabled();
    if (!asyncCompute_.available()) {
        ImGui::TextDisabled("Async compute queue unavailable; cluster passes run on the graphics queue.");
        ImGui::TextDisabled("(MoltenVK: set MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1 to expose one.)");
    } else {
        ImGui::Text("Async compute: %s", frameAsyncComputeActive_ ? "active" : "inactive");
        if (frameAsyncComputeActive_) {
            ImGui::TextDisabled("ClusterBuild/LightCull run on the compute queue; their GPU profiler rows are "
                                "not captured there.");
        }
    }
    if (!useClusteredLighting_) {
        showClusterHeatmap_ = false;
    }
    ImGui::SetItemTooltip("Heatmap tints each froxel by its light count (blue=few, red=many).");

    ImGui::SeparatorText("Demo light swarm");
    ImGui::SliderInt("Light count", &demoLightCount_, 0, 512);
    ImGui::Checkbox("Animate##demoLights", &animateLights_);
    ImGui::SliderFloat("Intensity##demoLights", &demoLightIntensity_, 0.0f, 40.0f, "%.1f");
    ImGui::SliderFloat("Range", &demoLightRange_, 1.0f, 30.0f, "%.1f");
    ImGui::TextDisabled("Drive the count up to stress-test the clustered path.");
}

void Renderer::drawSkeletalAnimationDebugUi()
{
    if (!ImGui::CollapsingHeader("Skeletal Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (!skinnedMesh_.valid()) {
        ImGui::TextDisabled("Skinned mesh unavailable.");
        return;
    }

    ImGui::Text("Source: %s (%u joints, GPU linear-blend skinning)",
                skinnedMesh_.sourceName().c_str(),
                skinnedMesh_.jointCount());
    ImGui::Text("Animation: %s", skinnedMesh_.usesImportedClip() ? "imported glTF clip" : "procedural bend");
    ImGui::Checkbox("Show skinned mesh", &showSkinnedMesh_);
    ImGui::SameLine();
    ImGui::Checkbox("Animate##skinnedMesh", &animateSkinnedMesh_);
    ImGui::SliderFloat("Playback speed", &skinnedAnimationSpeed_, 0.0f, 4.0f, "%.2fx");
    if (ImGui::Button("Reset to bind pose")) {
        skinnedAnimationTime_ = 0.0f;
        animateSkinnedMesh_ = false;
    }
}

void Renderer::drawMeshLodDebugUi()
{
    if (!ImGui::CollapsingHeader("Mesh LOD", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    // Selection runs inside the GPU cull dispatch, so every control here is a
    // field of GpuCullFrameParams::lodSettings uploaded next frame -- nothing is
    // recomputed on the CPU.
    ImGui::Checkbox("LOD selection enabled", &lodSettings_.enabled);
    ImGui::SameLine();
    ImGui::TextDisabled("(off pins every draw to level 0)");

    ImGui::BeginDisabled(!lodSettings_.enabled);

    ImGui::DragFloat("Reference radius (px)", &lodSettings_.referenceRadiusPixels, 1.0f, 8.0f, 4096.0f, "%.0f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Projected sphere radius at which level 0 is still chosen.\n"
                          "Each halving of the on-screen radius steps one level down.");
    }
    ImGui::DragFloat("Bias", &lodSettings_.bias, 0.05f, -4.0f, 4.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Positive biases toward lower detail. One unit is one level.");
    }
    ImGui::DragFloat("Shadow bias", &lodSettings_.shadowBias, 0.05f, -4.0f, 4.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Added on top of Bias for shadow-cascade dispatches, which hide\n"
                          "simplification far better than the main pass.");
    }

    // -1 is the "select by distance" sentinel, so the combo is offset by one.
    static const char* kForcedLodLabels[] = {"Auto (by distance)", "Force 0", "Force 1", "Force 2", "Force 3"};
    int forcedLodChoice = std::clamp(lodSettings_.forcedLod + 1, 0, static_cast<int>(renderer::kMaxMeshLods));
    if (ImGui::Combo("Forced level", &forcedLodChoice, kForcedLodLabels, IM_ARRAYSIZE(kForcedLodLabels))) {
        lodSettings_.forcedLod = forcedLodChoice - 1;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Pins every draw item to one level. Clamped per mesh to the\n"
                          "levels its chain actually has, so low-poly meshes stay at 0.");
    }

    ImGui::Checkbox("Color by LOD", &lodSettings_.debugHeatmap);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Green -> yellow -> orange -> red as detail drops.\n"
                          "The level is packed into the high bits of gl_InstanceIndex\n"
                          "by the cull pass, since only it knows what was selected.");
    }

    ImGui::EndDisabled();

    if (ImGui::Button("Reset LOD settings")) {
        lodSettings_ = LodSettings{};
    }

    clampRuntimeSettings();

    ImGui::Separator();

    // Emitted-draw counts per level, read back from the cull stats block. Meshes
    // with no chain are not counted, so the total can be below the visible count.
    renderer::GpuCullCounters counters{};
    if (gpuCulling_.readMainCounters(isGpuCullingActive(), currentFrame_, counters)) {
        uint32_t countedDraws = 0;
        for (const uint32_t levelCount : counters.lodDrawItems) {
            countedDraws += levelCount;
        }

        ImGui::Text("Selected levels (%u draws with a chain):", countedDraws);
        for (size_t level = 0; level < counters.lodDrawItems.size(); ++level) {
            const float fraction =
                countedDraws > 0 ? static_cast<float>(counters.lodDrawItems[level]) / static_cast<float>(countedDraws)
                                 : 0.0f;
            const std::string overlay =
                "L" + std::to_string(level) + ": " + std::to_string(counters.lodDrawItems[level]);
            ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), overlay.c_str());
        }
    } else {
        ImGui::TextDisabled("Level distribution needs GPU culling (readback comes from the cull stats).");
    }

    // Chain lengths are a property of the loaded meshes, so report what actually
    // got built -- a mesh too small to simplify is a level-0-only chain, and that
    // is the usual reason a scene shows no level variety at all.
    ImGui::Separator();
    ImGui::Text("Mesh chains:");
    std::vector<const renderer::Mesh*> reportedMeshes;
    for (const renderer::RenderObject& object : renderObjects_) {
        const renderer::Mesh* mesh = object.mesh;
        if (!mesh || !mesh->valid()) {
            continue;
        }
        if (std::find(reportedMeshes.begin(), reportedMeshes.end(), mesh) != reportedMeshes.end()) {
            continue;
        }
        reportedMeshes.push_back(mesh);

        const std::span<const renderer::MeshLod> lods = mesh->lods();
        std::string summary;
        for (size_t level = 0; level < lods.size(); ++level) {
            summary += (level == 0 ? "" : " -> ") + std::to_string(lods[level].indexCount / 3) + "tri";
        }
        if (lods.size() <= 1) {
            summary += " (too small to simplify)";
        }
        ImGui::BulletText("%s: %s", mesh->debugName().c_str(), summary.c_str());
    }
}

void Renderer::drawGpuCullingDebugUi()
{
    if (!ImGui::CollapsingHeader("GPU Culling", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    // Completes the settings path for enableTransientAliasing: field, clamp, JSON
    // read, JSON write, control, consumer. Toggling it takes effect on the next
    // plan application rather than immediately, because the pool is rebuilt with
    // the post-process resources, so the state is spelled out rather than left
    // to look broken for a frame.
    if (ImGui::Checkbox("Transient memory aliasing (bloom)", &useTransientAliasing_)) {
        // Re-apply from scratch either way: turning it off has to hand the bloom
        // chain back its private allocations, not just stop planning.
        transientAliasingApplied_ = false;
        if (!useTransientAliasing_) {
            waitIdle();
            postProcess_.setBloomAliasPlan({}, 0, 0, 0);
            recreatePostProcessResources();
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Binds the bloom chain into one shared allocation: 17.48 MiB saved, and a\n"
                          "measured +0.176 ms (+1.2%%) of frame time, which is why it defaults off.\n"
                          "Applied after the next recorded frame, because the offsets come from\n"
                          "measured resource lifetimes.");
    }
    if (postProcess_.bloomImagesAreAliased()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(active, %.2f MiB pool)", static_cast<double>(postProcess_.bloomPoolBytes()) / (1024.0 * 1024.0));
    } else if (useTransientAliasing_) {
        ImGui::SameLine();
        ImGui::TextDisabled("(pending)");
    }

    if (!gpuCulling_.available()) {
        ImGui::BeginDisabled();
    }
    ImGui::Checkbox("Main GPU culling enabled", &useGpuCulling_);
    if (!gpuCulling_.available()) {
        ImGui::EndDisabled();
    }

    if (!gpuCulling_.shadowAvailable()) {
        ImGui::BeginDisabled();
    }
    ImGui::Checkbox("Shadow GPU culling enabled", &useGpuShadowCulling_);
    if (!gpuCulling_.shadowAvailable()) {
        ImGui::EndDisabled();
    }

    ImGui::Text("Bindless material textures: %s", isBindlessMaterialTextureActive() ? "active" : "fallback");
    bool bindlessEnabled = useBindlessMaterialTextures_;
    ImGui::BeginDisabled();
    ImGui::Checkbox("Bindless material textures enabled (startup)", &bindlessEnabled);
    ImGui::EndDisabled();
    ImGui::Text("Main indirect count path: %s",
                isFrameIndirectCountPathActive(currentFrame_) ? "active" : "fallback");
    ImGui::Text("Shadow indirect count path: %s",
                isShadowIndirectCountPathActive(currentFrame_) ? "active" : "fallback");
    if (ImGui::Button("Enable Occlusion Test Settings")) {
        enableOcclusionTestSettings();
    }
    const bool occlusionControlsAvailable =
        isGpuCullingActive() && depthPyramid_.buildAvailable() && depthPyramid_.image() != VK_NULL_HANDLE;
    if (!occlusionControlsAvailable) {
        ImGui::BeginDisabled();
    }
    ImGui::Checkbox("GPU occlusion culling enabled", &useGpuOcclusionCulling_);
    ImGui::Checkbox("Two-phase occlusion (Hi-Z re-test)", &useTwoPhaseOcclusion_);
    ImGui::SliderFloat("Occlusion depth bias", &gpuOcclusionDepthBias_, 0.0f, 0.05f, "%.4f");
    ImGui::SliderFloat("Near-object skip distance", &gpuOcclusionNearDisableDistance_, 0.0f, 10.0f, "%.2f");
    ImGui::SliderFloat("Max occlusion screen coverage", &gpuOcclusionMaxScreenCoverage_, 0.01f, 1.0f, "%.2f");
    ImGui::SliderFloat("Min occlusion size pixels", &gpuOcclusionMinScreenPixels_, 1.0f, 64.0f, "%.1f");
    if (!occlusionControlsAvailable) {
        ImGui::EndDisabled();
    }
    const char* depthPyramidStatus =
        depthPyramid_.buildAvailable() ? (depthPyramid_.valid() ? "valid" : "invalid/warming up") : "unavailable";
    ImGui::Text("Depth pyramid: %s, %u mip(s)", depthPyramidStatus, depthPyramid_.mipLevels());
}

void Renderer::drawEnvironmentDebugUi()
{
    if (!ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::Text("Environment source: %s", hdrEnvironmentLoaded_ ? "HDR environment loaded" : "procedural fallback");
    ImGui::Text("Tone mapping exposure: %.4f", postProcess_.currentToneMappingExposure());
}

void Renderer::drawScenePresetDebugUi()
{
    if (!ImGui::CollapsingHeader("Scene Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    size_t occlusionObjectCount = 0;
    for (const renderer::RenderObject& object : renderObjects_) {
        if (object.sourceType == renderer::RenderObjectSourceType::OcclusionTest) {
            ++occlusionObjectCount;
        }
    }

    const bool occlusionTestVisible = occlusionTestSceneActive_ && !portfolioCaptureMode_;
    ImGui::Text("Occlusion test scene: %s", occlusionTestVisible ? "active" : "inactive");
    ImGui::Text("Occlusion test objects: %zu", occlusionObjectCount);

    if (ImGui::Button("Load Occlusion Test Scene")) {
        loadOcclusionTestScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("Return to Default Scene")) {
        if (portfolioCaptureMode_) {
            setPortfolioCaptureMode(false);
        }
        occlusionTestSceneActive_ = false;
        resetCameraToDefault();
        occlusionTestSceneStatus_ = "Default scene active; occlusion test objects are hidden.";
    }

    if (ImGui::Button("Reset Occlusion Test Camera##culling")) {
        resetCameraToOcclusionTestPreset();
        occlusionTestSceneStatus_ = "Occlusion test camera preset reapplied.";
    }

    ImGui::TextWrapped("Status: %s", occlusionTestSceneStatus_.c_str());

    ImGui::SeparatorText("Stress scene");
    ImGui::TextWrapped("%d objects behind occluder slabs. The default scene is eleven, which is too few for "
                       "GPU culling, LOD selection, or the parallel frame-prep loops to do anything.",
                       renderer::kStressObjectCount);
    if (ImGui::Button("Load Stress Scene")) {
        loadStressScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("Exit Stress Scene")) {
        removeStressSceneObjects();
        stressSceneActive_ = false;
        resetCameraToDefault();
        stressSceneStatus_ = "Stress scene inactive; objects removed.";
    }
    ImGui::TextWrapped("Status: %s", stressSceneStatus_.c_str());

    ImGui::SeparatorText("Fragment stress scene");
    ImGui::TextWrapped("%d full-frame layers and %d dense point lights. The geometry stress scene above runs "
                       "faster than the default one because its objects are small on screen; this is the "
                       "opposite test, loading screen coverage and the per-fragment light loop.",
                       renderer::kFragmentStressLayerCount,
                       renderer::kFragmentStressLightCount);
    if (ImGui::Button("Load Fragment Stress")) {
        loadFragmentStressScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("Exit Fragment Stress")) {
        removeFragmentStressSceneObjects();
        fragmentStressSceneActive_ = false;
        resetCameraToDefault();
        fragmentStressSceneStatus_ = "Fragment stress inactive; objects removed.";
    }
    ImGui::TextWrapped("Status: %s", fragmentStressSceneStatus_.c_str());

    ImGui::SeparatorText("Cornell box");
    ImGui::TextDisabled("A closed, coloured room. The only scene here that shows indirect light:");
    ImGui::TextDisabled("colour bleeding needs saturated walls, and a second bounce needs");
    ImGui::TextDisabled("somewhere for light to be trapped. Loading it switches the sun off,");
    ImGui::TextDisabled("fits the probe grid to the interior, and turns probes on.");
    if (ImGui::Button("Load Cornell Box")) {
        loadCornellBoxScene();
    }
    ImGui::SameLine();
    if (ImGui::Button("Exit Cornell Box")) {
        cornellBoxSceneActive_ = false;
        resetCameraToDefault();
        resetDirectionalLightToDefault();
        cornellBoxSceneStatus_ = "Cornell box inactive; default scene and sun restored.";
    }
    ImGui::TextWrapped("Status: %s", cornellBoxSceneStatus_.c_str());
}

void Renderer::drawPortfolioCaptureDebugUi()
{
    if (!ImGui::CollapsingHeader("Portfolio Capture", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    bool portfolioMode = portfolioCaptureMode_;
    if (ImGui::Checkbox("Portfolio Capture Mode", &portfolioMode)) {
        setPortfolioCaptureMode(portfolioMode);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("F11");

    if (ImGui::Button("Load Portfolio Showcase Scene")) {
        setPortfolioCaptureMode(true);
        applyPortfolioCaptureSettings();
        screenshotCapture_.setStatus("Portfolio showcase scene preset is active.");
    }

    if (ImGui::Button("Capture Portfolio Screenshot")) {
        requestPortfolioScreenshot();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("F12");

    const std::filesystem::path outputDirectory = portfolioScreenshotDirectory();
    ImGui::TextWrapped("Output: %s", (outputDirectory / "vulkan_engine_portfolio_latest.png").string().c_str());
    if (!screenshotCapture_.lastSavedPath().empty()) {
        ImGui::TextWrapped("Last timestamped: %s", screenshotCapture_.lastSavedPath().string().c_str());
    }
    ImGui::TextWrapped("Status: %s", screenshotCapture_.status().c_str());
    ImGui::TextDisabled("Captures the final composite before ImGui, so debug UI is excluded from the PNG.");

    if (!swapchain_.supportsTransferSrc()) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f),
                           "Current swapchain does not support transfer-source screenshot capture.");
    }
}

void Renderer::drawRuntimeSettingsDebugUi()
{
    if (!ImGui::CollapsingHeader("Runtime Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    const std::string settingsPath = runtimeSettingsPath_.string();
    ImGui::TextWrapped("File: %s", settingsPath.c_str());

    if (ImGui::Button("Save Settings")) {
        saveRuntimeSettingsFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reload Settings")) {
        reloadRuntimeSettingsFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset to Defaults")) {
        resetRuntimeSettingsToDefaults();
    }

    ImGui::TextWrapped("Last load: %s", lastRuntimeSettingsLoadStatus_.c_str());
    ImGui::TextWrapped("Last save: %s", lastRuntimeSettingsSaveStatus_.c_str());
    if (!runtimeSettingsWarning_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f), "Warning: %s", runtimeSettingsWarning_.c_str());
    }
    ImGui::TextDisabled("Runtime-safe: tone mapping, exposure, bloom, TAA, CSM lambda/distance/stability/debug, "
                        "available GPU culling toggles, panel visibility, and render-target preview UI state.");
    ImGui::TextDisabled("Startup-applied: CSM cascade count, bindless material texture heap, and culling resources "
                        "that were disabled before initialization.");
}

void Renderer::drawTaaDebugUi()
{
    if (!ImGui::CollapsingHeader("Temporal AA", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    bool changed = false;
    changed |= ImGui::Checkbox("Enabled##taa", &taaSettings_.enabled);
    changed |= ImGui::Checkbox("Jitter enabled", &taaSettings_.jitterEnabled);
    changed |= ImGui::Checkbox("Neighborhood clamp", &taaSettings_.neighborhoodClampEnabled);
    changed |= ImGui::Checkbox("Motion reprojection", &taaSettings_.reprojectionEnabled);
    changed |= ImGui::SliderFloat("History feedback", &taaSettings_.feedback, 0.0f, 0.98f, "%.3f");

    ImGui::SeparatorText("Ghosting rejection");
    ImGui::TextDisabled("Both off by default. Stricter rejection always costs accumulated");
    ImGui::TextDisabled("history, so it is worth turning on only when ghosting is visible.");
    changed |= ImGui::Checkbox("Variance clipping", &taaSettings_.varianceClipping);
    ImGui::SetItemTooltip("Bounds the history by the neighbourhood's mean and standard deviation in\n"
                          "YCoCg instead of its RGB extremes. An extremes box is set by its two most\n"
                          "extreme samples, so one bright speck widens it enough for a ghost to sit\n"
                          "inside. Off restores the extremes box for comparison.");
    changed |= ImGui::SliderFloat("Variance gamma", &taaSettings_.varianceGamma, 0.25f, 3.0f, "%.2f");
    ImGui::SetItemTooltip("Half-width of that box in standard deviations. Lower rejects more history:\n"
                          "less ghosting, and less of the accumulated detail upsampling exists to gather.");
    changed |= ImGui::Checkbox("Rejection feedback", &taaSettings_.rejectionFeedback);
    changed |= ImGui::Checkbox("Catmull-Rom history", &taaSettings_.catmullRomHistory);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Resample history with Catmull-Rom instead of one bilinear tap.\n"
                          "Off reproduces the repeated-bilinear softening this replaces.");
    }
    ImGui::SetItemTooltip("Lowers a pixel's feedback by how far its history had to be pulled to become\n"
                          "acceptable. Clipping alone leaves a ghost at the nearest plausible colour and\n"
                          "still gives it most of the pixel; this is what actually removes it.");
    if (changed) {
        clampRuntimeSettings();
        invalidateTaaHistory();
    }

    if (ImGui::Button("Reset TAA History")) {
        invalidateTaaHistory();
    }

    ImGui::Text("Active: %s", postProcess_.isTaaActive() ? "yes" : "no");
    ImGui::Text("History valid: %s", postProcess_.taaHistoryValid() ? "yes" : "no");
    ImGui::Text("Read/Write: %u / %u", postProcess_.taaHistoryReadIndex(), postProcess_.taaHistoryWriteIndex());
    ImGui::Text("Jitter index: %u", postProcess_.taaJitterIndex());
    ImGui::Text("Current jitter pixels: %.3f, %.3f",
                postProcess_.taaCurrentJitterPixels().x,
                postProcess_.taaCurrentJitterPixels().y);
    ImGui::Text(
        "Current jitter NDC: %.6f, %.6f", postProcess_.taaCurrentJitterNdc().x, postProcess_.taaCurrentJitterNdc().y);
}

void Renderer::drawSsrDebugUi()
{
    if (!ImGui::CollapsingHeader("Screen-Space Reflections", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (!ssr_.available()) {
        ImGui::TextDisabled("SSR unavailable (requires a samplable main depth image).");
        return;
    }

    // "##ssr" suffixes keep these IDs distinct from same-labelled widgets in
    // other sections (e.g. the light swarm's Intensity slider) — collapsing
    // headers do not scope ImGui IDs.
    bool changed = false;
    changed |= ImGui::Checkbox("Enabled##ssr", &ssrSettings_.enabled);
    changed |= ImGui::SliderInt("March steps##ssr", &ssrSettings_.maxSteps, 8, 128);
    changed |= ImGui::SliderInt("Refinement steps##ssr", &ssrSettings_.refinementSteps, 0, 8);
    changed |= ImGui::SliderFloat("Max distance##ssr", &ssrSettings_.maxDistance, 1.0f, 200.0f, "%.1f");
    changed |= ImGui::SliderFloat("Thickness##ssr", &ssrSettings_.thickness, 0.01f, 2.0f, "%.3f");
    changed |= ImGui::SliderFloat("Intensity##ssr", &ssrSettings_.intensity, 0.0f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("Max roughness##ssr", &ssrSettings_.maxRoughness, 0.05f, 1.0f, "%.2f");
    if (changed) {
        clampRuntimeSettings();
    }

    ImGui::Text("Active: %s", frameSsrActive_ ? "yes" : "no");
    ImGui::TextDisabled("Traces the main depth buffer; reflections blend in before TAA. Lower Max");
    ImGui::TextDisabled("roughness for mirror-only, raise Intensity to exaggerate for inspection.");
}

void Renderer::drawRenderGraphDebugUi()
{
    const auto& passes = renderGraph_.debugPasses();
    const auto& resources = renderGraph_.debugResources();
    ImGui::Text("Declared pass order: %zu passes, %zu resources", passes.size(), resources.size());

    // ScrollX makes the table a child window whose default height is "remaining
    // visible space" — near the bottom of a scrolled panel that collapses to a
    // bare scrollbar. Pin an explicit height (with ScrollY) so both tables stay
    // readable regardless of where the section sits in the debug window.
    // Sizing must be FixedFit: stretch-proportional columns degenerate under
    // ScrollX (the long text columns collapse to a few characters and wrapped
    // text turns into one giant row).
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit |
                                      ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;
    const ImVec2 tableSize(0.0f, ImGui::GetTextLineHeightWithSpacing() * 12.0f);
    const float wideColumnWidth = ImGui::GetFontSize() * 16.0f;
    if (ImGui::BeginTable("RenderGraphPassesV2", 8, flags, tableSize)) {
        ImGui::TableSetupColumn("#");
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Status");
        ImGui::TableSetupColumn("Side effect");
        ImGui::TableSetupColumn("Reads", ImGuiTableColumnFlags_WidthFixed, wideColumnWidth);
        ImGui::TableSetupColumn("Writes", ImGuiTableColumnFlags_WidthFixed, wideColumnWidth);
        ImGui::TableSetupColumn("Barriers / notes", ImGuiTableColumnFlags_WidthFixed, wideColumnWidth);
        ImGui::TableHeadersRow();

        for (size_t index = 0; index < passes.size(); ++index) {
            const renderer::RenderPassNode& pass = passes[index];
            const std::string reads = resourceUsageList(pass, renderer::RenderResourceAccess::Read);
            const std::string writes = resourceUsageList(pass, renderer::RenderResourceAccess::Write);
            const char* status = pass.executed ? "executed" : (pass.culled ? "culled" : "not executed");

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", index);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(pass.name.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%s / %s",
                        renderer::renderPassExecutionTypeName(pass.executionType),
                        renderer::renderPassTypeName(pass.type));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(status);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(pass.sideEffect ? "yes" : "no");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", reads.c_str());
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", writes.c_str());
            ImGui::TableNextColumn();
            if (pass.culled && !pass.cullReason.empty()) {
                ImGui::TextWrapped("%s", pass.cullReason.c_str());
            } else {
                ImGui::TextWrapped("%s", pass.transitionSummary.c_str());
            }
        }

        ImGui::EndTable();
    }

    if (ImGui::BeginTable("RenderGraphResourcesV2", 8, flags, tableSize)) {
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("Kind");
        ImGui::TableSetupColumn("Lifetime");
        ImGui::TableSetupColumn("Extent / size");
        ImGui::TableSetupColumn("Format");
        ImGui::TableSetupColumn("Mips/layers");
        ImGui::TableSetupColumn("Initial layout");
        ImGui::TableSetupColumn("Final layout");
        ImGui::TableHeadersRow();

        for (const renderer::RenderGraphResourceDebugInfo& resource : resources) {
            const bool isTexture = resource.kind == renderer::RGResourceKind::Texture;
            std::string dimensions;
            if (isTexture) {
                dimensions = std::to_string(resource.extent.width) + " x " + std::to_string(resource.extent.height);
            } else {
                dimensions = std::to_string(resource.size) + " bytes";
            }

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(resource.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(renderer::renderGraphResourceKindName(resource.kind));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(resource.imported ? "imported" : "transient");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(dimensions.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(isTexture ? vkFormatName(resource.format) : "-");
            ImGui::TableNextColumn();
            ImGui::Text("%u / %u", resource.mipLevels, resource.arrayLayers);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(isTexture ? imageLayoutName(resource.initialLayout) : "-");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(isTexture ? imageLayoutName(resource.finalLayout) : "-");
        }

        ImGui::EndTable();
    }
}

void Renderer::drawSceneEditingDebugUi()
{
    const std::string scenePath = sceneDocumentPath_.string();
    ImGui::TextWrapped("Scene JSON: %s", scenePath.c_str());

    if (ImGui::Button("Save Scene")) {
        saveSceneFromUi();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load Scene")) {
        loadSceneFromUi();
    }

    ImGui::TextWrapped("Last save: %s", lastSceneSaveStatus_.c_str());
    if (lastSceneLoadStatus_.starts_with(kNoSavedSceneFoundMessage)) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f), "Last load: %s", lastSceneLoadStatus_.c_str());
    } else {
        ImGui::TextWrapped("Last load: %s", lastSceneLoadStatus_.c_str());
    }

    if (ImGui::CollapsingHeader("Camera and Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawCameraLightEditorDebugUi();
    }

    ImGui::Separator();
}

void Renderer::drawCameraLightEditorDebugUi()
{
    if (portfolioCaptureMode_) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f),
                           "Portfolio mode is active. F12 and Load Portfolio Showcase reapply portfolio camera and "
                           "lighting presets.");
    }

    ImGui::TextDisabled("Viewport: RMB+WASD/QE fly, scroll speed; LMB pick; Alt+LMB orbit; MMB pan; W/E/R gizmo.");

    if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool cameraChanged = false;
        cameraChanged |= ImGui::DragFloat3("Position", &camera_.position.x, 0.02f, -1000.0f, 1000.0f, "%.3f");
        cameraChanged |= ImGui::DragFloat3("Target", &camera_.target.x, 0.02f, -1000.0f, 1000.0f, "%.3f");
        cameraChanged |= ImGui::DragFloat3("Up", &camera_.up.x, 0.01f, -1.0f, 1.0f, "%.3f");

        float fovDegrees = glm::degrees(camera_.verticalFovRadians);
        if (ImGui::DragFloat("FOV", &fovDegrees, 0.25f, 1.0f, 160.0f, "%.2f deg")) {
            camera_.verticalFovRadians = glm::radians(std::clamp(fovDegrees, 1.0f, 160.0f));
            cameraChanged = true;
        }

        if (ImGui::DragFloat("Near plane", &camera_.nearPlane, 0.005f, 0.001f, 1000.0f, "%.3f")) {
            cameraChanged = true;
        }
        if (ImGui::DragFloat("Far plane", &camera_.farPlane, 0.1f, 0.01f, 10000.0f, "%.2f")) {
            cameraChanged = true;
        }

        if (cameraChanged) {
            camera_.nearPlane = std::max(camera_.nearPlane, 0.001f);
            camera_.farPlane = std::max(camera_.farPlane, camera_.nearPlane + 0.001f);
            camera_.up = normalizedOrFallback(camera_.up, {0.0f, 1.0f, 0.0f});
            if (glm::length(camera_.target - camera_.position) <= 0.001f) {
                camera_.target = camera_.position + glm::vec3{0.0f, 0.0f, -1.0f};
            }
            csmSettings_.nearPlane = camera_.nearPlane;
            csmSettings_.farPlane = camera_.farPlane;
            clampRuntimeSettings();
            invalidateTaaHistory();
        }

        if (ImGui::Button("Reset Default Camera")) {
            resetCameraToDefault();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Portfolio Camera")) {
            resetCameraToPortfolioPreset();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Occlusion Test Camera##scene")) {
            resetCameraToOcclusionTestPreset();
        }

        ImGui::TextDisabled("Camera movement speed: not available; there is no free-camera controller yet.");
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Direction", &directionalLightSettings_.direction.x, 0.01f, -1.0f, 1.0f, "%.3f");
        directionalLightSettings_.direction =
            normalizedOrFallback(directionalLightSettings_.direction,
                                 {kDirectionalLightDirection.x,
                                  kDirectionalLightDirection.y,
                                  kDirectionalLightDirection.z});

        ImGui::ColorEdit3("Color", &directionalLightSettings_.color.x);
        directionalLightSettings_.color = glm::max(directionalLightSettings_.color, glm::vec3{0.0f});
        ImGui::DragFloat("Intensity##directional", &directionalLightSettings_.intensity, 0.01f, 0.0f, 16.0f, "%.3f");
        directionalLightSettings_.intensity = std::max(directionalLightSettings_.intensity, 0.0f);

        if (portfolioCaptureMode_) {
            ImGui::TextDisabled("The editable directional light is used when portfolio mode is disabled.");
        }
        if (ImGui::Button("Reset Default Light")) {
            resetDirectionalLightToDefault();
        }
        ImGui::TreePop();
    }
}

void Renderer::drawSceneHierarchyDebugUi()
{
    if (selectedRenderObjectIndex_ >= renderObjects_.size()) {
        selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
    }

    drawSceneEditingDebugUi();

    ImGui::Text("RenderObjects: %zu", renderObjects_.size());
    ImGui::SameLine();
    ImGui::Text("DrawItems: %zu", allDrawItems_.size());
    ImGui::SameLine();
    ImGui::Text("Main batches: %zu", meshDrawBatches_.size());

    ImGui::BeginChild("SceneHierarchyList", ImVec2(0.0f, 280.0f), true);
    const ImGuiTreeNodeFlags sceneFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (ImGui::TreeNodeEx("Scene", sceneFlags)) {
        for (size_t objectIndex = 0; objectIndex < renderObjects_.size(); ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            const ObjectDrawDebugInfo debugInfo =
                objectDrawDebugInfo(static_cast<uint32_t>(objectIndex));
            const bool selected = selectedRenderObjectIndex_ == objectIndex;
            ImGuiTreeNodeFlags objectFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (selected) {
                objectFlags |= ImGuiTreeNodeFlags_Selected;
            }

            const uint32_t objectId = renderObjectEditorId(object);
            const std::string objectName = object.debugName.empty() ? "(unnamed)" : object.debugName;
            const std::string label = std::string(object.visible ? "" : "[hidden] ") + objectName + "##" +
                                      std::to_string(objectId);
            const bool open = ImGui::TreeNodeEx(label.c_str(), objectFlags);
            if (ImGui::IsItemClicked()) {
                selectedRenderObjectIndex_ = objectIndex;
            }

            if (open) {
                const std::string materialLabel = materialDebugLabel(object);
                const std::string mainCullingLabel = mainCullingDebugLabel(debugInfo);
                const std::string shadowCullingLabel = shadowCullingDebugLabel(debugInfo);
                ImGui::Text("Object ID: %u", objectId);
                ImGui::Text("Debug ID: %u", object.debugId);
                ImGui::Text("Visible: %s", object.visible ? "yes" : "no");
                ImGui::Text("Source: %s", renderObjectSourceTypeName(object.sourceType));
                ImGui::Text("Mesh: %s", meshDebugLabel(object.mesh).c_str());
                ImGui::Text("Material: %s", materialLabel.c_str());
                ImGui::Text("Submeshes: %u", meshSubmeshCount(object.mesh));
                ImGui::Text("Draw items: %zu", debugInfo.drawItemCount);
                ImGui::Text("Main: %s", mainCullingLabel.c_str());
                ImGui::Text("Shadow: %s", shadowCullingLabel.c_str());
                if (object.mesh) {
                    ImGui::TextWrapped("Local bounds: %s", formatAabb(object.mesh->localBounds()).c_str());
                }
                ImGui::TextWrapped("World bounds: %s", formatAabb(object.worldBounds()).c_str());
                ImGui::TreePop();
            }
        }
        ImGui::TreePop();
    }
    ImGui::EndChild();

    ImGui::Separator();
    ImGui::TextUnformatted("Inspector");
    if (selectedRenderObjectIndex_ == kInvalidRenderObjectIndex) {
        ImGui::TextDisabled("No RenderObject selected.");
        return;
    }

    drawSelectedRenderObjectInspector(static_cast<uint32_t>(selectedRenderObjectIndex_));
}

void Renderer::drawSelectedRenderObjectInspector(uint32_t objectIndex)
{
    if (objectIndex >= renderObjects_.size()) {
        ImGui::TextDisabled("Selected RenderObject is no longer available.");
        return;
    }

    renderer::RenderObject& object = renderObjects_[objectIndex];
    const ObjectDrawDebugInfo debugInfo = objectDrawDebugInfo(objectIndex);
    const std::string materialLabel = materialDebugLabel(object);
    const std::string mainCullingLabel = mainCullingDebugLabel(debugInfo);
    const std::string shadowCullingLabel = shadowCullingDebugLabel(debugInfo);
    const uint32_t objectId = renderObjectEditorId(object);

    ImGui::Text("Name: %s", object.debugName.empty() ? "(unnamed)" : object.debugName.c_str());
    ImGui::Text("Object index: %u", objectIndex);
    ImGui::Text("Object ID: %u", objectId);
    ImGui::Text("Debug ID: %u", object.debugId);
    ImGui::Text("Source: %s", renderObjectSourceTypeName(object.sourceType));
    ImGui::Text("Mesh: %s", meshDebugLabel(object.mesh).c_str());
    ImGui::Text("Mesh pointer: %p", static_cast<const void*>(object.mesh));
    if (object.mesh) {
        ImGui::Text("Mesh indices: %u", object.mesh->indexCount());
    }
    ImGui::Text("Material: %s", materialLabel.c_str());
    ImGui::Text("Material slots: %zu", object.materialCount);
    ImGui::Text("Submeshes: %u", meshSubmeshCount(object.mesh));
    ImGui::Text("Draw items: %zu", debugInfo.drawItemCount);
    if (debugInfo.hasObjectDataIndex) {
        ImGui::Text("First object-data index: %u", debugInfo.firstObjectDataIndex);
    } else {
        ImGui::TextDisabled("First object-data index: unavailable until draw data is built");
    }
    ImGui::Text("Main culling: %s", mainCullingLabel.c_str());
    ImGui::Text("Shadow culling: %s", shadowCullingLabel.c_str());
    ImGui::Text("Shadow draw items visible: %zu", debugInfo.visibleShadowDrawItemCount);

    bool visible = object.visible;
    if (ImGui::Checkbox("Visible", &visible)) {
        object.visible = visible;
        invalidateDepthPyramid();
    }

    ImGui::SeparatorText("Transform");

    // Viewport gizmo mode (also switchable with W / E / R, and X for space).
    auto gizmoRadio = [this](const char* label, GizmoOperation op) {
        if (ImGui::RadioButton(label, gizmoOperation_ == op)) {
            gizmoOperation_ = op;
        }
    };
    gizmoRadio("Move (W)", GizmoOperation::Translate);
    ImGui::SameLine();
    gizmoRadio("Rotate (E)", GizmoOperation::Rotate);
    ImGui::SameLine();
    gizmoRadio("Scale (R)", GizmoOperation::Scale);
    ImGui::SameLine();
    ImGui::Checkbox("World (X)", &gizmoWorldSpace_);

    if (object.transform.useMatrixOverride) {
        ImGui::TextDisabled("Matrix override transform; editing TRS converts it to position/rotation/scale.");
        if (ImGui::Button("Convert to Editable TRS")) {
            if (convertMatrixOverrideToEditableTrs(object.transform)) {
                object.animateTransform = false;
                invalidateDepthPyramid();
            }
        }
    }

    const TransformComponents transformComponents = editableTransformComponents(object.transform);
    if (!transformComponents.valid) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f),
                           "Transform cannot be decomposed for editing.");
    } else {
        glm::vec3 position = transformComponents.position;
        if (ImGui::DragFloat3("Position", &position.x, 0.01f, -1000.0f, 1000.0f, "%.3f")) {
            if (convertMatrixOverrideToEditableTrs(object.transform)) {
                object.transform.position = position;
                object.animateTransform = false;
                invalidateDepthPyramid();
            }
        }

        glm::vec3 rotationDegrees = glm::degrees(transformComponents.rotationRadians);
        if (ImGui::DragFloat3("Rotation", &rotationDegrees.x, 0.25f, -3600.0f, 3600.0f, "%.2f deg")) {
            if (convertMatrixOverrideToEditableTrs(object.transform)) {
                object.transform.rotationRadians = glm::radians(rotationDegrees);
                object.animateTransform = false;
                invalidateDepthPyramid();
            }
        }

        glm::vec3 scale = transformComponents.scale;
        if (ImGui::DragFloat3("Scale", &scale.x, 0.01f, -1000.0f, 1000.0f, "%.3f")) {
            if (convertMatrixOverrideToEditableTrs(object.transform)) {
                object.transform.scale = scale;
                object.animateTransform = false;
                invalidateDepthPyramid();
            }
        }
    }

    ImGui::Text("Demo animation: %s", object.animateTransform ? "enabled" : "disabled");
    ImGui::TextWrapped("Transform summary: %s", transformDebugSummary(object.transform).c_str());
    if (object.mesh) {
        ImGui::TextWrapped("Local bounds: %s", formatAabb(object.mesh->localBounds()).c_str());
    } else {
        ImGui::TextDisabled("Local bounds: unavailable");
    }
    ImGui::TextWrapped("World bounds: %s", formatAabb(object.worldBounds()).c_str());

    if (ImGui::TreeNodeEx("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        const std::vector<const renderer::Material*> materials = materialsForObject(object);
        if (materials.empty()) {
            ImGui::TextDisabled("Selected RenderObject has no material.");
        } else {
            if (materials.size() > 1) {
                ImGui::Text("Material count: %zu (showing first resolved material)", materials.size());
            }
            drawMaterialDebugSection(materials.front(), true, mutableMaterialFromPointer(materials.front()));
        }
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("World transform matrix")) {
        const glm::mat4 model = object.transform.modelMatrix();
        for (int row = 0; row < 4; ++row) {
            const std::string matrixRow = formatMatrixRow(model, row);
            ImGui::TextUnformatted(matrixRow.c_str());
        }
        ImGui::TreePop();
    }
}

void Renderer::drawMaterialInspectorDebugUi()
{
    if (selectedRenderObjectIndex_ == kInvalidRenderObjectIndex || selectedRenderObjectIndex_ >= renderObjects_.size()) {
        ImGui::TextDisabled("No RenderObject selected.");
        return;
    }

    const renderer::RenderObject& object = renderObjects_[selectedRenderObjectIndex_];
    ImGui::Text("RenderObject: %s", object.debugName.empty() ? "(unnamed)" : object.debugName.c_str());
    ImGui::Text("Object index: %zu", selectedRenderObjectIndex_);

    const std::vector<const renderer::Material*> materials = materialsForObject(object);
    if (materials.empty()) {
        ImGui::TextDisabled("Selected RenderObject has no material.");
        return;
    }

    if (materials.size() > 1) {
        ImGui::Text("Resolved materials: %zu (showing first material)", materials.size());
        constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("MaterialInspectorMaterialList", 3, flags)) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Material");
            ImGui::TableSetupColumn("Source");
            ImGui::TableHeadersRow();
            for (size_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
                const renderer::Material* material = materials[materialIndex];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%zu", materialIndex);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(materialNameOrFallback(material));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(material ? materialSourceName(material->source) : "unknown");
            }
            ImGui::EndTable();
        }
    }

    drawMaterialDebugSection(materials.front(), true, mutableMaterialFromPointer(materials.front()));
}

void Renderer::drawTextureDebugUi()
{
    if (ImGui::CollapsingHeader("Texture Debug Views", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (selectedRenderObjectIndex_ == kInvalidRenderObjectIndex ||
            selectedRenderObjectIndex_ >= renderObjects_.size()) {
            ImGui::TextDisabled("No RenderObject selected.");
        } else {
            const renderer::RenderObject& object = renderObjects_[selectedRenderObjectIndex_];
            const renderer::Material* material = primaryMaterialForObject(object);
            if (!material) {
                ImGui::TextDisabled("Selected RenderObject has no material textures to inspect.");
            } else {
                ImGui::Text("RenderObject: %s", object.debugName.empty() ? "(unnamed)" : object.debugName.c_str());
                ImGui::Text("Material: %s", materialNameOrFallback(material));
                drawMaterialTextureSlotDebugUi("Base color",
                                               "sRGB base color",
                                               material->baseColorTexture,
                                               material->baseColorTextureIndex,
                                               material->baseColorTextureFallback,
                                               true);
                drawMaterialTextureSlotDebugUi("Normal",
                                               "linear normal",
                                               material->normalTexture,
                                               material->normalTextureIndex,
                                               material->normalTextureFallback,
                                               true);
                drawMaterialTextureSlotDebugUi("Metallic-roughness",
                                               "linear metallic-roughness",
                                               material->metallicRoughnessTexture,
                                               material->metallicRoughnessTextureIndex,
                                               material->metallicRoughnessTextureFallback,
                                               true);
            }
        }
    }

    if (ImGui::CollapsingHeader("Global/Post-process Texture Metadata", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawGlobalTextureMetadata();
    }
}

void Renderer::drawMaterialDebugSection(const renderer::Material* material,
                                        bool includeTextureSummary,
                                        renderer::Material* editableMaterial)
{
    if (!material) {
        ImGui::TextDisabled("Material: unavailable");
        return;
    }

    ImGui::PushID(static_cast<const void*>(material));

    const auto runtimeMaterialIndexLabel = [this, material]() {
        for (size_t materialIndex = 0; materialIndex < materialVariants_.size(); ++materialIndex) {
            if (&materialVariants_[materialIndex] == material) {
                return "materialVariants[" + std::to_string(materialIndex) + "]";
            }
        }
        for (size_t materialIndex = 0; materialIndex < importedMaterials_.size(); ++materialIndex) {
            if (&importedMaterials_[materialIndex] == material) {
                return "importedMaterials[" + std::to_string(materialIndex) + "]";
            }
        }
        if (&checkerboardMaterial_ == material) {
            return std::string("checkerboardMaterial");
        }
        return std::string("untracked");
    };

    const std::string descriptorSetLabel = descriptorSetDebugString(material->descriptorSet);
    ImGui::Text("Material debug name: %s", materialNameOrFallback(material));
    ImGui::Text("Source: %s", materialSourceName(material->source));
    ImGui::Text("Runtime material index: %s", runtimeMaterialIndexLabel().c_str());
    ImGui::Text("Asset name: %s", material->assetName.empty() ? "(none)" : material->assetName.c_str());
    ImGui::Text("Shader: %s", material->shader.empty() ? "(default)" : material->shader.c_str());
    ImGui::Text("Alpha mode: %s", material->alphaMode.empty() ? "OPAQUE" : material->alphaMode.c_str());
    if (!material->sourceAssetPath.empty()) {
        ImGui::TextWrapped("Source asset path: %s", stableProjectPathString(material->sourceAssetPath).c_str());
    } else {
        ImGui::TextDisabled("Source asset path: none");
    }

    if (editableMaterial) {
        glm::vec4 baseColor = editableMaterial->baseColorFactor;
        if (ImGui::ColorEdit4("baseColorFactor", &baseColor.x, ImGuiColorEditFlags_Float)) {
            editableMaterial->baseColorFactor = glm::clamp(baseColor, glm::vec4(0.0f), glm::vec4(16.0f));
        }

        float metallic = editableMaterial->metallic;
        if (ImGui::DragFloat("metallic", &metallic, 0.01f, 0.0f, 1.0f, "%.3f")) {
            editableMaterial->metallic = std::clamp(metallic, 0.0f, 1.0f);
        }

        float roughness = editableMaterial->roughness;
        if (ImGui::DragFloat("roughness", &roughness, 0.01f, 0.0f, 1.0f, "%.3f")) {
            editableMaterial->roughness = std::clamp(roughness, 0.0f, 1.0f);
        }

        glm::vec3 emissive = editableMaterial->emissiveFactor;
        if (ImGui::ColorEdit3("emissiveFactor", &emissive.x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)) {
            editableMaterial->emissiveFactor = glm::clamp(emissive, glm::vec3(0.0f), glm::vec3(64.0f));
        }

        float alphaCutoff = editableMaterial->alphaCutoff;
        if (ImGui::DragFloat("alphaCutoff", &alphaCutoff, 0.01f, 0.0f, 1.0f, "%.3f")) {
            editableMaterial->alphaCutoff = std::clamp(alphaCutoff, 0.0f, 1.0f);
        }

        bool doubleSided = editableMaterial->doubleSided;
        if (ImGui::Checkbox("doubleSided metadata", &doubleSided)) {
            editableMaterial->doubleSided = doubleSided;
        }

        if (ImGui::Button("Save Material")) {
            saveMaterialAssetFromUi(*editableMaterial);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload Material")) {
            reloadMaterialAssetFromUi(*editableMaterial);
        }
        ImGui::TextWrapped("Material asset status: %s", lastMaterialAssetStatus_.c_str());
    } else {
        ImGui::Text("baseColorFactor: %s", formatVec4(material->baseColorFactor).c_str());
        ImGui::Text("metallic: %.3f", material->metallic);
        ImGui::Text("roughness: %.3f", material->roughness);
        ImGui::Text("alphaCutoff: %.3f", material->alphaCutoff);
        ImGui::Text("doubleSided metadata: %s", material->doubleSided ? "true" : "false");
    }

    ImGui::Text("multiScatterStrength: %.3f", material->multiScatterStrength);
    ImGui::Text("baseColorTextureIndex: %u", material->baseColorTextureIndex);
    ImGui::Text("normalTextureIndex: %u", material->normalTextureIndex);
    ImGui::Text("metallicRoughnessTextureIndex: %u", material->metallicRoughnessTextureIndex);
    ImGui::TextWrapped("Base color texture path: %s",
                       material->baseColorTexturePath.empty()
                           ? "(none)"
                           : material->baseColorTexturePath.generic_string().c_str());
    ImGui::TextWrapped("Normal texture path: %s",
                       material->normalTexturePath.empty() ? "(none)" : material->normalTexturePath.generic_string().c_str());
    ImGui::TextWrapped("Metallic-roughness texture path: %s",
                       material->metallicRoughnessTexturePath.empty()
                           ? "(none)"
                           : material->metallicRoughnessTexturePath.generic_string().c_str());
    ImGui::Text("Bindless material textures: %s", isBindlessMaterialTextureActive() ? "active" : "inactive");
    ImGui::Text("Legacy material descriptor fallback: %s",
                isBindlessMaterialTextureActive() ? "inactive" : "active");
    ImGui::Text("Material descriptor set: %s", descriptorSetLabel.c_str());

    if (!includeTextureSummary) {
        ImGui::PopID();
        return;
    }

    if (ImGui::TreeNode("Material texture slots")) {
        drawMaterialTextureSlotDebugUi("Base color",
                                       "sRGB base color",
                                       material->baseColorTexture,
                                       material->baseColorTextureIndex,
                                       material->baseColorTextureFallback,
                                       false);
        drawMaterialTextureSlotDebugUi("Normal",
                                       "linear normal",
                                       material->normalTexture,
                                       material->normalTextureIndex,
                                       material->normalTextureFallback,
                                       false);
        drawMaterialTextureSlotDebugUi("Metallic-roughness",
                                       "linear metallic-roughness",
                                       material->metallicRoughnessTexture,
                                       material->metallicRoughnessTextureIndex,
                                       material->metallicRoughnessTextureFallback,
                                       false);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void Renderer::drawMaterialTextureSlotDebugUi(const char* slotName,
                                              const char* semantic,
                                              const rhi::VulkanTexture* texture,
                                              uint32_t bindlessIndex,
                                              bool fallbackUsed,
                                              bool showPreview)
{
    if (!ImGui::TreeNode(slotName)) {
        return;
    }

    if (!texture || !texture->valid()) {
        ImGui::TextDisabled("Texture: unavailable");
        ImGui::TreePop();
        return;
    }

    const rhi::TextureDebugMetadata& metadata = texture->debugMetadata();
    const std::string debugName = metadata.debugName.empty() ? "(unnamed texture)" : metadata.debugName;
    const bool effectiveFallback = fallbackUsed || metadata.fallback;

    if (showPreview) {
        drawTexturePreview(*texture, 112.0f);
    }

    ImGui::Text("Texture debug name: %s", debugName.c_str());
    ImGui::Text("Bindless index: %u", bindlessIndex);
    ImGui::Text("Dimensions: %u x %u", texture->width(), texture->height());
    ImGui::Text("Mip levels: %u", texture->mipLevels());
    ImGui::Text("Vulkan format: %s", vkFormatName(texture->format()));
    ImGui::Text("Color space / semantic: %s", semantic);
    ImGui::Text("Tracked color space: %s", std::string(colorSpaceName(metadata.colorSpace)).c_str());
    ImGui::Text("Source: %s", textureDebugSourceName(metadata.source));
    if (!metadata.sourcePath.empty()) {
        ImGui::TextWrapped("Path: %s", metadata.sourcePath.c_str());
    }
    ImGui::Text("Fallback texture used: %s", effectiveFallback ? "yes" : "no");
    ImGui::TreePop();
}

void Renderer::drawTexturePreview(const rhi::VulkanTexture& texture, float size)
{
    const VkDescriptorSet descriptorSet =
        imguiLayer_.texturePreviewDescriptor(texture.imageView(), texture.sampler(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (descriptorSet == VK_NULL_HANDLE) {
        ImGui::TextDisabled("Preview unavailable.");
        return;
    }

    const float width = static_cast<float>(std::max(texture.width(), 1U));
    const float height = static_cast<float>(std::max(texture.height(), 1U));
    ImVec2 imageSize{size, size};
    if (width > height) {
        imageSize.y = size * (height / width);
    } else if (height > width) {
        imageSize.x = size * (width / height);
    }

    ImGui::Image((ImTextureID)descriptorSet, imageSize);
}

void Renderer::drawRenderTargetPreview(VkImageView imageView,
                                       VkSampler sampler,
                                       VkImageLayout imageLayout,
                                       uint32_t width,
                                       uint32_t height,
                                       float size,
                                       float exposureScale)
{
    const VkDescriptorSet descriptorSet =
        imguiLayer_.renderTargetPreviewDescriptor(imageView, sampler, imageLayout);
    if (descriptorSet == VK_NULL_HANDLE) {
        ImGui::TextDisabled("Preview unavailable.");
        return;
    }

    const float imageWidth = static_cast<float>(std::max(width, 1U));
    const float imageHeight = static_cast<float>(std::max(height, 1U));
    ImVec2 imageSize{size, size};
    if (imageWidth > imageHeight) {
        imageSize.y = size * (imageHeight / imageWidth);
    } else if (imageHeight > imageWidth) {
        imageSize.x = size * (imageWidth / imageHeight);
    }

    const float tintScale = std::clamp(exposureScale, 0.0f, 16.0f);
    ImGui::Image((ImTextureID)descriptorSet,
                 imageSize,
                 ImVec2(0.0f, 0.0f),
                 ImVec2(1.0f, 1.0f),
                 ImVec4(tintScale, tintScale, tintScale, 1.0f),
                 ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
}

void Renderer::drawRenderTargetDebugUi()
{
    ImGui::SliderFloat("Preview scale", &debugUiSettings_.renderTargetPreviewScale, 0.25f, 2.0f, "%.2f");
    ImGui::SliderFloat("Preview exposure", &debugUiSettings_.renderTargetPreviewExposure, 0.05f, 8.0f, "%.2f");
    ImGui::TextDisabled("HDR previews are raw ImGui samples multiplied by preview exposure; no channel remap.");

    if (ImGui::CollapsingHeader("Preview Toggles", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show scene color", &showRenderTargetSceneColor_);
        ImGui::Checkbox("Show TAA history", &showRenderTargetTaaHistory_);
        ImGui::Checkbox("Show bloom extract", &showRenderTargetBloomExtract_);
        ImGui::Checkbox("Show blurred bloom", &showRenderTargetBlurredBloom_);
        ImGui::Checkbox("Show bloom mip-chain", &showRenderTargetBloomMipChain_);
        ImGui::Checkbox("Show final composite metadata", &showRenderTargetFinalCompositeMetadata_);
        ImGui::Checkbox("Show BRDF LUT", &showRenderTargetBrdfLut_);
        ImGui::Checkbox("Show CSM cascades", &showRenderTargetCsmCascades_);
    }

    drawRenderTargetMetadataTable();
    drawRenderTargetPreviews();
}

void Renderer::drawRenderTargetMetadataTable()
{
    const auto extentString = [](uint32_t width, uint32_t height) {
        return std::to_string(width) + " x " + std::to_string(height);
    };
    const auto cubeExtentString = [](uint32_t faceSize) {
        return std::to_string(faceSize) + " x " + std::to_string(faceSize) + " x 6";
    };
    const auto layoutUsage = [](const char* usage, VkImageLayout layout) {
        return std::string(usage) + "; current layout " + imageLayoutName(layout);
    };

    if (ImGui::CollapsingHeader("Resource Metadata", ImGuiTreeNodeFlags_DefaultOpen)) {
        constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_ScrollX;
        if (ImGui::BeginTable("RenderTargetDebugMetadata", 8, flags)) {
            ImGui::TableSetupColumn("Debug name");
            ImGui::TableSetupColumn("Dimensions");
            ImGui::TableSetupColumn("Format");
            ImGui::TableSetupColumn("Mips", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Layers", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Usage");
            ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Sampled as");
            ImGui::TableHeadersRow();

            const auto addMetadataRow = [](const RenderTargetDebugMetadata& metadata) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(metadata.debugName);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(metadata.dimensions.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(vkFormatName(metadata.format));
                ImGui::TableNextColumn();
                ImGui::Text("%u", metadata.mipLevels);
                ImGui::TableNextColumn();
                ImGui::Text("%u", metadata.arrayLayers);
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", metadata.usage.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(metadata.previewable ? "yes" : "no");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(metadata.sampledAs.c_str());
            };

            if (postProcess_.sceneColor().image() != VK_NULL_HANDLE) {
                const VkExtent3D extent = postProcess_.sceneColor().extent();
                addMetadataRow(RenderTargetDebugMetadata{
                    "SceneColorHDR",
                    extentString(extent.width, extent.height),
                    postProcess_.sceneColor().format(),
                    1,
                    1,
                    layoutUsage("HDR scene color attachment sampled by bloom, exposure, composite, and preview",
                                postProcess_.sceneColorLayout()),
                    true,
                    "2D HDR"});
            }
            for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
                if (postProcess_.taaHistoryImages()[historyIndex].image() != VK_NULL_HANDLE) {
                    const VkExtent3D extent = postProcess_.taaHistoryImages()[historyIndex].extent();
                    const std::string debugName = "TAAHistory" + std::to_string(historyIndex);
                    addMetadataRow(RenderTargetDebugMetadata{
                        debugName.c_str(),
                        extentString(extent.width, extent.height),
                        postProcess_.taaHistoryImages()[historyIndex].format(),
                        1,
                        1,
                        layoutUsage("Persistent HDR TAA history sampled by resolve and post-process",
                                    postProcess_.taaHistoryLayouts()[historyIndex]),
                        true,
                        "2D HDR"});
                }
            }
            if (postProcess_.bloomExtract().image() != VK_NULL_HANDLE) {
                const VkExtent3D extent = postProcess_.bloomExtract().extent();
                addMetadataRow(RenderTargetDebugMetadata{
                    "BloomExtract",
                    extentString(extent.width, extent.height),
                    postProcess_.bloomExtract().format(),
                    1,
                    1,
                    layoutUsage("Bloom bright-pass output sampled by horizontal blur and preview",
                                postProcess_.bloomExtractLayout()),
                    true,
                    "2D HDR"});
            }
            if (postProcess_.bloomPing().image() != VK_NULL_HANDLE) {
                const VkExtent3D extent = postProcess_.bloomPing().extent();
                addMetadataRow(
                    RenderTargetDebugMetadata{"BloomPing",
                                              extentString(extent.width, extent.height),
                                              postProcess_.bloomPing().format(),
                                              1,
                                              1,
                                              layoutUsage("Horizontal blur output sampled by vertical blur and preview",
                                                          postProcess_.bloomPingLayout()),
                                              true,
                                              "2D HDR"});
            }
            if (postProcess_.bloomPong().image() != VK_NULL_HANDLE) {
                const VkExtent3D extent = postProcess_.bloomPong().extent();
                addMetadataRow(RenderTargetDebugMetadata{
                    "BloomPong",
                    extentString(extent.width, extent.height),
                    postProcess_.bloomPong().format(),
                    1,
                    1,
                    layoutUsage("Final blurred bloom sampled by composite and preview", postProcess_.bloomPongLayout()),
                    true,
                    "2D HDR"});
            }
            for (size_t level = 0; level < postProcess_.bloomMipDownsampleImages().size(); ++level) {
                const VkExtent3D extent = postProcess_.bloomMipDownsampleImages()[level].extent();
                const std::string debugName = "BloomMipDownsample" + std::to_string(level);
                addMetadataRow(
                    RenderTargetDebugMetadata{debugName.c_str(),
                                              extentString(extent.width, extent.height),
                                              postProcess_.bloomMipDownsampleImages()[level].format(),
                                              1,
                                              1,
                                              layoutUsage("Mip-chain bloom downsample level",
                                                          level < postProcess_.bloomMipDownsampleLayouts().size()
                                                              ? postProcess_.bloomMipDownsampleLayouts()[level]
                                                              : VK_IMAGE_LAYOUT_UNDEFINED),
                                              true,
                                              "2D HDR"});
            }
            for (size_t level = 0; level < postProcess_.bloomMipUpsampleImages().size(); ++level) {
                const VkExtent3D extent = postProcess_.bloomMipUpsampleImages()[level].extent();
                const std::string debugName = "BloomMipUpsample" + std::to_string(level);
                addMetadataRow(
                    RenderTargetDebugMetadata{debugName.c_str(),
                                              extentString(extent.width, extent.height),
                                              postProcess_.bloomMipUpsampleImages()[level].format(),
                                              1,
                                              1,
                                              layoutUsage("Mip-chain bloom upsample accumulation level",
                                                          level < postProcess_.bloomMipUpsampleLayouts().size()
                                                              ? postProcess_.bloomMipUpsampleLayouts()[level]
                                                              : VK_IMAGE_LAYOUT_UNDEFINED),
                                              true,
                                              "2D HDR"});
            }
            if (showRenderTargetFinalCompositeMetadata_) {
                const VkExtent2D extent = swapchain_.extent();
                addMetadataRow(RenderTargetDebugMetadata{
                    "FinalCompositeSwapchain",
                    extentString(extent.width, extent.height) + ", " + std::to_string(swapchain_.imageCount()) +
                        " images",
                    swapchain_.colorFormat(),
                    1,
                    1,
                    "CompositePass color attachment, then ImGui overlay and present",
                    false,
                    "swapchain"});
            }
            if (brdfLutTexture_.valid()) {
                addMetadataRow(RenderTargetDebugMetadata{
                    "BrdfLut",
                    extentString(brdfLutTexture_.width(), brdfLutTexture_.height()),
                    brdfLutTexture_.format(),
                    1,
                    1,
                    layoutUsage("Split-sum IBL data texture sampled by PBR shading; linear, not sRGB",
                                brdfLutTexture_.layout()),
                    true,
                    "2D LUT"});
            }
            if (shadowMap_.valid()) {
                const VkExtent2D extent = shadowMap_.extent();
                addMetadataRow(RenderTargetDebugMetadata{
                    "CascadedShadowMapArray",
                    extentString(extent.width, extent.height),
                    shadowMap_.format(),
                    1,
                    shadowMap_.layerCount(),
                    layoutUsage("CSM depth attachment array sampled by lighting and per-layer debug previews",
                                shadowMap_.layout()),
                    true,
                    "2D array / per-layer 2D"});
            }
            if (depthPyramid_.image() != VK_NULL_HANDLE) {
                const VkExtent3D extent = depthPyramid_.extent();
                addMetadataRow(RenderTargetDebugMetadata{
                    "DepthPyramidHiZ",
                    extentString(extent.width, extent.height),
                    depthPyramid_.format(),
                    depthPyramid_.mipLevels(),
                    1,
                    layoutUsage("Normal-Z max-depth Hi-Z pyramid written by compute and sampled by GPU culling",
                                depthPyramid_.layout()),
                    true,
                    "2D mip chain"});
            }
            if (diffuseIrradianceMap_.valid()) {
                addMetadataRow(RenderTargetDebugMetadata{
                    "DiffuseIrradianceCubemap",
                    cubeExtentString(diffuseIrradianceMap_.faceSize()),
                    diffuseIrradianceMap_.format(),
                    diffuseIrradianceMap_.mipLevels(),
                    6,
                    hdrEnvironmentLoaded_ ? "HDR-derived diffuse IBL cubemap" : "Procedural diffuse IBL cubemap",
                    false,
                    "cube"});
            }
            if (prefilteredEnvironmentMap_.valid()) {
                addMetadataRow(RenderTargetDebugMetadata{
                    "PrefilteredSpecularCubemap",
                    cubeExtentString(prefilteredEnvironmentMap_.faceSize()),
                    prefilteredEnvironmentMap_.format(),
                    prefilteredEnvironmentMap_.mipLevels(),
                    6,
                    hdrEnvironmentLoaded_ ? "HDR-derived specular IBL mip chain"
                                          : "Procedural specular IBL mip chain",
                    false,
                    "cube"});
            }
            if (environmentMap_.valid()) {
                addMetadataRow(RenderTargetDebugMetadata{
                    "VisibleEnvironmentCubemap",
                    cubeExtentString(environmentMap_.faceSize()),
                    environmentMap_.format(),
                    environmentMap_.mipLevels(),
                    6,
                    hdrEnvironmentLoaded_ ? "HDR environment skybox source" : "Procedural skybox source",
                    false,
                    "cube"});
            }

            ImGui::EndTable();
        }
    }
}

void Renderer::drawRenderTargetPreviews()
{
    const float previewSize = 160.0f * std::clamp(debugUiSettings_.renderTargetPreviewScale, 0.25f, 2.0f);
    const float hdrPreviewExposure = std::clamp(debugUiSettings_.renderTargetPreviewExposure, 0.05f, 8.0f);

    if (showRenderTargetSceneColor_ && postProcess_.sceneColor().imageView() != VK_NULL_HANDLE &&
        ImGui::CollapsingHeader("HDR Scene Color", ImGuiTreeNodeFlags_DefaultOpen)) {
        const VkExtent3D extent = postProcess_.sceneColor().extent();
        ImGui::Text("Dimensions: %u x %u", extent.width, extent.height);
        ImGui::Text("Format: %s", vkFormatName(postProcess_.sceneColor().format()));
        ImGui::Text("Layout: %s", imageLayoutName(postProcess_.sceneColorLayout()));
        drawRenderTargetPreview(postProcess_.sceneColor().imageView(),
                                postProcess_.sampler(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                extent.width,
                                extent.height,
                                previewSize,
                                hdrPreviewExposure);
    }

    if (postProcess_.ambientOcclusion().imageView() != VK_NULL_HANDLE &&
        postProcess_.ambientOcclusionLayout() == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
        ImGui::CollapsingHeader("GTAO Visibility", ImGuiTreeNodeFlags_DefaultOpen)) {
        const VkExtent3D extent = postProcess_.ambientOcclusion().extent();
        ImGui::Text("Dimensions: %u x %u", extent.width, extent.height);
        ImGui::Text("Format: %s", vkFormatName(postProcess_.ambientOcclusion().format()));
        ImGui::TextDisabled("%s", frameGtaoActive_ ? "Denoised visibility (1 = lit)."
                                                    : "GTAO disabled; showing the last-written term.");
        // AO is an LDR [0,1] visibility term, so preview it without HDR exposure scaling.
        drawRenderTargetPreview(postProcess_.ambientOcclusion().imageView(),
                                postProcess_.sampler(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                extent.width,
                                extent.height,
                                previewSize,
                                1.0f);
    }

    if (showRenderTargetTaaHistory_ && ImGui::CollapsingHeader("TAA History", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
            if (postProcess_.taaHistoryImages()[historyIndex].imageView() == VK_NULL_HANDLE) {
                continue;
            }
            const VkExtent3D extent = postProcess_.taaHistoryImages()[historyIndex].extent();
            ImGui::Text("History %u: %u x %u, %s, %s",
                        historyIndex,
                        extent.width,
                        extent.height,
                        vkFormatName(postProcess_.taaHistoryImages()[historyIndex].format()),
                        imageLayoutName(postProcess_.taaHistoryLayouts()[historyIndex]));
            if (postProcess_.taaHistoryLayouts()[historyIndex] == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                drawRenderTargetPreview(postProcess_.taaHistoryImages()[historyIndex].imageView(),
                                        postProcess_.sampler(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        extent.width,
                                        extent.height,
                                        previewSize,
                                        hdrPreviewExposure);
            } else {
                ImGui::TextDisabled("Preview available after the history image reaches shader-read layout.");
            }
        }
    }

    if ((showRenderTargetBloomExtract_ || showRenderTargetBlurredBloom_) &&
        ImGui::CollapsingHeader("Bloom Targets", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (showRenderTargetBloomExtract_ && postProcess_.bloomExtract().imageView() != VK_NULL_HANDLE) {
            const VkExtent3D extent = postProcess_.bloomExtract().extent();
            ImGui::Text("Bloom extract: %u x %u, %s",
                        extent.width,
                        extent.height,
                        vkFormatName(postProcess_.bloomExtract().format()));
            drawRenderTargetPreview(postProcess_.bloomExtract().imageView(),
                                    postProcess_.sampler(),
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    extent.width,
                                    extent.height,
                                    previewSize,
                                    hdrPreviewExposure);
        }
        if (showRenderTargetBlurredBloom_) {
            if (postProcess_.bloomPing().imageView() != VK_NULL_HANDLE) {
                const VkExtent3D extent = postProcess_.bloomPing().extent();
                ImGui::Text("Bloom ping: %u x %u, %s",
                            extent.width,
                            extent.height,
                            vkFormatName(postProcess_.bloomPing().format()));
                drawRenderTargetPreview(postProcess_.bloomPing().imageView(),
                                        postProcess_.sampler(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        extent.width,
                                        extent.height,
                                        previewSize,
                                        hdrPreviewExposure);
            }
            if (postProcess_.bloomPong().imageView() != VK_NULL_HANDLE) {
                const VkExtent3D extent = postProcess_.bloomPong().extent();
                ImGui::Text("Bloom pong: %u x %u, %s",
                            extent.width,
                            extent.height,
                            vkFormatName(postProcess_.bloomPong().format()));
                drawRenderTargetPreview(postProcess_.bloomPong().imageView(),
                                        postProcess_.sampler(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        extent.width,
                                        extent.height,
                                        previewSize,
                                        hdrPreviewExposure);
            }
        }
    }

    if (showRenderTargetBloomMipChain_ && !postProcess_.bloomMipDownsampleImages().empty() &&
        ImGui::CollapsingHeader("Bloom Mip Chain", ImGuiTreeNodeFlags_DefaultOpen)) {
        const int maxMip = static_cast<int>(postProcess_.bloomMipDownsampleImages().size() - 1u);
        int selectedMip = static_cast<int>(std::min(
            selectedBloomMipDebugLevel_, static_cast<uint32_t>(postProcess_.bloomMipDownsampleImages().size() - 1u)));
        ImGui::SliderInt("Selected bloom mip", &selectedMip, 0, maxMip);
        selectedBloomMipDebugLevel_ = static_cast<uint32_t>(std::max(selectedMip, 0));

        const rhi::VulkanImage& downsampleImage = postProcess_.bloomMipDownsampleImages()[selectedBloomMipDebugLevel_];
        const VkExtent3D downsampleExtent = downsampleImage.extent();
        ImGui::Text("Downsample %u: %u x %u, %s",
                    selectedBloomMipDebugLevel_,
                    downsampleExtent.width,
                    downsampleExtent.height,
                    vkFormatName(downsampleImage.format()));
        drawRenderTargetPreview(downsampleImage.imageView(),
                                postProcess_.sampler(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                downsampleExtent.width,
                                downsampleExtent.height,
                                previewSize,
                                hdrPreviewExposure);

        if (selectedBloomMipDebugLevel_ < postProcess_.bloomMipUpsampleImages().size()) {
            const rhi::VulkanImage& upsampleImage = postProcess_.bloomMipUpsampleImages()[selectedBloomMipDebugLevel_];
            const VkExtent3D upsampleExtent = upsampleImage.extent();
            ImGui::Text("Upsample %u: %u x %u, %s",
                        selectedBloomMipDebugLevel_,
                        upsampleExtent.width,
                        upsampleExtent.height,
                        vkFormatName(upsampleImage.format()));
            drawRenderTargetPreview(upsampleImage.imageView(),
                                    postProcess_.sampler(),
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    upsampleExtent.width,
                                    upsampleExtent.height,
                                    previewSize,
                                    hdrPreviewExposure);
        }
    }

    if (depthPyramid_.image() != VK_NULL_HANDLE && !depthPyramid_.mipImageViews().empty() &&
        ImGui::CollapsingHeader("Depth Pyramid", ImGuiTreeNodeFlags_DefaultOpen)) {
        uint32_t selectedMipLevel = std::min(depthPyramid_.selectedDebugMip(),
                                             static_cast<uint32_t>(depthPyramid_.mipImageViews().size() - 1));
        int selectedMip = static_cast<int>(selectedMipLevel);
        ImGui::SliderInt("Selected mip", &selectedMip, 0, static_cast<int>(depthPyramid_.mipImageViews().size() - 1));
        selectedMipLevel = static_cast<uint32_t>(std::max(selectedMip, 0));
        depthPyramid_.setSelectedDebugMip(selectedMipLevel);
        const VkExtent2D extent = mipExtent(renderResolution_.extent(), selectedMipLevel);
        ImGui::Text("Dimensions: %u x %u", extent.width, extent.height);
        ImGui::Text("Format: %s", vkFormatName(depthPyramid_.format()));
        ImGui::Text("Layout: %s", imageLayoutName(depthPyramid_.layout()));
        ImGui::TextDisabled("Normal-Z max-depth reduction; white/far regions are intentionally hard to cull through.");
        drawRenderTargetPreview(depthPyramid_.mipImageViews()[selectedMipLevel],
                                depthPyramid_.sampler(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                extent.width,
                                extent.height,
                                previewSize,
                                1.0f);
    }

    if (showRenderTargetBrdfLut_ && brdfLutTexture_.valid() &&
        ImGui::CollapsingHeader("BRDF LUT", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Dimensions: %u x %u", brdfLutTexture_.width(), brdfLutTexture_.height());
        ImGui::Text("Format: %s", vkFormatName(brdfLutTexture_.format()));
        ImGui::TextDisabled("Data texture for split-sum IBL; linear UNORM, not sRGB.");
        drawRenderTargetPreview(brdfLutTexture_.imageView(),
                                brdfLutTexture_.sampler(),
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                brdfLutTexture_.width(),
                                brdfLutTexture_.height(),
                                previewSize,
                                1.0f);
    }

    if (showRenderTargetCsmCascades_ && ImGui::CollapsingHeader("CSM Cascades", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawCsmCascadeDebugUi(previewSize);
    }
}

void Renderer::drawCsmCascadeDebugUi(float previewSize)
{
    if (!shadowMap_.valid()) {
        ImGui::TextDisabled("CSM shadow map is unavailable.");
        return;
    }

    const uint32_t cascadeCount = std::min(activeCascadeCount(), shadowMap_.layerCount());
    if (cascadeCount == 0) {
        ImGui::TextDisabled("No active CSM cascades.");
        return;
    }

    int selectedCascade = static_cast<int>(std::min(debugUiSettings_.selectedCsmCascade, cascadeCount - 1));
    if (ImGui::SliderInt("Selected cascade", &selectedCascade, 0, static_cast<int>(cascadeCount - 1))) {
        debugUiSettings_.selectedCsmCascade = static_cast<uint32_t>(std::max(selectedCascade, 0));
    }
    debugUiSettings_.selectedCsmCascade = std::min(debugUiSettings_.selectedCsmCascade, cascadeCount - 1);

    const VkExtent2D shadowExtent = shadowMap_.extent();
    ImGui::Text("Shadow resolution: %u x %u", shadowExtent.width, shadowExtent.height);
    ImGui::Text("Texel snapping: %s", csmSettings_.enableTexelSnapping ? "enabled" : "disabled");
    ImGui::Text("GPU shadow culling: %s", isGpuShadowCullingActive() ? "active" : "inactive");
    if (isGpuShadowCullingActive()) {
        uint32_t gpuVisibleShadowDraws = 0;
        if (readGpuShadowVisibleCount(currentFrame_, gpuVisibleShadowDraws)) {
            ImGui::Text("Latest aggregate GPU visible shadow draws: %u", gpuVisibleShadowDraws);
        } else {
            ImGui::TextDisabled("Aggregate GPU shadow readback pending.");
        }
        ImGui::TextDisabled("Per-cascade draw and batch counts below are CPU frustum estimates.");
    }

    constexpr ImGuiTableFlags tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                           ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("CsmCascadeMetadata", 7, tableFlags)) {
        ImGui::TableSetupColumn("Cascade", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Split range");
        ImGui::TableSetupColumn("Layer", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Resolution");
        ImGui::TableSetupColumn("Coverage");
        ImGui::TableSetupColumn("Visible draws");
        ImGui::TableSetupColumn("Batches");
        ImGui::TableHeadersRow();

        for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            const CascadeFrameData& cascade = frameCascades_[cascadeIndex];
            const float range = std::max(cascade.farDepth - cascade.nearDepth, 0.0f);
            const float coverage = csmSettings_.shadowDistance > 0.0f
                                       ? (range / csmSettings_.shadowDistance) * 100.0f
                                       : 0.0f;

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%u", cascadeIndex);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f - %.3f", cascade.nearDepth, cascade.farDepth);
            ImGui::TableNextColumn();
            ImGui::Text("%u", cascadeIndex);
            ImGui::TableNextColumn();
            ImGui::Text("%u x %u", shadowExtent.width, shadowExtent.height);
            ImGui::TableNextColumn();
            ImGui::Text("%.1f%%", coverage);
            ImGui::TableNextColumn();
            ImGui::Text("%u", shadowVisibleDrawItemsPerCascade_[cascadeIndex]);
            ImGui::TableNextColumn();
            ImGui::Text("%u", shadowBatchCountPerCascade_[cascadeIndex]);
        }

        ImGui::EndTable();
    }

    const uint32_t selected = debugUiSettings_.selectedCsmCascade;
    const CascadeFrameData& cascade = frameCascades_[selected];
    ImGui::Separator();
    ImGui::Text("Selected cascade %u", selected);
    ImGui::Text("Split depth: %.3f", cascade.splitDepth);
    ImGui::Text("Split range: %.3f - %.3f", cascade.nearDepth, cascade.farDepth);
    ImGui::Text("Layer index: %u", selected);
    ImGui::Text("Visible shadow draw count: %u", shadowVisibleDrawItemsPerCascade_[selected]);
    ImGui::Text("Shadow batch count: %u", shadowBatchCountPerCascade_[selected]);
    if (ImGui::TreeNode("Light view-projection matrix")) {
        for (int row = 0; row < 4; ++row) {
            const std::string matrixRow = formatMatrixRow(cascade.lightViewProjection, row);
            ImGui::TextUnformatted(matrixRow.c_str());
        }
        ImGui::TreePop();
    }

    ImGui::TextDisabled("Depth preview uses the existing per-layer 2D shadow image view sampled as raw depth.");
    drawRenderTargetPreview(shadowMap_.layerImageView(selected),
                            shadowMap_.sampler(),
                            VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                            shadowExtent.width,
                            shadowExtent.height,
                            previewSize,
                            1.0f);

    if (ImGui::TreeNodeEx("Cascade thumbnails", ImGuiTreeNodeFlags_DefaultOpen)) {
        const float thumbnailSize = std::max(64.0f, previewSize * 0.45f);
        for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            ImGui::PushID(static_cast<int>(cascadeIndex));
            ImGui::BeginGroup();
            ImGui::Text("Cascade %u", cascadeIndex);
            drawRenderTargetPreview(shadowMap_.layerImageView(cascadeIndex),
                                    shadowMap_.sampler(),
                                    VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                                    shadowExtent.width,
                                    shadowExtent.height,
                                    thumbnailSize,
                                    1.0f);
            if (ImGui::IsItemClicked()) {
                debugUiSettings_.selectedCsmCascade = cascadeIndex;
            }
            ImGui::EndGroup();
            ImGui::PopID();
            if (cascadeIndex + 1 < cascadeCount) {
                ImGui::SameLine();
            }
        }
        ImGui::TreePop();
    }
}

void Renderer::drawGlobalTextureMetadata()
{
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("GlobalTextureMetadata", 6, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Resource");
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupColumn("Dimensions");
    ImGui::TableSetupColumn("Mip/layers");
    ImGui::TableSetupColumn("Format");
    ImGui::TableSetupColumn("Layout / source");
    ImGui::TableHeadersRow();

    const auto addRow = [](const char* name,
                           const char* type,
                           const std::string& dimensions,
                           const std::string& mipLayers,
                           VkFormat format,
                           const std::string& layoutOrSource) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(name);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(type);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(dimensions.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(mipLayers.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(vkFormatName(format));
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", layoutOrSource.c_str());
    };

    if (shadowMap_.valid()) {
        const VkExtent2D extent = shadowMap_.extent();
        addRow("Cascaded shadow map array",
               "depth 2D array",
               std::to_string(extent.width) + " x " + std::to_string(extent.height),
               std::to_string(shadowMap_.layerCount()) + " layers",
               shadowMap_.format(),
               imageLayoutName(shadowMap_.layout()));
    }
    if (diffuseIrradianceMap_.valid()) {
        addRow("Diffuse irradiance cubemap",
               "cubemap",
               std::to_string(diffuseIrradianceMap_.faceSize()) + " x " +
                   std::to_string(diffuseIrradianceMap_.faceSize()) + " x 6",
               std::to_string(diffuseIrradianceMap_.mipLevels()) + " mips",
               diffuseIrradianceMap_.format(),
               hdrEnvironmentLoaded_ ? "HDR-derived" : "procedural fallback");
    }
    if (prefilteredEnvironmentMap_.valid()) {
        addRow("Prefiltered specular cubemap",
               "cubemap",
               std::to_string(prefilteredEnvironmentMap_.faceSize()) + " x " +
                   std::to_string(prefilteredEnvironmentMap_.faceSize()) + " x 6",
               std::to_string(prefilteredEnvironmentMap_.mipLevels()) + " mips",
               prefilteredEnvironmentMap_.format(),
               hdrEnvironmentLoaded_ ? "HDR-derived" : "procedural fallback");
    }
    if (brdfLutTexture_.valid()) {
        addRow("BRDF LUT",
               "2D LUT",
               std::to_string(brdfLutTexture_.width()) + " x " + std::to_string(brdfLutTexture_.height()),
               "1 mip",
               brdfLutTexture_.format(),
               imageLayoutName(brdfLutTexture_.layout()));
    }
    if (postProcess_.sceneColor().image() != VK_NULL_HANDLE) {
        const VkExtent3D extent = postProcess_.sceneColor().extent();
        addRow("SceneColor",
               "HDR render target",
               std::to_string(extent.width) + " x " + std::to_string(extent.height),
               "1 mip",
               postProcess_.sceneColor().format(),
               imageLayoutName(postProcess_.sceneColorLayout()));
    }
    for (uint32_t historyIndex = 0; historyIndex < kTaaHistoryCount; ++historyIndex) {
        if (postProcess_.taaHistoryImages()[historyIndex].image() != VK_NULL_HANDLE) {
            const VkExtent3D extent = postProcess_.taaHistoryImages()[historyIndex].extent();
            const std::string name = "TAAHistory" + std::to_string(historyIndex);
            addRow(name.c_str(),
                   "persistent HDR history",
                   std::to_string(extent.width) + " x " + std::to_string(extent.height),
                   "1 mip",
                   postProcess_.taaHistoryImages()[historyIndex].format(),
                   imageLayoutName(postProcess_.taaHistoryLayouts()[historyIndex]));
        }
    }
    if (postProcess_.bloomExtract().image() != VK_NULL_HANDLE) {
        const VkExtent3D extent = postProcess_.bloomExtract().extent();
        addRow("BloomExtract",
               "post-process render target",
               std::to_string(extent.width) + " x " + std::to_string(extent.height),
               "1 mip",
               postProcess_.bloomExtract().format(),
               imageLayoutName(postProcess_.bloomExtractLayout()));
    }
    if (postProcess_.bloomPing().image() != VK_NULL_HANDLE) {
        const VkExtent3D extent = postProcess_.bloomPing().extent();
        addRow("BloomPing",
               "post-process render target",
               std::to_string(extent.width) + " x " + std::to_string(extent.height),
               "1 mip",
               postProcess_.bloomPing().format(),
               imageLayoutName(postProcess_.bloomPingLayout()));
    }
    if (postProcess_.bloomPong().image() != VK_NULL_HANDLE) {
        const VkExtent3D extent = postProcess_.bloomPong().extent();
        addRow("BloomPong",
               "post-process render target",
               std::to_string(extent.width) + " x " + std::to_string(extent.height),
               "1 mip",
               postProcess_.bloomPong().format(),
               imageLayoutName(postProcess_.bloomPongLayout()));
    }

    ImGui::EndTable();
}

void Renderer::drawGpuTimingDebugUi()
{
    bool profilerEnabled = gpuProfilerEnabled_ && gpuProfiler_.available();
    if (ImGui::Checkbox("GPU profiler enabled", &profilerEnabled)) {
        gpuProfilerEnabled_ = profilerEnabled;
        gpuProfiler_.setEnabled(gpuProfilerEnabled_);
        if (!gpuProfilerEnabled_) {
            latestGpuProfilerResults_ = {};
        }
    }

    if (!gpuProfiler_.available()) {
        ImGui::TextDisabled("GPU profiler unavailable.");
        if (!gpuProfiler_.unavailableReason().empty()) {
            ImGui::TextWrapped("Reason: %s", gpuProfiler_.unavailableReason().c_str());
        }
        ImGui::Text("CPU frame delta: %.3f ms (%.1f FPS)", cpuFrameDeltaMs_, cpuFps_);
        return;
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset averages")) {
        resetGpuProfilerHistory();
        latestGpuProfilerResults_ = {};
    }

    if (!gpuProfilerEnabled_) {
        ImGui::TextDisabled("GPU profiler disabled.");
        ImGui::Text("CPU frame delta: %.3f ms (%.1f FPS)", cpuFrameDeltaMs_, cpuFps_);
        return;
    }

    if (!latestGpuProfilerResults_.valid) {
        ImGui::TextDisabled("GPU profiler is waiting for the first completed frame.");
        ImGui::Text("CPU frame delta: %.3f ms (%.1f FPS)", cpuFrameDeltaMs_, cpuFps_);
        return;
    }

    ImGui::Text("GPU frame total: %.3f ms", gpuFrameTimeHistory_.latest());
    ImGui::Text("CPU frame delta: %.3f ms (%.1f FPS)", cpuFrameDeltaMs_, cpuFps_);
    ImGui::Checkbox("Parallel frame prep (JobSystem)", &parallelFramePrepEnabled_);
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu workers)", jobSystem_.threadCount());
    if (!framePrepCpuHistory_.empty()) {
        ImGui::Text("Frame prep CPU: %.3f ms (avg %.3f, max %.3f)",
                    framePrepCpuHistory_.latest(),
                    framePrepCpuHistory_.average(),
                    framePrepCpuHistory_.max());
    }
    ImGui::Text("Timestamp queries: %u / %u",
                latestGpuProfilerResults_.queryCount,
                latestGpuProfilerResults_.maxQueryCount);
    if (latestGpuProfilerResults_.queryLimitExceeded) {
        ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.25f, 1.0f), "Warning: profiler query capacity exceeded.");
    }
    ImGui::TextDisabled("Timings are read back after frame-fence completion. Nested scopes are shown in execution order.");

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("GpuTimingHistoryTable", 5, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Range");
    ImGui::TableSetupColumn("Current ms", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Avg ms", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Max ms", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("History");
    ImGui::TableHeadersRow();

    drawTimingHistoryRow("GPU frame total", gpuFrameTimeHistory_);
    for (const renderer::GpuProfiler::ScopeResult& scope : latestGpuProfilerResults_.scopes) {
        if (const DebugHistory* history = gpuTimingHistoryForPass(scope.name)) {
            drawTimingHistoryRow(scope.name.c_str(), *history);
        }
    }

    ImGui::EndTable();
}

void Renderer::drawCullingDebugUi()
{
    const CullingDebugSnapshot snapshot = cullingDebugSnapshot(currentFrame_);
    const float occlusionRejectionPercent =
        snapshot.totalDrawItems > 0
            ? (100.0f * static_cast<float>(snapshot.occlusionCulledDrawItems) /
               static_cast<float>(snapshot.totalDrawItems))
            : 0.0f;
    ImGui::Text("Occlusion test scene: %s", snapshot.occlusionTestSceneActive ? "active" : "inactive");
    ImGui::Text("Total objects: %u", snapshot.totalObjects);
    ImGui::Text("Total draw items: %u", snapshot.totalDrawItems);

    // Render-bucket split. Opaque and Mask share the main pipeline (the alpha test
    // is data-driven through ObjectFrameData::materialParams.w) but take different
    // shadow pipelines, and batches never straddle a bucket, so this line also
    // explains any jump in the batch count.
    std::array<uint32_t, kRenderBucketCount> bucketDrawItemCounts{};
    for (const DrawItem& drawItem : allDrawItems_) {
        bucketDrawItemCounts[static_cast<size_t>(drawItem.bucket)] += 1;
    }
    ImGui::Text("Render buckets: opaque %u, mask %u, blend %u",
                bucketDrawItemCounts[static_cast<size_t>(RenderBucket::Opaque)],
                bucketDrawItemCounts[static_cast<size_t>(RenderBucket::Mask)],
                bucketDrawItemCounts[static_cast<size_t>(RenderBucket::Blend)]);
    ImGui::Text("Alpha-tested shadow pipeline: %s",
                maskedShadowPipeline_.pipeline() != VK_NULL_HANDLE ? "active" : "unavailable (bindless required)");

    ImGui::Text("Visible after culling: %u", snapshot.visibleDrawItems);
    ImGui::Text("Culled draw items: %u", snapshot.culledDrawItems);
    ImGui::Text("Frustum culled: %u", snapshot.frustumCulledDrawItems);
    ImGui::Text("Occlusion culled: %u", snapshot.occlusionCulledDrawItems);
    ImGui::Text("Phase-2 rescued: %u", snapshot.phase2RescuedDrawItems);
    ImGui::Text("Occlusion rejection: %.1f%%", occlusionRejectionPercent);
    ImGui::Text("Shadow draw items: %u", snapshot.shadowDrawItems);
    ImGui::Text("Visible shadow draw items: %u", snapshot.visibleShadowDrawItems);
    ImGui::Text("Culled shadow draw items: %u", snapshot.culledShadowDrawItems);
    ImGui::Text("Shadow batches: %zu", snapshot.shadowBatchCount);
    ImGui::Text("GPU culling: %s", snapshot.gpuCulling ? "enabled" : "disabled");
    ImGui::Text("GPU occlusion culling: %s", snapshot.gpuOcclusionCulling ? "enabled" : "disabled");
    ImGui::Text("Two-phase occlusion: %s", snapshot.twoPhaseOcclusion ? "active" : "inactive");
    const char* depthPyramidStatus = snapshot.depthPyramidBuildAvailable
                                         ? (snapshot.depthPyramidValid ? "valid" : "invalid/warming up")
                                         : "unavailable";
    ImGui::Text("Depth pyramid: %s, %u mip(s)", depthPyramidStatus, snapshot.depthPyramidMipCount);
    ImGui::Text("Previous-frame depth: %s", snapshot.previousFrameDepthValid ? "valid" : "invalid/warming up");
    ImGui::Text("Occlusion settings: enabled=%s bias=%.4f near-skip=%.2f max-coverage=%.2f min-pixels=%.1f",
                useGpuOcclusionCulling_ ? "true" : "false",
                gpuOcclusionDepthBias_,
                gpuOcclusionNearDisableDistance_,
                gpuOcclusionMaxScreenCoverage_,
                gpuOcclusionMinScreenPixels_);
    ImGui::Text("GPU shadow culling: %s", snapshot.gpuShadowCulling ? "enabled" : "disabled");

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("CullingHistoryTable", 5, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Current", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("History");
    ImGui::TableHeadersRow();

    drawScalarHistoryRow("Visible main draw items", visibleMainDrawItemsHistory_, "%.0f");
    drawScalarHistoryRow("Culled main draw items", culledMainDrawItemsHistory_, "%.0f");
    drawScalarHistoryRow("Visible shadow draw items", visibleShadowDrawItemsHistory_, "%.0f");
    drawScalarHistoryRow("Culled shadow draw items", culledShadowDrawItemsHistory_, "%.0f");

    ImGui::EndTable();
}

void Renderer::drawExposureDebugUi()
{
    const ExposureMode mode = exposureModeValue(toneMappingSettings_.enableAutoExposure ? toneMappingSettings_.exposureMode
                                                                                        : 0);
    ImGui::Text("Current exposure: %.4f", postProcess_.currentToneMappingExposure());
    ImGui::Text("Log-average luminance: %.4f", averageLuminance_);
    ImGui::Text("Histogram clipped luminance: %.4f", histogramClippedLuminance_);
    ImGui::Text("Exposure mode: %s", exposureModeName(mode).data());
    ImGui::Text("Composite exposure source: %s",
                postProcess_.isGpuExposureActive() ? "GPU exposure buffer" : "push constant");

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("ExposureHistoryTable", 5, flags)) {
        return;
    }

    ImGui::TableSetupColumn("Metric");
    ImGui::TableSetupColumn("Current", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableSetupColumn("History");
    ImGui::TableHeadersRow();

    drawScalarHistoryRow("Exposure", exposureHistory_, "%.4f");
    drawScalarHistoryRow("Log-average luminance", averageLuminanceHistory_, "%.4f");
    drawScalarHistoryRow("Histogram clipped luminance", histogramClippedLuminanceHistory_, "%.4f");

    ImGui::EndTable();
}

void Renderer::drawTimingHistoryRow(const char* label, const DebugHistory& history) const
{
    ImGui::PushID(label);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", history.latest());
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", history.average());
    ImGui::TableNextColumn();
    ImGui::Text("%.3f", history.max());
    ImGui::TableNextColumn();
    drawHistoryPlot(history, 42.0f);
    ImGui::PopID();
}

void Renderer::drawScalarHistoryRow(const char* label, const DebugHistory& history, const char* valueFormat) const
{
    ImGui::PushID(label);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::Text(valueFormat, history.latest());
    ImGui::TableNextColumn();
    ImGui::Text(valueFormat, history.average());
    ImGui::TableNextColumn();
    ImGui::Text(valueFormat, history.max());
    ImGui::TableNextColumn();
    drawHistoryPlot(history, 42.0f);
    ImGui::PopID();
}

void Renderer::drawHistoryPlot(const DebugHistory& history, float height) const
{
    if (history.empty()) {
        ImGui::TextDisabled("waiting");
        return;
    }

    std::array<float, kDebugHistoryCapacity> values{};
    const size_t sampleCount = history.copyChronological(values);
    const float scaleMax = std::max(history.max(), 0.001f);
    ImGui::PlotLines("##history",
                     values.data(),
                     static_cast<int>(sampleCount),
                     0,
                     nullptr,
                     0.0f,
                     scaleMax,
                     ImVec2(180.0f, height));
}

} // namespace ve
