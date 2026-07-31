#pragma once

#include "assets/AssetManager.h"
#include "core/JobSystem.h"
#include "renderer/BindlessTextureHeap.h"
#include "renderer/BuiltinTextureFactory.h"
#include "renderer/EditorCamera.h"
#include "renderer/Bounds.h"
#include "renderer/Camera.h"
#include "renderer/CascadeMath.h"
#include "renderer/ClusteredLighting.h"
#include "renderer/DepthPyramid.h"
#include "renderer/GpuCulling.h"
#include "renderer/FrameResources.h"
#include "renderer/GpuProfiler.h"
#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/PostProcessStack.h"
#include "renderer/RenderGraph.h"
#include "renderer/RenderObject.h"
#include "renderer/RuntimeSettings.h"
#include "renderer/SceneBuilder.h"
#include "renderer/GroundTruthAmbientOcclusion.h"
#include "renderer/ScreenSpaceReflections.h"
#include "renderer/ScreenshotCapture.h"
#include "renderer/SkinnedMesh.h"
#include "rhi/VulkanAsyncCompute.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanBrdfLut.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanComputePipeline.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDescriptor.h"
#include "rhi/VulkanEnvironmentMap.h"
#include "rhi/VulkanImage.h"
#include "rhi/VulkanPipeline.h"
#include "rhi/VulkanShadowMap.h"
#include "rhi/VulkanSwapchain.h"
#include "rhi/VulkanSync.h"
#include "rhi/VulkanTexture.h"
#include "ui/ImGuiLayer.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

typedef union SDL_Event SDL_Event;

namespace ve {

class Window;

class Renderer final {
public:
    explicit Renderer(Window& window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void drawFrame();
    void handleEvent(const SDL_Event& event);
    void waitIdle();

private:
    static constexpr uint32_t kMaxShadowCascades = renderer::kMaxShadowCascades;
    static constexpr size_t kDebugHistoryCapacity = 240;
    static constexpr size_t kInvalidRenderObjectIndex = std::numeric_limits<size_t>::max();

    // Which transform handle the viewport gizmo manipulates (mapped to ImGuizmo
    // operations in the .cpp so this header stays free of the ImGuizmo include).
    enum class GizmoOperation {
        Translate,
        Rotate,
        Scale
    };

    struct ShadowSettings {
        uint32_t resolution = 2048;
        bool enablePcf = true;
        int pcfRadius = 1;
        float rasterDepthBiasConstantFactor = 1.25f;
        float rasterDepthBiasSlopeFactor = 1.75f;
    };

    // SsaoSettings now lives in RuntimeSettings.h alongside the other
    // post-process settings structs so PostProcessStack and Renderer can share
    // the type without a circular include. The ssaoSettings_ member stays here.

    // Pure cascade fit data now lives in renderer::ShadowCascade (CascadeMath.h)
    // so the math is shared with the GPU-independent unit tests. Aliased here to
    // keep the many frameCascades_ usages in the .cpp unchanged.
    using CascadeFrameData = renderer::ShadowCascade;

    struct DrawItem {
        const renderer::Mesh* mesh = nullptr;
        const renderer::Material* material = nullptr;
        uint32_t objectIndex = 0;
        uint32_t submeshIndex = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        int32_t vertexOffset = 0;
        uint32_t frameDataIndex = 0;
    };

    struct MeshDrawBatch {
        const renderer::Mesh* mesh = nullptr;
        uint32_t beginDrawItem = 0;
        uint32_t drawItemCount = 0;
        uint32_t compactedCommandOffset = 0;
        uint32_t visibleCountOffset = 0;
    };

    struct CullingStats {
        size_t totalObjects = 0;
        size_t visibleObjects = 0;
        size_t culledObjects = 0;
        size_t totalDrawItems = 0;
        size_t batchCount = 0;
        size_t commandCount = 0;
        bool gpuCulling = false;
    };

    // GpuCullCounters now lives in renderer/GpuCulling.h alongside the subsystem.

    struct ShadowCullingStats {
        size_t cascadeCount = 0;
        size_t totalDrawItems = 0;
        size_t visibleDrawItems = 0;
        size_t culledDrawItems = 0;
        size_t batchCount = 0;
        bool gpuCulling = false;
        bool indirectDrawing = false;
    };

    struct DebugHistory {
        std::array<float, kDebugHistoryCapacity> samples{};
        size_t cursor = 0;
        size_t count = 0;

        void push(float value);
        [[nodiscard]] float latest() const;
        [[nodiscard]] float average() const;
        [[nodiscard]] float max() const;
        [[nodiscard]] size_t copyChronological(std::array<float, kDebugHistoryCapacity>& output) const;
        [[nodiscard]] bool empty() const { return count == 0; }
    };

    struct GpuTimingHistory {
        std::string name;
        DebugHistory history;
    };

    enum class RuntimeSettingsApplyMode {
        Startup,
        Runtime
    };

    struct CullingDebugSnapshot {
        uint32_t totalObjects = 0;
        uint32_t totalDrawItems = 0;
        uint32_t visibleDrawItems = 0;
        uint32_t culledDrawItems = 0;
        uint32_t frustumCulledDrawItems = 0;
        uint32_t occlusionCulledDrawItems = 0;
        uint32_t shadowDrawItems = 0;
        uint32_t visibleShadowDrawItems = 0;
        uint32_t culledShadowDrawItems = 0;
        size_t shadowBatchCount = 0;
        bool gpuCulling = false;
        bool gpuOcclusionCulling = false;
        bool gpuShadowCulling = false;
        bool occlusionTestSceneActive = false;
        bool depthPyramidBuildAvailable = false;
        bool depthPyramidValid = false;
        bool previousFrameDepthValid = false;
        uint32_t depthPyramidMipCount = 0;
        uint32_t phase2RescuedDrawItems = 0;
        bool twoPhaseOcclusion = false;
    };

    struct ObjectDrawDebugInfo {
        size_t drawItemCount = 0;
        size_t visibleMainDrawItemCount = 0;
        size_t visibleShadowDrawItemCount = 0;
        size_t shadowVisibleCascadeCount = 0;
        uint32_t firstObjectDataIndex = 0;
        bool hasObjectDataIndex = false;
    };

    struct DirectionalLightSettings {
        glm::vec3 direction{0.35f, -0.65f, -0.55f};
        glm::vec3 color{0.85f, 0.85f, 0.85f};
        float intensity = 1.0f;
    };

    struct PortfolioCaptureSavedState {
        renderer::Camera camera;
        ToneMappingSettings toneMapping;
        BloomSettings bloom;
        CsmSettings csm;
        float currentExposure = 1.0f;
        bool valid = false;
    };

    void createMaterialDescriptorSetLayout();
    void createBindlessMaterialTextureHeap();
    void createSkyboxDescriptorSetLayout();
    // Resets the Renderer-owned jittered view-projection matrices, then delegates
    // the TAA history/jitter reset to PostProcessStack.
    void invalidateTaaHistory();
    void createDepthPyramidDescriptorSetLayout();
    // Recreates PostProcessStack resources plus the interleaved depth-pyramid and
    // ImGui render-target-preview reset the former monolithic method performed.
    void recreatePostProcessResources();
    void createDepthPyramidResources();
    void destroyDepthPyramidResources();
    // Thin wrappers over the gpuCulling_ subsystem (resource lifetime + depth
    // pyramid descriptor rebind); the GPU-culling implementation lives in
    // renderer::GpuCulling.
    void updateGpuCullingDepthPyramidDescriptors();
    void invalidateDepthPyramid();
    void createGpuCullingResources();
    void destroyGpuCullingResources();
    void createShadowMap();
    void createPipeline();
    // createPipeline() helpers (see Renderer.cpp); each builds one pipeline group,
    // a verbatim slice of the former monolithic function.
    void createMainGraphicsPipeline();
    void createSkinnedPipeline();
    void createSkyboxPipeline();
    void createShadowPipeline();
    void createComputePipelines();
    void createScene();
    // createScene() helpers (see Renderer.cpp): reset scene state, build the shared
    // meshes/textures/material, try the glTF scene, else the built-in cube fallback.
    // The CPU-side scene-object layout itself lives in renderer::SceneBuilder.
    void resetSceneState();
    void createSceneSharedResources();
    [[nodiscard]] bool tryLoadGltfScene();
    // Constructs a SceneBuilder borrowing the renderer's shared meshes, material
    // array, and debug-id allocator. Cheap; call per scene-build operation.
    [[nodiscard]] renderer::SceneBuilder makeSceneBuilder();
    void resetPortfolioShowcaseObjectsToPreset();
    [[nodiscard]] bool currentFrameHasPortfolioShowcaseDrawItems() const;
    [[nodiscard]] bool ensurePortfolioShowcaseSceneReady();
    void resetOcclusionTestSceneToPreset();
    [[nodiscard]] uint32_t allocateRenderObjectDebugId();
    void createEnvironmentMap();
    void createDiffuseIrradianceMap();
    void createPrefilteredEnvironmentMap();
    void createBrdfLutTexture();
    void createMaterial();
    // createMaterial() helpers (see Renderer.cpp): build the built-in checkerboard
    // variants and the portfolio-showcase variants (after shared bindless fallback
    // registration in createMaterial).
    void createBuiltInMaterialVariants();
    void createPortfolioMaterialVariants();
    [[nodiscard]] renderer::Material createMaterialFromAsset(const assets::MaterialAsset& materialAsset,
                                                             const rhi::VulkanTexture& baseColorFallback,
                                                             const rhi::VulkanTexture& normalFallback,
                                                             const rhi::VulkanTexture& metallicRoughnessFallback,
                                                             float multiScatterStrength,
                                                             renderer::MaterialSource fallbackSource);
    void assignBindlessTextureIndices(renderer::Material& material);
    void createMaterialDescriptorSet(renderer::Material& material);
    [[nodiscard]] const rhi::VulkanTexture* loadMaterialAssetTextureOrFallback(
        const std::filesystem::path& materialPath,
        const std::filesystem::path& texturePath,
        rhi::TextureColorSpace colorSpace,
        std::string_view slotName,
        const rhi::VulkanTexture& fallbackTexture,
        bool& fallbackUsed);
    [[nodiscard]] assets::MaterialAsset runtimeMaterialToAsset(const renderer::Material& material) const;
    [[nodiscard]] std::filesystem::path makeNewMaterialAssetPath(const renderer::Material& material) const;
    bool saveMaterialAssetFromUi(renderer::Material& material);
    bool reloadMaterialAssetFromUi(renderer::Material& material);
    void createImportedGltfTextures(const std::vector<renderer::GltfTextureInfo>& textureInfos,
                                    const std::vector<renderer::GltfMaterialInfo>& materialInfos);
    void createImportedGltfMaterials(const std::vector<renderer::GltfMaterialInfo>& materialInfos);
    void createSkyboxDescriptorSet();
    void createObjectFrameDataBuffers();
    void createIndirectDrawBuffers();
    void createShadowIndirectDrawBuffers();
    void updateCascades(float aspectRatio);
    void buildFrameMeshLodTable();
    // Shared by the main and shadow cull-input updates so both dispatches see the
    // same camera, viewport, and LOD settings regardless of upload order.
    void uploadGpuCullFrameParams(uint32_t frameIndex, bool occlusionEnabledThisFrame);
    void updateGpuCullInputBuffer(uint32_t frameIndex);
    void updateGpuShadowCullInputBuffer(uint32_t frameIndex);
    void updateFrameData(uint32_t frameIndex);
    // Captures this frame's view-projection and per-object model matrices as the
    // "previous frame" inputs for next frame's motion vectors.
    void capturePreviousFrameMatrices();
    // Recomputes every active object's world AABB once per frame into
    // frameWorldBounds_ so visibility, shadow cascades, and GPU-cull input
    // builds share it instead of re-deriving the model matrix per use.
    void updateFrameWorldBounds();
    // Runs body(begin, end) over [0, count): chunked across the JobSystem when
    // parallel frame prep is enabled, inline on the calling thread otherwise.
    // Callers must not nest framePrepParallelFor inside a parallel body.
    void framePrepParallelFor(size_t count, const std::function<void(size_t, size_t)>& body);
    // updateFrameData() helpers (see Renderer.cpp); each is a verbatim slice of the
    // former monolithic function, kept private and behaviour-preserving.
    void resetFrameStateForEmptyScene(uint32_t frameIndex);
    bool updateAnimatedTransforms(float elapsedSeconds);
    void resetGpuCullFrameCounters(uint32_t frameIndex);
    void buildShadowFrameData(uint32_t frameIndex);
    void buildMainCullingFrameData(uint32_t frameIndex, const renderer::Frustum& cameraFrustum);
    void uploadObjectFrameData(uint32_t frameIndex);
    void buildDrawItems();
    void buildVisibleDrawItems(const renderer::Frustum& frustum);
    void buildMeshDrawBatches();
    void buildShadowDrawItems(uint32_t cascadeIndex, const renderer::Frustum& lightFrustum);
    void buildShadowMeshDrawBatches();
    void buildMeshDrawBatchesForItems(const std::vector<DrawItem>& drawItems,
                                      std::vector<MeshDrawBatch>& batches) const;
    bool appendDrawItemsForObject(uint32_t objectIndex, std::vector<DrawItem>& drawItems) const;
    [[nodiscard]] bool isRenderObjectActive(const renderer::RenderObject& object) const;
    void updateIndirectDrawBuffer(uint32_t frameIndex);
    void updateShadowIndirectDrawBuffer(uint32_t frameIndex);
    [[nodiscard]] const renderer::Material* resolveMaterial(const renderer::RenderObject& object,
                                                            const renderer::MeshPrimitive* primitive) const;
    void recreateSwapchain();
    void recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void recordGpuCullingCommands(VkCommandBuffer commandBuffer);
    void recordGpuShadowCullingCommands(VkCommandBuffer commandBuffer, uint32_t cascadeIndex);
    void ensureDepthPyramidShaderReadLayout(VkCommandBuffer commandBuffer);
    void recordDepthPyramidCommands(VkCommandBuffer commandBuffer, bool midFrame = false);
    [[nodiscard]] renderer::RenderGraphFrameResources renderGraphFrameResources();
    bool readGpuVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    bool readGpuCullCounters(uint32_t frameIndex, renderer::GpuCullCounters& counters);
    bool readGpuShadowVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    [[nodiscard]] bool isGpuCullingActive() const;
    [[nodiscard]] bool isGpuOcclusionCullingActive() const;
    [[nodiscard]] bool isGpuShadowCullingActive() const;
    [[nodiscard]] bool isBindlessMaterialTextureActive() const;
    [[nodiscard]] bool isMainPassMultiDrawIndirectActive() const;
    [[nodiscard]] bool isMainPassIndirectCountSupported() const;
    [[nodiscard]] bool isFrameIndirectCountPathActive(uint32_t frameIndex) const;
    [[nodiscard]] bool isShadowIndirectCountSupported() const;
    [[nodiscard]] bool isShadowIndirectCountPathActive(uint32_t frameIndex) const;
    [[nodiscard]] bool isShadowIndirectActive() const;
    [[nodiscard]] uint32_t activeCascadeCount() const;
    [[nodiscard]] VkDescriptorSet globalMaterialDescriptorSet() const;
    void nameTextureResources(const rhi::VulkanTexture& texture, std::string_view name) const;
    void nameEnvironmentMapResources(const rhi::VulkanEnvironmentMap& environmentMap, std::string_view name) const;
    void nameBrdfLutResources(const rhi::VulkanBrdfLut& brdfLut, std::string_view name) const;
    void tryPrintExposureStats();
    void tryPrintGpuTimings(uint32_t frameIndex);
    void loadRuntimeSettingsAtStartup();
    void applyRuntimeSettings(const RuntimeSettings& settings, RuntimeSettingsApplyMode mode);
    [[nodiscard]] RuntimeSettings captureRuntimeSettings() const;
    void saveRuntimeSettingsFromUi();
    void reloadRuntimeSettingsFromUi();
    void resetRuntimeSettingsToDefaults();
    void saveSceneFromUi();
    void loadSceneFromUi();
    void resetCameraToDefault();
    void resetCameraToPortfolioPreset();
    void resetCameraToOcclusionTestPreset();
    void resetDirectionalLightToDefault();
    // Regenerates clusteredLighting_ each frame from the demo-light controls: a
    // scattered, orbiting swarm of colored point lights plus an overhead spot.
    // Driving the count up is how the clustered path is stress-tested.
    void updateDemoLights(float elapsedSeconds);
    [[nodiscard]] glm::vec4 activeDirectionalLightDirection() const;
    [[nodiscard]] glm::vec4 activeDirectionalLightColor() const;
    void loadOcclusionTestScene();
    void enableOcclusionTestSettings();
    [[nodiscard]] bool previousFrameDepthValidForOcclusion() const;
    // Editor viewport interaction: free-fly/orbit camera, click-to-select picking,
    // and ImGuizmo transform handles. See EditorCamera and Bounds::intersectRay.
    void updateEditorCamera(float deltaSeconds);
    void pickObjectAtCursor(float pixelX, float pixelY);
    [[nodiscard]] renderer::Ray screenPointToRay(float pixelX, float pixelY) const;
    void drawViewportGizmo();
    void buildDebugUi();
    // buildDebugUi() panel sections (see RendererDebugUi.cpp); each owns its
    // CollapsingHeader, matching the existing drawXxxDebugUi pattern.
    void drawDebugViewToggles();
    void drawControlsDebugUi();
    void drawToneMappingDebugUi();
    void drawBloomDebugUi();
    void drawSsaoDebugUi();
    void drawCsmSettingsDebugUi();
    void drawLightsDebugUi();
    void drawSkeletalAnimationDebugUi();
    void drawGpuCullingDebugUi();
    void drawEnvironmentDebugUi();
    void drawScenePresetDebugUi();
    void drawPortfolioCaptureDebugUi();
    void drawRuntimeSettingsDebugUi();
    void drawRenderGraphDebugUi();
    void drawSceneEditingDebugUi();
    void drawCameraLightEditorDebugUi();
    void drawSceneHierarchyDebugUi();
    void drawSelectedRenderObjectInspector(uint32_t objectIndex);
    void drawMaterialInspectorDebugUi();
    void drawTextureDebugUi();
    void drawRenderTargetDebugUi();
    // drawRenderTargetDebugUi() sections (see RendererDebugUi.cpp).
    void drawRenderTargetMetadataTable();
    void drawRenderTargetPreviews();
    void drawMaterialDebugSection(const renderer::Material* material,
                                  bool includeTextureSummary,
                                  renderer::Material* editableMaterial = nullptr);
    void drawMaterialTextureSlotDebugUi(const char* slotName,
                                        const char* semantic,
                                        const rhi::VulkanTexture* texture,
                                        uint32_t bindlessIndex,
                                        bool fallbackUsed,
                                        bool showPreview);
    void drawTexturePreview(const rhi::VulkanTexture& texture, float size);
    void drawRenderTargetPreview(VkImageView imageView,
                                 VkSampler sampler,
                                 VkImageLayout imageLayout,
                                 uint32_t width,
                                 uint32_t height,
                                 float size,
                                 float exposureScale);
    void drawGlobalTextureMetadata();
    void drawCsmCascadeDebugUi(float previewSize);
    void drawGpuTimingDebugUi();
    void drawCullingDebugUi();
    void drawExposureDebugUi();
    void drawTaaDebugUi();
    void drawSsrDebugUi();
    void drawTimingHistoryRow(const char* label, const DebugHistory& history) const;
    void drawScalarHistoryRow(const char* label, const DebugHistory& history, const char* valueFormat) const;
    void drawHistoryPlot(const DebugHistory& history, float height) const;
    void clampRuntimeSettings();
    void updateCpuFrameTime();
    void pushGpuTimingSample(const renderer::GpuProfiler::FrameResults& results);
    void resetGpuProfilerHistory();
    [[nodiscard]] DebugHistory* gpuTimingHistoryForPass(std::string_view name);
    [[nodiscard]] const DebugHistory* gpuTimingHistoryForPass(std::string_view name) const;
    void pushCullingHistorySample(uint32_t frameIndex);
    void pushExposureHistorySample();
    [[nodiscard]] CullingDebugSnapshot cullingDebugSnapshot(uint32_t frameIndex);
    [[nodiscard]] ObjectDrawDebugInfo objectDrawDebugInfo(uint32_t objectIndex) const;
    [[nodiscard]] std::vector<const renderer::Material*> materialsForObject(
        const renderer::RenderObject& object) const;
    [[nodiscard]] const renderer::Material* primaryMaterialForObject(
        const renderer::RenderObject& object) const;
    [[nodiscard]] renderer::Material* mutableMaterialFromPointer(const renderer::Material* material);
    [[nodiscard]] renderer::Material* primaryMutableMaterialForObject(renderer::RenderObject& object);
    [[nodiscard]] renderer::Material* findRuntimeMaterialByAssetPath(const std::filesystem::path& path);
    [[nodiscard]] std::string materialDebugLabel(const renderer::RenderObject& object) const;
    [[nodiscard]] std::string mainCullingDebugLabel(const ObjectDrawDebugInfo& debugInfo) const;
    [[nodiscard]] std::string shadowCullingDebugLabel(const ObjectDrawDebugInfo& debugInfo) const;
    void requestPortfolioScreenshot();
    void recordPortfolioScreenshotCopy(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void processPortfolioScreenshotReadback(uint32_t frameIndex);
    void setPortfolioCaptureMode(bool enabled);
    void applyPortfolioCaptureSettings();
    void restorePortfolioCaptureSettings();
    [[nodiscard]] bool hasPendingPortfolioScreenshotReadback() const;

    Window& window_;
    JobSystem jobSystem_;
    rhi::VulkanContext context_;
    std::vector<renderer::FrameResources> frames_;
    renderer::GpuProfiler gpuProfiler_;
    rhi::VulkanSwapchain swapchain_;
    renderer::RenderGraph renderGraph_;
    assets::AssetManager assetManager_;
    ui::ImGuiLayer imguiLayer_;
    rhi::VulkanDescriptorSetLayout materialDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout skyboxDescriptorSetLayout_;
    rhi::VulkanShadowMap shadowMap_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanPipeline skinnedPipeline_;
    rhi::VulkanPipeline skyboxPipeline_;
    rhi::VulkanPipeline shadowPipeline_;
    rhi::VulkanCommandContext commandContext_;
    // Async compute: per-frame command buffers + semaphores for the queue that
    // runs ClusterBuild/LightCull in parallel with the shadow passes. Falls
    // back to inline graphics-queue recording when unavailable or disabled.
    rhi::VulkanAsyncCompute asyncCompute_;
    rhi::VulkanSync sync_;
    rhi::VulkanTexture checkerboardTexture_;
    rhi::VulkanTexture portfolioBaseColorTexture_;
    rhi::VulkanTexture portfolioBackdropTexture_;
    rhi::VulkanTexture normalMapTexture_;
    rhi::VulkanTexture flatNormalTexture_;
    rhi::VulkanTexture metallicRoughnessTexture_;
    rhi::VulkanTexture neutralMetallicRoughnessTexture_;
    rhi::VulkanEnvironmentMap environmentMap_;
    rhi::VulkanEnvironmentMap diffuseIrradianceMap_;
    rhi::VulkanEnvironmentMap prefilteredEnvironmentMap_;
    rhi::VulkanBrdfLut brdfLutTexture_;
    std::vector<rhi::VulkanTexture> importedBaseColorTextures_;
    std::vector<rhi::VulkanTexture> importedNormalTextures_;
    std::vector<rhi::VulkanTexture> importedMetallicRoughnessTextures_;
    std::vector<std::unique_ptr<rhi::VulkanTexture>> materialAssetTextures_;
    renderer::BindlessTextureHeap bindlessTextureHeap_;
    renderer::BuiltinTextureFactory builtinTextureFactory_{};
    renderer::ClusteredLighting clusteredLighting_;
    renderer::SkinnedMesh skinnedMesh_;
    rhi::VulkanDescriptorPool materialDescriptorPool_;
    rhi::VulkanDescriptorPool skyboxDescriptorPool_;
    VkDescriptorSet skyboxDescriptorSet_ = VK_NULL_HANDLE;
    renderer::Camera camera_;
    EditorCamera editorCamera_{};
    DirectionalLightSettings directionalLightSettings_{};
    renderer::Mesh cubeMesh_;
    renderer::Mesh portfolioSphereMesh_;
    std::vector<renderer::Mesh> importedMeshes_;
    renderer::Material checkerboardMaterial_;
    std::vector<renderer::Material> materialVariants_;
    std::vector<renderer::Material> importedMaterials_;
    std::vector<renderer::RenderObject> renderObjects_;
    std::vector<DrawItem> allDrawItems_;
    std::vector<DrawItem> visibleDrawItems_;
    std::vector<DrawItem> shadowDrawItems_;
    std::array<std::vector<DrawItem>, kMaxShadowCascades> shadowCascadeDrawItems_;
    std::vector<MeshDrawBatch> meshDrawBatches_;
    std::vector<MeshDrawBatch> shadowMeshDrawBatches_;
    std::array<std::vector<MeshDrawBatch>, kMaxShadowCascades> shadowCascadeMeshDrawBatches_;
    std::vector<MeshDrawBatch> gpuShadowMeshDrawBatches_;
    std::vector<rhi::VulkanBuffer> frameObjectDataBuffers_;
    // Indirect-draw output buffers stay owned here (the main/shadow draw passes read
    // them); gpuCulling_ borrows them by reference to write the compacted commands.
    std::vector<rhi::VulkanBuffer> frameIndirectDrawBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowIndirectDrawBuffers_;
    CsmSettings csmSettings_{};
    std::vector<VkFence> imagesInFlight_;
    ShadowSettings shadowSettings_{};
    SsaoSettings ssaoSettings_{};
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat shadowPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t selectedBloomMipDebugLevel_ = 0;
    uint32_t currentFrame_ = 0;
    uint32_t bindlessBaseColorFallbackIndex_ = 0;
    uint32_t bindlessNormalFallbackIndex_ = 0;
    uint32_t bindlessMetallicRoughnessFallbackIndex_ = 0;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastGpuTimingPrint_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastExposureLogPrint_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastFrameStartTime_ = std::chrono::steady_clock::now();
    std::array<glm::vec4, 6> frameFrustumPlanes_{};
    // Discrete-LOD selection knobs. Selection itself runs in cull.comp; these are
    // uploaded in GpuCullFrameParams::lodSettings. renderer::MeshLod.h holds the
    // unit-tested reference copy of the selection math.
    struct LodSettings {
        bool enabled = true;
        float referenceRadiusPixels = 220.0f;
        float bias = 0.0f;
        // Shadow cascades drop extra detail on top of `bias`.
        float shadowBias = 1.0f;
        // >= 0 pins every draw item to that level (debug view).
        int32_t forcedLod = -1;
    };

    LodSettings lodSettings_{};
    // Flat per-frame LOD table uploaded to the cull pass, plus each draw item's
    // (base, count) range into it. Rebuilt every frame alongside the cull input
    // because scene edits add and remove meshes; deduped by mesh so a mesh's
    // chain is uploaded once no matter how many draw items reference it.
    // renderer::MeshLod is already the GPU record: two uint32s, std430-compatible
    // (static_assert'd in Renderer.cpp), so the table uploads without conversion.
    std::vector<renderer::MeshLod> frameMeshLodTable_;
    std::vector<glm::uvec2> frameDrawItemLodRanges_;

    glm::mat4 frameViewProjection_{1.0f};
    glm::mat4 frameJitteredProjection_{1.0f};
    glm::mat4 frameJitteredViewProjection_{1.0f};
    // Previous frame's unjittered view-projection, used with the per-object
    // previous model matrices to build motion vectors. Invalid after a TAA
    // history reset so the first frame reprojects with zero camera motion.
    glm::mat4 previousFrameViewProjection_{1.0f};
    bool previousFrameViewProjectionValid_ = false;
    glm::mat4 previousSkinnedModelMatrix_{1.0f};
    bool previousSkinnedModelValid_ = false;
    glm::vec3 frameCameraPosition_{0.0f};
    std::array<CascadeFrameData, kMaxShadowCascades> frameCascades_{};
    glm::vec4 frameCascadeSplits_{};
    std::array<std::array<glm::vec4, 6>, kMaxShadowCascades> frameShadowCascadeFrustumPlanes_{};
    std::array<uint32_t, kMaxShadowCascades> shadowVisibleDrawItemsPerCascade_{};
    std::array<uint32_t, kMaxShadowCascades> shadowBatchCountPerCascade_{};
    CullingStats cullingStats_{};
    ShadowCullingStats shadowCullingStats_{};
    ToneMappingSettings toneMappingSettings_{};
    BloomSettings bloomSettings_{};
    TaaSettings taaSettings_{};
    SsrSettings ssrSettings_{};
    DebugUiSettings debugUiSettings_{};
    DebugHistory gpuFrameTimeHistory_{};
    std::vector<GpuTimingHistory> gpuTimingHistories_;
    DebugHistory visibleMainDrawItemsHistory_{};
    DebugHistory culledMainDrawItemsHistory_{};
    DebugHistory visibleShadowDrawItemsHistory_{};
    DebugHistory culledShadowDrawItemsHistory_{};
    DebugHistory exposureHistory_{};
    DebugHistory averageLuminanceHistory_{};
    DebugHistory histogramClippedLuminanceHistory_{};
    renderer::GpuProfiler::FrameResults latestGpuProfilerResults_{};
    std::filesystem::path runtimeSettingsPath_;
    std::filesystem::path sceneDocumentPath_;
    std::string lastRuntimeSettingsLoadStatus_ = "Not loaded yet.";
    std::string lastRuntimeSettingsSaveStatus_ = "Not saved this session.";
    std::string runtimeSettingsWarning_;
    std::string lastSceneLoadStatus_ = "Not loaded yet.";
    std::string lastSceneSaveStatus_ = "Not saved this session.";
    std::string lastMaterialAssetStatus_ = "No material asset saved or reloaded this session.";
    std::string occlusionTestSceneStatus_ = "Occlusion test scene not loaded.";
    renderer::ScreenshotCapture screenshotCapture_;
    PortfolioCaptureSavedState portfolioCaptureSavedState_{};
    float cpuFrameDeltaMs_ = 0.0f;
    float cpuFps_ = 0.0f;
    // CPU frame-preparation cost (updateFrameData: transforms, bounds, draw
    // items, culling inputs, per-object frame data) and the JobSystem toggle
    // that lets the debug UI A/B single-threaded vs parallel prep.
    bool parallelFramePrepEnabled_ = true;
    DebugHistory framePrepCpuHistory_{};
    std::vector<renderer::Aabb> frameWorldBounds_;
    float currentExposure_ = 1.0f;
    float averageLuminance_ = 0.18f;
    float histogramClippedLuminance_ = 0.18f;
    size_t selectedRenderObjectIndex_ = kInvalidRenderObjectIndex;
    uint32_t nextRenderObjectDebugId_ = 1;
    // Editor viewport interaction state.
    GizmoOperation gizmoOperation_ = GizmoOperation::Translate;
    bool gizmoWorldSpace_ = true;
    bool cameraFlying_ = false;   // RMB held: free-fly look + WASD
    bool cameraOrbiting_ = false; // Alt+LMB: orbit around target
    bool cameraPanning_ = false;  // MMB: pan
    bool leftMouseDown_ = false;
    bool leftMouseDragged_ = false;
    glm::vec2 leftMouseDownPosition_{0.0f, 0.0f};
    glm::vec2 pendingLookDelta_{0.0f, 0.0f}; // accumulated mouse motion while flying (pixels)
    float pendingScroll_ = 0.0f;             // accumulated wheel this frame
    bool initialized_ = false;
    bool useBindlessMaterialTextures_ = true;
    bool bindlessMaterialTexturesAvailable_ = false;
    bool useGpuCulling_ = true;
    // When true (and clustered resources are available), the main pass walks the
    // per-froxel light list; otherwise it brute-forces every light. The runtime
    // toggle also enables a brute-force-vs-clustered comparison.
    bool useClusteredLighting_ = true;
    bool showClusterHeatmap_ = false;
    // Procedural skinned bone-chain demo: draw it, and animate (vs hold bind pose).
    bool showSkinnedMesh_ = true;
    bool animateSkinnedMesh_ = true;
    float skinnedAnimationSpeed_ = 1.0f;
    float skinnedAnimationTime_ = 0.0f;   // accumulated playback time (scaled by speed)
    float previousElapsedSeconds_ = 0.0f; // for the per-frame delta
    // Demo-light controls (see updateDemoLights). The count slider drives the
    // clustered-path stress test; animation orbits the swarm each frame.
    int demoLightCount_ = 24;
    bool animateLights_ = true;
    float demoLightIntensity_ = 10.0f;
    float demoLightRange_ = 8.0f;
    bool useGpuOcclusionCulling_ = false;
    // Two-phase Hi-Z occlusion: phase 1 culls against the previous frame's
    // pyramid, phase 2 re-tests the occluded candidates against a mid-frame
    // rebuild so disocclusions never drop draws. Requires the indirect-count
    // path; frameTwoPhaseOcclusionActive_ is the per-frame resolved predicate.
    bool useTwoPhaseOcclusion_ = true;
    bool frameTwoPhaseOcclusionActive_ = false;
    bool useAsyncCompute_ = true;
    bool frameAsyncComputeActive_ = false;
    bool frameSsrActive_ = false;
    bool frameGtaoActive_ = false;
    uint32_t ssrFrameCounter_ = 0;
    uint32_t gtaoFrameCounter_ = 0;
    bool useGpuShadowCulling_ = true;
    bool shadowIndirectAvailable_ = false;
    bool ssaoAvailable_ = false;
    bool gpuProfilerEnabled_ = true;
    bool portfolioCaptureMode_ = false;
    bool occlusionTestSceneActive_ = false;
    bool portfolioScreenshotRequested_ = false;
    bool normalMapAssetLoaded_ = false;
    bool metallicRoughnessMapAssetLoaded_ = false;
    bool hdrEnvironmentLoaded_ = false;
    bool showRenderTargetSceneColor_ = true;
    bool showRenderTargetBloomExtract_ = true;
    bool showRenderTargetBlurredBloom_ = true;
    bool showRenderTargetBloomMipChain_ = true;
    bool showRenderTargetTaaHistory_ = true;
    bool showRenderTargetFinalCompositeMetadata_ = true;
    bool showRenderTargetBrdfLut_ = true;
    bool showRenderTargetCsmCascades_ = true;
    float gpuOcclusionDepthBias_ = 0.002f;
    float gpuOcclusionNearDisableDistance_ = 1.0f;
    float gpuOcclusionMaxScreenCoverage_ = 0.35f;
    float gpuOcclusionMinScreenPixels_ = 4.0f;

    // HDR post-process subsystem (bloom, TAA, auto-exposure, composite). Owns its
    // own GPU resources; borrows the services + settings declared above by
    // reference, so it is declared last to guarantee those are constructed first.
    renderer::PostProcessStack postProcess_{context_,
                                            renderGraph_,
                                            gpuProfiler_,
                                            swapchain_,
                                            toneMappingSettings_,
                                            bloomSettings_,
                                            taaSettings_,
                                            ssaoSettings_,
                                            currentExposure_,
                                            averageLuminance_,
                                            histogramClippedLuminance_,
                                            ssaoAvailable_};

    // Screen-space reflections: view-space march against main depth using the
    // thin G-buffer, additively blended into scene color before TAA. Declared
    // after the services + settings it borrows.
    renderer::ScreenSpaceReflections ssr_{context_, swapchain_, renderGraph_, gpuProfiler_, ssrSettings_};

    // Ground-truth ambient occlusion: horizon-search pass reading main depth +
    // the thin G-buffer normal, writing the visibility target the composite
    // multiplies into scene color. Borrows the same services + the SSAO settings.
    renderer::GroundTruthAmbientOcclusion gtao_{context_, swapchain_, renderGraph_, gpuProfiler_, ssaoSettings_};

    // Hi-Z depth pyramid subsystem. Like postProcess_, it owns its GPU resources
    // and borrows the rendering services by reference, so it is declared last to
    // guarantee those are constructed first.
    renderer::DepthPyramid depthPyramid_{context_, swapchain_, renderGraph_, gpuProfiler_};

    // GPU-driven visibility culling (main frustum/occlusion + per-cascade shadow).
    // Owns its cull pipeline/descriptors/buffers; borrows the services, the depth
    // pyramid, and the indirect-draw output buffers. Declared last so all of those
    // are constructed first.
    renderer::GpuCulling gpuCulling_{context_,
                                     depthPyramid_,
                                     renderGraph_,
                                     gpuProfiler_,
                                     frameIndirectDrawBuffers_,
                                     frameShadowIndirectDrawBuffers_};
};

} // namespace ve
