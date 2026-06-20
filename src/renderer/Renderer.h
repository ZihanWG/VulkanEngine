#pragma once

#include "assets/AssetManager.h"
#include "core/JobSystem.h"
#include "renderer/BindlessTextureHeap.h"
#include "renderer/EditorCamera.h"
#include "renderer/Bounds.h"
#include "renderer/Camera.h"
#include "renderer/FrameResources.h"
#include "renderer/GpuProfiler.h"
#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/RenderGraph.h"
#include "renderer/RenderObject.h"
#include "renderer/RuntimeSettings.h"
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
    static constexpr uint32_t kMaxShadowCascades = 4;
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

    // Screen-space ambient occlusion computed in the composite pass from the main
    // depth buffer. Disabled by default; the toggle is only honoured when the
    // depth image supports sampling (ssaoAvailable_).
    struct SsaoSettings {
        bool enabled = false;
        float radius = 0.5f;
        float bias = 0.025f;
        float intensity = 1.0f;
        float power = 2.0f;
        int sampleCount = 16;
    };

    struct CascadeFrameData {
        glm::mat4 lightViewProjection{1.0f};
        renderer::Frustum lightFrustum{};
        float splitDepth = 0.0f;
        float nearDepth = 0.0f;
        float farDepth = 0.0f;
    };

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

    struct GpuCullCounters {
        uint32_t totalDrawItems = 0;
        uint32_t visibleDrawItems = 0;
        uint32_t frustumCulledDrawItems = 0;
        uint32_t occlusionCulledDrawItems = 0;
    };

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

    struct PortfolioScreenshotReadback {
        rhi::VulkanBuffer buffer;
        VkExtent2D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::filesystem::path timestampedPath;
        std::filesystem::path latestPath;
        bool pending = false;
    };

    void createMaterialDescriptorSetLayout();
    void createBindlessMaterialTextureHeap();
    void createSkyboxDescriptorSetLayout();
    void createPostProcessDescriptorSetLayouts();
    void createPostProcessSampler();
    void destroyPostProcessSampler();
    void createPostProcessResources();
    void createPostProcessDescriptorSets();
    void createTaaResources();
    void destroyTaaResources();
    void invalidateTaaHistory();
    void createExposureResources();
    void destroyExposureResources();
    void createDepthPyramidDescriptorSetLayout();
    void createDepthPyramidResources();
    void destroyDepthPyramidResources();
    void updateGpuCullingDepthPyramidDescriptors();
    void invalidateDepthPyramid();
    void createLuminanceResources();
    void destroyLuminanceResources();
    void createHistogramResources();
    void destroyHistogramResources();
    void disableAutoExposureFallback(std::string_view reason);
    void disableLogAverageExposureFallback(std::string_view reason);
    void disableHistogramExposureFallback(std::string_view reason);
    void updateAutoExposureFromReadback(uint32_t frameIndex);
    void recordLuminanceCommands(VkCommandBuffer commandBuffer);
    void recordHistogramCommands(VkCommandBuffer commandBuffer);
    void recordTaaResolveCommands(VkCommandBuffer commandBuffer);
    void recordLegacyBloomCommands(VkCommandBuffer commandBuffer);
    void recordMipChainBloomCommands(VkCommandBuffer commandBuffer);
    void createGpuCullingResources();
    void destroyGpuCullingResources();
    void createGpuShadowCullingResources();
    void destroyGpuShadowCullingResources();
    void createShadowMap();
    void createPipeline();
    void createScene();
    void addPortfolioShowcaseObjects();
    void resetPortfolioShowcaseObjectsToPreset();
    [[nodiscard]] bool hasPortfolioShowcaseScene() const;
    [[nodiscard]] bool currentFrameHasPortfolioShowcaseDrawItems() const;
    [[nodiscard]] bool ensurePortfolioShowcaseSceneReady();
    void addOcclusionTestSceneObjects();
    void resetOcclusionTestSceneToPreset();
    [[nodiscard]] bool hasOcclusionTestScene() const;
    [[nodiscard]] uint32_t allocateRenderObjectDebugId();
    void createCheckerboardTexture();
    void createPortfolioBaseColorTexture();
    void createPortfolioBackdropTexture();
    void createNormalTexture();
    void createFlatNormalTexture();
    void createMetallicRoughnessTexture();
    void createNeutralMetallicRoughnessTexture();
    void createEnvironmentMap();
    void createDiffuseIrradianceMap();
    void createPrefilteredEnvironmentMap();
    void createBrdfLutTexture();
    void createMaterial();
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
    void updateGpuCullInputBuffer(uint32_t frameIndex);
    void updateGpuShadowCullInputBuffer(uint32_t frameIndex);
    void updateFrameData(uint32_t frameIndex);
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
    void recordDepthPyramidCommands(VkCommandBuffer commandBuffer);
    [[nodiscard]] renderer::RenderGraphFrameResources renderGraphFrameResources();
    bool readGpuVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    bool readGpuCullCounters(uint32_t frameIndex, GpuCullCounters& counters);
    bool readGpuShadowVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    [[nodiscard]] bool isGpuCullingActive() const;
    [[nodiscard]] bool isGpuOcclusionCullingActive() const;
    [[nodiscard]] bool isGpuShadowCullingActive() const;
    [[nodiscard]] bool isAutoExposureActive() const;
    [[nodiscard]] bool isLogAverageExposureActive() const;
    [[nodiscard]] bool isHistogramExposureActive() const;
    [[nodiscard]] bool isGpuExposureActive() const;
    [[nodiscard]] bool isTaaActive() const;
    [[nodiscard]] bool isTaaJitterActive() const;
    [[nodiscard]] uint32_t taaHistoryReadIndex() const;
    [[nodiscard]] uint32_t taaHistoryWriteIndex() const;
    [[nodiscard]] VkDescriptorSet activeBloomExtractDescriptorSet() const;
    [[nodiscard]] VkDescriptorSet activeBloomMipDownsampleDescriptorSet(uint32_t level) const;
    [[nodiscard]] VkDescriptorSet activeCompositeDescriptorSet() const;
    [[nodiscard]] VkDescriptorSet activeLuminanceDescriptorSet() const;
    [[nodiscard]] VkDescriptorSet activeHistogramDescriptorSet() const;
    [[nodiscard]] bool isBindlessMaterialTextureActive() const;
    [[nodiscard]] bool isMainPassMultiDrawIndirectActive() const;
    [[nodiscard]] bool isMainPassIndirectCountSupported() const;
    [[nodiscard]] bool isFrameIndirectCountPathActive(uint32_t frameIndex) const;
    [[nodiscard]] bool isShadowIndirectCountSupported() const;
    [[nodiscard]] bool isShadowIndirectCountPathActive(uint32_t frameIndex) const;
    [[nodiscard]] bool isShadowIndirectActive() const;
    [[nodiscard]] uint32_t activeCascadeCount() const;
    [[nodiscard]] VkDescriptorSet globalMaterialDescriptorSet() const;
    [[nodiscard]] float currentToneMappingExposure() const;
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
    rhi::VulkanDescriptorSetLayout postProcessSingleImageDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessDualImageDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessCompositeDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessLuminanceDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessExposureReduceDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout gpuCullDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout depthPyramidDescriptorSetLayout_;
    rhi::VulkanShadowMap shadowMap_;
    rhi::VulkanImage sceneColor_;
    rhi::VulkanImage bloomExtract_;
    rhi::VulkanImage bloomPing_;
    rhi::VulkanImage bloomPong_;
    std::array<rhi::VulkanImage, 2> taaHistoryImages_;
    std::vector<rhi::VulkanImage> bloomMipDownsampleImages_;
    std::vector<rhi::VulkanImage> bloomMipUpsampleImages_;
    rhi::VulkanImage depthPyramid_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanPipeline skyboxPipeline_;
    rhi::VulkanPipeline shadowPipeline_;
    rhi::VulkanPipeline bloomExtractPipeline_;
    rhi::VulkanPipeline bloomBlurPipeline_;
    rhi::VulkanPipeline bloomDownsamplePipeline_;
    rhi::VulkanPipeline bloomUpsamplePipeline_;
    rhi::VulkanPipeline taaResolvePipeline_;
    rhi::VulkanPipeline compositePipeline_;
    rhi::VulkanComputePipeline luminancePipeline_;
    rhi::VulkanComputePipeline histogramPipeline_;
    rhi::VulkanComputePipeline exposureReducePipeline_;
    rhi::VulkanComputePipeline gpuCullPipeline_;
    rhi::VulkanComputePipeline depthPyramidPipeline_;
    rhi::VulkanCommandContext commandContext_;
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
    rhi::VulkanDescriptorPool materialDescriptorPool_;
    rhi::VulkanDescriptorPool skyboxDescriptorPool_;
    rhi::VulkanDescriptorPool postProcessDescriptorPool_;
    rhi::VulkanDescriptorPool gpuCullDescriptorPool_;
    rhi::VulkanDescriptorPool shadowCullDescriptorPool_;
    rhi::VulkanDescriptorPool depthPyramidDescriptorPool_;
    VkSampler postProcessSampler_ = VK_NULL_HANDLE;
    VkSampler depthPyramidSampler_ = VK_NULL_HANDLE;
    VkDescriptorSet skyboxDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomExtractDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurHorizontalDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurVerticalDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet compositeDescriptorSet_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> taaResolveDescriptorSets_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDescriptorSet, 2> taaBloomExtractDescriptorSets_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDescriptorSet, 2> taaBloomMipDownsampleDescriptorSets_{VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> bloomMipDownsampleDescriptorSets_;
    std::vector<VkDescriptorSet> bloomMipUpsampleDescriptorSets_;
    std::vector<VkDescriptorSet> compositeDescriptorSets_;
    std::vector<VkDescriptorSet> luminanceDescriptorSets_;
    std::vector<VkDescriptorSet> histogramDescriptorSets_;
    std::array<std::vector<VkDescriptorSet>, 2> taaCompositeDescriptorSets_;
    std::array<std::vector<VkDescriptorSet>, 2> taaLuminanceDescriptorSets_;
    std::array<std::vector<VkDescriptorSet>, 2> taaHistogramDescriptorSets_;
    std::vector<VkDescriptorSet> exposureReduceDescriptorSets_;
    std::vector<VkDescriptorSet> gpuCullDescriptorSets_;
    std::vector<VkDescriptorSet> shadowCullDescriptorSets_;
    std::vector<VkDescriptorSet> depthPyramidDescriptorSets_;
    std::vector<VkImageView> depthPyramidMipImageViews_;
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
    std::vector<rhi::VulkanBuffer> frameCullInputBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowCullInputBuffers_;
    std::vector<rhi::VulkanBuffer> frameGpuCullParamBuffers_;
    std::vector<rhi::VulkanBuffer> frameIndirectDrawBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowIndirectDrawBuffers_;
    std::vector<rhi::VulkanBuffer> frameBatchVisibleCountBuffers_;
    std::vector<rhi::VulkanBuffer> frameBatchVisibleCountReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowBatchVisibleCountBuffers_;
    std::vector<rhi::VulkanBuffer> frameShadowBatchVisibleCountReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameLuminanceBuffers_;
    std::vector<rhi::VulkanBuffer> frameLuminanceReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameHistogramBuffers_;
    std::vector<rhi::VulkanBuffer> frameHistogramReadbackBuffers_;
    std::vector<rhi::VulkanBuffer> frameExposureBuffers_;
    std::vector<PortfolioScreenshotReadback> portfolioScreenshotReadbacks_;
    std::vector<uint32_t> frameGpuCullTotalDrawItems_;
    std::vector<uint32_t> frameGpuCullBatchCounts_;
    std::vector<uint32_t> frameGpuShadowCullTotalDrawItems_;
    std::vector<uint32_t> frameGpuShadowCullBatchCounts_;
    std::vector<uint8_t> frameGpuCullReadbackReady_;
    std::vector<uint8_t> frameGpuCullIndirectCountPath_;
    std::vector<uint8_t> frameGpuShadowCullReadbackReady_;
    std::vector<uint8_t> frameGpuShadowCullIndirectCountPath_;
    std::vector<uint8_t> frameLuminanceReadbackReady_;
    std::vector<uint8_t> frameHistogramReadbackReady_;
    std::vector<uint8_t> frameExposureReadbackReady_;
    CsmSettings csmSettings_{};
    std::vector<VkFence> imagesInFlight_;
    ShadowSettings shadowSettings_{};
    SsaoSettings ssaoSettings_{};
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat shadowPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomExtractPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomBlurPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomDownsamplePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomUpsamplePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat taaResolvePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat compositePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D bloomExtent_{};
    VkImageLayout sceneColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomExtractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPingLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPongLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::array<VkImageLayout, 2> taaHistoryLayouts_{VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    std::vector<VkImageLayout> bloomMipDownsampleLayouts_;
    std::vector<VkImageLayout> bloomMipUpsampleLayouts_;
    VkImageLayout depthPyramidLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t luminanceGroupCountX_ = 0;
    uint32_t luminanceGroupCountY_ = 0;
    uint32_t luminancePartialCount_ = 0;
    uint32_t depthPyramidMipLevels_ = 0;
    uint32_t selectedDepthPyramidDebugMip_ = 0;
    uint32_t selectedBloomMipDebugLevel_ = 0;
    uint32_t currentFrame_ = 0;
    uint32_t bindlessBaseColorFallbackIndex_ = 0;
    uint32_t bindlessNormalFallbackIndex_ = 0;
    uint32_t bindlessMetallicRoughnessFallbackIndex_ = 0;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastGpuTimingPrint_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastExposureLogPrint_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastAutoExposureUpdate_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastFrameStartTime_ = std::chrono::steady_clock::now();
    std::array<glm::vec4, 6> frameFrustumPlanes_{};
    glm::mat4 frameViewProjection_{1.0f};
    glm::mat4 frameJitteredProjection_{1.0f};
    glm::mat4 frameJitteredViewProjection_{1.0f};
    glm::mat4 previousFrameViewProjection_{1.0f};
    glm::mat4 depthPyramidViewProjection_{1.0f};
    glm::vec3 frameCameraPosition_{0.0f};
    glm::vec3 depthPyramidCameraPosition_{0.0f};
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
    std::string portfolioScreenshotStatus_ = "No capture requested.";
    std::filesystem::path lastPortfolioScreenshotPath_;
    PortfolioCaptureSavedState portfolioCaptureSavedState_{};
    float cpuFrameDeltaMs_ = 0.0f;
    float cpuFps_ = 0.0f;
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
    bool gpuCullingAvailable_ = false;
    bool useGpuOcclusionCulling_ = false;
    bool depthPyramidValid_ = false;
    bool depthPyramidBuildAvailable_ = false;
    bool useGpuShadowCulling_ = true;
    bool gpuShadowCullingAvailable_ = false;
    bool autoExposureAvailable_ = false;
    bool histogramExposureAvailable_ = false;
    bool exposureReduceAvailable_ = false;
    bool autoExposureWarningLogged_ = false;
    bool logAverageExposureWarningLogged_ = false;
    bool histogramExposureWarningLogged_ = false;
    bool shadowIndirectAvailable_ = false;
    bool ssaoAvailable_ = false;
    bool taaHistoryValid_ = false;
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
    glm::vec2 taaCurrentJitterPixels_{0.0f, 0.0f};
    glm::vec2 taaPreviousJitterPixels_{0.0f, 0.0f};
    glm::vec2 taaCurrentJitterNdc_{0.0f, 0.0f};
    glm::vec2 taaPreviousJitterNdc_{0.0f, 0.0f};
    uint32_t taaJitterIndex_ = 0;
    uint32_t taaHistoryWriteIndex_ = 0;
    uint32_t taaPostProcessHistoryIndex_ = 0;
};

} // namespace ve
