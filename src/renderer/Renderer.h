#pragma once

#include "renderer/BindlessTextureHeap.h"
#include "renderer/Bounds.h"
#include "renderer/Camera.h"
#include "renderer/FrameResources.h"
#include "renderer/Material.h"
#include "renderer/Mesh.h"
#include "renderer/RenderGraph.h"
#include "renderer/RenderObject.h"
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
#include "rhi/VulkanTimestampQuery.h"
#include "rhi/VulkanTexture.h"
#include "ui/ImGuiLayer.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
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

    struct CsmSettings {
        uint32_t cascadeCount = kMaxShadowCascades;
        float lambda = 0.5f;
        float nearPlane = 0.1f;
        float farPlane = 100.0f;
        float shadowDistance = 40.0f;
        bool enableTexelSnapping = true;
        bool enableCascadeDebugColors = false;
        float depthBiasConstant = 0.002f;
        float depthBiasSlope = 0.005f;
    };

    struct ShadowSettings {
        uint32_t resolution = 2048;
        bool enablePcf = true;
        int pcfRadius = 1;
        float rasterDepthBiasConstantFactor = 1.25f;
        float rasterDepthBiasSlopeFactor = 1.75f;
    };

    struct ToneMappingSettings {
        float manualExposure = 1.0f;
        bool enableAutoExposure = true;

        int exposureMode = 2;
        // 0 = manual
        // 1 = log-average luminance
        // 2 = histogram percentile

        float targetLuminance = 0.18f;
        float minExposure = 0.1f;
        float maxExposure = 8.0f;
        float adaptationRate = 1.5f;

        float histogramMinLogLuminance = -10.0f;
        float histogramMaxLogLuminance = 4.0f;
        float lowPercentile = 0.05f;
        float highPercentile = 0.95f;

        int operatorType = 0;
        // 0 = Reinhard
        // 1 = ACES fitted approximation
    };

    struct BloomSettings {
        bool enabled = true;
        float threshold = 1.0f;
        float intensity = 0.1f;
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
        DebugHistory shadowPass;
        DebugHistory mainPass;
        DebugHistory bloom;
        DebugHistory composite;
        DebugHistory autoExposure;
        DebugHistory histogramExposure;
        DebugHistory skybox;
        DebugHistory renderObjects;
        DebugHistory knownFrameTotal;
    };

    struct DebugUiSettings {
        bool showRenderGraphPanel = true;
        bool showGpuTimingGraphs = true;
        bool showCullingStats = true;
        bool showExposureGraphs = true;
    };

    struct CullingDebugSnapshot {
        uint32_t totalDrawItems = 0;
        uint32_t visibleDrawItems = 0;
        uint32_t culledDrawItems = 0;
        uint32_t shadowDrawItems = 0;
        uint32_t visibleShadowDrawItems = 0;
        uint32_t culledShadowDrawItems = 0;
        size_t shadowBatchCount = 0;
        bool gpuCulling = false;
        bool gpuShadowCulling = false;
    };

    void createMaterialDescriptorSetLayout();
    void createBindlessMaterialTextureHeap();
    void createSkyboxDescriptorSetLayout();
    void createPostProcessDescriptorSetLayouts();
    void createPostProcessSampler();
    void destroyPostProcessSampler();
    void createPostProcessResources();
    void createPostProcessDescriptorSets();
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
    void createGpuCullingResources();
    void destroyGpuCullingResources();
    void createGpuShadowCullingResources();
    void destroyGpuShadowCullingResources();
    void createShadowMap();
    void createPipeline();
    void createScene();
    void createCheckerboardTexture();
    void createNormalTexture();
    void createFlatNormalTexture();
    void createMetallicRoughnessTexture();
    void createNeutralMetallicRoughnessTexture();
    void createEnvironmentMap();
    void createDiffuseIrradianceMap();
    void createPrefilteredEnvironmentMap();
    void createBrdfLutTexture();
    void createMaterial();
    void assignBindlessTextureIndices(renderer::Material& material);
    void createMaterialDescriptorSet(renderer::Material& material);
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
    void updateIndirectDrawBuffer(uint32_t frameIndex);
    void updateShadowIndirectDrawBuffer(uint32_t frameIndex);
    [[nodiscard]] const renderer::Material* resolveMaterial(const renderer::RenderObject& object,
                                                            const renderer::MeshPrimitive* primitive) const;
    void recreateSwapchain();
    void recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void recordGpuCullingCommands(VkCommandBuffer commandBuffer);
    void recordGpuShadowCullingCommands(VkCommandBuffer commandBuffer, uint32_t cascadeIndex);
    [[nodiscard]] renderer::RenderGraphFrameResources renderGraphFrameResources();
    bool readGpuVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    bool readGpuShadowVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    [[nodiscard]] bool isGpuCullingActive() const;
    [[nodiscard]] bool isGpuShadowCullingActive() const;
    [[nodiscard]] bool isAutoExposureActive() const;
    [[nodiscard]] bool isLogAverageExposureActive() const;
    [[nodiscard]] bool isHistogramExposureActive() const;
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
    void buildDebugUi();
    void drawRenderGraphDebugUi();
    void drawGpuTimingDebugUi();
    void drawCullingDebugUi();
    void drawExposureDebugUi();
    void drawTimingHistoryRow(const char* label, const DebugHistory& history) const;
    void drawScalarHistoryRow(const char* label, const DebugHistory& history, const char* valueFormat) const;
    void drawHistoryPlot(const DebugHistory& history, float height) const;
    void clampRuntimeSettings();
    void updateCpuFrameTime();
    void pushGpuTimingSample(const rhi::VulkanTimestampQuery::Results& results);
    void pushCullingHistorySample(uint32_t frameIndex);
    void pushExposureHistorySample();
    [[nodiscard]] CullingDebugSnapshot cullingDebugSnapshot(uint32_t frameIndex);

    Window& window_;
    rhi::VulkanContext context_;
    std::vector<renderer::FrameResources> frames_;
    rhi::VulkanTimestampQuery timestampQuery_;
    rhi::VulkanSwapchain swapchain_;
    renderer::RenderGraph renderGraph_;
    ui::ImGuiLayer imguiLayer_;
    rhi::VulkanDescriptorSetLayout materialDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout skyboxDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessSingleImageDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessCompositeDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout postProcessLuminanceDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout gpuCullDescriptorSetLayout_;
    rhi::VulkanShadowMap shadowMap_;
    rhi::VulkanImage sceneColor_;
    rhi::VulkanImage bloomExtract_;
    rhi::VulkanImage bloomPing_;
    rhi::VulkanImage bloomPong_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanPipeline skyboxPipeline_;
    rhi::VulkanPipeline shadowPipeline_;
    rhi::VulkanPipeline bloomExtractPipeline_;
    rhi::VulkanPipeline bloomBlurPipeline_;
    rhi::VulkanPipeline compositePipeline_;
    rhi::VulkanComputePipeline luminancePipeline_;
    rhi::VulkanComputePipeline histogramPipeline_;
    rhi::VulkanComputePipeline gpuCullPipeline_;
    rhi::VulkanCommandContext commandContext_;
    rhi::VulkanSync sync_;
    rhi::VulkanTexture checkerboardTexture_;
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
    renderer::BindlessTextureHeap bindlessTextureHeap_;
    rhi::VulkanDescriptorPool materialDescriptorPool_;
    rhi::VulkanDescriptorPool skyboxDescriptorPool_;
    rhi::VulkanDescriptorPool postProcessDescriptorPool_;
    rhi::VulkanDescriptorPool gpuCullDescriptorPool_;
    rhi::VulkanDescriptorPool shadowCullDescriptorPool_;
    VkSampler postProcessSampler_ = VK_NULL_HANDLE;
    VkDescriptorSet skyboxDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomExtractDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurHorizontalDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurVerticalDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet compositeDescriptorSet_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> luminanceDescriptorSets_;
    std::vector<VkDescriptorSet> histogramDescriptorSets_;
    std::vector<VkDescriptorSet> gpuCullDescriptorSets_;
    std::vector<VkDescriptorSet> shadowCullDescriptorSets_;
    renderer::Camera camera_;
    renderer::Mesh cubeMesh_;
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
    CsmSettings csmSettings_{};
    std::vector<VkFence> imagesInFlight_;
    ShadowSettings shadowSettings_{};
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat shadowPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomExtractPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat bloomBlurPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat compositePipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D bloomExtent_{};
    VkImageLayout sceneColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomExtractLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPingLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout bloomPongLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t luminanceGroupCountX_ = 0;
    uint32_t luminanceGroupCountY_ = 0;
    uint32_t luminancePartialCount_ = 0;
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
    std::array<CascadeFrameData, kMaxShadowCascades> frameCascades_{};
    glm::vec4 frameCascadeSplits_{};
    std::array<std::array<glm::vec4, 6>, kMaxShadowCascades> frameShadowCascadeFrustumPlanes_{};
    std::array<uint32_t, kMaxShadowCascades> shadowVisibleDrawItemsPerCascade_{};
    std::array<uint32_t, kMaxShadowCascades> shadowBatchCountPerCascade_{};
    CullingStats cullingStats_{};
    ShadowCullingStats shadowCullingStats_{};
    ToneMappingSettings toneMappingSettings_{};
    BloomSettings bloomSettings_{};
    DebugUiSettings debugUiSettings_{};
    GpuTimingHistory gpuTimingHistory_{};
    DebugHistory visibleMainDrawItemsHistory_{};
    DebugHistory culledMainDrawItemsHistory_{};
    DebugHistory visibleShadowDrawItemsHistory_{};
    DebugHistory culledShadowDrawItemsHistory_{};
    DebugHistory exposureHistory_{};
    DebugHistory averageLuminanceHistory_{};
    DebugHistory histogramClippedLuminanceHistory_{};
    rhi::VulkanTimestampQuery::Results latestGpuTimings_{};
    float cpuFrameDeltaMs_ = 0.0f;
    float cpuFps_ = 0.0f;
    float currentExposure_ = 1.0f;
    float averageLuminance_ = 0.18f;
    float histogramClippedLuminance_ = 0.18f;
    bool initialized_ = false;
    bool useBindlessMaterialTextures_ = true;
    bool bindlessMaterialTexturesAvailable_ = false;
    bool useGpuCulling_ = true;
    bool gpuCullingAvailable_ = false;
    bool useGpuShadowCulling_ = true;
    bool gpuShadowCullingAvailable_ = false;
    bool autoExposureAvailable_ = false;
    bool histogramExposureAvailable_ = false;
    bool autoExposureWarningLogged_ = false;
    bool logAverageExposureWarningLogged_ = false;
    bool histogramExposureWarningLogged_ = false;
    bool shadowIndirectAvailable_ = false;
    bool normalMapAssetLoaded_ = false;
    bool metallicRoughnessMapAssetLoaded_ = false;
    bool hdrEnvironmentLoaded_ = false;
};

} // namespace ve
