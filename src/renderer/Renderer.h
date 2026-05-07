#pragma once

#include "renderer/BindlessTextureHeap.h"
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
#include "rhi/VulkanPipeline.h"
#include "rhi/VulkanShadowMap.h"
#include "rhi/VulkanSwapchain.h"
#include "rhi/VulkanSync.h"
#include "rhi/VulkanTimestampQuery.h"
#include "rhi/VulkanTexture.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <glm/vec4.hpp>
#include <string_view>
#include <vector>

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
    void waitIdle();

private:
    struct ShadowSettings {
        uint32_t resolution = 2048;
        float constantBias = 0.002f;
        float slopeBias = 0.005f;
        bool enablePcf = true;
        int pcfRadius = 1;
        float rasterDepthBiasConstantFactor = 1.25f;
        float rasterDepthBiasSlopeFactor = 1.75f;
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
        uint32_t firstCommand = 0;
        uint32_t commandCount = 0;
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

    void createMaterialDescriptorSetLayout();
    void createBindlessMaterialTextureHeap();
    void createSkyboxDescriptorSetLayout();
    void createGpuCullingResources();
    void destroyGpuCullingResources();
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
    void createImportedGltfTextures(const std::vector<renderer::GltfTextureInfo>& textureInfos);
    void createImportedGltfMaterials(const std::vector<renderer::GltfMaterialInfo>& materialInfos);
    void createSkyboxDescriptorSet();
    void createObjectFrameDataBuffers();
    void createIndirectDrawBuffers();
    void updateGpuCullInputBuffer(uint32_t frameIndex);
    void updateFrameData(uint32_t frameIndex);
    void buildDrawItems();
    void buildVisibleDrawItems(const renderer::Frustum& frustum);
    void buildMeshDrawBatches();
    bool appendDrawItemsForObject(uint32_t objectIndex, std::vector<DrawItem>& drawItems) const;
    void updateIndirectDrawBuffer(uint32_t frameIndex);
    [[nodiscard]] const renderer::Material* resolveMaterial(
        const renderer::RenderObject& object,
        const renderer::MeshPrimitive* primitive) const;
    void recreateSwapchain();
    void recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void recordGpuCullingCommands(VkCommandBuffer commandBuffer);
    bool readGpuVisibleCount(uint32_t frameIndex, uint32_t& visibleCount);
    [[nodiscard]] bool isGpuCullingActive() const;
    [[nodiscard]] bool isBindlessMaterialTextureActive() const;
    [[nodiscard]] bool isMainPassMultiDrawIndirectActive() const;
    [[nodiscard]] VkDescriptorSet globalMaterialDescriptorSet() const;
    void nameTextureResources(const rhi::VulkanTexture& texture, std::string_view name) const;
    void nameEnvironmentMapResources(const rhi::VulkanEnvironmentMap& environmentMap, std::string_view name) const;
    void nameBrdfLutResources(const rhi::VulkanBrdfLut& brdfLut, std::string_view name) const;
    void tryPrintGpuTimings(uint32_t frameIndex);

    Window& window_;
    rhi::VulkanContext context_;
    std::vector<renderer::FrameResources> frames_;
    rhi::VulkanTimestampQuery timestampQuery_;
    rhi::VulkanSwapchain swapchain_;
    renderer::RenderGraph renderGraph_;
    rhi::VulkanDescriptorSetLayout materialDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout skyboxDescriptorSetLayout_;
    rhi::VulkanDescriptorSetLayout gpuCullDescriptorSetLayout_;
    rhi::VulkanShadowMap shadowMap_;
    rhi::VulkanPipeline pipeline_;
    rhi::VulkanPipeline skyboxPipeline_;
    rhi::VulkanPipeline shadowPipeline_;
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
    std::vector<rhi::VulkanTexture> importedTextures_;
    renderer::BindlessTextureHeap bindlessTextureHeap_;
    rhi::VulkanDescriptorPool materialDescriptorPool_;
    rhi::VulkanDescriptorPool skyboxDescriptorPool_;
    rhi::VulkanDescriptorPool gpuCullDescriptorPool_;
    VkDescriptorSet skyboxDescriptorSet_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> gpuCullDescriptorSets_;
    renderer::Camera camera_;
    renderer::Mesh cubeMesh_;
    std::vector<renderer::Mesh> importedMeshes_;
    renderer::Material checkerboardMaterial_;
    std::vector<renderer::Material> materialVariants_;
    std::vector<renderer::Material> importedMaterials_;
    std::vector<renderer::RenderObject> renderObjects_;
    std::vector<DrawItem> drawItems_;
    std::vector<DrawItem> visibleDrawItems_;
    std::vector<MeshDrawBatch> meshDrawBatches_;
    std::vector<rhi::VulkanBuffer> frameObjectDataBuffers_;
    std::vector<rhi::VulkanBuffer> frameCullInputBuffers_;
    std::vector<rhi::VulkanBuffer> frameIndirectDrawBuffers_;
    std::vector<rhi::VulkanBuffer> frameVisibleCountBuffers_;
    std::vector<rhi::VulkanBuffer> frameVisibleCountReadbackBuffers_;
    std::vector<uint32_t> frameGpuCullTotalDrawItems_;
    std::vector<uint8_t> frameGpuCullReadbackReady_;
    std::vector<VkFence> imagesInFlight_;
    ShadowSettings shadowSettings_{};
    VkFormat pipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat pipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineColorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat skyboxPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat shadowPipelineDepthFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t currentFrame_ = 0;
    uint32_t bindlessBaseColorFallbackIndex_ = 0;
    uint32_t bindlessNormalFallbackIndex_ = 0;
    uint32_t bindlessMetallicRoughnessFallbackIndex_ = 0;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point lastGpuTimingPrint_ = std::chrono::steady_clock::now();
    std::array<glm::vec4, 6> frameFrustumPlanes_{};
    CullingStats cullingStats_{};
    bool initialized_ = false;
    bool useBindlessMaterialTextures_ = true;
    bool bindlessMaterialTexturesAvailable_ = false;
    bool useGpuCulling_ = true;
    bool gpuCullingAvailable_ = false;
    bool normalMapAssetLoaded_ = false;
    bool metallicRoughnessMapAssetLoaded_ = false;
};

} // namespace ve
