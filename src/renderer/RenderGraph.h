#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ve::rhi {

class VulkanShadowMap;
class VulkanSwapchain;

} // namespace ve::rhi

namespace ve::renderer {

inline constexpr uint32_t kInvalidRenderGraphHandle = std::numeric_limits<uint32_t>::max();

struct RGTextureHandle {
    uint32_t index = kInvalidRenderGraphHandle;

    [[nodiscard]] bool valid() const { return index != kInvalidRenderGraphHandle; }
};

struct RGBufferHandle {
    uint32_t index = kInvalidRenderGraphHandle;

    [[nodiscard]] bool valid() const { return index != kInvalidRenderGraphHandle; }
};

enum class RGResourceKind {
    Texture,
    Buffer
};

enum class RGAccess {
    Unknown,
    ShaderRead,
    ColorAttachmentWrite,
    DepthStencilAttachmentWrite,
    StorageImageRead,
    StorageImageWrite,
    StorageImageReadWrite,
    TransferSrc,
    TransferDst,
    Present,
    StorageBufferRead,
    StorageBufferWrite,
    StorageBufferReadWrite,
    IndirectRead,
    HostRead
};

enum class RenderPassType {
    Shadow,
    ShadowGpuCulling,
    VolumetricFog,
    ProbeCapture,
    IrradianceProbes,
    MainGpuCulling,
    DepthPyramid,
    MainHdr,
    Ssr,
    Gtao,
    GtaoBlur,
    Transparent,
    TaaResolve,
    BloomExtract,
    BloomBlur,
    BloomDownsample,
    BloomUpsample,
    Luminance,
    HistogramExposure,
    Composite,
    ImGui
};

enum class RenderPassExecutionType {
    Graphics,
    Compute,
    Transfer
};

enum class RenderResourceAccess {
    Read,
    Write,
    ReadWrite
};

struct RGTextureDesc {
    std::string name;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    VkImageUsageFlags usage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkClearValue clearValue{};
    bool hasClearValue = false;
    bool imported = true;
};

struct RGBufferDesc {
    std::string name;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    bool imported = true;
};

struct RenderResourceHandle {
    std::string name;
    RGResourceKind kind = RGResourceKind::Texture;
    uint32_t index = kInvalidRenderGraphHandle;
};

struct RenderResourceUsage {
    RenderResourceHandle resource{};
    RenderResourceAccess access = RenderResourceAccess::Read;
    RGAccess declaredAccess = RGAccess::Unknown;
    std::string description;
};

struct RenderPassNode {
    std::string name;
    RenderPassType type = RenderPassType::MainHdr;
    std::vector<RenderResourceUsage> resourceUsages;
    RenderPassExecutionType executionType = RenderPassExecutionType::Graphics;
    std::string transitionSummary;
    bool sideEffect = false;
    bool executed = false;
    bool culled = false;
    std::string cullReason;
    uint32_t generatedBarrierCount = 0;
    uint32_t generatedImageBarrierCount = 0;
    uint32_t generatedBufferBarrierCount = 0;
};

struct RenderGraphImageResource {
    std::string name;
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkImageLayout* layout = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkClearValue clearValue{};
    bool hasClearValue = false;
    bool imported = true;
};

struct RenderGraphBufferResource {
    std::string name;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    bool imported = true;
};

struct RenderGraphFrameResources {
    RenderGraphImageResource sceneColor;
    RenderGraphImageResource velocity;
    RenderGraphImageResource normalRoughness;
    RenderGraphImageResource ambientOcclusion;
    RenderGraphImageResource ambientOcclusionRaw;
    RenderGraphImageResource ssrSceneColorCopy;
    RenderGraphImageResource taaHistoryRead;
    RenderGraphImageResource taaHistoryWrite;
    RenderGraphImageResource bloomExtract;
    RenderGraphImageResource bloomPing;
    RenderGraphImageResource bloomPong;
    std::vector<RenderGraphImageResource> bloomDownsampleChain;
    std::vector<RenderGraphImageResource> bloomUpsampleChain;
    RenderGraphImageResource depthPyramid;
    RenderGraphImageResource probeIrradianceAtlas;
    RenderGraphImageResource probeDepthAtlas;
    RenderGraphImageResource probeCaptureAtlas;
    RenderGraphImageResource probeCaptureDepth;
    RenderGraphBufferResource mainCullInput;
    RenderGraphBufferResource mainCullIndirectOutput;
    RenderGraphBufferResource mainCullVisibleCounts;
    RenderGraphBufferResource mainCullReadback;
    RenderGraphBufferResource luminancePartials;
    RenderGraphBufferResource luminanceReadback;
    RenderGraphBufferResource luminanceHistogram;
    RenderGraphBufferResource histogramReadback;
    RenderGraphBufferResource exposureState;
    bool taaEnabled = false;
    // Declares the two-phase occlusion passes (mid-frame depth pyramid, cull
    // phase 2, second main HDR pass) for this frame.
    bool twoPhaseOcclusionEnabled = false;
    // Declares the SSR copy + trace passes for this frame.
    bool ssrEnabled = false;
    // Declares the GTAO horizon-search pass for this frame.
    bool gtaoEnabled = false;
    // Number of alpha-blended draw items this frame. Zero skips declaring the
    // transparent pass entirely, so a fully opaque scene costs nothing.
    uint32_t transparentDrawCount = 0;
    // Punctual shadow atlas tiles allocated this frame. Zero skips declaring the
    // atlas pass, but the atlas texture is still imported and still declared as a
    // main-pass read so the graph transitions it into the layout the material
    // descriptors claim -- otherwise a frame that casts nothing would leave the
    // image in whatever layout it happened to be in.
    uint32_t punctualShadowSlotCount = 0;
    // Declares the volumetric fog compute pass for this frame. It only needs to
    // exist so the graph moves the cascaded shadow map into a sampled layout
    // before the injection dispatch reads it -- without it the fog runs while
    // the map is still a depth attachment.
    bool volumetricFogEnabled = false;
    // Declares the probe-atlas update compute pass for this frame. The two
    // atlases are imported and read by the main pass whenever they exist, the
    // same asymmetry the punctual shadow atlas uses: a frame that updates no
    // probe still has to leave the atlases in the layout their samplers claim.
    bool irradianceProbeUpdateEnabled = false;
    // Declares the probe capture pass: the scene rasterised from each probe in
    // this frame's batch. Separate from the update flag because the update also
    // runs on frames with nothing to capture -- the cold-start seed, and the
    // debug-pattern path.
    bool probeCaptureEnabled = false;
};

struct RenderGraphResourceDebugInfo {
    std::string name;
    RGResourceKind kind = RGResourceKind::Texture;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    VkDeviceSize size = 0;
    VkImageUsageFlags imageUsage = 0;
    VkBufferUsageFlags bufferUsage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    bool imported = true;
    bool transient = false;
    bool graphManaged = false;
    VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

class RenderGraph;

class RenderGraphBuilder final {
public:
    RGTextureHandle readTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    RGTextureHandle writeTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    RGTextureHandle readWriteTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    RGBufferHandle readBuffer(RGBufferHandle handle, RGAccess access, std::string description = {});
    RGBufferHandle writeBuffer(RGBufferHandle handle, RGAccess access, std::string description = {});
    RGBufferHandle readWriteBuffer(RGBufferHandle handle, RGAccess access, std::string description = {});
    void sideEffect(std::string reason = {});

private:
    RenderGraphBuilder(RenderGraph& graph, RenderPassNode& pass);

    RenderGraph& graph_;
    RenderPassNode& pass_;

    friend class RenderGraph;
};

class RenderGraphContext final {
public:
    [[nodiscard]] VkCommandBuffer commandBuffer() const;
    [[nodiscard]] VkImage image(RGTextureHandle handle) const;
    [[nodiscard]] VkImageView imageView(RGTextureHandle handle) const;
    [[nodiscard]] VkBuffer buffer(RGBufferHandle handle) const;

private:
    RenderGraphContext(RenderGraph& graph, VkCommandBuffer commandBuffer);

    RenderGraph& graph_;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;

    friend class RenderGraph;
};

// Render Graph 2.0 keeps the renderer's existing pass recorders in place while
// adding logical handles, declarations, conservative inferred image and buffer
// barriers, transient/imported resource metadata, pass liveness state, and
// debug UI data.
class RenderGraph final {
public:
    using SetupCallback = std::function<void(RenderGraphBuilder&)>;
    using ExecuteCallback = std::function<void(RenderGraphContext&)>;

    RenderGraph();

    // punctualShadowAtlas is nullable: the atlas is an optional subsystem, and a
    // null pointer simply leaves its texture and pass out of this frame's graph.
    void beginFrame(VkCommandBuffer commandBuffer,
                    rhi::VulkanSwapchain& swapchain,
                    rhi::VulkanShadowMap& shadowMap,
                    rhi::VulkanShadowMap* punctualShadowAtlas,
                    uint32_t imageIndex,
                    RenderGraphFrameResources frameResources);
    void beginShadowPass(uint32_t cascadeLayer);
    void endShadowPass(bool finalCascade);
    // Punctual shadow atlas. One rendering scope covers the whole atlas; the
    // caller sets a viewport/scissor per slot inside it.
    //
    // clearWholeAtlas selects the load op. False preserves the tiles already in
    // the atlas so cached ones survive, which is what per-slot invalidation
    // needs -- the caller then clears just the tiles it is about to redraw.
    // True is for the first frame after the image is (re)created, when its
    // contents are undefined and no tile can be trusted.
    void beginPunctualShadowPass(bool clearWholeAtlas);
    void endPunctualShadowPass();
    // Volumetric fog's compute passes. No rendering scope; this exists so the
    // graph transitions the cascaded shadow map the injection pass samples.
    void beginVolumetricFogPass();
    void endVolumetricFogPass();
    // Irradiance probe atlas maintenance. Compute, no rendering scope; the graph
    // moves both atlases into GENERAL for the dispatches and back to a sampled
    // layout for whoever reads them.
    void beginIrradianceProbePass();
    void endIrradianceProbePass();
    // Probe capture. One rendering scope over the whole capture atlas; the
    // caller sets a viewport per (probe, cube face) inside it.
    void beginProbeCapturePass();
    void endProbeCapturePass();
    void beginMainGpuCullingPass();
    void endMainGpuCullingPass();
    void beginMainHdrPass();
    void endMainHdrPass();
    // Two-phase occlusion: mid-frame pyramid build, candidate re-test, and the
    // second (load, no clear) main HDR pass for the rescued draws.
    void beginDepthPyramidMidPass();
    void endDepthPyramidMidPass();
    void beginMainGpuCullingPhase2Pass();
    void endMainGpuCullingPhase2Pass();
    void beginMainHdrPhase2Pass();
    void endMainHdrPhase2Pass();
    // SSR: transfer copy of the lit scene color (the trace's reflection source),
    // then the trace pass blending reflections back into scene color (LOAD).
    void beginSsrCopyPass();
    void endSsrCopyPass();
    void beginSsrTracePass();
    void endSsrTracePass();
    // GTAO: horizon-search fullscreen pass writing the raw AO target, then a
    // bilateral blur denoising it into the composite-visible AO target.
    void beginGtaoPass();
    void endGtaoPass();
    void beginGtaoBlurPass();
    void endGtaoBlurPass();
    // Alpha-blended geometry. Scene color only (no velocity, no G-buffer) with
    // depth test on and depth write off, recorded after every screen-space effect
    // that needs a clean opaque depth/G-buffer and before the TAA resolve.
    void beginTransparentPass();
    void endTransparentPass();
    void beginDepthPyramidPass();
    void endDepthPyramidPass();
    void beginTaaResolvePass();
    void endTaaResolvePass();
    void beginBloomExtractPass();
    void endBloomExtractPass();
    void beginBloomBlurPass(bool horizontal);
    void endBloomBlurPass();
    void beginBloomDownsamplePass(uint32_t level);
    void endBloomDownsamplePass();
    void beginBloomUpsamplePass(uint32_t level);
    void endBloomUpsamplePass();
    void beginLuminancePass();
    void endLuminancePass();
    void beginHistogramExposurePass();
    void endHistogramExposurePass();
    void beginCompositePass();
    void endCompositePass();
    void beginImGuiPass();
    void endImGuiPass();
    void endFrame();

    uint32_t addPass(std::string name, SetupCallback setup, ExecuteCallback execute = {});
    uint32_t addPass(std::string name,
                     RenderPassType type,
                     RenderPassExecutionType executionType,
                     bool sideEffect,
                     SetupCallback setup,
                     ExecuteCallback execute = {});

    RGTextureHandle importTexture(const RenderGraphImageResource& resource);
    RGTextureHandle createTransientTexture(const RGTextureDesc& desc, const RenderGraphImageResource& backingResource);
    RGBufferHandle importBuffer(const RenderGraphBufferResource& resource);

    [[nodiscard]] VkImage image(RGTextureHandle handle) const;
    [[nodiscard]] VkImageView imageView(RGTextureHandle handle) const;
    [[nodiscard]] VkBuffer buffer(RGBufferHandle handle) const;

    [[nodiscard]] const std::vector<RenderPassNode>& passes() const
    {
        return passes_;
    }

    [[nodiscard]] const std::vector<RenderPassNode>& debugPasses() const
    {
        return passes_;
    }

    [[nodiscard]] const std::vector<RenderGraphResourceDebugInfo>& debugResources() const
    {
        return debugResources_;
    }

    struct TextureAccessState {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;
        RGAccess declaredAccess = RGAccess::Unknown;
    };

    struct BufferAccessState {
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;
        RGAccess declaredAccess = RGAccess::Unknown;
    };

private:
    enum class ActivePass {
        None,
        Shadow,
        PunctualShadow,
        VolumetricFog,
        ProbeCapture,
        IrradianceProbes,
        MainGpuCulling,
        MainHdr,
        MainGpuCullingPhase2,
        MainHdrPhase2,
        SsrCopy,
        SsrTrace,
        Gtao,
        GtaoBlur,
        Transparent,
        DepthPyramidMid,
        DepthPyramid,
        TaaResolve,
        BloomExtract,
        BloomBlur,
        BloomDownsample,
        BloomUpsample,
        Luminance,
        HistogramExposure,
        Composite,
        ImGui
    };

    enum class TextureOwner {
        ExternalLayoutPointer,
        SwapchainColor,
        SwapchainDepth,
        ShadowMap,
        PunctualShadowAtlas
    };

    struct TextureResource {
        RGTextureDesc desc;
        VkImage image = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkImageLayout* externalLayout = nullptr;
        TextureOwner owner = TextureOwner::ExternalLayoutPointer;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        TextureAccessState lastAccess{};
        bool usedThisFrame = false;
        bool graphManaged = false;
    };

    struct BufferResource {
        RGBufferDesc desc;
        VkBuffer buffer = VK_NULL_HANDLE;
        BufferAccessState lastAccess{};
        bool usedThisFrame = false;
        bool graphManaged = false;
    };

    struct BuiltinPassIndices {
        uint32_t shadow = kInvalidRenderGraphHandle;
        uint32_t punctualShadow = kInvalidRenderGraphHandle;
        uint32_t volumetricFog = kInvalidRenderGraphHandle;
        uint32_t probeCapture = kInvalidRenderGraphHandle;
        uint32_t irradianceProbes = kInvalidRenderGraphHandle;
        uint32_t mainGpuCulling = kInvalidRenderGraphHandle;
        uint32_t mainHdr = kInvalidRenderGraphHandle;
        uint32_t depthPyramidMid = kInvalidRenderGraphHandle;
        uint32_t mainGpuCullingPhase2 = kInvalidRenderGraphHandle;
        uint32_t mainHdrPhase2 = kInvalidRenderGraphHandle;
        uint32_t ssrCopy = kInvalidRenderGraphHandle;
        uint32_t ssrTrace = kInvalidRenderGraphHandle;
        uint32_t gtao = kInvalidRenderGraphHandle;
        uint32_t gtaoBlur = kInvalidRenderGraphHandle;
        uint32_t transparent = kInvalidRenderGraphHandle;
        uint32_t depthPyramid = kInvalidRenderGraphHandle;
        uint32_t taaResolve = kInvalidRenderGraphHandle;
        uint32_t bloomExtract = kInvalidRenderGraphHandle;
        uint32_t bloomBlurHorizontal = kInvalidRenderGraphHandle;
        uint32_t bloomBlurVertical = kInvalidRenderGraphHandle;
        std::vector<uint32_t> bloomDownsampleChain;
        std::vector<uint32_t> bloomUpsampleChain;
        uint32_t luminance = kInvalidRenderGraphHandle;
        uint32_t histogramExposure = kInvalidRenderGraphHandle;
        uint32_t composite = kInvalidRenderGraphHandle;
        uint32_t imgui = kInvalidRenderGraphHandle;
    };

    struct FrameState {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        rhi::VulkanSwapchain* swapchain = nullptr;
        rhi::VulkanShadowMap* shadowMap = nullptr;
        // Null when the punctual atlas failed to allocate; every use is guarded,
        // and the pass is simply never declared in that case.
        rhi::VulkanShadowMap* punctualShadowAtlas = nullptr;
        RenderGraphFrameResources resources{};
        BuiltinPassIndices passIndices{};
        uint32_t imageIndex = 0;
        VkImage swapchainImage = VK_NULL_HANDLE;
        RGTextureHandle swapchainColor{};
        RGTextureHandle mainDepth{};
        RGTextureHandle shadowMapDepth{};
        RGTextureHandle punctualShadowAtlasDepth{};
        RGTextureHandle sceneColor{};
        RGTextureHandle velocity{};
        RGTextureHandle normalRoughness{};
        RGTextureHandle ambientOcclusion{};
        RGTextureHandle ambientOcclusionRaw{};
        RGTextureHandle ssrSceneColorCopy{};
        RGTextureHandle taaHistoryRead{};
        RGTextureHandle taaHistoryWrite{};
        RGTextureHandle postProcessSceneColor{};
        RGTextureHandle bloomExtract{};
        RGTextureHandle bloomPing{};
        RGTextureHandle bloomPong{};
        std::vector<RGTextureHandle> bloomDownsampleChain;
        std::vector<RGTextureHandle> bloomUpsampleChain;
        RGTextureHandle depthPyramid{};
        RGTextureHandle probeIrradianceAtlas{};
        RGTextureHandle probeDepthAtlas{};
        RGTextureHandle probeCaptureAtlas{};
        RGTextureHandle probeCaptureDepth{};
        RGBufferHandle mainCullInput{};
        RGBufferHandle mainCullIndirectOutput{};
        RGBufferHandle mainCullVisibleCounts{};
        RGBufferHandle mainCullReadback{};
        RGBufferHandle luminancePartials{};
        RGBufferHandle luminanceReadback{};
        RGBufferHandle luminanceHistogram{};
        RGBufferHandle histogramReadback{};
        RGBufferHandle exposureState{};
    };

    void requireFrameActive(const char* operation) const;
    // beginFrame() helpers (see RenderGraph.cpp): import the externally-owned
    // swapchain/shadow targets, create the transient frame textures, and import
    // the per-frame buffers. Verbatim slices of the former monolithic beginFrame.
    void importExternalFrameTargets(rhi::VulkanSwapchain& swapchain,
                                    rhi::VulkanShadowMap& shadowMap,
                                    rhi::VulkanShadowMap* punctualShadowAtlas,
                                    uint32_t imageIndex);
    void createTransientFrameTextures();
    void importFrameBuffers();
    void buildFrameGraphDeclarations();
    // buildFrameGraphDeclarations() groups (see RenderGraph.cpp); each declares a
    // contiguous run of passes in the same order as before.
    void declareGeometryPasses();
    void declareBloomAndTaaPasses();
    void declareExposureCompositePasses();
    void compilePassCulling();
    bool beginDeclaredPass(uint32_t passIndex);
    uint32_t transitionTexture(RGTextureHandle handle, RGAccess access);
    uint32_t transitionBuffer(RGBufferHandle handle, RGAccess access);
    [[nodiscard]] TextureAccessState accessStateForTexture(const TextureResource& resource, RGAccess access) const;
    [[nodiscard]] BufferAccessState accessStateForBuffer(RGAccess access) const;
    [[nodiscard]] VkImageLayout currentTextureLayout(const TextureResource& resource) const;
    void setTextureLayout(TextureResource& resource, VkImageLayout layout);
    [[nodiscard]] RenderResourceHandle textureResourceHandle(RGTextureHandle handle) const;
    [[nodiscard]] RenderResourceHandle bufferResourceHandle(RGBufferHandle handle) const;
    void addTextureUsage(RenderPassNode& pass,
                         RGTextureHandle handle,
                         RenderResourceAccess resourceAccess,
                         RGAccess declaredAccess,
                         std::string description);
    void addBufferUsage(RenderPassNode& pass,
                        RGBufferHandle handle,
                        RenderResourceAccess resourceAccess,
                        RGAccess declaredAccess,
                        std::string description);
    void refreshDebugResources();
    void beginColorRendering(const TextureResource& resource, VkClearValue clearValue);
    // Shared main-HDR dynamic-rendering setup; loadExisting selects LOAD ops for
    // the phase-2 pass instead of the phase-1 clears.
    void beginMainHdrRendering(bool loadExisting);
    void beginSwapchainRendering(VkClearValue clearValue, VkAttachmentLoadOp loadOp);

    FrameState frame_{};
    std::vector<TextureResource> textures_;
    std::vector<BufferResource> buffers_;
    std::vector<RenderPassNode> passes_;
    std::vector<ExecuteCallback> executeCallbacks_;
    std::vector<RenderGraphResourceDebugInfo> debugResources_;
    bool frameActive_ = false;
    ActivePass activePass_ = ActivePass::None;

    friend class RenderGraphBuilder;
    friend class RenderGraphContext;
};

// Backward liveness sweep that marks passes whose declared writes nothing later
// reads. A free function rather than a private method because it is the graph's
// one piece of non-trivial pure logic and needs no device to exercise -- the same
// split ClusterGrid.h, CascadeMath.h, and VolumetricFog.h use.
//
// `textureCount` and `bufferCount` bound the resource indices; a usage pointing
// past either is ignored rather than treated as live, matching how the graph
// tolerates handles it never imported.
void cullUnusedPasses(std::vector<RenderPassNode>& passes, size_t textureCount, size_t bufferCount);

[[nodiscard]] const char* renderPassTypeName(RenderPassType type);
[[nodiscard]] const char* renderPassExecutionTypeName(RenderPassExecutionType executionType);
[[nodiscard]] const char* renderResourceAccessName(RenderResourceAccess access);
[[nodiscard]] const char* renderGraphResourceKindName(RGResourceKind kind);
[[nodiscard]] const char* rgAccessName(RGAccess access);

} // namespace ve::renderer
