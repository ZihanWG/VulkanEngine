#include "renderer/RenderGraph.h"

#include "renderer/IrradianceProbes.h"
#include "rhi/VulkanShadowMap.h"
#include "rhi/VulkanSwapchain.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace ve::renderer {

namespace {

bool validImageResource(const RenderGraphImageResource& resource)
{
    return resource.image != VK_NULL_HANDLE && resource.imageView != VK_NULL_HANDLE && resource.layout != nullptr &&
           resource.extent.width > 0 && resource.extent.height > 0;
}

void requireImageResource(const RenderGraphImageResource& resource, const char* operation)
{
    if (!validImageResource(resource)) {
        throw std::logic_error(std::string(operation) + " requires a valid " + resource.name + " image resource.");
    }
}

bool validBufferResource(const RenderGraphBufferResource& resource)
{
    return resource.buffer != VK_NULL_HANDLE && resource.size > 0;
}

bool isDepthAspect(VkImageAspectFlags aspectMask)
{
    return (aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
}

VkImageLayout depthAttachmentLayout(VkImageAspectFlags aspectMask)
{
    return (aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                           : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
}

VkImageLayout depthReadOnlyLayout(VkImageAspectFlags aspectMask)
{
    return (aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                           : VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
}

bool accessWrites(RenderResourceAccess access)
{
    return access == RenderResourceAccess::Write || access == RenderResourceAccess::ReadWrite;
}

bool accessReads(RenderResourceAccess access)
{
    return access == RenderResourceAccess::Read || access == RenderResourceAccess::ReadWrite;
}

bool accessMaskWrites(VkAccessFlags2 access)
{
    constexpr VkAccessFlags2 kWriteMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                          VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
    return (access & kWriteMask) != 0;
}

RenderGraph::TextureAccessState accessStateFromLayout(VkImageLayout layout, VkImageAspectFlags aspectMask)
{
    (void)aspectMask;

    RenderGraph::TextureAccessState state{};
    state.layout = layout;

    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        state.stage = VK_PIPELINE_STAGE_2_NONE;
        state.access = VK_ACCESS_2_NONE;
        break;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        state.stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        state.access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        state.declaredAccess = RGAccess::ColorAttachmentWrite;
        break;
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        state.stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        state.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        state.declaredAccess = RGAccess::DepthStencilAttachmentWrite;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        state.declaredAccess = RGAccess::ShaderRead;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        state.declaredAccess = RGAccess::StorageImageReadWrite;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        state.stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        state.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        state.declaredAccess = RGAccess::TransferSrc;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        state.stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        state.access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        state.declaredAccess = RGAccess::TransferDst;
        break;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        state.stage = VK_PIPELINE_STAGE_2_NONE;
        state.access = VK_ACCESS_2_NONE;
        state.declaredAccess = RGAccess::Present;
        break;
    default:
        state.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        state.access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        break;
    }

    return state;
}

std::string passBarrierSummary(uint32_t imageBarrierCount, uint32_t bufferBarrierCount)
{
    return std::to_string(imageBarrierCount) + " inferred image barrier" +
           (imageBarrierCount == 1 ? "" : "s") + ", " + std::to_string(bufferBarrierCount) +
           " inferred buffer barrier" + (bufferBarrierCount == 1 ? "" : "s");
}

} // namespace

RenderGraphBuilder::RenderGraphBuilder(RenderGraph& graph, RenderPassNode& pass)
    : graph_(graph)
    , pass_(pass)
{}

RGTextureHandle RenderGraphBuilder::readTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    graph_.addTextureUsage(pass_, handle, RenderResourceAccess::Read, access, std::move(description));
    return handle;
}

RGTextureHandle RenderGraphBuilder::writeTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    graph_.addTextureUsage(pass_, handle, RenderResourceAccess::Write, access, std::move(description));
    return handle;
}

RGTextureHandle RenderGraphBuilder::readWriteTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    graph_.addTextureUsage(pass_, handle, RenderResourceAccess::ReadWrite, access, std::move(description));
    return handle;
}

RGBufferHandle RenderGraphBuilder::readBuffer(RGBufferHandle handle, RGAccess access, std::string description)
{
    graph_.addBufferUsage(pass_, handle, RenderResourceAccess::Read, access, std::move(description));
    return handle;
}

RGBufferHandle RenderGraphBuilder::writeBuffer(RGBufferHandle handle, RGAccess access, std::string description)
{
    graph_.addBufferUsage(pass_, handle, RenderResourceAccess::Write, access, std::move(description));
    return handle;
}

RGBufferHandle RenderGraphBuilder::readWriteBuffer(RGBufferHandle handle, RGAccess access, std::string description)
{
    graph_.addBufferUsage(pass_, handle, RenderResourceAccess::ReadWrite, access, std::move(description));
    return handle;
}

void RenderGraphBuilder::sideEffect(std::string reason)
{
    pass_.sideEffect = true;
    if (!reason.empty()) {
        pass_.cullReason = std::move(reason);
    }
}

RenderGraphContext::RenderGraphContext(RenderGraph& graph, VkCommandBuffer commandBuffer)
    : graph_(graph)
    , commandBuffer_(commandBuffer)
{}

VkCommandBuffer RenderGraphContext::commandBuffer() const
{
    return commandBuffer_;
}

VkImage RenderGraphContext::image(RGTextureHandle handle) const
{
    return graph_.image(handle);
}

VkImageView RenderGraphContext::imageView(RGTextureHandle handle) const
{
    return graph_.imageView(handle);
}

VkBuffer RenderGraphContext::buffer(RGBufferHandle handle) const
{
    return graph_.buffer(handle);
}

RenderGraph::RenderGraph() = default;

const char* renderPassTypeName(RenderPassType type)
{
    switch (type) {
    case RenderPassType::Shadow:
        return "Shadow";
    case RenderPassType::ShadowGpuCulling:
        return "Shadow GPU Culling";
    case RenderPassType::VolumetricFog:
        return "Volumetric Fog";
    case RenderPassType::ProbeCapture:
        return "Probe Capture";
    case RenderPassType::IrradianceProbes:
        return "Irradiance Probes";
    case RenderPassType::MainGpuCulling:
        return "Main GPU Culling";
    case RenderPassType::DepthPyramid:
        return "Depth Pyramid";
    case RenderPassType::MainHdr:
        return "Main HDR";
    case RenderPassType::Ssr:
        return "SSR";
    case RenderPassType::Gtao:
        return "GTAO";
    case RenderPassType::GtaoBlur:
        return "GTAO Blur";
    case RenderPassType::Transparent:
        return "Transparent";
    case RenderPassType::TaaResolve:
        return "TAA Resolve";
    case RenderPassType::BloomExtract:
        return "Bloom Extract";
    case RenderPassType::BloomBlur:
        return "Bloom Blur";
    case RenderPassType::BloomDownsample:
        return "Bloom Downsample";
    case RenderPassType::BloomUpsample:
        return "Bloom Upsample";
    case RenderPassType::Luminance:
        return "Luminance";
    case RenderPassType::HistogramExposure:
        return "Histogram Exposure";
    case RenderPassType::Composite:
        return "Composite";
    case RenderPassType::ImGui:
        return "ImGui";
    }

    return "Unknown";
}

const char* renderPassExecutionTypeName(RenderPassExecutionType executionType)
{
    switch (executionType) {
    case RenderPassExecutionType::Graphics:
        return "Graphics";
    case RenderPassExecutionType::Compute:
        return "Compute";
    case RenderPassExecutionType::Transfer:
        return "Transfer";
    }

    return "Unknown";
}

const char* renderResourceAccessName(RenderResourceAccess access)
{
    switch (access) {
    case RenderResourceAccess::Read:
        return "Read";
    case RenderResourceAccess::Write:
        return "Write";
    case RenderResourceAccess::ReadWrite:
        return "Read/Write";
    }

    return "Unknown";
}

const char* renderGraphResourceKindName(RGResourceKind kind)
{
    switch (kind) {
    case RGResourceKind::Texture:
        return "Texture";
    case RGResourceKind::Buffer:
        return "Buffer";
    }

    return "Unknown";
}

const char* rgAccessName(RGAccess access)
{
    switch (access) {
    case RGAccess::Unknown:
        return "Unknown";
    case RGAccess::ShaderRead:
        return "ShaderRead";
    case RGAccess::ColorAttachmentWrite:
        return "ColorAttachmentWrite";
    case RGAccess::DepthStencilAttachmentWrite:
        return "DepthStencilAttachmentWrite";
    case RGAccess::StorageImageRead:
        return "StorageImageRead";
    case RGAccess::StorageImageWrite:
        return "StorageImageWrite";
    case RGAccess::StorageImageReadWrite:
        return "StorageImageReadWrite";
    case RGAccess::TransferSrc:
        return "TransferSrc";
    case RGAccess::TransferDst:
        return "TransferDst";
    case RGAccess::Present:
        return "Present";
    case RGAccess::StorageBufferRead:
        return "StorageBufferRead";
    case RGAccess::StorageBufferWrite:
        return "StorageBufferWrite";
    case RGAccess::StorageBufferReadWrite:
        return "StorageBufferReadWrite";
    case RGAccess::IndirectRead:
        return "IndirectRead";
    case RGAccess::HostRead:
        return "HostRead";
    }

    return "Unknown";
}

void RenderGraph::beginFrame(VkCommandBuffer commandBuffer,
                             rhi::VulkanSwapchain& swapchain,
                             rhi::VulkanShadowMap& shadowMap,
                             rhi::VulkanShadowMap* punctualShadowAtlas,
                             uint32_t imageIndex,
                             RenderGraphFrameResources frameResources)
{
    if (frameActive_) {
        throw std::logic_error("RenderGraph::beginFrame called while a frame is already active.");
    }
    if (commandBuffer == VK_NULL_HANDLE) {
        throw std::logic_error("RenderGraph::beginFrame requires a valid command buffer.");
    }
    requireImageResource(frameResources.sceneColor, "RenderGraph::beginFrame");
    requireImageResource(frameResources.velocity, "RenderGraph::beginFrame");
    requireImageResource(frameResources.normalRoughness, "RenderGraph::beginFrame");
    requireImageResource(frameResources.ambientOcclusion, "RenderGraph::beginFrame");
    requireImageResource(frameResources.bloomExtract, "RenderGraph::beginFrame");
    requireImageResource(frameResources.bloomPing, "RenderGraph::beginFrame");
    requireImageResource(frameResources.bloomPong, "RenderGraph::beginFrame");

    textures_.clear();
    buffers_.clear();
    passes_.clear();
    executeCallbacks_.clear();
    debugResources_.clear();

    frame_ = {};
    frame_.commandBuffer = commandBuffer;
    frame_.swapchain = &swapchain;
    frame_.shadowMap = &shadowMap;
    // Only treated as present when it actually has an image; a failed atlas
    // allocation leaves the pointer non-null but invalid.
    frame_.punctualShadowAtlas =
        (punctualShadowAtlas != nullptr && punctualShadowAtlas->valid()) ? punctualShadowAtlas : nullptr;
    frame_.resources = std::move(frameResources);
    frame_.imageIndex = imageIndex;
    frame_.swapchainImage = swapchain.image(imageIndex);

    importExternalFrameTargets(swapchain, shadowMap, frame_.punctualShadowAtlas, imageIndex);
    createTransientFrameTextures();
    importFrameBuffers();

    buildFrameGraphDeclarations();
    compilePassCulling();
    refreshDebugResources();

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    frameActive_ = true;
    activePass_ = ActivePass::None;
}

void RenderGraph::importExternalFrameTargets(rhi::VulkanSwapchain& swapchain,
                                             rhi::VulkanShadowMap& shadowMap,
                                             rhi::VulkanShadowMap* punctualShadowAtlas,
                                             uint32_t imageIndex)
{
    RenderGraphImageResource swapchainColor{};
    swapchainColor.name = "SwapchainColor";
    swapchainColor.image = frame_.swapchainImage;
    swapchainColor.imageView = swapchain.imageView(imageIndex);
    swapchainColor.extent = swapchain.extent();
    swapchainColor.format = swapchain.colorFormat();
    swapchainColor.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           (swapchain.supportsTransferSrc() ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    swapchainColor.imported = true;
    frame_.swapchainColor = importTexture(swapchainColor);
    textures_.at(frame_.swapchainColor.index).owner = TextureOwner::SwapchainColor;
    textures_.at(frame_.swapchainColor.index).initialLayout = swapchain.imageLayout(imageIndex);
    textures_.at(frame_.swapchainColor.index).lastAccess =
        accessStateFromLayout(swapchain.imageLayout(imageIndex), VK_IMAGE_ASPECT_COLOR_BIT);

    RenderGraphImageResource mainDepth{};
    mainDepth.name = "MainDepth";
    mainDepth.image = swapchain.depthImage();
    mainDepth.imageView = swapchain.depthImageView();
    // Not swapchain.extent(): under render scale the depth image is smaller than
    // the presentation images, and this extent becomes the renderArea of every
    // pass that writes it.
    mainDepth.extent = swapchain.depthExtent();
    mainDepth.format = swapchain.depthFormat();
    mainDepth.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      (swapchain.depthSupportsSampling() ? VK_IMAGE_USAGE_SAMPLED_BIT : 0);
    mainDepth.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    mainDepth.imported = true;
    frame_.mainDepth = importTexture(mainDepth);
    textures_.at(frame_.mainDepth.index).owner = TextureOwner::SwapchainDepth;
    textures_.at(frame_.mainDepth.index).initialLayout = swapchain.depthImageLayout();
    textures_.at(frame_.mainDepth.index).lastAccess =
        accessStateFromLayout(swapchain.depthImageLayout(), VK_IMAGE_ASPECT_DEPTH_BIT);

    RenderGraphImageResource shadowDepth{};
    shadowDepth.name = "CascadedShadowMapArray";
    shadowDepth.image = shadowMap.image();
    shadowDepth.imageView = shadowMap.imageView();
    shadowDepth.extent = shadowMap.extent();
    shadowDepth.format = shadowMap.format();
    shadowDepth.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    shadowDepth.arrayLayers = shadowMap.layerCount();
    shadowDepth.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    shadowDepth.imported = true;
    frame_.shadowMapDepth = importTexture(shadowDepth);
    textures_.at(frame_.shadowMapDepth.index).owner = TextureOwner::ShadowMap;
    textures_.at(frame_.shadowMapDepth.index).initialLayout = shadowMap.layout();
    textures_.at(frame_.shadowMapDepth.index).lastAccess =
        accessStateFromLayout(shadowMap.layout(), VK_IMAGE_ASPECT_DEPTH_BIT);

    if (punctualShadowAtlas == nullptr) {
        return;
    }

    RenderGraphImageResource punctualDepth{};
    punctualDepth.name = "PunctualShadowAtlas";
    punctualDepth.image = punctualShadowAtlas->image();
    punctualDepth.imageView = punctualShadowAtlas->imageView();
    punctualDepth.extent = punctualShadowAtlas->extent();
    punctualDepth.format = punctualShadowAtlas->format();
    punctualDepth.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    punctualDepth.arrayLayers = 1;
    punctualDepth.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    punctualDepth.imported = true;
    frame_.punctualShadowAtlasDepth = importTexture(punctualDepth);
    textures_.at(frame_.punctualShadowAtlasDepth.index).owner = TextureOwner::PunctualShadowAtlas;
    textures_.at(frame_.punctualShadowAtlasDepth.index).initialLayout = punctualShadowAtlas->layout();
    textures_.at(frame_.punctualShadowAtlasDepth.index).lastAccess =
        accessStateFromLayout(punctualShadowAtlas->layout(), VK_IMAGE_ASPECT_DEPTH_BIT);
}

void RenderGraph::createTransientFrameTextures()
{
    const auto makeTransientDesc = [](const RenderGraphImageResource& resource) {
        RGTextureDesc desc{};
        desc.name = resource.name;
        desc.format = resource.format;
        desc.extent = resource.extent;
        desc.usage = resource.usage;
        desc.mipLevels = resource.mipLevels;
        desc.arrayLayers = resource.arrayLayers;
        desc.aspectMask = resource.aspectMask;
        desc.clearValue = resource.clearValue;
        desc.hasClearValue = resource.hasClearValue;
        desc.imported = false;
        desc.aliased = resource.aliased;
        return desc;
    };

    frame_.sceneColor = createTransientTexture(makeTransientDesc(frame_.resources.sceneColor), frame_.resources.sceneColor);
    frame_.velocity = createTransientTexture(makeTransientDesc(frame_.resources.velocity), frame_.resources.velocity);
    frame_.normalRoughness = createTransientTexture(makeTransientDesc(frame_.resources.normalRoughness),
                                                    frame_.resources.normalRoughness);
    frame_.ambientOcclusion = createTransientTexture(makeTransientDesc(frame_.resources.ambientOcclusion),
                                                     frame_.resources.ambientOcclusion);
    if (frame_.resources.gtaoEnabled && validImageResource(frame_.resources.ambientOcclusionRaw)) {
        frame_.ambientOcclusionRaw = createTransientTexture(makeTransientDesc(frame_.resources.ambientOcclusionRaw),
                                                            frame_.resources.ambientOcclusionRaw);
    }
    if (frame_.resources.ssrEnabled && validImageResource(frame_.resources.ssrSceneColorCopy)) {
        frame_.ssrSceneColorCopy = createTransientTexture(makeTransientDesc(frame_.resources.ssrSceneColorCopy),
                                                          frame_.resources.ssrSceneColorCopy);
    }
    frame_.postProcessSceneColor = frame_.sceneColor;
    if (frame_.resources.taaEnabled && validImageResource(frame_.resources.taaHistoryRead) &&
        validImageResource(frame_.resources.taaHistoryWrite)) {
        frame_.taaHistoryRead = importTexture(frame_.resources.taaHistoryRead);
        frame_.taaHistoryWrite = importTexture(frame_.resources.taaHistoryWrite);
        if (frame_.taaHistoryWrite.valid()) {
            frame_.postProcessSceneColor = frame_.taaHistoryWrite;
        }
    }
    frame_.bloomExtract =
        createTransientTexture(makeTransientDesc(frame_.resources.bloomExtract), frame_.resources.bloomExtract);
    frame_.bloomPing = createTransientTexture(makeTransientDesc(frame_.resources.bloomPing), frame_.resources.bloomPing);
    frame_.bloomPong = createTransientTexture(makeTransientDesc(frame_.resources.bloomPong), frame_.resources.bloomPong);
    frame_.bloomDownsampleChain.reserve(frame_.resources.bloomDownsampleChain.size());
    for (const RenderGraphImageResource& resource : frame_.resources.bloomDownsampleChain) {
        if (validImageResource(resource)) {
            frame_.bloomDownsampleChain.push_back(createTransientTexture(makeTransientDesc(resource), resource));
        }
    }
    frame_.bloomUpsampleChain.reserve(frame_.resources.bloomUpsampleChain.size());
    for (const RenderGraphImageResource& resource : frame_.resources.bloomUpsampleChain) {
        if (validImageResource(resource)) {
            frame_.bloomUpsampleChain.push_back(createTransientTexture(makeTransientDesc(resource), resource));
        }
    }
    frame_.depthPyramid = importTexture(frame_.resources.depthPyramid);
    // Imported rather than transient: the probe atlases persist across frames --
    // that is the point of amortising probe updates -- so the graph must not
    // treat their contents as discardable.
    frame_.probeIrradianceAtlas = importTexture(frame_.resources.probeIrradianceAtlas);
    frame_.probeDepthAtlas = importTexture(frame_.resources.probeDepthAtlas);
    // The capture targets are scratch -- overwritten every frame that captures --
    // but still imported rather than transient: they are owned by
    // IrradianceProbeVolume, and the graph only aliases memory for the resources
    // it allocates itself.
    frame_.probeCaptureAtlas = importTexture(frame_.resources.probeCaptureAtlas);
    frame_.probeCaptureDepth = importTexture(frame_.resources.probeCaptureDepth);
}

void RenderGraph::importFrameBuffers()
{
    frame_.mainCullInput = importBuffer(frame_.resources.mainCullInput);
    frame_.mainCullIndirectOutput = importBuffer(frame_.resources.mainCullIndirectOutput);
    frame_.mainCullVisibleCounts = importBuffer(frame_.resources.mainCullVisibleCounts);
    frame_.mainCullReadback = importBuffer(frame_.resources.mainCullReadback);
    frame_.luminancePartials = importBuffer(frame_.resources.luminancePartials);
    frame_.luminanceReadback = importBuffer(frame_.resources.luminanceReadback);
    frame_.luminanceHistogram = importBuffer(frame_.resources.luminanceHistogram);
    frame_.histogramReadback = importBuffer(frame_.resources.histogramReadback);
    frame_.exposureState = importBuffer(frame_.resources.exposureState);
}

void RenderGraph::beginShadowPass(uint32_t cascadeLayer)
{
    beginShadowPassInternal(cascadeLayer, /*viewMask=*/0);
}

void RenderGraph::beginLayeredShadowPass(uint32_t cascadeCount)
{
    if (cascadeCount == 0) {
        throw std::invalid_argument("RenderGraph::beginLayeredShadowPass needs at least one cascade.");
    }
    // Every cascade is a view of the same pass, so the mask is the low
    // cascadeCount bits and the attachment is the whole array view.
    beginShadowPassInternal(0, (1u << cascadeCount) - 1u);
}

void RenderGraph::beginShadowPassInternal(uint32_t cascadeLayer, uint32_t viewMask)
{
    requireFrameActive("RenderGraph::beginShadowPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginShadowPass called while another pass is active.");
    }
    const bool layered = viewMask != 0;
    if (layered) {
        // Views index array layers directly, so the mask cannot outrun the array.
        if (viewMask >= (1u << frame_.shadowMap->layerCount())) {
            throw std::out_of_range("RenderGraph::beginLayeredShadowPass view mask exceeds the cascade array.");
        }
    } else if (cascadeLayer >= frame_.shadowMap->layerCount()) {
        throw std::out_of_range("RenderGraph::beginShadowPass cascade layer is out of range.");
    }

    if (!beginDeclaredPass(frame_.passIndices.shadow)) {
        throw std::logic_error("RenderGraph::beginShadowPass was culled but the renderer attempted to record it.");
    }

    VkClearValue shadowDepthClear{};
    shadowDepthClear.depthStencil.depth = 1.0f;
    shadowDepthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo shadowDepthAttachment{};
    shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    // The layered pass attaches the array view and lets the view mask pick the
    // layers; the per-cascade path attaches one layer's own view.
    shadowDepthAttachment.imageView =
        layered ? frame_.shadowMap->imageView() : frame_.shadowMap->layerImageView(cascadeLayer);
    shadowDepthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    shadowDepthAttachment.clearValue = shadowDepthClear;

    VkRenderingInfo shadowRenderingInfo{};
    shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    shadowRenderingInfo.renderArea.offset = {0, 0};
    shadowRenderingInfo.renderArea.extent = frame_.shadowMap->extent();
    // layerCount must be 0 when a view mask is present -- the mask defines the
    // layers instead, and a non-zero count alongside it is invalid.
    shadowRenderingInfo.layerCount = layered ? 0 : 1;
    shadowRenderingInfo.viewMask = viewMask;
    shadowRenderingInfo.colorAttachmentCount = 0;
    shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &shadowRenderingInfo);
    activePass_ = ActivePass::Shadow;
}

void RenderGraph::endShadowPass(bool /*finalCascade*/)
{
    requireFrameActive("RenderGraph::endShadowPass");
    if (activePass_ != ActivePass::Shadow) {
        throw std::logic_error("RenderGraph::endShadowPass called without an active shadow pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginPunctualShadowPass(bool clearWholeAtlas)
{
    requireFrameActive("RenderGraph::beginPunctualShadowPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginPunctualShadowPass called while another pass is active.");
    }
    if (frame_.punctualShadowAtlas == nullptr) {
        throw std::logic_error("RenderGraph::beginPunctualShadowPass requires an imported punctual shadow atlas.");
    }
    if (!beginDeclaredPass(frame_.passIndices.punctualShadow)) {
        throw std::logic_error(
            "RenderGraph::beginPunctualShadowPass was culled but the renderer attempted to record it.");
    }

    VkClearValue atlasDepthClear{};
    atlasDepthClear.depthStencil.depth = 1.0f;
    atlasDepthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo atlasDepthAttachment{};
    atlasDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    atlasDepthAttachment.imageView = frame_.punctualShadowAtlas->imageView();
    atlasDepthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    // CLEAR wipes the whole atlas, which is only correct when nothing in it can
    // be trusted -- the first frame after the image is created. Otherwise LOAD
    // preserves the tiles the caller means to reuse, and the caller clears just
    // the ones it is about to redraw. The graph's write transition uses the
    // tracked current layout rather than UNDEFINED, so contents survive it.
    atlasDepthAttachment.loadOp =
        clearWholeAtlas ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    atlasDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    atlasDepthAttachment.clearValue = atlasDepthClear;

    VkRenderingInfo atlasRenderingInfo{};
    atlasRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    atlasRenderingInfo.renderArea.offset = {0, 0};
    atlasRenderingInfo.renderArea.extent = frame_.punctualShadowAtlas->extent();
    atlasRenderingInfo.layerCount = 1;
    atlasRenderingInfo.colorAttachmentCount = 0;
    atlasRenderingInfo.pDepthAttachment = &atlasDepthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &atlasRenderingInfo);
    activePass_ = ActivePass::PunctualShadow;
}

void RenderGraph::endPunctualShadowPass()
{
    requireFrameActive("RenderGraph::endPunctualShadowPass");
    if (activePass_ != ActivePass::PunctualShadow) {
        throw std::logic_error("RenderGraph::endPunctualShadowPass called without an active punctual shadow pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginVolumetricFogPass()
{
    requireFrameActive("RenderGraph::beginVolumetricFogPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginVolumetricFogPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.volumetricFog)) {
        throw std::logic_error(
            "RenderGraph::beginVolumetricFogPass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::VolumetricFog;
}

void RenderGraph::endVolumetricFogPass()
{
    requireFrameActive("RenderGraph::endVolumetricFogPass");
    if (activePass_ != ActivePass::VolumetricFog) {
        throw std::logic_error("RenderGraph::endVolumetricFogPass called without an active fog pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginProbeCapturePass()
{
    requireFrameActive("RenderGraph::beginProbeCapturePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginProbeCapturePass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.probeCapture)) {
        throw std::logic_error(
            "RenderGraph::beginProbeCapturePass was culled but the renderer attempted to record it.");
    }

    const TextureResource& colorResource = textures_.at(frame_.probeCaptureAtlas.index);
    const TextureResource& depthResource = textures_.at(frame_.probeCaptureDepth.index);

    // Cleared, not loaded. Unlike the punctual shadow atlas there is nothing to
    // preserve: the capture atlas holds only the probes being captured right
    // now, and a tile whose face culled every draw item has to read as empty sky
    // rather than as whatever probe used that row last frame.
    // The clear is what a probe sees in directions with no geometry, so it is
    // the sky term rather than a blank. Black here would be a real error, not a
    // cosmetic one: outdoors most of a probe's hemisphere is sky, and treating
    // it as unlit makes every probe far too dark while still looking like
    // plausible -- just moody -- indirect light.
    //
    // The caller supplies it; see renderGraphFrameResources.
    VkClearValue colorClear = colorResource.desc.clearValue;
    // Alpha is distance, so "nothing here" is the far bound rather than zero --
    // a zero would tell the visibility test every direction is blocked at the
    // probe itself.
    colorClear.color.float32[3] = kProbeMaxDistance;

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = colorResource.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = colorClear;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthResource.imageView;
    depthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = colorResource.desc.extent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::ProbeCapture;
}

void RenderGraph::endProbeCapturePass()
{
    requireFrameActive("RenderGraph::endProbeCapturePass");
    if (activePass_ != ActivePass::ProbeCapture) {
        throw std::logic_error("RenderGraph::endProbeCapturePass called without an active probe capture pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginIrradianceProbePass()
{
    requireFrameActive("RenderGraph::beginIrradianceProbePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginIrradianceProbePass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.irradianceProbes)) {
        throw std::logic_error(
            "RenderGraph::beginIrradianceProbePass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::IrradianceProbes;
}

void RenderGraph::endIrradianceProbePass()
{
    requireFrameActive("RenderGraph::endIrradianceProbePass");
    if (activePass_ != ActivePass::IrradianceProbes) {
        throw std::logic_error("RenderGraph::endIrradianceProbePass called without an active probe pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginMainGpuCullingPass()
{
    requireFrameActive("RenderGraph::beginMainGpuCullingPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginMainGpuCullingPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.mainGpuCulling)) {
        throw std::logic_error(
            "RenderGraph::beginMainGpuCullingPass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::MainGpuCulling;
}

void RenderGraph::endMainGpuCullingPass()
{
    requireFrameActive("RenderGraph::endMainGpuCullingPass");
    if (activePass_ != ActivePass::MainGpuCulling) {
        throw std::logic_error(
            "RenderGraph::endMainGpuCullingPass called without an active main GPU culling pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginMainHdrPass()
{
    requireFrameActive("RenderGraph::beginMainHdrPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginMainHdrPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.mainHdr)) {
        throw std::logic_error("RenderGraph::beginMainHdrPass was culled but the renderer attempted to record it.");
    }

    beginMainHdrRendering(false);
    activePass_ = ActivePass::MainHdr;
}

void RenderGraph::beginMainHdrRendering(bool loadExisting)
{
    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.03f;
    clearColor.color.float32[1] = 0.04f;
    clearColor.color.float32[2] = 0.07f;
    clearColor.color.float32[3] = 1.0f;

    const TextureResource& sceneColor = textures_.at(frame_.sceneColor.index);
    const TextureResource& velocity = textures_.at(frame_.velocity.index);
    const TextureResource& normalRoughness = textures_.at(frame_.normalRoughness.index);

    std::array<VkRenderingAttachmentInfo, 3> colorAttachments{};
    VkRenderingAttachmentInfo& colorAttachment = colorAttachments[0];
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = sceneColor.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearColor;

    VkRenderingAttachmentInfo& velocityAttachment = colorAttachments[1];
    velocityAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    velocityAttachment.imageView = velocity.imageView;
    velocityAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    velocityAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    velocityAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    velocityAttachment.clearValue = VkClearValue{};

    VkRenderingAttachmentInfo& normalRoughnessAttachment = colorAttachments[2];
    normalRoughnessAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    normalRoughnessAttachment.imageView = normalRoughness.imageView;
    normalRoughnessAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    normalRoughnessAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    normalRoughnessAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    normalRoughnessAttachment.clearValue = VkClearValue{};

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = frame_.swapchain->depthImageView();
    depthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    depthAttachment.loadOp = loadExisting ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue = depthClear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = sceneRenderArea(sceneColor.desc.extent);
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
    renderingInfo.pColorAttachments = colorAttachments.data();
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
}

void RenderGraph::endMainHdrPass()
{
    requireFrameActive("RenderGraph::endMainHdrPass");
    if (activePass_ != ActivePass::MainHdr) {
        throw std::logic_error("RenderGraph::endMainHdrPass called without an active main HDR pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginMainHdrPhase2Pass()
{
    requireFrameActive("RenderGraph::beginMainHdrPhase2Pass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginMainHdrPhase2Pass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.mainHdrPhase2)) {
        throw std::logic_error(
            "RenderGraph::beginMainHdrPhase2Pass was culled but the renderer attempted to record it.");
    }

    beginMainHdrRendering(true);
    activePass_ = ActivePass::MainHdrPhase2;
}

void RenderGraph::endMainHdrPhase2Pass()
{
    requireFrameActive("RenderGraph::endMainHdrPhase2Pass");
    if (activePass_ != ActivePass::MainHdrPhase2) {
        throw std::logic_error("RenderGraph::endMainHdrPhase2Pass called without an active phase-2 main HDR pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginSsrCopyPass()
{
    requireFrameActive("RenderGraph::beginSsrCopyPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginSsrCopyPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.ssrCopy)) {
        throw std::logic_error("RenderGraph::beginSsrCopyPass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::SsrCopy;
}

void RenderGraph::endSsrCopyPass()
{
    requireFrameActive("RenderGraph::endSsrCopyPass");
    if (activePass_ != ActivePass::SsrCopy) {
        throw std::logic_error("RenderGraph::endSsrCopyPass called without an active SSR copy pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginSsrTracePass()
{
    requireFrameActive("RenderGraph::beginSsrTracePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginSsrTracePass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.ssrTrace)) {
        throw std::logic_error("RenderGraph::beginSsrTracePass was culled but the renderer attempted to record it.");
    }

    const TextureResource& sceneColor = textures_.at(frame_.sceneColor.index);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = sceneColor.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = sceneRenderArea(sceneColor.desc.extent);
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::SsrTrace;
}

void RenderGraph::endSsrTracePass()
{
    requireFrameActive("RenderGraph::endSsrTracePass");
    if (activePass_ != ActivePass::SsrTrace) {
        throw std::logic_error("RenderGraph::endSsrTracePass called without an active SSR trace pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginGtaoPass()
{
    requireFrameActive("RenderGraph::beginGtaoPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginGtaoPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.gtao)) {
        throw std::logic_error("RenderGraph::beginGtaoPass was culled but the renderer attempted to record it.");
    }

    const TextureResource& rawAo = textures_.at(frame_.ambientOcclusionRaw.index);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = rawAo.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // every texel is overwritten
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = rawAo.desc.extent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::Gtao;
}

void RenderGraph::endGtaoPass()
{
    requireFrameActive("RenderGraph::endGtaoPass");
    if (activePass_ != ActivePass::Gtao) {
        throw std::logic_error("RenderGraph::endGtaoPass called without an active GTAO pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginGtaoBlurPass()
{
    requireFrameActive("RenderGraph::beginGtaoBlurPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginGtaoBlurPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.gtaoBlur)) {
        throw std::logic_error("RenderGraph::beginGtaoBlurPass was culled but the renderer attempted to record it.");
    }

    const TextureResource& ambientOcclusion = textures_.at(frame_.ambientOcclusion.index);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = ambientOcclusion.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // every texel is overwritten
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = sceneRenderArea(ambientOcclusion.desc.extent);
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::GtaoBlur;
}

void RenderGraph::endGtaoBlurPass()
{
    requireFrameActive("RenderGraph::endGtaoBlurPass");
    if (activePass_ != ActivePass::GtaoBlur) {
        throw std::logic_error("RenderGraph::endGtaoBlurPass called without an active GTAO blur pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginTransparentPass()
{
    requireFrameActive("RenderGraph::beginTransparentPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginTransparentPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.transparent)) {
        throw std::logic_error("RenderGraph::beginTransparentPass was culled but the renderer attempted to record it.");
    }

    const TextureResource& sceneColor = textures_.at(frame_.sceneColor.index);
    const TextureResource& velocity = textures_.at(frame_.velocity.index);
    const TextureResource& normalRoughness = textures_.at(frame_.normalRoughness.index);

    // Same three attachments as the main pass, all loaded rather than cleared.
    // Only scene color blends; velocity and the thin G-buffer are overwritten by
    // the transparent fragments. Writing velocity is deliberate -- it is what lets
    // the TAA resolve reproject blended surfaces with their own motion instead of
    // the opaque geometry's. Overwriting the G-buffer is harmless because SSR and
    // GTAO, its only readers, have already run by this point in the frame.
    std::array<VkRenderingAttachmentInfo, 3> colorAttachments{};
    VkRenderingAttachmentInfo& colorAttachment = colorAttachments[0];
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = sceneColor.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo& velocityAttachment = colorAttachments[1];
    velocityAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    velocityAttachment.imageView = velocity.imageView;
    velocityAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    velocityAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    velocityAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo& normalRoughnessAttachment = colorAttachments[2];
    normalRoughnessAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    normalRoughnessAttachment.imageView = normalRoughness.imageView;
    normalRoughnessAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    normalRoughnessAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    normalRoughnessAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    // Loaded and stored but never written: the pipeline disables depth writes, so
    // transparents occlude against opaque geometry without occluding each other.
    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = frame_.swapchain->depthImageView();
    depthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = sceneRenderArea(sceneColor.desc.extent);
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size());
    renderingInfo.pColorAttachments = colorAttachments.data();
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::Transparent;
}

void RenderGraph::endTransparentPass()
{
    requireFrameActive("RenderGraph::endTransparentPass");
    if (activePass_ != ActivePass::Transparent) {
        throw std::logic_error("RenderGraph::endTransparentPass called without an active transparent pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginMainGpuCullingPhase2Pass()
{
    requireFrameActive("RenderGraph::beginMainGpuCullingPhase2Pass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginMainGpuCullingPhase2Pass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.mainGpuCullingPhase2)) {
        throw std::logic_error(
            "RenderGraph::beginMainGpuCullingPhase2Pass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::MainGpuCullingPhase2;
}

void RenderGraph::endMainGpuCullingPhase2Pass()
{
    requireFrameActive("RenderGraph::endMainGpuCullingPhase2Pass");
    if (activePass_ != ActivePass::MainGpuCullingPhase2) {
        throw std::logic_error(
            "RenderGraph::endMainGpuCullingPhase2Pass called without an active phase-2 culling pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginDepthPyramidMidPass()
{
    requireFrameActive("RenderGraph::beginDepthPyramidMidPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginDepthPyramidMidPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.depthPyramidMid)) {
        throw std::logic_error(
            "RenderGraph::beginDepthPyramidMidPass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::DepthPyramidMid;
}

void RenderGraph::endDepthPyramidMidPass()
{
    requireFrameActive("RenderGraph::endDepthPyramidMidPass");
    if (activePass_ != ActivePass::DepthPyramidMid) {
        throw std::logic_error(
            "RenderGraph::endDepthPyramidMidPass called without an active mid-frame depth pyramid pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginDepthPyramidPass()
{
    requireFrameActive("RenderGraph::beginDepthPyramidPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginDepthPyramidPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.depthPyramid)) {
        throw std::logic_error(
            "RenderGraph::beginDepthPyramidPass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::DepthPyramid;
}

void RenderGraph::endDepthPyramidPass()
{
    requireFrameActive("RenderGraph::endDepthPyramidPass");
    if (activePass_ != ActivePass::DepthPyramid) {
        throw std::logic_error("RenderGraph::endDepthPyramidPass called without an active depth pyramid pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginTaaResolvePass()
{
    requireFrameActive("RenderGraph::beginTaaResolvePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginTaaResolvePass called while another pass is active.");
    }
    if (!frame_.taaHistoryWrite.valid()) {
        throw std::logic_error("RenderGraph::beginTaaResolvePass requires a valid TAA history write target.");
    }
    if (!beginDeclaredPass(frame_.passIndices.taaResolve)) {
        throw std::logic_error("RenderGraph::beginTaaResolvePass was culled but the renderer attempted to record it.");
    }

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    // The resolve writes the history in FULL, not in the render sub-rect: it is
    // the pass that turns a low-resolution frame into an output-resolution one.
    beginColorRendering(textures_.at(frame_.taaHistoryWrite.index), clearColor);
    activePass_ = ActivePass::TaaResolve;
}

void RenderGraph::endTaaResolvePass()
{
    requireFrameActive("RenderGraph::endTaaResolvePass");
    if (activePass_ != ActivePass::TaaResolve) {
        throw std::logic_error("RenderGraph::endTaaResolvePass called without an active TAA resolve pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginBloomExtractPass()
{
    requireFrameActive("RenderGraph::beginBloomExtractPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginBloomExtractPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.bloomExtract)) {
        throw std::logic_error("RenderGraph::beginBloomExtractPass was culled but the renderer attempted to record it.");
    }

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginColorRendering(textures_.at(frame_.bloomExtract.index), clearColor);
    activePass_ = ActivePass::BloomExtract;
}

void RenderGraph::endBloomExtractPass()
{
    requireFrameActive("RenderGraph::endBloomExtractPass");
    if (activePass_ != ActivePass::BloomExtract) {
        throw std::logic_error("RenderGraph::endBloomExtractPass called without an active bloom extract pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginBloomBlurPass(bool horizontal)
{
    requireFrameActive("RenderGraph::beginBloomBlurPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginBloomBlurPass called while another pass is active.");
    }
    const uint32_t passIndex = horizontal ? frame_.passIndices.bloomBlurHorizontal : frame_.passIndices.bloomBlurVertical;
    if (!beginDeclaredPass(passIndex)) {
        throw std::logic_error("RenderGraph::beginBloomBlurPass was culled but the renderer attempted to record it.");
    }

    const RGTextureHandle output = horizontal ? frame_.bloomPing : frame_.bloomPong;
    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginColorRendering(textures_.at(output.index), clearColor);
    activePass_ = ActivePass::BloomBlur;
}

void RenderGraph::endBloomBlurPass()
{
    requireFrameActive("RenderGraph::endBloomBlurPass");
    if (activePass_ != ActivePass::BloomBlur) {
        throw std::logic_error("RenderGraph::endBloomBlurPass called without an active bloom blur pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginBloomDownsamplePass(uint32_t level)
{
    requireFrameActive("RenderGraph::beginBloomDownsamplePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginBloomDownsamplePass called while another pass is active.");
    }
    if (level >= frame_.passIndices.bloomDownsampleChain.size() ||
        level >= frame_.bloomDownsampleChain.size()) {
        throw std::out_of_range("RenderGraph::beginBloomDownsamplePass level is out of range.");
    }
    if (!beginDeclaredPass(frame_.passIndices.bloomDownsampleChain[level])) {
        throw std::logic_error(
            "RenderGraph::beginBloomDownsamplePass was culled but the renderer attempted to record it.");
    }

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginColorRendering(textures_.at(frame_.bloomDownsampleChain[level].index), clearColor);
    activePass_ = ActivePass::BloomDownsample;
}

void RenderGraph::endBloomDownsamplePass()
{
    requireFrameActive("RenderGraph::endBloomDownsamplePass");
    if (activePass_ != ActivePass::BloomDownsample) {
        throw std::logic_error(
            "RenderGraph::endBloomDownsamplePass called without an active bloom downsample pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginBloomUpsamplePass(uint32_t level)
{
    requireFrameActive("RenderGraph::beginBloomUpsamplePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginBloomUpsamplePass called while another pass is active.");
    }
    if (level >= frame_.passIndices.bloomUpsampleChain.size() || level >= frame_.bloomUpsampleChain.size()) {
        throw std::out_of_range("RenderGraph::beginBloomUpsamplePass level is out of range.");
    }
    if (!beginDeclaredPass(frame_.passIndices.bloomUpsampleChain[level])) {
        throw std::logic_error(
            "RenderGraph::beginBloomUpsamplePass was culled but the renderer attempted to record it.");
    }

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginColorRendering(textures_.at(frame_.bloomUpsampleChain[level].index), clearColor);
    activePass_ = ActivePass::BloomUpsample;
}

void RenderGraph::endBloomUpsamplePass()
{
    requireFrameActive("RenderGraph::endBloomUpsamplePass");
    if (activePass_ != ActivePass::BloomUpsample) {
        throw std::logic_error("RenderGraph::endBloomUpsamplePass called without an active bloom upsample pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginLuminancePass()
{
    requireFrameActive("RenderGraph::beginLuminancePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginLuminancePass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.luminance)) {
        throw std::logic_error("RenderGraph::beginLuminancePass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::Luminance;
}

void RenderGraph::endLuminancePass()
{
    requireFrameActive("RenderGraph::endLuminancePass");
    if (activePass_ != ActivePass::Luminance) {
        throw std::logic_error("RenderGraph::endLuminancePass called without an active luminance pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginHistogramExposurePass()
{
    requireFrameActive("RenderGraph::beginHistogramExposurePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginHistogramExposurePass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.histogramExposure)) {
        throw std::logic_error(
            "RenderGraph::beginHistogramExposurePass was culled but the renderer attempted to record it.");
    }

    activePass_ = ActivePass::HistogramExposure;
}

void RenderGraph::endHistogramExposurePass()
{
    requireFrameActive("RenderGraph::endHistogramExposurePass");
    if (activePass_ != ActivePass::HistogramExposure) {
        throw std::logic_error(
            "RenderGraph::endHistogramExposurePass called without an active histogram exposure pass.");
    }

    activePass_ = ActivePass::None;
}

void RenderGraph::beginCompositePass()
{
    requireFrameActive("RenderGraph::beginCompositePass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginCompositePass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.composite)) {
        throw std::logic_error("RenderGraph::beginCompositePass was culled but the renderer attempted to record it.");
    }

    VkClearValue clearColor{};
    clearColor.color.float32[0] = 0.0f;
    clearColor.color.float32[1] = 0.0f;
    clearColor.color.float32[2] = 0.0f;
    clearColor.color.float32[3] = 1.0f;
    beginSwapchainRendering(clearColor, VK_ATTACHMENT_LOAD_OP_CLEAR);
    activePass_ = ActivePass::Composite;
}

void RenderGraph::endCompositePass()
{
    requireFrameActive("RenderGraph::endCompositePass");
    if (activePass_ != ActivePass::Composite) {
        throw std::logic_error("RenderGraph::endCompositePass called without an active composite pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::None;
}

void RenderGraph::beginImGuiPass()
{
    requireFrameActive("RenderGraph::beginImGuiPass");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::beginImGuiPass called while another pass is active.");
    }
    if (!beginDeclaredPass(frame_.passIndices.imgui)) {
        throw std::logic_error("RenderGraph::beginImGuiPass was culled but the renderer attempted to record it.");
    }

    VkClearValue clearColor{};
    beginSwapchainRendering(clearColor, VK_ATTACHMENT_LOAD_OP_LOAD);
    activePass_ = ActivePass::ImGui;
}

void RenderGraph::endImGuiPass()
{
    requireFrameActive("RenderGraph::endImGuiPass");
    if (activePass_ != ActivePass::ImGui) {
        throw std::logic_error("RenderGraph::endImGuiPass called without an active ImGui pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    const uint32_t presentBarriers = transitionTexture(frame_.swapchainColor, RGAccess::Present);
    if (frame_.passIndices.imgui != kInvalidRenderGraphHandle) {
        RenderPassNode& pass = passes_.at(frame_.passIndices.imgui);
        pass.generatedBarrierCount += presentBarriers;
        pass.generatedImageBarrierCount += presentBarriers;
        pass.transitionSummary =
            passBarrierSummary(pass.generatedImageBarrierCount, pass.generatedBufferBarrierCount);
    }
    activePass_ = ActivePass::None;
}

void RenderGraph::endFrame()
{
    requireFrameActive("RenderGraph::endFrame");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::endFrame called while a pass is still active.");
    }

    refreshDebugResources();
    VK_CHECK(vkEndCommandBuffer(frame_.commandBuffer));

    frame_ = {};
    frameActive_ = false;
}

uint32_t RenderGraph::addPass(std::string name, SetupCallback setup, ExecuteCallback execute)
{
    return addPass(std::move(name),
                   RenderPassType::MainHdr,
                   RenderPassExecutionType::Graphics,
                   false,
                   std::move(setup),
                   std::move(execute));
}

uint32_t RenderGraph::addPass(std::string name,
                              RenderPassType type,
                              RenderPassExecutionType executionType,
                              bool sideEffect,
                              SetupCallback setup,
                              ExecuteCallback execute)
{
    RenderPassNode pass{};
    pass.name = std::move(name);
    pass.type = type;
    pass.executionType = executionType;
    pass.sideEffect = sideEffect;
    pass.transitionSummary = "Declared; waiting for execution.";

    passes_.push_back(std::move(pass));
    executeCallbacks_.push_back(std::move(execute));
    const uint32_t index = static_cast<uint32_t>(passes_.size() - 1);

    if (setup) {
        RenderGraphBuilder builder(*this, passes_.back());
        setup(builder);
    }

    return index;
}

RGTextureHandle RenderGraph::importTexture(const RenderGraphImageResource& resource)
{
    if (resource.image == VK_NULL_HANDLE || resource.imageView == VK_NULL_HANDLE || resource.extent.width == 0 ||
        resource.extent.height == 0) {
        return {};
    }

    TextureResource texture{};
    texture.desc.name = resource.name;
    texture.desc.format = resource.format;
    texture.desc.extent = resource.extent;
    texture.desc.usage = resource.usage;
    texture.desc.mipLevels = resource.mipLevels;
    texture.desc.arrayLayers = resource.arrayLayers;
    texture.desc.aspectMask = resource.aspectMask;
    texture.desc.clearValue = resource.clearValue;
    texture.desc.hasClearValue = resource.hasClearValue;
    texture.desc.imported = true;
    texture.image = resource.image;
    texture.imageView = resource.imageView;
    texture.externalLayout = resource.layout;
    texture.owner = TextureOwner::ExternalLayoutPointer;
    texture.initialLayout = resource.layout ? *resource.layout : VK_IMAGE_LAYOUT_UNDEFINED;
    texture.lastAccess = accessStateFromLayout(texture.initialLayout, texture.desc.aspectMask);
    texture.graphManaged = false;

    textures_.push_back(texture);
    return RGTextureHandle{static_cast<uint32_t>(textures_.size() - 1)};
}

RGTextureHandle RenderGraph::createTransientTexture(const RGTextureDesc& desc,
                                                    const RenderGraphImageResource& backingResource)
{
    if (backingResource.image == VK_NULL_HANDLE || backingResource.imageView == VK_NULL_HANDLE ||
        backingResource.extent.width == 0 || backingResource.extent.height == 0 || backingResource.layout == nullptr) {
        return {};
    }

    TextureResource texture{};
    texture.desc = desc;
    texture.desc.imported = false;
    texture.image = backingResource.image;
    texture.imageView = backingResource.imageView;
    texture.externalLayout = backingResource.layout;
    texture.owner = TextureOwner::ExternalLayoutPointer;
    texture.initialLayout = *backingResource.layout;
    texture.lastAccess = accessStateFromLayout(texture.initialLayout, texture.desc.aspectMask);
    texture.graphManaged = true;

    textures_.push_back(texture);
    return RGTextureHandle{static_cast<uint32_t>(textures_.size() - 1)};
}

RGBufferHandle RenderGraph::importBuffer(const RenderGraphBufferResource& resource)
{
    if (!validBufferResource(resource)) {
        return {};
    }

    BufferResource bufferResource{};
    bufferResource.desc.name = resource.name;
    bufferResource.desc.size = resource.size;
    bufferResource.desc.usage = resource.usage;
    bufferResource.desc.imported = resource.imported;
    bufferResource.buffer = resource.buffer;
    bufferResource.graphManaged = false;

    buffers_.push_back(bufferResource);
    return RGBufferHandle{static_cast<uint32_t>(buffers_.size() - 1)};
}

VkImage RenderGraph::image(RGTextureHandle handle) const
{
    if (!handle.valid() || handle.index >= textures_.size()) {
        return VK_NULL_HANDLE;
    }
    return textures_[handle.index].image;
}

VkImageView RenderGraph::imageView(RGTextureHandle handle) const
{
    if (!handle.valid() || handle.index >= textures_.size()) {
        return VK_NULL_HANDLE;
    }
    return textures_[handle.index].imageView;
}

VkBuffer RenderGraph::buffer(RGBufferHandle handle) const
{
    if (!handle.valid() || handle.index >= buffers_.size()) {
        return VK_NULL_HANDLE;
    }
    return buffers_[handle.index].buffer;
}

void RenderGraph::requireFrameActive(const char* operation) const
{
    if (!frameActive_) {
        throw std::logic_error(std::string(operation) + " requires an active frame.");
    }
}

void RenderGraph::buildFrameGraphDeclarations()
{
    declareGeometryPasses();
    declareBloomAndTaaPasses();
    declareExposureCompositePasses();
}

void RenderGraph::declareGeometryPasses()
{
    frame_.passIndices.shadow = addPass(
        "CSMShadowPass",
        RenderPassType::Shadow,
        RenderPassExecutionType::Graphics,
        false,
        [this](RenderGraphBuilder& builder) {
            builder.writeTexture(frame_.shadowMapDepth,
                                 RGAccess::DepthStencilAttachmentWrite,
                                 "Writes cascaded shadow-map depth array layers.");
        });

    // Only declared when a light actually got a tile. The atlas texture is
    // still imported and still read by the main pass below, so a frame that
    // casts nothing gets the read-layout transition without the write pass.
    if (frame_.punctualShadowAtlas != nullptr && frame_.resources.punctualShadowSlotCount > 0) {
        frame_.passIndices.punctualShadow = addPass(
            "PunctualShadowAtlasPass",
            RenderPassType::Shadow,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.writeTexture(frame_.punctualShadowAtlasDepth,
                                     RGAccess::DepthStencilAttachmentWrite,
                                     "Writes per-slot spot-light depth tiles into the punctual shadow atlas.");
            });
    }

    if (frame_.resources.volumetricFogEnabled) {
        frame_.passIndices.volumetricFog = addPass(
            "VolumetricFogPass",
            RenderPassType::VolumetricFog,
            RenderPassExecutionType::Compute,
            // Side effect: its output volumes are not graph resources, so
            // nothing downstream declares a read on them and liveness analysis
            // would otherwise cull the pass away.
            true,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.shadowMapDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the cascaded shadow map to shadow the fog froxels.");
                if (frame_.punctualShadowAtlas != nullptr) {
                    // Fog runs between the atlas pass and the main pass, so
                    // without this the atlas is still a depth attachment when
                    // the injection dispatch samples it for light shafts.
                    builder.readTexture(frame_.punctualShadowAtlasDepth,
                                        RGAccess::ShaderRead,
                                        "Samples the punctual shadow atlas for fog light shafts.");
                }
            });
    }

    if (frame_.resources.probeCaptureEnabled && frame_.probeCaptureAtlas.valid() &&
        frame_.probeCaptureDepth.valid()) {
        frame_.passIndices.probeCapture = addPass(
            "ProbeCapture",
            RenderPassType::ProbeCapture,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.shadowMapDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the cascaded shadow map so captured radiance is shadowed.");
                if (frame_.punctualShadowAtlas != nullptr) {
                    // The capture evaluates punctual lights too, so the atlas has
                    // to be out of its depth-attachment layout before this pass
                    // rather than only before the main pass.
                    builder.readTexture(frame_.punctualShadowAtlasDepth,
                                        RGAccess::ShaderRead,
                                        "Samples the punctual shadow atlas so captured radiance is shadowed.");
                }
                if (frame_.probeIrradianceAtlas.valid() && frame_.probeDepthAtlas.valid()) {
                    // Multi-bounce: the capture reads the irradiance the grid
                    // already holds. The update pass writes those same images
                    // later this frame, so what the capture sees is the previous
                    // frame's contents -- which is exactly the feedback wanted,
                    // and why the read has to be declared before that write.
                    builder.readTexture(frame_.probeIrradianceAtlas,
                                        RGAccess::ShaderRead,
                                        "Reads the previous bounce's irradiance so light can bounce again.");
                    builder.readTexture(frame_.probeDepthAtlas,
                                        RGAccess::ShaderRead,
                                        "Reads probe visibility to weight the previous bounce.");
                }
                builder.writeTexture(frame_.probeCaptureAtlas,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes radiance and distance for every face of this frame's probes.");
                builder.writeTexture(frame_.probeCaptureDepth,
                                     RGAccess::DepthStencilAttachmentWrite,
                                     "Resolves which surface each capture texel sees.");
            });
    }

    if (frame_.resources.irradianceProbeUpdateEnabled && frame_.probeIrradianceAtlas.valid() &&
        frame_.probeDepthAtlas.valid()) {
        frame_.passIndices.irradianceProbes = addPass(
            "IrradianceProbeUpdate",
            RenderPassType::IrradianceProbes,
            RenderPassExecutionType::Compute,
            // Side effect, even though both atlases are graph resources with a
            // declared reader. The atlases persist across frames by design, so
            // the pass's real consumer is the *next* frame's update, which
            // liveness analysis cannot see. Without this the pass survives only
            // because the main pass declares a read it does not yet use, and the
            // day that read is conditioned or removed the pass is culled while
            // the renderer still tries to record it -- which throws rather than
            // degrades.
            true,
            [this](RenderGraphBuilder& builder) {
                // Read-write, not write: the border dispatch copies core texels
                // the fill dispatch produced, and later phases blend new radiance
                // against what the atlas already holds.
                builder.readWriteTexture(frame_.probeIrradianceAtlas,
                                         RGAccess::StorageImageReadWrite,
                                         "Writes probe irradiance tiles and wraps their octahedral border.");
                builder.readWriteTexture(frame_.probeDepthAtlas,
                                         RGAccess::StorageImageReadWrite,
                                         "Writes probe visibility tiles and wraps their octahedral border.");
                if (frame_.resources.probeCaptureEnabled && frame_.probeCaptureAtlas.valid()) {
                    // Moves the capture atlas out of the colour-attachment
                    // layout the pass above left it in and into the one the
                    // convolution reads it as.
                    builder.readTexture(frame_.probeCaptureAtlas,
                                        RGAccess::StorageImageRead,
                                        "Reads this frame's captured cube faces to convolve into probe tiles.");
                }
            });
    }

    frame_.passIndices.mainGpuCulling = addPass(
        "MainGpuCullingPass",
        RenderPassType::MainGpuCulling,
        RenderPassExecutionType::Compute,
        true,
        [this](RenderGraphBuilder& builder) {
            builder.readBuffer(frame_.mainCullInput,
                               RGAccess::StorageBufferRead,
                               "Reads per-draw AABB, draw-command, and batch metadata.");
            builder.readTexture(frame_.depthPyramid,
                                RGAccess::ShaderRead,
                                "Optionally samples the previous-frame Hi-Z depth pyramid for occlusion tests.");
            builder.writeBuffer(frame_.mainCullIndirectOutput,
                                RGAccess::StorageBufferWrite,
                                "Writes indirect draw commands for the main pass.");
            builder.writeBuffer(frame_.mainCullVisibleCounts,
                                RGAccess::StorageBufferReadWrite,
                                "Clears and writes visible counts plus culling debug counters.");
            builder.writeBuffer(frame_.mainCullReadback,
                                RGAccess::TransferDst,
                                "Receives copied culling counters for frame-latency CPU readback.");
        });

    frame_.passIndices.mainHdr = addPass(
        "MainHDRPass",
        RenderPassType::MainHdr,
        RenderPassExecutionType::Graphics,
        false,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.shadowMapDepth,
                                RGAccess::ShaderRead,
                                "Samples the cascaded shadow-map array for lighting.");
            if (frame_.punctualShadowAtlas != nullptr) {
                // Declared unconditionally (not just when slots exist) so the
                // atlas always reaches the layout the material descriptors
                // record, even on frames where nothing cast.
                builder.readTexture(frame_.punctualShadowAtlasDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the punctual shadow atlas for spot-light visibility.");
            }
            if (frame_.probeIrradianceAtlas.valid() && frame_.probeDepthAtlas.valid()) {
                // Same asymmetry as the punctual atlas above, for the same
                // reason: declared whenever the atlases exist rather than only
                // when probes updated, so they always reach a sampled layout.
                // Nothing samples them until the shading phase lands; the
                // declaration is what keeps them out of UNDEFINED until then.
                builder.readTexture(frame_.probeIrradianceAtlas,
                                    RGAccess::ShaderRead,
                                    "Samples probe irradiance for indirect diffuse.");
                builder.readTexture(frame_.probeDepthAtlas,
                                    RGAccess::ShaderRead,
                                    "Samples probe visibility to reject leaked probe contributions.");
            }
            if (frame_.ambientOcclusion.valid()) {
                // Read before the GTAO pass writes it later this frame, so what
                // lands here is the previous frame's occlusion, reprojected per
                // fragment in the shader. Declared whenever the target exists,
                // for the same reason as the atlases above: the declaration is
                // what keeps it in a sampled layout rather than UNDEFINED, and
                // the shader gates the fetch on a push constant anyway.
                builder.readTexture(frame_.ambientOcclusion,
                                    RGAccess::ShaderRead,
                                    "Samples the previous frame's ambient occlusion for the ambient term.");
            }
            builder.writeTexture(frame_.sceneColor,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes linear HDR skybox and mesh lighting.");
            builder.writeTexture(frame_.velocity,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes UV-space motion vectors for TAA history reprojection.");
            builder.writeTexture(frame_.normalRoughness,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes the thin G-buffer (normal, roughness, metallic) for SSR.");
            builder.writeTexture(frame_.mainDepth,
                                 RGAccess::DepthStencilAttachmentWrite,
                                 "Clears and writes the main depth attachment.");
            builder.readBuffer(frame_.mainCullIndirectOutput,
                               RGAccess::IndirectRead,
                               "Reads CPU- or GPU-generated indirect draw commands.");
            builder.readBuffer(frame_.mainCullVisibleCounts,
                               RGAccess::IndirectRead,
                               "Reads per-batch visible counts when indirect-count drawing is active.");
        });

    if (frame_.resources.twoPhaseOcclusionEnabled) {
        frame_.passIndices.depthPyramidMid = addPass(
            "DepthPyramidMidPass",
            RenderPassType::DepthPyramid,
            RenderPassExecutionType::Compute,
            true,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Samples phase-1 main depth for the mid-frame Hi-Z rebuild.");
                builder.writeTexture(frame_.depthPyramid,
                                     RGAccess::StorageImageWrite,
                                     "Rebuilds the Hi-Z pyramid so phase 2 can re-test occlusion candidates.");
            });

        frame_.passIndices.mainGpuCullingPhase2 = addPass(
            "MainGpuCullingPhase2",
            RenderPassType::MainGpuCulling,
            RenderPassExecutionType::Compute,
            true,
            [this](RenderGraphBuilder& builder) {
                builder.readBuffer(frame_.mainCullInput,
                                   RGAccess::StorageBufferRead,
                                   "Re-reads per-draw AABBs for the phase-1 occlusion candidates.");
                builder.readTexture(frame_.depthPyramid,
                                    RGAccess::ShaderRead,
                                    "Samples the mid-frame Hi-Z pyramid for the candidate re-test.");
                builder.writeBuffer(frame_.mainCullIndirectOutput,
                                    RGAccess::StorageBufferWrite,
                                    "Writes indirect draw commands for rescued (disoccluded) draws.");
                builder.writeBuffer(frame_.mainCullVisibleCounts,
                                    RGAccess::StorageBufferReadWrite,
                                    "Resets per-batch counts and appends the rescued stats counter.");
                builder.writeBuffer(frame_.mainCullReadback,
                                    RGAccess::TransferDst,
                                    "Receives the combined two-phase culling counters for CPU readback.");
            });

        frame_.passIndices.mainHdrPhase2 = addPass(
            "MainHDRPhase2",
            RenderPassType::MainHdr,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.shadowMapDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the cascaded shadow-map array for lighting.");
                if (frame_.ambientOcclusion.valid()) {
                    builder.readTexture(frame_.ambientOcclusion,
                                        RGAccess::ShaderRead,
                                        "Samples ambient occlusion for the ambient term, as the main pass does.");
                }
                builder.writeTexture(frame_.sceneColor,
                                     RGAccess::ColorAttachmentWrite,
                                     "Draws rescued disoccluded objects into the existing HDR color.");
                builder.writeTexture(frame_.velocity,
                                     RGAccess::ColorAttachmentWrite,
                                     "Appends motion vectors for the rescued draws.");
                builder.writeTexture(frame_.normalRoughness,
                                     RGAccess::ColorAttachmentWrite,
                                     "Appends thin G-buffer data for the rescued draws.");
                builder.writeTexture(frame_.mainDepth,
                                     RGAccess::DepthStencilAttachmentWrite,
                                     "Loads and extends phase-1 depth with the rescued draws.");
                builder.readBuffer(frame_.mainCullIndirectOutput,
                                   RGAccess::IndirectRead,
                                   "Reads the phase-2 compacted indirect draw commands.");
                builder.readBuffer(frame_.mainCullVisibleCounts,
                                   RGAccess::IndirectRead,
                                   "Reads the phase-2 per-batch visible counts.");
            });
    }

    if (frame_.resources.ssrEnabled && frame_.ssrSceneColorCopy.valid()) {
        frame_.passIndices.ssrCopy = addPass(
            "SSRCopyPass",
            RenderPassType::Ssr,
            RenderPassExecutionType::Transfer,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.sceneColor,
                                    RGAccess::TransferSrc,
                                    "Copies the lit opaque scene color as the SSR reflection source.");
                builder.writeTexture(frame_.ssrSceneColorCopy,
                                     RGAccess::TransferDst,
                                     "Receives the scene-color copy the trace samples.");
            });

        frame_.passIndices.ssrTrace = addPass(
            "SSRTracePass",
            RenderPassType::Ssr,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.ssrSceneColorCopy,
                                    RGAccess::ShaderRead,
                                    "Samples the pre-reflection scene color at ray hit points.");
                builder.readTexture(frame_.normalRoughness,
                                    RGAccess::ShaderRead,
                                    "Reads surface normal/roughness/metallic for ray setup and weighting.");
                builder.readTexture(frame_.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Marches rays against the main depth buffer.");
                // Read-modify-write, like the transparent pass: an additive blend
                // reads the destination. Declared write-only, the culler is free to
                // treat the main pass's write to scene colour as dead -- harmless
                // only for as long as nothing else writes scene colour in between.
                // The transparent pass hit exactly this and the note in
                // docs/transparency.md flagged this one as the same latent gap.
                builder.readWriteTexture(frame_.sceneColor,
                                        RGAccess::ColorAttachmentWrite,
                                        "Blends the reflection correction into scene color; additive, and the "
                                        "correction is signed, so it reads what the main pass already wrote.");
            });
    }

    if (frame_.resources.gtaoEnabled && frame_.ambientOcclusion.valid() && frame_.ambientOcclusionRaw.valid()) {
        frame_.passIndices.gtao = addPass(
            "GTAOPass",
            RenderPassType::Gtao,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Reconstructs view positions from the main depth buffer for the horizon search.");
                builder.readTexture(frame_.normalRoughness,
                                    RGAccess::ShaderRead,
                                    "Reads the surface normal for GTAO slice integration.");
                builder.writeTexture(frame_.ambientOcclusionRaw,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes the raw (pre-denoise) GTAO visibility term.");
            });

        frame_.passIndices.gtaoBlur = addPass(
            "GTAOBlurPass",
            RenderPassType::GtaoBlur,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.ambientOcclusionRaw,
                                    RGAccess::ShaderRead,
                                    "Samples the raw GTAO term for depth-aware bilateral denoising.");
                builder.readTexture(frame_.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Samples main depth for the bilateral blur's edge-stopping weights.");
                builder.writeTexture(frame_.ambientOcclusion,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes the denoised GTAO visibility term the composite multiplies in.");
            });
    }

    // Transparents come after SSR and GTAO, which both need an opaque-only depth
    // buffer and G-buffer, and before the TAA resolve so blended edges still get
    // antialiased. They read depth without writing it, so the depth pyramid built
    // below is unaffected.
    if (frame_.resources.transparentDrawCount > 0) {
        frame_.passIndices.transparent = addPass(
            "TransparentPass",
            RenderPassType::Transparent,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.mainDepth,
                                    RGAccess::DepthStencilAttachmentWrite,
                                    "Depth-tests blended geometry against the opaque depth buffer (no writes).");
                if (frame_.ambientOcclusion.valid()) {
                    // Same shared-shader reason as normalRoughness below: this
                    // pass reuses the main fragment shader, so it samples the
                    // occlusion binding whether or not blended surfaces need it.
                    // GTAO already ran and left the image a colour attachment,
                    // so without this declaration it would still be in that
                    // layout when the descriptor is read.
                    builder.readTexture(frame_.ambientOcclusion,
                                        RGAccess::ShaderRead,
                                        "Samples ambient occlusion through the shared main-pass fragment shader.");
                }
                // Read-modify-write, not a plain write: the "over" blend reads the
                // destination. Declaring it write-only makes the pass culler treat
                // every earlier write to scene color -- the main pass, and SSR's
                // additive blend -- as dead, and cull them.
                builder.readWriteTexture(frame_.sceneColor,
                                         RGAccess::ColorAttachmentWrite,
                                         "Alpha-blends sorted transparent geometry over the lit opaque scene color.");
                builder.readWriteTexture(frame_.velocity,
                                         RGAccess::ColorAttachmentWrite,
                                         "Overwrites motion vectors for blended pixels so TAA reprojects them "
                                         "with their own motion; loaded, so opaque velocity survives.");
                builder.readWriteTexture(frame_.normalRoughness,
                                         RGAccess::ColorAttachmentWrite,
                                         "Written because the shared fragment shader emits it; SSR and GTAO, its "
                                         "only readers, already ran earlier this frame.");
            });
    }

    frame_.passIndices.depthPyramid = addPass(
        "DepthPyramidPass",
        RenderPassType::DepthPyramid,
        RenderPassExecutionType::Compute,
        true,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.mainDepth,
                                RGAccess::ShaderRead,
                                "Samples the completed normal-Z main depth buffer.");
            builder.writeTexture(frame_.depthPyramid,
                                 RGAccess::StorageImageWrite,
                                 "Writes the max-depth Hi-Z pyramid for later-frame occlusion culling.");
        });
}

void RenderGraph::declareBloomAndTaaPasses()
{
    if (frame_.taaHistoryRead.valid() && frame_.taaHistoryWrite.valid()) {
        frame_.passIndices.taaResolve = addPass(
            "TAAResolvePass",
            RenderPassType::TaaResolve,
            RenderPassExecutionType::Graphics,
            false,
            [this](RenderGraphBuilder& builder) {
                builder.readTexture(frame_.sceneColor,
                                    RGAccess::ShaderRead,
                                    "Samples the current jittered HDR scene color.");
                builder.readTexture(frame_.taaHistoryRead,
                                    RGAccess::ShaderRead,
                                    "Samples the previous HDR TAA history image.");
                builder.readTexture(frame_.velocity,
                                    RGAccess::ShaderRead,
                                    "Samples motion vectors to reproject the history UV.");
                if (frame_.swapchain->depthSupportsSampling()) {
                    builder.readTexture(frame_.mainDepth,
                                        RGAccess::ShaderRead,
                                        "Samples main depth for closest-depth velocity dilation.");
                }
                builder.writeTexture(frame_.taaHistoryWrite,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes the resolved HDR TAA history image.");
            });
    }

    frame_.passIndices.bloomExtract = addPass(
        "BloomExtractPass",
        RenderPassType::BloomExtract,
        RenderPassExecutionType::Graphics,
        false,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.postProcessSceneColor,
                                RGAccess::ShaderRead,
                                "Samples the active HDR scene color target.");
            builder.writeTexture(frame_.bloomExtract,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes bright pixels above the bloom threshold.");
        });

    frame_.passIndices.bloomBlurHorizontal = addPass(
        "BloomBlurHorizontal",
        RenderPassType::BloomBlur,
        RenderPassExecutionType::Graphics,
        false,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.bloomExtract, RGAccess::ShaderRead, "Samples extracted bloom highlights.");
            builder.writeTexture(frame_.bloomPing,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes the horizontal blur result.");
        });

    frame_.passIndices.bloomBlurVertical = addPass(
        "BloomBlurVertical",
        RenderPassType::BloomBlur,
        RenderPassExecutionType::Graphics,
        false,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.bloomPing, RGAccess::ShaderRead, "Samples the horizontal blur result.");
            builder.writeTexture(frame_.bloomPong,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes the final vertical blur result.");
        });

    frame_.passIndices.bloomDownsampleChain.reserve(frame_.bloomDownsampleChain.size());
    for (uint32_t level = 0; level < frame_.bloomDownsampleChain.size(); ++level) {
        const RGTextureHandle source =
            level == 0 ? frame_.postProcessSceneColor : frame_.bloomDownsampleChain[level - 1];
        const RGTextureHandle output = frame_.bloomDownsampleChain[level];
        frame_.passIndices.bloomDownsampleChain.push_back(addPass(
            "BloomDownsampleMip" + std::to_string(level),
            RenderPassType::BloomDownsample,
            RenderPassExecutionType::Graphics,
            false,
            [source, output, level](RenderGraphBuilder& builder) {
                builder.readTexture(source,
                                    RGAccess::ShaderRead,
                                    level == 0 ? "Samples HDR scene color and extracts conservative bright bloom."
                                               : "Samples the previous bloom mip.");
                builder.writeTexture(output,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes a downsampled bloom mip-chain level.");
            }));
    }

    frame_.passIndices.bloomUpsampleChain.assign(frame_.bloomUpsampleChain.size(), kInvalidRenderGraphHandle);
    for (uint32_t reverseIndex = 0; reverseIndex < frame_.bloomUpsampleChain.size(); ++reverseIndex) {
        const uint32_t level =
            static_cast<uint32_t>(frame_.bloomUpsampleChain.size() - 1u - reverseIndex);
        const RGTextureHandle currentMip = frame_.bloomDownsampleChain[level];
        const RGTextureHandle lowerMip =
            level + 1u == frame_.bloomDownsampleChain.size() - 1u ? frame_.bloomDownsampleChain[level + 1u]
                                                                  : frame_.bloomUpsampleChain[level + 1u];
        const RGTextureHandle output = frame_.bloomUpsampleChain[level];
        frame_.passIndices.bloomUpsampleChain[level] = addPass(
            "BloomUpsampleMip" + std::to_string(level),
            RenderPassType::BloomUpsample,
            RenderPassExecutionType::Graphics,
            false,
            [currentMip, lowerMip, output](RenderGraphBuilder& builder) {
                builder.readTexture(currentMip, RGAccess::ShaderRead, "Samples this bloom mip's local highlights.");
                builder.readTexture(lowerMip, RGAccess::ShaderRead, "Samples the accumulated lower-resolution bloom.");
                builder.writeTexture(output,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes the progressively upsampled bloom chain.");
            });
    }
}

void RenderGraph::declareExposureCompositePasses()
{
    frame_.passIndices.luminance = addPass(
        "LuminancePass",
        RenderPassType::Luminance,
        RenderPassExecutionType::Compute,
        true,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.postProcessSceneColor,
                                RGAccess::ShaderRead,
                                "Samples active scene color for log-average luminance reduction.");
            builder.writeBuffer(frame_.luminancePartials,
                                RGAccess::StorageBufferWrite,
                                "Writes per-workgroup luminance partials.");
        });

    frame_.passIndices.histogramExposure = addPass(
        "HistogramExposurePass",
        RenderPassType::HistogramExposure,
        RenderPassExecutionType::Compute,
        true,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.postProcessSceneColor,
                                RGAccess::ShaderRead,
                                "Samples active scene color for log2 luminance histogram binning.");
            builder.writeBuffer(frame_.luminanceHistogram,
                                RGAccess::StorageBufferReadWrite,
                                "Clears and writes 256 luminance histogram bins.");
            builder.readBuffer(frame_.luminancePartials,
                               RGAccess::StorageBufferRead,
                               "Reads log-average luminance partials for GPU exposure fallback.");
            builder.readWriteBuffer(frame_.exposureState,
                                    RGAccess::StorageBufferReadWrite,
                                    "Reads previous exposure and writes GPU exposure/luminance state.");
        });

    frame_.passIndices.composite = addPass(
        "CompositePass",
        RenderPassType::Composite,
        RenderPassExecutionType::Graphics,
        true,
        [this](RenderGraphBuilder& builder) {
            builder.readTexture(frame_.postProcessSceneColor,
                                RGAccess::ShaderRead,
                                "Samples the active HDR scene color target.");
            builder.readTexture(frame_.bloomPong, RGAccess::ShaderRead, "Samples the legacy blurred bloom texture.");
            if (!frame_.bloomUpsampleChain.empty()) {
                builder.readTexture(frame_.bloomUpsampleChain.front(),
                                    RGAccess::ShaderRead,
                                    "Samples the final mip-chain bloom texture.");
            } else if (!frame_.bloomDownsampleChain.empty()) {
                builder.readTexture(frame_.bloomDownsampleChain.front(),
                                    RGAccess::ShaderRead,
                                    "Samples the single-level mip-chain bloom texture.");
            }
            builder.readBuffer(frame_.exposureState,
                               RGAccess::StorageBufferRead,
                               "Reads GPU exposure state for auto exposure modes.");
            builder.readTexture(frame_.ambientOcclusion,
                                RGAccess::ShaderRead,
                                "Samples the ground-truth ambient-occlusion term applied to scene color.");
            builder.writeTexture(frame_.swapchainColor,
                                 RGAccess::ColorAttachmentWrite,
                                 "Writes the exposed and tone-mapped final color.");
        });

    frame_.passIndices.imgui = addPass(
        "ImGuiPass",
        RenderPassType::ImGui,
        RenderPassExecutionType::Graphics,
        true,
        [this](RenderGraphBuilder& builder) {
            builder.readWriteTexture(frame_.swapchainColor,
                                     RGAccess::ColorAttachmentWrite,
                                     "Loads the composited swapchain image and draws the debug overlay.");
        });
}

void RenderGraph::compilePassCulling()
{
    cullUnusedPasses(passes_, textures_.size(), buffers_.size());
}

std::vector<RenderGraphResourceLifetime> computeTextureLifetimes(const std::vector<RenderPassNode>& passes,
                                                                   size_t textureCount)
{
    std::vector<RenderGraphResourceLifetime> lifetimes(textureCount);

    for (uint32_t passIndex = 0; passIndex < passes.size(); ++passIndex) {
        const RenderPassNode& pass = passes[passIndex];
        if (pass.culled) {
            continue;
        }

        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (usage.resource.kind != RGResourceKind::Texture || usage.resource.index >= lifetimes.size()) {
                // Same tolerance cullUnusedPasses applies: a handle the graph
                // never imported is ignored rather than treated as live.
                continue;
            }

            RenderGraphResourceLifetime& lifetime = lifetimes[usage.resource.index];
            if (!lifetime.used) {
                lifetime.used = true;
                lifetime.firstPass = passIndex;
                lifetime.lastPass = passIndex;
                continue;
            }
            lifetime.firstPass = std::min(lifetime.firstPass, passIndex);
            lifetime.lastPass = std::max(lifetime.lastPass, passIndex);
        }
    }

    return lifetimes;
}

void cullUnusedPasses(std::vector<RenderPassNode>& passes, size_t textureCount, size_t bufferCount)
{
    std::vector<uint8_t> neededTextures(textureCount, 0);
    std::vector<uint8_t> neededBuffers(bufferCount, 0);

    for (auto passIt = passes.rbegin(); passIt != passes.rend(); ++passIt) {
        RenderPassNode& pass = *passIt;
        bool hasWrite = false;
        bool writesNeededOutput = false;

        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (!accessWrites(usage.access)) {
                continue;
            }

            hasWrite = true;
            if (usage.resource.kind == RGResourceKind::Texture && usage.resource.index < neededTextures.size()) {
                writesNeededOutput = writesNeededOutput || neededTextures[usage.resource.index] != 0;
            } else if (usage.resource.kind == RGResourceKind::Buffer &&
                       usage.resource.index < neededBuffers.size()) {
                writesNeededOutput = writesNeededOutput || neededBuffers[usage.resource.index] != 0;
            }
        }

        const bool passLive = pass.sideEffect || writesNeededOutput;
        pass.culled = hasWrite && !passLive;
        if (pass.culled) {
            pass.cullReason = "All declared writes are unused and the pass has no side effects.";
            continue;
        }

        if (!pass.sideEffect && !hasWrite && !writesNeededOutput) {
            pass.culled = true;
            pass.cullReason = "The pass has no side effects and declares no writes.";
            continue;
        }

        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (accessWrites(usage.access)) {
                if (usage.resource.kind == RGResourceKind::Texture && usage.resource.index < neededTextures.size()) {
                    neededTextures[usage.resource.index] = 0;
                } else if (usage.resource.kind == RGResourceKind::Buffer &&
                           usage.resource.index < neededBuffers.size()) {
                    neededBuffers[usage.resource.index] = 0;
                }
            }
            if (accessReads(usage.access)) {
                if (usage.resource.kind == RGResourceKind::Texture && usage.resource.index < neededTextures.size()) {
                    neededTextures[usage.resource.index] = 1;
                } else if (usage.resource.kind == RGResourceKind::Buffer &&
                           usage.resource.index < neededBuffers.size()) {
                    neededBuffers[usage.resource.index] = 1;
                }
            }
        }
    }
}

bool RenderGraph::beginDeclaredPass(uint32_t passIndex)
{
    if (passIndex == kInvalidRenderGraphHandle || passIndex >= passes_.size()) {
        return true;
    }

    RenderPassNode& pass = passes_[passIndex];
    if (pass.culled) {
        pass.executed = false;
        return false;
    }

    uint32_t imageBarrierCount = 0;
    uint32_t bufferBarrierCount = 0;
    for (const RenderResourceUsage& usage : pass.resourceUsages) {
        if (usage.resource.kind == RGResourceKind::Texture) {
            imageBarrierCount += transitionTexture(RGTextureHandle{usage.resource.index}, usage.declaredAccess);
        } else if (usage.resource.kind == RGResourceKind::Buffer) {
            bufferBarrierCount += transitionBuffer(RGBufferHandle{usage.resource.index}, usage.declaredAccess);
        }
    }

    pass.executed = true;
    pass.generatedBarrierCount += imageBarrierCount + bufferBarrierCount;
    pass.generatedImageBarrierCount += imageBarrierCount;
    pass.generatedBufferBarrierCount += bufferBarrierCount;
    pass.transitionSummary =
        passBarrierSummary(pass.generatedImageBarrierCount, pass.generatedBufferBarrierCount);
    return true;
}

uint32_t RenderGraph::transitionTexture(RGTextureHandle handle, RGAccess access)
{
    if (!handle.valid() || handle.index >= textures_.size()) {
        return 0;
    }

    TextureResource& resource = textures_[handle.index];
    if (resource.image == VK_NULL_HANDLE || access == RGAccess::Unknown) {
        return 0;
    }

    const TextureAccessState desired = accessStateForTexture(resource, access);
    VkImageLayout oldLayout = currentTextureLayout(resource);
    TextureAccessState previous = resource.lastAccess;
    if (previous.declaredAccess == RGAccess::Unknown && oldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
        previous = accessStateFromLayout(oldLayout, resource.desc.aspectMask);
    }

    // Alias handoff. The first use of a pool-bound resource in a frame inherits
    // bytes that another resource owned earlier, so two things must hold that do
    // not for a privately allocated image:
    //
    //   - its contents are genuinely undefined, hence UNDEFINED as the old
    //     layout, regardless of what layout this resource was left in last frame;
    //   - the barrier must wait for whatever wrote those bytes, which is not
    //     tracked in this resource's own lastAccess.
    //
    // The source scope is deliberately conservative rather than the exact union
    // of overlapping predecessors. This graph already documents its barriers as
    // conservative and not heavily optimised, tracking predecessors would mean
    // threading the memory plan through the barrier path, and the cost is what
    // Phase 4 measures. Tightening it is a measurement-driven change, not a
    // correctness one.
    const bool aliasHandoff = resource.desc.aliased && !resource.usedThisFrame;
    if (aliasHandoff) {
        oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        previous.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        previous.access = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
        previous.declaredAccess = RGAccess::Unknown;
    }

    if (!aliasHandoff && !textureBarrierRequired(oldLayout,
                                desired.layout,
                                resource.usedThisFrame,
                                previous.declaredAccess,
                                previous.access,
                                desired.access)) {
        resource.lastAccess = desired;
        resource.usedThisFrame = true;
        return 0;
    }

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = previous.stage;
    barrier.srcAccessMask = previous.access;
    barrier.dstStageMask = desired.stage;
    barrier.dstAccessMask = desired.access;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = desired.layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = resource.image;
    barrier.subresourceRange.aspectMask = resource.desc.aspectMask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = resource.desc.mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = resource.desc.arrayLayers;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);
    setTextureLayout(resource, desired.layout);
    resource.lastAccess = desired;
    resource.usedThisFrame = true;
    return 1;
}

uint32_t RenderGraph::transitionBuffer(RGBufferHandle handle, RGAccess access)
{
    if (!handle.valid() || handle.index >= buffers_.size()) {
        return 0;
    }

    BufferResource& resource = buffers_[handle.index];
    if (resource.buffer == VK_NULL_HANDLE || resource.desc.size == 0 || access == RGAccess::Unknown) {
        return 0;
    }

    const BufferAccessState desired = accessStateForBuffer(access);
    if (desired.declaredAccess == RGAccess::Unknown || desired.stage == VK_PIPELINE_STAGE_2_NONE ||
        desired.access == VK_ACCESS_2_NONE) {
        return 0;
    }

    const BufferAccessState previous = resource.lastAccess;
    const bool needsOrdering = bufferBarrierRequired(
        resource.usedThisFrame, previous.declaredAccess, previous.access, desired.access);
    if (!needsOrdering) {
        resource.lastAccess = desired;
        resource.usedThisFrame = true;
        return 0;
    }

    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = previous.stage;
    barrier.srcAccessMask = previous.access;
    barrier.dstStageMask = desired.stage;
    barrier.dstAccessMask = desired.access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = resource.buffer;
    barrier.offset = 0;
    barrier.size = resource.desc.size;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);
    resource.lastAccess = desired;
    resource.usedThisFrame = true;
    return 1;
}

RenderGraph::TextureAccessState RenderGraph::accessStateForTexture(const TextureResource& resource, RGAccess access) const
{
    return textureAccessState(resource.desc.aspectMask, access, currentTextureLayout(resource));
}

TextureAccessState textureAccessState(VkImageAspectFlags aspectMask, RGAccess access, VkImageLayout currentLayout)
{
    TextureAccessState state{};
    state.declaredAccess = access;

    switch (access) {
    case RGAccess::ShaderRead:
        state.layout = isDepthAspect(aspectMask) ? depthReadOnlyLayout(aspectMask)
                                                 : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        break;
    case RGAccess::ColorAttachmentWrite:
        state.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        state.stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        state.access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        break;
    case RGAccess::DepthStencilAttachmentWrite:
        state.layout = depthAttachmentLayout(aspectMask);
        state.stage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        state.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        break;
    case RGAccess::StorageImageRead:
        state.layout = VK_IMAGE_LAYOUT_GENERAL;
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;
    case RGAccess::StorageImageWrite:
        state.layout = VK_IMAGE_LAYOUT_GENERAL;
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    case RGAccess::StorageImageReadWrite:
        state.layout = VK_IMAGE_LAYOUT_GENERAL;
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    case RGAccess::TransferSrc:
        state.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        state.stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        state.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;
    case RGAccess::TransferDst:
        state.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        state.stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        state.access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;
    case RGAccess::Present:
        state.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        state.stage = VK_PIPELINE_STAGE_2_NONE;
        state.access = VK_ACCESS_2_NONE;
        break;
    case RGAccess::Unknown:
    case RGAccess::StorageBufferRead:
    case RGAccess::StorageBufferWrite:
    case RGAccess::StorageBufferReadWrite:
    case RGAccess::IndirectRead:
    case RGAccess::HostRead:
        state.layout = currentLayout;
        state.stage = VK_PIPELINE_STAGE_2_NONE;
        state.access = VK_ACCESS_2_NONE;
        break;
    }

    return state;
}

RenderGraph::BufferAccessState RenderGraph::accessStateForBuffer(RGAccess access) const
{
    return bufferAccessState(access);
}

bool bufferBarrierRequired(bool usedThisFrame,
                           RGAccess previousDeclared,
                           VkAccessFlags2 previousAccess,
                           VkAccessFlags2 desiredAccess)
{
    const bool touched = usedThisFrame || previousDeclared != RGAccess::Unknown;
    return touched && (accessMaskWrites(previousAccess) || accessMaskWrites(desiredAccess));
}

bool textureBarrierRequired(VkImageLayout oldLayout,
                            VkImageLayout desiredLayout,
                            bool usedThisFrame,
                            RGAccess previousDeclared,
                            VkAccessFlags2 previousAccess,
                            VkAccessFlags2 desiredAccess)
{
    if (oldLayout != desiredLayout) {
        return true;
    }
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        return false;
    }
    return bufferBarrierRequired(usedThisFrame, previousDeclared, previousAccess, desiredAccess);
}

BufferAccessState bufferAccessState(RGAccess access)
{
    BufferAccessState state{};
    state.declaredAccess = access;

    constexpr VkPipelineStageFlags2 kShaderBufferStages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    switch (access) {
    case RGAccess::StorageBufferRead:
        state.stage = kShaderBufferStages;
        state.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;
    case RGAccess::StorageBufferWrite:
        state.stage = kShaderBufferStages;
        state.access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    case RGAccess::StorageBufferReadWrite:
        state.stage = kShaderBufferStages;
        state.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    case RGAccess::IndirectRead:
        state.stage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        state.access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        break;
    case RGAccess::TransferSrc:
        state.stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        state.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;
    case RGAccess::TransferDst:
        state.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        state.access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;
    case RGAccess::HostRead:
        state.stage = VK_PIPELINE_STAGE_2_HOST_BIT;
        state.access = VK_ACCESS_2_HOST_READ_BIT;
        break;
    case RGAccess::Unknown:
    case RGAccess::ShaderRead:
    case RGAccess::ColorAttachmentWrite:
    case RGAccess::DepthStencilAttachmentWrite:
    case RGAccess::StorageImageRead:
    case RGAccess::StorageImageWrite:
    case RGAccess::StorageImageReadWrite:
    case RGAccess::Present:
        state.declaredAccess = RGAccess::Unknown;
        state.stage = VK_PIPELINE_STAGE_2_NONE;
        state.access = VK_ACCESS_2_NONE;
        break;
    }

    return state;
}

VkImageLayout RenderGraph::currentTextureLayout(const TextureResource& resource) const
{
    switch (resource.owner) {
    case TextureOwner::ExternalLayoutPointer:
        return resource.externalLayout ? *resource.externalLayout : VK_IMAGE_LAYOUT_UNDEFINED;
    case TextureOwner::SwapchainColor:
        return frame_.swapchain ? frame_.swapchain->imageLayout(frame_.imageIndex) : VK_IMAGE_LAYOUT_UNDEFINED;
    case TextureOwner::SwapchainDepth:
        return frame_.swapchain ? frame_.swapchain->depthImageLayout() : VK_IMAGE_LAYOUT_UNDEFINED;
    case TextureOwner::ShadowMap:
        return frame_.shadowMap ? frame_.shadowMap->layout() : VK_IMAGE_LAYOUT_UNDEFINED;
    case TextureOwner::PunctualShadowAtlas:
        return frame_.punctualShadowAtlas ? frame_.punctualShadowAtlas->layout() : VK_IMAGE_LAYOUT_UNDEFINED;
    }

    return VK_IMAGE_LAYOUT_UNDEFINED;
}

void RenderGraph::setTextureLayout(TextureResource& resource, VkImageLayout layout)
{
    switch (resource.owner) {
    case TextureOwner::ExternalLayoutPointer:
        if (resource.externalLayout) {
            *resource.externalLayout = layout;
        }
        break;
    case TextureOwner::SwapchainColor:
        if (frame_.swapchain) {
            frame_.swapchain->setImageLayout(frame_.imageIndex, layout);
        }
        break;
    case TextureOwner::SwapchainDepth:
        if (frame_.swapchain) {
            frame_.swapchain->setDepthImageLayout(layout);
        }
        break;
    case TextureOwner::ShadowMap:
        if (frame_.shadowMap) {
            frame_.shadowMap->setLayout(layout);
        }
        break;
    case TextureOwner::PunctualShadowAtlas:
        if (frame_.punctualShadowAtlas) {
            frame_.punctualShadowAtlas->setLayout(layout);
        }
        break;
    }
}

RenderResourceHandle RenderGraph::textureResourceHandle(RGTextureHandle handle) const
{
    if (!handle.valid() || handle.index >= textures_.size()) {
        return {};
    }

    return RenderResourceHandle{textures_[handle.index].desc.name, RGResourceKind::Texture, handle.index};
}

RenderResourceHandle RenderGraph::bufferResourceHandle(RGBufferHandle handle) const
{
    if (!handle.valid() || handle.index >= buffers_.size()) {
        return {};
    }

    return RenderResourceHandle{buffers_[handle.index].desc.name, RGResourceKind::Buffer, handle.index};
}

void RenderGraph::addTextureUsage(RenderPassNode& pass,
                                  RGTextureHandle handle,
                                  RenderResourceAccess resourceAccess,
                                  RGAccess declaredAccess,
                                  std::string description)
{
    if (!handle.valid() || handle.index >= textures_.size()) {
        return;
    }

    pass.resourceUsages.push_back(RenderResourceUsage{
        textureResourceHandle(handle),
        resourceAccess,
        declaredAccess,
        std::move(description),
    });
}

void RenderGraph::addBufferUsage(RenderPassNode& pass,
                                 RGBufferHandle handle,
                                 RenderResourceAccess resourceAccess,
                                 RGAccess declaredAccess,
                                 std::string description)
{
    if (!handle.valid() || handle.index >= buffers_.size()) {
        return;
    }

    pass.resourceUsages.push_back(RenderResourceUsage{
        bufferResourceHandle(handle),
        resourceAccess,
        declaredAccess,
        std::move(description),
    });
}

std::vector<RenderGraph::TransientTextureRecord> RenderGraph::transientTextures() const
{
    std::vector<TransientTextureRecord> records;
    records.reserve(textures_.size());
    for (uint32_t index = 0; index < textures_.size(); ++index) {
        const TextureResource& texture = textures_[index];
        // Imported resources are owned outside the graph and are not candidates
        // for the transient pool, whatever their lifetime looks like.
        if (texture.desc.imported || texture.image == VK_NULL_HANDLE) {
            continue;
        }
        records.push_back(TransientTextureRecord{texture.desc.name, texture.image, index});
    }
    return records;
}

void RenderGraph::refreshDebugResources()
{
    debugResources_.clear();
    debugResources_.reserve(textures_.size() + buffers_.size());

    for (const TextureResource& texture : textures_) {
        debugResources_.push_back(RenderGraphResourceDebugInfo{
            texture.desc.name,
            RGResourceKind::Texture,
            texture.desc.format,
            texture.desc.extent,
            0,
            texture.desc.usage,
            0,
            texture.desc.mipLevels,
            texture.desc.arrayLayers,
            texture.desc.imported,
            !texture.desc.imported,
            texture.graphManaged,
            texture.initialLayout,
            currentTextureLayout(texture),
        });
    }

    for (const BufferResource& bufferResource : buffers_) {
        debugResources_.push_back(RenderGraphResourceDebugInfo{
            bufferResource.desc.name,
            RGResourceKind::Buffer,
            VK_FORMAT_UNDEFINED,
            {},
            bufferResource.desc.size,
            0,
            bufferResource.desc.usage,
            1,
            1,
            bufferResource.desc.imported,
            !bufferResource.desc.imported,
            bufferResource.graphManaged,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_UNDEFINED,
        });
    }
}

VkExtent2D RenderGraph::sceneRenderArea(VkExtent2D resourceExtent) const
{
    if (frame_.resources.renderExtent.width == 0 || frame_.resources.renderExtent.height == 0) {
        return resourceExtent;
    }
    return VkExtent2D{std::min(frame_.resources.renderExtent.width, resourceExtent.width),
                      std::min(frame_.resources.renderExtent.height, resourceExtent.height)};
}

void RenderGraph::beginColorRendering(const TextureResource& resource,
                                      VkClearValue clearValue,
                                      VkExtent2D extentOverride)
{
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = resource.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = extentOverride.width != 0 && extentOverride.height != 0
                                          ? VkExtent2D{std::min(extentOverride.width, resource.desc.extent.width),
                                                       std::min(extentOverride.height, resource.desc.extent.height)}
                                          : resource.desc.extent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
}

void RenderGraph::beginSwapchainRendering(VkClearValue clearValue, VkAttachmentLoadOp loadOp)
{
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = frame_.swapchain->imageView(frame_.imageIndex);
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = loadOp;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = frame_.swapchain->extent();
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
}

} // namespace ve::renderer
