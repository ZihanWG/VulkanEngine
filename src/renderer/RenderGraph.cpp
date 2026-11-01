#include "renderer/RenderGraph.h"

#include "core/Logger.h"
#include "renderer/IrradianceProbes.h"
#include "renderer/RayTracedShadows.h"
#include "rhi/VulkanShadowMap.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanSwapchain.h"

#include <algorithm>
#include <chrono>
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
    // Every write bit a declared access can produce. One left out turns that
    // writer's edge into a read-after-read, which emits no barrier at all -- the
    // acceleration-structure write was the most recent way to get it wrong.
    constexpr VkAccessFlags2 kWriteMask =
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT |
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    return (access & kWriteMask) != 0;
}

// Whether some barrier since the last write named every stage in `stage`
// together with every access in `access`. Checked one stage bit at a time,
// because each recorded scope only makes its own accesses visible to its own
// stages.
bool bufferWriteVisibleTo(const BufferSyncState& state, VkPipelineStageFlags2 stage, VkAccessFlags2 access)
{
    for (VkPipelineStageFlags2 remaining = stage; remaining != 0; remaining &= remaining - 1) {
        const VkPipelineStageFlags2 stageBit = remaining & (~remaining + 1);
        VkAccessFlags2 visible = VK_ACCESS_2_NONE;
        for (uint32_t i = 0; i < state.visibleScopeCount; ++i) {
            if ((state.visibleScopes[i].stage & stageBit) != 0) {
                visible |= state.visibleScopes[i].access;
            }
        }
        if ((visible & access) != access) {
            return false;
        }
    }
    return true;
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
        state.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
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
        state.stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        state.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        state.declaredAccess = RGAccess::TransferSrc;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        state.stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        state.access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        state.declaredAccess = RGAccess::TransferDst;
        break;
    // Leaving PRESENT_SRC is the one source scope that is not about a previous
    // command in this command buffer: the previous reader is the presentation
    // engine, and what orders the frame against it is the image-available
    // semaphore, which Renderer waits at COLOR_ATTACHMENT_OUTPUT.
    //
    // NONE here meant the layout transition was ordered against nothing and
    // could execute before that wait was satisfied -- a write racing the
    // presentation engine's read of the same image. Naming the stage the
    // semaphore is waited at is what puts the transition after it.
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        state.stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
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

std::string passBarrierSummary(uint32_t imageBarrierCount, uint32_t bufferBarrierCount, uint32_t submitCount)
{
    return std::to_string(imageBarrierCount) + " inferred image barrier" + (imageBarrierCount == 1 ? "" : "s") + ", " +
           std::to_string(bufferBarrierCount) + " inferred buffer barrier" + (bufferBarrierCount == 1 ? "" : "s") +
           " in " + std::to_string(submitCount) + " submission" + (submitCount == 1 ? "" : "s");
}

} // namespace

RenderGraphBuilder::RenderGraphBuilder(RenderGraph& graph, RenderPassNode& pass) : graph_(graph), pass_(pass)
{}

RGTextureHandle RenderGraphBuilder::readTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    return graph_.addTextureUsage(pass_, handle, RenderResourceAccess::Read, access, std::move(description));
}

RGTextureHandle RenderGraphBuilder::readHistoryTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    return graph_.addTextureUsage(
        pass_, handle, RenderResourceAccess::Read, access, std::move(description), RGReadKind::History);
}

RGTextureHandle
RenderGraphBuilder::readTextureForLayout(RGTextureHandle handle, RGAccess access, std::string description)
{
    return graph_.addTextureUsage(
        pass_, handle, RenderResourceAccess::Read, access, std::move(description), RGReadKind::LayoutOnly);
}

RGTextureHandle RenderGraphBuilder::writeTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    return graph_.addTextureUsage(pass_, handle, RenderResourceAccess::Write, access, std::move(description));
}

RGTextureHandle RenderGraphBuilder::readWriteTexture(RGTextureHandle handle, RGAccess access, std::string description)
{
    return graph_.addTextureUsage(pass_, handle, RenderResourceAccess::ReadWrite, access, std::move(description));
}

RGBufferHandle RenderGraphBuilder::readBuffer(RGBufferHandle handle, RGAccess access, std::string description)
{
    return graph_.addBufferUsage(pass_, handle, RenderResourceAccess::Read, access, std::move(description));
}

RGBufferHandle RenderGraphBuilder::writeBuffer(RGBufferHandle handle, RGAccess access, std::string description)
{
    return graph_.addBufferUsage(pass_, handle, RenderResourceAccess::Write, access, std::move(description));
}

RGBufferHandle RenderGraphBuilder::readWriteBuffer(RGBufferHandle handle, RGAccess access, std::string description)
{
    return graph_.addBufferUsage(pass_, handle, RenderResourceAccess::ReadWrite, access, std::move(description));
}

void RenderGraphBuilder::sideEffect(std::string reason)
{
    pass_.sideEffect = true;
    if (!reason.empty()) {
        pass_.cullReason = std::move(reason);
    }
}

void RenderGraphBuilder::colorAttachment(RGTextureHandle handle, VkAttachmentLoadOp loadOp, VkClearValue clearValue)
{
    pass_.colorAttachment = RGColorAttachment{handle, loadOp, clearValue};
}

void RenderGraphBuilder::transitionAfterPass(RGTextureHandle handle, RGAccess access)
{
    pass_.postPassTransitions.push_back({handle, access});
}

void RenderGraphBuilder::orderAfter(uint32_t passIndex)
{
    if (passIndex == kInvalidRenderGraphHandle) {
        return;
    }
    // The pass being declared is the last one in the list, so anything it can
    // name is earlier. A later index would be an edge pointing forward, which
    // the declaration order cannot express and the schedule would ignore.
    if (passIndex + 1 >= graph_.passes_.size()) {
        throw std::logic_error("RenderGraphBuilder::orderAfter names a pass that is not declared before this one.");
    }
    pass_.orderAfter.push_back(passIndex);
}

RenderGraph::RenderGraph() = default;

const char* renderPassTypeName(RenderPassType type)
{
    switch (type) {
    case RenderPassType::RayTracingSceneBuild:
        return "Ray Tracing Scene Build";
    case RenderPassType::RayTracedShadowTrace:
        return "Ray Traced Shadow Trace";
    case RenderPassType::RayTracedShadowTemporal:
        return "Ray Traced Shadow Temporal";
    case RenderPassType::RayTracedShadowFilter:
        return "Ray Traced Shadow Filter";
    case RenderPassType::RayTracedShadowResolve:
        return "Ray Traced Shadow Resolve";
    case RenderPassType::Shadow:
        return "Shadow";
    case RenderPassType::ShadowGpuCulling:
        return "Shadow GPU Culling";
    case RenderPassType::VsmPageMark:
        return "VSM Page Mark";
    case RenderPassType::VsmPage:
        return "VSM Pages";
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
    case RenderPassType::DepthPrepass:
        return "Depth Prepass";
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
    case RenderPassType::VsmPageCull:
        return "VSM Page Cull";
    case RenderPassType::VsmPagePoolIdle:
        return "VSM Page Pool Idle";
    case RenderPassType::CascadeShadowCache:
        return "Cascade Shadow Cache";
    case RenderPassType::FogVolumeInit:
        return "Fog Volume Init";
    case RenderPassType::ClusterBuild:
        return "Cluster Build";
    case RenderPassType::ProbeShadingParams:
        return "Probe Shading Params";
    case RenderPassType::FrameCapture:
        return "Frame Capture";
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
    case RGAccess::StorageBufferClearAndReadWrite:
        return "StorageBufferClearAndReadWrite";
    case RGAccess::IndirectRead:
        return "IndirectRead";
    case RGAccess::HostRead:
        return "HostRead";
    case RGAccess::AccelerationStructureBuild:
        return "AccelerationStructureBuild";
    case RGAccess::AccelerationStructureRead:
        return "AccelerationStructureRead";
    case RGAccess::AccelerationStructureComputeRead:
        return "AccelerationStructureComputeRead";
    }

    return "Unknown";
}

void RenderGraph::beginFrame(VkCommandBuffer commandBuffer,
                             rhi::VulkanSwapchain& swapchain,
                             rhi::VulkanShadowMap& shadowMap,
                             rhi::VulkanShadowMap* punctualShadowAtlas,
                             rhi::VulkanShadowMap* vsmPagePool,
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
    barrierBatch_.reset();
    declarationIssues_.clear();
    passSchedule_.clear();
    executionOrder_.clear();
    executionOrderCycleDetected_ = false;
    recordedOrder_.clear();
    recordedOrderViolations_.clear();
    unrecordedPassIndices_.clear();
    debugResources_.clear();
    scopes_.clear();
    currentScope_ = kInvalidRenderGraphHandle;
    openScopes_.clear();
    openScopeProfiled_.clear();

    frame_ = {};
    frame_.commandBuffer = commandBuffer;
    frame_.swapchain = &swapchain;
    frame_.shadowMap = &shadowMap;
    // Only treated as present when it actually has an image; a failed atlas
    // allocation leaves the pointer non-null but invalid.
    frame_.punctualShadowAtlas =
        (punctualShadowAtlas != nullptr && punctualShadowAtlas->valid()) ? punctualShadowAtlas : nullptr;
    frame_.vsmPagePool = (vsmPagePool != nullptr && vsmPagePool->valid()) ? vsmPagePool : nullptr;
    frame_.resources = std::move(frameResources);
    frame_.imageIndex = imageIndex;
    frame_.swapchainImage = swapchain.image(imageIndex);

    frame_.punctualShadowAtlasImported = frame_.punctualShadowAtlas != nullptr;
    frame_.vsmPagePoolImported = frame_.vsmPagePool != nullptr;

    importExternalFrameTargets(swapchain, shadowMap, frame_.punctualShadowAtlas, frame_.vsmPagePool, imageIndex);
    createTransientFrameTextures();
    importFrameBuffers();

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
                                             rhi::VulkanShadowMap* vsmPagePool,
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

    if (vsmPagePool != nullptr) {
        RenderGraphImageResource poolDepth{};
        poolDepth.name = "VsmPagePool";
        poolDepth.image = vsmPagePool->image();
        poolDepth.imageView = vsmPagePool->imageView();
        poolDepth.extent = vsmPagePool->extent();
        poolDepth.format = vsmPagePool->format();
        poolDepth.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        poolDepth.arrayLayers = 1;
        poolDepth.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        poolDepth.imported = true;
        frame_.vsmPagePoolDepth = importTexture(poolDepth);
        textures_.at(frame_.vsmPagePoolDepth.index).owner = TextureOwner::VsmPagePool;
        textures_.at(frame_.vsmPagePoolDepth.index).initialLayout = vsmPagePool->layout();
        textures_.at(frame_.vsmPagePoolDepth.index).lastAccess =
            accessStateFromLayout(vsmPagePool->layout(), VK_IMAGE_ASPECT_DEPTH_BIT);
    }

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
        desc.depth = resource.depth;
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

    frame_.sceneColor =
        createTransientTexture(makeTransientDesc(frame_.resources.sceneColor), frame_.resources.sceneColor);
    frame_.velocity = createTransientTexture(makeTransientDesc(frame_.resources.velocity), frame_.resources.velocity);
    frame_.normalRoughness =
        createTransientTexture(makeTransientDesc(frame_.resources.normalRoughness), frame_.resources.normalRoughness);
    frame_.ambientOcclusion =
        createTransientTexture(makeTransientDesc(frame_.resources.ambientOcclusion), frame_.resources.ambientOcclusion);
    if (frame_.resources.gtaoEnabled && validImageResource(frame_.resources.ambientOcclusionRaw)) {
        frame_.ambientOcclusionRaw = createTransientTexture(makeTransientDesc(frame_.resources.ambientOcclusionRaw),
                                                            frame_.resources.ambientOcclusionRaw);
    }
    if (frame_.resources.ssrEnabled && validImageResource(frame_.resources.ssrSceneColorCopy)) {
        frame_.ssrSceneColorCopy = createTransientTexture(makeTransientDesc(frame_.resources.ssrSceneColorCopy),
                                                          frame_.resources.ssrSceneColorCopy);
    }

    if (frame_.resources.taaEnabled && validImageResource(frame_.resources.taaHistoryRead) &&
        validImageResource(frame_.resources.taaHistoryWrite)) {
        frame_.taaHistoryRead = importTexture(frame_.resources.taaHistoryRead);
        frame_.taaHistoryWrite = importTexture(frame_.resources.taaHistoryWrite);
        if (frame_.taaHistoryWrite.valid()) {
            frame_.taaHistoryIsPostProcessSource = true;
        }
    }
    frame_.bloomExtract =
        createTransientTexture(makeTransientDesc(frame_.resources.bloomExtract), frame_.resources.bloomExtract);
    frame_.bloomPing =
        createTransientTexture(makeTransientDesc(frame_.resources.bloomPing), frame_.resources.bloomPing);
    frame_.bloomPong =
        createTransientTexture(makeTransientDesc(frame_.resources.bloomPong), frame_.resources.bloomPong);
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
    // Imported whenever the volumes exist, not only when fog runs: the material
    // descriptor set binds the integrated volume unconditionally, so the main
    // pass declares a layout-only read of it on a frame with fog off and the
    // graph needs the resource to declare against.
    frame_.fogScatterWrite = importTexture(frame_.resources.fogScatterWrite);
    frame_.fogScatterRead = importTexture(frame_.resources.fogScatterRead);
    frame_.fogIntegrated = importTexture(frame_.resources.fogIntegrated);
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
    frame_.shadowCullIndirectOutput = importBuffer(frame_.resources.shadowCullIndirectOutput);
    frame_.shadowCullVisibleCounts = importBuffer(frame_.resources.shadowCullVisibleCounts);
    frame_.shadowCullReadback = importBuffer(frame_.resources.shadowCullReadback);
    frame_.punctualCullIndirectOutput = importBuffer(frame_.resources.punctualCullIndirectOutput);
    frame_.punctualCullVisibleCounts = importBuffer(frame_.resources.punctualCullVisibleCounts);
    frame_.luminancePartials = importBuffer(frame_.resources.luminancePartials);
    frame_.luminanceReadback = importBuffer(frame_.resources.luminanceReadback);
    frame_.luminanceHistogram = importBuffer(frame_.resources.luminanceHistogram);
    frame_.histogramReadback = importBuffer(frame_.resources.histogramReadback);
    frame_.exposureState = importBuffer(frame_.resources.exposureState);
    // Only on a frame that traces: an imported buffer nothing declares is
    // harmless, but a valid handle is also what tells the main passes to
    // declare their read, and that read must not outlive the build.
    if (frame_.resources.rayTracedShadowsEnabled) {
        frame_.rayTracingTopLevel = importBuffer(frame_.resources.rayTracingTopLevel);
    }
    // An image, imported here beside the structure it is traced against. On
    // every frame it exists, unlike the TLAS: see declareRayTracedShadowMaskRead.
    // Imported, not graph-managed, so a frame that reads it without tracing it
    // reads defined contents -- whatever the last traced frame left, which the
    // main pass never uses because the flag that would make it is clear.
    if (frame_.resources.rayTracedShadowMask.image != VK_NULL_HANDLE) {
        frame_.rayTracedShadowMask = importTexture(frame_.resources.rayTracedShadowMask);
    }
    if (frame_.resources.rayTracedShadowMaskEnabled && frame_.resources.rayTracedShadowTrace.image != VK_NULL_HANDLE) {
        frame_.rayTracedShadowTrace = importTexture(frame_.resources.rayTracedShadowTrace);
    }
    if (frame_.resources.rayTracedShadowMaskEnabled && frame_.resources.rayTracedShadowFilterEnabled &&
        frame_.resources.rayTracedShadowFilter.image != VK_NULL_HANDLE) {
        frame_.rayTracedShadowFilter = importTexture(frame_.resources.rayTracedShadowFilter);
        if (frame_.resources.rayTracedShadowHistoryWrite.image != VK_NULL_HANDLE) {
            frame_.rayTracedShadowHistoryWrite = importTexture(frame_.resources.rayTracedShadowHistoryWrite);
        }
        if (frame_.resources.rayTracedShadowTemporalEnabled &&
            frame_.resources.rayTracedShadowHistoryRead.image != VK_NULL_HANDLE) {
            frame_.rayTracedShadowHistoryRead = importTexture(frame_.resources.rayTracedShadowHistoryRead);
        }
    }
}

void RenderGraph::beginVsmPageRendering(bool clearWholePool)
{
    requireExecuting(RenderPassType::VsmPage, "RenderGraph::beginVsmPageRendering");
    if (frame_.vsmPagePool == nullptr) {
        throw std::logic_error("RenderGraph::beginVsmPageRendering requires an imported page pool.");
    }

    VkClearValue poolDepthClear{};
    poolDepthClear.depthStencil.depth = 1.0f;
    poolDepthClear.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo poolDepthAttachment{};
    poolDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    poolDepthAttachment.imageView = frame_.vsmPagePool->imageView();
    poolDepthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    // Same reasoning as the punctual atlas: CLEAR wipes every page, which is
    // only correct when nothing in the pool can be trusted. Otherwise LOAD keeps
    // the cached pages -- which is the entire point of the page grid being
    // absolute -- and the caller clears just the ones it is about to redraw.
    poolDepthAttachment.loadOp = clearWholePool ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    poolDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    poolDepthAttachment.clearValue = poolDepthClear;

    VkRenderingInfo poolRenderingInfo{};
    poolRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    poolRenderingInfo.renderArea.offset = {0, 0};
    poolRenderingInfo.renderArea.extent = frame_.vsmPagePool->extent();
    poolRenderingInfo.layerCount = 1;
    poolRenderingInfo.colorAttachmentCount = 0;
    poolRenderingInfo.pDepthAttachment = &poolDepthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &poolRenderingInfo);
    activePass_ = ActivePass::VsmPage;
}

void RenderGraph::endVsmPageRendering()
{
    requireFrameActive("RenderGraph::endVsmPageRendering");
    if (activePass_ != ActivePass::VsmPage) {
        throw std::logic_error("RenderGraph::endVsmPageRendering called without an active page pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::requireExecuting(RenderPassType type, const char* operation) const
{
    requireFrameActive(operation);
    if (activePass_ != ActivePass::Executor || executingPass_ >= passes_.size() ||
        passes_[executingPass_].type != type) {
        throw std::logic_error(std::string(operation) + " must be called from its pass's execute callback.");
    }
}

void RenderGraph::beginShadowCascadeRendering(uint32_t cascadeLayer)
{
    beginShadowRenderingInternal(cascadeLayer, /*viewMask=*/0);
}

void RenderGraph::beginLayeredShadowRendering(uint32_t cascadeCount)
{
    if (cascadeCount == 0) {
        throw std::invalid_argument("RenderGraph::beginLayeredShadowRendering needs at least one cascade.");
    }
    // Every cascade is a view of the same pass, so the mask is the low
    // cascadeCount bits and the attachment is the whole array view.
    beginShadowRenderingInternal(0, (1u << cascadeCount) - 1u);
}

void RenderGraph::beginShadowRenderingInternal(uint32_t cascadeLayer, uint32_t viewMask)
{
    requireExecuting(RenderPassType::Shadow, "RenderGraph::beginShadowCascadeRendering");
    const bool layered = viewMask != 0;
    if (layered) {
        // Views index array layers directly, so the mask cannot outrun the array.
        if (viewMask >= (1u << frame_.shadowMap->layerCount())) {
            throw std::out_of_range("RenderGraph::beginLayeredShadowRendering view mask exceeds the cascade array.");
        }
    } else if (cascadeLayer >= frame_.shadowMap->layerCount()) {
        throw std::out_of_range("RenderGraph::beginShadowCascadeRendering cascade layer is out of range.");
    }

    // The graph emitted the pass's barriers ahead of the callback, which covers
    // the first cascade. Each later one re-emits them, exactly as a per-cascade
    // begin always did.
    if (executingRenderingScopes_++ > 0 && !beginDeclaredPass(executingPass_)) {
        throw std::logic_error("RenderGraph::beginShadowCascadeRendering reached a culled pass.");
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

void RenderGraph::endShadowRendering()
{
    requireFrameActive("RenderGraph::endShadowRendering");
    if (activePass_ != ActivePass::Shadow) {
        throw std::logic_error("RenderGraph::endShadowRendering called without an open shadow cascade scope.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginPunctualShadowRendering(bool clearWholeAtlas)
{
    requireExecuting(RenderPassType::Shadow, "RenderGraph::beginPunctualShadowRendering");
    if (frame_.punctualShadowAtlas == nullptr) {
        throw std::logic_error("RenderGraph::beginPunctualShadowRendering requires an imported punctual shadow atlas.");
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
    atlasDepthAttachment.loadOp = clearWholeAtlas ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
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

void RenderGraph::endPunctualShadowRendering()
{
    requireFrameActive("RenderGraph::endPunctualShadowRendering");
    if (activePass_ != ActivePass::PunctualShadow) {
        throw std::logic_error(
            "RenderGraph::endPunctualShadowRendering called without an active punctual shadow pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginProbeCaptureRendering()
{
    requireExecuting(RenderPassType::ProbeCapture, "RenderGraph::beginProbeCaptureRendering");

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

void RenderGraph::endProbeCaptureRendering()
{
    requireFrameActive("RenderGraph::endProbeCaptureRendering");
    if (activePass_ != ActivePass::ProbeCapture) {
        throw std::logic_error("RenderGraph::endProbeCaptureRendering called without an active probe capture pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginMainHdrPassRendering()
{
    requireExecuting(RenderPassType::MainHdr, "RenderGraph::beginMainHdrPassRendering");

    // Colour is cleared as always; depth is kept when the prepass has already
    // filled it. Clearing it here instead would throw away the entire pass.
    beginMainHdrRendering(false, declaresPassOfType(RenderPassType::DepthPrepass));
    activePass_ = ActivePass::MainHdr;
}

void RenderGraph::beginDepthPrepassRendering()
{
    requireExecuting(RenderPassType::DepthPrepass, "RenderGraph::beginDepthPrepassRendering");

    VkClearValue depthClear{};
    depthClear.depthStencil.depth = 1.0f;
    depthClear.depthStencil.stencil = 0;

    // Depth only: no colour attachment at all, which is what makes this cheaper
    // than the pass it exists to speed up.
    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = frame_.swapchain->depthImageView();
    depthAttachment.imageLayout = depthAttachmentLayout(VK_IMAGE_ASPECT_DEPTH_BIT);
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue = depthClear;

    const TextureResource& sceneColor = textures_.at(frame_.sceneColor.index);
    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    // The same sub-rect the main pass uses, so a render-scale change moves both
    // together. A prepass covering a different area than the pass reading its
    // depth would leave a band of stale or cleared depth along the seam.
    renderingInfo.renderArea.extent = sceneRenderArea(sceneColor.desc.extent);
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(frame_.commandBuffer, &renderingInfo);
    activePass_ = ActivePass::DepthPrepass;
}

void RenderGraph::endDepthPrepassRendering()
{
    requireFrameActive("RenderGraph::endDepthPrepassRendering");
    if (activePass_ != ActivePass::DepthPrepass) {
        throw std::logic_error("RenderGraph::endDepthPrepassRendering called without an active depth prepass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginMainHdrRendering(bool loadExisting, bool loadDepth)
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
    depthAttachment.loadOp = loadDepth ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
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

void RenderGraph::endMainHdrPassRendering()
{
    requireFrameActive("RenderGraph::endMainHdrPassRendering");
    if (activePass_ != ActivePass::MainHdr) {
        throw std::logic_error("RenderGraph::endMainHdrPassRendering called without an active main HDR pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginMainHdrPhase2Rendering()
{
    requireExecuting(RenderPassType::MainHdr, "RenderGraph::beginMainHdrPhase2Rendering");

    // Phase 2 keeps everything phase 1 left behind, depth included.
    beginMainHdrRendering(true, true);
    activePass_ = ActivePass::MainHdrPhase2;
}

void RenderGraph::endMainHdrPhase2Rendering()
{
    requireFrameActive("RenderGraph::endMainHdrPhase2Rendering");
    if (activePass_ != ActivePass::MainHdrPhase2) {
        throw std::logic_error(
            "RenderGraph::endMainHdrPhase2Rendering called without an active phase-2 main HDR pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginSsrTraceRendering()
{
    requireExecuting(RenderPassType::Ssr, "RenderGraph::beginSsrTraceRendering");

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

void RenderGraph::endSsrTraceRendering()
{
    requireFrameActive("RenderGraph::endSsrTraceRendering");
    if (activePass_ != ActivePass::SsrTrace) {
        throw std::logic_error("RenderGraph::endSsrTraceRendering called without an active SSR trace pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginGtaoRendering()
{
    requireExecuting(RenderPassType::Gtao, "RenderGraph::beginGtaoRendering");

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

void RenderGraph::endGtaoRendering()
{
    requireFrameActive("RenderGraph::endGtaoRendering");
    if (activePass_ != ActivePass::Gtao) {
        throw std::logic_error("RenderGraph::endGtaoRendering called without an active GTAO pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginGtaoBlurRendering()
{
    requireExecuting(RenderPassType::GtaoBlur, "RenderGraph::beginGtaoBlurRendering");

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

void RenderGraph::endGtaoBlurRendering()
{
    requireFrameActive("RenderGraph::endGtaoBlurRendering");
    if (activePass_ != ActivePass::GtaoBlur) {
        throw std::logic_error("RenderGraph::endGtaoBlurRendering called without an active GTAO blur pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::beginTransparentRendering()
{
    requireExecuting(RenderPassType::Transparent, "RenderGraph::beginTransparentRendering");

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

void RenderGraph::endTransparentRendering()
{
    requireFrameActive("RenderGraph::endTransparentRendering");
    if (activePass_ != ActivePass::Transparent) {
        throw std::logic_error("RenderGraph::endTransparentRendering called without an active transparent pass.");
    }

    vkCmdEndRendering(frame_.commandBuffer);
    activePass_ = ActivePass::Executor;
}

void RenderGraph::endFrame()
{
    requireFrameActive("RenderGraph::endFrame");
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::endFrame called while a pass is still active.");
    }

    // The backstop. The declarations in buildFrameGraphDeclarations and the
    // recording spread across the renderer's translation units are two sequences
    // maintained by hand; this is what notices them drifting apart. Reported, not
    // thrown, for the same reason validateDeclarations reports: a graph that
    // refuses to render is a worse diagnostic than one that renders and says
    // what looks wrong.
    recordedOrderViolations_ = validatePassOrder(passSchedule_, recordedOrder_);
    unrecordedPassIndices_ = unrecordedPasses(passSchedule_, recordedOrder_);

    // Roll the frame into the run-level summary before the next beginFrame
    // clears the vectors. Reported once at shutdown; see
    // RenderGraphBackstopSummary for why peaks and not sums.
    ++backstopSummary_.framesObserved;
    backstopSummary_.peakOrderViolations =
        std::max(backstopSummary_.peakOrderViolations, recordedOrderViolations_.size());
    backstopSummary_.peakUnrecordedPasses =
        std::max(backstopSummary_.peakUnrecordedPasses, unrecordedPassIndices_.size());
    backstopSummary_.peakDeclarationIssues =
        std::max(backstopSummary_.peakDeclarationIssues, declarationIssues_.size());
    backstopSummary_.executionOrderCycleDetected =
        backstopSummary_.executionOrderCycleDetected || executionOrderCycleDetected_;
    if (!recordedOrderViolations_.empty() || !unrecordedPassIndices_.empty() || !declarationIssues_.empty() ||
        executionOrderCycleDetected_) {
        ++backstopSummary_.framesWithIssues;
        if (backstopSummary_.firstIssueDetails.empty()) {
            backstopSummary_.firstIssueDetails = describeBackstopFindings();
        }
    }

    refreshDebugResources();
    VK_CHECK(vkEndCommandBuffer(frame_.commandBuffer));

    frame_ = {};
    frameActive_ = false;
}

uint32_t RenderGraph::addPass(std::string name, SetupCallback setup)
{
    return addPass(
        std::move(name), RenderPassType::MainHdr, RenderPassExecutionType::Graphics, false, std::move(setup));
}

uint32_t RenderGraph::addPass(
    std::string name, RenderPassType type, RenderPassExecutionType executionType, bool sideEffect, SetupCallback setup)
{
    RenderPassNode pass{};
    pass.name = std::move(name);
    pass.type = type;
    pass.executionType = executionType;
    pass.sideEffect = sideEffect;
    pass.transitionSummary = "Declared; waiting for execution.";
    pass.scope = currentScope_;

    passes_.push_back(std::move(pass));
    const uint32_t index = static_cast<uint32_t>(passes_.size() - 1);

    if (setup) {
        RenderGraphBuilder builder(*this, passes_.back());
        setup(builder);
    }

    return index;
}

uint32_t RenderGraph::addPass(std::string name,
                              RenderPassType type,
                              RenderPassExecutionType executionType,
                              bool sideEffect,
                              SetupCallback setup,
                              RenderGraphPassExecutor executor)
{
    const uint32_t index = addPass(std::move(name), type, executionType, sideEffect, std::move(setup));
    RenderPassNode& node = passes_[index];
    node.execute = std::move(executor.execute);
    node.executeLabel = std::move(executor.label);
    node.executeProfileScope = executor.profileScope;
    return index;
}

RenderGraphPassContext::RenderGraphPassContext(const RenderGraph& graph,
                                               VkCommandBuffer commandBuffer,
                                               uint32_t passIndex)
    : graph_(graph), commandBuffer_(commandBuffer), passIndex_(passIndex)
{}

VkImage RenderGraphPassContext::image(RGTextureHandle handle) const
{
    return graph_.image(handle);
}

VkImageView RenderGraphPassContext::imageView(RGTextureHandle handle) const
{
    return graph_.imageView(handle);
}

VkBuffer RenderGraphPassContext::buffer(RGBufferHandle handle) const
{
    return graph_.buffer(handle);
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
    texture.desc.depth = resource.depth;
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

void RenderGraph::pushScope(const char* label, const char* profileScope)
{
    scopes_.push_back({label, profileScope, currentScope_});
    currentScope_ = static_cast<uint32_t>(scopes_.size() - 1);
}

void RenderGraph::popScope()
{
    if (currentScope_ == kInvalidRenderGraphHandle) {
        throw std::logic_error("RenderGraph::popScope called with no scope open.");
    }
    currentScope_ = scopes_[currentScope_].parent;
}

void RenderGraph::transitionScopes(uint32_t target)
{
    const VkCommandBuffer commandBuffer = frame_.commandBuffer;
    const RenderGraphScopeTransition transition = planScopeTransition(scopes_, openScopes_, target);
    for (size_t closed = 0; closed < transition.closeCount; ++closed) {
        const RenderGraphScope& scope = scopes_[openScopes_.back()];
        if (openScopeProfiled_.back() != 0 && profilerHooks_.endScope) {
            profilerHooks_.endScope(commandBuffer);
        }
        if (scope.label != nullptr) {
            rhi::debug::endLabel(commandBuffer);
        }
        openScopes_.pop_back();
        openScopeProfiled_.pop_back();
    }
    for (const uint32_t scopeIndex : transition.toOpen) {
        const RenderGraphScope& scope = scopes_[scopeIndex];
        if (scope.label != nullptr) {
            rhi::debug::beginLabel(commandBuffer, scope.label);
        }
        const bool profiled = scope.profileScope != nullptr && profilerHooks_.beginScope &&
                              profilerHooks_.beginScope(commandBuffer, scope.profileScope);
        openScopes_.push_back(scopeIndex);
        openScopeProfiled_.push_back(profiled ? 1 : 0);
    }
}

void RenderGraph::setProfilerHooks(RenderGraphProfilerHooks hooks)
{
    profilerHooks_ = std::move(hooks);
}

void RenderGraph::recordPasses()
{
    requireFrameActive("RenderGraph::recordPasses");

    const std::vector<uint32_t> plan = buildExecutionPlan(executionOrder_, passes_);

    // Cleared here rather than in beginFrame: the report that reads this runs at
    // the top of the next frame, after beginFrame has already run, so clearing
    // there would hand it an empty list every time.
    passRecordCosts_.clear();
    passRecordCosts_.reserve(plan.size());

    for (const uint32_t passIndex : plan) {
        transitionScopes(passes_[passIndex].scope);
        const auto start = std::chrono::steady_clock::now();
        executePass(passIndex);
        passRecordCosts_.push_back(
            {renderPassTypeName(passes_[passIndex].type),
             std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count(),
             passIndex});
    }
    transitionScopes(kInvalidRenderGraphHandle);
}

void RenderGraph::executePass(uint32_t passIndex)
{
    if (activePass_ != ActivePass::None) {
        throw std::logic_error("RenderGraph::executePass called while another pass is active.");
    }
    const VkCommandBuffer commandBuffer = frame_.commandBuffer;
    const RenderPassNode& pass = passes_[passIndex];

    // Label and profiler scope outside the barriers, in that order: where every
    // recorder put them, so a pass keeps its timestamp range and the command
    // stream does not change when the graph takes the pass over.
    if (!pass.executeLabel.empty()) {
        rhi::debug::beginLabel(commandBuffer, pass.executeLabel);
    }
    const bool profileScope = pass.executeProfileScope != nullptr && profilerHooks_.beginScope &&
                              profilerHooks_.beginScope(commandBuffer, pass.executeProfileScope);

    // buildExecutionPlan only emits scheduled passes, so a culled one cannot reach
    // here; beginDeclaredPass returning false would mean the plan and the culling
    // disagree, which is a graph bug rather than a renderer one.
    if (!beginDeclaredPass(passIndex)) {
        throw std::logic_error("RenderGraph::executePass reached a culled pass.");
    }
    if (pass.colorAttachment) {
        const RGColorAttachment& attachment = *pass.colorAttachment;
        beginColorRendering(
            textures_.at(attachment.texture.index), attachment.clearValue, VkExtent2D{0, 0}, attachment.loadOp);
    }

    activePass_ = ActivePass::Executor;
    executingPass_ = passIndex;
    executingRenderingScopes_ = 0;
    RenderGraphPassContext context(*this, commandBuffer, passIndex);
    pass.execute(context);
    if (activePass_ != ActivePass::Executor) {
        throw std::logic_error("RenderGraph::executePass: the callback left a pass open or closed one it did not own.");
    }
    activePass_ = ActivePass::None;
    executingPass_ = kInvalidRenderGraphHandle;

    if (pass.colorAttachment) {
        vkCmdEndRendering(commandBuffer);
    }
    recordPostPassTransitions(passIndex);
    if (profileScope && profilerHooks_.endScope) {
        profilerHooks_.endScope(commandBuffer);
    }
    if (!pass.executeLabel.empty()) {
        rhi::debug::endLabel(commandBuffer);
    }
}

void RenderGraph::recordPostPassTransitions(uint32_t passIndex)
{
    RenderPassNode& pass = passes_.at(passIndex);
    if (pass.postPassTransitions.empty()) {
        return;
    }

    // Its own batch, and its own submission: these are recorded after the pass
    // body rather than before it, so they cannot join the pass's set.
    barrierBatch_.reset();
    uint32_t imageBarriers = 0;
    for (const RGPostPassTransition& transition : pass.postPassTransitions) {
        imageBarriers += transitionTexture(transition.texture, transition.access, barrierBatch_);
    }
    flushBarrierBatch(barrierBatch_);
    pass.generatedBarrierCount += imageBarriers;
    pass.generatedImageBarrierCount += imageBarriers;
    pass.generatedBarrierSubmitCount += barrierBatch_.submitCount;
    pass.transitionSummary = passBarrierSummary(
        pass.generatedImageBarrierCount, pass.generatedBufferBarrierCount, pass.generatedBarrierSubmitCount);
}

std::string RenderGraph::frameTrace() const
{
    return formatRenderGraphFrameTrace(passes_, recordedOrder_);
}

void RenderGraph::requireFrameActive(const char* operation) const
{
    if (!frameActive_) {
        throw std::logic_error(std::string(operation) + " requires an active frame.");
    }
}

uint32_t RenderGraph::addWorkPass(std::string name,
                                  RenderPassType type,
                                  RenderPassExecutionType executionType,
                                  std::initializer_list<uint32_t> after,
                                  RenderGraphPassExecutor executor)
{
    // Side effect: what these passes write is not a graph resource, so nothing
    // downstream declares a read and liveness analysis would cull them away.
    const std::vector<uint32_t> predecessors(after);
    return addPass(
        std::move(name),
        type,
        executionType,
        true,
        [predecessors](RenderGraphBuilder& builder) {
            for (const uint32_t predecessor : predecessors) {
                builder.orderAfter(predecessor);
            }
        },
        std::move(executor));
}

void RenderGraph::compile()
{
    requireFrameActive("RenderGraph::compile");
    compilePassCulling();
    validateFrameDeclarations();
    refreshDebugResources();
}

bool RenderGraph::declaresPassOfType(RenderPassType type) const
{
    return std::any_of(passes_.begin(), passes_.end(), [type](const RenderPassNode& pass) {
        return pass.type == type && !pass.culled;
    });
}

void RenderGraph::compilePassCulling()
{
    cullUnusedPasses(passes_, textures_.size(), buffers_.size());
}

void RenderGraph::validateFrameDeclarations()
{
    textureValidationInfo_.clear();
    textureValidationInfo_.reserve(textures_.size());
    for (const TextureResource& texture : textures_) {
        textureValidationInfo_.push_back(RGResourceValidationInfo{texture.graphManaged, texture.desc.aliased});
    }

    bufferValidationInfo_.clear();
    bufferValidationInfo_.reserve(buffers_.size());
    for (const BufferResource& buffer : buffers_) {
        bufferValidationInfo_.push_back(RGResourceValidationInfo{buffer.graphManaged, false});
    }

    declarationIssues_ = validateDeclarations(passes_, textureValidationInfo_, bufferValidationInfo_);
    passSchedule_ = computePassSchedule(passes_);
    const RenderGraphExecutionOrder executionOrder = computeExecutionOrder(passSchedule_);
    executionOrder_ = executionOrder.order;
    executionOrderCycleDetected_ = executionOrder.cycleDetected;

    for (const RenderGraphDeclarationIssue& issue : declarationIssues_) {
        if (issue.passIndex >= passes_.size()) {
            continue;
        }

        const std::string resourceName = issue.resourceKind == RGResourceKind::Texture
                                             ? textures_.at(issue.resourceIndex).desc.name
                                             : buffers_.at(issue.resourceIndex).desc.name;
        std::string& text = passes_[issue.passIndex].declarationIssues;
        if (!text.empty()) {
            text += "; ";
        }
        text += resourceName;
        text += ": ";
        text += renderGraphDeclarationIssueName(issue.issue);
    }
}

std::vector<RenderGraphResourceLifetime>
computeTextureLifetimes(const std::vector<RenderPassNode>& passes, std::span<const uint32_t> order, size_t textureCount)
{
    std::vector<RenderGraphResourceLifetime> lifetimes(textureCount);
    // A resource something history-reads is consumed by the *next* frame, so the
    // interval this frame's usages describe is not the interval its bytes have to
    // survive. Written at slot 17 and history-read at slot 4, it looks live over
    // [4, 17] -- and anything the allocator packs into [0, 3] would then overwrite,
    // at the start of frame N+1, exactly what frame N left for it to read. The
    // read happens after the clobber, so nothing catches it: not the validation
    // layer, not a barrier, only the pixels.
    std::vector<uint8_t> historyRead(textureCount, 0);

    for (uint32_t slot = 0; slot < order.size(); ++slot) {
        if (order[slot] >= passes.size()) {
            continue;
        }
        const RenderPassNode& pass = passes[order[slot]];

        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (usage.resource.kind != RGResourceKind::Texture || usage.resource.index >= lifetimes.size()) {
                // Same tolerance cullUnusedPasses applies: a handle the graph
                // never imported is ignored rather than treated as live.
                continue;
            }

            // Every usage extends the interval, a layout-only read included: the
            // image has to be in a valid state where a descriptor binds it, so
            // its bytes cannot belong to something else at that point.
            if (usage.readKind == RGReadKind::History) {
                historyRead[usage.resource.index] = 1;
            }

            RenderGraphResourceLifetime& lifetime = lifetimes[usage.resource.index];
            if (!lifetime.used) {
                lifetime.used = true;
                lifetime.firstPass = slot;
                lifetime.lastPass = slot;
                continue;
            }
            lifetime.firstPass = std::min(lifetime.firstPass, slot);
            lifetime.lastPass = std::max(lifetime.lastPass, slot);
        }
    }

    // Widened to the whole frame rather than excluded outright: the allocator
    // already drops a resource whose interval covers everything from sharing with
    // anything, and saying "live all frame" keeps one concept instead of two. The
    // effect is that a history resource can still be placed in the pool -- it just
    // cannot have its bytes reused by anything.
    if (!order.empty()) {
        const uint32_t lastSlot = static_cast<uint32_t>(order.size()) - 1;
        for (size_t index = 0; index < lifetimes.size(); ++index) {
            if (historyRead[index] != 0 && lifetimes[index].used) {
                lifetimes[index].firstPass = 0;
                lifetimes[index].lastPass = lastSlot;
            }
        }
    }

    return lifetimes;
}

void cullUnusedPasses(std::vector<RenderPassNode>& passes, size_t textureCount, size_t bufferCount)
{
    std::vector<uint8_t> neededTextures(textureCount, 0);
    std::vector<uint8_t> neededBuffers(bufferCount, 0);

    // A resource something declares a history read on is consumed by the *next*
    // frame, so whatever writes it last this frame has to survive even though
    // nothing here reads it. Seeding the needed set is the whole mechanism: it is
    // a reader at the end of the frame, and the sweep below already knows how to
    // keep the last writer of a needed resource and drop the ones before it.
    //
    // The end-of-frame depth pyramid rebuild is exactly this shape. Nothing this
    // frame reads it -- the phase-2 re-test reads the mid-frame build -- and the
    // next frame's page marking and main cull history-read it. It was kept alive
    // by a hand-set side-effect flag, which is a fine outcome and a bad mechanism:
    // a pass added in that shape without the flag is culled silently.
    //
    // Seeded from every pass, including ones this sweep is about to cull. A
    // history reader that gets culled would be culled next frame too, so treating
    // its declaration as live is conservative in the safe direction.
    for (const RenderPassNode& pass : passes) {
        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (usage.readKind != RGReadKind::History) {
                continue;
            }
            if (usage.resource.kind == RGResourceKind::Texture && usage.resource.index < neededTextures.size()) {
                neededTextures[usage.resource.index] = 1;
            } else if (usage.resource.kind == RGResourceKind::Buffer && usage.resource.index < neededBuffers.size()) {
                neededBuffers[usage.resource.index] = 1;
            }
        }
    }

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
            } else if (usage.resource.kind == RGResourceKind::Buffer && usage.resource.index < neededBuffers.size()) {
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
            // Only a read of what this frame produced marks the resource needed.
            // A layout-only read consumes nothing, which is what lets culling
            // remove the bloom chain the composite does not sample; a history read
            // consumes the previous frame, and its claim on this frame's producer
            // was already made by the seeding above.
            if (accessReads(usage.access) && usage.readKind == RGReadKind::Produced) {
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

const char* renderGraphDeclarationIssueName(RGDeclarationIssue issue)
{
    switch (issue) {
    case RGDeclarationIssue::ReadsContentNoPassProduced:
        return "reads content no pass produced this frame";
    case RGDeclarationIssue::HistoryReadOfAliasedResource:
        return "history read of a pool-bound resource";
    case RGDeclarationIssue::ResourceDeclaredTwice:
        return "resource declared more than once";
    case RGDeclarationIssue::HistoryReadAfterProducer:
        return "history read placed after this frame's producer";
    case RGDeclarationIssue::StaleVersionRead:
        return "reads a version older than the one current here";
    }

    return "unknown declaration issue";
}

std::vector<RenderGraphDeclarationIssue> validateDeclarations(const std::vector<RenderPassNode>& passes,
                                                              std::span<const RGResourceValidationInfo> textures,
                                                              std::span<const RGResourceValidationInfo> buffers)
{
    std::vector<RenderGraphDeclarationIssue> issues;

    // Whether any surviving pass so far has written each resource. Forward sweep,
    // the opposite direction to cullUnusedPasses: liveness looks back from the
    // consumers, production looks forward from the producers.
    std::vector<uint8_t> textureWritten(textures.size(), 0);
    std::vector<uint8_t> bufferWritten(buffers.size(), 0);
    std::vector<uint32_t> textureVersion(textures.size(), 0);
    std::vector<uint32_t> bufferVersion(buffers.size(), 0);
    std::vector<uint32_t> seenTextures;
    std::vector<uint32_t> seenBuffers;

    for (uint32_t passIndex = 0; passIndex < passes.size(); ++passIndex) {
        const RenderPassNode& pass = passes[passIndex];
        if (pass.culled) {
            continue;
        }

        seenTextures.clear();
        seenBuffers.clear();

        const auto report = [&](RGDeclarationIssue issue, const RenderResourceUsage& usage) {
            issues.push_back(RenderGraphDeclarationIssue{passIndex, issue, usage.resource.kind, usage.resource.index});
        };

        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            const bool isTexture = usage.resource.kind == RGResourceKind::Texture;
            const size_t resourceCount = isTexture ? textures.size() : buffers.size();
            if (usage.resource.index >= resourceCount) {
                continue;
            }

            std::vector<uint32_t>& seen = isTexture ? seenTextures : seenBuffers;
            if (std::find(seen.begin(), seen.end(), usage.resource.index) != seen.end()) {
                report(RGDeclarationIssue::ResourceDeclaredTwice, usage);
            } else {
                seen.push_back(usage.resource.index);
            }

            const RGResourceValidationInfo& info =
                isTexture ? textures[usage.resource.index] : buffers[usage.resource.index];
            const std::vector<uint8_t>& written = isTexture ? textureWritten : bufferWritten;
            const std::vector<uint32_t>& current = isTexture ? textureVersion : bufferVersion;

            // A layout-only read consumes nothing, so none of the content rules
            // below have anything to say about it.
            if (accessReads(usage.access) && usage.readKind != RGReadKind::LayoutOnly) {
                const bool producedThisFrame = written[usage.resource.index] != 0;
                // The version current at this point is what the last write left;
                // a read naming an older one is holding a handle from before it.
                // History reads are exempt: naming version 0 is the whole point,
                // and HistoryReadAfterProducer already covers a stale one.
                if (usage.readKind == RGReadKind::Produced && usage.inputVersion < current[usage.resource.index]) {
                    report(RGDeclarationIssue::StaleVersionRead, usage);
                }
                if (usage.readKind == RGReadKind::History && producedThisFrame) {
                    // The declaration claims the previous frame's contents but an
                    // earlier pass already overwrote them. Applies whoever owns
                    // the memory: what is wrong here is the ordering.
                    report(RGDeclarationIssue::HistoryReadAfterProducer, usage);
                } else if (!producedThisFrame && info.graphManaged) {
                    // Only a graph-managed resource can be read before it is
                    // produced: an imported one keeps its contents by contract.
                    if (usage.readKind == RGReadKind::Produced) {
                        report(RGDeclarationIssue::ReadsContentNoPassProduced, usage);
                    } else if (info.aliased) {
                        // A declared history read of a private resource is
                        // exactly right; it is only wrong when the bytes are
                        // shared, because then there is no previous frame left.
                        report(RGDeclarationIssue::HistoryReadOfAliasedResource, usage);
                    }
                }
            }
        }

        // Writes land after the whole pass is checked, so a pass that reads and
        // writes the same resource is judged on what existed before it ran.
        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (!accessWrites(usage.access)) {
                continue;
            }
            if (usage.resource.kind == RGResourceKind::Texture && usage.resource.index < textureWritten.size()) {
                textureWritten[usage.resource.index] = 1;
                textureVersion[usage.resource.index] = usage.outputVersion;
            } else if (usage.resource.kind == RGResourceKind::Buffer && usage.resource.index < bufferWritten.size()) {
                bufferWritten[usage.resource.index] = 1;
                bufferVersion[usage.resource.index] = usage.outputVersion;
            }
        }
    }

    return issues;
}

std::vector<RenderGraphPassSchedule> computePassSchedule(const std::vector<RenderPassNode>& passes)
{
    std::vector<RenderGraphPassSchedule> schedule(passes.size());

    // Per resource: the last surviving pass to write it, and the passes that
    // have read it since. Both index spaces are separate, the same way
    // cullUnusedPasses keeps them separate.
    std::vector<uint32_t> textureLastWriter;
    std::vector<uint32_t> bufferLastWriter;
    std::vector<std::vector<uint32_t>> textureReaders;
    std::vector<std::vector<uint32_t>> bufferReaders;
    // producers[R][v - 1] is the pass that produced version v of resource R.
    std::vector<std::vector<uint32_t>> textureProducers;
    std::vector<std::vector<uint32_t>> bufferProducers;

    const auto grow = [](auto& lastWriter, auto& readers, auto& producers, uint32_t index) {
        if (index >= lastWriter.size()) {
            lastWriter.resize(index + 1, kInvalidRenderGraphHandle);
            readers.resize(index + 1);
            producers.resize(index + 1);
        }
    };

    uint32_t recordedSlot = 0;
    for (uint32_t passIndex = 0; passIndex < passes.size(); ++passIndex) {
        const RenderPassNode& pass = passes[passIndex];
        if (pass.culled) {
            continue;
        }

        RenderGraphPassSchedule& entry = schedule[passIndex];
        entry.scheduled = true;
        entry.recordedSlot = recordedSlot++;

        const auto addPredecessor = [&entry, passIndex](uint32_t predecessor) {
            if (predecessor == kInvalidRenderGraphHandle || predecessor == passIndex) {
                return;
            }
            if (std::find(entry.predecessors.begin(), entry.predecessors.end(), predecessor) ==
                entry.predecessors.end()) {
                entry.predecessors.push_back(predecessor);
            }
        };

        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            const bool isTexture = usage.resource.kind == RGResourceKind::Texture;
            auto& lastWriter = isTexture ? textureLastWriter : bufferLastWriter;
            auto& readers = isTexture ? textureReaders : bufferReaders;
            auto& producers = isTexture ? textureProducers : bufferProducers;
            grow(lastWriter, readers, producers, usage.resource.index);

            // Read after write, resolved through the version the handle names
            // rather than through whichever write happens to be most recent: a
            // reader holding an older handle depends on the pass that produced
            // *that* version, which is what its declaration actually says.
            //
            // A history read makes no such edge at all -- it consumes the
            // previous frame, not whatever this frame's producer will write.
            if (accessReads(usage.access) && usage.readKind == RGReadKind::Produced && usage.inputVersion > 0) {
                const std::vector<uint32_t>& versionProducers = producers[usage.resource.index];
                if (usage.inputVersion <= versionProducers.size()) {
                    addPredecessor(versionProducers[usage.inputVersion - 1]);
                }
            }

            if (accessWrites(usage.access)) {
                // Write after write, and write after read: overwriting means
                // waiting both for the previous writer and for everyone who read
                // what it left.
                addPredecessor(lastWriter[usage.resource.index]);
                for (const uint32_t reader : readers[usage.resource.index]) {
                    addPredecessor(reader);
                }
            }
        }

        // Declared ordering with no resource behind it. Only an earlier pass that
        // survived culling: a culled one does not run, so there is nothing to
        // wait for.
        for (const uint32_t predecessor : pass.orderAfter) {
            if (predecessor < passIndex && schedule[predecessor].scheduled) {
                addPredecessor(predecessor);
            }
        }

        // Committed after the whole pass is examined, so a read-modify-write is
        // judged against the state the pass found rather than the one it leaves.
        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (!accessWrites(usage.access)) {
                continue;
            }
            const bool isTexture = usage.resource.kind == RGResourceKind::Texture;
            auto& lastWriter = isTexture ? textureLastWriter : bufferLastWriter;
            auto& readers = isTexture ? textureReaders : bufferReaders;
            auto& producers = isTexture ? textureProducers : bufferProducers;
            grow(lastWriter, readers, producers, usage.resource.index);
            lastWriter[usage.resource.index] = passIndex;
            readers[usage.resource.index].clear();
            if (usage.outputVersion > 0) {
                producers[usage.resource.index].resize(usage.outputVersion, kInvalidRenderGraphHandle);
                producers[usage.resource.index][usage.outputVersion - 1] = passIndex;
            }
        }
        for (const RenderResourceUsage& usage : pass.resourceUsages) {
            if (!accessReads(usage.access)) {
                continue;
            }
            const bool isTexture = usage.resource.kind == RGResourceKind::Texture;
            auto& lastWriter = isTexture ? textureLastWriter : bufferLastWriter;
            auto& readers = isTexture ? textureReaders : bufferReaders;
            auto& producers = isTexture ? textureProducers : bufferProducers;
            grow(lastWriter, readers, producers, usage.resource.index);
            readers[usage.resource.index].push_back(passIndex);
        }

        std::sort(entry.predecessors.begin(), entry.predecessors.end());

        // Every edge points backwards, so one forward pass settles the longest
        // chain: a predecessor's own earliest slot is already final here.
        for (const uint32_t predecessor : entry.predecessors) {
            entry.earliestSlot = std::max(entry.earliestSlot, schedule[predecessor].earliestSlot + 1);
        }
        entry.slack = entry.recordedSlot - entry.earliestSlot;
    }

    return schedule;
}

std::vector<RenderGraphOrderViolation> validatePassOrder(const std::vector<RenderGraphPassSchedule>& schedule,
                                                         std::span<const uint32_t> order)
{
    std::vector<RenderGraphOrderViolation> violations;

    // Position of each pass in the proposed order, or absent.
    constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
    std::vector<size_t> position(schedule.size(), kAbsent);
    for (size_t slot = 0; slot < order.size(); ++slot) {
        if (order[slot] < position.size()) {
            position[order[slot]] = slot;
        }
    }

    for (uint32_t passIndex = 0; passIndex < schedule.size(); ++passIndex) {
        const RenderGraphPassSchedule& entry = schedule[passIndex];
        if (!entry.scheduled) {
            continue;
        }

        for (const uint32_t predecessor : entry.predecessors) {
            const size_t passSlot = position[passIndex];
            const size_t predecessorSlot = predecessor < position.size() ? position[predecessor] : kAbsent;
            // Absent counts as broken on either side: a pass the graph schedules
            // cannot be dropped from an order, and a predecessor that is not run
            // cannot be waited on.
            if (passSlot == kAbsent || predecessorSlot == kAbsent || predecessorSlot >= passSlot) {
                violations.push_back(RenderGraphOrderViolation{passIndex, predecessor});
            }
        }
    }

    return violations;
}

RenderGraphExecutionOrder computeExecutionOrder(const std::vector<RenderGraphPassSchedule>& schedule)
{
    RenderGraphExecutionOrder result;

    std::vector<uint32_t> remaining(schedule.size(), 0);
    // Successors, so emitting a pass can cheaply release what waited on it.
    std::vector<std::vector<uint32_t>> successors(schedule.size());
    uint32_t scheduledPasses = 0;

    for (uint32_t passIndex = 0; passIndex < schedule.size(); ++passIndex) {
        const RenderGraphPassSchedule& entry = schedule[passIndex];
        if (!entry.scheduled) {
            continue;
        }

        ++scheduledPasses;
        for (const uint32_t predecessor : entry.predecessors) {
            if (predecessor < schedule.size() && schedule[predecessor].scheduled) {
                ++remaining[passIndex];
                successors[predecessor].push_back(passIndex);
            }
        }
    }

    // Kahn's algorithm. The ready set is kept as a sorted vector rather than a
    // heap: the counts here are in the tens, and taking the lowest declaration
    // index is what makes the result reproduce the declaration order.
    std::vector<uint32_t> ready;
    for (uint32_t passIndex = 0; passIndex < schedule.size(); ++passIndex) {
        if (schedule[passIndex].scheduled && remaining[passIndex] == 0) {
            ready.push_back(passIndex);
        }
    }

    result.order.reserve(scheduledPasses);
    while (!ready.empty()) {
        const auto next = std::min_element(ready.begin(), ready.end());
        const uint32_t passIndex = *next;
        ready.erase(next);
        result.order.push_back(passIndex);

        for (const uint32_t successor : successors[passIndex]) {
            if (--remaining[successor] == 0) {
                ready.push_back(successor);
            }
        }
    }

    result.cycleDetected = result.order.size() != scheduledPasses;
    return result;
}

std::vector<uint32_t> unrecordedPasses(const std::vector<RenderGraphPassSchedule>& schedule,
                                       std::span<const uint32_t> order)
{
    std::vector<uint8_t> present(schedule.size(), 0);
    for (const uint32_t passIndex : order) {
        if (passIndex < present.size()) {
            present[passIndex] = 1;
        }
    }

    std::vector<uint32_t> missing;
    for (uint32_t passIndex = 0; passIndex < schedule.size(); ++passIndex) {
        if (schedule[passIndex].scheduled && present[passIndex] == 0) {
            missing.push_back(passIndex);
        }
    }

    return missing;
}

std::vector<std::string> RenderGraph::describeBackstopFindings() const
{
    // Names, not indices. A pass index is meaningless in a log: it depends on
    // which passes this configuration declared, which is exactly the thing the
    // reader does not know yet.
    const auto passName = [this](uint32_t index) -> std::string {
        return index < passes_.size() ? passes_[index].name : ("pass#" + std::to_string(index));
    };

    std::vector<std::string> findings;
    for (const RenderGraphOrderViolation& violation : recordedOrderViolations_) {
        findings.push_back("  order violation: " + passName(violation.passIndex) + " was recorded at or before " +
                           passName(violation.predecessorIndex) + ", which it depends on.");
    }
    for (const uint32_t passIndex : unrecordedPassIndices_) {
        findings.push_back("  unrecorded pass: " + passName(passIndex) +
                           " was declared and scheduled but never recorded.");
    }
    for (const RenderGraphDeclarationIssue& issue : declarationIssues_) {
        findings.push_back("  declaration issue: " + passName(issue.passIndex) + " -- " +
                           renderGraphDeclarationIssueName(issue.issue) + ".");
    }
    if (executionOrderCycleDetected_) {
        findings.emplace_back("  execution order: the derived dependency graph could not be fully ordered.");
    }

    return findings;
}

std::vector<uint32_t> buildExecutionPlan(std::span<const uint32_t> executionOrder,
                                         const std::vector<RenderPassNode>& passes)
{
    std::vector<uint32_t> plan;
    plan.reserve(executionOrder.size());
    for (const uint32_t passIndex : executionOrder) {
        if (passIndex < passes.size() && passes[passIndex].execute) {
            plan.push_back(passIndex);
        }
    }
    return plan;
}

RenderGraphScopeTransition
planScopeTransition(const std::vector<RenderGraphScope>& scopes, std::span<const uint32_t> open, uint32_t target)
{
    // The target's chain, outermost first. Bounded by the scope count so a
    // malformed parent link cannot loop.
    std::vector<uint32_t> chain;
    for (uint32_t scope = target; scope < scopes.size() && chain.size() <= scopes.size();
         scope = scopes[scope].parent) {
        chain.push_back(scope);
    }
    std::reverse(chain.begin(), chain.end());

    size_t shared = 0;
    while (shared < open.size() && shared < chain.size() && open[shared] == chain[shared]) {
        ++shared;
    }

    RenderGraphScopeTransition transition;
    transition.closeCount = open.size() - shared;
    transition.toOpen.assign(chain.begin() + static_cast<std::ptrdiff_t>(shared), chain.end());
    return transition;
}

std::string formatRenderGraphFrameTrace(const std::vector<RenderPassNode>& passes,
                                        std::span<const uint32_t> recordedOrder)
{
    std::string text = "Render graph frame trace:";
    for (const uint32_t passIndex : recordedOrder) {
        if (passIndex >= passes.size()) {
            continue;
        }
        const RenderPassNode& pass = passes[passIndex];
        text += ' ';
        text += pass.name;
        text += '[';
        text += std::to_string(pass.generatedImageBarrierCount);
        text += '/';
        text += std::to_string(pass.generatedBufferBarrierCount);
        text += '/';
        text += std::to_string(pass.generatedBarrierSubmitCount);
        text += ']';
    }
    return text;
}

std::string formatRenderGraphBackstopSummary(const RenderGraphBackstopSummary& summary)
{
    // The prefix and the three count phrases are the CI contract. Keep them
    // literal and in this order; tests/test_render_graph.cpp pins them, because
    // a reworded line makes the gate pass by matching nothing.
    std::string text = "Render graph backstop: ";
    text += std::to_string(summary.peakOrderViolations);
    text += " order violations, ";
    text += std::to_string(summary.peakUnrecordedPasses);
    text += " unrecorded passes, ";
    text += std::to_string(summary.peakDeclarationIssues);
    text += " declaration issues";

    // Counts alone cannot distinguish "clean over 30 frames" from "the graph
    // never ended a frame", and the second is how a no-op run reads green.
    text += " over ";
    text += std::to_string(summary.framesObserved);
    text += " frames";

    if (summary.framesWithIssues > 0) {
        text += " (";
        text += std::to_string(summary.framesWithIssues);
        text += " affected)";
    }
    if (summary.executionOrderCycleDetected) {
        text += "; execution order cycle detected";
    }

    return text;
}

uint32_t longestPassChain(const std::vector<RenderGraphPassSchedule>& schedule)
{
    uint32_t longest = 0;
    for (const RenderGraphPassSchedule& entry : schedule) {
        if (entry.scheduled) {
            longest = std::max(longest, entry.earliestSlot + 1);
        }
    }

    return longest;
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

    // First occurrence only: the cascaded shadow pass is begun once per cascade,
    // and the frame's recording order should say it ran once.
    if (std::find(recordedOrder_.begin(), recordedOrder_.end(), passIndex) == recordedOrder_.end()) {
        recordedOrder_.push_back(passIndex);
    }

    // Every barrier this pass infers sits between the same two points in the
    // command stream -- after whatever the previous pass recorded, before this
    // pass's first command -- so they are accumulated and submitted as one
    // dependency rather than one vkCmdPipelineBarrier2 per resource.
    barrierBatch_.reset();
    BarrierBatch& batch = barrierBatch_;
    uint32_t imageBarrierCount = 0;
    uint32_t bufferBarrierCount = 0;
    for (const RenderResourceUsage& usage : pass.resourceUsages) {
        if (usage.resource.kind == RGResourceKind::Texture) {
            imageBarrierCount += transitionTexture(RGTextureHandle{usage.resource.index}, usage.declaredAccess, batch);
        } else if (usage.resource.kind == RGResourceKind::Buffer) {
            bufferBarrierCount += transitionBuffer(RGBufferHandle{usage.resource.index}, usage.declaredAccess, batch);
        }
    }
    flushBarrierBatch(batch);

    pass.executed = true;
    pass.generatedBarrierCount += imageBarrierCount + bufferBarrierCount;
    pass.generatedImageBarrierCount += imageBarrierCount;
    pass.generatedBufferBarrierCount += bufferBarrierCount;
    pass.generatedBarrierSubmitCount += batch.submitCount;
    pass.transitionSummary = passBarrierSummary(
        pass.generatedImageBarrierCount, pass.generatedBufferBarrierCount, pass.generatedBarrierSubmitCount);
    return true;
}

void RenderGraph::flushBarrierBatch(BarrierBatch& batch)
{
    if (batch.imageBarriers.empty() && batch.bufferBarriers.empty()) {
        return;
    }

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(batch.imageBarriers.size());
    dependencyInfo.pImageMemoryBarriers = batch.imageBarriers.data();
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(batch.bufferBarriers.size());
    dependencyInfo.pBufferMemoryBarriers = batch.bufferBarriers.data();

    vkCmdPipelineBarrier2(frame_.commandBuffer, &dependencyInfo);

    batch.imageBarriers.clear();
    batch.bufferBarriers.clear();
    batch.touchedTextures.clear();
    batch.touchedBuffers.clear();
    ++batch.submitCount;
}

uint32_t RenderGraph::transitionTexture(RGTextureHandle handle, RGAccess access, BarrierBatch& batch)
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

    // The swapchain image's previous reader is the presentation engine, not a
    // command this graph recorded, and what orders the frame against it is the
    // image-available semaphore Renderer waits at COLOR_ATTACHMENT_OUTPUT. The
    // first transition of the frame therefore has to name that stage as its
    // source whatever the tracked layout says -- including on the frame where
    // the layout is still UNDEFINED, which is exactly when the mapping from
    // PRESENT_SRC does not apply and the transition would otherwise be ordered
    // against nothing at all.
    if (resource.owner == TextureOwner::SwapchainColor && !resource.usedThisFrame) {
        previous.stage |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
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

    // The tracked state is updated here, not at flush time, so a second
    // transition of this resource later in the pass computes its barrier from
    // the layout and access this one leaves behind -- exactly as it did when
    // every transition submitted immediately. Only the submission is deferred.
    if (barrierBatchNeedsFlush(batch.touchedTextures, handle.index)) {
        flushBarrierBatch(batch);
    }
    batch.imageBarriers.push_back(barrier);
    batch.touchedTextures.push_back(handle.index);

    setTextureLayout(resource, desired.layout);
    resource.lastAccess = desired;
    resource.usedThisFrame = true;
    return 1;
}

uint32_t RenderGraph::transitionBuffer(RGBufferHandle handle, RGAccess access, BarrierBatch& batch)
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

    const std::optional<BufferBarrierScopes> scopes = advanceBufferSync(resource.sync, desired);
    if (!scopes) {
        return 0;
    }

    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = scopes->srcStage;
    barrier.srcAccessMask = scopes->srcAccess;
    barrier.dstStageMask = scopes->dstStage;
    barrier.dstAccessMask = scopes->dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = resource.buffer;
    barrier.offset = 0;
    barrier.size = resource.desc.size;

    if (barrierBatchNeedsFlush(batch.touchedBuffers, handle.index)) {
        flushBarrierBatch(batch);
    }
    batch.bufferBarriers.push_back(barrier);
    batch.touchedBuffers.push_back(handle.index);
    return 1;
}

RenderGraph::TextureAccessState RenderGraph::accessStateForTexture(const TextureResource& resource,
                                                                   RGAccess access) const
{
    return textureAccessState(resource.desc.aspectMask, access, currentTextureLayout(resource));
}

TextureAccessState textureAccessState(VkImageAspectFlags aspectMask, RGAccess access, VkImageLayout currentLayout)
{
    TextureAccessState state{};
    state.declaredAccess = access;

    switch (access) {
    case RGAccess::ShaderRead:
        state.layout =
            isDepthAspect(aspectMask) ? depthReadOnlyLayout(aspectMask) : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
        state.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
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
    // ALL_TRANSFER rather than COPY: the declaration says "a transfer touches
    // this", not which transfer command does it. COPY covers vkCmdCopy* and
    // nothing else, so a barrier built from it establishes no dependency at all
    // against the blit the SSR half-res copy actually records -- both the
    // barrier and the blit are individually legal, and only synchronization
    // validation can see that they never meet. ALL_TRANSFER covers copy, blit,
    // resolve and clear alike, which is the conservative scope this graph's
    // barriers are documented to use.
    case RGAccess::TransferSrc:
        state.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        state.stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        state.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;
    case RGAccess::TransferDst:
        state.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        state.stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
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
    case RGAccess::StorageBufferClearAndReadWrite:
    case RGAccess::IndirectRead:
    case RGAccess::HostRead:
    case RGAccess::AccelerationStructureBuild:
    case RGAccess::AccelerationStructureRead:
    case RGAccess::AccelerationStructureComputeRead:
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

std::optional<BufferBarrierScopes> advanceBufferSync(BufferSyncState& state, const BufferAccessState& access)
{
    const bool firstUse = !state.touched;
    state.touched = true;

    if (accessMaskWrites(access.access)) {
        // The last write is in the first scope as well as the readers since it:
        // with no reader in between it is the whole of what this write follows.
        std::optional<BufferBarrierScopes> barrier;
        if (!firstUse) {
            barrier = BufferBarrierScopes{state.writeStage | state.readStages,
                                          state.writeAccess | state.readAccesses,
                                          access.stage,
                                          access.access};
        }
        state.writeStage = access.stage;
        state.writeAccess = access.access;
        state.readStages = VK_PIPELINE_STAGE_2_NONE;
        state.readAccesses = VK_ACCESS_2_NONE;
        state.visibleScopeCount = 0;
        return barrier;
    }

    state.readStages |= access.stage;
    state.readAccesses |= access.access;
    if (state.writeAccess == VK_ACCESS_2_NONE || bufferWriteVisibleTo(state, access.stage, access.access)) {
        return std::nullopt;
    }

    // Full only past eight distinct read scopes after one write, which no frame
    // comes near. A scope left out costs a redundant barrier later, never a
    // missing one.
    if (state.visibleScopeCount < BufferSyncState::kMaxVisibleScopes) {
        state.visibleScopes[state.visibleScopeCount++] = BufferAccessScope{access.stage, access.access};
    }
    return BufferBarrierScopes{state.writeStage, state.writeAccess, access.stage, access.access};
}

bool barrierBatchNeedsFlush(std::span<const uint32_t> batched, uint32_t resource)
{
    return std::find(batched.begin(), batched.end(), resource) != batched.end();
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
    const bool touched = usedThisFrame || previousDeclared != RGAccess::Unknown;
    return touched && (accessMaskWrites(previousAccess) || accessMaskWrites(desiredAccess));
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
    case RGAccess::StorageBufferClearAndReadWrite:
        state.stage = kShaderBufferStages | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        state.access =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;
    case RGAccess::IndirectRead:
        state.stage = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        state.access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        break;
    // The same scope as the texture mapping above, and for the same reason. The
    // two spellings used to disagree here -- the source side named COPY and the
    // destination side named TRANSFER -- which is the shape of the bug the
    // texture side actually had.
    case RGAccess::TransferSrc:
        state.stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        state.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;
    case RGAccess::TransferDst:
        state.stage = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        state.access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;
    case RGAccess::HostRead:
        state.stage = VK_PIPELINE_STAGE_2_HOST_BIT;
        state.access = VK_ACCESS_2_HOST_READ_BIT;
        break;
    // The build writes the structure and, as a source of the declared edge,
    // also stands for its reads of the instance data; the shader-read half of
    // that is the host-written instance buffer, which the queue submit already
    // makes visible.
    case RGAccess::AccelerationStructureBuild:
        state.stage = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        state.access =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        break;
    case RGAccess::AccelerationStructureRead:
        state.stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        state.access = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        break;
    // The compute stage alone. On a frame the mask traces it is the first reader
    // after the build, and it used to name the fragment stage as well, because
    // a read after a read got no barrier whatever its stage and the main pass's
    // rays had no other way to be ordered against the build. advanceBufferSync
    // gives them their own barrier now.
    case RGAccess::AccelerationStructureComputeRead:
        state.stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        state.access = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
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
    case TextureOwner::VsmPagePool:
        return frame_.vsmPagePool ? frame_.vsmPagePool->layout() : VK_IMAGE_LAYOUT_UNDEFINED;
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
    case TextureOwner::VsmPagePool:
        if (frame_.vsmPagePool) {
            frame_.vsmPagePool->setLayout(layout);
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

RGTextureHandle RenderGraph::addTextureUsage(RenderPassNode& pass,
                                             RGTextureHandle handle,
                                             RenderResourceAccess resourceAccess,
                                             RGAccess declaredAccess,
                                             std::string description,
                                             RGReadKind readKind)
{
    if (!handle.valid() || handle.index >= textures_.size()) {
        return handle;
    }

    TextureResource& resource = textures_[handle.index];
    const uint32_t inputVersion = handle.version;
    // A write produces the next version and the caller is handed a handle for
    // it; a plain read leaves the resource where it was.
    const uint32_t outputVersion = accessWrites(resourceAccess) ? ++resource.currentVersion : inputVersion;

    pass.resourceUsages.push_back(RenderResourceUsage{
        textureResourceHandle(handle),
        resourceAccess,
        declaredAccess,
        std::move(description),
        readKind,
        inputVersion,
        outputVersion,
    });

    return RGTextureHandle{handle.index, outputVersion};
}

RGBufferHandle RenderGraph::addBufferUsage(RenderPassNode& pass,
                                           RGBufferHandle handle,
                                           RenderResourceAccess resourceAccess,
                                           RGAccess declaredAccess,
                                           std::string description)
{
    if (!handle.valid() || handle.index >= buffers_.size()) {
        return handle;
    }

    BufferResource& resource = buffers_[handle.index];
    const uint32_t inputVersion = handle.version;
    const uint32_t outputVersion = accessWrites(resourceAccess) ? ++resource.currentVersion : inputVersion;

    pass.resourceUsages.push_back(RenderResourceUsage{
        bufferResourceHandle(handle),
        resourceAccess,
        declaredAccess,
        std::move(description),
        // No buffer is graph-managed today, so neither a history nor a
        // layout-only read of one has anything to say; the field exists to keep
        // the usage type uniform.
        RGReadKind::Produced,
        inputVersion,
        outputVersion,
    });

    return RGBufferHandle{handle.index, outputVersion};
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
            texture.desc.depth,
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
            1,
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
                                      VkExtent2D extentOverride,
                                      VkAttachmentLoadOp loadOp)
{
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = resource.imageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = loadOp;
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

} // namespace ve::renderer
