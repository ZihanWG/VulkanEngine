// Command recording: the main frame recording routine and the passes it drives,
// plus the render-graph frame resource description they are recorded against.
//
// Split out of Renderer.cpp, where recordRenderCommands alone was 784 lines.
// Definitions only -- these remain Renderer member functions, so no call site
// changed.
#include "renderer/Renderer.h"
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"

#include <imgui.h>

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


namespace ve {

void Renderer::recordPortfolioScreenshotCopy(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    if (!portfolioScreenshotRequested_) {
        return;
    }

    if (!ensurePortfolioShowcaseSceneReady()) {
        portfolioScreenshotRequested_ = false;
        return;
    }
    if (!portfolioCaptureMode_) {
        setPortfolioCaptureMode(true);
        screenshotCapture_.setStatus(
            "Screenshot deferred: portfolio capture mode was not active for the frame being copied.");
        Logger::warn(screenshotCapture_.status());
        return;
    }
    if (!currentFrameHasPortfolioShowcaseDrawItems()) {
        applyPortfolioCaptureSettings();
        screenshotCapture_.setStatus(
            "Screenshot deferred: portfolio showcase draw items were not active for the frame being copied.");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    portfolioScreenshotRequested_ = false;
    if (!swapchain_.supportsTransferSrc()) {
        screenshotCapture_.setStatus("Screenshot failed: swapchain transfer-source usage is unsupported.");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    const VkFormat format = swapchain_.colorFormat();
    if (extent.width == 0 || extent.height == 0 || !supportedScreenshotFormat(format)) {
        screenshotCapture_.setStatus(
            std::string("Screenshot failed: unsupported extent or format ") + vkFormatName(format) + ".");
        Logger::warn(screenshotCapture_.status());
        return;
    }

    // Policy gate passed; hand the validated frame to the capture subsystem, which
    // owns the readback buffers and records the copy + barriers.
    screenshotCapture_.recordCopy(commandBuffer, currentFrame_, swapchain_, imageIndex);
}

bool Renderer::isGpuPunctualShadowCullingActive() const
{
    return useGpuPunctualShadowCulling_ && punctualShadows_.cullAvailable() && punctualShadows_.valid() &&
           usePunctualShadows_ && punctualShadows_.slotCount() > 0 && !allDrawItems_.empty() &&
           !punctualShadowCacheHit_ && context_.device().drawIndirectFirstInstanceEnabled();
}

bool Renderer::isProbeCaptureActive() const
{
    return giSettings_.enabled && !giSettings_.debugPattern && irradianceProbes_.available() &&
           irradianceProbes_.convolveAvailable() && probeCapturePipeline_.pipeline() != VK_NULL_HANDLE &&
           giSettings_.probesPerFrame > 0 && !allDrawItems_.empty();
}

void Renderer::recordProbeCapturePass(VkCommandBuffer commandBuffer)
{
    const std::vector<uint32_t>& batch = irradianceProbes_.captureBatch();
    if (batch.empty()) {
        return;
    }

    const VkDescriptorSet globalDescriptorSet = globalMaterialDescriptorSet();
    if (globalDescriptorSet == VK_NULL_HANDLE) {
        return;
    }

    const bool profileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "ProbeCapture");
    rhi::debug::beginLabel(commandBuffer, "ProbeCapture");

    // One rendering scope over the whole capture atlas, with a viewport per
    // (probe, face) inside it. Same reasoning as the punctual shadow atlas: the
    // clear runs once for the image instead of once per tile, and the 96 tiles
    // do not each pay for a render pass.
    renderGraph_.beginProbeCapturePass();

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, probeCapturePipeline_.pipeline());

    const std::array<VkDescriptorSet, 2> descriptorSets{globalDescriptorSet,
                                                        bindlessTextureHeap_.descriptorSet()};
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            probeCapturePipeline_.layout(),
                            0,
                            static_cast<uint32_t>(descriptorSets.size()),
                            descriptorSets.data(),
                            0,
                            nullptr);

    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const renderer::ProbeGridBounds bounds = irradianceProbes_.bounds();
    // The same offset the convolution will subtract. Applied here as a
    // sub-texel slide of every face so successive captures of an unchanged
    // scene are not bit-identical -- without that, accumulating them averages
    // the same numbers forever and buys nothing.
    const glm::vec2 captureJitter = irradianceProbes_.captureJitter();

    // Every punctual light, flat. The cluster lists cannot be used here: they
    // index the camera's froxel grid, and a probe has no froxel. The slot
    // address is zero when nothing got a shadow tile, which every light's
    // negative slot sentinel already implies, so the shader needs no null test.
    const VkDeviceAddress probeLightBufferAddress = clusteredLighting_.lightBufferAddress(currentFrame_);
    const VkDeviceAddress probeShadowSlotAddress =
        punctualShadows_.slotCount() > 0 ? punctualShadows_.slotBufferAddress(currentFrame_) : 0;
    const uint32_t probeLightCount = probeLightBufferAddress != 0 ? clusteredLighting_.lightCount() : 0;

    // Multi-bounce. Held at zero until every probe has been captured once: until
    // then most of the grid still holds its neutral seed, and blending a surface
    // toward that would make the scene darker rather than brighter -- the
    // opposite of what another bounce is supposed to do. Same gate the
    // accumulation hysteresis uses, for the same reason.
    const float probeBounceWeight =
        irradianceProbes_.gridConverged() ? std::clamp(giSettings_.bounceWeight, 0.0f, 1.0f) : 0.0f;

    const renderer::Mesh* boundMesh = nullptr;
    uint32_t recordedDraws = 0;

    const auto captureCpuStart = std::chrono::high_resolution_clock::now();

    for (uint32_t slot = 0; slot < batch.size(); ++slot) {
        const glm::vec3 probePosition = renderer::probeWorldPosition(batch[slot], bounds);

        for (uint32_t face = 0; face < renderer::kProbeCaptureFaceCount; ++face) {
            const glm::uvec2 origin = renderer::probeCaptureTileOrigin(slot, face);

            VkViewport viewport{};
            viewport.x = static_cast<float>(origin.x);
            viewport.y = static_cast<float>(origin.y);
            viewport.width = static_cast<float>(renderer::kProbeCaptureFaceResolution);
            viewport.height = static_cast<float>(renderer::kProbeCaptureFaceResolution);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = {static_cast<int32_t>(origin.x), static_cast<int32_t>(origin.y)};
            scissor.extent = {renderer::kProbeCaptureFaceResolution, renderer::kProbeCaptureFaceResolution};
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            const glm::mat4 faceViewProjection =
                renderer::probeCaptureFaceViewProjection(probePosition, face, captureJitter);
            // Reuses the spot-shadow frustum extractor: it takes any perspective
            // view-projection, and a cube face is one. Without this every probe
            // would submit every draw item six times over.
            const renderer::Frustum faceFrustum = renderer::computeSpotShadowFrustum(faceViewProjection);

            for (const DrawItem& drawItem : allDrawItems_) {
                if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems || drawItem.indexCount == 0) {
                    continue;
                }
                // Alpha-blended geometry is skipped for the same reason it does
                // not cast shadows: it never wrote opaque depth, and treating it
                // as an opaque occluder here would block light it should let
                // through.
                if (drawItem.bucket == RenderBucket::Blend) {
                    continue;
                }
                if (drawItem.objectIndex < frameWorldBounds_.size() &&
                    !faceFrustum.testAabb(frameWorldBounds_[drawItem.objectIndex])) {
                    continue;
                }

                if (boundMesh != drawItem.mesh) {
                    const VkBuffer vertexBuffers[] = {drawItem.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, drawItem.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundMesh = drawItem.mesh;
                }

                ProbeCapturePushConstants pushConstants{};
                pushConstants.lightBufferAddress = probeLightBufferAddress;
                pushConstants.punctualShadowSlotAddress = probeShadowSlotAddress;
                pushConstants.lightCount = probeLightCount;
                pushConstants.bounceWeight = probeBounceWeight;
                // Offset per draw so the vertex stage reads objects[0] with an
                // instance index of zero, matching the direct-draw shadow path.
                pushConstants.objectFrameDataAddress =
                    objectFrameDataBaseAddress +
                    static_cast<VkDeviceAddress>(drawItem.frameDataIndex) * sizeof(ObjectFrameData);
                pushConstants.faceViewProjection = faceViewProjection;
                pushConstants.probePosition = glm::vec4{probePosition, 0.0f};
                vkCmdPushConstants(commandBuffer,
                                   probeCapturePipeline_.layout(),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   static_cast<uint32_t>(sizeof(ProbeCapturePushConstants)),
                                   &pushConstants);

                vkCmdDrawIndexed(commandBuffer, drawItem.indexCount, 1, drawItem.firstIndex, 0, 0);
                ++recordedDraws;
            }
        }
    }

    probeCaptureCpuMicroseconds_ = std::chrono::duration<float, std::micro>(
                                       std::chrono::high_resolution_clock::now() - captureCpuStart)
                                       .count();

    renderGraph_.endProbeCapturePass();
    probeCaptureDrawsRecorded_ = recordedDraws;

    rhi::debug::endLabel(commandBuffer);
    if (profileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
}

void Renderer::recordPunctualShadowPass(VkCommandBuffer commandBuffer, bool gpuCullActive)
{
    const uint32_t slotCount = punctualShadows_.slotCount();
    if (!punctualShadows_.valid() || punctualShadowPipeline_.pipeline() == VK_NULL_HANDLE || slotCount == 0 ||
        allDrawItems_.empty()) {
        punctualShadowDrawsRecorded_ = 0;
        return;
    }

    // Cache hit: the atlas already holds exactly what this frame would draw, so
    // the pass is skipped entirely -- no clear, no draws, no transition. The
    // image stays in the sampled layout the main pass left it in, which is the
    // layout the main pass wants next, so the graph emits nothing either.
    if (punctualShadowCacheHit_) {
        punctualShadowDrawsRecorded_ = 0;
        ++punctualShadowCachedFrames_;
        return;
    }

    const bool profileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "PunctualShadowAtlas");
    rhi::debug::beginLabel(commandBuffer, "PunctualShadowAtlas");

    // Layout transitions come from the graph: the pass declares a
    // depth-attachment write on the imported atlas texture and the main HDR pass
    // declares the matching sampled read, so the barriers are inferred in both
    // directions.
    //
    // The clear does not. A full-image clear would wipe the cached tiles this
    // whole mechanism exists to keep, so only the first frame after the image is
    // created clears everything; after that the attachment loads and the dirty
    // tiles are cleared individually below.
    const bool clearWholeAtlas = punctualShadowNeedsFullClear_;
    renderGraph_.beginPunctualShadowPass(clearWholeAtlas);

    if (!clearWholeAtlas) {
        // One vkCmdClearAttachments with a rect per dirty tile. Batched because
        // the clear is a real draw-time operation, not a load-op, and issuing it
        // per tile would serialise them for no reason.
        punctualShadowClearRects_.clear();
        punctualShadowClearRects_.reserve(punctualShadowDirtySlots_.size());
        for (const uint32_t slot : punctualShadowDirtySlots_) {
            const renderer::ShadowAtlasRect rect = punctualShadows_.slotRect(slot);
            if (rect.size == 0) {
                continue;
            }
            VkClearRect clearRect{};
            clearRect.rect.offset = {static_cast<int32_t>(rect.x), static_cast<int32_t>(rect.y)};
            clearRect.rect.extent = {rect.size, rect.size};
            clearRect.baseArrayLayer = 0;
            clearRect.layerCount = 1;
            punctualShadowClearRects_.push_back(clearRect);
        }

        if (!punctualShadowClearRects_.empty()) {
            VkClearAttachment clearAttachment{};
            clearAttachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            clearAttachment.clearValue.depthStencil = {1.0f, 0};
            vkCmdClearAttachments(commandBuffer,
                                  1,
                                  &clearAttachment,
                                  static_cast<uint32_t>(punctualShadowClearRects_.size()),
                                  punctualShadowClearRects_.data());
        }
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, punctualShadowPipeline_.pipeline());

    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const renderer::Mesh* boundMesh = nullptr;
    uint32_t recordedDraws = 0;

    // The CPU cost of culling and recording this pass. Kept because it is the
    // number that decides whether GPU caster culling is worth enabling: it is
    // what that path removes, and on this scene it is small enough that the
    // dispatch does not pay for itself.
    const auto punctualCpuStart = std::chrono::high_resolution_clock::now();

    // Only the dirty tiles. Everything else in the atlas is already exactly what
    // this frame would have drawn there.
    for (const uint32_t slot : punctualShadowDirtySlots_) {
        // Rects vary in size, so the pass reads each slot's own rect back rather
        // than deriving it from the index.
        const renderer::ShadowAtlasRect rect = punctualShadows_.slotRect(slot);
        if (rect.size == 0) {
            continue;
        }

        VkViewport viewport{};
        viewport.x = static_cast<float>(rect.x);
        viewport.y = static_cast<float>(rect.y);
        viewport.width = static_cast<float>(rect.size);
        viewport.height = static_cast<float>(rect.size);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {static_cast<int32_t>(rect.x), static_cast<int32_t>(rect.y)};
        scissor.extent = {rect.size, rect.size};
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // GPU-culled path: the compute dispatch above already compacted this
        // slot's surviving casters into its own region of the indirect buffer,
        // so the whole slot is one draw call regardless of caster count. There
        // is no indirect-count on this platform, so the full per-slot stride is
        // submitted and the zeroed commands the dispatch left behind are no-ops.
        if (gpuCullActive && slot < renderer::PunctualShadows::kMaxGpuCulledSlots) {
            const VkBuffer indirectBuffer = punctualShadows_.cullIndirectBuffer(currentFrame_);
            if (indirectBuffer != VK_NULL_HANDLE && !allDrawItems_.empty()) {
                const renderer::Mesh* indirectMesh = allDrawItems_.front().mesh;
                if (indirectMesh != nullptr) {
                    if (boundMesh != indirectMesh) {
                        const VkBuffer vertexBuffers[] = {indirectMesh->vertexBuffer()};
                        const VkDeviceSize vertexOffsets[] = {0};
                        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                        vkCmdBindIndexBuffer(commandBuffer, indirectMesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                        boundMesh = indirectMesh;
                    }

                    // Base address, not a per-draw offset: indirect draws carry
                    // the object-data index in firstInstance, which the vertex
                    // stage reads back out of gl_InstanceIndex.
                    PunctualShadowPushConstants pushConstants{};
                    pushConstants.objectFrameDataAddress = objectFrameDataBaseAddress;
                    pushConstants.lightViewProjection = punctualShadows_.slotViewProjection(slot);
                    vkCmdPushConstants(commandBuffer,
                                       punctualShadowPipeline_.layout(),
                                       VK_SHADER_STAGE_VERTEX_BIT,
                                       0,
                                       static_cast<uint32_t>(sizeof(PunctualShadowPushConstants)),
                                       &pushConstants);

                    const uint32_t stride = punctualShadows_.cullSlotCommandStride();
                    const uint32_t drawCount = std::min(stride, static_cast<uint32_t>(allDrawItems_.size()));
                    vkCmdDrawIndexedIndirect(
                        commandBuffer,
                        indirectBuffer,
                        static_cast<VkDeviceSize>(slot) * stride * sizeof(VkDrawIndexedIndirectCommand),
                        drawCount,
                        sizeof(VkDrawIndexedIndirectCommand));
                    recordedDraws += drawCount;
                }
            }
            continue;
        }

        const renderer::Frustum& slotFrustum = punctualShadows_.slotFrustum(slot);

        for (const DrawItem& drawItem : allDrawItems_) {
            if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems || drawItem.indexCount == 0) {
                continue;
            }
            // Blended geometry does not write depth in the main pass, so letting
            // it cast an opaque shadow here would be wrong.
            if (drawItem.bucket == RenderBucket::Blend) {
                continue;
            }
            // CPU cull against the slot's own frustum. With one tile per light
            // this is a handful of sphere tests per slot; moving punctual slots
            // onto the existing GPU shadow-cull path is a follow-up.
            if (drawItem.objectIndex < frameWorldBounds_.size() &&
                !slotFrustum.testAabb(frameWorldBounds_[drawItem.objectIndex])) {
                continue;
            }

            PunctualShadowPushConstants pushConstants{};
            // Offsetting the base address per draw lets the vertex stage read
            // objects[0] with an instance index of zero, matching the CSM
            // direct-draw fallback.
            pushConstants.objectFrameDataAddress =
                objectFrameDataBaseAddress +
                static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData));
            pushConstants.lightViewProjection = punctualShadows_.slotViewProjection(slot);
            vkCmdPushConstants(commandBuffer,
                               punctualShadowPipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PunctualShadowPushConstants)),
                               &pushConstants);

            if (boundMesh != drawItem.mesh) {
                const VkBuffer vertexBuffers[] = {drawItem.mesh->vertexBuffer()};
                const VkDeviceSize vertexOffsets[] = {0};
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, drawItem.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                boundMesh = drawItem.mesh;
            }

            vkCmdDrawIndexed(
                commandBuffer, drawItem.indexCount, 1, drawItem.firstIndex, drawItem.vertexOffset, 0);
            ++recordedDraws;
        }
    }

    renderGraph_.endPunctualShadowPass();

    const auto punctualCpuEnd = std::chrono::high_resolution_clock::now();
    punctualShadowCpuMicros_ =
        std::chrono::duration_cast<std::chrono::microseconds>(punctualCpuEnd - punctualCpuStart).count();

    punctualShadowDrawsRecorded_ = recordedDraws;
    punctualShadowSlotsRedrawn_ = static_cast<uint32_t>(punctualShadowDirtySlots_.size());

    // Record what the atlas now holds, keyed by tile. Only the tiles just drawn
    // are updated; the rest keep the entries that made them cacheable.
    for (const uint32_t slot : punctualShadowDirtySlots_) {
        const renderer::ShadowAtlasRect rect = punctualShadows_.slotRect(slot);
        if (rect.size == 0) {
            continue;
        }
        punctualShadowResidentTiles_[renderer::packShadowAtlasRect(rect)] = punctualShadowSlotKeys_[slot];
    }
    // A full clear wiped every tile, so any entry for a tile not redrawn above
    // now describes contents that no longer exist.
    if (clearWholeAtlas) {
        for (auto it = punctualShadowResidentTiles_.begin(); it != punctualShadowResidentTiles_.end();) {
            const bool redrawn = std::any_of(
                punctualShadowDirtySlots_.begin(), punctualShadowDirtySlots_.end(), [&](uint32_t slot) {
                    return renderer::packShadowAtlasRect(punctualShadows_.slotRect(slot)) == it->first;
                });
            it = redrawn ? std::next(it) : punctualShadowResidentTiles_.erase(it);
        }
    }
    punctualShadowNeedsFullClear_ = false;

    rhi::debug::beginLabel(commandBuffer,
                           "PunctualShadowSlots " + std::to_string(slotCount) + " draws " +
                               std::to_string(recordedDraws));
    rhi::debug::endLabel(commandBuffer);
    rhi::debug::endLabel(commandBuffer);
    if (profileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
}

renderer::RenderGraphFrameResources Renderer::renderGraphFrameResources()
{
    const VkExtent3D sceneExtent = postProcess_.sceneColor().extent();
    const VkExtent3D depthPyramidExtent = depthPyramid_.extent();
    VkClearValue sceneClear{};
    sceneClear.color.float32[0] = 0.03f;
    sceneClear.color.float32[1] = 0.04f;
    sceneClear.color.float32[2] = 0.07f;
    sceneClear.color.float32[3] = 1.0f;

    VkClearValue bloomClear{};
    bloomClear.color.float32[0] = 0.0f;
    bloomClear.color.float32[1] = 0.0f;
    bloomClear.color.float32[2] = 0.0f;
    bloomClear.color.float32[3] = 1.0f;

    const auto bufferResource = [](const char* name,
                                   const std::vector<rhi::VulkanBuffer>& buffers,
                                   uint32_t frameIndex,
                                   VkBufferUsageFlags usage) {
        if (frameIndex >= buffers.size() || !buffers[frameIndex].valid()) {
            return renderer::RenderGraphBufferResource{};
        }

        return renderer::RenderGraphBufferResource{
            name,
            buffers[frameIndex].buffer(),
            buffers[frameIndex].size(),
            usage,
            true,
        };
    };

    const auto bloomResource = [&bloomClear](const char* name, const rhi::VulkanImage& image, VkImageLayout& layout) {
        const VkExtent3D extent = image.extent();
        return renderer::RenderGraphImageResource{
            name,
            image.image(),
            image.imageView(),
            VkExtent2D{extent.width, extent.height},
            &layout,
            image.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        };
    };

    const auto taaHistoryResource = [&bloomClear](
                                        const char* name, const rhi::VulkanImage& image, VkImageLayout& layout) {
        const VkExtent3D extent = image.extent();
        return renderer::RenderGraphImageResource{
            name,
            image.image(),
            image.imageView(),
            VkExtent2D{extent.width, extent.height},
            &layout,
            image.format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            true,
        };
    };

    std::vector<renderer::RenderGraphImageResource> bloomDownsampleResources;
    bloomDownsampleResources.reserve(postProcess_.bloomMipDownsampleImages().size());
    for (size_t level = 0; level < postProcess_.bloomMipDownsampleImages().size() &&
                           level < postProcess_.bloomMipDownsampleLayouts().size();
         ++level) {
        const std::string name = "BloomMipDownsample" + std::to_string(level);
        bloomDownsampleResources.push_back(bloomResource(name.c_str(),
                                                         postProcess_.bloomMipDownsampleImages()[level],
                                                         postProcess_.bloomMipDownsampleLayouts()[level]));
    }

    std::vector<renderer::RenderGraphImageResource> bloomUpsampleResources;
    bloomUpsampleResources.reserve(postProcess_.bloomMipUpsampleImages().size());
    for (size_t level = 0;
         level < postProcess_.bloomMipUpsampleImages().size() && level < postProcess_.bloomMipUpsampleLayouts().size();
         ++level) {
        const std::string name = "BloomMipUpsample" + std::to_string(level);
        bloomUpsampleResources.push_back(bloomResource(
            name.c_str(), postProcess_.bloomMipUpsampleImages()[level], postProcess_.bloomMipUpsampleLayouts()[level]));
    }

    renderer::RenderGraphImageResource gtaoRawResource{};
    if (gtao_.available()) {
        // Raw AO is half resolution; import its actual extent, not the scene's.
        const VkExtent3D gtaoRawExtent = gtao_.rawAmbientOcclusion().extent();
        gtaoRawResource = renderer::RenderGraphImageResource{
            "GtaoRawAmbientOcclusion",
            gtao_.rawAmbientOcclusion().image(),
            gtao_.rawAmbientOcclusion().imageView(),
            VkExtent2D{gtaoRawExtent.width, gtaoRawExtent.height},
            gtao_.rawAmbientOcclusionLayoutPtr(),
            gtao_.rawAmbientOcclusion().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            false,
            false,
        };
    }

    renderer::RenderGraphImageResource ssrSceneColorCopyResource{};
    if (ssr_.available()) {
        ssrSceneColorCopyResource = renderer::RenderGraphImageResource{
            "SsrSceneColorCopy",
            ssr_.sceneColorCopy().image(),
            ssr_.sceneColorCopy().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            ssr_.sceneColorCopyLayoutPtr(),
            ssr_.sceneColorCopy().format(),
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            false,
            false,
        };
    }

    // Probe atlases. Sized by the probe grid rather than the window, so unlike
    // every other target here their extent comes from the image itself. A failed
    // probe subsystem leaves null handles, which importTexture turns into an
    // invalid handle and every declaration then skips.
    // What a probe records in directions where nothing was drawn. The same
    // ambient term the main pass adds to every surface, standing in for the sky:
    // a probe that saw black there would be far too dark outdoors, and it would
    // look like moody indirect light rather than like a missing term.
    const glm::vec4 ambientSky = portfolioCaptureMode_ ? kPortfolioAmbientLightColor : kAmbientLightColor;
    VkClearValue probeCaptureSkyClear{};
    probeCaptureSkyClear.color.float32[0] = ambientSky.r;
    probeCaptureSkyClear.color.float32[1] = ambientSky.g;
    probeCaptureSkyClear.color.float32[2] = ambientSky.b;
    probeCaptureSkyClear.color.float32[3] = renderer::kProbeMaxDistance;

    const auto probeCaptureResource = [](const char* name,
                                         const rhi::VulkanImage& image,
                                         VkImageLayout* layout,
                                         VkImageUsageFlags usage,
                                         VkImageAspectFlags aspect,
                                         VkClearValue clearValue = VkClearValue{}) {
        const VkExtent3D extent = image.extent();
        return renderer::RenderGraphImageResource{
            name,
            image.image(),
            image.imageView(),
            VkExtent2D{extent.width, extent.height},
            layout,
            image.format(),
            usage,
            1,
            1,
            aspect,
            clearValue,
            true,
            true,
        };
    };

    const auto probeAtlasResource = [](const char* name, const rhi::VulkanImage& image, VkImageLayout* layout) {
        const VkExtent3D extent = image.extent();
        return renderer::RenderGraphImageResource{
            name,
            image.image(),
            image.imageView(),
            VkExtent2D{extent.width, extent.height},
            layout,
            image.format(),
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            false,
            true,
        };
    };

    return renderer::RenderGraphFrameResources{
        renderer::RenderGraphImageResource{
            "SceneColor",
            postProcess_.sceneColor().image(),
            postProcess_.sceneColor().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.sceneColorLayout(),
            postProcess_.sceneColor().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            sceneClear,
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "VelocityBuffer",
            postProcess_.velocity().image(),
            postProcess_.velocity().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.velocityLayout(),
            postProcess_.velocity().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "NormalRoughnessGBuffer",
            postProcess_.normalRoughness().image(),
            postProcess_.normalRoughness().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.normalRoughnessLayout(),
            postProcess_.normalRoughness().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "AmbientOcclusion",
            postProcess_.ambientOcclusion().image(),
            postProcess_.ambientOcclusion().imageView(),
            VkExtent2D{sceneExtent.width, sceneExtent.height},
            &postProcess_.ambientOcclusionLayout(),
            postProcess_.ambientOcclusion().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VkClearValue{},
            true,
            false,
        },
        gtaoRawResource,
        ssrSceneColorCopyResource,
        taaHistoryResource("TAAHistoryRead",
                           postProcess_.taaHistoryImages()[postProcess_.taaHistoryReadIndex()],
                           postProcess_.taaHistoryLayouts()[postProcess_.taaHistoryReadIndex()]),
        taaHistoryResource("TAAHistoryWrite",
                           postProcess_.taaHistoryImages()[postProcess_.taaHistoryWriteIndex()],
                           postProcess_.taaHistoryLayouts()[postProcess_.taaHistoryWriteIndex()]),
        renderer::RenderGraphImageResource{
            "BloomExtract",
            postProcess_.bloomExtract().image(),
            postProcess_.bloomExtract().imageView(),
            postProcess_.bloomExtent(),
            &postProcess_.bloomExtractLayout(),
            postProcess_.bloomExtract().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "BloomPing",
            postProcess_.bloomPing().image(),
            postProcess_.bloomPing().imageView(),
            postProcess_.bloomExtent(),
            &postProcess_.bloomPingLayout(),
            postProcess_.bloomPing().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        },
        renderer::RenderGraphImageResource{
            "BloomPong",
            postProcess_.bloomPong().image(),
            postProcess_.bloomPong().imageView(),
            postProcess_.bloomExtent(),
            &postProcess_.bloomPongLayout(),
            postProcess_.bloomPong().format(),
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            1,
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            bloomClear,
            true,
            false,
        },
        std::move(bloomDownsampleResources),
        std::move(bloomUpsampleResources),
        renderer::RenderGraphImageResource{
            "DepthPyramidHiZ",
            depthPyramid_.image(),
            depthPyramid_.imageView(),
            VkExtent2D{depthPyramidExtent.width, depthPyramidExtent.height},
            depthPyramid_.layoutPtr(),
            depthPyramid_.format(),
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            depthPyramid_.mipLevels(),
            1,
            VK_IMAGE_ASPECT_COLOR_BIT,
            {},
            false,
            true,
        },
        probeAtlasResource("ProbeIrradianceAtlas",
                           irradianceProbes_.irradianceAtlas(),
                           irradianceProbes_.irradianceAtlasLayoutPtr()),
        probeAtlasResource(
            "ProbeDepthAtlas", irradianceProbes_.depthAtlas(), irradianceProbes_.depthAtlasLayoutPtr()),
        probeCaptureResource("ProbeCaptureAtlas",
                             irradianceProbes_.captureAtlas(),
                             irradianceProbes_.captureAtlasLayoutPtr(),
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                 VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             probeCaptureSkyClear),
        probeCaptureResource("ProbeCaptureDepth",
                             irradianceProbes_.captureDepth(),
                             irradianceProbes_.captureDepthLayoutPtr(),
                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                             VK_IMAGE_ASPECT_DEPTH_BIT),
        bufferResource(
            "MainCullInput", gpuCulling_.cullInputBuffers(), currentFrame_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        bufferResource("MainCullIndirectOutput",
                       frameIndirectDrawBuffers_,
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT),
        bufferResource("MainCullVisibleCounts",
                       gpuCulling_.visibleCountBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        bufferResource("MainCullReadback",
                       gpuCulling_.visibleCountReadbackBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource(
            "LuminancePartials", postProcess_.luminanceBuffers(), currentFrame_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        bufferResource("LuminanceReadback",
                       postProcess_.luminanceReadbackBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("LuminanceHistogram",
                       postProcess_.histogramBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource("HistogramReadback",
                       postProcess_.histogramReadbackBuffers(),
                       currentFrame_,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        bufferResource(
            "ExposureState", postProcess_.exposureBuffers(), currentFrame_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        postProcess_.isTaaActive(),
        frameTwoPhaseOcclusionActive_,
        frameSsrActive_,
        frameGtaoActive_,
        frameVisibleBucketRanges_[static_cast<size_t>(RenderBucket::Blend)].count(),
        // Zero on a cache hit, which stops the graph declaring the atlas write
        // pass at all. The main pass still declares its read, so the image keeps
        // the layout it already has and no barrier is emitted for it.
        punctualShadowCacheHit_ ? 0u : punctualShadows_.slotCount(),
        isVolumetricFogActive(),
        isIrradianceProbeUpdateActive(),
        frameProbeCaptureActive_,
    };
}

void Renderer::recordGpuCullingCommands(VkCommandBuffer commandBuffer)
{
    gpuCulling_.recordMainCull(commandBuffer,
                               currentFrame_,
                               isGpuCullingActive(),
                               static_cast<uint32_t>(allDrawItems_.size()),
                               frameFrustumPlanes_,
                               isMainPassMultiDrawIndirectActive(),
                               /*copyReadback=*/!frameTwoPhaseOcclusionActive_);
}

void Renderer::recordGpuShadowCullingCommands(VkCommandBuffer commandBuffer, uint32_t cascadeIndex)
{
    gpuCulling_.recordShadowCull(commandBuffer,
                                 currentFrame_,
                                 isGpuShadowCullingActive(),
                                 cascadeIndex,
                                 activeCascadeCount(),
                                 static_cast<uint32_t>(allDrawItems_.size()),
                                 frameShadowCascadeFrustumPlanes_[cascadeIndex]);
}

void Renderer::ensureDepthPyramidShaderReadLayout(VkCommandBuffer commandBuffer)
{
    depthPyramid_.ensureShaderReadLayout(commandBuffer);
}

void Renderer::recordDepthPyramidCommands(VkCommandBuffer commandBuffer, bool midFrame)
{
    depthPyramid_.recordCommands(commandBuffer, currentFrame_, frameViewProjection_, frameCameraPosition_, midFrame);
}

void Renderer::recordRenderCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    const VkDeviceAddress objectFrameDataBaseAddress = frameObjectDataBuffers_.at(currentFrame_).deviceAddress();
    const size_t mainDrawItemCount = visibleDrawItems_.size();
    const bool clusteredLightingActive = clusteredLighting_.available() && useClusteredLighting_ &&
                                         clusteredLighting_.lightCount() > 0 && !allDrawItems_.empty();
    const bool taaActiveThisFrame = postProcess_.isTaaActive();
    postProcess_.beginFrame(currentFrame_, taaActiveThisFrame);

    renderGraph_.beginFrame(commandBuffer,
                            swapchain_,
                            shadowMap_,
                            punctualShadows_.valid() ? &punctualShadows_.atlas() : nullptr,
                            imageIndex,
                            renderGraphFrameResources());
    rhi::debug::beginLabel(commandBuffer, "Frame");
    gpuProfiler_.beginFrame(currentFrame_, commandBuffer);

    const bool gpuShadowCullingActive = isGpuShadowCullingActive() && !allDrawItems_.empty();
    const uint32_t cascadeCount = activeCascadeCount();

    const bool csmProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "CSMShadowPass");
    rhi::debug::beginLabel(commandBuffer, "CSMShadowPass");

    const VkExtent2D shadowExtent = shadowMap_.extent();
    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = static_cast<float>(shadowExtent.width);
    shadowViewport.height = static_cast<float>(shadowExtent.height);
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = shadowExtent;

    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        if (gpuShadowCullingActive) {
            recordGpuShadowCullingCommands(commandBuffer, cascadeIndex);
        }

        const std::vector<DrawItem>& activeShadowDrawItems =
            gpuShadowCullingActive ? allDrawItems_ : shadowCascadeDrawItems_[cascadeIndex];
        const std::vector<MeshDrawBatch>& activeShadowMeshDrawBatches =
            gpuShadowCullingActive ? gpuShadowMeshDrawBatches_ : shadowCascadeMeshDrawBatches_[cascadeIndex];
        const size_t shadowDrawItemCount = activeShadowDrawItems.size();
        const bool shadowIndirectCountPathActive =
            gpuShadowCullingActive && isShadowIndirectCountPathActive(currentFrame_);

        rhi::debug::beginLabel(commandBuffer, "ShadowCascade" + std::to_string(cascadeIndex));
        renderGraph_.beginShadowPass(cascadeIndex);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_.pipeline());
        vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

        const std::string shadowCullingLabel = gpuShadowCullingActive
                                                   ? "ShadowCasterCulling GPU cascade " + std::to_string(cascadeIndex) +
                                                         " max " + std::to_string(shadowDrawItemCount)
                                                   : "ShadowCasterCulling cascade " + std::to_string(cascadeIndex) +
                                                         " visible " + std::to_string(shadowDrawItemCount) + "/" +
                                                         std::to_string(allDrawItems_.size());
        rhi::debug::beginLabel(commandBuffer, shadowCullingLabel);
        rhi::debug::endLabel(commandBuffer);

        const std::string shadowDrawLabel =
            "ShadowCascade" + std::to_string(cascadeIndex) + " DrawItems " + std::to_string(shadowDrawItemCount) +
            (gpuShadowCullingActive
                 ? (shadowIndirectCountPathActive ? " GPU culling indirect-count" : " GPU culling indirect fallback")
                 : " CPU culling direct fallback");
        rhi::debug::beginLabel(commandBuffer, shadowDrawLabel);

        // Cutout casters swap to the alpha-tested shadow pipeline. Its layout adds
        // the bindless set, so the layouts are not push-constant compatible and
        // every switch has to re-push. Returns true when a switch happened.
        const bool maskedShadowPipelineReady = maskedShadowPipeline_.pipeline() != VK_NULL_HANDLE;
        VkPipelineLayout boundShadowLayout = shadowPipeline_.layout();
        RenderBucket boundShadowBucket = RenderBucket::Opaque;
        auto bindShadowPipelineForBucket = [&](RenderBucket bucket) {
            // Without the bindless heap there is nothing to sample, so MASK falls
            // back to the depth-only pipeline and casts a solid silhouette.
            const bool wantMasked = maskedShadowPipelineReady && bucket == RenderBucket::Mask;
            const RenderBucket effectiveBucket = wantMasked ? RenderBucket::Mask : RenderBucket::Opaque;
            if (boundShadowBucket == effectiveBucket) {
                return false;
            }

            const rhi::VulkanPipeline& pipeline = wantMasked ? maskedShadowPipeline_ : shadowPipeline_;
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline());
            if (wantMasked) {
                const VkDescriptorSet bindlessSet = bindlessTextureHeap_.descriptorSet();
                vkCmdBindDescriptorSets(commandBuffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipeline.layout(),
                                        0,
                                        1,
                                        &bindlessSet,
                                        0,
                                        nullptr);
            }
            boundShadowLayout = pipeline.layout();
            boundShadowBucket = effectiveBucket;
            return true;
        };

        const renderer::Mesh* boundShadowMesh = nullptr;
        if (gpuShadowCullingActive && !activeShadowDrawItems.empty()) {
            const PushConstants pushConstants{objectFrameDataBaseAddress, cascadeIndex, 0};
            vkCmdPushConstants(commandBuffer,
                               boundShadowLayout,
                               VK_SHADER_STAGE_VERTEX_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &pushConstants);

            const VkBuffer shadowIndirectDrawBuffer = frameShadowIndirectDrawBuffers_.at(currentFrame_).buffer();
            const VkBuffer shadowBatchVisibleCountBuffer =
                shadowIndirectCountPathActive ? gpuCulling_.shadowVisibleCountBuffer(currentFrame_).buffer()
                                              : VK_NULL_HANDLE;
            const std::string shadowBatchesLabel = "ShadowIndirectDrawBatches " +
                                                   std::to_string(activeShadowMeshDrawBatches.size()) +
                                                   " max commands " + std::to_string(shadowDrawItemCount);
            rhi::debug::beginLabel(commandBuffer, shadowBatchesLabel);
            for (const MeshDrawBatch& batch : activeShadowMeshDrawBatches) {
                if (!batch.mesh || batch.drawItemCount == 0) {
                    continue;
                }
                // Blended geometry only depth-tests in the main pass, so treating
                // it as a solid occluder here would have clear glass throw an
                // opaque silhouette. Skipped at replay rather than filtered out of
                // the caster list, exactly like the main pass does: the GPU cull
                // still emits commands for these slots and they are simply never
                // drawn, which keeps every batch's command range valid.
                if (batch.bucket == RenderBucket::Blend) {
                    continue;
                }

                if (bindShadowPipelineForBucket(batch.bucket)) {
                    vkCmdPushConstants(commandBuffer,
                                       boundShadowLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT,
                                       0,
                                       static_cast<uint32_t>(sizeof(PushConstants)),
                                       &pushConstants);
                }

                if (boundShadowMesh != batch.mesh) {
                    const VkBuffer vertexBuffers[] = {batch.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, batch.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundShadowMesh = batch.mesh;
                }

                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(batch.compactedCommandOffset * sizeof(VkDrawIndexedIndirectCommand));
                if (shadowIndirectCountPathActive && shadowBatchVisibleCountBuffer != VK_NULL_HANDLE) {
                    vkCmdDrawIndexedIndirectCount(commandBuffer,
                                                  shadowIndirectDrawBuffer,
                                                  indirectOffset,
                                                  shadowBatchVisibleCountBuffer,
                                                  batch.visibleCountOffset,
                                                  batch.drawItemCount,
                                                  sizeof(VkDrawIndexedIndirectCommand));
                } else {
                    vkCmdDrawIndexedIndirect(commandBuffer,
                                             shadowIndirectDrawBuffer,
                                             indirectOffset,
                                             batch.drawItemCount,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }
            }
            rhi::debug::endLabel(commandBuffer);
        } else {
            for (size_t drawIndex = 0; drawIndex < shadowDrawItemCount; ++drawIndex) {
                const DrawItem& drawItem = activeShadowDrawItems[drawIndex];
                if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems) {
                    continue;
                }
                // Same rule as the indirect path above: blended geometry is not
                // an occluder.
                if (drawItem.bucket == RenderBucket::Blend) {
                    continue;
                }

                bindShadowPipelineForBucket(drawItem.bucket);

                const PushConstants pushConstants{
                    objectFrameDataBaseAddress +
                        static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData)),
                    cascadeIndex,
                    0};

                vkCmdPushConstants(commandBuffer,
                                   boundShadowLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT,
                                   0,
                                   static_cast<uint32_t>(sizeof(PushConstants)),
                                   &pushConstants);

                const renderer::Mesh* mesh = drawItem.mesh;
                if (boundShadowMesh != mesh) {
                    const VkBuffer vertexBuffers[] = {mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundShadowMesh = mesh;
                }

                if (drawItem.indexCount > 0) {
                    vkCmdDrawIndexed(
                        commandBuffer, drawItem.indexCount, 1, drawItem.firstIndex, drawItem.vertexOffset, 0);
                }
            }
        }
        rhi::debug::endLabel(commandBuffer);

        renderGraph_.endShadowPass(cascadeIndex + 1 == cascadeCount);
        rhi::debug::endLabel(commandBuffer);
    }

    rhi::debug::endLabel(commandBuffer);
    if (csmProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    // GPU caster culling for the atlas, when enabled. Recorded here rather than
    // inside recordPunctualShadowPass because compute cannot run inside a
    // dynamic-rendering scope, and that pass is one scope so its cached tiles
    // survive a partial clear. Every slot is therefore culled up front.
    const bool gpuPunctualCullActive = isGpuPunctualShadowCullingActive();
    if (gpuPunctualCullActive) {
        const renderer::GpuProfileScope cullScope(
            gpuProfiler_, currentFrame_, commandBuffer, "PunctualShadowGpuCull");
        rhi::debug::beginLabel(commandBuffer, "PunctualShadowGpuCull");
        punctualShadows_.uploadSlotFrustums(currentFrame_);

        // The shared cull input carries no bucket, so the "does this cast" test
        // travels beside it. Blended geometry only depth-tests in the main pass;
        // letting it cast would put an opaque silhouette back into the atlas.
        punctualShadowCasterFlags_.assign(allDrawItems_.size(), 1u);
        for (size_t drawIndex = 0; drawIndex < allDrawItems_.size(); ++drawIndex) {
            const DrawItem& drawItem = allDrawItems_[drawIndex];
            if (drawItem.bucket == RenderBucket::Blend || drawItem.mesh == nullptr ||
                drawItem.indexCount == 0) {
                punctualShadowCasterFlags_[drawIndex] = 0u;
            }
        }

        rhi::VulkanBuffer& cullInput = gpuCulling_.shadowCullInputBuffer(currentFrame_);
        punctualShadows_.recordCull(
            commandBuffer,
            currentFrame_,
            cullInput.buffer(),
            cullInput.size(),
            static_cast<uint32_t>(allDrawItems_.size()),
            std::span<const uint32_t>(punctualShadowCasterFlags_.data(), punctualShadowCasterFlags_.size()));
        rhi::debug::endLabel(commandBuffer);
    }

    // Punctual casters go into the atlas right after the directional cascades,
    // so both shadow sources are resident before the main HDR pass samples them.
    recordPunctualShadowPass(commandBuffer, gpuPunctualCullActive);

    // Fog injection samples the cascaded shadow map, so it has to follow the
    // shadow passes; the main HDR pass samples the integrated volume, so it has
    // to precede that.
    // Runs whether or not fog is on: binding 8 claims a sampled layout
    // unconditionally, so the volume has to reach it even on a cold start with
    // fog disabled. No-op after the first frame.
    // Fog switching off has to leave a neutral volume behind, because the
    // skybox samples it unconditionally. Detected here rather than in the UI so
    // any path that disables fog -- settings load, preset, toggle -- is covered.
    const bool fogActiveThisFrame = isVolumetricFogActive();
    if (fogWasActive_ && !fogActiveThisFrame) {
        volumetricFog_.markVolumeNeedsClear();
    }
    fogWasActive_ = fogActiveThisFrame;

    volumetricFog_.ensureVolumeInitialized(commandBuffer);

    recordGpuCullingCommands(commandBuffer);

    // Clustered (Forward+) light assignment: rebuild the froxel AABBs, then cull
    // every light into its froxels. Both write buffers the main HDR fragment
    // shader reads, so the assignment pass barriers into the fragment stage.
    if (clusteredLightingActive && !frameAsyncComputeActive_) {
        {
            const renderer::GpuProfileScope buildScope(gpuProfiler_, currentFrame_, commandBuffer, "ClusterBuild");
            rhi::debug::beginLabel(commandBuffer, "ClusterBuild");
            clusteredLighting_.recordClusterBuild(commandBuffer, currentFrame_);
            rhi::debug::endLabel(commandBuffer);
        }
        {
            const renderer::GpuProfileScope cullScope(gpuProfiler_, currentFrame_, commandBuffer, "LightCull");
            rhi::debug::beginLabel(commandBuffer, "LightCull");
            clusteredLighting_.recordLightCull(commandBuffer, currentFrame_);
            rhi::debug::endLabel(commandBuffer);
        }
    }

    // Fog runs here, not with the shadow passes: injection walks the per-cluster
    // light lists, so it has to follow the cluster build and light cull above.
    // It still has to precede the main HDR pass, which samples the volume.
    if (isVolumetricFogActive()) {
        const bool fogProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "VolumetricFog");
        // The graph pass carries no rendering scope; it exists so the cascaded
        // shadow map is transitioned out of its depth-attachment layout before
        // the injection dispatch samples it.
        // Fog reuses the light lists the cluster passes just produced and the
        // atlas tiles the punctual shadow pass just filled, rather than
        // building either for itself.
        renderer::FogInjectPushConstants fogPushConstants{};
        fogPushConstants.lightBufferAddress = clusteredLighting_.lightBufferAddress(currentFrame_);
        fogPushConstants.clusterGridAddress =
            clusteredLightingActive ? clusteredLighting_.clusterGridAddress(currentFrame_) : 0;
        fogPushConstants.lightIndexListAddress =
            clusteredLightingActive ? clusteredLighting_.lightIndexListAddress(currentFrame_) : 0;
        fogPushConstants.punctualShadowSlotAddress =
            punctualShadows_.slotCount() > 0 ? punctualShadows_.slotBufferAddress(currentFrame_) : 0;
        fogPushConstants.lightCount = clusteredLighting_.lightCount();
        fogPushConstants.useClustered = clusteredLightingActive ? 1u : 0u;
        fogPushConstants.clusterZNear = camera_.nearPlane;
        fogPushConstants.clusterZFar = camera_.farPlane;

        renderGraph_.beginVolumetricFogPass();
        volumetricFog_.recordCommands(commandBuffer, currentFrame_, fogPushConstants);
        renderGraph_.endVolumetricFogPass();
        if (fogProfileScope) {
            gpuProfiler_.endScope(currentFrame_, commandBuffer);
        }
    }

    // Probe shading parameters, refreshed inside this frame's own command buffer
    // so a single buffer serves every frame in flight.
    //
    // Before the capture pass, not just before the main pass: the capture reads
    // these too, for the multi-bounce lookup. Updating after it would have the
    // capture read the previous frame's grid placement -- and on the very first
    // frame, a buffer nothing had written yet.
    {
        renderer::ProbeShadingParams probeParams{};
        const renderer::ProbeGridBounds bounds = irradianceProbes_.bounds();
        // Intensity carries the off state, so everything downstream needs no
        // separate flag: no atlases, no capture pipeline, or the toggle off all
        // collapse to zero here.
        const bool probeShadingActive = giSettings_.enabled && irradianceProbes_.hasAtlases() &&
                                        irradianceProbes_.convolveAvailable() && !giSettings_.debugPattern;
        probeParams.gridOrigin =
            glm::vec4{bounds.origin, probeShadingActive ? giSettings_.intensity : 0.0f};
        probeParams.gridSpacing = glm::vec4{bounds.spacing, giSettings_.surfaceBias};
        probeParams.debug.x =
            (probeShadingActive && giSettings_.debugIrradianceOnly) ? 1.0f : 0.0f;
        irradianceProbes_.updateShadingParams(commandBuffer, probeParams);
    }

    // Probe capture, then the convolution that turns it into probe tiles. Both
    // sit after the shadow passes -- the capture samples the cascades so the
    // radiance it records is shadowed -- and before the main pass, which
    // declares a read on the probe atlases.
    if (frameProbeCaptureActive_) {
        recordProbeCapturePass(commandBuffer);
    } else {
        probeCaptureDrawsRecorded_ = 0;
    }

    if (isIrradianceProbeUpdateActive()) {
        const bool probeProfileScope =
            gpuProfiler_.beginScope(currentFrame_, commandBuffer, "IrradianceProbeUpdate");
        renderGraph_.beginIrradianceProbePass();
        irradianceProbes_.recordUpdate(commandBuffer, giSettings_.debugPattern);
        renderGraph_.endIrradianceProbePass();
        if (probeProfileScope) {
            gpuProfiler_.endScope(currentFrame_, commandBuffer);
        }
    }


    const bool mainHdrProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "MainHDRPass");
    rhi::debug::beginLabel(commandBuffer, "MainHDRPass");
    renderGraph_.beginMainHdrPass();

    const VkExtent2D extent = swapchain_.extent();
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    // Viewport and scissor depend on the current swapchain extent, so they stay
    // dynamic instead of forcing a new pipeline for every resize.
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    if (skyboxDescriptorSet_ != VK_NULL_HANDLE) {
        const renderer::GpuProfileScope skyboxScope(gpuProfiler_, currentFrame_, commandBuffer, "Skybox");
        rhi::debug::beginLabel(commandBuffer, "Skybox");
        glm::mat4 skyboxView = camera_.viewMatrix();
        skyboxView[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const glm::mat4 projection = frameJitteredProjection_;
        // The fragment shader projects the sky direction with w = 0, which drops
        // the translation column, so the full previous view-projection doubles as
        // the rotation-only sky reprojection matrix.
        const SkyboxPushConstants skyboxPushConstants{glm::inverse(projection * skyboxView),
                                                      previousFrameViewProjection_};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline_.pipeline());
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skyboxPipeline_.layout(),
                                0,
                                1,
                                &skyboxDescriptorSet_,
                                0,
                                nullptr);
        vkCmdPushConstants(commandBuffer,
                           skyboxPipeline_.layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(SkyboxPushConstants)),
                           &skyboxPushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        rhi::debug::endLabel(commandBuffer);
    }

    const std::string objectDrawLabel = cullingStats_.gpuCulling
                                            ? "MainHDRPass IndirectDrawItems " + std::to_string(mainDrawItemCount) +
                                                  " GPU culling batches " + std::to_string(meshDrawBatches_.size())
                                            : "MainHDRPass IndirectDrawItems " + std::to_string(mainDrawItemCount) +
                                                  " visible objects " + std::to_string(cullingStats_.visibleObjects) +
                                                  "/" + std::to_string(cullingStats_.totalObjects) + " batches " +
                                                  std::to_string(meshDrawBatches_.size());
    const bool renderObjectsProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "RenderObjects");
    rhi::debug::beginLabel(commandBuffer, objectDrawLabel);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());

    const VkDescriptorSet globalDescriptorSet = globalMaterialDescriptorSet();
    const bool bindlessMaterialTexturesActive = isBindlessMaterialTextureActive();
    const bool multiDrawIndirectActive = isMainPassMultiDrawIndirectActive();
    bool bindlessDescriptorSetsBound = false;
    if (bindlessMaterialTexturesActive) {
        if (globalDescriptorSet != VK_NULL_HANDLE) {
            const std::array<VkDescriptorSet, 2> descriptorSets{
                globalDescriptorSet,
                bindlessTextureHeap_.descriptorSet(),
            };
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_.layout(),
                                    0,
                                    static_cast<uint32_t>(descriptorSets.size()),
                                    descriptorSets.data(),
                                    0,
                                    nullptr);
            bindlessDescriptorSetsBound = true;
        }
    }

    const renderer::Mesh* boundMesh = nullptr;
    const VkBuffer indirectDrawBuffer = frameIndirectDrawBuffers_.at(currentFrame_).buffer();
    const bool indirectCountPathActive = isFrameIndirectCountPathActive(currentFrame_);
    const VkBuffer batchVisibleCountBuffer =
        indirectCountPathActive ? gpuCulling_.visibleCountBuffer(currentFrame_).buffer() : VK_NULL_HANDLE;
    const uint32_t toneMappingOperator = toneMappingOperatorValue(toneMappingSettings_.operatorType);
    const float exposure = postProcess_.currentToneMappingExposure();
    // Shared lighting push constant for the main pass. The clustered fields are
    // zeroed when the per-froxel path is inactive, which makes the fragment
    // shader fall back to brute-force light evaluation.
    PushConstants basePushConstants{};
    basePushConstants.objectFrameDataAddress = objectFrameDataBaseAddress;
    basePushConstants.cascadeIndex = 0;
    basePushConstants.toneMappingOperator = toneMappingOperator;
    basePushConstants.exposure = exposure;
    basePushConstants.lightCount = clusteredLighting_.lightCount();
    basePushConstants.lightBufferAddress = clusteredLighting_.lightBufferAddress(currentFrame_);
    basePushConstants.clusterGridAddress =
        clusteredLightingActive ? clusteredLighting_.clusterGridAddress(currentFrame_) : 0;
    basePushConstants.lightIndexListAddress =
        clusteredLightingActive ? clusteredLighting_.lightIndexListAddress(currentFrame_) : 0;
    basePushConstants.clusterZNear = camera_.nearPlane;
    basePushConstants.clusterZFar = camera_.farPlane;
    basePushConstants.screenWidth = static_cast<float>(extent.width);
    basePushConstants.screenHeight = static_cast<float>(extent.height);
    basePushConstants.useClustered = clusteredLightingActive ? 1u : 0u;
    basePushConstants.debugClusterHeatmap = showClusterHeatmap_ ? 1u : 0u;
    basePushConstants.debugLodHeatmap = lodSettings_.debugHeatmap ? 1u : 0u;
    // Zero when nothing got a tile this frame, so the shader's null-address test
    // is enough to skip the atlas fetch -- it never has to consult a count.
    basePushConstants.punctualShadowSlotAddress =
        punctualShadows_.slotCount() > 0 ? punctualShadows_.slotBufferAddress(currentFrame_) : 0;
    basePushConstants.debugPunctualShadows = showPunctualShadowDebug_ ? 1u : 0u;
    // Zero disables the fog fetch in the shader, so the off state needs no
    // separate flag.
    basePushConstants.fogMaxDistance = isVolumetricFogActive() ? fogSettings_.maxDistance : 0.0f;
    if (multiDrawIndirectActive) {
        if (bindlessDescriptorSetsBound) {
            const PushConstants pushConstants = basePushConstants;
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &pushConstants);

            const std::string batchesLabel =
                std::string(indirectCountPathActive ? "IndirectCountBatches " : "MultiDrawIndirectBatches ") +
                std::to_string(meshDrawBatches_.size()) + " max commands " + std::to_string(mainDrawItemCount);
            rhi::debug::beginLabel(commandBuffer, batchesLabel);
            for (const MeshDrawBatch& batch : meshDrawBatches_) {
                if (!batch.mesh || batch.drawItemCount == 0) {
                    continue;
                }
                // Blended geometry is drawn later, by recordTransparentPass. The
                // GPU cull still emits commands for it into the compacted buffer;
                // those slots are simply never replayed here.
                if (batch.bucket == RenderBucket::Blend) {
                    continue;
                }

                if (boundMesh != batch.mesh) {
                    const VkBuffer vertexBuffers[] = {batch.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, batch.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    boundMesh = batch.mesh;
                }

                const std::string batchLabel =
                    std::string(indirectCountPathActive ? "MeshBatchIndirectCount max commands "
                                                        : "MeshBatchDraw commands ") +
                    std::to_string(batch.drawItemCount);
                rhi::debug::beginLabel(commandBuffer, batchLabel);
                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(batch.compactedCommandOffset * sizeof(VkDrawIndexedIndirectCommand));
                if (indirectCountPathActive && batchVisibleCountBuffer != VK_NULL_HANDLE) {
                    vkCmdDrawIndexedIndirectCount(commandBuffer,
                                                  indirectDrawBuffer,
                                                  indirectOffset,
                                                  batchVisibleCountBuffer,
                                                  batch.visibleCountOffset,
                                                  batch.drawItemCount,
                                                  sizeof(VkDrawIndexedIndirectCommand));
                } else {
                    vkCmdDrawIndexedIndirect(commandBuffer,
                                             indirectDrawBuffer,
                                             indirectOffset,
                                             batch.drawItemCount,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }
                rhi::debug::endLabel(commandBuffer);
            }
            rhi::debug::endLabel(commandBuffer);
        }
    } else {
        for (size_t drawIndex = 0; drawIndex < mainDrawItemCount; ++drawIndex) {
            const DrawItem& drawItem = visibleDrawItems_[drawIndex];
            if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems) {
                continue;
            }
            if (drawItem.bucket == RenderBucket::Blend) {
                continue;
            }

            if (bindlessMaterialTexturesActive) {
                if (!bindlessDescriptorSetsBound) {
                    continue;
                }
            } else {
                if (!drawItem.material || drawItem.material->descriptorSet == VK_NULL_HANDLE) {
                    continue;
                }

                const VkDescriptorSet descriptorSet = drawItem.material->descriptorSet;
                vkCmdBindDescriptorSets(commandBuffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipeline_.layout(),
                                        0,
                                        1,
                                        &descriptorSet,
                                        0,
                                        nullptr);
            }

            PushConstants pushConstants = basePushConstants;
            pushConstants.objectFrameDataAddress =
                objectFrameDataBaseAddress +
                static_cast<VkDeviceAddress>(drawItem.frameDataIndex * sizeof(ObjectFrameData));

            // Fallback recording pushes the address of this draw's object data; firstInstance stays zero.
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &pushConstants);

            const renderer::Mesh* mesh = drawItem.mesh;
            if (boundMesh != mesh) {
                const VkBuffer vertexBuffers[] = {mesh->vertexBuffer()};
                const VkDeviceSize vertexOffsets[] = {0};
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                boundMesh = mesh;
            }

            if (drawItem.indexCount > 0) {
                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(drawIndex * sizeof(VkDrawIndexedIndirectCommand));
                vkCmdDrawIndexedIndirect(
                    commandBuffer, indirectDrawBuffer, indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
            }
        }
    }
    rhi::debug::endLabel(commandBuffer);
    if (renderObjectsProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    // Skinned demo mesh: GPU vertex skinning via the joint-matrix palette. Shares
    // the main descriptor sets + push constant layout, adds the joint-palette
    // address, and binds a second vertex stream for joint indices/weights.
    if (showSkinnedMesh_ && skinnedMesh_.valid() && skinnedPipeline_.pipeline() != VK_NULL_HANDLE &&
        bindlessMaterialTexturesActive && globalDescriptorSet != VK_NULL_HANDLE) {
        const renderer::GpuProfileScope skinnedScope(gpuProfiler_, currentFrame_, commandBuffer, "SkinnedMesh");
        rhi::debug::beginLabel(commandBuffer, "SkinnedMesh");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skinnedPipeline_.pipeline());

        const std::array<VkDescriptorSet, 2> skinnedSets{globalDescriptorSet, bindlessTextureHeap_.descriptorSet()};
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                skinnedPipeline_.layout(),
                                0,
                                static_cast<uint32_t>(skinnedSets.size()),
                                skinnedSets.data(),
                                0,
                                nullptr);

        PushConstants skinnedPush = basePushConstants;
        skinnedPush.objectFrameDataAddress = objectFrameDataBaseAddress +
                                             static_cast<VkDeviceAddress>(kSkinnedObjectFrameSlot) *
                                                 sizeof(ObjectFrameData);
        skinnedPush.jointMatricesAddress = skinnedMesh_.jointPaletteAddress(currentFrame_);
        vkCmdPushConstants(commandBuffer,
                           skinnedPipeline_.layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(PushConstants)),
                           &skinnedPush);

        const std::array<VkBuffer, 2> skinnedVertexBuffers{skinnedMesh_.geometryBuffer(), skinnedMesh_.skinningBuffer()};
        const std::array<VkDeviceSize, 2> skinnedOffsets{0, 0};
        vkCmdBindVertexBuffers(commandBuffer,
                               0,
                               static_cast<uint32_t>(skinnedVertexBuffers.size()),
                               skinnedVertexBuffers.data(),
                               skinnedOffsets.data());
        vkCmdBindIndexBuffer(commandBuffer, skinnedMesh_.indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, skinnedMesh_.indexCount(), 1, 0, 0, 0);
        rhi::debug::endLabel(commandBuffer);
    }

    renderGraph_.endMainHdrPass();
    rhi::debug::endLabel(commandBuffer);
    if (mainHdrProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }

    // Two-phase Hi-Z occlusion: rebuild the pyramid from phase-1 depth, re-test
    // the phase-1 occlusion candidates against it, then draw the rescued
    // (disoccluded) objects into the existing attachments with LOAD ops. The
    // final end-of-frame pyramid rebuild below then includes the rescued draws.
    if (frameTwoPhaseOcclusionActive_) {
        recordDepthPyramidCommands(commandBuffer, /*midFrame=*/true);
        gpuCulling_.recordMainCullPhase2(commandBuffer,
                                         currentFrame_,
                                         isGpuCullingActive(),
                                         static_cast<uint32_t>(allDrawItems_.size()),
                                         frameFrustumPlanes_,
                                         isMainPassMultiDrawIndirectActive());

        const bool phase2ProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "MainHDRPhase2");
        rhi::debug::beginLabel(commandBuffer, "MainHDRPhase2");
        renderGraph_.beginMainHdrPhase2Pass();

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.pipeline());
        if (bindlessDescriptorSetsBound) {
            const std::array<VkDescriptorSet, 2> phase2Sets{
                globalDescriptorSet,
                bindlessTextureHeap_.descriptorSet(),
            };
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline_.layout(),
                                    0,
                                    static_cast<uint32_t>(phase2Sets.size()),
                                    phase2Sets.data(),
                                    0,
                                    nullptr);
            vkCmdPushConstants(commandBuffer,
                               pipeline_.layout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0,
                               static_cast<uint32_t>(sizeof(PushConstants)),
                               &basePushConstants);

            const renderer::Mesh* phase2BoundMesh = nullptr;
            for (const MeshDrawBatch& batch : meshDrawBatches_) {
                if (!batch.mesh || batch.drawItemCount == 0) {
                    continue;
                }

                if (phase2BoundMesh != batch.mesh) {
                    const VkBuffer vertexBuffers[] = {batch.mesh->vertexBuffer()};
                    const VkDeviceSize vertexOffsets[] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, batch.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                    phase2BoundMesh = batch.mesh;
                }

                const VkDeviceSize indirectOffset =
                    static_cast<VkDeviceSize>(batch.compactedCommandOffset * sizeof(VkDrawIndexedIndirectCommand));
                if (indirectCountPathActive && batchVisibleCountBuffer != VK_NULL_HANDLE) {
                    vkCmdDrawIndexedIndirectCount(commandBuffer,
                                                  indirectDrawBuffer,
                                                  indirectOffset,
                                                  batchVisibleCountBuffer,
                                                  batch.visibleCountOffset,
                                                  batch.drawItemCount,
                                                  sizeof(VkDrawIndexedIndirectCommand));
                } else {
                    vkCmdDrawIndexedIndirect(commandBuffer,
                                             indirectDrawBuffer,
                                             indirectOffset,
                                             batch.drawItemCount,
                                             sizeof(VkDrawIndexedIndirectCommand));
                }
            }
        }

        renderGraph_.endMainHdrPhase2Pass();
        rhi::debug::endLabel(commandBuffer);
        if (phase2ProfileScope) {
            gpuProfiler_.endScope(currentFrame_, commandBuffer);
        }
    }

    if (frameSsrActive_) {
        ssr_.recordCommands(commandBuffer, currentFrame_, postProcess_.sceneColor().image(), extent);
    }

    if (frameGtaoActive_) {
        gtao_.recordCommands(commandBuffer, currentFrame_, extent);
    }

    // Transparent pass. After SSR and GTAO, which both need opaque-only depth and
    // G-buffer data, and before the TAA resolve so blended edges are still
    // antialiased. Recorded inline because it reuses this scope's push constants,
    // descriptor sets, and viewport rather than rebuilding them.
    const RenderBucketRange blendRange = frameVisibleBucketRanges_[static_cast<size_t>(RenderBucket::Blend)];
    if (!blendRange.empty() && transparentPipeline_.pipeline() != VK_NULL_HANDLE &&
        globalDescriptorSet != VK_NULL_HANDLE && bindlessMaterialTexturesActive) {
        const renderer::GpuProfileScope transparentScope(gpuProfiler_, currentFrame_, commandBuffer, "Transparent");
        rhi::debug::beginLabel(commandBuffer, "Transparent " + std::to_string(blendRange.count()) + " draws");
        renderGraph_.beginTransparentPass();

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipeline_.pipeline());
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        const std::array<VkDescriptorSet, 2> transparentSets{globalDescriptorSet,
                                                             bindlessTextureHeap_.descriptorSet()};
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                transparentPipeline_.layout(),
                                0,
                                static_cast<uint32_t>(transparentSets.size()),
                                transparentSets.data(),
                                0,
                                nullptr);

        const PushConstants transparentPushConstants = basePushConstants;
        vkCmdPushConstants(commandBuffer,
                           transparentPipeline_.layout(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           static_cast<uint32_t>(sizeof(PushConstants)),
                           &transparentPushConstants);

        // Direct draws rather than multi-draw indirect. The GPU cull compacts
        // visible commands within a batch, and compaction is by definition
        // order-destroying -- it would undo the back-to-front sort these draws
        // depend on. firstInstance still carries the object-data slot exactly as
        // the indirect path does, so the vertex shader is unchanged.
        const renderer::Mesh* boundTransparentMesh = nullptr;
        for (uint32_t drawIndex = blendRange.begin; drawIndex < blendRange.end; ++drawIndex) {
            const DrawItem& drawItem = visibleDrawItems_[drawIndex];
            if (!drawItem.mesh || drawItem.indexCount == 0 || drawItem.frameDataIndex >= kMaxDrawItems) {
                continue;
            }

            if (boundTransparentMesh != drawItem.mesh) {
                const VkBuffer vertexBuffers[] = {drawItem.mesh->vertexBuffer()};
                const VkDeviceSize vertexOffsets[] = {0};
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, drawItem.mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                boundTransparentMesh = drawItem.mesh;
            }

            vkCmdDrawIndexed(commandBuffer,
                             drawItem.indexCount,
                             1,
                             drawItem.firstIndex,
                             drawItem.vertexOffset,
                             drawItem.frameDataIndex);
        }

        renderGraph_.endTransparentPass();
        rhi::debug::endLabel(commandBuffer);
    }

    recordDepthPyramidCommands(commandBuffer);

    if (taaActiveThisFrame) {
        postProcess_.recordTaaResolveCommands(commandBuffer);
    }

    postProcess_.recordLegacyBloomCommands(commandBuffer);
    postProcess_.recordMipChainBloomCommands(commandBuffer);

    postProcess_.recordLuminanceCommands(commandBuffer);
    postProcess_.recordHistogramCommands(commandBuffer);

    // The probe-only view is a view of a linear radiance value, so it bypasses
    // the display pipeline entirely. Auto-exposure would otherwise cancel
    // exactly the brightness change the view exists to show.
    const float probeDebugGain =
        (giSettings_.enabled && giSettings_.debugIrradianceOnly) ? std::max(giSettings_.previewGain, 0.01f) : 0.0f;
    postProcess_.recordCompositeCommands(commandBuffer, frameJitteredProjection_, probeDebugGain);

    recordPortfolioScreenshotCopy(commandBuffer, imageIndex);

    rhi::debug::beginLabel(commandBuffer, "ImGuiPass");
    const bool imguiProfileScope = gpuProfiler_.beginScope(currentFrame_, commandBuffer, "ImGuiPass");
    renderGraph_.beginImGuiPass();
    imguiLayer_.render(commandBuffer);
    renderGraph_.endImGuiPass();
    if (imguiProfileScope) {
        gpuProfiler_.endScope(currentFrame_, commandBuffer);
    }
    rhi::debug::endLabel(commandBuffer);

    if (taaActiveThisFrame) {
        postProcess_.advanceTaaHistory();
    }

    gpuProfiler_.endFrame(currentFrame_, commandBuffer);
    rhi::debug::endLabel(commandBuffer);
    renderGraph_.endFrame();
}

} // namespace ve
