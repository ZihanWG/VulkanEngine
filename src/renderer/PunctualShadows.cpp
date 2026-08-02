#include "renderer/PunctualShadows.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <algorithm>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

// Identity slot handed back for out-of-range queries so a bookkeeping slip
// degrades into "no shadow" instead of reading past the end of the vectors.
const glm::mat4 kIdentityViewProjection{1.0f};
const Frustum kEmptyFrustum{};

} // namespace

void PunctualShadows::create(rhi::VulkanContext& context, uint32_t frameCount)
{
    reset();
    context_ = &context;

    try {
        atlas_.create(context, kPunctualShadowAtlasSize, kPunctualShadowAtlasSize, /*layerCount=*/1);

        slotBuffers_.resize(frameCount);
        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
            rhi::VulkanBufferCreateInfo bufferInfo{};
            bufferInfo.size = static_cast<VkDeviceSize>(kMaxPunctualShadowSlots) * sizeof(GpuShadowSlot);
            bufferInfo.usage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
            bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            bufferInfo.requestDeviceAddress = true;
            slotBuffers_[frameIndex].createBuffer(context, bufferInfo);
            rhi::debug::setObjectName(context.vkDevice(),
                                      slotBuffers_[frameIndex].buffer(),
                                      VK_OBJECT_TYPE_BUFFER,
                                      "PunctualShadowSlotBuffer" + std::to_string(frameIndex));
        }

        slots_.reserve(kMaxPunctualShadowSlots);
        slotFrustums_.reserve(kMaxPunctualShadowSlots);

        Logger::info("Punctual shadow atlas enabled: " + std::to_string(kPunctualShadowAtlasSize) + "x" +
                     std::to_string(kPunctualShadowAtlasSize) + " with " +
                     std::to_string(kMaxPunctualShadowSlots) + " tiles of " +
                     std::to_string(kPunctualShadowTileSize) + "px.");
    } catch (const std::exception& error) {
        // Punctual lights still shade, they just do not cast. Matching the
        // clustered path's policy: a missing optional subsystem degrades the
        // frame instead of failing device creation.
        reset();
        Logger::warn(std::string("Punctual shadow atlas unavailable; point/spot lights will not cast shadows: ") +
                     error.what());
    }
}

void PunctualShadows::reset()
{
    slotBuffers_.clear();
    atlas_.reset();
    slots_.clear();
    slotFrustums_.clear();
    allocator_.reset();
    context_ = nullptr;
}

void PunctualShadows::beginFrame()
{
    allocator_.reset();
    slots_.clear();
    slotFrustums_.clear();
}

uint32_t PunctualShadows::addSpotLight(const glm::vec3& position,
                                       const glm::vec3& direction,
                                       float outerAngleRadians,
                                       float range)
{
    if (!valid()) {
        return kInvalidPunctualShadowSlot;
    }

    const uint32_t slot = allocator_.allocate();
    if (slot == kInvalidPunctualShadowSlot) {
        return kInvalidPunctualShadowSlot;
    }

    const glm::mat4 viewProjection =
        computeSpotShadowViewProjection(position, direction, outerAngleRadians, range);

    GpuShadowSlot record{};
    record.viewProjection = viewProjection;
    record.atlasUvOffsetScale = punctualShadowSlotUvOffsetScale(slot);
    record.params = glm::vec4(constantBias_,
                              normalBias_,
                              // One atlas texel in UV: the PCF tap step. Derived
                              // from the atlas (not the tile) because the shader
                              // steps in atlas UV after the tile remap.
                              1.0f / static_cast<float>(kPunctualShadowAtlasSize),
                              0.0f);

    // slots_ is indexed by slot, and the bump allocator hands them out in order,
    // so appending keeps the two in step.
    slots_.push_back(record);
    slotFrustums_.push_back(computeSpotShadowFrustum(viewProjection));

    return slot;
}

uint32_t PunctualShadows::addPointLight(const glm::vec3& position, float range)
{
    if (!valid()) {
        return kInvalidPunctualShadowSlot;
    }

    const uint32_t baseSlot = allocator_.allocateRange(kPointShadowFaceCount);
    if (baseSlot == kInvalidPunctualShadowSlot) {
        return kInvalidPunctualShadowSlot;
    }

    for (uint32_t face = 0; face < kPointShadowFaceCount; ++face) {
        const glm::mat4 viewProjection = computePointShadowFaceViewProjection(position, face, range);

        GpuShadowSlot record{};
        record.viewProjection = viewProjection;
        record.atlasUvOffsetScale = punctualShadowSlotUvOffsetScale(baseSlot + face);
        record.params = glm::vec4(
            constantBias_, normalBias_, 1.0f / static_cast<float>(kPunctualShadowAtlasSize), 0.0f);

        slots_.push_back(record);
        slotFrustums_.push_back(computeSpotShadowFrustum(viewProjection));
    }

    return baseSlot;
}

void PunctualShadows::upload(uint32_t frameIndex)
{
    if (frameIndex >= slotBuffers_.size() || slots_.empty()) {
        return;
    }

    const size_t count = std::min<size_t>(slots_.size(), kMaxPunctualShadowSlots);
    slotBuffers_[frameIndex].upload(std::as_bytes(std::span<const GpuShadowSlot>(slots_.data(), count)));
}

VkDeviceAddress PunctualShadows::slotBufferAddress(uint32_t frameIndex) const
{
    if (frameIndex >= slotBuffers_.size()) {
        return 0;
    }

    return slotBuffers_[frameIndex].deviceAddress();
}

const glm::mat4& PunctualShadows::slotViewProjection(uint32_t slot) const
{
    if (slot >= slots_.size()) {
        return kIdentityViewProjection;
    }

    return slots_[slot].viewProjection;
}

const Frustum& PunctualShadows::slotFrustum(uint32_t slot) const
{
    if (slot >= slotFrustums_.size()) {
        return kEmptyFrustum;
    }

    return slotFrustums_[slot];
}

} // namespace ve::renderer
