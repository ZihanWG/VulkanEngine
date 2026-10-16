#include "renderer/RayTracedShadows.h"

#include "renderer/GltfGeometry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ve::renderer {

// The shader reads these out of a raw float array, so a field added to Vertex
// moves the UVs it samples without a single validation message.
static_assert(sizeof(Vertex) == kRtVertexStrideFloats * sizeof(float));
static_assert(offsetof(Vertex, uv) == kRtVertexUvOffsetFloats * sizeof(float));

RayTracedCasterKind rayTracedCasterKind(RenderBucket bucket)
{
    switch (bucket) {
    case RenderBucket::Opaque:
        return RayTracedCasterKind::Opaque;
    case RenderBucket::Mask:
        return RayTracedCasterKind::AlphaTested;
    case RenderBucket::Blend:
        break;
    }
    return RayTracedCasterKind::None;
}

std::array<float, 12> rayTracingInstanceTransform(const glm::mat4& model)
{
    std::array<float, 12> rows{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            // glm indexes [column][row].
            rows[static_cast<size_t>(row * 4 + column)] = model[column][row];
        }
    }
    return rows;
}

float rayTracedShadowOffsetPerDistance(float pixelFootprints,
                                       float lodErrorPixels,
                                       float verticalFovRadians,
                                       uint32_t renderHeight)
{
    if (renderHeight == 0) {
        return 0.0f;
    }
    const float pixels = std::max(pixelFootprints, 0.0f) + std::max(lodErrorPixels, 0.0f);
    const float pixelAngle = 2.0f * std::tan(0.5f * verticalFovRadians) / static_cast<float>(renderHeight);
    return pixels * pixelAngle;
}

float rayTracedShadowRayLength(const Aabb& sceneBounds)
{
    if (!sceneBounds.valid()) {
        return kRtFallbackRayLength;
    }
    const glm::vec3 extent = sceneBounds.max - sceneBounds.min;
    const float diagonal = std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
    if (!std::isfinite(diagonal)) {
        return kRtFallbackRayLength;
    }
    // The margin covers the origin's own offset off its surface, and a scene
    // whose draw items all sit at one point.
    return diagonal * 1.01f + 1.0f;
}

uint32_t rayTracedShadowMaskGroupCount(uint32_t pixels)
{
    return (pixels + kRtMaskLocalSize - 1u) / kRtMaskLocalSize;
}

uint32_t rayTracingPreSkinGroupCount(uint32_t vertexCount)
{
    return (vertexCount + kRtPreSkinLocalSize - 1) / kRtPreSkinLocalSize;
}

uint64_t alignUpPowerOfTwo(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

void RayTracingTopLevelTracker::reset(uint32_t frameSlots)
{
    content_.clear();
    hasContent_ = false;
    generation_ = 0;
    builtGeneration_ = 0;
    refittedPose_ = 0;
    slotGenerations_.assign(frameSlots, 0);
    planned_ = {};
    plannedGeneration_ = 0;
    plannedPose_ = 0;
}

RayTracingTopLevelPlan RayTracingTopLevelTracker::plan(uint32_t frameSlot,
                                                       std::span<const std::byte> instances,
                                                       uint64_t skinnedPose,
                                                       bool rebuildEveryFrame)
{
    // Against the previous frame's content, not the built one's: a generation
    // is a run of frames with equal instance records, and each slot and the
    // structure record which run they last caught up with. The pose is not part
    // of it -- a slot's buffers hold the records, which do not move with it.
    if (!hasContent_ || !std::ranges::equal(instances, content_)) {
        content_.assign(instances.begin(), instances.end());
        hasContent_ = true;
        ++generation_;
    }
    if (frameSlot >= slotGenerations_.size()) {
        slotGenerations_.resize(frameSlot + 1, 0);
    }

    RayTracingTopLevelPlan plan{};
    plan.refitSkinned = skinnedPose != 0 && (rebuildEveryFrame || skinnedPose != refittedPose_);
    // A refit moves the skinned BLAS's bounds, which the TLAS copied into its
    // instance when it was built, so it rebuilds even over equal records.
    plan.build = rebuildEveryFrame || plan.refitSkinned || builtGeneration_ != generation_;
    plan.writeSlot = rebuildEveryFrame || slotGenerations_[frameSlot] != generation_;
    // Recorded now, because the caller writes the slot before anything this
    // frame reads it. The structure is different: a planned build may never be
    // recorded, so it waits for commitBuild.
    slotGenerations_[frameSlot] = generation_;

    planned_ = plan;
    plannedGeneration_ = generation_;
    plannedPose_ = skinnedPose;
    return plan;
}

void RayTracingTopLevelTracker::commitBuild()
{
    if (!planned_.build) {
        return;
    }
    builtGeneration_ = plannedGeneration_;
    if (planned_.refitSkinned) {
        refittedPose_ = plannedPose_;
    }
    planned_ = {};
}

} // namespace ve::renderer
