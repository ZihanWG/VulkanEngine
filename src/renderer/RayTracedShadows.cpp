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

} // namespace ve::renderer
