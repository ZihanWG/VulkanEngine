#pragma once

// A self-contained skinned mesh: geometry + a parallel skinning stream (joint
// indices/weights), a skeleton, and per-frame joint-matrix palette buffers the
// vertex shader reads through buffer-device-address. Phase 1 fills it with a
// procedural bone chain + a sine-wave bend animation so skinning is demoable
// without an external asset; the GPU only consumes the palette, all pose math
// comes from the unit-tested SkeletalAnimation core.

#include "renderer/SkeletalAnimation.h"
#include "rhi/VulkanBuffer.h"

#include <array>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace ve::rhi {
class VulkanContext;
class VulkanCommandContext;
} // namespace ve::rhi

namespace ve::renderer {

// Per-vertex skinning data (vertex binding 1, parallel to the geometry stream).
struct SkinningVertex {
    glm::uvec4 jointIndices{0, 0, 0, 0};
    glm::vec4 weights{0.0f, 0.0f, 0.0f, 0.0f};
};

// Vertex input for the skinning pipeline: binding 0 is the standard geometry
// Vertex, binding 1 is SkinningVertex. Attributes 0-4 match the static mesh,
// attributes 5-6 are the joint indices/weights.
[[nodiscard]] std::array<VkVertexInputBindingDescription, 2> skinnedVertexBindingDescriptions();
[[nodiscard]] std::array<VkVertexInputAttributeDescription, 7> skinnedVertexAttributeDescriptions();

class SkinnedMesh final {
public:
    static constexpr uint32_t kMaxJoints = 64;

    void create(rhi::VulkanContext& context, const rhi::VulkanCommandContext& commandContext, uint32_t frameCount);

    // Recomputes the joint-matrix palette for the frame from the procedural
    // animation and uploads it to paletteBuffers_[frameIndex].
    void update(uint32_t frameIndex, float timeSeconds);

    [[nodiscard]] bool valid() const { return indexCount_ > 0; }
    [[nodiscard]] VkBuffer geometryBuffer() const { return geometryBuffer_.buffer(); }
    [[nodiscard]] VkBuffer skinningBuffer() const { return skinningBuffer_.buffer(); }
    [[nodiscard]] VkBuffer indexBuffer() const { return indexBuffer_.buffer(); }
    [[nodiscard]] uint32_t indexCount() const { return indexCount_; }
    [[nodiscard]] uint32_t jointCount() const { return static_cast<uint32_t>(skeleton_.jointCount()); }
    [[nodiscard]] VkDeviceAddress jointPaletteAddress(uint32_t frameIndex) const;
    [[nodiscard]] const glm::mat4& modelMatrix() const { return modelMatrix_; }

private:
    glm::mat4 modelMatrix_{1.0f};
    Skeleton skeleton_;
    rhi::VulkanBuffer geometryBuffer_;
    rhi::VulkanBuffer skinningBuffer_;
    rhi::VulkanBuffer indexBuffer_;
    uint32_t indexCount_ = 0;
    std::vector<rhi::VulkanBuffer> paletteBuffers_;
};

} // namespace ve::renderer
