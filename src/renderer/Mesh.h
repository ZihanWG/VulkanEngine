#pragma once

#include "rhi/VulkanBuffer.h"

#include <array>
#include <cstdint>
#include <glm/vec3.hpp>

namespace ve::renderer {

struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

[[nodiscard]] VkVertexInputBindingDescription vertexBindingDescription();
[[nodiscard]] std::array<VkVertexInputAttributeDescription, 2> vertexAttributeDescriptions();

class Mesh final {
public:
    Mesh() = default;
    ~Mesh() = default;

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;

    [[nodiscard]] static Mesh createCube(
        rhi::VulkanContext& context,
        const rhi::VulkanCommandContext& commandContext);

    [[nodiscard]] VkBuffer vertexBuffer() const { return vertexBuffer_.buffer(); }
    [[nodiscard]] VkBuffer indexBuffer() const { return indexBuffer_.buffer(); }
    [[nodiscard]] uint32_t indexCount() const { return indexCount_; }

private:
    // Mesh owns the GPU-local buffers for one drawable piece of geometry.
    rhi::VulkanBuffer vertexBuffer_;
    rhi::VulkanBuffer indexBuffer_;
    uint32_t indexCount_ = 0;
};

} // namespace ve::renderer
