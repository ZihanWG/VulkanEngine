#include "renderer/Mesh.h"

#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <array>
#include <cstddef>
#include <span>

namespace ve::renderer {

namespace {

const std::array<Vertex, 24> kCubeVertices = {{
    // Front
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
    // Back
    {{0.5f, -0.5f, -0.5f}, {0.9f, 0.9f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.9f, 0.9f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.9f, 0.9f, 1.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.9f, 0.9f, 1.0f}, {0.0f, 0.0f}},
    // Left
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.9f, 0.9f}, {0.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.9f, 0.9f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.9f, 0.9f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.9f, 0.9f}, {0.0f, 0.0f}},
    // Right
    {{0.5f, -0.5f, 0.5f}, {0.9f, 1.0f, 0.9f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.9f, 1.0f, 0.9f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.9f, 1.0f, 0.9f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.9f, 1.0f, 0.9f}, {0.0f, 0.0f}},
    // Top
    {{-0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 0.9f}, {0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 0.9f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.9f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.9f}, {0.0f, 0.0f}},
    // Bottom
    {{-0.5f, -0.5f, -0.5f}, {0.9f, 1.0f, 1.0f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.9f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.9f, 1.0f, 1.0f}, {1.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.9f, 1.0f, 1.0f}, {0.0f, 0.0f}}
}};

const std::array<uint16_t, 36> kCubeIndices = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
};

} // namespace

VkVertexInputBindingDescription vertexBindingDescription()
{
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = static_cast<uint32_t>(sizeof(Vertex));
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding;
}

std::array<VkVertexInputAttributeDescription, 3> vertexAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 3> attributes{};

    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = static_cast<uint32_t>(offsetof(Vertex, position));

    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = static_cast<uint32_t>(offsetof(Vertex, color));

    attributes[2].location = 2;
    attributes[2].binding = 0;
    attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[2].offset = static_cast<uint32_t>(offsetof(Vertex, uv));

    return attributes;
}

Mesh Mesh::createCube(rhi::VulkanContext& context, const rhi::VulkanCommandContext& commandContext)
{
    Mesh mesh;

    mesh.vertexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const Vertex>(kCubeVertices.data(), kCubeVertices.size())),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    mesh.indexBuffer_.createDeviceLocal(
        context,
        commandContext,
        std::as_bytes(std::span<const uint16_t>(kCubeIndices.data(), kCubeIndices.size())),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    mesh.indexCount_ = static_cast<uint32_t>(kCubeIndices.size());
    return mesh;
}

} // namespace ve::renderer
