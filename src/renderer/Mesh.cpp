#include "renderer/Mesh.h"

#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <array>
#include <cstddef>
#include <span>

namespace ve::renderer {

namespace {

const std::array<Vertex, 8> kCubeVertices = {{
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.1f, 0.1f}},
    {{0.5f, -0.5f, -0.5f}, {0.1f, 1.0f, 0.1f}},
    {{0.5f, 0.5f, -0.5f}, {0.1f, 0.2f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 0.1f}},
    {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.1f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.1f, 1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.5f, 0.1f}},
    {{-0.5f, 0.5f, 0.5f}, {0.8f, 0.8f, 0.8f}}
}};

const std::array<uint16_t, 36> kCubeIndices = {
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    3, 6, 2, 3, 7, 6,
    1, 2, 6, 1, 6, 5,
    0, 4, 7, 0, 7, 3
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

std::array<VkVertexInputAttributeDescription, 2> vertexAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 2> attributes{};

    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = static_cast<uint32_t>(offsetof(Vertex, position));

    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = static_cast<uint32_t>(offsetof(Vertex, color));

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
