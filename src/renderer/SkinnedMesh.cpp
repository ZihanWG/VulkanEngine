#include "renderer/SkinnedMesh.h"

#include "core/Logger.h"
#include "renderer/GltfSkinnedImport.h"
#include "rhi/VulkanContext.h"

#include <cmath>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace ve::renderer {

namespace {

constexpr uint32_t kSegments = 6;
constexpr float kSegmentHeight = 0.6f;
constexpr float kHalfWidth = 0.22f;
const glm::mat4 kDemoModelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-2.6f, 0.0f, 0.0f));

} // namespace

std::array<VkVertexInputBindingDescription, 2> skinnedVertexBindingDescriptions()
{
    std::array<VkVertexInputBindingDescription, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].stride = static_cast<uint32_t>(sizeof(Vertex));
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = static_cast<uint32_t>(sizeof(SkinningVertex));
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindings;
}

std::array<VkVertexInputAttributeDescription, 7> skinnedVertexAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 7> attributes{};
    // Geometry attributes (binding 0) reuse the static mesh layout.
    const std::array<VkVertexInputAttributeDescription, 5> geometry = vertexAttributeDescriptions();
    for (size_t i = 0; i < geometry.size(); ++i) {
        attributes[i] = geometry[i];
    }

    attributes[5].location = 5;
    attributes[5].binding = 1;
    attributes[5].format = VK_FORMAT_R32G32B32A32_UINT;
    attributes[5].offset = static_cast<uint32_t>(offsetof(SkinningVertex, jointIndices));

    attributes[6].location = 6;
    attributes[6].binding = 1;
    attributes[6].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[6].offset = static_cast<uint32_t>(offsetof(SkinningVertex, weights));
    return attributes;
}

void SkinnedMesh::buildBuffers(rhi::VulkanContext& context,
                               const rhi::VulkanCommandContext& commandContext,
                               uint32_t frameCount,
                               std::span<const Vertex> geometry,
                               std::span<const SkinningVertex> skinning,
                               std::span<const uint32_t> indices)
{
    indexCount_ = static_cast<uint32_t>(indices.size());

    geometryBuffer_.createDeviceLocal(
        context, commandContext, std::as_bytes(geometry), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    skinningBuffer_.createDeviceLocal(
        context, commandContext, std::as_bytes(skinning), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    indexBuffer_.createDeviceLocal(
        context, commandContext, std::as_bytes(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    paletteBuffers_.clear();
    paletteBuffers_.resize(frameCount);
    for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
        rhi::VulkanBufferCreateInfo bufferInfo{};
        bufferInfo.size = static_cast<VkDeviceSize>(kMaxJoints) * sizeof(glm::mat4);
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        bufferInfo.allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        bufferInfo.requestDeviceAddress = true;
        paletteBuffers_[frameIndex].createBuffer(context, bufferInfo);
    }
}

void SkinnedMesh::create(rhi::VulkanContext& context,
                         const rhi::VulkanCommandContext& commandContext,
                         uint32_t frameCount)
{
    // Procedural bone chain: kSegments rigid box segments stacked along +Y, each
    // bound fully to one joint. Rigid weighting keeps the demo unambiguous while
    // exercising the full hierarchy/palette/GPU-skinning path.
    std::vector<Vertex> vertices;
    std::vector<SkinningVertex> skinning;
    std::vector<uint32_t> indices;
    vertices.reserve(kSegments * 8);
    skinning.reserve(kSegments * 8);
    indices.reserve(kSegments * 36);

    // Per-segment box: corner c packs x=bit0, y=bit1, z=bit2.
    static const uint32_t boxIndices[36] = {0, 1, 3, 0, 3, 2,  // -Z
                                            4, 6, 7, 4, 7, 5,  // +Z
                                            0, 4, 5, 0, 5, 1,  // -Y
                                            2, 3, 7, 2, 7, 6,  // +Y
                                            0, 2, 6, 0, 6, 4,  // -X
                                            1, 5, 7, 1, 7, 3}; // +X

    for (uint32_t segment = 0; segment < kSegments; ++segment) {
        // Shrink each box inside its slot so adjacent segments do not share a cap
        // plane (which would z-fight) and have room to not intersect when bent.
        const float slotCenter = (static_cast<float>(segment) + 0.5f) * kSegmentHeight;
        const float halfHeight = kSegmentHeight * 0.5f * 0.78f;
        const float base = slotCenter - halfHeight;
        const float top = slotCenter + halfHeight;
        const glm::vec3 center{0.0f, slotCenter, 0.0f};
        const float gradient = static_cast<float>(segment) / static_cast<float>(kSegments - 1);
        const glm::vec3 color = glm::mix(glm::vec3(0.15f, 0.65f, 0.75f), glm::vec3(0.95f, 0.5f, 0.2f), gradient);

        const uint32_t baseVertex = static_cast<uint32_t>(vertices.size());
        for (uint32_t corner = 0; corner < 8; ++corner) {
            const float x = (corner & 1u) ? kHalfWidth : -kHalfWidth;
            const float y = (corner & 2u) ? top : base;
            const float z = (corner & 4u) ? kHalfWidth : -kHalfWidth;
            const glm::vec3 position{x, y, z};

            Vertex vertex{};
            vertex.position = position;
            vertex.color = color;
            vertex.uv = glm::vec2(0.0f);
            vertex.normal = glm::normalize(position - center);
            vertex.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            vertices.push_back(vertex);

            SkinningVertex skin{};
            skin.jointIndices = glm::uvec4(segment, 0, 0, 0);
            skin.weights = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            skinning.push_back(skin);
        }

        for (uint32_t index : boxIndices) {
            indices.push_back(baseVertex + index);
        }
    }

    buildBuffers(context, commandContext, frameCount, vertices, skinning, indices);

    // Joint chain: joint i is offset +kSegmentHeight from its parent, so its bind
    // global position is (0, i*kSegmentHeight, 0); inverse-bind is the inverse.
    skeleton_ = Skeleton{};
    skeleton_.parents.resize(kSegments);
    skeleton_.bindPose.resize(kSegments);
    skeleton_.inverseBind.resize(kSegments);
    for (uint32_t joint = 0; joint < kSegments; ++joint) {
        skeleton_.parents[joint] = joint == 0 ? -1 : static_cast<int>(joint) - 1;
        JointPose pose;
        pose.translation = joint == 0 ? glm::vec3(0.0f) : glm::vec3(0.0f, kSegmentHeight, 0.0f);
        skeleton_.bindPose[joint] = pose;
        skeleton_.inverseBind[joint] =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -static_cast<float>(joint) * kSegmentHeight, 0.0f));
    }

    hasClip_ = false;
    clipDuration_ = 0.0f;
    sourceName_ = "procedural bone chain";
    modelMatrix_ = kDemoModelMatrix;
}

bool SkinnedMesh::createFromGltf(rhi::VulkanContext& context,
                                 const rhi::VulkanCommandContext& commandContext,
                                 uint32_t frameCount,
                                 const std::filesystem::path& path)
{
    const SkinnedGltf imported = loadSkinnedGltf(path);
    if (!imported.valid) {
        if (!imported.error.empty()) {
            Logger::warn("Skinned glTF import failed; using procedural demo instead: " + imported.error);
        }
        return false;
    }

    std::vector<Vertex> geometry(imported.vertices.size());
    std::vector<SkinningVertex> skinning(imported.vertices.size());
    for (size_t i = 0; i < imported.vertices.size(); ++i) {
        const SkinnedImportVertex& source = imported.vertices[i];
        Vertex vertex{};
        vertex.position = source.position;
        vertex.color = glm::vec3(0.8f);
        vertex.uv = source.uv;
        vertex.normal = source.normal;
        vertex.tangent = source.tangent;
        geometry[i] = vertex;

        SkinningVertex skin{};
        skin.jointIndices = source.joints;
        skin.weights = source.weights;
        skinning[i] = skin;
    }

    buildBuffers(context, commandContext, frameCount, geometry, skinning, imported.indices);

    skeleton_ = imported.skeleton;
    if (!imported.clips.empty()) {
        clip_ = imported.clips.front();
        clipDuration_ = clip_.duration;
        hasClip_ = clipDuration_ > 0.0f;
    } else {
        hasClip_ = false;
        clipDuration_ = 0.0f;
    }
    sourceName_ = path.filename().string();
    modelMatrix_ = kDemoModelMatrix;

    Logger::info("Loaded skinned glTF '" + sourceName_ + "': " + std::to_string(skeleton_.jointCount()) +
                 " joints, " + std::to_string(imported.clips.size()) + " clip(s).");
    return valid();
}

void SkinnedMesh::update(uint32_t frameIndex, float timeSeconds)
{
    if (frameIndex >= paletteBuffers_.size() || skeleton_.jointCount() == 0) {
        return;
    }

    std::vector<glm::mat4> jointMatrices;
    if (hasClip_) {
        const float clipTime = clipDuration_ > 0.0f ? std::fmod(timeSeconds, clipDuration_) : 0.0f;
        jointMatrices = computeJointMatricesAtTime(skeleton_, clip_, clipTime);
    } else {
        // Procedural sine-wave bend: each joint rotates about Z, phase-shifted down
        // the chain so it curls. Pose math goes through the unit-tested core.
        constexpr float kAmplitude = 0.39f; // ~22 degrees
        constexpr float kSpeed = 2.2f;
        constexpr float kPhase = 0.7f;

        std::vector<JointPose> poses = skeleton_.bindPose;
        for (size_t joint = 0; joint < poses.size(); ++joint) {
            const float angle = kAmplitude * std::sin(timeSeconds * kSpeed + static_cast<float>(joint) * kPhase);
            poses[joint].rotation = glm::angleAxis(angle, glm::vec3(0.0f, 0.0f, 1.0f)) * poses[joint].rotation;
        }
        jointMatrices = computeJointMatrices(skeleton_, poses);
    }

    if (jointMatrices.size() > kMaxJoints) {
        jointMatrices.resize(kMaxJoints);
    }
    paletteBuffers_[frameIndex].upload(std::as_bytes(std::span<const glm::mat4>(jointMatrices)));
}

VkDeviceAddress SkinnedMesh::jointPaletteAddress(uint32_t frameIndex) const
{
    return frameIndex < paletteBuffers_.size() ? paletteBuffers_[frameIndex].deviceAddress() : 0;
}

} // namespace ve::renderer
