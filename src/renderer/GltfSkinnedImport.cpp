#include "renderer/GltfSkinnedImport.h"

// This is the single tinygltf implementation translation unit for the whole
// build; other users (e.g. Mesh.cpp) include the header without the define.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <glm/gtc/quaternion.hpp>
#include <unordered_map>
#include <utility>

namespace ve::renderer {

namespace {

// Pointer + stride for an accessor's tightly-/interleaved-packed elements.
struct AccessorView {
    const unsigned char* data = nullptr;
    size_t stride = 0;
    size_t count = 0;
    int componentType = 0;
    int type = 0;
};

[[nodiscard]] AccessorView accessorView(const tinygltf::Model& model, int accessorIndex)
{
    AccessorView view;
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) {
        return view;
    }
    const tinygltf::Accessor& accessor = model.accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.bufferView < 0) {
        return view;
    }
    const tinygltf::BufferView& bufferView = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView.buffer)];
    view.data = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    view.count = accessor.count;
    view.componentType = accessor.componentType;
    view.type = accessor.type;
    const int stride = accessor.ByteStride(bufferView);
    view.stride = stride > 0 ? static_cast<size_t>(stride) : 0;
    return view;
}

[[nodiscard]] float readComponentAsFloat(const unsigned char* element, int componentType, size_t component)
{
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        float value = 0.0f;
        std::memcpy(&value, element + component * sizeof(float), sizeof(float));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<float>(element[component]);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t value = 0;
        std::memcpy(&value, element + component * sizeof(uint16_t), sizeof(uint16_t));
        return static_cast<float>(value);
    }
    default:
        return 0.0f;
    }
}

[[nodiscard]] uint32_t readComponentAsUint(const unsigned char* element, int componentType, size_t component)
{
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return static_cast<uint32_t>(element[component]);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t value = 0;
        std::memcpy(&value, element + component * sizeof(uint16_t), sizeof(uint16_t));
        return static_cast<uint32_t>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        uint32_t value = 0;
        std::memcpy(&value, element + component * sizeof(uint32_t), sizeof(uint32_t));
        return value;
    }
    default:
        return 0;
    }
}

[[nodiscard]] size_t componentSize(int componentType)
{
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return 4;
    default:
        return 4;
    }
}

[[nodiscard]] size_t componentCountForType(int type)
{
    switch (type) {
    case TINYGLTF_TYPE_SCALAR:
        return 1;
    case TINYGLTF_TYPE_VEC2:
        return 2;
    case TINYGLTF_TYPE_VEC3:
        return 3;
    case TINYGLTF_TYPE_VEC4:
        return 4;
    case TINYGLTF_TYPE_MAT4:
        return 16;
    default:
        return 1;
    }
}

[[nodiscard]] const unsigned char* element(const AccessorView& view, size_t index)
{
    const size_t stride =
        view.stride != 0 ? view.stride : componentSize(view.componentType) * componentCountForType(view.type);
    return view.data + index * stride;
}

JointPose nodeLocalPose(const tinygltf::Node& node)
{
    JointPose pose;
    if (node.translation.size() == 3) {
        pose.translation = glm::vec3(static_cast<float>(node.translation[0]),
                                     static_cast<float>(node.translation[1]),
                                     static_cast<float>(node.translation[2]));
    }
    if (node.rotation.size() == 4) {
        pose.rotation = glm::quat(static_cast<float>(node.rotation[3]),
                                  static_cast<float>(node.rotation[0]),
                                  static_cast<float>(node.rotation[1]),
                                  static_cast<float>(node.rotation[2]));
    }
    if (node.scale.size() == 3) {
        pose.scale = glm::vec3(
            static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]), static_cast<float>(node.scale[2]));
    }
    return pose;
}

AnimationPath pathFromString(const std::string& path)
{
    if (path == "rotation") {
        return AnimationPath::Rotation;
    }
    if (path == "scale") {
        return AnimationPath::Scale;
    }
    return AnimationPath::Translation;
}

} // namespace

SkinnedGltf loadSkinnedGltf(const std::filesystem::path& path)
{
    SkinnedGltf result;

    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string error;
    std::string warning;
    const std::string pathString = path.string();
    const bool binary = path.extension() == ".glb";
    const bool loaded = binary ? loader.LoadBinaryFromFile(&model, &error, &warning, pathString)
                               : loader.LoadASCIIFromFile(&model, &error, &warning, pathString);
    if (!loaded) {
        result.error = error.empty() ? ("failed to load glTF: " + pathString) : error;
        return result;
    }

    // Find the first node that has both a mesh and a skin.
    int meshNodeIndex = -1;
    for (size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex) {
        const tinygltf::Node& node = model.nodes[nodeIndex];
        if (node.mesh >= 0 && node.skin >= 0) {
            meshNodeIndex = static_cast<int>(nodeIndex);
            break;
        }
    }
    if (meshNodeIndex < 0) {
        result.error = "glTF has no skinned mesh (a node with both a mesh and a skin).";
        return result;
    }

    const tinygltf::Node& meshNode = model.nodes[static_cast<size_t>(meshNodeIndex)];
    const tinygltf::Skin& skin = model.skins[static_cast<size_t>(meshNode.skin)];
    const size_t jointCount = skin.joints.size();
    if (jointCount == 0) {
        result.error = "glTF skin has no joints.";
        return result;
    }

    // node index -> joint index within this skin.
    std::unordered_map<int, uint32_t> nodeToJoint;
    for (size_t jointIndex = 0; jointIndex < jointCount; ++jointIndex) {
        nodeToJoint[skin.joints[jointIndex]] = static_cast<uint32_t>(jointIndex);
    }

    // node index -> parent node index (scan children).
    std::unordered_map<int, int> nodeParent;
    for (size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex) {
        for (int child : model.nodes[nodeIndex].children) {
            nodeParent[child] = static_cast<int>(nodeIndex);
        }
    }

    Skeleton& skeleton = result.skeleton;
    skeleton.parents.resize(jointCount);
    skeleton.bindPose.resize(jointCount);
    skeleton.inverseBind.assign(jointCount, glm::mat4(1.0f));
    for (size_t jointIndex = 0; jointIndex < jointCount; ++jointIndex) {
        const int jointNode = skin.joints[jointIndex];
        const tinygltf::Node& node = model.nodes[static_cast<size_t>(jointNode)];
        skeleton.bindPose[jointIndex] = nodeLocalPose(node);
        const auto parentIt = nodeParent.find(jointNode);
        if (parentIt != nodeParent.end()) {
            const auto jointIt = nodeToJoint.find(parentIt->second);
            skeleton.parents[jointIndex] = jointIt != nodeToJoint.end() ? static_cast<int>(jointIt->second) : -1;
        } else {
            skeleton.parents[jointIndex] = -1;
        }
    }

    if (skin.inverseBindMatrices >= 0) {
        const AccessorView ibmView = accessorView(model, skin.inverseBindMatrices);
        for (size_t jointIndex = 0; jointIndex < jointCount && jointIndex < ibmView.count; ++jointIndex) {
            const unsigned char* matrixData = element(ibmView, jointIndex);
            glm::mat4 matrix(1.0f);
            for (size_t component = 0; component < 16; ++component) {
                const float value = readComponentAsFloat(matrixData, ibmView.componentType, component);
                matrix[static_cast<int>(component / 4)][static_cast<int>(component % 4)] = value;
            }
            skeleton.inverseBind[jointIndex] = matrix;
        }
    }

    // Geometry: first primitive of the skinned mesh.
    const tinygltf::Mesh& mesh = model.meshes[static_cast<size_t>(meshNode.mesh)];
    if (mesh.primitives.empty()) {
        result.error = "glTF skinned mesh has no primitives.";
        return result;
    }
    const tinygltf::Primitive& primitive = mesh.primitives.front();

    const auto attribute = [&](const char* name) {
        const auto it = primitive.attributes.find(name);
        return it != primitive.attributes.end() ? it->second : -1;
    };

    const AccessorView positions = accessorView(model, attribute("POSITION"));
    if (positions.data == nullptr) {
        result.error = "glTF skinned primitive has no POSITION attribute.";
        return result;
    }
    const AccessorView normals = accessorView(model, attribute("NORMAL"));
    const AccessorView tangents = accessorView(model, attribute("TANGENT"));
    const AccessorView uvs = accessorView(model, attribute("TEXCOORD_0"));
    const AccessorView joints = accessorView(model, attribute("JOINTS_0"));
    const AccessorView weights = accessorView(model, attribute("WEIGHTS_0"));

    result.vertices.resize(positions.count);
    for (size_t vertexIndex = 0; vertexIndex < positions.count; ++vertexIndex) {
        SkinnedImportVertex& vertex = result.vertices[vertexIndex];
        const unsigned char* p = element(positions, vertexIndex);
        vertex.position = glm::vec3(readComponentAsFloat(p, positions.componentType, 0),
                                    readComponentAsFloat(p, positions.componentType, 1),
                                    readComponentAsFloat(p, positions.componentType, 2));
        if (normals.data != nullptr && vertexIndex < normals.count) {
            const unsigned char* n = element(normals, vertexIndex);
            vertex.normal = glm::vec3(readComponentAsFloat(n, normals.componentType, 0),
                                      readComponentAsFloat(n, normals.componentType, 1),
                                      readComponentAsFloat(n, normals.componentType, 2));
        }
        if (tangents.data != nullptr && vertexIndex < tangents.count) {
            const unsigned char* t = element(tangents, vertexIndex);
            vertex.tangent = glm::vec4(readComponentAsFloat(t, tangents.componentType, 0),
                                       readComponentAsFloat(t, tangents.componentType, 1),
                                       readComponentAsFloat(t, tangents.componentType, 2),
                                       readComponentAsFloat(t, tangents.componentType, 3));
        }
        if (uvs.data != nullptr && vertexIndex < uvs.count) {
            const unsigned char* u = element(uvs, vertexIndex);
            vertex.uv = glm::vec2(readComponentAsFloat(u, uvs.componentType, 0),
                                  readComponentAsFloat(u, uvs.componentType, 1));
        }
        if (joints.data != nullptr && vertexIndex < joints.count) {
            const unsigned char* j = element(joints, vertexIndex);
            vertex.joints = glm::uvec4(readComponentAsUint(j, joints.componentType, 0),
                                       readComponentAsUint(j, joints.componentType, 1),
                                       readComponentAsUint(j, joints.componentType, 2),
                                       readComponentAsUint(j, joints.componentType, 3));
        }
        if (weights.data != nullptr && vertexIndex < weights.count) {
            const unsigned char* w = element(weights, vertexIndex);
            vertex.weights = glm::vec4(readComponentAsFloat(w, weights.componentType, 0),
                                       readComponentAsFloat(w, weights.componentType, 1),
                                       readComponentAsFloat(w, weights.componentType, 2),
                                       readComponentAsFloat(w, weights.componentType, 3));
        }
    }

    if (primitive.indices >= 0) {
        const AccessorView indexView = accessorView(model, primitive.indices);
        result.indices.resize(indexView.count);
        for (size_t i = 0; i < indexView.count; ++i) {
            result.indices[i] = readComponentAsUint(element(indexView, i), indexView.componentType, 0);
        }
    }

    // Animations.
    for (const tinygltf::Animation& animation : model.animations) {
        AnimationClip clip;
        clip.name = animation.name;
        for (const tinygltf::AnimationChannel& channel : animation.channels) {
            const auto jointIt = nodeToJoint.find(channel.target_node);
            if (jointIt == nodeToJoint.end()) {
                continue; // channel targets a non-joint node
            }
            if (channel.sampler < 0 || channel.sampler >= static_cast<int>(animation.samplers.size())) {
                continue;
            }
            const tinygltf::AnimationSampler& sampler = animation.samplers[static_cast<size_t>(channel.sampler)];
            const AccessorView input = accessorView(model, sampler.input);
            const AccessorView output = accessorView(model, sampler.output);
            if (input.data == nullptr || output.data == nullptr) {
                continue;
            }

            AnimationChannel outChannel;
            outChannel.joint = jointIt->second;
            outChannel.path = pathFromString(channel.target_path);
            outChannel.times.resize(input.count);
            for (size_t k = 0; k < input.count; ++k) {
                outChannel.times[k] = readComponentAsFloat(element(input, k), input.componentType, 0);
                clip.duration = std::max(clip.duration, outChannel.times[k]);
            }

            const size_t componentsPerValue = outChannel.path == AnimationPath::Rotation ? 4 : 3;
            outChannel.values.resize(output.count);
            for (size_t k = 0; k < output.count; ++k) {
                const unsigned char* valueData = element(output, k);
                glm::vec4 value(0.0f);
                for (size_t component = 0; component < componentsPerValue; ++component) {
                    value[static_cast<int>(component)] =
                        readComponentAsFloat(valueData, output.componentType, component);
                }
                outChannel.values[k] = value;
            }
            clip.channels.push_back(std::move(outChannel));
        }
        if (!clip.channels.empty()) {
            result.clips.push_back(std::move(clip));
        }
    }

    result.valid = !result.vertices.empty() && !result.indices.empty();
    if (!result.valid && result.error.empty()) {
        result.error = "glTF skinned mesh produced no geometry.";
    }
    return result;
}

} // namespace ve::renderer
