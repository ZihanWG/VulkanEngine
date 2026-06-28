#pragma once

// GPU-independent glTF skin + animation import. Parses a rigged, animated glTF
// into CPU data: a combined skinned-vertex array, indices, a Skeleton (parents,
// inverse-bind, bind pose), and AnimationClips — all consumed by the shared,
// unit-tested SkeletalAnimation core. Lives in VulkanEngineCore so the import can
// be unit-tested without a GPU; the renderer splits the vertices into GPU streams.

#include "renderer/SkeletalAnimation.h"

#include <cstdint>
#include <filesystem>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <vector>

namespace ve::renderer {

// One imported skinned vertex (geometry + skinning influences combined).
struct SkinnedImportVertex {
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    glm::vec2 uv{0.0f, 0.0f};
    glm::uvec4 joints{0, 0, 0, 0};
    glm::vec4 weights{1.0f, 0.0f, 0.0f, 0.0f};
};

struct SkinnedGltf {
    bool valid = false;
    std::string error;
    std::vector<SkinnedImportVertex> vertices;
    std::vector<uint32_t> indices;
    Skeleton skeleton;
    std::vector<AnimationClip> clips;
};

// Loads the first skinned mesh + its skin and all animations from an ASCII or
// binary glTF. On failure returns { valid = false, error = ... }.
[[nodiscard]] SkinnedGltf loadSkinnedGltf(const std::filesystem::path& path);

} // namespace ve::renderer
