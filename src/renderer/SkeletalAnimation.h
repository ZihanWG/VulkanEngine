#pragma once

// GPU-independent skeletal animation core: joint poses, keyframe sampling, and
// the hierarchy flatten that produces skinning matrices. This is the testable
// reference shared by the procedural demo and the glTF import path; the GPU only
// consumes the final joint-matrix palette (uploaded per frame), so all the
// interpolation/hierarchy logic lives here and is unit-tested without a device.

#include <cstdint>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <vector>

namespace ve::renderer {

// Local transform of one joint (TRS). matrix() = T * R * S.
struct JointPose {
    glm::vec3 translation{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // (w, x, y, z) identity
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    [[nodiscard]] glm::mat4 matrix() const;
};

// Joint hierarchy. parents[i] is the parent joint index (-1 for a root),
// inverseBind[i] maps a vertex from model space into joint i's bind space, and
// bindPose[i] is the joint's default local transform (used where a clip has no
// channel for that joint). All three vectors are parallel and sized jointCount().
struct Skeleton {
    std::vector<int> parents;
    std::vector<glm::mat4> inverseBind;
    std::vector<JointPose> bindPose;

    [[nodiscard]] size_t jointCount() const { return parents.size(); }
};

enum class AnimationPath : uint8_t {
    Translation,
    Rotation,
    Scale,
};

// Keyframes for a single joint's translation, rotation, or scale. times is sorted
// ascending; values holds vec3 data in xyz for Translation/Scale and a quaternion
// (x, y, z, w) for Rotation. times.size() == values.size().
struct AnimationChannel {
    uint32_t joint = 0;
    AnimationPath path = AnimationPath::Translation;
    std::vector<float> times;
    std::vector<glm::vec4> values;
};

struct AnimationClip {
    std::string name;
    float duration = 0.0f;
    std::vector<AnimationChannel> channels;
};

// Sample a channel at time t. vec3 channels interpolate linearly; rotation
// channels slerp. Times outside the range clamp to the first/last keyframe.
[[nodiscard]] glm::vec3 sampleVec3Channel(const AnimationChannel& channel, float time);
[[nodiscard]] glm::quat sampleQuatChannel(const AnimationChannel& channel, float time);

// Local poses for every joint at time t: starts from the skeleton bind pose and
// overrides each joint with whatever channels target it.
[[nodiscard]] std::vector<JointPose> sampleLocalPoses(const Skeleton& skeleton,
                                                      const AnimationClip& clip,
                                                      float time);

// Skinning palette: jointMatrix[i] = global[i] * inverseBind[i], where
// global[i] = global[parent[i]] * local[i]. Handles arbitrary parent ordering.
// At the bind pose this yields identity matrices (no deformation).
[[nodiscard]] std::vector<glm::mat4> computeJointMatrices(const Skeleton& skeleton,
                                                         const std::vector<JointPose>& localPoses);

[[nodiscard]] std::vector<glm::mat4> computeJointMatricesAtTime(const Skeleton& skeleton,
                                                               const AnimationClip& clip,
                                                               float time);

} // namespace ve::renderer
