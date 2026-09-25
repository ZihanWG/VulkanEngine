#pragma once

// GPU-independent skeletal animation core: joint poses, keyframe sampling, and
// the hierarchy flatten that produces skinning matrices. This is the testable
// reference shared by the procedural demo and the glTF import path; the GPU only
// consumes the final joint-matrix palette (uploaded per frame), so all the
// interpolation/hierarchy logic lives here and is unit-tested without a device.

#include "renderer/Bounds.h"

#include <cstdint>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <span>
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

    [[nodiscard]] size_t jointCount() const
    {
        return parents.size();
    }
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
[[nodiscard]] std::vector<JointPose> sampleLocalPoses(const Skeleton& skeleton, const AnimationClip& clip, float time);

// Skinning palette: jointMatrix[i] = global[i] * inverseBind[i], where
// global[i] = global[parent[i]] * local[i]. Handles arbitrary parent ordering.
// At the bind pose this yields identity matrices (no deformation).
[[nodiscard]] std::vector<glm::mat4> computeJointMatrices(const Skeleton& skeleton,
                                                          const std::vector<JointPose>& localPoses);

[[nodiscard]] std::vector<glm::mat4>
computeJointMatricesAtTime(const Skeleton& skeleton, const AnimationClip& clip, float time);

// --- Bounds of a posed skinned mesh -----------------------------------------
//
// A skinned mesh has no fixed bounds: its model matrix does not move while its
// vertices do, which is exactly what defeats every cache and cull that keys on
// the transform. These two produce a bound that follows the pose instead, and
// they are here rather than next to the GPU mesh because the math is the usual
// GPU-free kind and the failure mode -- a bound that is too small -- is a
// missing shadow somewhere far from its cause.

// Bind-space bounds of the vertices each joint actually influences, computed
// once at build time. A vertex counts toward a joint only if its weight there is
// non-zero: a joint that moves nothing gets an empty (invalid) box, which the
// pose bound below then skips.
[[nodiscard]] std::vector<Aabb> computeJointBindBounds(size_t jointCount,
                                                       std::span<const glm::vec3> positions,
                                                       std::span<const glm::uvec4> jointIndices,
                                                       std::span<const glm::vec4> weights);

// Conservative world bounds of the posed mesh.
//
// Conservative and provably so: linear-blend skinning puts a vertex at
// sum(w_i * M_i * p) with weights that are non-negative and sum to one, which is
// a convex combination of the points M_i * p. Each of those lies inside
// M_i applied to that joint's bind box, so the union of the transformed boxes
// contains every posed vertex. Never tight -- a rotating box's AABB grows -- but
// never wrong, which is the direction a shadow bound has to err in.
[[nodiscard]] Aabb skinnedWorldBounds(std::span<const Aabb> jointBindBounds,
                                      std::span<const glm::mat4> jointMatrices,
                                      const glm::mat4& model);

// --- Previous-frame palette, for motion vectors -------------------------------
//
// The velocity buffer needs where a vertex was last frame as well as where it
// is. For a rigid object the previous model matrix answers that; a skinned
// vertex also moves with its joints while the model matrix holds still, so it
// needs last frame's palette too.
//
// Each per-frame palette buffer carries both, this frame's matrices first and
// the previous frame's at kSkinPreviousPaletteOffset. Keeping the pair inside
// one buffer is what makes it safe to read: that buffer is guarded by its own
// frame slot's fence, where reading "last frame's" slot instead would race the
// CPU rewriting it for the frame after -- and with one frame in flight there
// is no other slot at all. Mirrored in joint_palette.glsl.
inline constexpr uint32_t kMaxSkinJoints = 64;
inline constexpr uint32_t kSkinPreviousPaletteOffset = kMaxSkinJoints;
static_assert(kSkinPreviousPaletteOffset >= kMaxSkinJoints,
              "the previous palette must start past the last current joint, or the halves overlap");

// Remembers the palette last passed to advance(), so the next frame can upload
// it as the previous one.
class JointPaletteHistory {
public:
    // Returns the palette to upload as the previous frame's, then remembers
    // `current` for the next call. With nothing to compare against -- the first
    // call after reset(), or a joint count that changed -- it returns `current`
    // itself: zero joint motion is the honest answer there, where a palette
    // from an unrelated pose would be a large, wrong velocity.
    //
    // The returned span stays valid until the next advance() or reset().
    [[nodiscard]] std::span<const glm::mat4> advance(std::span<const glm::mat4> current);

    // Forget the previous palette. For a rebuilt rig, whose joints need not mean
    // what the remembered matrices meant.
    void reset();

private:
    std::vector<glm::mat4> previous_;
    std::vector<glm::mat4> returned_;
    bool hasPrevious_ = false;
};

} // namespace ve::renderer
