#include "renderer/SkeletalAnimation.h"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

namespace ve::renderer {

namespace {

// Find the keyframe segment [i, i+1] containing time and the local interpolation
// factor. Returns false when there is a single keyframe or time clamps to an end.
bool keyframeSegment(const std::vector<float>& times, float time, size_t& index, float& factor)
{
    index = 0;
    factor = 0.0f;
    if (times.size() < 2) {
        return false;
    }
    if (time <= times.front()) {
        index = 0;
        return false;
    }
    if (time >= times.back()) {
        index = times.size() - 1;
        return false;
    }

    // First keyframe strictly after time; the segment is [it-1, it].
    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    const size_t next = static_cast<size_t>(upper - times.begin());
    index = next - 1;
    const float t0 = times[index];
    const float t1 = times[next];
    const float span = t1 - t0;
    factor = span > 0.0f ? (time - t0) / span : 0.0f;
    return true;
}

} // namespace

glm::mat4 JointPose::matrix() const
{
    const glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
    const glm::mat4 r = glm::mat4_cast(rotation);
    const glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
    return t * r * s;
}

glm::vec3 sampleVec3Channel(const AnimationChannel& channel, float time)
{
    if (channel.values.empty()) {
        return glm::vec3(0.0f);
    }

    size_t index = 0;
    float factor = 0.0f;
    if (!keyframeSegment(channel.times, time, index, factor)) {
        return glm::vec3(channel.values[index]);
    }

    const glm::vec3 a{channel.values[index]};
    const glm::vec3 b{channel.values[index + 1]};
    return glm::mix(a, b, factor);
}

glm::quat sampleQuatChannel(const AnimationChannel& channel, float time)
{
    if (channel.values.empty()) {
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }

    const auto toQuat = [](const glm::vec4& v) {
        // Stored as (x, y, z, w); glm::quat constructor takes (w, x, y, z).
        return glm::quat(v.w, v.x, v.y, v.z);
    };

    size_t index = 0;
    float factor = 0.0f;
    if (!keyframeSegment(channel.times, time, index, factor)) {
        return glm::normalize(toQuat(channel.values[index]));
    }

    return glm::normalize(glm::slerp(toQuat(channel.values[index]), toQuat(channel.values[index + 1]), factor));
}

std::vector<JointPose> sampleLocalPoses(const Skeleton& skeleton, const AnimationClip& clip, float time)
{
    std::vector<JointPose> poses = skeleton.bindPose;
    poses.resize(skeleton.jointCount());

    for (const AnimationChannel& channel : clip.channels) {
        if (channel.joint >= poses.size()) {
            continue;
        }
        JointPose& pose = poses[channel.joint];
        switch (channel.path) {
        case AnimationPath::Translation:
            pose.translation = sampleVec3Channel(channel, time);
            break;
        case AnimationPath::Rotation:
            pose.rotation = sampleQuatChannel(channel, time);
            break;
        case AnimationPath::Scale:
            pose.scale = sampleVec3Channel(channel, time);
            break;
        }
    }

    return poses;
}

std::vector<glm::mat4> computeJointMatrices(const Skeleton& skeleton, const std::vector<JointPose>& localPoses)
{
    const size_t jointCount = skeleton.jointCount();
    std::vector<glm::mat4> global(jointCount, glm::mat4(1.0f));
    std::vector<glm::mat4> jointMatrices(jointCount, glm::mat4(1.0f));
    if (localPoses.size() < jointCount) {
        return jointMatrices;
    }

    std::vector<char> resolved(jointCount, 0);

    // Iterative resolve up the parent chain so arbitrary joint ordering works
    // (glTF does not guarantee parents precede children).
    std::vector<size_t> stack;
    for (size_t start = 0; start < jointCount; ++start) {
        if (resolved[start]) {
            continue;
        }
        stack.clear();
        size_t current = start;
        // Walk up until a resolved joint or a root, pushing the unresolved chain.
        while (true) {
            stack.push_back(current);
            const int parent = skeleton.parents[current];
            if (parent < 0 || resolved[static_cast<size_t>(parent)]) {
                break;
            }
            current = static_cast<size_t>(parent);
        }
        // Resolve from the top of the chain down.
        for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
            const size_t joint = *it;
            const glm::mat4 local = localPoses[joint].matrix();
            const int parent = skeleton.parents[joint];
            global[joint] = parent < 0 ? local : global[static_cast<size_t>(parent)] * local;
            resolved[joint] = 1;
        }
    }

    for (size_t joint = 0; joint < jointCount; ++joint) {
        jointMatrices[joint] = global[joint] * skeleton.inverseBind[joint];
    }
    return jointMatrices;
}

std::vector<glm::mat4> computeJointMatricesAtTime(const Skeleton& skeleton, const AnimationClip& clip, float time)
{
    return computeJointMatrices(skeleton, sampleLocalPoses(skeleton, clip, time));
}

std::vector<Aabb> computeJointBindBounds(size_t jointCount,
                                         std::span<const glm::vec3> positions,
                                         std::span<const glm::uvec4> jointIndices,
                                         std::span<const glm::vec4> weights)
{
    std::vector<Aabb> bounds(jointCount);
    const size_t vertexCount = std::min({positions.size(), jointIndices.size(), weights.size()});

    for (size_t vertex = 0; vertex < vertexCount; ++vertex) {
        for (int influence = 0; influence < 4; ++influence) {
            if (!(weights[vertex][influence] > 0.0f)) {
                continue;
            }
            // An index past the palette is dropped rather than clamped: clamping
            // would grow joint 0's box by geometry it does not move, which shows
            // up as a shadow bound that is too large everywhere instead of an
            // import that is visibly wrong once.
            const uint32_t joint = jointIndices[vertex][influence];
            if (joint >= jointCount) {
                continue;
            }
            bounds[joint].expand(positions[vertex]);
        }
    }

    return bounds;
}

Aabb skinnedWorldBounds(std::span<const Aabb> jointBindBounds,
                        std::span<const glm::mat4> jointMatrices,
                        const glm::mat4& model)
{
    Aabb bounds{};
    const size_t jointCount = std::min(jointBindBounds.size(), jointMatrices.size());

    for (size_t joint = 0; joint < jointCount; ++joint) {
        if (!jointBindBounds[joint].valid()) {
            continue;
        }
        bounds.merge(jointBindBounds[joint].transform(jointMatrices[joint]));
    }

    if (!bounds.valid()) {
        return bounds;
    }

    return bounds.transform(model);
}

} // namespace ve::renderer
