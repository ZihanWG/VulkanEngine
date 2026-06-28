#include "renderer/SkeletalAnimation.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using ve::renderer::AnimationChannel;
using ve::renderer::AnimationClip;
using ve::renderer::AnimationPath;
using ve::renderer::computeJointMatricesAtTime;
using ve::renderer::JointPose;
using ve::renderer::sampleQuatChannel;
using ve::renderer::sampleVec3Channel;
using ve::renderer::Skeleton;

namespace {

// A two-joint chain: root at the origin, child offset +1 on X. Inverse-bind is
// the inverse of each joint's bind-space global transform, so the bind pose must
// produce identity skinning matrices.
Skeleton makeTwoJointChain()
{
    Skeleton skeleton;
    skeleton.parents = {-1, 0};

    JointPose rootBind; // identity at origin
    JointPose childBind;
    childBind.translation = glm::vec3(1.0f, 0.0f, 0.0f);
    skeleton.bindPose = {rootBind, childBind};

    const glm::mat4 rootGlobal = rootBind.matrix();
    const glm::mat4 childGlobal = rootGlobal * childBind.matrix();
    skeleton.inverseBind = {glm::inverse(rootGlobal), glm::inverse(childGlobal)};
    return skeleton;
}

} // namespace

TEST_CASE("JointPose::matrix composes translation, rotation, scale", "[skinning]")
{
    JointPose pose;
    pose.translation = glm::vec3(2.0f, 3.0f, 4.0f);
    pose.scale = glm::vec3(2.0f);
    const glm::vec4 point = pose.matrix() * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    CHECK(point.x == Catch::Approx(4.0f)); // 1*2 + 2
    CHECK(point.y == Catch::Approx(3.0f));
    CHECK(point.z == Catch::Approx(4.0f));
}

TEST_CASE("Vec3 channel sampling clamps and interpolates", "[skinning]")
{
    AnimationChannel channel;
    channel.path = AnimationPath::Translation;
    channel.times = {0.0f, 1.0f};
    channel.values = {glm::vec4(0.0f), glm::vec4(10.0f, 0.0f, 0.0f, 0.0f)};

    CHECK(sampleVec3Channel(channel, -1.0f).x == Catch::Approx(0.0f));  // clamp low
    CHECK(sampleVec3Channel(channel, 2.0f).x == Catch::Approx(10.0f));  // clamp high
    CHECK(sampleVec3Channel(channel, 0.5f).x == Catch::Approx(5.0f));   // midpoint lerp
    CHECK(sampleVec3Channel(channel, 0.25f).x == Catch::Approx(2.5f));
}

TEST_CASE("Rotation channel slerps between keyframes", "[skinning]")
{
    AnimationChannel channel;
    channel.path = AnimationPath::Rotation;
    channel.times = {0.0f, 1.0f};
    const glm::quat identity(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::quat ninety = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    channel.values = {glm::vec4(identity.x, identity.y, identity.z, identity.w),
                      glm::vec4(ninety.x, ninety.y, ninety.z, ninety.w)};

    // Halfway should be a 45-degree rotation: it maps +X to (cos45, sin45, 0).
    const glm::quat mid = sampleQuatChannel(channel, 0.5f);
    const glm::vec3 rotated = mid * glm::vec3(1.0f, 0.0f, 0.0f);
    CHECK(rotated.x == Catch::Approx(std::cos(glm::radians(45.0f))).margin(1e-5));
    CHECK(rotated.y == Catch::Approx(std::sin(glm::radians(45.0f))).margin(1e-5));
}

TEST_CASE("Bind pose yields identity skinning matrices", "[skinning]")
{
    const Skeleton skeleton = makeTwoJointChain();
    AnimationClip emptyClip; // no channels -> bind pose

    const std::vector<glm::mat4> matrices = computeJointMatricesAtTime(skeleton, emptyClip, 0.0f);
    REQUIRE(matrices.size() == 2);
    for (const glm::mat4& m : matrices) {
        CHECK(m[0][0] == Catch::Approx(1.0f));
        CHECK(m[3][0] == Catch::Approx(0.0f).margin(1e-5));
        CHECK(m[3][1] == Catch::Approx(0.0f).margin(1e-5));
        CHECK(m[3][2] == Catch::Approx(0.0f).margin(1e-5));
    }
}

TEST_CASE("Rotating the root propagates to the child joint", "[skinning]")
{
    const Skeleton skeleton = makeTwoJointChain();

    // Rotate the root 90 degrees about Z for the whole clip.
    const glm::quat ninety = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    AnimationChannel rootRotation;
    rootRotation.joint = 0;
    rootRotation.path = AnimationPath::Rotation;
    rootRotation.times = {0.0f};
    rootRotation.values = {glm::vec4(ninety.x, ninety.y, ninety.z, ninety.w)};

    AnimationClip clip;
    clip.channels = {rootRotation};

    const std::vector<glm::mat4> matrices = computeJointMatricesAtTime(skeleton, clip, 0.0f);
    REQUIRE(matrices.size() == 2);

    // A vertex bound to the child sits at the child's bind position (1,0,0). After
    // rotating the root 90 deg about Z, it should move to about (0,1,0).
    const glm::vec4 skinned = matrices[1] * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    CHECK(skinned.x == Catch::Approx(0.0f).margin(1e-5));
    CHECK(skinned.y == Catch::Approx(1.0f).margin(1e-5));
    CHECK(skinned.z == Catch::Approx(0.0f).margin(1e-5));
}
