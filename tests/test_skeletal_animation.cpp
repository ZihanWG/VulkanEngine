#include "renderer/SkeletalAnimation.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using ve::renderer::AnimationChannel;
using ve::renderer::AnimationClip;
using ve::renderer::AnimationPath;
using ve::renderer::computeJointMatrices;
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

// --- Bounds of a posed skinned mesh ------------------------------------------
//
// The property that matters is one-sided: the bound may be loose, but a posed
// vertex outside it is a shadow that silently stops being drawn.

namespace {

using ve::renderer::Aabb;
using ve::renderer::computeJointBindBounds;
using ve::renderer::skinnedWorldBounds;

bool containsPoint(const Aabb& bounds, const glm::vec3& point, float epsilon = 1e-4f)
{
    return point.x >= bounds.min.x - epsilon && point.x <= bounds.max.x + epsilon &&
           point.y >= bounds.min.y - epsilon && point.y <= bounds.max.y + epsilon &&
           point.z >= bounds.min.z - epsilon && point.z <= bounds.max.z + epsilon;
}

// Linear-blend skinning, exactly as the vertex shader does it.
glm::vec3 skinVertex(const std::vector<glm::mat4>& jointMatrices,
                     const glm::mat4& model,
                     const glm::vec3& position,
                     const glm::uvec4& joints,
                     const glm::vec4& weights)
{
    glm::vec4 skinned(0.0f);
    for (int influence = 0; influence < 4; ++influence) {
        if (weights[influence] <= 0.0f) {
            continue;
        }
        skinned += weights[influence] * (jointMatrices[joints[influence]] * glm::vec4(position, 1.0f));
    }
    return glm::vec3(model * skinned);
}

} // namespace

TEST_CASE("Joint bind bounds only cover the vertices a joint moves", "[skinning][bounds]")
{
    const std::vector<glm::vec3> positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f}};
    const std::vector<glm::uvec4> joints = {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}};
    const std::vector<glm::vec4> weights = {
        {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};

    const std::vector<Aabb> bounds = computeJointBindBounds(2, positions, joints, weights);
    REQUIRE(bounds.size() == 2);

    // Joint 0 owns the first two vertices and must not have been grown by the
    // third, which belongs to joint 1 and sits far away.
    CHECK(bounds[0].min.x == Catch::Approx(0.0f));
    CHECK(bounds[0].max.x == Catch::Approx(1.0f));
    CHECK(bounds[1].min.x == Catch::Approx(5.0f));
    CHECK(bounds[1].max.x == Catch::Approx(5.0f));
}

TEST_CASE("A joint that influences nothing gets an empty box", "[skinning][bounds]")
{
    const std::vector<glm::vec3> positions = {{0.0f, 0.0f, 0.0f}};
    const std::vector<glm::uvec4> joints = {{0, 0, 0, 0}};
    // The three trailing influences carry zero weight and must not pull joint 0's
    // box onto vertices they do not move.
    const std::vector<glm::vec4> weights = {{1.0f, 0.0f, 0.0f, 0.0f}};

    const std::vector<Aabb> bounds = computeJointBindBounds(3, positions, joints, weights);
    REQUIRE(bounds.size() == 3);
    CHECK(bounds[0].valid());
    CHECK_FALSE(bounds[1].valid());
    CHECK_FALSE(bounds[2].valid());
}

TEST_CASE("An out-of-range joint index is dropped, not clamped", "[skinning][bounds]")
{
    const std::vector<glm::vec3> positions = {{0.0f, 0.0f, 0.0f}, {100.0f, 0.0f, 0.0f}};
    const std::vector<glm::uvec4> joints = {{0, 0, 0, 0}, {7, 0, 0, 0}};
    const std::vector<glm::vec4> weights = {{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};

    const std::vector<Aabb> bounds = computeJointBindBounds(1, positions, joints, weights);
    REQUIRE(bounds.size() == 1);
    // Clamping index 7 to joint 0 would have stretched this box to x=100 and
    // hidden a broken import behind an enormous but "valid" bound.
    CHECK(bounds[0].max.x == Catch::Approx(0.0f));
}

TEST_CASE("The bind pose bounds the mesh where it actually is", "[skinning][bounds]")
{
    const Skeleton skeleton = makeTwoJointChain();
    const std::vector<glm::vec3> positions = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    const std::vector<glm::uvec4> joints = {{0, 0, 0, 0}, {1, 0, 0, 0}};
    const std::vector<glm::vec4> weights = {{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};

    const std::vector<Aabb> bindBounds = computeJointBindBounds(skeleton.jointCount(), positions, joints, weights);
    const std::vector<glm::mat4> identityPalette = computeJointMatrices(skeleton, skeleton.bindPose);

    const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 0.0f, 0.0f));
    const Aabb bounds = skinnedWorldBounds(bindBounds, identityPalette, model);

    REQUIRE(bounds.valid());
    // Bind pose skinning matrices are identity, so this is just the model
    // transform applied to the mesh: 10 and 11 on X, nothing on the other axes.
    CHECK(bounds.min.x == Catch::Approx(10.0f));
    CHECK(bounds.max.x == Catch::Approx(11.0f));
    CHECK(bounds.min.y == Catch::Approx(0.0f));
    CHECK(bounds.max.y == Catch::Approx(0.0f));
}

TEST_CASE("A posed vertex never escapes the bound", "[skinning][bounds]")
{
    const Skeleton skeleton = makeTwoJointChain();
    // Deliberately blended: a vertex split between both joints is the case a
    // per-joint box could miss if the union were not a convex-hull argument.
    const std::vector<glm::vec3> positions = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 0.25f, 0.0f}, {1.0f, -0.3f, 0.2f}};
    const std::vector<glm::uvec4> joints = {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};
    const std::vector<glm::vec4> weights = {{1.0f, 0.0f, 0.0f, 0.0f},
                                            {1.0f, 0.0f, 0.0f, 0.0f},
                                            {0.5f, 0.5f, 0.0f, 0.0f},
                                            {1.0f, 0.0f, 0.0f, 0.0f}};

    const std::vector<Aabb> bindBounds = computeJointBindBounds(skeleton.jointCount(), positions, joints, weights);
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(-2.6f, 0.0f, 0.0f));

    // Sweep the pose rather than testing one: the bound has to hold at every
    // angle the animation passes through, not at a lucky one.
    for (int step = 0; step < 24; ++step) {
        const float angle = glm::radians(static_cast<float>(step) * 15.0f);
        std::vector<JointPose> poses = skeleton.bindPose;
        poses[0].rotation = glm::angleAxis(angle, glm::vec3(0.0f, 0.0f, 1.0f));
        poses[1].rotation = glm::angleAxis(-angle * 0.5f, glm::vec3(0.0f, 0.0f, 1.0f));

        const std::vector<glm::mat4> palette = computeJointMatrices(skeleton, poses);
        const Aabb bounds = skinnedWorldBounds(bindBounds, palette, model);
        REQUIRE(bounds.valid());

        for (size_t vertex = 0; vertex < positions.size(); ++vertex) {
            const glm::vec3 skinned =
                skinVertex(palette, model, positions[vertex], joints[vertex], weights[vertex]);
            INFO("step " << step << " vertex " << vertex);
            CHECK(containsPoint(bounds, skinned));
        }
    }
}

TEST_CASE("An empty rig produces an invalid bound rather than a huge one", "[skinning][bounds]")
{
    const Aabb bounds = skinnedWorldBounds({}, {}, glm::mat4(1.0f));
    // Invalid is the honest answer, and callers skip it. A zero-size box at the
    // origin would be a bound that silently covers nothing while looking real.
    CHECK_FALSE(bounds.valid());
}
