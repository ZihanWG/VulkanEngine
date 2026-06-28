#include "renderer/GltfSkinnedImport.h"
#include "renderer/SkeletalAnimation.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <glm/glm.hpp>

using ve::renderer::AnimationPath;
using ve::renderer::computeJointMatricesAtTime;
using ve::renderer::loadSkinnedGltf;
using ve::renderer::SkinnedGltf;

namespace {
std::filesystem::path rigPath()
{
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR) / "models" / "skinned_rig.gltf";
}
} // namespace

TEST_CASE("Skinned glTF import parses the skeleton, mesh, and animation", "[gltf][skinning]")
{
    const SkinnedGltf rig = loadSkinnedGltf(rigPath());
    REQUIRE(rig.valid);
    CHECK(rig.error.empty());

    // Two joints in a chain: joint 0 is the root, joint 1 its child.
    REQUIRE(rig.skeleton.jointCount() == 2);
    CHECK(rig.skeleton.parents[0] == -1);
    CHECK(rig.skeleton.parents[1] == 0);

    // Three rings of four vertices; the tube has side quads + two caps.
    CHECK(rig.vertices.size() == 12);
    CHECK(rig.indices.size() == 60);

    // One animation with a single rotation channel targeting the child joint.
    REQUIRE(rig.clips.size() == 1);
    REQUIRE(rig.clips[0].channels.size() == 1);
    CHECK(rig.clips[0].channels[0].joint == 1u);
    CHECK(rig.clips[0].channels[0].path == AnimationPath::Rotation);
    CHECK(rig.clips[0].channels[0].times.size() == 5);
    CHECK(rig.clips[0].duration == Catch::Approx(2.0f));

    // A vertex must reference a real joint with a normalized-ish weight set.
    const glm::vec4 weights = rig.vertices.front().weights;
    CHECK(weights.x + weights.y + weights.z + weights.w == Catch::Approx(1.0f));
}

TEST_CASE("Imported rig is identity at t=0 and bends the child joint mid-clip", "[gltf][skinning]")
{
    const SkinnedGltf rig = loadSkinnedGltf(rigPath());
    REQUIRE(rig.valid);
    REQUIRE(rig.clips.size() == 1);

    // At t=0 the rotation keyframe is identity, so skinning matrices are identity.
    const std::vector<glm::mat4> rest = computeJointMatricesAtTime(rig.skeleton, rig.clips[0], 0.0f);
    REQUIRE(rest.size() == 2);
    for (const glm::mat4& m : rest) {
        CHECK(m[3][0] == Catch::Approx(0.0f).margin(1e-5));
        CHECK(m[3][1] == Catch::Approx(0.0f).margin(1e-5));
        CHECK(m[0][0] == Catch::Approx(1.0f).margin(1e-5));
    }

    // At t=0.5 the child joint is rotated +35 deg about Z. The top of the mesh
    // (bound to joint 1, bind position (0,2,0)) should swing in -X.
    const std::vector<glm::mat4> bent = computeJointMatricesAtTime(rig.skeleton, rig.clips[0], 0.5f);
    const glm::vec4 skinnedTop = bent[1] * glm::vec4(0.0f, 2.0f, 0.0f, 1.0f);
    CHECK(skinnedTop.x < -0.3f);
    CHECK(skinnedTop.y > 1.5f);
}
