#include "renderer/GltfSkinnedImport.h"
#include "renderer/SkeletalAnimation.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <json.hpp>
#include <string>
#include <vector>

using ve::renderer::AnimationPath;
using ve::renderer::computeJointMatricesAtTime;
using ve::renderer::loadSkinnedGltf;
using ve::renderer::sampleQuatChannel;
using ve::renderer::sampleVec3Channel;
using ve::renderer::SkinnedGltf;

namespace {
std::filesystem::path rigPath()
{
    return std::filesystem::path(VULKAN_ENGINE_ASSET_DIR) / "models" / "skinned_rig.gltf";
}

// glTF component types, spelled out rather than pulled from tinygltf so the
// test states the encoding it exercises.
constexpr int kByte = 5120;
constexpr int kUnsignedByte = 5121;
constexpr int kShort = 5122;
constexpr int kUnsignedShort = 5123;
constexpr int kFloat = 5126;

// The smallest skinned glTF that isolates one accessor encoding or one sampler:
// a three-vertex triangle skinned to a two-joint chain (node 1 the root, node 2
// its child), written as a .gltf beside its .bin in a scratch directory.
class SkinnedGltfBuilder {
public:
    // Appends `data` as its own buffer view and accessor, 4-byte aligned as
    // glTF requires, and returns the accessor index.
    template <typename T>
    int accessor(const std::vector<T>& data, int componentType, const char* type, size_t count, bool normalized = false)
    {
        while (bin_.size() % 4 != 0) {
            bin_.push_back(0);
        }
        const size_t offset = bin_.size();
        const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
        bin_.insert(bin_.end(), bytes, bytes + data.size() * sizeof(T));
        views_.push_back({{"buffer", 0}, {"byteOffset", offset}, {"byteLength", data.size() * sizeof(T)}});
        nlohmann::json entry = {
            {"bufferView", views_.size() - 1}, {"componentType", componentType}, {"count", count}, {"type", type}};
        if (normalized) {
            entry["normalized"] = true;
        }
        accessors_.push_back(entry);
        return static_cast<int>(accessors_.size() - 1);
    }

    // The triangle. Weights are supplied by the caller, since their encoding is
    // what several tests are about.
    void triangle(int weightsAccessor)
    {
        const std::vector<float> positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        const std::vector<uint8_t> joints = {0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
        const std::vector<uint16_t> indices = {0, 1, 2};
        primitive_ = {{"attributes",
                       {{"POSITION", accessor(positions, kFloat, "VEC3", 3)},
                        {"JOINTS_0", accessor(joints, kUnsignedByte, "VEC4", 3)},
                        {"WEIGHTS_0", weightsAccessor}}},
                      {"indices", accessor(indices, kUnsignedShort, "SCALAR", 3)}};
    }

    // One animation channel on the child joint (node 2).
    void channel(const char* path, int input, int output, const char* interpolation)
    {
        samplers_.push_back({{"input", input}, {"output", output}, {"interpolation", interpolation}});
        channels_.push_back({{"sampler", samplers_.size() - 1}, {"target", {{"node", 2}, {"path", path}}}});
    }

    [[nodiscard]] SkinnedGltf load(const std::string& caseName) const
    {
        const std::filesystem::path directory =
            std::filesystem::temp_directory_path() / "VulkanEngineTests" / "skinned_gltf" / caseName;
        std::filesystem::create_directories(directory);
        {
            std::ofstream binFile(directory / "rig.bin", std::ios::binary | std::ios::trunc);
            binFile.write(reinterpret_cast<const char*>(bin_.data()), static_cast<std::streamsize>(bin_.size()));
        }
        nlohmann::json document = {
            {"asset", {{"version", "2.0"}}},
            {"nodes",
             nlohmann::json::array({{{"mesh", 0}, {"skin", 0}}, {{"children", {2}}}, {{"translation", {0, 1, 0}}}})},
            {"skins", nlohmann::json::array({{{"joints", {1, 2}}}})},
            {"meshes", nlohmann::json::array({{{"primitives", nlohmann::json::array({primitive_})}}})},
            {"buffers", nlohmann::json::array({{{"uri", "rig.bin"}, {"byteLength", bin_.size()}}})},
            {"bufferViews", views_},
            {"accessors", accessors_},
        };
        if (!channels_.empty()) {
            document["animations"] = nlohmann::json::array({{{"samplers", samplers_}, {"channels", channels_}}});
        }
        {
            std::ofstream gltfFile(directory / "rig.gltf", std::ios::trunc);
            gltfFile << document.dump(2);
        }
        return loadSkinnedGltf(directory / "rig.gltf");
    }

private:
    std::vector<uint8_t> bin_;
    nlohmann::json views_ = nlohmann::json::array();
    nlohmann::json accessors_ = nlohmann::json::array();
    nlohmann::json primitive_;
    nlohmann::json samplers_ = nlohmann::json::array();
    nlohmann::json channels_ = nlohmann::json::array();
};

int floatWeights(SkinnedGltfBuilder& builder)
{
    const std::vector<float> weights = {1, 0, 0, 0, 0.5f, 0.5f, 0, 0, 0, 1, 0, 0};
    return builder.accessor(weights, kFloat, "VEC4", 3);
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

TEST_CASE("Normalized integer weights decode to fractions of one", "[gltf][skinning]")
{
    const auto expectUnitSums = [](const SkinnedGltf& rig) {
        REQUIRE(rig.vertices.size() == 3);
        for (const auto& vertex : rig.vertices) {
            const glm::vec4 w = vertex.weights;
            CHECK(w.x + w.y + w.z + w.w == Catch::Approx(1.0f));
        }
    };

    SECTION("unsigned byte")
    {
        SkinnedGltfBuilder builder;
        const std::vector<uint8_t> weights = {255, 0, 0, 0, 128, 127, 0, 0, 0, 255, 0, 0};
        builder.triangle(builder.accessor(weights, kUnsignedByte, "VEC4", 3, /*normalized=*/true));
        const SkinnedGltf rig = builder.load("normalized_ubyte_weights");
        REQUIRE(rig.valid);
        expectUnitSums(rig);
        CHECK(rig.vertices[1].weights.x == Catch::Approx(128.0f / 255.0f));
        CHECK(rig.vertices[1].weights.y == Catch::Approx(127.0f / 255.0f));
    }

    SECTION("unsigned short")
    {
        SkinnedGltfBuilder builder;
        const std::vector<uint16_t> weights = {65535, 0, 0, 0, 32768, 32767, 0, 0, 0, 65535, 0, 0};
        builder.triangle(builder.accessor(weights, kUnsignedShort, "VEC4", 3, /*normalized=*/true));
        const SkinnedGltf rig = builder.load("normalized_ushort_weights");
        REQUIRE(rig.valid);
        expectUnitSums(rig);
        CHECK(rig.vertices[1].weights.x == Catch::Approx(32768.0f / 65535.0f));
    }

    SECTION("float weights are read unchanged")
    {
        SkinnedGltfBuilder builder;
        builder.triangle(floatWeights(builder));
        const SkinnedGltf rig = builder.load("float_weights");
        REQUIRE(rig.valid);
        expectUnitSums(rig);
        CHECK(rig.vertices[1].weights.x == Catch::Approx(0.5f));
    }
}

TEST_CASE("Normalized signed rotation keys decode, down to the most negative code", "[gltf][skinning]")
{
    SkinnedGltfBuilder builder;
    builder.triangle(floatWeights(builder));
    const std::vector<float> times = {0.0f, 1.0f, 2.0f};
    // Identity, 90 degrees about +Z, and identity again written as w = -32768,
    // which glTF decodes to exactly -1 -- the same rotation as +1.
    const std::vector<int16_t> rotations = {0, 0, 0, 32767, 0, 0, 23170, 23170, 0, 0, 0, -32768};
    builder.channel("rotation",
                    builder.accessor(times, kFloat, "SCALAR", 3),
                    builder.accessor(rotations, kShort, "VEC4", 3, /*normalized=*/true),
                    "LINEAR");
    const SkinnedGltf rig = builder.load("normalized_short_rotation");
    REQUIRE(rig.valid);
    REQUIRE(rig.clips.size() == 1);
    REQUIRE(rig.clips[0].channels.size() == 1);
    const auto& channel = rig.clips[0].channels[0];

    REQUIRE(channel.values.size() == 3);
    CHECK(channel.values[1].z == Catch::Approx(23170.0f / 32767.0f));
    CHECK(channel.values[2].w == -1.0f);

    const glm::vec3 quarterTurn = sampleQuatChannel(channel, 1.0f) * glm::vec3(1.0f, 0.0f, 0.0f);
    CHECK(quarterTurn.x == Catch::Approx(0.0f).margin(1e-4));
    CHECK(quarterTurn.y == Catch::Approx(1.0f).margin(1e-4));
    const glm::vec3 fullTurn = sampleQuatChannel(channel, 2.0f) * glm::vec3(1.0f, 0.0f, 0.0f);
    CHECK(fullTurn.x == Catch::Approx(1.0f).margin(1e-4));
}

TEST_CASE("A STEP sampler holds each keyframe until the next", "[gltf][skinning]")
{
    SkinnedGltfBuilder builder;
    builder.triangle(floatWeights(builder));
    const std::vector<float> times = {0.0f, 1.0f};
    const std::vector<float> translations = {0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};
    builder.channel("translation",
                    builder.accessor(times, kFloat, "SCALAR", 2),
                    builder.accessor(translations, kFloat, "VEC3", 2),
                    "STEP");
    const SkinnedGltf rig = builder.load("step_translation");
    REQUIRE(rig.valid);
    REQUIRE(rig.clips.size() == 1);
    REQUIRE(rig.clips[0].channels.size() == 1);
    const auto& channel = rig.clips[0].channels[0];

    // Linear would give 1.0 at the midpoint.
    CHECK(sampleVec3Channel(channel, 0.5f).x == Catch::Approx(0.0f));
    CHECK(sampleVec3Channel(channel, 0.999f).x == Catch::Approx(0.0f));
    CHECK(sampleVec3Channel(channel, 1.0f).x == Catch::Approx(2.0f));
}

TEST_CASE("A CUBICSPLINE sampler reads the tangents around each keyframe", "[gltf][skinning]")
{
    SkinnedGltfBuilder builder;
    builder.triangle(floatWeights(builder));
    const std::vector<float> times = {0.0f, 1.0f};
    // Per keyframe: in-tangent, value, out-tangent. From 0 to 1 along X, leaving
    // the first key with slope 2 and arriving at the second with slope 0.
    const std::vector<float> output = {
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        2.0f,
        0.0f,
        0.0f, // key 0
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f, // key 1
    };
    builder.channel("translation",
                    builder.accessor(times, kFloat, "SCALAR", 2),
                    builder.accessor(output, kFloat, "VEC3", 6),
                    "CUBICSPLINE");
    const SkinnedGltf rig = builder.load("cubic_translation");
    REQUIRE(rig.valid);
    REQUIRE(rig.clips.size() == 1);
    REQUIRE(rig.clips[0].channels.size() == 1);
    const auto& channel = rig.clips[0].channels[0];

    // The curve passes through the values, never the tangents stored beside them.
    CHECK(sampleVec3Channel(channel, 0.0f).x == Catch::Approx(0.0f));
    CHECK(sampleVec3Channel(channel, 1.0f).x == Catch::Approx(1.0f));
    // Hermite at t = 0.5: h01 * 1 + h10 * 2 * (1 s) = 0.5 + 0.125 * 2. Linear
    // would give 0.5.
    CHECK(sampleVec3Channel(channel, 0.5f).x == Catch::Approx(0.75f));
}

TEST_CASE("Channels the importer cannot honour are skipped and reported", "[gltf][skinning]")
{
    const std::vector<float> times = {0.0f, 1.0f};

    SECTION("a morph-target weights channel")
    {
        SkinnedGltfBuilder builder;
        builder.triangle(floatWeights(builder));
        const std::vector<float> morphWeights = {0.0f, 1.0f};
        builder.channel("weights",
                        builder.accessor(times, kFloat, "SCALAR", 2),
                        builder.accessor(morphWeights, kFloat, "SCALAR", 2),
                        "LINEAR");
        const SkinnedGltf rig = builder.load("morph_weights_channel");
        REQUIRE(rig.valid);
        CHECK(rig.clips.empty());
        REQUIRE(rig.warnings.size() == 1);
        CHECK(rig.warnings[0].find("morph") != std::string::npos);
    }

    SECTION("a sampler with the wrong number of values for its keyframes")
    {
        SkinnedGltfBuilder builder;
        builder.triangle(floatWeights(builder));
        // Three linear values for two keyframes, and a cubic sampler with one
        // value per keyframe instead of three.
        const std::vector<float> threeValues = {0, 0, 0, 1, 0, 0, 2, 0, 0};
        const std::vector<float> twoValues = {0, 0, 0, 1, 0, 0};
        builder.channel("translation",
                        builder.accessor(times, kFloat, "SCALAR", 2),
                        builder.accessor(threeValues, kFloat, "VEC3", 3),
                        "LINEAR");
        builder.channel("scale",
                        builder.accessor(times, kFloat, "SCALAR", 2),
                        builder.accessor(twoValues, kFloat, "VEC3", 2),
                        "CUBICSPLINE");
        const SkinnedGltf rig = builder.load("mismatched_value_counts");
        REQUIRE(rig.valid);
        CHECK(rig.clips.empty());
        CHECK(rig.warnings.size() == 2);
    }
}
