// The "Asset inputs:" digest: which asset files a run read, hashed so two logs
// can be diffed for "did these runs render the same inputs".

#include "assets/AssetInputs.h"

#include "assets/AssetManager.h"
#include "core/JobSystem.h"
#include "renderer/GltfSkinnedImport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using ve::assets::AssetInputDigest;
using ve::assets::AssetInputFile;
using ve::assets::AssetInputRecorder;
using ve::assets::AssetInputRoot;

std::filesystem::path makeCaseDir(std::string_view caseName)
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "VulkanEngineTests" / "asset_inputs" / caseName;
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    std::filesystem::create_directories(directory, ec);
    return directory;
}

void writeFile(const std::filesystem::path& path, std::string_view contents)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

// A miniature assets/ tree: a texture, a material and a model, the three kinds
// of file the renderer records.
void writeAssetTree(const std::filesystem::path& root)
{
    writeFile(root / "textures" / "checker_mr.png", "metallic-roughness texels");
    writeFile(root / "materials" / "ground.material.json", R"({"roughness": 0.86})");
    writeFile(root / "models" / "rig.gltf", R"({"asset": {"version": "2.0"}})");
}

std::vector<std::filesystem::path> assetTreePaths(const std::filesystem::path& root)
{
    return {root / "textures" / "checker_mr.png",
            root / "materials" / "ground.material.json",
            root / "models" / "rig.gltf"};
}

AssetInputDigest digestTree(const std::filesystem::path& root,
                            const std::vector<std::filesystem::path>& paths,
                            ve::JobSystem* jobSystem = nullptr)
{
    const std::vector<AssetInputRoot> roots{{"assets", root}};
    return ve::assets::digestAssetInputs(ve::assets::assetInputFiles(paths, roots), jobSystem);
}

} // namespace

TEST_CASE("Asset input digest is stable for the same files", "[asset_inputs]")
{
    const std::filesystem::path root = makeCaseDir("stable") / "assets";
    writeAssetTree(root);

    const AssetInputDigest first = digestTree(root, assetTreePaths(root));
    const AssetInputDigest second = digestTree(root, assetTreePaths(root));
    REQUIRE(first.fileCount == 3);
    REQUIRE(first.unreadableCount == 0);
    REQUIRE(first.digest == second.digest);

    SECTION("the same files in another checkout hash the same")
    {
        // A worktree holds the same assets at a different absolute path; the
        // digest is about the files, so it must not see the difference.
        const std::filesystem::path otherRoot = makeCaseDir("stable_other_checkout") / "nested" / "assets";
        writeAssetTree(otherRoot);
        REQUIRE(digestTree(otherRoot, assetTreePaths(otherRoot)).digest == first.digest);
    }

    SECTION("hashing on the job system gives the same digest as hashing inline")
    {
        ve::JobSystem jobs(4);
        REQUIRE(digestTree(root, assetTreePaths(root), &jobs).digest == first.digest);
    }
}

TEST_CASE("Asset input digest changes when one byte of one file does", "[asset_inputs]")
{
    const std::filesystem::path root = makeCaseDir("one_byte") / "assets";
    writeAssetTree(root);
    const AssetInputDigest original = digestTree(root, assetTreePaths(root));

    // Same size, one byte different -- the shape of PR #110's re-encoded
    // checker_mr.png, which a size or file-count check would have missed.
    writeFile(root / "textures" / "checker_mr.png", "metallic-roughness texelS");
    const AssetInputDigest edited = digestTree(root, assetTreePaths(root));
    REQUIRE(edited.fileCount == original.fileCount);
    REQUIRE(edited.digest != original.digest);

    writeFile(root / "textures" / "checker_mr.png", "metallic-roughness texels");
    REQUIRE(digestTree(root, assetTreePaths(root)).digest == original.digest);
}

TEST_CASE("Asset input digest does not depend on load order", "[asset_inputs]")
{
    const std::filesystem::path root = makeCaseDir("order") / "assets";
    writeAssetTree(root);
    std::vector<std::filesystem::path> paths = assetTreePaths(root);
    const AssetInputDigest forward = digestTree(root, paths);

    std::reverse(paths.begin(), paths.end());
    REQUIRE(digestTree(root, paths).digest == forward.digest);

    SECTION("a file listed twice counts once")
    {
        paths.push_back(paths.front());
        const AssetInputDigest repeated = digestTree(root, paths);
        REQUIRE(repeated.fileCount == forward.fileCount);
        REQUIRE(repeated.digest == forward.digest);
    }

    SECTION("through the recorder, in any order and under any spelling")
    {
        AssetInputRecorder::reset();
        AssetInputRecorder::record(root / "models" / "rig.gltf");
        AssetInputRecorder::record(root / "textures" / ".." / "textures" / "checker_mr.png");
        AssetInputRecorder::record(root / "materials" / "ground.material.json");
        AssetInputRecorder::record(root / "models" / "rig.gltf");
        const std::vector<std::filesystem::path> recorded = AssetInputRecorder::snapshot();
        AssetInputRecorder::reset();

        REQUIRE(recorded.size() == 3);
        REQUIRE(digestTree(root, recorded).digest == forward.digest);
    }
}

TEST_CASE("Asset input digest covers which files were read, not just their bytes", "[asset_inputs]")
{
    const std::filesystem::path root = makeCaseDir("membership") / "assets";
    writeAssetTree(root);
    const std::vector<std::filesystem::path> paths = assetTreePaths(root);
    const AssetInputDigest all = digestTree(root, paths);

    SECTION("one file fewer")
    {
        const std::vector<std::filesystem::path> fewer(paths.begin(), paths.end() - 1);
        const AssetInputDigest digest = digestTree(root, fewer);
        REQUIRE(digest.fileCount == 2);
        REQUIRE(digest.digest != all.digest);
    }

    SECTION("the same bytes under another name")
    {
        // A cooked sidecar read instead of its source is a different input even
        // if, improbably, the bytes matched.
        std::filesystem::rename(root / "models" / "rig.gltf", root / "models" / "rig2.gltf");
        std::vector<std::filesystem::path> renamed = paths;
        renamed.back() = root / "models" / "rig2.gltf";
        REQUIRE(digestTree(root, renamed).digest != all.digest);
    }

    SECTION("a file that has gone since it was loaded")
    {
        std::filesystem::remove(root / "models" / "rig.gltf");
        const AssetInputDigest digest = digestTree(root, paths);
        REQUIRE(digest.fileCount == 3);
        REQUIRE(digest.unreadableCount == 1);
        REQUIRE(digest.digest != all.digest);
    }

    SECTION("a file that has gone is not the same as an empty one")
    {
        writeFile(root / "models" / "rig.gltf", "");
        const AssetInputDigest empty = digestTree(root, paths);
        std::filesystem::remove(root / "models" / "rig.gltf");
        const AssetInputDigest missing = digestTree(root, paths);
        REQUIRE(empty.unreadableCount == 0);
        REQUIRE(empty.digest != missing.digest);
    }
}

TEST_CASE("Asset input keys are relative to the most specific root", "[asset_inputs]")
{
    const std::filesystem::path base = makeCaseDir("keys");
    const std::vector<AssetInputRoot> roots{{"assets", base / "assets"}, {"sponza", base / "assets" / "sponza"}};

    REQUIRE(ve::assets::assetInputKey(base / "assets" / "textures" / "checker.png", roots) ==
            "assets/textures/checker.png");
    REQUIRE(ve::assets::assetInputKey(base / "assets" / "sponza" / "Sponza.gltf", roots) == "sponza/Sponza.gltf");
    REQUIRE(ve::assets::assetInputKey(base / "assets" / "models" / ".." / "textures" / "a.png", roots) ==
            "assets/textures/a.png");

    // Outside every root: the whole normalized path, forward slashes included,
    // rather than a short name that two unrelated files could share.
    const std::filesystem::path outside = base / "elsewhere" / "env.hdr";
    REQUIRE(ve::assets::assetInputKey(outside, roots) == outside.lexically_normal().generic_string());
    // A sibling whose name merely starts with the root's is not inside it.
    REQUIRE(ve::assets::assetInputKey(base / "assets2" / "x.png", roots) ==
            (base / "assets2" / "x.png").lexically_normal().generic_string());
}

TEST_CASE("Asset input recorder resolves glTF buffers", "[asset_inputs]")
{
    const std::filesystem::path root = makeCaseDir("gltf_buffers");
    writeFile(root / "scene.gltf", "{}");
    writeFile(root / "scene.bin", "vertex bytes");

    AssetInputRecorder::reset();
    AssetInputRecorder::recordGltf(
        root / "scene.gltf",
        std::vector<std::string>{"scene.bin", "data:application/octet-stream;base64,AAAA", "", "missing.bin"});
    const std::vector<std::filesystem::path> recorded = AssetInputRecorder::snapshot();
    AssetInputRecorder::reset();

    // The glTF and the one external buffer that exists; never the embedded or
    // the unresolvable ones.
    const std::vector<AssetInputRoot> roots{{"scene", root}};
    std::vector<std::string> keys;
    for (const std::filesystem::path& path : recorded) {
        keys.push_back(ve::assets::assetInputKey(path, roots));
    }
    REQUIRE(keys == std::vector<std::string>{"scene/scene.bin", "scene/scene.gltf"});
}

TEST_CASE("Asset loaders record the files they read", "[asset_inputs]")
{
    const std::filesystem::path assetRoot(VULKAN_ENGINE_ASSET_DIR);
    const std::vector<AssetInputRoot> roots{{"assets", assetRoot}};
    const auto recordedKeys = [&roots]() {
        std::vector<std::string> keys;
        for (const std::filesystem::path& path : AssetInputRecorder::snapshot()) {
            keys.push_back(ve::assets::assetInputKey(path, roots));
        }
        return keys;
    };

    SECTION("glTF import")
    {
        AssetInputRecorder::reset();
        const ve::renderer::SkinnedGltf rig = ve::renderer::loadSkinnedGltf(assetRoot / "models" / "skinned_rig.gltf");
        REQUIRE(rig.error.empty());
        const std::vector<std::string> keys = recordedKeys();
        AssetInputRecorder::reset();
        REQUIRE(keys == std::vector<std::string>{"assets/models/skinned_rig.gltf"});
    }

    SECTION("material asset")
    {
        AssetInputRecorder::reset();
        ve::assets::AssetManager manager;
        std::string error;
        const auto handle =
            manager.loadMaterialAsset(assetRoot / "materials" / "portfolio_matte_gray.material.json", &error);
        REQUIRE(error.empty());
        REQUIRE(handle.id != 0);
        const std::vector<std::string> keys = recordedKeys();
        AssetInputRecorder::reset();
        REQUIRE(keys == std::vector<std::string>{"assets/materials/portfolio_matte_gray.material.json"});
    }

    SECTION("a load that fails records nothing")
    {
        AssetInputRecorder::reset();
        ve::assets::AssetManager manager;
        std::string error;
        (void)manager.loadMaterialAsset(assetRoot / "materials" / "does_not_exist.material.json", &error);
        REQUIRE_FALSE(error.empty());
        REQUIRE(AssetInputRecorder::count() == 0);
    }
}

TEST_CASE("Asset input line is fixed width and names the count", "[asset_inputs]")
{
    AssetInputDigest digest{};
    digest.fileCount = 12;
    digest.digest = 0xabcULL;
    REQUIRE(ve::assets::formatAssetInputDigest(digest) == "Asset inputs: 12 files, digest 0x0000000000000abc");

    digest.fileCount = 1;
    digest.unreadableCount = 1;
    REQUIRE(ve::assets::formatAssetInputDigest(digest, "Asset inputs at exit") ==
            "Asset inputs at exit: 1 file, digest 0x0000000000000abc (1 no longer readable)");
}
