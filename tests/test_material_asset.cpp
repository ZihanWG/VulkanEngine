#include "assets/AssetManager.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>

using ve::assets::AssetManager;
using ve::assets::MaterialAsset;
using ve::assets::MaterialAssetHandle;

namespace {

std::filesystem::path makeTempMaterialPath()
{
    std::random_device device;
    const std::string name = "ve_material_asset_" + std::to_string(device()) + ".json";
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

TEST_CASE("Material asset save -> load round-trips the emissive factor", "[material]")
{
    const std::filesystem::path path = makeTempMaterialPath();

    MaterialAsset source;
    source.name = "Emissive Test";
    source.baseColorFactor = {0.2f, 0.4f, 0.6f, 1.0f};
    source.emissiveFactor = {1.5f, 0.75f, 0.1f}; // includes an HDR (>1) component
    source.metallicFactor = 0.3f;
    source.roughnessFactor = 0.7f;

    AssetManager manager;
    std::string saveError;
    REQUIRE(manager.saveMaterialAsset(path, source, &saveError));
    REQUIRE(saveError.empty());

    std::string loadError;
    const MaterialAssetHandle handle = manager.loadMaterialAsset(path, &loadError);
    REQUIRE(static_cast<bool>(handle));
    REQUIRE(loadError.empty());

    const MaterialAsset* loaded = manager.materialAsset(handle);
    REQUIRE(loaded != nullptr);
    CHECK(loaded->emissiveFactor.x == Catch::Approx(1.5f));
    CHECK(loaded->emissiveFactor.y == Catch::Approx(0.75f));
    CHECK(loaded->emissiveFactor.z == Catch::Approx(0.1f));
    CHECK(loaded->baseColorFactor.r == Catch::Approx(0.2f));
    CHECK(loaded->metallicFactor == Catch::Approx(0.3f));
    CHECK(loaded->roughnessFactor == Catch::Approx(0.7f));

    std::error_code cleanup;
    std::filesystem::remove(path, cleanup);
}

TEST_CASE("Material asset without an emissive factor defaults to zero", "[material]")
{
    const std::filesystem::path path = makeTempMaterialPath();
    {
        std::ofstream output(path);
        output << R"({"name":"Legacy","baseColorFactor":[1.0,1.0,1.0,1.0],)"
               << R"("metallicFactor":0.0,"roughnessFactor":0.5})";
    }

    AssetManager manager;
    std::string loadError;
    const MaterialAssetHandle handle = manager.loadMaterialAsset(path, &loadError);
    REQUIRE(static_cast<bool>(handle));
    REQUIRE(loadError.empty());

    const MaterialAsset* loaded = manager.materialAsset(handle);
    REQUIRE(loaded != nullptr);
    CHECK(loaded->emissiveFactor.x == Catch::Approx(0.0f));
    CHECK(loaded->emissiveFactor.y == Catch::Approx(0.0f));
    CHECK(loaded->emissiveFactor.z == Catch::Approx(0.0f));

    std::error_code cleanup;
    std::filesystem::remove(path, cleanup);
}
