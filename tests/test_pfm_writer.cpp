// The PFM writer feeds tools/dev/compare_shadow_visibility.py, which reads the
// format the way the specification lays it out. So these parse the bytes back by
// the specification too -- header, byte order, bottom-to-top rows -- rather than
// trusting the writer's own idea of them.

#include "core/PfmWriter.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::filesystem::path scratchPath(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

TEST_CASE("A PFM holds linear floats, bottom row first, little-endian", "[pfm]")
{
    const std::filesystem::path path = scratchPath("ve_pfm_writer_test.pfm");
    // Top row 0, 0.25, 0.5; bottom row 0.75, 1, NaN -- the NaN is how the shadow
    // dump marks a pixel nothing traced, and it must survive the trip.
    const std::vector<float> rows{0.0f, 0.25f, 0.5f, 0.75f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
    REQUIRE(ve::writePfmGray(path, 3, 2, rows));

    std::ifstream file(path, std::ios::binary);
    REQUIRE(file);
    std::string magic;
    uint32_t width = 0;
    uint32_t height = 0;
    double scale = 0.0;
    file >> magic >> width >> height >> scale;
    file.get(); // the single whitespace byte before the data
    CHECK(magic == "Pf");
    CHECK(width == 3);
    CHECK(height == 2);
    // Negative scale is little-endian.
    CHECK(scale < 0.0);

    std::vector<float> stored(6);
    file.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(stored.size() * sizeof(float)));
    REQUIRE(file.gcount() == static_cast<std::streamsize>(stored.size() * sizeof(float)));
    // The bottom row comes first in the file.
    CHECK(stored[0] == 0.75f);
    CHECK(stored[1] == 1.0f);
    CHECK(std::isnan(stored[2]));
    CHECK(stored[3] == 0.0f);
    CHECK(stored[4] == 0.25f);
    CHECK(stored[5] == 0.5f);
    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("A PFM write refuses values that do not fill the image", "[pfm]")
{
    const std::filesystem::path path = scratchPath("ve_pfm_writer_short.pfm");
    const std::vector<float> tooFew{0.0f, 1.0f, 0.5f};
    CHECK_FALSE(ve::writePfmGray(path, 2, 2, tooFew));
    CHECK_FALSE(ve::writePfmGray(path, 0, 2, {}));
    std::filesystem::remove(path);
}
