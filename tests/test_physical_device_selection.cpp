#include "rhi/PhysicalDeviceSelection.h"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using ve::rhi::kUnsuitableDeviceScore;
using ve::rhi::PhysicalDeviceChoice;
using ve::rhi::selectPhysicalDevice;

namespace {

PhysicalDeviceChoice select(const std::vector<int>& scores)
{
    return selectPhysicalDevice(std::span<const int>(scores));
}

} // namespace

TEST_CASE("The highest-scoring device is chosen", "[physical-device]")
{
    // A discrete GPU (+1000) beside an integrated one with a larger image limit.
    const PhysicalDeviceChoice choice = select({16384, 1000 + 32768, 8192});
    CHECK(choice.index == 1);
    CHECK(choice.tiedWith.empty());
}

TEST_CASE("One GPU enumerated by two drivers is a tie the first one wins", "[physical-device]")
{
    // An Apple M3 under MoltenVK and under KosmicKrisp: same type, same limit.
    const PhysicalDeviceChoice choice = select({16384, 16384});
    CHECK(choice.index == 0);
    REQUIRE(choice.tiedWith.size() == 1);
    CHECK(choice.tiedWith[0] == 1);
}

TEST_CASE("Unsuitable devices neither win nor tie", "[physical-device]")
{
    const PhysicalDeviceChoice choice = select({kUnsuitableDeviceScore, 16384, kUnsuitableDeviceScore, 16384});
    CHECK(choice.index == 1);
    REQUIRE(choice.tiedWith.size() == 1);
    CHECK(choice.tiedWith[0] == 3);
}

TEST_CASE("A lower score is not reported as a tie", "[physical-device]")
{
    const PhysicalDeviceChoice choice = select({8192, 16384, 16384, 4096});
    CHECK(choice.index == 1);
    REQUIRE(choice.tiedWith.size() == 1);
    CHECK(choice.tiedWith[0] == 2);
}

TEST_CASE("No suitable device yields no choice", "[physical-device]")
{
    CHECK_FALSE(select({}).available());
    CHECK_FALSE(select({kUnsuitableDeviceScore, kUnsuitableDeviceScore}).available());
}
