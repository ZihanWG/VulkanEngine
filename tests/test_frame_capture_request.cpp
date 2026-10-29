#include "renderer/FrameCaptureRequest.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

using ve::renderer::FrameCaptureRequest;
using ve::renderer::frameSequenceFileName;

TEST_CASE("A default capture request covers no frame", "[capture]")
{
    const FrameCaptureRequest request{};
    REQUIRE_FALSE(request.active());
    REQUIRE(request.lastFrame() == 0);
    REQUIRE_FALSE(request.covers(0));
    REQUIRE_FALSE(request.covers(1));
}

TEST_CASE("A single capture covers exactly its frame and writes its named file", "[capture]")
{
    const FrameCaptureRequest request{30, 1, "out/frame30.png", false};
    REQUIRE(request.active());
    REQUIRE(request.lastFrame() == 30);
    REQUIRE_FALSE(request.covers(29));
    REQUIRE(request.covers(30));
    REQUIRE_FALSE(request.covers(31));
    REQUIRE(request.outputPathFor(30) == std::filesystem::path("out/frame30.png"));
}

TEST_CASE("A sequence covers its whole range and nothing either side", "[capture]")
{
    const FrameCaptureRequest request{10, 5, "frames", true};
    REQUIRE(request.lastFrame() == 14);
    REQUIRE_FALSE(request.covers(9));
    for (uint64_t frame = 10; frame <= 14; ++frame) {
        REQUIRE(request.covers(frame));
    }
    REQUIRE_FALSE(request.covers(15));
}

TEST_CASE("Sequence files are numbered from zero in capture order", "[capture]")
{
    const FrameCaptureRequest request{10, 5, "frames", true};
    REQUIRE(request.outputPathFor(10) == std::filesystem::path("frames") / "frame_0000.png");
    REQUIRE(request.outputPathFor(14) == std::filesystem::path("frames") / "frame_0004.png");
}

TEST_CASE("Sequence file names are zero-padded so they sort by name", "[capture]")
{
    REQUIRE(frameSequenceFileName(0) == "frame_0000.png");
    REQUIRE(frameSequenceFileName(7) == "frame_0007.png");
    REQUIRE(frameSequenceFileName(123) == "frame_0123.png");
    REQUIRE(frameSequenceFileName(ve::renderer::kMaxFrameSequenceLength - 1) == "frame_9998.png");
    REQUIRE(frameSequenceFileName(9) < frameSequenceFileName(10));
}
