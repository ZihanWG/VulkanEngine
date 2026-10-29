#pragma once

// Which frames a scripted capture reads back, and where each one is written.
//
// One request covers both scripted captures: a single frame to one named file
// (--capture-frame), and a run of consecutive frames to numbered files in a
// directory (--capture-sequence). A single capture is a sequence of one, so the
// renderer arms, counts and completes both the same way. GPU-free, so the frame
// range and the file naming are unit-testable without a device.

#include <cstdint>
#include <filesystem>
#include <string>

namespace ve::renderer {

// Four digits, so a sequence of at most this many frames sorts by name.
inline constexpr uint32_t kMaxFrameSequenceLength = 9999;

// "frame_0000.png" for index 0. Zero-padded so a directory listing, a glob, and
// tools/make_gif.py all see the frames in capture order.
[[nodiscard]] std::string frameSequenceFileName(uint32_t index);

struct FrameCaptureRequest {
    // 1-based, matching the frame clock. 0 means no capture was asked for.
    uint64_t firstFrame = 0;
    uint32_t frameCount = 0;
    // A file for a single capture, a directory for a sequence.
    std::filesystem::path output;
    bool sequence = false;

    [[nodiscard]] bool active() const
    {
        return firstFrame != 0 && frameCount != 0;
    }

    [[nodiscard]] uint64_t lastFrame() const
    {
        return active() ? firstFrame + frameCount - 1 : 0;
    }

    [[nodiscard]] bool covers(uint64_t frame) const
    {
        return active() && frame >= firstFrame && frame <= lastFrame();
    }

    // Only meaningful for a frame this request covers.
    [[nodiscard]] std::filesystem::path outputPathFor(uint64_t frame) const;
};

} // namespace ve::renderer
