#pragma once

// Command-line parsing for the engine executable.
//
// Deliberately its own translation unit in VulkanEngineCore rather than a member
// of Application. Parsing is GPU-free policy, so the layering rule puts it here;
// and just as importantly, the unit tests that cover it must not drag in
// Application.cpp, which references Window and therefore imports SDL3. On
// Windows that import makes the test executable fail to start with
// STATUS_DLL_NOT_FOUND (0xc0000135) during Catch2's build-time test discovery,
// because SDL3.dll does not sit next to the test binary.

#include <cstdint>
#include <string>

namespace ve {

struct LaunchOptions {
    std::string title = "VulkanEngine";
    int width = 1280;
    int height = 720;

    // Asset-load baseline instrumentation. Recording is off unless asked for, so
    // a normal run pays nothing but the branch.
    bool assetLoadStats = false;

    // 0 keeps the usual "run until the window closes" behavior; any positive
    // value makes a measurement run scriptable and repeatable.
    uint32_t exitAfterFrames = 0;

    // Makes validation-layer output a failure rather than a log line the reader
    // has to notice. Only meaningful in a build that enables the validation
    // layer at all.
    bool failOnValidationError = false;

    // Fixed-timestep frame clock plus dynamic resolution pinned off, so what
    // gets rendered depends on the frame number and not on machine speed.
    // Required for any frame-to-frame image comparison.
    bool deterministic = false;

    // Capture the swapchain image of this frame (1-based) to captureOutput. The
    // loop keeps drawing past it until the readback lands, then exits.
    uint64_t captureFrame = 0;
    std::string captureOutput;
};

// Parses the recognized flags and leaves defaults in place otherwise. Returns
// false when an argument is malformed, so main can fail loudly rather than
// silently running something other than what was asked for.
//
// Fills `options` rather than resetting it: callers pass a default-constructed
// value.
[[nodiscard]] bool parseLaunchOptions(int argc, char** argv, LaunchOptions& options);

} // namespace ve
