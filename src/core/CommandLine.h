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
#include <optional>
#include <string>
#include <string_view>

namespace ve {

// Which scene the renderer builds at startup.
//
// The presets were reachable only through ImGui buttons, which reset on every
// launch and therefore could not be scripted -- so the one scene that actually
// loads the CPU frame path (geometry stress, 2311 objects) was unmeasurable
// without a human driving the UI. Each loader already pins its own camera, so
// selecting one here yields a fully reproducible configuration.
enum class ScenePreset {
    Default,
    Stress,
    FragmentStress,
    Occlusion,
    CornellBox,
};

// Parses a --scene value. Returns false for an unknown name rather than
// silently falling back to the default, which would measure the wrong scene.
[[nodiscard]] bool parseScenePreset(std::string_view name, ScenePreset& preset);

// The spelling accepted on the command line, for error messages and tests.
[[nodiscard]] std::string_view scenePresetName(ScenePreset preset);

// Which VSM stages a run turns on, overriding whatever config/runtime_settings.json
// persisted.
//
// The stages are nested rather than independent -- rendering pages without
// marking them, or sampling a pool nothing filled, are not configurations -- so
// one cumulative mode names all three booleans and cannot spell an impossible
// combination. Off is a real value, not the absence of the flag: `--vsm off`
// pins the cascade baseline even when the persisted file asks for VSM, which is
// exactly what the A/B control needs.
//
// enableMarking gates a 64 MiB page pool allocated during renderer construction,
// so it can only be decided before the renderer exists. That is why this arrives
// on the command line at all: the ImGui checkbox cannot reach it, and editing
// the git-ignored settings file to run an A/B overwrites whatever the person at
// the keyboard had configured.
enum class VsmMode {
    Off,
    Mark,
    Render,
    Shadows,
};

// Parses a --vsm value. Returns false for an unknown name rather than silently
// picking a stage, which would report on a configuration nobody asked for.
[[nodiscard]] bool parseVsmMode(std::string_view name, VsmMode& mode);

// The spelling accepted on the command line, for error messages and tests.
[[nodiscard]] std::string_view vsmModeName(VsmMode mode);

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

    // Take the capture AFTER the ImGui overlay instead of before it.
    //
    // Off by default because a regression capture wants the rendered frame and
    // not a debug panel drawn over a third of it. On, it is the only way to see
    // the debug UI from a scripted run at all: the panel exists solely on screen,
    // so anything it reports and the log does not -- an amber capacity warning,
    // a colour, a layout -- has no other evidence path.
    bool captureIncludeUi = false;

    // Reports whether this driver can bind two images into one allocation, and
    // whether they provably share bytes, then continues as normal. Kept as the
    // standing check for that -- see rhi/VulkanAliasingProbe.h.
    bool probeAliasing = false;

    // Startup scene. Default keeps whatever createScene() builds on its own.
    ScenePreset scene = ScenePreset::Default;

    // Unset leaves the persisted VSM settings untouched, which is what a normal
    // desktop run wants. Any value overrides all three stage toggles.
    std::optional<VsmMode> vsm;
};

// Parses the recognized flags and leaves defaults in place otherwise. Returns
// false when an argument is malformed, so main can fail loudly rather than
// silently running something other than what was asked for.
//
// Fills `options` rather than resetting it: callers pass a default-constructed
// value.
[[nodiscard]] bool parseLaunchOptions(int argc, char** argv, LaunchOptions& options);

} // namespace ve
