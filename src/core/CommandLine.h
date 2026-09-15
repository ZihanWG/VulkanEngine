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
#include <filesystem>
#include <optional>
#include <span>
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
    // Strong low sun over a broad ground plane. The one preset whose directional
    // shadows are legible rather than a faint tint, which is what makes shadow
    // work judgeable from a picture instead of from patch means.
    SunlitYard,
    // fragment-stress turned up until the GPU frame is large enough to measure
    // against. See kGpuStressLayerCount.
    GpuStress,
    // The fetched Sponza scene. Unlike every preset above it, this one is not
    // procedural: it needs -DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON at configure
    // time, so it is the only preset that can be named on a build that cannot
    // load it. That case is a hard failure, never a fallback -- see
    // Renderer::loadScenePreset.
    Sponza,

    // Not a scene. One past the last preset, so a test can walk every value and
    // check it round-trips through its name.
    //
    // It is here because the obvious guard does not work: scenePresetName falls
    // back to "default" for a preset with no table entry, so a scene added to
    // this enum and forgotten in kScenePresetNames would report itself as the
    // one scene it is not -- the "measured the wrong scene" failure --scene was
    // written to prevent. The switch over this enum does not catch it either:
    // MSVC at /W4 compiles an unhandled enum case without a warning, which was
    // measured rather than assumed.
    Count,
};

// One table, read by both directions, so a name can never parse to one preset
// and print back as another.
struct ScenePresetName {
    std::string_view name;
    ScenePreset preset;
};

// Every --scene spelling, in the order the error message lists them.
//
// Public so a test can walk the real table instead of a copy of it. The
// hand-maintained version of that test named six of the seven presets and
// reported success, which is the failure mode a coverage test is supposed to be
// immune to. Enum values with no entry here are caught separately, by the
// switch in Renderer::loadScenePreset having no default case.
[[nodiscard]] std::span<const ScenePresetName> scenePresets();

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
    // Presentation size, and with it the render target size. Settable from the
    // command line because it is the only load knob with no ceiling and no CPU
    // cost: renderScale only scales down, the scene presets are bounded by
    // early-Z and by kMaxLightsPerCluster, and every one of them leaves an
    // RTX 3080 Ti at a 1-2 ms GPU frame that measure_gpu.py's drift gate cannot
    // resolve a change against.
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

    // Turns on the validation layer's synchronization validation, which is the
    // only thing that checks the render graph's inferred barriers against what
    // the frame actually does. Off by default: it costs real CPU time inside the
    // layer, and it is a scripted-run check rather than a launch-time one.
    //
    // Instance-creation policy, so it takes effect only at startup and there is
    // no runtime equivalent.
    bool syncValidation = false;

    // Provoke a deliberate hazard at startup, report whether the layer caught
    // it, and exit without rendering.
    //
    // The negative control for the flag above. "Rendered a frame, no hazards
    // reported" is only evidence if something was watching, and a setting the
    // layer ignored looks identical to a frame with nothing wrong in it. Implies
    // --sync-validation, because a self-test with the check off would report the
    // failure it was built to detect and mean nothing by it.
    bool syncValidationSelfTest = false;

    // Fixed-timestep frame clock plus dynamic resolution pinned off, so what
    // gets rendered depends on the frame number and not on machine speed.
    // Required for any frame-to-frame image comparison.
    bool deterministic = false;

    // Report fragment shader invocations per rendered pixel once a second
    // (renderer/OverdrawQuery.h). Off by default: it is a diagnostic for the
    // depth-prepass question, it needs an optional device feature, and a query
    // bracketing the main geometry is not something a measurement run should
    // carry unasked.
    bool overdraw = false;

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
    // Dump the VSM page pool to this PNG (plus a .txt page manifest) at the
    // capture frame. Requires --capture-frame, so the pool and the shaded frame
    // it produced are from the same moment, and a VSM mode that allocates the
    // pool at all.
    std::string vsmDumpPool;

    // Reports whether this driver can bind two images into one allocation, and
    // whether they provably share bytes, then continues as normal. Kept as the
    // standing check for that -- see rhi/VulkanAliasingProbe.h.
    bool probeAliasing = false;

    // Reports, once a second, how much meshlet-level culling WOULD remove from
    // this frame if it ran: meshlets tested, rejected by cone and by frustum,
    // triangles before and after, and the indirect command count it would need.
    //
    // A measurement instrument, not a feature toggle -- nothing about the
    // rendered frame changes. It exists to answer whether a GPU meshlet cull
    // pass is worth building on the content that actually exists here, before
    // any of it is built. Off by default: the analysis is O(visible meshlets)
    // on the CPU every frame and would otherwise show up in frame prep.
    bool meshletAnalysis = false;

    // Startup scene. Default keeps whatever createScene() builds on its own.
    ScenePreset scene = ScenePreset::Default;

    // Unset leaves the persisted VSM settings untouched, which is what a normal
    // desktop run wants. Any value overrides all three stage toggles.
    std::optional<VsmMode> vsm;

    // Read runtime settings from here instead of config/runtime_settings.json.
    //
    // Exists for the same reason --scene and --vsm do, generalised: those two
    // were added because a configuration reachable only through ImGui buttons
    // cannot be scripted, and the same is true of every other persisted setting.
    // The headless job could therefore only ever run one configuration -- the
    // default -- which is exactly the blind spot the render-graph backstop
    // warns about, since a declaration/recording mismatch that only appears with
    // fog or GI on is invisible to a gate that never turns them on.
    //
    // Needs no read-only guard: saving is an ImGui button
    // (Renderer::saveRuntimeSettingsFromUi), so a headless run never writes back
    // and a sweep cannot corrupt whoever's settings it borrowed.
    //
    // Unset keeps the default path. A path that does not load is a hard failure
    // rather than a fallback to defaults: a sweep that quietly ran the default
    // configuration ten times is worse than no sweep, because it reports green.
    std::optional<std::filesystem::path> settingsPath;
};

// Parses the recognized flags and leaves defaults in place otherwise. Returns
// false when an argument is malformed, so main can fail loudly rather than
// silently running something other than what was asked for.
//
// Fills `options` rather than resetting it: callers pass a default-constructed
// value.
[[nodiscard]] bool parseLaunchOptions(int argc, char** argv, LaunchOptions& options);

} // namespace ve
