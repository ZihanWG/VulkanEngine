#pragma once

#include "core/CommandLine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace ve {

class Renderer;
class Window;

class Application final {
public:
    // Defined in core/CommandLine.h so that parsing -- and the tests covering
    // it -- stay clear of this translation unit, which pulls in SDL3.
    using Config = LaunchOptions;

    // Distinct from the -1 an exception returns, so CI can tell "the renderer
    // produced validation errors" apart from "the renderer crashed".
    static constexpr int kValidationFailureExitCode = 2;

    // A capture was requested but never written: the run must not report success.
    static constexpr int kCaptureFailureExitCode = 3;

    // --scene named a preset this build cannot load, which today means
    // ScenePreset::Sponza without -DVULKAN_ENGINE_FETCH_SAMPLE_SCENE=ON. Distinct
    // from the -1 a crash returns so a measurement harness can tell "this build
    // has no such scene" apart from "the renderer fell over", and distinct from
    // success because a run that quietly rendered a different scene than the one
    // named would report a clean number for the wrong question.
    static constexpr int kSceneFailureExitCode = 4;

    // --sync-validation-selftest ran and the layer did NOT report the hazard it
    // was handed, so synchronization validation is not actually watching. Its
    // own code because it is the opposite kind of failure from the ones above:
    // nothing is wrong with the frame, the check itself is dead, and a run that
    // treated that as success would be reporting on nothing.
    static constexpr int kSyncValidationSelfTestFailureExitCode = 5;

    // How far past the capture frame to keep drawing before declaring the
    // readback lost. Comfortably above the in-flight frame count.
    static constexpr uint64_t kCaptureReadbackGraceFrames = 16;

    Application();
    explicit Application(Config config);
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    int run();

private:
    void initialize();
    void mainLoop();
    void shutdown();

    // Called after shutdown so teardown-time validation messages are included.
    [[nodiscard]] int reportValidationTally() const;

    Config config_;
    std::unique_ptr<Window> window_;
    std::unique_ptr<Renderer> renderer_;

    // Measured in initialize(), consumed in mainLoop() once the first frame has
    // also been timed. Meaningless unless config_.assetLoadStats is set.
    double rendererInitMs_ = 0.0;

    // Latched in mainLoop, because reportValidationTally runs after the renderer
    // has already been destroyed.
    bool captureCompleted_ = false;

    // Set by initialize() when --sync-validation-selftest ran. Its presence also
    // means the run ends there: the self-test deliberately produces validation
    // errors, so the frames and the tally that would follow could only describe
    // the probe rather than the renderer.
    std::optional<int> syncValidationSelfTestExitCode_;
};

} // namespace ve