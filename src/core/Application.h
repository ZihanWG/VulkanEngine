#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ve {

class Renderer;
class Window;

class Application final {
public:
    struct Config {
        std::string title = "VulkanEngine";
        int width = 1280;
        int height = 720;

        // Asset-load baseline instrumentation. Recording is off unless asked for,
        // so a normal run pays nothing but the branch. exitAfterFrames = 0 keeps
        // the usual "run until the window closes" behavior; any positive value
        // makes a measurement run scriptable and repeatable.
        bool assetLoadStats = false;
        uint32_t exitAfterFrames = 0;

        // Makes validation-layer output a failure rather than a log line the
        // reader has to notice. Only meaningful in a build that enables the
        // validation layer at all (NDEBUG undefined).
        bool failOnValidationError = false;

        // Fixed-timestep frame clock plus dynamic resolution pinned off, so what
        // gets rendered depends on the frame number and not on machine speed.
        // Required for any frame-to-frame image comparison.
        bool deterministic = false;

        // Capture the swapchain image of this frame (1-based) to captureOutput.
        // The loop keeps drawing past it until the readback lands, then exits.
        uint64_t captureFrame = 0;
        std::string captureOutput;
    };

    // A capture was requested but never written: the run must not report success.
    static constexpr int kCaptureFailureExitCode = 3;

    // How far past the capture frame to keep drawing before declaring the
    // readback lost. Comfortably above the in-flight frame count.
    static constexpr uint64_t kCaptureReadbackGraceFrames = 16;

    // Distinct from the -1 an exception returns, so CI can tell "the renderer
    // produced validation errors" apart from "the renderer crashed".
    static constexpr int kValidationFailureExitCode = 2;

    // Parses the recognized flags and leaves config defaults in place otherwise.
    // Returns false when an argument is malformed, so main can fail loudly rather
    // than silently measuring the wrong thing.
    [[nodiscard]] static bool parseArguments(int argc, char** argv, Config& config);

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
};

} // namespace ve