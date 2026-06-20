#pragma once

#include "renderer/Camera.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace ve {

// Per-frame editor camera input, already gathered and normalized by the platform
// layer (SDL). Kept free of any windowing types so the controller stays testable.
struct EditorCameraInput {
    // Movement request in camera-local axes, each component in [-1, 1]:
    // x = right, y = world-up, z = forward.
    glm::vec3 moveAxis{0.0f};
    // Look delta in radians accumulated this frame (x = yaw, y = pitch).
    glm::vec2 lookDelta{0.0f};
    float speedScale = 1.0f; // extra multiplier on move speed (sprint, etc.)
};

// Fly / orbit editor camera controller. Pure math: it reads input + a time step
// and writes position/target back into a renderer::Camera. No SDL or Vulkan
// dependencies, so it lives in the GPU-independent core library and is unit-tested.
class EditorCamera {
public:
    // Initialize yaw/pitch from a camera's current look direction. Call this when
    // the controller takes over (e.g. when the user presses the look button).
    void syncFromCamera(const renderer::Camera& camera);

    // Advance a free-fly camera by deltaSeconds and write the result into camera.
    void updateFly(renderer::Camera& camera, const EditorCameraInput& input, float deltaSeconds);

    // Orbit the camera position around its (fixed) target. Deltas in radians.
    void orbit(renderer::Camera& camera, float yawDelta, float pitchDelta);

    // Pan camera and target together along the view's right/up axes.
    void pan(renderer::Camera& camera, float rightDelta, float upDelta);

    // Move the camera toward (positive) or away from its target.
    void dolly(renderer::Camera& camera, float amount);

    [[nodiscard]] float yaw() const { return yaw_; }
    [[nodiscard]] float pitch() const { return pitch_; }
    void setMoveSpeed(float speed) { moveSpeed_ = speed; }
    [[nodiscard]] float moveSpeed() const { return moveSpeed_; }

    // Forward direction implied by the current yaw/pitch (normalized).
    [[nodiscard]] glm::vec3 forward() const;

private:
    float yaw_ = 0.0f;        // radians around world up (+Y)
    float pitch_ = 0.0f;      // radians, clamped away from straight up/down
    float moveSpeed_ = 5.0f;  // units per second
};

} // namespace ve
