#include "renderer/EditorCamera.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace ve {

namespace {

constexpr glm::vec3 kWorldUp{0.0f, 1.0f, 0.0f};
// Keep pitch a hair under +/-90 degrees to avoid the forward vector aligning with
// world up (which would make the right/up basis degenerate).
constexpr float kMaxPitch = 1.55334f; // ~89 degrees in radians

glm::vec3 directionFromAngles(float yaw, float pitch)
{
    const float cosPitch = std::cos(pitch);
    return glm::vec3{
        std::sin(yaw) * cosPitch,
        std::sin(pitch),
        -std::cos(yaw) * cosPitch,
    };
}

} // namespace

glm::vec3 EditorCamera::forward() const
{
    return directionFromAngles(yaw_, pitch_);
}

void EditorCamera::syncFromCamera(const renderer::Camera& camera)
{
    const glm::vec3 delta = camera.target - camera.position;
    if (glm::length(delta) <= 1e-6f) {
        return;
    }

    const glm::vec3 direction = glm::normalize(delta);
    pitch_ = std::clamp(std::asin(std::clamp(direction.y, -1.0f, 1.0f)), -kMaxPitch, kMaxPitch);
    yaw_ = std::atan2(direction.x, -direction.z);
}

void EditorCamera::updateFly(renderer::Camera& camera, const EditorCameraInput& input, float deltaSeconds)
{
    yaw_ += input.lookDelta.x;
    pitch_ = std::clamp(pitch_ + input.lookDelta.y, -kMaxPitch, kMaxPitch);

    const glm::vec3 forwardDir = directionFromAngles(yaw_, pitch_);
    const glm::vec3 rightDir = glm::normalize(glm::cross(forwardDir, kWorldUp));

    const float distance = std::max(moveSpeed_, 0.0f) * std::max(input.speedScale, 0.0f) * std::max(deltaSeconds, 0.0f);
    glm::vec3 position = camera.position;
    position += rightDir * input.moveAxis.x * distance;
    position += kWorldUp * input.moveAxis.y * distance;
    position += forwardDir * input.moveAxis.z * distance;

    camera.position = position;
    camera.target = position + forwardDir;
    camera.up = kWorldUp;
}

void EditorCamera::orbit(renderer::Camera& camera, float yawDelta, float pitchDelta)
{
    const glm::vec3 offset = camera.position - camera.target;
    const float radius = glm::length(offset);
    if (radius <= 1e-6f) {
        return;
    }

    syncFromCamera(camera); // refresh yaw/pitch from the current look direction
    yaw_ += yawDelta;
    pitch_ = std::clamp(pitch_ + pitchDelta, -kMaxPitch, kMaxPitch);

    const glm::vec3 forwardDir = directionFromAngles(yaw_, pitch_);
    camera.position = camera.target - forwardDir * radius;
    camera.up = kWorldUp;
}

void EditorCamera::pan(renderer::Camera& camera, float rightDelta, float upDelta)
{
    const glm::vec3 delta = camera.target - camera.position;
    if (glm::length(delta) <= 1e-6f) {
        return;
    }

    const glm::vec3 forwardDir = glm::normalize(delta);
    const glm::vec3 rightDir = glm::normalize(glm::cross(forwardDir, kWorldUp));
    const glm::vec3 upDir = glm::normalize(glm::cross(rightDir, forwardDir));

    const glm::vec3 shift = rightDir * rightDelta + upDir * upDelta;
    camera.position += shift;
    camera.target += shift;
}

void EditorCamera::dolly(renderer::Camera& camera, float amount)
{
    const glm::vec3 delta = camera.target - camera.position;
    const float distance = glm::length(delta);
    if (distance <= 1e-6f) {
        return;
    }

    const glm::vec3 forwardDir = delta / distance;
    // Don't allow the camera to cross past (or land exactly on) the target.
    const float move = std::min(amount, distance - 0.05f);
    camera.position += forwardDir * move;
}

} // namespace ve
