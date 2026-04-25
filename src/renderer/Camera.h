#pragma once

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

struct Camera {
    glm::vec3 position{0.0f, 0.0f, 3.0f};
    glm::vec3 target{0.0f, 0.0f, 0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float verticalFovRadians = glm::radians(60.0f);
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    [[nodiscard]] glm::mat4 viewMatrix() const
    {
        return glm::lookAt(position, target, up);
    }

    [[nodiscard]] glm::mat4 projectionMatrix(float aspectRatio) const
    {
        glm::mat4 projection = glm::perspective(verticalFovRadians, aspectRatio, nearPlane, farPlane);

        // GLM is configured for Vulkan's 0..1 depth range. Flipping Y converts
        // from OpenGL-style clip coordinates to Vulkan's framebuffer orientation.
        projection[1][1] *= -1.0f;
        return projection;
    }
};

} // namespace ve::renderer
