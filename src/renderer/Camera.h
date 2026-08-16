#pragma once

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <glm/mat4x4.hpp>
#include "renderer/Bounds.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cmath>

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

    [[nodiscard]] glm::mat4 viewProjectionMatrix(float aspectRatio) const
    {
        return projectionMatrix(aspectRatio) * viewMatrix();
    }
};

// Places a camera so an axis-aligned volume fits in frame.
//
// Exists because a fetched scene has no idea what the shipped portfolio camera
// preset was framed for: that preset is positioned for the sphere showcase, and
// pointing it at Sponza puts the near plane inside a column. Framing from the
// scene's own bounds works for whatever asset happens to be fetched.
//
// `direction` is the way the camera looks (it is normalised here, so it need not
// be). `fill` is how much of the smaller field-of-view axis the bounding sphere
// should occupy: 1.0 touches the frame edges, lower values leave margin.
//
// Pure, so the framing is unit tested rather than eyeballed once and trusted.
[[nodiscard]] inline Camera framedCamera(const Aabb& bounds,
                                         float aspectRatio,
                                         const glm::vec3& direction,
                                         float verticalFovRadians = glm::radians(60.0f),
                                         float fill = 0.9f)
{
    Camera camera{};
    camera.verticalFovRadians = verticalFovRadians;
    if (!bounds.valid()) {
        return camera;
    }

    const glm::vec3 center = (bounds.min + bounds.max) * 0.5f;
    // A bounding sphere rather than the box: it makes the fit independent of
    // which way the camera ends up looking at the box.
    const float radius = glm::length(bounds.max - bounds.min) * 0.5f;
    if (!(radius > 0.0f)) {
        camera.position = center + glm::vec3(0.0f, 0.0f, 1.0f);
        camera.target = center;
        return camera;
    }

    const float safeFill = glm::clamp(fill, 0.05f, 1.0f);
    const float halfVertical = verticalFovRadians * 0.5f;
    // The horizontal half-angle for this aspect. Whichever axis is tighter is the
    // one that decides the distance, or a wide scene overflows the sides of a
    // narrow window.
    const float halfHorizontal =
        std::atan(std::tan(halfVertical) * (aspectRatio > 0.0f ? aspectRatio : 1.0f));
    const float halfAngle = std::min(halfVertical, halfHorizontal);
    const float distance = radius / (std::sin(halfAngle) * safeFill);

    const float directionLength = glm::length(direction);
    const glm::vec3 forward =
        directionLength > 0.0f ? direction / directionLength : glm::vec3(0.0f, 0.0f, -1.0f);

    camera.position = center - forward * distance;
    camera.target = center;
    // Sized from the framing rather than left at the preset's values, which would
    // clip a scene orders of magnitude larger or smaller than the sphere showcase.
    camera.nearPlane = std::max(distance * 0.005f, 0.01f);
    camera.farPlane = distance + radius * 4.0f;
    return camera;
}

} // namespace ve::renderer
