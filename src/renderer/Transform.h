#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

struct Transform {
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 rotationRadians{0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    [[nodiscard]] glm::mat4 modelMatrix() const
    {
        glm::mat4 model{1.0f};
        model = glm::translate(model, position);
        model = glm::rotate(model, rotationRadians.x, glm::vec3{1.0f, 0.0f, 0.0f});
        model = glm::rotate(model, rotationRadians.y, glm::vec3{0.0f, 1.0f, 0.0f});
        model = glm::rotate(model, rotationRadians.z, glm::vec3{0.0f, 0.0f, 1.0f});
        model = glm::scale(model, scale);
        return model;
    }
};

} // namespace ve::renderer
