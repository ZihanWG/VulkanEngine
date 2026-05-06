#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

struct Transform {
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 rotationRadians{0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};
    glm::mat4 matrixOverride{1.0f};
    bool useMatrixOverride = false;

    [[nodiscard]] static Transform fromMatrix(const glm::mat4& matrix)
    {
        Transform transform{};
        transform.matrixOverride = matrix;
        transform.useMatrixOverride = true;
        return transform;
    }

    [[nodiscard]] glm::mat4 modelMatrix() const
    {
        if (useMatrixOverride) {
            return matrixOverride;
        }

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
