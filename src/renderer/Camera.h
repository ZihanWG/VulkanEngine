#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace ve::renderer {

struct Camera {
    glm::vec3 position{0.0f, 0.0f, 3.0f};
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
};

} // namespace ve::renderer
