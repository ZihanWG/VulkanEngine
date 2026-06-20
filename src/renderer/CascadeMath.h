#pragma once

#include "renderer/Bounds.h"

#include <array>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace ve::renderer {

// Maximum number of cascaded-shadow-map slices the engine supports. Lives here
// (rather than nested in Renderer) so the GPU-independent cascade math and its
// unit tests can use it without pulling in the Vulkan-heavy renderer headers.
inline constexpr uint32_t kMaxShadowCascades = 4;

// One fitted cascade: the light-space view-projection used to render the shadow
// slice, its extracted frustum (for CPU culling), and the camera-space depth
// range it covers. Pure data -- no Vulkan handles -- so it can be computed and
// tested off the GPU.
struct ShadowCascade {
    glm::mat4 lightViewProjection{1.0f};
    Frustum lightFrustum{};
    float splitDepth = 0.0f;
    float nearDepth = 0.0f;
    float farDepth = 0.0f;
};

// Everything computeShadowCascades needs to fit the cascades for one frame.
// Raw (unclamped) settings are passed through so the clamping/derivation logic
// is part of the tested function rather than the caller.
struct CascadeBuildInput {
    uint32_t requestedCascadeCount = kMaxShadowCascades;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    float shadowDistance = 40.0f;
    float lambda = 0.5f;
    bool enableTexelSnapping = true;
    uint32_t shadowResolution = 2048;

    glm::vec3 cameraPosition{0.0f, 0.0f, 0.0f};
    glm::vec3 cameraTarget{0.0f, 0.0f, -1.0f};
    glm::vec3 cameraUp{0.0f, 1.0f, 0.0f};
    float cameraVerticalFovRadians = 1.0f;

    // Direction the light travels along; normalized internally.
    glm::vec3 lightDirection{0.0f, -1.0f, 0.0f};
    float aspectRatio = 1.0f;
};

struct CascadeBuildOutput {
    std::array<ShadowCascade, kMaxShadowCascades> cascades{};
    // Camera-space far split distance per cascade (x..w), used by the shader to
    // select a cascade. Unused slots repeat the last active cascade's far split.
    glm::vec4 splitDistances{0.0f};
};

// Fit one shadow cascade per active slice using practical-split-scheme depths
// (blended uniform/logarithmic by `lambda`) and a stabilized light frustum.
// GPU-independent: pure trig/matrix work, safe to unit-test.
[[nodiscard]] CascadeBuildOutput computeShadowCascades(const CascadeBuildInput& input);

} // namespace ve::renderer
