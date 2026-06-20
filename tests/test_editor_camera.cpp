#include "renderer/EditorCamera.h"
#include "renderer/Camera.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/geometric.hpp>

using ve::EditorCamera;
using ve::EditorCameraInput;
using ve::renderer::Camera;

TEST_CASE("syncFromCamera + zero input keeps the look direction stable", "[editorcamera]")
{
    Camera camera;
    camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f); // looking down -Z

    EditorCamera controller;
    controller.syncFromCamera(camera);

    EditorCameraInput input; // all zero
    controller.updateFly(camera, input, 0.016f);

    // Position unchanged; the forward direction stays -Z.
    CHECK(camera.position.x == Catch::Approx(0.0f));
    CHECK(camera.position.z == Catch::Approx(5.0f));
    const glm::vec3 forward = glm::normalize(camera.target - camera.position);
    CHECK(forward.z == Catch::Approx(-1.0f).margin(1e-4));
}

TEST_CASE("Moving forward advances the camera along its view direction", "[editorcamera]")
{
    Camera camera;
    camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f);

    EditorCamera controller;
    controller.syncFromCamera(camera);
    controller.setMoveSpeed(10.0f);

    EditorCameraInput input;
    input.moveAxis.z = 1.0f; // forward
    controller.updateFly(camera, input, 0.5f); // 10 * 0.5 = 5 units forward (-Z)

    CHECK(camera.position.z == Catch::Approx(0.0f).margin(1e-4)); // 5 - 5
    CHECK(camera.position.x == Catch::Approx(0.0f).margin(1e-4));
}

TEST_CASE("Strafing right moves along the camera right axis", "[editorcamera]")
{
    Camera camera;
    camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f); // forward = -Z, so right = +X

    EditorCamera controller;
    controller.syncFromCamera(camera);
    controller.setMoveSpeed(10.0f);

    EditorCameraInput input;
    input.moveAxis.x = 1.0f; // right
    controller.updateFly(camera, input, 0.5f);

    CHECK(camera.position.x == Catch::Approx(5.0f).margin(1e-4));
    CHECK(camera.position.z == Catch::Approx(5.0f).margin(1e-4));
}

TEST_CASE("Look delta rotates the view and pitch is clamped", "[editorcamera]")
{
    Camera camera;
    camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f);

    EditorCamera controller;
    controller.syncFromCamera(camera);

    EditorCameraInput input;
    input.lookDelta = glm::vec2(0.0f, 100.0f); // absurd upward pitch
    controller.updateFly(camera, input, 0.0f);

    // Pitch clamp keeps |pitch| < 90deg, so forward.y must stay below 1.
    CHECK(controller.pitch() < 1.56f);
    const glm::vec3 forward = glm::normalize(camera.target - camera.position);
    CHECK(forward.y < 1.0f);
    CHECK(forward.y > 0.9f); // still pitched strongly upward
}

TEST_CASE("Orbit keeps the target fixed and preserves radius", "[editorcamera]")
{
    Camera camera;
    camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f);

    EditorCamera controller;
    const float radiusBefore = glm::length(camera.position - camera.target);

    controller.orbit(camera, 1.0f, 0.0f); // yaw a radian around the target

    CHECK(camera.target == glm::vec3(0.0f)); // target unchanged
    const float radiusAfter = glm::length(camera.position - camera.target);
    CHECK(radiusAfter == Catch::Approx(radiusBefore));
    CHECK(camera.position.z < 5.0f); // it actually moved around
}

TEST_CASE("Dolly moves toward the target without crossing it", "[editorcamera]")
{
    Camera camera;
    camera.position = glm::vec3(0.0f, 0.0f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.0f, 0.0f);

    EditorCamera controller;
    controller.dolly(camera, 2.0f);
    CHECK(camera.position.z == Catch::Approx(3.0f).margin(1e-4));

    // A huge dolly is clamped so the camera never passes the target.
    controller.dolly(camera, 1000.0f);
    CHECK(camera.position.z > 0.0f);
}
