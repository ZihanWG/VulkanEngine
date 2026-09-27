#include "renderer/Camera.h"
#include "renderer/TaaDisocclusion.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <glm/glm.hpp>

using ve::renderer::Camera;
using ve::renderer::kTaaHistoryDepthOffset;
using ve::renderer::kTaaHistorySkyDepth;
using ve::renderer::taaCurrentViewDepth;
using ve::renderer::taaDecodeHistoryDepth;
using ve::renderer::taaEncodeHistoryDepth;
using ve::renderer::taaHistoryDisoccluded;
using ve::renderer::taaPreviousDepthRows;
using ve::renderer::taaPreviousViewDepth;

namespace {

constexpr float kAspect = 16.0f / 9.0f;

Camera cameraAt(const glm::vec3& position)
{
    Camera camera;
    camera.position = position;
    camera.target = position + glm::vec3(0.0f, 0.0f, -1.0f);
    return camera;
}

// Where a world point lands in a frame rendered through `viewProjection`: the
// (ndc, depth) the resolve reads back out of the depth buffer.
struct Projected {
    glm::vec2 ndc;
    float depth;
};

Projected project(const glm::mat4& viewProjection, const glm::vec3& world)
{
    const glm::vec4 clip = viewProjection * glm::vec4(world, 1.0f);
    return {glm::vec2(clip) / clip.w, clip.z / clip.w};
}

// A projection jittered the way Renderer::updateFrameData does it: an NDC offset
// in the third column, so clip x and y shift in proportion to w.
glm::mat4 jittered(glm::mat4 projection, glm::vec2 jitterNdc)
{
    projection[2][0] += jitterNdc.x;
    projection[2][1] += jitterNdc.y;
    return projection;
}

} // namespace

TEST_CASE("A still camera expects the depth the surface has now", "[taa]")
{
    const Camera camera = cameraAt({0.0f, 1.0f, 5.0f});
    const glm::mat4 viewProjection = camera.viewProjectionMatrix(kAspect);
    const auto rows = taaPreviousDepthRows(viewProjection, viewProjection);

    // A point 7 units in front of the camera, off-centre so x and y matter.
    const glm::vec3 world(1.5f, 0.25f, -2.0f);
    const Projected p = project(viewProjection, world);

    CHECK(taaCurrentViewDepth(rows, p.ndc, p.depth) == Catch::Approx(7.0f).epsilon(1e-4));
    CHECK(taaPreviousViewDepth(rows, p.ndc, p.depth) == Catch::Approx(7.0f).epsilon(1e-4));
}

TEST_CASE("A dolly moves the expected depth by exactly the distance travelled", "[taa]")
{
    // Last frame the camera stood 0.5 further back, so a static surface was 0.5
    // further away then than it is now.
    const Camera previous = cameraAt({0.0f, 1.0f, 5.5f});
    const Camera current = cameraAt({0.0f, 1.0f, 5.0f});
    const glm::mat4 currentViewProjection = current.viewProjectionMatrix(kAspect);
    const auto rows = taaPreviousDepthRows(currentViewProjection, previous.viewProjectionMatrix(kAspect));

    const glm::vec3 world(-0.75f, 1.5f, -3.0f);
    const Projected p = project(currentViewProjection, world);

    CHECK(taaCurrentViewDepth(rows, p.ndc, p.depth) == Catch::Approx(8.0f).epsilon(1e-4));
    CHECK(taaPreviousViewDepth(rows, p.ndc, p.depth) == Catch::Approx(8.5f).epsilon(1e-4));
}

TEST_CASE("The expected depth matches a direct reprojection under rotation and jitter", "[taa]")
{
    Camera previous = cameraAt({2.0f, 1.0f, 6.0f});
    previous.target = {0.0f, 0.5f, 0.0f};
    Camera current = cameraAt({1.6f, 1.2f, 5.7f});
    current.target = {0.3f, 0.4f, 0.1f};

    const glm::mat4 previousViewProjection = previous.viewProjectionMatrix(kAspect);
    // The depth buffer is rendered through the jittered projection, so that is
    // what the rows are built from; the jitter must not leak into the depths.
    const glm::mat4 currentJittered =
        jittered(current.projectionMatrix(kAspect), {0.0013f, -0.0021f}) * current.viewMatrix();
    const auto rows = taaPreviousDepthRows(currentJittered, previousViewProjection);

    for (const glm::vec3& world :
         {glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 2.0f, -1.5f), glm::vec3(-2.0f, 0.1f, 1.0f)}) {
        const Projected p = project(currentJittered, world);
        const float directPrevious = (previousViewProjection * glm::vec4(world, 1.0f)).w;
        const float directCurrent = -(current.viewMatrix() * glm::vec4(world, 1.0f)).z;

        CHECK(taaPreviousViewDepth(rows, p.ndc, p.depth) == Catch::Approx(directPrevious).epsilon(1e-4));
        CHECK(taaCurrentViewDepth(rows, p.ndc, p.depth) == Catch::Approx(directCurrent).epsilon(1e-4));
    }
}

TEST_CASE("History depth encoding keeps alpha at or above one", "[taa]")
{
    CHECK(taaEncodeHistoryDepth(0.0f) == kTaaHistoryDepthOffset);
    CHECK(taaDecodeHistoryDepth(taaEncodeHistoryDepth(12.5f)) == Catch::Approx(12.5f));
    // The value the resolve wrote before it recorded depth decodes to "unknown".
    CHECK(taaDecodeHistoryDepth(1.0f) <= 0.0f);
    // Sky is clamped into range, and stays far past any surface.
    CHECK(taaDecodeHistoryDepth(taaEncodeHistoryDepth(1.0e9f)) == Catch::Approx(kTaaHistorySkyDepth));
    // 16-bit float alpha tops out at 65504.
    CHECK(taaEncodeHistoryDepth(kTaaHistorySkyDepth) < 65504.0f);
}

TEST_CASE("History that matches the current surface is kept", "[taa]")
{
    // Current neighbourhood spans 9.8-10.2 m of expected previous depth.
    CHECK_FALSE(taaHistoryDisoccluded({10.0f, 10.0f, 10.0f, 10.0f}, 9.8f, 10.2f, false, 0.05f));
    // An edge: two texels show the background, two the surface. One match is enough.
    CHECK_FALSE(taaHistoryDisoccluded({3.0f, 3.0f, 10.1f, 3.0f}, 9.8f, 10.2f, false, 0.05f));
    // Inside the tolerance band but outside the raw range.
    CHECK_FALSE(taaHistoryDisoccluded({10.6f, 10.6f, 10.6f, 10.6f}, 9.8f, 10.2f, false, 0.05f));
}

TEST_CASE("History showing a surface that is gone is rejected", "[taa]")
{
    // The occluder that stood 3 m away has moved; the wall behind it is at 10 m.
    CHECK(taaHistoryDisoccluded({3.0f, 3.0f, 3.1f, 2.9f}, 9.8f, 10.2f, false, 0.05f));
    // And the reverse: history saw the far wall where a surface has since appeared.
    CHECK(taaHistoryDisoccluded({10.0f, 10.0f, 10.0f, 10.0f}, 2.9f, 3.1f, false, 0.05f));
    // Tolerance is relative: the same absolute gap passes at 5% and fails at 1%.
    CHECK_FALSE(taaHistoryDisoccluded({10.6f, 10.6f, 10.6f, 10.6f}, 9.8f, 10.2f, false, 0.05f));
    CHECK(taaHistoryDisoccluded({10.6f, 10.6f, 10.6f, 10.6f}, 9.8f, 10.2f, false, 0.01f));
}

TEST_CASE("Unknown history depth and an all-sky neighbourhood never reject", "[taa]")
{
    // A history written before depth was recorded, or sampled off the written
    // region, carries no depth to disagree with.
    CHECK_FALSE(taaHistoryDisoccluded({0.0f, 0.0f, 0.0f, 0.0f}, 9.8f, 10.2f, false, 0.05f));
    // Unknown texels are skipped rather than counted as mismatches.
    CHECK(taaHistoryDisoccluded({0.0f, 3.0f, 0.0f, 0.0f}, 9.8f, 10.2f, false, 0.05f));
    // Only sky in the current neighbourhood: min stays above max, nothing to judge.
    CHECK_FALSE(taaHistoryDisoccluded({3.0f, 3.0f, 3.0f, 3.0f}, 1.0e30f, 0.0f, true, 0.05f));
}

TEST_CASE("Sky in the neighbourhood leaves the upper bound open", "[taa]")
{
    const float sky = kTaaHistorySkyDepth;
    // A silhouette against the sky: history recorded sky there, which matches.
    CHECK_FALSE(taaHistoryDisoccluded({sky, sky, sky, sky}, 9.8f, 10.2f, true, 0.05f));
    // Without sky in the current neighbourhood, recorded sky is a mismatch.
    CHECK(taaHistoryDisoccluded({sky, sky, sky, sky}, 9.8f, 10.2f, false, 0.05f));
    // The lower bound still applies: an occluder in front of both is rejected.
    CHECK(taaHistoryDisoccluded({2.0f, 2.0f, 2.0f, 2.0f}, 9.8f, 10.2f, true, 0.05f));
}
