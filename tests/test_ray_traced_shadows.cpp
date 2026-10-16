#include "renderer/RayTracedShadows.h"
#include "renderer/RuntimeSettings.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

#include <cmath>
#include <numbers>

using ve::renderer::alignUpPowerOfTwo;
using ve::renderer::RayTracedCasterKind;
using ve::renderer::rayTracedCasterKind;
using ve::renderer::rayTracedShadowOffsetPerDistance;
using ve::renderer::rayTracingInstanceTransform;
using ve::renderer::RenderBucket;

namespace {

// Applies the 3x4 row-major instance transform the way the acceleration
// structure does: each row dotted with (x, y, z, 1).
glm::vec3 applyInstanceTransform(const std::array<float, 12>& rows, const glm::vec3& point)
{
    glm::vec3 result{};
    for (int row = 0; row < 3; ++row) {
        const size_t base = static_cast<size_t>(row * 4);
        result[row] = rows[base] * point.x + rows[base + 1] * point.y + rows[base + 2] * point.z + rows[base + 3];
    }
    return result;
}

} // namespace

TEST_CASE("The instance transform puts the translation in the last column of each row", "[raytracing]")
{
    // The copy-the-first-twelve-floats mistake keeps the identity rows of an
    // untranslated object intact, so only a translated one can catch it.
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
    const std::array<float, 12> rows = rayTracingInstanceTransform(model);

    CHECK(rows[3] == Catch::Approx(1.0f));
    CHECK(rows[7] == Catch::Approx(2.0f));
    CHECK(rows[11] == Catch::Approx(3.0f));
    CHECK(rows[0] == Catch::Approx(1.0f));
    CHECK(rows[5] == Catch::Approx(1.0f));
    CHECK(rows[10] == Catch::Approx(1.0f));
}

TEST_CASE("The instance transform moves points exactly as the model matrix does", "[raytracing]")
{
    // Rotation, non-uniform scale and translation together: a transposed
    // rotation still passes a symmetric test, and a uniform scale hides a
    // row/column swap of the scale terms.
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 0.5f, 7.0f));
    model = glm::rotate(model, 0.7f, glm::normalize(glm::vec3(0.3f, 1.0f, -0.2f)));
    model = glm::scale(model, glm::vec3(2.0f, 0.5f, 3.0f));
    const std::array<float, 12> rows = rayTracingInstanceTransform(model);

    for (const glm::vec3 point : {glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.2f, -3.0f, 5.0f)}) {
        const glm::vec3 expected = glm::vec3(model * glm::vec4(point, 1.0f));
        const glm::vec3 actual = applyInstanceTransform(rows, point);
        CHECK(actual.x == Catch::Approx(expected.x).margin(1e-5));
        CHECK(actual.y == Catch::Approx(expected.y).margin(1e-5));
        CHECK(actual.z == Catch::Approx(expected.z).margin(1e-5));
    }
}

TEST_CASE("Ray-traced casters follow the cascades' bucket rules", "[raytracing]")
{
    // docs/transparency.md: opaque casts, a cutout casts what its alpha test
    // keeps, blended geometry casts nothing. A ray path that let glass cast
    // would disagree with the cascades on every window in the scene.
    CHECK(rayTracedCasterKind(RenderBucket::Opaque) == RayTracedCasterKind::Opaque);
    CHECK(rayTracedCasterKind(RenderBucket::Mask) == RayTracedCasterKind::AlphaTested);
    CHECK(rayTracedCasterKind(RenderBucket::Blend) == RayTracedCasterKind::None);
}

TEST_CASE("The distance offset is one pixel's world footprint per requested pixel", "[raytracing]")
{
    // At a 90-degree field of view a frame 2 pixels tall spans 2 units at
    // distance 1, so one pixel is one unit per unit of distance.
    const float rightAngle = std::numbers::pi_v<float> * 0.5f;
    CHECK(rayTracedShadowOffsetPerDistance(1.0f, 0.0f, rightAngle, 2) == Catch::Approx(1.0f));
    CHECK(rayTracedShadowOffsetPerDistance(2.0f, 0.0f, rightAngle, 2) == Catch::Approx(2.0f));

    // The LOD budget adds on top of the margin, which is the whole point of it:
    // a coarse level is at most that many pixels off level 0.
    CHECK(rayTracedShadowOffsetPerDistance(1.0f, 1.5f, rightAngle, 2) == Catch::Approx(2.5f));

    // Taller frames have smaller pixels.
    const float at720 = rayTracedShadowOffsetPerDistance(1.0f, 0.0f, rightAngle, 720);
    const float at1440 = rayTracedShadowOffsetPerDistance(1.0f, 0.0f, rightAngle, 1440);
    CHECK(at1440 == Catch::Approx(at720 * 0.5f));
}

TEST_CASE("The distance offset degenerates to zero rather than to infinity", "[raytracing]")
{
    const float fov = std::numbers::pi_v<float> / 3.0f;
    CHECK(rayTracedShadowOffsetPerDistance(1.0f, 1.0f, fov, 0) == 0.0f);
    // Negative inputs from a hand-edited file add nothing instead of pulling the
    // origin into the surface.
    CHECK(rayTracedShadowOffsetPerDistance(-3.0f, -1.0f, fov, 720) == 0.0f);
}

TEST_CASE("The shadow ray reaches across the whole scene, not to the far plane", "[raytracing]")
{
    using ve::renderer::Aabb;
    using ve::renderer::rayTracedShadowRayLength;

    // Any receiver and any caster are both inside these bounds, so the diagonal
    // is the farthest an occluder can be.
    Aabb bounds{};
    bounds.expand(glm::vec3(-100.0f, 0.0f, -50.0f));
    bounds.expand(glm::vec3(200.0f, 40.0f, 350.0f));
    const float diagonal = glm::length(bounds.max - bounds.min);
    CHECK(rayTracedShadowRayLength(bounds) >= diagonal);
    // With a margin, not a multiple: the ray stays proportionate to the scene.
    CHECK(rayTracedShadowRayLength(bounds) < diagonal * 1.1f + 2.0f);

    // A scene collapsed to a point still gets a ray longer than the origin offset.
    Aabb point{};
    point.expand(glm::vec3(3.0f));
    CHECK(rayTracedShadowRayLength(point) >= 1.0f);

    // No bounds at all must not produce a zero-length ray, which would light
    // every pixel.
    CHECK(rayTracedShadowRayLength(Aabb{}) == ve::renderer::kRtFallbackRayLength);
}

TEST_CASE("Scratch addresses round up to the device's alignment", "[raytracing]")
{
    CHECK(alignUpPowerOfTwo(0, 128) == 0);
    CHECK(alignUpPowerOfTwo(1, 128) == 128);
    CHECK(alignUpPowerOfTwo(128, 128) == 128);
    CHECK(alignUpPowerOfTwo(129, 128) == 256);
    // An alignment of one or zero leaves the value alone.
    CHECK(alignUpPowerOfTwo(77, 1) == 77);
    CHECK(alignUpPowerOfTwo(77, 0) == 77);
}

TEST_CASE("The pre-skin dispatch covers every vertex with no spare group", "[raytracing]")
{
    using ve::renderer::kRtPreSkinLocalSize;
    using ve::renderer::rayTracingPreSkinGroupCount;

    CHECK(rayTracingPreSkinGroupCount(0) == 0);
    CHECK(rayTracingPreSkinGroupCount(1) == 1);
    CHECK(rayTracingPreSkinGroupCount(kRtPreSkinLocalSize) == 1);
    CHECK(rayTracingPreSkinGroupCount(kRtPreSkinLocalSize + 1) == 2);
    // Enough threads for the last vertex, never a whole idle group past it.
    for (const uint32_t vertices : {7u, 64u, 65u, 1000u, 4096u}) {
        const uint32_t groups = rayTracingPreSkinGroupCount(vertices);
        CHECK(groups * kRtPreSkinLocalSize >= vertices);
        CHECK((groups - 1) * kRtPreSkinLocalSize < vertices);
    }
}

TEST_CASE("Ray-traced shadow flag bits are distinct single bits", "[raytracing]")
{
    // The shader tests them with &, so two sharing a bit would turn one debug
    // view on whenever the other was.
    constexpr uint32_t enabled = ve::renderer::kRtShadowFlagEnabled;
    constexpr uint32_t visibility = ve::renderer::kRtShadowFlagDebugVisibility;
    constexpr uint32_t compare = ve::renderer::kRtShadowFlagDebugCompareCascades;
    for (const uint32_t bit : {enabled, visibility, compare}) {
        CHECK(bit != 0u);
        CHECK((bit & (bit - 1u)) == 0u);
    }
    CHECK((enabled & visibility) == 0u);
    CHECK((enabled & compare) == 0u);
    CHECK((visibility & compare) == 0u);
}

TEST_CASE("Ray-traced shadow settings clamp into their ranges", "[raytracing][settings]")
{
    ve::RayTracedShadowSettings settings{};
    settings.normalOffset = -1.0f;
    settings.pixelFootprintOffset = 1000.0f;
    ve::clampRayTracedShadowSettings(settings);
    CHECK(settings.normalOffset == 0.0f);
    CHECK(settings.pixelFootprintOffset == 16.0f);

    // Zero is a real A/B point for both and must survive.
    settings.normalOffset = 0.0f;
    settings.pixelFootprintOffset = 0.0f;
    ve::clampRayTracedShadowSettings(settings);
    CHECK(settings.normalOffset == 0.0f);
    CHECK(settings.pixelFootprintOffset == 0.0f);

    // The defaults are already inside the ranges.
    ve::RayTracedShadowSettings defaults{};
    ve::RayTracedShadowSettings clamped = defaults;
    ve::clampRayTracedShadowSettings(clamped);
    CHECK(clamped == defaults);
    CHECK_FALSE(defaults.enabled);
}
