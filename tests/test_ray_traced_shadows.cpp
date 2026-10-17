#include "renderer/RayTracedShadows.h"
#include "renderer/RuntimeSettings.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <vector>

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

TEST_CASE("The shadow mask covers the sub-rect in whole workgroups", "[raytracing][mask]")
{
    using ve::renderer::kRtMaskLocalSize;
    using ve::renderer::rayTracedShadowMaskGroupCount;

    CHECK(rayTracedShadowMaskGroupCount(0) == 0);
    CHECK(rayTracedShadowMaskGroupCount(1) == 1);
    CHECK(rayTracedShadowMaskGroupCount(kRtMaskLocalSize) == 1);
    CHECK(rayTracedShadowMaskGroupCount(kRtMaskLocalSize + 1) == 2);
    // 720 rows are exactly 90 groups; the 1280 columns of a full-resolution
    // frame are 160. A render-scaled sub-rect rounds up, and the shader returns
    // early on the overhang.
    CHECK(rayTracedShadowMaskGroupCount(720) == 90);
    CHECK(rayTracedShadowMaskGroupCount(1280) == 160);
    CHECK(rayTracedShadowMaskGroupCount(721) == 91);
}

TEST_CASE("The shadow mask runs only where it has depth to trace from", "[raytracing][mask]")
{
    using ve::renderer::rayTracedShadowMaskActive;

    CHECK(rayTracedShadowMaskActive(true, true, true, true));
    // Each condition alone turns it off: a frame that does not trace, a frame
    // with no depth prepass to reconstruct surfaces from, a device or build
    // where the pass could not be created, and the setting, which is the A/B
    // control. In every case the main pass traces inline, as before the mask.
    CHECK_FALSE(rayTracedShadowMaskActive(false, true, true, true));
    CHECK_FALSE(rayTracedShadowMaskActive(true, false, true, true));
    CHECK_FALSE(rayTracedShadowMaskActive(true, true, false, true));
    CHECK_FALSE(rayTracedShadowMaskActive(true, true, true, false));

    // Off by default: measured, it only costs while the shadow is hard.
    const ve::RayTracedShadowSettings defaults{};
    CHECK_FALSE(defaults.screenSpaceMask);
    CHECK_FALSE(defaults.debugMaskSource);
}

TEST_CASE("A soft shadow or the reference asks for the mask by itself", "[raytracing][mask]")
{
    using ve::renderer::rayTracedShadowMaskRequested;

    // The hard shadow asks only through the setting, which measured as a cost.
    CHECK_FALSE(rayTracedShadowMaskRequested(false, 0.0f, 0));
    CHECK(rayTracedShadowMaskRequested(true, 0.0f, 0));
    // A sun with extent needs the mask: its filter has to see the visibility
    // before anything is shaded with it. So does the reference mode.
    CHECK(rayTracedShadowMaskRequested(false, 0.53f, 0));
    CHECK(rayTracedShadowMaskRequested(false, 0.0f, 16));

    // Off by default: no extent and no reference.
    const ve::RayTracedShadowSettings defaults{};
    CHECK_FALSE(rayTracedShadowMaskRequested(
        defaults.screenSpaceMask, defaults.sunAngularDiameterDegrees, defaults.debugReferenceSamples));
    CHECK(defaults.halfResolutionMask);
}

TEST_CASE("The trace grid covers the sub-rect at either resolution", "[raytracing][mask]")
{
    using ve::renderer::rayTracedShadowTraceExtent;
    using ve::renderer::rayTracedShadowTraceScale;

    CHECK(rayTracedShadowTraceScale(true, 0) == 2u);
    CHECK(rayTracedShadowTraceScale(false, 0) == 1u);
    // The reference is the converged answer for every pixel, so it never halves.
    CHECK(rayTracedShadowTraceScale(true, 8) == 1u);

    CHECK(rayTracedShadowTraceExtent(1280, 1) == 1280u);
    CHECK(rayTracedShadowTraceExtent(1280, 2) == 640u);
    // An odd sub-rect: texel t stands for pixel 2t, so the last pixel, 720, is
    // texel 360 -- 361 texels.
    CHECK(rayTracedShadowTraceExtent(721, 2) == 361u);
    for (uint32_t pixels : {1u, 2u, 3u, 719u, 720u, 721u}) {
        const uint32_t texels = rayTracedShadowTraceExtent(pixels, 2);
        // The last texel's pixel is inside the sub-rect, and no pixel is past
        // the last texel's block.
        CHECK((texels - 1) * 2 < pixels);
        CHECK(texels * 2 >= pixels);
    }
    CHECK(rayTracedShadowTraceExtent(100, 0) == 0u);
}

TEST_CASE("The sun's disc maps area-uniformly onto a cone of its angle", "[raytracing][mask]")
{
    using ve::renderer::concentricDiskSample;
    using ve::renderer::orthonormalBasis;
    using ve::renderer::sunConeDirection;
    using ve::renderer::sunConeTangent;

    // The cone's tangent: zero is the hard shadow, and the real sun's 0.53
    // degrees is about 0.0046.
    CHECK(sunConeTangent(0.0f) == 0.0f);
    CHECK(sunConeTangent(-1.0f) == 0.0f);
    CHECK(sunConeTangent(0.53f) == Catch::Approx(std::tan(0.265f * std::numbers::pi_v<float> / 180.0f)));

    // The centre of the square is the centre of the disc, the corners its rim,
    // and a regular grid covers the disc area-uniformly: the mean squared
    // radius of a uniform unit disc is 1/2.
    CHECK(glm::length(concentricDiskSample(glm::vec2(0.5f))) == Catch::Approx(0.0f).margin(1e-6));
    CHECK(glm::length(concentricDiskSample(glm::vec2(1.0f, 0.5f))) == Catch::Approx(1.0f));
    CHECK(glm::length(concentricDiskSample(glm::vec2(0.0f, 0.0f))) == Catch::Approx(1.0f));
    constexpr int kGrid = 64;
    double radiusSquaredSum = 0.0;
    double xSum = 0.0;
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            const glm::vec2 u((static_cast<float>(x) + 0.5f) / kGrid, (static_cast<float>(y) + 0.5f) / kGrid);
            const glm::vec2 disk = concentricDiskSample(u);
            CHECK(glm::length(disk) <= 1.0f + 1e-5f);
            radiusSquaredSum += glm::dot(disk, disk);
            xSum += disk.x;
        }
    }
    CHECK(radiusSquaredSum / (kGrid * kGrid) == Catch::Approx(0.5).epsilon(0.01));
    CHECK(xSum / (kGrid * kGrid) == Catch::Approx(0.0).margin(1e-3));

    // The basis is orthonormal for any direction, including straight down,
    // where a basis built by crossing with +z would degenerate.
    for (const glm::vec3 n : {glm::vec3(0.0f, 0.0f, 1.0f),
                              glm::vec3(0.0f, 0.0f, -1.0f),
                              glm::normalize(glm::vec3(0.35f, -0.65f, -0.55f)),
                              glm::normalize(glm::vec3(1.0f, 1e-4f, 0.0f))}) {
        glm::vec3 tangent{};
        glm::vec3 bitangent{};
        orthonormalBasis(n, tangent, bitangent);
        CHECK(glm::length(tangent) == Catch::Approx(1.0f).epsilon(1e-5));
        CHECK(glm::length(bitangent) == Catch::Approx(1.0f).epsilon(1e-5));
        CHECK(glm::dot(tangent, n) == Catch::Approx(0.0f).margin(1e-5));
        CHECK(glm::dot(bitangent, n) == Catch::Approx(0.0f).margin(1e-5));
        CHECK(glm::dot(tangent, bitangent) == Catch::Approx(0.0f).margin(1e-5));
    }

    // Every direction stays inside the cone and the rim reaches it; the centre
    // is the sun's own direction.
    const glm::vec3 towardLight = glm::normalize(glm::vec3(-0.35f, 0.65f, 0.55f));
    const float tanRadius = sunConeTangent(2.0f);
    const float cosRadius = 1.0f / std::sqrt(1.0f + tanRadius * tanRadius);
    CHECK(glm::dot(sunConeDirection(towardLight, tanRadius, glm::vec2(0.5f)), towardLight) == Catch::Approx(1.0f));
    float smallestCosine = 1.0f;
    for (int y = 0; y <= 8; ++y) {
        for (int x = 0; x <= 8; ++x) {
            const glm::vec2 u(static_cast<float>(x) / 8.0f, static_cast<float>(y) / 8.0f);
            const float cosine = glm::dot(sunConeDirection(towardLight, tanRadius, u), towardLight);
            CHECK(cosine >= cosRadius - 1e-6f);
            smallestCosine = std::min(smallestCosine, cosine);
        }
    }
    CHECK(smallestCosine == Catch::Approx(cosRadius).epsilon(1e-5));
}

TEST_CASE("The frame the mask turns on rebuilds the top-level structure", "[raytracing][mask]")
{
    using ve::renderer::rayTracedShadowMaskNeedsFreshBuild;

    // A kept structure is only visible to the stages that read it on the frame
    // that built it. Built with the mask off, that is the fragment stage alone,
    // so the first frame the mask's compute rays need it has to build again.
    CHECK(rayTracedShadowMaskNeedsFreshBuild(true, false));
    // Built with the mask on, both stages were covered, and stay covered for as
    // long as the structure is kept -- including across the mask going off and
    // back on.
    CHECK_FALSE(rayTracedShadowMaskNeedsFreshBuild(true, true));
    // Turning the mask off leaves only the fragment reads, which every build
    // covers.
    CHECK_FALSE(rayTracedShadowMaskNeedsFreshBuild(false, true));
    CHECK_FALSE(rayTracedShadowMaskNeedsFreshBuild(false, false));
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

    // The sun's disc and the reference both have ceilings, and a NaN angle must
    // not survive to reach tan().
    settings.sunAngularDiameterDegrees = 90.0f;
    settings.debugReferenceSamples = 1000;
    ve::clampRayTracedShadowSettings(settings);
    CHECK(settings.sunAngularDiameterDegrees == 5.0f);
    CHECK(settings.debugReferenceSamples == ve::renderer::kRtReferenceMaxSamples);
    settings.sunAngularDiameterDegrees = std::numeric_limits<float>::quiet_NaN();
    ve::clampRayTracedShadowSettings(settings);
    CHECK(settings.sunAngularDiameterDegrees == 0.0f);

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

// The top-level structure's change detection. The tracker sees bytes, so the
// tests stand in short word sequences for the instance records; what matters is
// that equal content keeps the structure and any difference rebuilds it.
namespace {

using ve::renderer::RayTracingTopLevelPlan;
using ve::renderer::RayTracingTopLevelTracker;

std::vector<std::byte> instanceBytes(std::initializer_list<uint32_t> words)
{
    std::vector<std::byte> bytes(words.size() * sizeof(uint32_t));
    size_t offset = 0;
    for (const uint32_t word : words) {
        std::memcpy(bytes.data() + offset, &word, sizeof(word));
        offset += sizeof(word);
    }
    return bytes;
}

// One planned frame whose build is recorded, which is what every frame of a
// run that does not throw between frame prep and recording looks like.
RayTracingTopLevelPlan tracedFrame(RayTracingTopLevelTracker& tracker,
                                   uint32_t slot,
                                   const std::vector<std::byte>& instances,
                                   uint64_t pose = 0,
                                   bool rebuildEveryFrame = false)
{
    const RayTracingTopLevelPlan plan = tracker.plan(slot, instances, pose, rebuildEveryFrame);
    tracker.commitBuild();
    return plan;
}

} // namespace

TEST_CASE("The first traced frame builds, refits and writes its slot", "[raytracing][tlas]")
{
    // The structure built at startup is empty, so whatever the first frame
    // traces has to be built -- even when it is empty too.
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const RayTracingTopLevelPlan first = tracedFrame(tracker, 0, instanceBytes({1, 2, 3}), 7);
    CHECK(first.build);
    CHECK(first.refitSkinned);
    CHECK(first.writeSlot);

    RayTracingTopLevelTracker empty;
    empty.reset(2);
    const RayTracingTopLevelPlan nothing = tracedFrame(empty, 0, {});
    CHECK(nothing.build);
    CHECK_FALSE(nothing.refitSkinned);
}

TEST_CASE("An unchanged frame keeps the structure and catches up only a stale slot", "[raytracing][tlas]")
{
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> scene = instanceBytes({10, 20, 30, 40});
    REQUIRE(tracedFrame(tracker, 0, scene).build);

    // Slot 1 has never been written, and the frame's rays read its geometry
    // records whether or not the structure is rebuilt.
    const RayTracingTopLevelPlan second = tracedFrame(tracker, 1, scene);
    CHECK_FALSE(second.build);
    CHECK_FALSE(second.refitSkinned);
    CHECK(second.writeSlot);

    // Both slots hold this generation now: nothing left to do.
    for (uint32_t frame = 0; frame < 4; ++frame) {
        const RayTracingTopLevelPlan steady = tracedFrame(tracker, frame % 2, scene);
        CHECK_FALSE(steady.build);
        CHECK_FALSE(steady.writeSlot);
    }
}

TEST_CASE("Any difference in the instance records rebuilds the structure", "[raytracing][tlas]")
{
    // Stand-ins for the fields the bytes carry: a moved transform, a custom
    // index that moved because the draw list was reordered, an instance flag,
    // an instance dropped, one added, and all of them gone.
    const std::vector<std::byte> scene = instanceBytes({1, 2, 3, 4});
    const std::vector<std::vector<std::byte>> changes{
        instanceBytes({9, 2, 3, 4}),
        instanceBytes({1, 2, 3, 5}),
        instanceBytes({1, 2, 3u | 0x80000000u, 4}),
        instanceBytes({1, 2, 3}),
        instanceBytes({1, 2, 3, 4, 0}),
        {},
    };
    for (const std::vector<std::byte>& changed : changes) {
        RayTracingTopLevelTracker tracker;
        tracker.reset(2);
        REQUIRE(tracedFrame(tracker, 0, scene).build);
        REQUIRE_FALSE(tracedFrame(tracker, 1, scene).build);

        const RayTracingTopLevelPlan plan = tracedFrame(tracker, 0, changed);
        CHECK(plan.build);
        CHECK(plan.writeSlot);
        CHECK_FALSE(plan.refitSkinned);
        // And once built, the new content is kept like the old.
        CHECK_FALSE(tracedFrame(tracker, 1, changed).build);
    }
}

TEST_CASE("A moving pose refits and rebuilds; a held one does neither", "[raytracing][tlas]")
{
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> scene = instanceBytes({5, 6});
    REQUIRE(tracedFrame(tracker, 0, scene, 100).refitSkinned);

    // The instance records do not move with the pose -- same BLAS, same
    // transform -- which is exactly why the pose is tracked apart from them.
    const RayTracingTopLevelPlan moved = tracedFrame(tracker, 1, scene, 101);
    CHECK(moved.refitSkinned);
    CHECK(moved.build);

    const RayTracingTopLevelPlan held = tracedFrame(tracker, 0, scene, 101);
    CHECK_FALSE(held.refitSkinned);
    CHECK_FALSE(held.build);
    // Slot 0 was written on the first frame and the records have not moved
    // since, so a pose alone never costs a slot write.
    CHECK_FALSE(held.writeSlot);
}

TEST_CASE("The skinned caster coming back in the pose it left rebuilds without a refit", "[raytracing][tlas]")
{
    // Hiding the mesh drops its instance, which changes the content; showing it
    // again in the same pose changes it back. The BLAS was never touched in
    // between, so it still holds that pose -- only the TLAS has to move.
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> withSkinned = instanceBytes({1, 2, 99});
    const std::vector<std::byte> without = instanceBytes({1, 2});
    REQUIRE(tracedFrame(tracker, 0, withSkinned, 42).refitSkinned);

    const RayTracingTopLevelPlan hidden = tracedFrame(tracker, 1, without, 0);
    CHECK(hidden.build);
    CHECK_FALSE(hidden.refitSkinned);

    const RayTracingTopLevelPlan shown = tracedFrame(tracker, 0, withSkinned, 42);
    CHECK(shown.build);
    CHECK_FALSE(shown.refitSkinned);
}

TEST_CASE("Content that returns to an earlier state is still rebuilt", "[raytracing][tlas]")
{
    // The structure holds the latest build, not every build: A, then B, then A
    // again has to rebuild A. So does the slot that last held A, because it was
    // written for an older generation than the one now traced.
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> a = instanceBytes({1});
    const std::vector<std::byte> b = instanceBytes({2});
    REQUIRE(tracedFrame(tracker, 0, a).build);
    REQUIRE(tracedFrame(tracker, 1, b).build);
    const RayTracingTopLevelPlan back = tracedFrame(tracker, 0, a);
    CHECK(back.build);
    CHECK(back.writeSlot);
}

TEST_CASE("A build that was planned and never recorded is planned again", "[raytracing][tlas]")
{
    // Frame prep plans, recording commits. A frame that throws in between built
    // nothing, and the next one must not trust that it did.
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> scene = instanceBytes({3, 1, 4});
    REQUIRE(tracker.plan(0, scene, 8, false).build);

    const RayTracingTopLevelPlan retry = tracker.plan(1, scene, 8, false);
    CHECK(retry.build);
    CHECK(retry.refitSkinned);
    tracker.commitBuild();

    // A second commit with nothing planned changes nothing.
    tracker.commitBuild();
    CHECK_FALSE(tracker.plan(0, scene, 8, false).build);
}

TEST_CASE("Rebuilding every frame does all of it every frame", "[raytracing][tlas]")
{
    // The setting reproduces what every frame did before the tracker, which is
    // what makes it the control for an A/B and for the image.
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> scene = instanceBytes({1, 1, 2, 3, 5});
    for (uint32_t frame = 0; frame < 6; ++frame) {
        const RayTracingTopLevelPlan plan = tracedFrame(tracker, frame % 2, scene, 77, true);
        CHECK(plan.build);
        CHECK(plan.refitSkinned);
        CHECK(plan.writeSlot);
    }
    // Without the skinned caster there is nothing to refit, setting or not.
    CHECK_FALSE(tracedFrame(tracker, 0, scene, 0, true).refitSkinned);
    // And turning it off keeps what the last forced build left.
    CHECK_FALSE(tracedFrame(tracker, 1, scene, 0, false).build);
}

TEST_CASE("Reset forgets the structure and every slot", "[raytracing][tlas]")
{
    RayTracingTopLevelTracker tracker;
    tracker.reset(2);
    const std::vector<std::byte> scene = instanceBytes({4, 4});
    tracedFrame(tracker, 0, scene, 5);
    tracedFrame(tracker, 1, scene, 5);
    REQUIRE_FALSE(tracedFrame(tracker, 0, scene, 5).build);

    tracker.reset(2);
    const RayTracingTopLevelPlan after = tracedFrame(tracker, 0, scene, 5);
    CHECK(after.build);
    CHECK(after.refitSkinned);
    CHECK(after.writeSlot);
}

TEST_CASE("Following the plans keeps the structure, the BLAS and every slot current", "[raytracing][tlas]")
{
    // A model of what the GPU holds, driven only by the plans: a slot changes
    // when the plan writes it, the BLAS when it refits, the structure when it
    // builds -- from the slot, as the real build reads the slot's instance
    // buffer. Over a long random run of content, pose, slot, the setting, and
    // frames whose recording never happens, every frame that ran must trace
    // exactly what it meant to.
    const std::vector<std::vector<std::byte>> contents{
        instanceBytes({1, 2, 3}), instanceBytes({1, 2, 4}), instanceBytes({7}), {}};
    const std::array<uint64_t, 3> poses{0, 11, 12};
    constexpr uint32_t kSlots = 3;
    constexpr uint32_t kFrames = 4000;

    RayTracingTopLevelTracker tracker;
    tracker.reset(kSlots);
    std::array<std::vector<std::byte>, kSlots> slots{};
    std::array<bool, kSlots> slotWritten{};
    std::vector<std::byte> structure;
    bool structureBuilt = false;
    uint64_t blasPose = 0;
    uint32_t builds = 0;
    uint32_t mismatches = 0;

    // A fixed linear congruential sequence rather than <random>, whose
    // distributions differ between standard libraries.
    uint32_t state = 12345u;
    const auto next = [&state](uint32_t range) {
        state = state * 1664525u + 1013904223u;
        return (state >> 8) % range;
    };

    size_t content = 0;
    uint64_t pose = poses[1];
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        // Mostly still frames, which is the case being optimised.
        if (next(8) == 0) {
            content = next(static_cast<uint32_t>(contents.size()));
        }
        if (next(10) == 0) {
            pose = poses[next(static_cast<uint32_t>(poses.size()))];
        }
        const uint32_t slot = frame % kSlots;
        const bool forced = next(50) == 0;
        const bool recorded = next(40) != 0;

        const RayTracingTopLevelPlan plan = tracker.plan(slot, contents[content], pose, forced);
        if (plan.refitSkinned && (!plan.build || pose == 0)) {
            ++mismatches;
        }
        // The CPU writes the slot in frame prep, recorded or not.
        if (plan.writeSlot) {
            slots[slot] = contents[content];
            slotWritten[slot] = true;
        }
        if (!recorded) {
            continue;
        }
        if (plan.build) {
            if (plan.refitSkinned) {
                blasPose = pose;
            }
            if (!slotWritten[slot]) {
                ++mismatches;
            }
            structure = slots[slot];
            structureBuilt = true;
            ++builds;
            tracker.commitBuild();
        }

        // What this frame's rays see: the structure, the slot's geometry
        // records, and the skinned BLAS.
        if (!structureBuilt || structure != contents[content] || slots[slot] != contents[content] ||
            (pose != 0 && blasPose != pose)) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0u);
    // And it did skip: a change lands on about one frame in five here, so far
    // fewer builds than frames.
    CHECK(builds > 0u);
    CHECK(builds < kFrames / 2u);
}
